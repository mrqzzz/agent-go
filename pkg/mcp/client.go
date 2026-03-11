package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os/exec"
	"strings"
	"sync"
	"time"

	"agent-go/pkg/llm"
)

type Client struct {
	Name            string
	Debug           bool
	InternalTimeout time.Duration
	cmd             *exec.Cmd
	stdin           io.WriteCloser
	stdout          io.ReadCloser
	stderr          io.ReadCloser // Read tool errors
	seq             int
	pending         map[int]chan jsonRPCResponse
	mu              sync.Mutex
	closed          chan struct{}
	tools           map[string]bool
}

// JSON-RPC structures
type jsonRPCRequest struct {
	JSONRPC string      `json:"jsonrpc"`
	Method  string      `json:"method"`
	Params  interface{} `json:"params,omitempty"`
	ID      int         `json:"id"`
}

type jsonRPCResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *jsonRPCError   `json:"error,omitempty"`
	ID      int             `json:"id"`
}

type jsonRPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type mcpListToolsResult struct {
	Tools []llm.ToolDefinition `json:"tools"`
}

type mcpCallToolResult struct {
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	IsError bool `json:"isError"`
}

func NewClient(name string, command string, args []string, debug bool, internalTimeout time.Duration) (*Client, error) {
	if internalTimeout <= 0 {
		internalTimeout = 2 * time.Minute
	}

	cmd := exec.Command(command, args...)

	// Stdin pipe
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}

	// Stdout pipe (MCP protocol)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}

	// Stderr pipe (tool logs/errors)
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, err
	}

	client := &Client{
		Name:            name,
		Debug:           debug,
		InternalTimeout: internalTimeout,
		cmd:             cmd,
		stdin:           stdin,
		stdout:          stdout,
		stderr:          stderr,
		pending:         make(map[int]chan jsonRPCResponse),
		closed:          make(chan struct{}),
		tools:           make(map[string]bool),
	}

	if err := cmd.Start(); err != nil {
		return nil, err
	}

	// Start output monitoring
	go client.readLoop()      // Reads stdout (JSON-RPC)
	go client.monitorStderr() // Reads stderr (tool text logs)

	return client, nil
}

// monitorStderr reads standard error from the process and logs it
func (c *Client) monitorStderr() {
	scanner := bufio.NewScanner(c.stderr)
	for scanner.Scan() {
		text := scanner.Text()
		// Log with clear prefix
		log.Printf("🔴 [%s STDERR] %s", c.Name, text)
	}
}

// readLoop reads standard output (MCP data channel)
func (c *Client) readLoop() {
	scanner := bufio.NewScanner(c.stdout)
	buf := make([]byte, 0, 1024*1024)
	scanner.Buffer(buf, 10*1024*1024)

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		// Stdout traffic logging (JSON)
		// Useful for seeing what the tool actually responds
		// log.Printf("🔵 [%s STDOUT] %s", c.Name, string(line))

		var resp jsonRPCResponse
		if err := json.Unmarshal(line, &resp); err != nil {
			log.Printf("⚠️ [%s] Non-JSON output ignored: %s", c.Name, string(line))
			continue
		}

		c.mu.Lock()
		ch, ok := c.pending[resp.ID]
		if ok {
			delete(c.pending, resp.ID)
			select {
			case ch <- resp:
			default:
				log.Printf("⚠️ [%s] Warning: channel full for ID %d", c.Name, resp.ID)
			}
		}
		c.mu.Unlock()
	}
	close(c.closed)
}

func (c *Client) call(ctx context.Context, method string, params interface{}) (json.RawMessage, error) {
	c.mu.Lock()
	c.seq++
	id := c.seq
	ch := make(chan jsonRPCResponse, 1)
	c.pending[id] = ch
	c.mu.Unlock()

	defer func() {
		c.mu.Lock()
		delete(c.pending, id)
		c.mu.Unlock()
	}()

	req := jsonRPCRequest{
		JSONRPC: "2.0",
		Method:  method,
		Params:  params,
		ID:      id,
	}

	bytes, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	// Write to stdin
	if _, err := c.stdin.Write(append(bytes, '\n')); err != nil {
		return nil, fmt.Errorf("stdin write failed: %w", err)
	}

	// Wait for response
	select {
	case resp := <-ch:
		if resp.Error != nil {
			return nil, fmt.Errorf("MCP Error (%d): %s", resp.Error.Code, resp.Error.Message)
		}
		return resp.Result, nil

	case <-ctx.Done():
		return nil, ctx.Err()

	case <-c.closed:
		return nil, fmt.Errorf("MCP connection closed unexpectedly")

	case <-time.After(c.InternalTimeout): // Internal safety timeout
		log.Printf("⏰ [%s] Internal TIMEOUT on call ID %d", c.Name, id)
		return nil, fmt.Errorf("TIMEOUT: Tool '%s' did not respond in %s", c.Name, c.InternalTimeout)
	}
}

func (c *Client) CallTool(ctx context.Context, toolName string, argsJSON string) (string, error) {
	// --- LOG START ---
	if c.Debug {
		log.Printf("🚀 [%s] START CallTool: %s | Args: %s", c.Name, toolName, argsJSON)
	}
	startTime := time.Now()
	// -----------------

	var argsMap map[string]interface{}
	if err := json.Unmarshal([]byte(argsJSON), &argsMap); err != nil {
		return "", fmt.Errorf("invalid JSON arguments: %v", err)
	}

	params := map[string]interface{}{
		"name":      toolName,
		"arguments": argsMap,
	}

	// Execution-specific timeout
	timeoutDur := 120 * time.Second
	callCtx, cancel := context.WithTimeout(ctx, timeoutDur)
	defer cancel()

	raw, err := c.call(callCtx, "tools/call", params)

	// --- LOG END / ERROR / TIMEOUT ---
	duration := time.Since(startTime)
	if err != nil {
		if err == context.DeadlineExceeded {
			log.Printf("💀 [%s] TIMEOUT CallTool: %s dopo %v", c.Name, toolName, duration)
		} else {
			log.Printf("❌ [%s] ERROR CallTool: %s | Err: %v", c.Name, toolName, err)
		}
		return "", err
	}

	if c.Debug {
		log.Printf("🏁 [%s] END CallTool: %s | Durata: %v", c.Name, toolName, duration)
	}
	// ---------------------------------

	var result mcpCallToolResult
	if err := json.Unmarshal(raw, &result); err != nil {
		return "", err
	}

	output := ""
	for _, content := range result.Content {
		if content.Type == "text" {
			output += content.Text + "\n"
		}
	}

	if result.IsError {
		if output == "" {
			output = "Unknown tool error"
		}
		log.Printf("⚠️ [%s] TOOL RETURNED ERROR: %s", c.Name, strings.TrimSpace(output))
		return output, fmt.Errorf("%s", strings.TrimSpace(output))
	}

	return output, nil
}

func (c *Client) ListTools(ctx context.Context) ([]llm.ToolDefinition, error) {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	raw, err := c.call(ctx, "tools/list", map[string]string{})
	if err != nil {
		return nil, err
	}

	var result mcpListToolsResult
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, err
	}
	// MCP servers send "inputSchema" but LLM providers expect "parameters".
	// Copy inputSchema into Parameters when Parameters is missing.
	c.tools = make(map[string]bool)
	for i := range result.Tools {
		if result.Tools[i].Parameters == nil && result.Tools[i].InputSchema != nil {
			result.Tools[i].Parameters = result.Tools[i].InputSchema
		}
		c.tools[result.Tools[i].Name] = true
	}
	return result.Tools, nil
}

func (c *Client) HasTool(name string) bool {
	return c.tools[name]
}
