package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"
)

// OllamaProvider handles communication with a local or remote Ollama instance
type OllamaProvider struct {
	BaseURL     string
	Model       string
	ContextSize int
	Debug       bool
	Client      *http.Client
}

func NewOllama(baseURL, model string, ctxSize int, debug bool) *OllamaProvider {
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}
	if ctxSize == 0 {
		ctxSize = 8192
	}

	return &OllamaProvider{
		BaseURL:     baseURL,
		Model:       model,
		ContextSize: ctxSize,
		Debug:       debug,
		Client: &http.Client{
			// High overall timeout to allow for processing
			Timeout: 600 * time.Second,
			// Transport config (fix for 0% CPU hang)
			Transport: &http.Transport{
				DialContext: (&net.Dialer{
					Timeout:   30 * time.Second,
					KeepAlive: 30 * time.Second,
				}).DialContext,
				TLSHandshakeTimeout:   10 * time.Second,
				ResponseHeaderTimeout: 600 * time.Second,
				ExpectContinueTimeout: 1 * time.Second,
				DisableKeepAlives:     true, // Close connection after each request
				MaxIdleConns:          -1,
			},
		},
	}
}

// --- Internal structures ---
type ollamaRequest struct {
	Model    string                 `json:"model"`
	Messages []ollamaMessage        `json:"messages"`
	Tools    []ollamaTool           `json:"tools,omitempty"`
	Stream   bool                   `json:"stream"`
	Options  map[string]interface{} `json:"options,omitempty"`
}

type ollamaTool struct {
	Type     string         `json:"type"`
	Function ToolDefinition `json:"function"`
}

type ollamaResponse struct {
	Model     string    `json:"model"`
	CreatedAt time.Time `json:"created_at"`
	Message   struct {
		Role      string `json:"role"`
		Content   string `json:"content"`
		ToolCalls []struct {
			Function struct {
				Name      string                 `json:"name"`
				Arguments map[string]interface{} `json:"arguments"`
			} `json:"function"`
		} `json:"tool_calls"`
	} `json:"message"`
	Done bool `json:"done"`
}

// ollamaMessage is the Ollama-native message format for requests.
// Tool calls use Ollama's format with arguments as a JSON object,
// not a double-encoded string, to prevent XML template breakage.
type ollamaMessage struct {
	Role       string           `json:"role"`
	Content    string           `json:"content"`
	Name       string           `json:"name,omitempty"`
	ToolCalls  []ollamaToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
}

type ollamaToolCall struct {
	Function ollamaToolCallFunc `json:"function"`
}

type ollamaToolCallFunc struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
}

// toOllamaMessages converts internal Messages to Ollama's native format.
// Arguments are parsed from JSON strings back to objects so they appear as
// native JSON objects in the request (single encoding, not double-encoded strings).
// String values inside arguments are XML-escaped here (at the Ollama boundary)
// so the model's stored history retains raw values like && while the rendered
// XML template stays valid.
func toOllamaMessages(msgs []Message) []ollamaMessage {
	result := make([]ollamaMessage, 0, len(msgs))
	for _, m := range msgs {
		om := ollamaMessage{
			Role:       m.Role,
			Content:    m.Content,
			Name:       m.Name,
			ToolCallID: m.ToolCallID,
		}
		for _, tc := range m.ToolCalls {
			var args map[string]interface{}
			if err := json.Unmarshal([]byte(tc.Arguments), &args); err != nil {
				args = map[string]interface{}{}
			}
			escapeMapValues(args)
			om.ToolCalls = append(om.ToolCalls, ollamaToolCall{
				Function: ollamaToolCallFunc{
					Name:      tc.Name,
					Arguments: args,
				},
			})
		}
		result = append(result, om)
	}
	return result
}

// unescapeMapValues reverses XML entities in all string values of a map (recursively).
func unescapeMapValues(m map[string]interface{}) {
	for k, v := range m {
		switch val := v.(type) {
		case string:
			m[k] = unescapeXMLContent(val)
		case map[string]interface{}:
			unescapeMapValues(val)
		}
	}
}

// escapeMapValues XML-escapes all string values in a map (recursively).
// Un-escapes first to avoid double-escaping if the model already produced entities.
func escapeMapValues(m map[string]interface{}) {
	for k, v := range m {
		switch val := v.(type) {
		case string:
			m[k] = escapeXMLContent(unescapeXMLContent(val))
		case map[string]interface{}:
			escapeMapValues(val)
		}
	}
}

// escapeXMLContent replaces characters that break Qwen's XML chat template.
func escapeXMLContent(s string) string {
	s = strings.ReplaceAll(s, "&", "&amp;")
	s = strings.ReplaceAll(s, "<", "&lt;")
	s = strings.ReplaceAll(s, ">", "&gt;")
	return s
}

// unescapeXMLContent reverses escapeXMLContent.
// The model may reproduce XML entities it saw in the rendered template,
// so we un-escape them to recover the raw values.
func unescapeXMLContent(s string) string {
	s = strings.ReplaceAll(s, "&lt;", "<")
	s = strings.ReplaceAll(s, "&gt;", ">")
	s = strings.ReplaceAll(s, "&amp;", "&")
	return s
}

// sanitizeHistory fixes message patterns that confuse Qwen's XML tool-call template.
func (p *OllamaProvider) sanitizeHistory(history []Message) []Message {
	sanitized := make([]Message, 0, len(history))
	for _, msg := range history {
		m := msg
		// Escape XML-sensitive characters in ALL message content.
		// Qwen's chat template uses XML; raw <, >, & in content breaks parsing.
		// Tool output is the main offender (e.g. git's "<file>"), but user
		// and assistant messages can also contain & (from "&&" commands).
		m.Content = escapeXMLContent(m.Content)
		// NOTE: We do NOT escape tool call arguments here. Argument escaping
		// happens in toOllamaMessages (at the Ollama boundary) so the stored
		// history retains raw values. If we escaped here, the model would see
		// &amp;&amp; in its history and reproduce it in future tool calls.
		// Assistant messages with empty content + tool_calls break Qwen's XML template.
		// Give them a non-empty content so the template renders correctly.
		if m.Role == RoleModel && m.Content == "" && len(m.ToolCalls) > 0 {
			names := make([]string, len(m.ToolCalls))
			for i, tc := range m.ToolCalls {
				names[i] = tc.Name
			}
			m.Content = "Calling: " + strings.Join(names, ", ")
		}
		// Fix consecutive same-role messages: insert a placeholder between them.
		if len(sanitized) > 0 && m.Role != RoleSystem {
			prev := sanitized[len(sanitized)-1]
			if prev.Role == m.Role {
				if m.Role == RoleUser {
					sanitized = append(sanitized, Message{Role: RoleModel, Content: "Understood."})
				} else if m.Role == RoleModel {
					sanitized = append(sanitized, Message{Role: RoleUser, Content: "Continue."})
				}
			}
		}
		sanitized = append(sanitized, m)
	}
	return sanitized
}

// flattenHistory converts tool-call exchanges into plain text messages.
// This avoids the XML template issues entirely on retries.
func (p *OllamaProvider) flattenHistory(history []Message) []Message {
	flat := make([]Message, 0, len(history))
	for _, msg := range history {
		switch {
		case msg.Role == RoleModel && len(msg.ToolCalls) > 0:
			// Convert assistant tool-call message to plain text
			parts := make([]string, 0, len(msg.ToolCalls))
			for _, tc := range msg.ToolCalls {
				parts = append(parts, fmt.Sprintf("I called %s(%s)", tc.Name, tc.Arguments))
			}
			content := strings.Join(parts, ". ")
			if msg.Content != "" {
				content = msg.Content + " " + content
			}
			flat = append(flat, Message{Role: RoleModel, Content: escapeXMLContent(content)})
		case msg.Role == RoleTool:
			// Convert tool response to user message with result
			flat = append(flat, Message{
				Role:    RoleUser,
				Content: escapeXMLContent(fmt.Sprintf("Result of %s: %s", msg.Name, msg.Content)),
			})
		default:
			m := msg
			m.Content = escapeXMLContent(m.Content)
			flat = append(flat, m)
		}
	}
	// Merge consecutive same-role messages
	merged := make([]Message, 0, len(flat))
	for _, m := range flat {
		if len(merged) > 0 && merged[len(merged)-1].Role == m.Role {
			merged[len(merged)-1].Content += "\n" + m.Content
		} else {
			merged = append(merged, m)
		}
	}
	return merged
}

// truncateHistory keeps system prompt + the last keepLast messages.
func (p *OllamaProvider) truncateHistory(history []Message, keepLast int) []Message {
	if len(history) <= keepLast+1 {
		return history
	}
	result := make([]Message, 0, keepLast+1)
	if len(history) > 0 && history[0].Role == RoleSystem {
		result = append(result, history[0])
	}
	start := len(history) - keepLast
	if start < 1 {
		start = 1
	}
	result = append(result, history[start:]...)
	return result
}

func (p *OllamaProvider) ChatCompletion(ctx context.Context, history []Message, tools []ToolDefinition, onDelta func(string)) (*Response, error) {
	var oTools []ollamaTool
	if len(tools) > 0 {
		for _, t := range tools {
			// Fix parameters for Qwen's XML template:
			// - nil parameters cause malformed XML
			// - empty properties ({}) confuse the model when it needs to pass arguments
			// In both cases, use additionalProperties so the model knows it can pass any args.
			params := t.Parameters
			if params == nil {
				params = map[string]interface{}{
					"type":                 "object",
					"properties":           map[string]interface{}{},
					"additionalProperties": true,
				}
			} else if m, ok := params.(map[string]interface{}); ok {
				if props, exists := m["properties"]; exists {
					if pm, ok := props.(map[string]interface{}); ok && len(pm) == 0 {
						m["additionalProperties"] = true
					}
				}
			}
			t.Parameters = params
			oTools = append(oTools, ollamaTool{
				Type:     "function",
				Function: t,
			})
		}
	}

	maxRetries := 5
	var lastErr error
	xmlErrorCount := 0

	temps := []float64{0.0, 0.3, 0.5, 0.7, 0.9}

	for attempt := 1; attempt <= maxRetries; attempt++ {
		currentTemp := temps[0]
		if attempt <= len(temps) {
			currentTemp = temps[attempt-1]
		}

		// Sanitize history to fix patterns that break Qwen's XML template
		msgs := p.sanitizeHistory(history)

		// On the first XML error, flatten history to remove all tool_call XML
		// structures and truncate aggressively. This avoids the XML template
		// entirely and keeps context small enough to prevent truncation issues.
		sendTools := oTools
		if xmlErrorCount >= 1 {
			msgs = p.flattenHistory(history)
			msgs = p.truncateHistory(msgs, 4)
		}

		reqPayload := ollamaRequest{
			Model:    p.Model,
			Messages: toOllamaMessages(msgs),
			Tools:    sendTools,
			Stream:   onDelta != nil,
			Options: map[string]interface{}{
				"num_ctx":     p.ContextSize,
				"temperature": currentTemp,
				"num_predict": 4096,
			},
		}

		var reqBuf bytes.Buffer
		reqEnc := json.NewEncoder(&reqBuf)
		reqEnc.SetEscapeHTML(false)
		if err := reqEnc.Encode(reqPayload); err != nil {
			return nil, fmt.Errorf("error marshalling request: %w", err)
		}
		jsonBody := reqBuf.Bytes()

		url := fmt.Sprintf("%s/api/chat", p.BaseURL)
		req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonBody))
		if err != nil {
			return nil, fmt.Errorf("error creating request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")
		req.Close = true // Force close header on client side

		// Debug timing log
		fmt.Printf("⏳ [Attempt %d | T=%.1f] Sending to %s... ", attempt, currentTemp, p.Model)
		start := time.Now()

		resp, err := p.Client.Do(req)
		if err != nil {
			fmt.Println("❌ Network error!")
			return nil, fmt.Errorf("connection error: %w", err)
		}
		defer resp.Body.Close()

		if onDelta == nil {
			body, _ := io.ReadAll(resp.Body)
			fmt.Printf("Received in %.2fs! ⚡️\n", time.Since(start).Seconds())

			// Error handling
			if resp.StatusCode != 200 {
				errMsg := string(body)
				isXmlError := strings.Contains(errMsg, "XML syntax error") || strings.Contains(errMsg, "unexpected end element")

				if resp.StatusCode == 500 && isXmlError {
					// Log the request body that caused the error
					if p.Debug {
						fmt.Printf("\n📝 [DEBUG] Request Body (Glitch T=%d):\n%s\n\n", attempt, string(jsonBody))
					}
					fmt.Printf("⚠️  Ollama glitch: %s. Retrying in 1s...\n", strings.TrimSpace(errMsg))
					lastErr = fmt.Errorf("Ollama XML Error: %s", errMsg)
					xmlErrorCount++
					time.Sleep(1 * time.Second)
					continue
				}

				return nil, fmt.Errorf("Ollama API error (%d): %s", resp.StatusCode, errMsg)
			}

			// Parse response
			var oResp ollamaResponse
			if err := json.Unmarshal(body, &oResp); err != nil {
				return nil, fmt.Errorf("error decoding Ollama JSON: %w", err)
			}

			msg := Message{
				Role:    oResp.Message.Role,
				Content: oResp.Message.Content,
			}

			if len(oResp.Message.ToolCalls) > 0 {
				for _, tc := range oResp.Message.ToolCalls {
					// Un-escape XML entities in argument values. The model may
					// reproduce &amp; / &lt; / &gt; it saw in the rendered template.
					unescapeMapValues(tc.Function.Arguments)

					// Use SetEscapeHTML(false) so & stays as real & (not \u0026).
					var argsBuf bytes.Buffer
					argEnc := json.NewEncoder(&argsBuf)
					argEnc.SetEscapeHTML(false)
					if err := argEnc.Encode(tc.Function.Arguments); err != nil {
						argsBuf.Reset()
						argsBuf.WriteString("{}")
					}

					msg.ToolCalls = append(msg.ToolCalls, ToolCall{
						ID:        fmt.Sprintf("call_%d", time.Now().UnixNano()),
						Type:      "function",
						Name:      tc.Function.Name,
						Arguments: strings.TrimSpace(argsBuf.String()),
					})
				}
			}

			return &Response{Message: msg}, nil
		}

		if resp.StatusCode != 200 {
			body, _ := io.ReadAll(resp.Body)
			errMsg := string(body)
			isXmlError := strings.Contains(errMsg, "XML syntax error") || strings.Contains(errMsg, "unexpected end element")

			if resp.StatusCode == 500 && isXmlError {
				if p.Debug {
					fmt.Printf("\n📝 [DEBUG] Request Body (Glitch T=%d):\n%s\n\n", attempt, string(jsonBody))
				}
				fmt.Printf("⚠️  Ollama glitch: %s. Retrying in 1s...\n", strings.TrimSpace(errMsg))
				lastErr = fmt.Errorf("Ollama XML Error: %s", errMsg)
				xmlErrorCount++
				time.Sleep(1 * time.Second)
				continue
			}

			return nil, fmt.Errorf("Ollama API error (%d): %s", resp.StatusCode, errMsg)
		}

		fmt.Printf("Streaming in %.2fs... ⚡️\n", time.Since(start).Seconds())

		var aggregated ollamaResponse
		aggregated.Message.Role = RoleModel
		decoder := json.NewDecoder(resp.Body)
		for {
			var chunk ollamaResponse
			if err := decoder.Decode(&chunk); err != nil {
				if err == io.EOF {
					break
				}
				return nil, fmt.Errorf("error decoding Ollama stream: %w", err)
			}

			if aggregated.Message.Role == "" && chunk.Message.Role != "" {
				aggregated.Message.Role = chunk.Message.Role
			}
			if chunk.Message.Content != "" {
				aggregated.Message.Content += chunk.Message.Content
				onDelta(chunk.Message.Content)
			}
			if len(chunk.Message.ToolCalls) > 0 {
				aggregated.Message.ToolCalls = chunk.Message.ToolCalls
			}
			if chunk.Done {
				break
			}
		}

		msg := Message{
			Role:    aggregated.Message.Role,
			Content: aggregated.Message.Content,
		}

		if len(aggregated.Message.ToolCalls) > 0 {
			for _, tc := range aggregated.Message.ToolCalls {
				unescapeMapValues(tc.Function.Arguments)

				var argsBuf bytes.Buffer
				argEnc := json.NewEncoder(&argsBuf)
				argEnc.SetEscapeHTML(false)
				if err := argEnc.Encode(tc.Function.Arguments); err != nil {
					argsBuf.Reset()
					argsBuf.WriteString("{}")
				}

				msg.ToolCalls = append(msg.ToolCalls, ToolCall{
					ID:        fmt.Sprintf("call_%d", time.Now().UnixNano()),
					Type:      "function",
					Name:      tc.Function.Name,
					Arguments: strings.TrimSpace(argsBuf.String()),
				})
			}
		}

		return &Response{Message: msg}, nil
	}

	return nil, fmt.Errorf("failed after %d attempts. Last error: %v", maxRetries, lastErr)
}
