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
	Client      *http.Client
}

func NewOllama(baseURL, model string, ctxSize int) *OllamaProvider {
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
	Messages []Message              `json:"messages"`
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

// sanitizeHistory fixes message patterns that confuse Qwen's XML tool-call template.
func (p *OllamaProvider) sanitizeHistory(history []Message) []Message {
	sanitized := make([]Message, 0, len(history))
	for _, msg := range history {
		m := msg
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
			flat = append(flat, Message{Role: RoleModel, Content: content})
		case msg.Role == RoleTool:
			// Convert tool response to user message with result
			flat = append(flat, Message{
				Role:    RoleUser,
				Content: fmt.Sprintf("Result of %s: %s", msg.Name, msg.Content),
			})
		default:
			flat = append(flat, msg)
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

func (p *OllamaProvider) ChatCompletion(ctx context.Context, history []Message, tools []ToolDefinition) (*Response, error) {
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

		// After 2 XML errors, flatten history to remove all tool_call XML structures.
		// This converts tool exchanges to plain text, avoiding the XML template entirely.
		sendTools := oTools
		if xmlErrorCount >= 2 {
			msgs = p.flattenHistory(history)
			msgs = p.truncateHistory(msgs, 4)
		} else if xmlErrorCount >= 1 {
			msgs = p.truncateHistory(msgs, 6)
		}

		reqPayload := ollamaRequest{
			Model:    p.Model,
			Messages: msgs,
			Tools:    sendTools,
			Stream:   false,
			Options: map[string]interface{}{
				"num_ctx":     p.ContextSize,
				"temperature": currentTemp,
				"num_predict": 4096,
			},
		}

		jsonBody, err := json.Marshal(reqPayload)
		if err != nil {
			return nil, fmt.Errorf("error marshalling request: %w", err)
		}

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

		body, _ := io.ReadAll(resp.Body)
		fmt.Printf("Received in %.2fs! ⚡️\n", time.Since(start).Seconds())

		// Error handling
		if resp.StatusCode != 200 {
			errMsg := string(body)
			isXmlError := strings.Contains(errMsg, "XML syntax error") || strings.Contains(errMsg, "unexpected end element")

			if resp.StatusCode == 500 && isXmlError {
				// Log the request body that caused the error
				fmt.Printf("\n📝 [DEBUG] Request Body (Glitch T=%d):\n%s\n\n", attempt, string(jsonBody))

				fmt.Printf("\n📝🤖 [DEBUG] Response Body (Glitch T=%d):\n%s\n\n", attempt, string(body))

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
				argsBytes, err := json.Marshal(tc.Function.Arguments)
				if err != nil {
					argsBytes = []byte("{}")
				}

				msg.ToolCalls = append(msg.ToolCalls, ToolCall{
					ID:        fmt.Sprintf("call_%d", time.Now().UnixNano()),
					Type:      "function",
					Name:      tc.Function.Name,
					Arguments: string(argsBytes),
				})
			}
		}

		return &Response{Message: msg}, nil
	}

	return nil, fmt.Errorf("failed after %d attempts. Last error: %v", maxRetries, lastErr)
}
