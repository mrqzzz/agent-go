package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
)

type OpenAIProvider struct {
	APIKey string
	Model  string
	Client *http.Client
}

func NewOpenAI(apiKey, model string) *OpenAIProvider {
	return &OpenAIProvider{
		APIKey: apiKey,
		Model:  model,
		Client: &http.Client{},
	}
}

// Internal structures for OpenAI payload
type openAIRequest struct {
	Model    string       `json:"model"`
	Messages []Message    `json:"messages"`
	Tools    []openAITool `json:"tools,omitempty"`
	Stream   bool         `json:"stream,omitempty"`
}

type openAITool struct {
	Type     string         `json:"type"`
	Function ToolDefinition `json:"function"`
}

type openAIResponse struct {
	Choices []struct {
		Message struct {
			Role      string  `json:"role"`
			Content   *string `json:"content"` // Pointer because it can be null
			ToolCalls []struct {
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"message"`
	} `json:"choices"`
}

type openAIStreamChunk struct {
	Choices []struct {
		Delta struct {
			Role      string  `json:"role"`
			Content   *string `json:"content"`
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
	} `json:"choices"`
}

func (p *OpenAIProvider) ChatCompletion(ctx context.Context, history []Message, tools []ToolDefinition, onDelta func(string)) (*Response, error) {
	// 1. Prepare tools in OpenAI format
	var oaTools []openAITool
	if len(tools) > 0 {
		for _, t := range tools {
			oaTools = append(oaTools, openAITool{
				Type:     "function",
				Function: t,
			})
		}
	}

	reqBody := openAIRequest{
		Model:    p.Model,
		Messages: history,
		Tools:    oaTools,
		Stream:   onDelta != nil,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.openai.com/v1/chat/completions", bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.APIKey)

	resp, err := p.Client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if onDelta != nil {
		if resp.StatusCode != 200 {
			body, _ := io.ReadAll(resp.Body)
			return nil, fmt.Errorf("OpenAI API error: %s - %s", resp.Status, string(body))
		}

		msg := Message{}
		toolByIndex := make(map[int]*ToolCall)

		scanner := bufio.NewScanner(resp.Body)
		buf := make([]byte, 0, 64*1024)
		scanner.Buffer(buf, 1024*1024)

		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" || strings.HasPrefix(line, ":") {
				continue
			}
			if !strings.HasPrefix(line, "data:") {
				continue
			}

			payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if payload == "[DONE]" {
				break
			}

			var chunk openAIStreamChunk
			if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
				return nil, fmt.Errorf("failed to decode OpenAI stream chunk: %w", err)
			}

			for _, choice := range chunk.Choices {
				d := choice.Delta
				if d.Role != "" && msg.Role == "" {
					msg.Role = d.Role
				}
				if d.Content != nil {
					msg.Content += *d.Content
					onDelta(*d.Content)
				}
				for _, tc := range d.ToolCalls {
					entry, ok := toolByIndex[tc.Index]
					if !ok {
						entry = &ToolCall{}
						toolByIndex[tc.Index] = entry
					}
					if tc.ID != "" {
						entry.ID = tc.ID
					}
					if tc.Type != "" {
						entry.Type = tc.Type
					}
					if tc.Function.Name != "" {
						entry.Name = tc.Function.Name
					}
					if tc.Function.Arguments != "" {
						entry.Arguments += tc.Function.Arguments
					}
				}
			}
		}

		if err := scanner.Err(); err != nil {
			return nil, fmt.Errorf("OpenAI stream read error: %w", err)
		}

		if msg.Role == "" {
			msg.Role = RoleModel
		}

		if len(toolByIndex) > 0 {
			indices := make([]int, 0, len(toolByIndex))
			for idx := range toolByIndex {
				indices = append(indices, idx)
			}
			sort.Ints(indices)
			for _, idx := range indices {
				tc := toolByIndex[idx]
				if tc.ID == "" {
					tc.ID = fmt.Sprintf("call_%d", idx)
				}
				if tc.Type == "" {
					tc.Type = "function"
				}
				msg.ToolCalls = append(msg.ToolCalls, *tc)
			}
		}

		return &Response{Message: msg}, nil
	}

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("OpenAI API error: %s - %s", resp.Status, string(body))
	}

	var oaResp openAIResponse
	if err := json.Unmarshal(body, &oaResp); err != nil {
		return nil, err
	}

	if len(oaResp.Choices) == 0 {
		return nil, fmt.Errorf("no choices returned by OpenAI")
	}

	// Convert OpenAI response -> our Message struct
	choice := oaResp.Choices[0].Message
	msg := Message{
		Role: choice.Role,
	}
	if choice.Content != nil {
		msg.Content = *choice.Content
	}

	for _, tc := range choice.ToolCalls {
		msg.ToolCalls = append(msg.ToolCalls, ToolCall{
			ID:        tc.ID,
			Type:      tc.Type,
			Name:      tc.Function.Name,
			Arguments: tc.Function.Arguments,
		})
	}

	return &Response{Message: msg}, nil
}
