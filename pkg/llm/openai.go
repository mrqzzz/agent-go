package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
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

func (p *OpenAIProvider) ChatCompletion(ctx context.Context, history []Message, tools []ToolDefinition) (*Response, error) {
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
