package llm

import "context"

// Standard roles
const (
	RoleUser   = "user"
	RoleSystem = "system"
	RoleModel  = "assistant"
	RoleTool   = "tool"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	// Name is used when role is 'tool'
	Name       string     `json:"name,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"` // Used when role is 'tool'
}

type ToolCall struct {
	ID        string `json:"id"`
	Type      string `json:"type"`
	Name      string `json:"name"`      // Extracted from function.name
	Arguments string `json:"arguments"` // JSON string
}

// ToolDefinition is the definition we pass to the LLM (JSON Schema)
type ToolDefinition struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Parameters  interface{} `json:"parameters"`            // JSON Schema
	InputSchema interface{} `json:"inputSchema,omitempty"` // MCP uses this field name
}

// Provider is the interface that every AI client must implement
type Provider interface {
	ChatCompletion(ctx context.Context, history []Message, tools []ToolDefinition, onDelta func(string)) (*Response, error)
}

type Response struct {
	Message Message
}
