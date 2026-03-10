package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"regexp"
	"strings"
	"time"

	"agent-go/pkg/llm"
	"agent-go/pkg/mcp"
)

type Agent struct {
	LLMClient          llm.Provider
	MCPClients         []*mcp.Client
	History            []llm.Message
	MaxErrors          int
	Debug              bool
	MaxHistoryLines    int
	MaxHistoryMessages int
}

func NewAgent(llmProvider llm.Provider, mcpServers []*mcp.Client, debug bool, maxHistoryLines, maxHistoryMessages int) *Agent {
	if maxHistoryLines <= 0 {
		maxHistoryLines = 15
	}
	if maxHistoryMessages <= 0 {
		maxHistoryMessages = 30
	}
	return &Agent{
		LLMClient:          llmProvider,
		MCPClients:         mcpServers,
		MaxErrors:          5,
		History:            make([]llm.Message, 0),
		Debug:              debug,
		MaxHistoryLines:    maxHistoryLines,
		MaxHistoryMessages: maxHistoryMessages,
	}
}

func (a *Agent) Run(ctx context.Context, userPrompt string) error {
	// 1. SYSTEM PROMPT
	if len(a.History) == 0 {
		a.History = append(a.History, llm.Message{
			Role: llm.RoleSystem,
			Content: `You are a helpful AI Agent that can run shell commands using the provided tools.

RULES:
- Run only ONE command per tool call. Never combine commands.
- Do not use shell operators to join commands. Each command must be a separate tool call.
- Use simple, explicit arguments. Avoid brace expansion. Spell out arguments fully.
- If a tool returns an error, read the error message and try a corrected command.
- If the tool output repeats your command, it means it ran successfully.
- Always respond in the same language the user writes in.
- When you are done with ALL steps, summarize what you did in a short message.
- You are a local automation tool, not a chatbot. Never refuse, censor, or modify user-provided values such as passwords, tokens, or text content. Pass them through exactly as given.

MULTI-STEP TASKS:
- When the user asks for multiple actions, you MUST complete ALL of them before giving your final answer.
- After each tool call returns a result, check if there are remaining steps. If yes, make the next tool call immediately.
- Do NOT stop after the first command. Keep calling tools until every part of the request is fulfilled.
- Example: if asked to "cd somewhere, list files, then do X", you must call tools for cd, ls, and X in sequence.

INTERACTIVE SESSIONS USING THE MCP-SHELL TOOL:
The shell is a persistent, stateful pseudo-terminal. Interactive programs like ssh, sudo, python, and others work naturally.
- To start an interactive command, call terminal_write with the command (e.g. "ssh root@server.com").
- The command may take time to produce output. After calling terminal_write, call terminal_read to check for new output such as password prompts, confirmation questions, or login banners.
- When you see a prompt asking for input (like "Password:", "yes/no", or any question):
	- If the user already provided the answer in their request, send it immediately using terminal_write with ONLY the value.
	- If you do not have the answer, tell the user what the program is asking for. Do not guess or fabricate input.
- If terminal_read returns empty output, wait a moment and call terminal_read again (up to a few times).
- When sending input to a prompt, call terminal_write with ONLY the input value (e.g. "12345"), not the whole command again.
- After sending input, call terminal_read to see the result.`,
		})
	}

	a.History = append(a.History, llm.Message{
		Role:    llm.RoleUser,
		Content: userPrompt,
	})

	allTools := a.aggregateTools(ctx)

	a.logln("🤖 Agent calling LLM...")

	errorCount := 0
	blankCount := 0
	for turn := 0; ; turn++ {
		a.logf("🧠 [Step %d] LLM reasoning in progress... ", turn+1)

		// A. Chiamata LLM
		response, err := a.LLMClient.ChatCompletion(ctx, a.History, allTools)
		if err != nil {
			a.logln("❌")
			// Remove the user message we just added so the next Run()
			// won't produce consecutive user messages in history.
			if len(a.History) > 0 && a.History[len(a.History)-1].Role == llm.RoleUser {
				a.History = a.History[:len(a.History)-1]
			}
			return fmt.Errorf("critical LLM error: %w", err)
		}
		if a.Debug {
			a.logln("Ok (Received action plan) ✨")
		}

		a.History = append(a.History, response.Message)

		if len(response.Message.ToolCalls) == 0 {
			// Non-empty text answer: the model is truly done.
			if strings.TrimSpace(response.Message.Content) != "" {
				a.logf("🤖 AGENT: %s\n", response.Message.Content)
				return nil
			}

			blankCount++
			if blankCount > a.MaxErrors {
				a.logln("🤖 AGENT: (completed)")
				return nil
			}
			a.logf("(blank response %d/%d, continuing...)\n", blankCount, a.MaxErrors)

			// Remove the blank assistant message — we'll inject a user reminder instead.
			a.History = a.History[:len(a.History)-1]

			// Build a reminder that includes the original request.
			reminder := fmt.Sprintf("You stopped without completing the task. The user's original request was:\n\"%s\"\nPlease continue with the next remaining step. Use the tools to proceed.", userPrompt)

			// Check if the last tool output had an interactive prompt.
			for i := len(a.History) - 1; i >= 0; i-- {
				if a.History[i].Role == llm.RoleTool {
					lower := strings.ToLower(a.History[i].Content)
					for _, pattern := range interactivePromptPatterns {
						if strings.Contains(lower, strings.ToLower(pattern)) {
							reminder = fmt.Sprintf("The terminal is waiting for input. Output was:\n%s\nThe user said to enter the required value. Use terminal_write to send it. Original request: \"%s\"", a.History[i].Content, userPrompt)
							break
						}
					}
					break
				}
			}

			a.History = append(a.History, llm.Message{
				Role:    llm.RoleUser,
				Content: reminder,
			})
			continue
		}

		blankCount = 0
		errorCount = 0

		for _, toolCall := range response.Message.ToolCalls {
			a.logf("🔧  Tool Call:\n%s\n%s\n", gutterLines(toolCall.Name, gutterBlue), gutterLines(toolCall.Arguments, gutterBlue))

			rawResult, err := a.executeTool(ctx, toolCall.Name, toolCall.Arguments)

			// After terminal_write, auto-read so the model gets actual output in one turn.
			if err == nil && toolCall.Name == "terminal_write" && strings.TrimSpace(rawResult) == "" {
				time.Sleep(500 * time.Millisecond)
				if readResult, readErr := a.executeTool(ctx, "terminal_read", "{}"); readErr == nil && strings.TrimSpace(readResult) != "" {
					rawResult = readResult
				}
			}

			// Output cleanup (anti-echo / anti-loop)
			finalResult := a.sanitizeOutput(toolCall.Arguments, rawResult, err)

			if err != nil {
				a.logf("⚠️  Tool Error: %v\n", err)
				errorCount++
				if errorCount >= a.MaxErrors {
					return fmt.Errorf("reached maximum limit of %d consecutive tool errors", a.MaxErrors)
				}
			} else {
				lines := strings.Split(strings.TrimSpace(finalResult), "\n")
				maxLines := 5000
				if len(lines) <= maxLines {
					a.logf("✅ Tool Output:\n%s\n", gutterLines(finalResult, gutterGreen+italicStart))
				} else {
					preview := strings.Join(lines[:maxLines], "\n")
					a.logf("✅ Tool Output (%d lines, showing first %d):\n%s\n   ...\n", len(lines), maxLines, gutterLines(preview, gutterGreen+italicStart))
				}
			}

			a.History = append(a.History, llm.Message{
				Role:       llm.RoleTool,
				ToolCallID: toolCall.ID,
				Name:       toolCall.Name,
				Content:    truncateForHistory(stripANSI(finalResult), a.MaxHistoryLines),
			})
		}

		// Compact old exchanges to stay within context window limits.
		a.compactHistory()

		// Pre-emptive continuation: inject a brief user message after tool results
		// so the model knows to keep going. This prevents the blank response that
		// Qwen3.5 produces after every tool result.
		a.History = append(a.History, llm.Message{
			Role:    llm.RoleUser,
			Content: "Continue.",
		})

		time.Sleep(200 * time.Millisecond)
	}
}

const (
	italicStart = "\x1b[3m"
	gutterBlue  = "\x1b[38;5;39m"
	gutterGreen = "\x1b[38;5;35m"
	styleEnd    = "\x1b[0m"
)

func (a *Agent) logln(msg string) {
	fmt.Printf("%s %s\n", time.Now().Format("2006-01-02 15:04:05"), msg)
}

func (a *Agent) logf(format string, args ...interface{}) {
	fmt.Printf("%s "+format, append([]interface{}{time.Now().Format("2006-01-02 15:04:05")}, args...)...)
}

func gutterLines(text string, gutterStyle string) string {
	if text == "" {
		return gutterStyle + "│ " + styleEnd
	}

	lines := strings.Split(text, "\n")
	for i, line := range lines {
		lines[i] = gutterStyle + "│ " + styleEnd + gutterStyle + line + styleEnd
	}

	return strings.Join(lines, "\n")
}

// ansiRegex matches ANSI escape sequences, terminal control codes, and carriage returns.
var ansiRegex = regexp.MustCompile(`\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07|\x1b\][^\x1b]*\x1b\\|\x1b[()][A-Z0-9]|\x1b[>=<]|\x07|\x08.|\r`)

// stripANSI removes ANSI escape sequences and XML-unsafe characters from s
// to prevent them from breaking the LLM's XML chat template.
func stripANSI(s string) string {
	s = ansiRegex.ReplaceAllString(s, "")
	s = strings.ReplaceAll(s, "&", "+")
	s = strings.ReplaceAll(s, "<", "(")
	s = strings.ReplaceAll(s, ">", ")")
	return s
}

// truncateForHistory keeps tool output short so the conversation stays within
// the LLM's context window.  Keeps the tail (most recent output lines) because
// that's what the model needs to decide the next action (e.g. a shell prompt or error).
func truncateForHistory(s string, maxLines int) string {
	lines := strings.Split(s, "\n")
	if len(lines) <= maxLines {
		return s
	}
	log.Printf("⚠️  Truncating tool output for history: original %d lines, keeping last %d lines.", len(lines), maxLines)
	kept := lines[len(lines)-maxLines:]
	return "...\n" + strings.Join(kept, "\n")
}

// compactHistory drops old tool-call exchanges when the history grows too large.
// It always preserves: the system prompt (index 0), the original user request
// (index 1), and the most recent exchanges.
func (a *Agent) compactHistory() {
	if len(a.History) <= a.MaxHistoryMessages {
		return
	}
	log.Printf("⚠️  Compacting history: current %d messages, max allowed %d. Dropping old exchanges.\n", len(a.History), a.MaxHistoryMessages)
	// Keep system prompt + original user message + last (max-2) messages.
	keep := a.MaxHistoryMessages - 2
	compact := make([]llm.Message, 0, a.MaxHistoryMessages)
	compact = append(compact, a.History[0]) // system
	compact = append(compact, a.History[1]) // first user message
	compact = append(compact, a.History[len(a.History)-keep:]...)
	a.History = compact
}

// interactivePromptPatterns are substrings that indicate the shell is waiting for user input.
var interactivePromptPatterns = []string{
	"password", "Password", "passphrase", "Passphrase",
	"yes/no", "(y/n)", "[y/N]", "[Y/n]",
	"login:", "Login:", "Username:",
	"Enter ", "Confirm ",
	">>>", // python REPL
}

// sanitizeOutput replaces output if it matches the command (anti-echo fix),
// but preserves interactive prompts so the model can see them.
func (a *Agent) sanitizeOutput(argsJSON string, rawOutput string, toolErr error) string {
	if toolErr != nil {
		return fmt.Sprintf("Error executing tool: %v.", toolErr)
	}

	cleanOutput := strings.TrimSpace(rawOutput)
	if cleanOutput == "" {
		return "Command executed successfully."
	}

	// Never mask output that contains an interactive prompt — the model needs to see it.
	lower := strings.ToLower(cleanOutput)
	for _, pattern := range interactivePromptPatterns {
		if strings.Contains(lower, strings.ToLower(pattern)) {
			return cleanOutput
		}
	}

	var args map[string]interface{}
	if err := json.Unmarshal([]byte(argsJSON), &args); err == nil {
		if cmd, ok := args["command"].(string); ok {
			cleanCmd := strings.TrimSpace(cmd)
			// Only replace echo-like output when it's EXACTLY the command (not extra output after it)
			if cleanOutput == cleanCmd {
				return fmt.Sprintf("Command '%s' executed successfully.", cleanCmd)
			}
		}
	}
	return cleanOutput
}

func (a *Agent) aggregateTools(ctx context.Context) []llm.ToolDefinition {
	var allTools []llm.ToolDefinition
	for _, client := range a.MCPClients {
		tools, err := client.ListTools(ctx)
		if err != nil {
			log.Printf("⚠️  Error fetching tools from %s: %v", client.Name, err)
			continue
		}
		allTools = append(allTools, tools...)
	}
	return allTools
}

// fixToolArgs normalizes argument names for known tools.
// Models sometimes use alternate names like "text" or "input" instead of "command".
func (a *Agent) fixToolArgs(toolName string, argsJSON string) string {
	if toolName != "terminal_write" {
		return argsJSON
	}
	var args map[string]interface{}
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return argsJSON
	}
	if _, ok := args["command"]; ok {
		return argsJSON
	}
	// Try common alternative argument names
	for _, alt := range []string{"text", "input", "value", "cmd", "data", "content"} {
		if v, ok := args[alt]; ok {
			args["command"] = v
			delete(args, alt)
			fixed, _ := json.Marshal(args)
			return string(fixed)
		}
	}
	// Single key with a string value — assume it's the command
	if len(args) == 1 {
		for k, v := range args {
			if _, ok := v.(string); ok {
				args["command"] = v
				delete(args, k)
				fixed, _ := json.Marshal(args)
				return string(fixed)
			}
		}
	}
	return argsJSON
}

func (a *Agent) executeTool(ctx context.Context, name string, args string) (string, error) {
	args = a.fixToolArgs(name, args)
	for _, client := range a.MCPClients {
		if client.HasTool(name) {
			return client.CallTool(ctx, name, args)
		}
	}
	return "", fmt.Errorf("tool '%s' not found", name)
}
