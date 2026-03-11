package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
	"time"

	"github.com/chzyer/readline"

	"agent-go/pkg/agent"
	"agent-go/pkg/config"
	"agent-go/pkg/llm"
	"agent-go/pkg/mcp"
)

func main() {
	debug := flag.Bool("debug", false, "Enable debug logs")
	noHistory := flag.Bool("nohistory", false, "Disable saving readline history")
	flag.Parse()

	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	// 1. Load configuration
	logln("📂 Loading configuration...")
	cfg, err := config.LoadConfig("config.yaml")
	if err != nil {
		log.Fatalf("❌ Cannot load 'config.yaml'. Make sure the file exists.\nError: %v", err)
	}

	// 2. Initialize MCP servers
	var mcpClients []*mcp.Client
	if len(cfg.MCPServers) > 0 {
		logln("🔌 Connecting to MCP servers...")
		for name, serverCfg := range cfg.MCPServers {
			client, err := mcp.NewClient(name, serverCfg.Command, serverCfg.Args, *debug)
			if err != nil {
				log.Printf("⚠️  Warning: Cannot start MCP server '%s': %v", name, err)
				continue
			}
			mcpClients = append(mcpClients, client)
			logf("   ✅ Connected to: %s (cmd: %s)\n", name, serverCfg.Command)
		}
	} else {
		logln("ℹ️  No MCP servers configured. The agent will work in conversation-only mode.")
	}

	// 3. Initialize AI provider
	var aiProvider llm.Provider

	switch strings.ToLower(cfg.AI.Provider) {
	case "openai":
		if cfg.AI.APIKey == "" {
			log.Fatal("❌ Error: OpenAI requires an API key in config.yaml")
		}
		aiProvider = llm.NewOpenAI(cfg.AI.APIKey, cfg.AI.Model)
		logf("🧠 AI Provider: OpenAI (Model: %s)\n", cfg.AI.Model)

	case "ollama":
		host := "http://localhost:11434"
		if cfg.AI.APIKey != "" && cfg.AI.APIKey != "none" {
			host = cfg.AI.APIKey
		}
		aiProvider = llm.NewOllama(host, cfg.AI.Model, cfg.AI.ContextSize, *debug)
		logf("🦙 AI Provider: Ollama (Host: %s, Model: %s, Ctx: %d)\n", host, cfg.AI.Model, cfg.AI.ContextSize)
	default:
		log.Fatalf("❌ Unsupported AI provider: '%s'. Use 'openai' or 'ollama'.", cfg.AI.Provider)
	}

	// 4. Create agent
	ag := agent.NewAgent(aiProvider, mcpClients, *debug, cfg.AI.MaxHistoryLines, cfg.AI.MaxHistoryMessages)
	promptTimeout := 5 * time.Minute
	if strings.TrimSpace(cfg.AI.PromptTimeout) != "" {
		d, err := time.ParseDuration(cfg.AI.PromptTimeout)
		if err != nil {
			logf("⚠️  Invalid ai.prompt_timeout value %q, using default %s\n", cfg.AI.PromptTimeout, promptTimeout)
		} else if d > 0 {
			promptTimeout = d
		} else {
			logf("⚠️  ai.prompt_timeout must be > 0, using default %s\n", promptTimeout)
		}
	}
	logf("⏱️  Prompt timeout: %s\n", promptTimeout)

	if flag.NArg() > 0 {
		if err := runBatchMode(ag, flag.Arg(0), promptTimeout); err != nil {
			log.Fatalf("❌ Batch mode failed: %v", err)
		}
		return
	}

	// 5. Interactive loop (CLI)
	printWelcomeMessage()

	historyFile := "/tmp/agent-go-history"
	if *noHistory {
		historyFile = ""
	}
	defaultPrompt := "\033[38;5;202m▖\033[38;5;208m▗\033[38;5;214m▝\033[38;5;220m▞\033[38;5;226m█\033[0m "
	continuationPrompt := "\033[38;5;244m...\033[0m "
	continueLine := false
	collectedLines := make([]string, 0)

	rl, err := readline.NewEx(&readline.Config{
		Prompt:          defaultPrompt,
		HistoryFile:     historyFile,
		InterruptPrompt: "^C",
		EOFPrompt:       "exit",
		FuncFilterInputRune: func(r rune) (rune, bool) {
			// In most terminals Ctrl+Enter is delivered as Ctrl+J (LF).
			// Treat it as "continue to next line" rather than submit.
			if r == readline.CharCtrlJ {
				continueLine = true
				return readline.CharEnter, true
			}
			return r, true
		},
	})
	if err != nil {
		log.Fatalf("❌ Cannot initialize readline: %v", err)
	}
	defer rl.Close()

	for {
		input, err := rl.Readline()
		if err != nil {
			if err == readline.ErrInterrupt || err == io.EOF {
				logln("👋 Agent terminated. Goodbye!")
				break
			}
			log.Printf("Input read error: %v", err)
			continue
		}

		rawInput := input
		input = strings.TrimSpace(input)

		if continueLine {
			if input != "" {
				collectedLines = append(collectedLines, input)
			}
			continueLine = false
			rl.SetPrompt(continuationPrompt)
			continue
		}

		if len(collectedLines) > 0 {
			if input != "" {
				collectedLines = append(collectedLines, strings.TrimSpace(rawInput))
			}
			if err := runBatchLines(ag, collectedLines, promptTimeout); err != nil {
				logf("❌ Execution error: %v\n", err)
			}
			collectedLines = collectedLines[:0]
			rl.SetPrompt(defaultPrompt)
			continue
		}

		if input == "exit" || input == "quit" {
			logln("👋 Agent terminated. Goodbye!")
			break
		}

		if input == "" {
			continue
		}

		if err := runPrompt(ag, input, promptTimeout); err != nil {
			logf("❌ Execution error: %v\n", err)
		}
	}
}

func runBatchMode(ag *agent.Agent, promptFile string, promptTimeout time.Duration) error {
	file, err := os.Open(promptFile)
	if err != nil {
		return fmt.Errorf("cannot open prompt file %q: %w", promptFile, err)
	}
	defer file.Close()

	logf("📄 Running batch mode with prompt file: %s\n", promptFile)

	scanner := bufio.NewScanner(file)
	lines := make([]string, 0)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading prompt file %q: %w", promptFile, err)
	}

	if err := runBatchLines(ag, lines, promptTimeout); err != nil {
		return err
	}

	logln("✅ Batch mode completed.")
	return nil
}

func runBatchLines(ag *agent.Agent, lines []string, promptTimeout time.Duration) error {
	lineNumber := 0
	for _, line := range lines {
		lineNumber++
		prompt := strings.TrimSpace(line)
		if prompt == "" {
			continue
		}

		logf("➡️  Batch prompt %d: %s\n", lineNumber, prompt)
		if err := runPrompt(ag, prompt, promptTimeout); err != nil {
			return fmt.Errorf("line %d: %w", lineNumber, err)
		}
	}
	return nil
}

func runPrompt(ag *agent.Agent, input string, promptTimeout time.Duration) error {
	if promptTimeout <= 0 {
		promptTimeout = 5 * time.Minute
	}

	ctx, cancel := context.WithTimeout(context.Background(), promptTimeout)
	defer cancel()

	startTime := time.Now()
	err := ag.Run(ctx, input)
	duration := time.Since(startTime)

	if err != nil {
		return err
	}

	logf("\n✨ Completed in %.2f seconds.\n", duration.Seconds())
	return nil
}

func printWelcomeMessage() {
	logln(strings.Repeat("=", 60))
	logln("   🤖  AGENT-GO - READY")
	logln("   Type your request. The agent will use tools as needed.")
	logln("   Use Ctrl+J (Ctrl+Enter in many terminals) for multiline input.")
	logln("   Type 'exit' to quit.")
	logln(strings.Repeat("=", 60))
}

func ts() string {
	return time.Now().Format("2006-01-02 15:04:05")
}

func logln(msg string) {
	fmt.Printf("%s %s\n", ts(), msg)
}

func logf(format string, args ...interface{}) {
	fmt.Printf("%s "+format, append([]interface{}{ts()}, args...)...)
}
