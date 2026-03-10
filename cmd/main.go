package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
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
		aiProvider = llm.NewOllama(host, cfg.AI.Model, cfg.AI.ContextSize)
		logf("🦙 AI Provider: Ollama (Host: %s, Ctx: %d)\n", host, cfg.AI.ContextSize)
	default:
		log.Fatalf("❌ Unsupported AI provider: '%s'. Use 'openai' or 'ollama'.", cfg.AI.Provider)
	}

	// 4. Create agent
	ag := agent.NewAgent(aiProvider, mcpClients, *debug)

	// 5. Interactive loop (CLI)
	printWelcomeMessage()

	rl, err := readline.NewEx(&readline.Config{
		Prompt:          "👤 You > ",
		HistoryFile:     "/tmp/agent-go-history",
		InterruptPrompt: "^C",
		EOFPrompt:       "exit",
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

		input = strings.TrimSpace(input)

		if input == "exit" || input == "quit" {
			logln("👋 Agent terminated. Goodbye!")
			break
		}

		if input == "" {
			continue
		}

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)

		startTime := time.Now()
		err = ag.Run(ctx, input)
		duration := time.Since(startTime)

		cancel()

		if err != nil {
			logf("❌ Execution error: %v\n", err)
		} else {
			logf("✨ Completed in %.2f seconds.\n", duration.Seconds())
		}
	}
}

func printWelcomeMessage() {
	logln(strings.Repeat("=", 60))
	logln("   🤖  AGENT-GO - READY")
	logln("   Type your request. The agent will use tools as needed.")
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
