package main

import (
	"context"
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
	log.SetFlags(log.Ltime)

	// 1. Load configuration
	fmt.Println("📂 Loading configuration...")
	cfg, err := config.LoadConfig("config.yaml")
	if err != nil {
		log.Fatalf("❌ Cannot load 'config.yaml'. Make sure the file exists.\nError: %v", err)
	}

	// 2. Initialize MCP servers
	var mcpClients []*mcp.Client
	if len(cfg.MCPServers) > 0 {
		fmt.Println("🔌 Connecting to MCP servers...")
		for name, serverCfg := range cfg.MCPServers {
			client, err := mcp.NewClient(name, serverCfg.Command, serverCfg.Args)
			if err != nil {
				log.Printf("⚠️  Warning: Cannot start MCP server '%s': %v", name, err)
				continue
			}
			mcpClients = append(mcpClients, client)
			fmt.Printf("   ✅ Connected to: %s (cmd: %s)\n", name, serverCfg.Command)
		}
	} else {
		fmt.Println("ℹ️  No MCP servers configured. The agent will work in conversation-only mode.")
	}

	// 3. Initialize AI provider
	var aiProvider llm.Provider

	switch strings.ToLower(cfg.AI.Provider) {
	case "openai":
		if cfg.AI.APIKey == "" {
			log.Fatal("❌ Error: OpenAI requires an API key in config.yaml")
		}
		aiProvider = llm.NewOpenAI(cfg.AI.APIKey, cfg.AI.Model)
		fmt.Printf("🧠 AI Provider: OpenAI (Model: %s)\n", cfg.AI.Model)

	case "ollama":
		host := "http://localhost:11434"
		if cfg.AI.APIKey != "" && cfg.AI.APIKey != "none" {
			host = cfg.AI.APIKey
		}
		aiProvider = llm.NewOllama(host, cfg.AI.Model, cfg.AI.ContextSize)
		fmt.Printf("🦙 AI Provider: Ollama (Host: %s, Ctx: %d)\n", host, cfg.AI.ContextSize)
	default:
		log.Fatalf("❌ Unsupported AI provider: '%s'. Use 'openai' or 'ollama'.", cfg.AI.Provider)
	}

	// 4. Create agent
	ag := agent.NewAgent(aiProvider, mcpClients)

	// 5. Interactive loop (CLI)
	printWelcomeMessage()

	rl, err := readline.NewEx(&readline.Config{
		Prompt:          "\n👤 You > ",
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
				fmt.Println("👋 Agent terminated. Goodbye!")
				break
			}
			log.Printf("Input read error: %v", err)
			continue
		}

		input = strings.TrimSpace(input)

		if input == "exit" || input == "quit" {
			fmt.Println("👋 Agent terminated. Goodbye!")
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
			fmt.Printf("\n❌ Execution error: %v\n", err)
		} else {
			fmt.Printf("\n✨ Completed in %.2f seconds.\n", duration.Seconds())
		}
	}
}

func printWelcomeMessage() {
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("   🤖  GOLANG MCP AGENT - READY")
	fmt.Println("   Type your request. The agent will use tools as needed.")
	fmt.Println("   Type 'exit' to quit.")
	fmt.Println(strings.Repeat("=", 60))
}
