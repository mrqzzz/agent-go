# agent-go

Simple Agent with prompt and tooling capabilities.


- Connects to a configured LLM Model (far example a qwen3.5 running in Ollama locally)
- Configure MCP servers
- Configure Context size
- Chat via the build-in prompt
- Request to do several tasks with one prompt (it works if the LLM Model is smart enough to handle it)
- The prompts history is saved to `/tmp/agent-go-history`, run with `--nohistory=true` to prevent saving the history to file

This agent works well with **shell-mcp** https://github.com/mrqzzz/shell-mcp to execute shell commands in a stateful way.


## Flow

```mermaid
flowchart TD
    %% =========================
    %% Styles
    %% =========================
    classDef laneUser fill:#E8F0FE,stroke:#4C8BF5,stroke-width:1.5px,color:#0B3A8F
    classDef laneAgent fill:#EAF7EE,stroke:#2E8B57,stroke-width:1.5px,color:#145A32
    classDef laneModel fill:#FFF4E5,stroke:#F39C12,stroke-width:1.5px,color:#7D4E00
    classDef laneProtocolClient fill:#F3E8FF,stroke:#8E44AD,stroke-width:1.5px,color:#4A235A
    classDef laneProtocolServer fill:#FFEFF3,stroke:#E74C3C,stroke-width:1.5px,color:#7B241C
    classDef step fill:#FFFFFF,stroke:#6C757D,stroke-width:1.2px,color:#212529

    %% =========================
    %% Horizontal Alignment (Invisible)
    %% =========================
    User ~~~ Agent ~~~ LLM ~~~ Client ~~~ Server

    %% =========================
    %% Nodes
    %% =========================
    subgraph User["User"]
        U1["Submit prompt"]
        U2["Receive final<br/>response"]
    end

    subgraph Agent["Agent"]
        A1["Agent.Run"]
        A2["aggregateTools"]
        A3["executeTool"]
        A4["fixToolArgs"]
        A5["Append tool result<br/>to history"]
    end

    subgraph LLM["Large Language Model"]
        L1["ChatCompletion"]
        L2["Return tool<br/>call(s)"]
        L3["Return final<br/>answer"]
    end

    subgraph Client["MCP Client"]
        C1["Client.CallTool"]
        C2["call(tools/call)"]
        C3["Return tool<br/>output"]
    end

    subgraph Server["MCP Server"]
        S1["terminal_write<br/>handler"]
        S2["terminal_read<br/>handler"]
    end

    %% =========================
    %% Logic Flow
    %% =========================
    U1 --> A1
    A1 --> A2
    A2 --> L1
    L1 --> L2
    L2 --> A3
    A3 --> A4
    A4 --> C1
    C1 --> C2
    C2 --> S1
    C2 --> S2
    S1 --> C3
    S2 --> C3
    C3 --> A5
    A5 --> L1
    L1 --> L3
    L3 --> U2

    %% =========================
    %% Class Assignments
    %% =========================
    class User laneUser
    class Agent laneAgent
    class LLM laneModel
    class Client laneProtocolClient
    class Server laneProtocolServer
    class U1,U2,A1,A2,A3,A4,A5,L1,L2,L3,C1,C2,C3,S1,S2 step
```

[![Build Status](https://github.com/mrqzzz/agent-go/actions/workflows/build.yml/badge.svg)](https://github.com/mrqzzz/agent-go/actions)
[![Latest Release](https://img.shields.io/github/v/release/mrqzzz/agent-go?display_name=tag&color=blue)](https://github.com/mrqzzz/agent-go/releases/latest)

---

## Download Binaries

Download the latest compiled version for your platform. These links point to the most recent official release:

| Operating System | Architecture | Download |
| :--- | :--- | :--- |
| 🪟 **Windows** | x86_64 | [Download .exe](https://github.com/mrqzzz/agent-go/releases/latest/download/agent-go-windows-amd64.exe) |
| 🐧 **Linux** | x86_64 | [Download binary](https://github.com/mrqzzz/agent-go/releases/latest/download/agent-go-linux-amd64) |
| 🍎 **macOS** | Apple Silicon (M1/M2/M3) | [Download binary](https://github.com/mrqzzz/agent-go/releases/latest/download/agent-go-darwin-arm64) |
| 🍎 **macOS** | Intel | [Download binary](https://github.com/mrqzzz/agent-go/releases/latest/download/agent-go-darwin-amd64) |

## Build

Build locally:

Run build.sh from the repo root to build the binary.


## Configure

Edit the file `config.yaml`: 
change the LLM to use :
```
ai:
  provider: "ollama"
  model: "qwen3.5:latest" 
  api_key: "http://localhost:11434"
  prompt_timeout: "20m"
  context_size: 8192
  max_history_lines: 5000
  max_history_messages: 10000
```

add MCP servers (tools) :
```
mcp_servers:
  shell-mcp:
    command: "/Users/marcus.oblak/go/src/github.com/mrqzzz/shell-mcp/shell-mcp" 
    internal_timeout: "2m"
    args: [] 
```

## Distribute

You only need the binary `agent-go` and `config.yaml`

## Usage

### Interactive: 

Start the agent running `./agent-go` and interact normally.
Ask the agent to execute something that you MCP servers can handle. For example, if you have the "shell-mcp" tool in your config, your prompt can be something like this:

You >`evaluate the env.variable "SECRET_PASS" and remember the escaped value as the password to use later, then execute "ssh user@domain.com" and use that password when prompted and when a passphrase is asked, then call "ls -la"`


### Batch:
You can also pass agent-go a file with a sequence of prompts, one per line.

For example, run  `./agent-go prompts.txt >log.txt` — each line of `prompts.txt` will be executed individually in order, while the logs will be dumped to the file `log.txt`, and then the program will exit.

### Flags: 
`--debug`: enable verbose internal logs; 
`--nohistory`: disable saving/restoring persisted history.

### LLM temperature on failure

Behavior: If a command fails, the agent slightly increases the LLM's temperature parameter so the model can try alternative solutions. The change is small and temporary to encourage different approaches without making outputs wildly random.

### Context rot

Yes it rots, so work with few well designed prompts.

### MCP servers & prompt guidance

Advice: When using MCP servers (shell tools), make prompts precise and restrictive to avoid unwanted actions. Include safety constraints such as: "don't delete or modify any file, unless told to" to prevent accidental destructive operations.




### Disclaimer

Remember that AI Agents may produce unexpected results based on the AI model decision. I do not accept responsibility for the outputs or actions of these models.
