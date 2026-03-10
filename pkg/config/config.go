package config

import (
	"os"

	"gopkg.in/yaml.v3"
)

type Config struct {
	AI         AIConfig                   `yaml:"ai"`
	MCPServers map[string]MCPServerConfig `yaml:"mcp_servers"`
}

type AIConfig struct {
	Provider           string `yaml:"provider"` // e.g. "openai"
	Model              string `yaml:"model"`    // e.g. "gpt-4o"
	APIKey             string `yaml:"api_key"`
	ContextSize        int    `yaml:"context_size"`
	MaxHistoryLines    int    `yaml:"max_history_lines"`
	MaxHistoryMessages int    `yaml:"max_history_messages"`
}

type MCPServerConfig struct {
	Command string   `yaml:"command"`
	Args    []string `yaml:"args"`
}

func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}

	return &cfg, nil
}
