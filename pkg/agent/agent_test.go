package agent

import (
	"encoding/json"
	"testing"
)

func TestFixToolArgs_NormalizesEscapedTerminalCommand(t *testing.T) {
	a := &Agent{}
	input := `{"command":"for id in 1 2; do curl -s \\\"https://example.com/$id.json\\\"\\; done"}`

	fixed := a.fixToolArgs("terminal_write", input)

	var args map[string]interface{}
	if err := json.Unmarshal([]byte(fixed), &args); err != nil {
		t.Fatalf("failed to decode fixed args: %v", err)
	}

	command, ok := args["command"].(string)
	if !ok {
		t.Fatalf("command argument missing or not a string")
	}

	expected := `for id in 1 2; do curl -s "https://example.com/$id.json"; done`
	if command != expected {
		t.Fatalf("unexpected normalized command\nexpected: %q\nactual:   %q", expected, command)
	}
}

func TestFixToolArgs_DoesNotChangeNonTerminalWrite(t *testing.T) {
	a := &Agent{}
	input := `{"command":"echo \\\"hello\\\"\\;"}`

	fixed := a.fixToolArgs("other_tool", input)
	if fixed != input {
		t.Fatalf("non-terminal_write args should not be changed")
	}
}

func TestFixToolArgs_NormalizesDoubleEscapedShellSyntax(t *testing.T) {
	a := &Agent{}
	input := `{"command":"for id in $(echo 1 2); do curl -s \\\"https://example.com/${id}.json\\\"\\\\; done | python3 -m json.tool"}`

	fixed := a.fixToolArgs("terminal_write", input)

	var args map[string]interface{}
	if err := json.Unmarshal([]byte(fixed), &args); err != nil {
		t.Fatalf("failed to decode fixed args: %v", err)
	}

	command, ok := args["command"].(string)
	if !ok {
		t.Fatalf("command argument missing or not a string")
	}

	expected := `for id in $(echo 1 2); do curl -s "https://example.com/${id}.json"; done | python3 -m json.tool`
	if command != expected {
		t.Fatalf("unexpected normalized command\nexpected: %q\nactual:   %q", expected, command)
	}
}
