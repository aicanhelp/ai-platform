package shell

import (
	"errors"
	"fmt"
	"strings"

	"gopkg.openfuyao.cn/bkeagent/pkg/executor/exec"
)

type Shell interface {
	Execute(execCommands []string) ([]string, error)
}

type Task struct {
	Exec exec.Executor
}

// Execute Run the command
func (t *Task) Execute(execCommands []string) ([]string, error) {
	result := []string{}
	if len(execCommands) < 1 {
		return result, errors.New("The execution instruction is null ")
	}
	s, err := t.Exec.ExecuteCommandWithOutput("/bin/sh", "-c", strings.Join(execCommands, " "))
	if err != nil {
		return result, errors.New(fmt.Sprintf("{code:%s, reason: %s}", err.Error(), s))
	}
	result = append(result, s)
	return result, nil
}
