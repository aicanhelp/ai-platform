package executor

import (
	"context"
	"io"
	"os/exec"
	"time"
)

// LocalExecutor execute commands locally
type LocalExecutor struct {
	defaultTimeout time.Duration
}

// NewLocalExecutor creates a new LocalExecutor with the given timeout
func NewLocalExecutor(timeout time.Duration) *LocalExecutor {
	return &LocalExecutor{
		defaultTimeout: timeout,
	}
}

// Exec executes a command locally
func (l *LocalExecutor) Exec(cmd string, opts ...ExecOption) (*ExecResult, error) {
	ctx, cancel := context.WithTimeout(context.Background(), l.defaultTimeout)
	defer cancel()

	c := exec.CommandContext(ctx, "bash", "-c", cmd)

	stdout, _ := c.StdoutPipe()
	stderr, _ := c.StderrPipe()

	if err := c.Start(); err != nil {
		return nil, err
	}

	outBytes, _ := io.ReadAll(stdout)
	errBytes, _ := io.ReadAll(stderr)
	err := c.Wait()

	exitCode := 0
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			exitCode = exitErr.ExitCode()
		}
	}

	return &ExecResult{
		Command:  cmd,
		Stdout:   string(outBytes),
		Stderr:   string(errBytes),
		ExitCode: exitCode,
	}, err
}
