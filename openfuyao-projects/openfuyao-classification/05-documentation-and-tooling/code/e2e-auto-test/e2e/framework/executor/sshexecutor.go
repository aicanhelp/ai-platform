package executor

import (
	"fmt"
	"io"
	"strings"
	"sync"
	"time"

	"golang.org/x/crypto/ssh"
)

type SSHExecutor struct {
	client *ssh.Client
}

func NewSSHExecutor(host string, port int, user, pass string) (*SSHExecutor, error) {
	config := &ssh.ClientConfig{
		User: user,
		Auth: []ssh.AuthMethod{
			ssh.Password(pass),
		},
		HostKeyCallback: ssh.InsecureIgnoreHostKey(),
		Timeout:         5 * time.Second,
	}

	cli, err := ssh.Dial("tcp", fmt.Sprintf("%s:%d", host, port), config)
	if err != nil {
		return nil, err
	}

	return &SSHExecutor{client: cli}, nil
}

func (s *SSHExecutor) Exec(cmd string, opts ...ExecOption) (*ExecResult, error) {
	return s.exec(cmd, "")
}

// ExecWithInput executes cmd and sends input through the SSH session's stdin.
// Keep secrets such as generated configuration credentials out of the command
// string, because ExecResult.Command may be included in failure diagnostics.
func (s *SSHExecutor) ExecWithInput(cmd, input string) (*ExecResult, error) {
	return s.exec(cmd, input)
}

func (s *SSHExecutor) exec(cmd, input string) (*ExecResult, error) {
	session, err := s.client.NewSession()
	if err != nil {
		return nil, err
	}
	defer session.Close()
	if input != "" {
		session.Stdin = strings.NewReader(input)
	}

	stdout, err := session.StdoutPipe()
	if err != nil {
		return nil, err
	}
	stderr, err := session.StderrPipe()
	if err != nil {
		return nil, err
	}

	err = session.Start(cmd)
	if err != nil {
		return nil, err
	}

	// stdout 与 stderr 共享 SSH channel 的固定缓冲，必须并发消费：
	// 若顺序读取，stderr 输出填满缓冲会使远端进程阻塞，stdout 永远等不到 EOF，导致死锁。
	var outBytes, errBytes []byte
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		outBytes, _ = io.ReadAll(stdout)
	}()
	go func() {
		defer wg.Done()
		errBytes, _ = io.ReadAll(stderr)
	}()
	wg.Wait()

	waitErr := session.Wait()

	exitCode := 0
	if waitErr != nil {
		if exit, ok := waitErr.(*ssh.ExitError); ok {
			exitCode = exit.ExitStatus()
		}
	}

	return &ExecResult{
		Command:  cmd,
		Stdout:   string(outBytes),
		Stderr:   string(errBytes),
		ExitCode: exitCode,
	}, waitErr
}

// Close closes the SSH client connection
func (s *SSHExecutor) Close() error {
	if s.client != nil {
		return s.client.Close()
	}
	return nil
}
