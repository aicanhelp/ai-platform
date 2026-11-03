/******************************************************************
 * Copyright (c) 2026 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package preflightbridge

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"golang.org/x/crypto/ssh"
	"golang.org/x/crypto/ssh/agent"
	configv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"

	"gopkg.openfuyao.cn/preflight/api"
	sshcollector "gopkg.openfuyao.cn/preflight/collectors/ssh"
)

const (
	sshConnectionTimeout = 10 * time.Second
	sshAgentTimeout      = 2 * time.Second
)

const maxSSHOutput = 1 << 20

type nodeCredential struct{ user, port, password string }

// nodeSSHExecutor owns credential material only in the bkeadm host layer. It
// never places passwords in AdapterInput, Facts, Evidence, logs or errors.
type nodeSSHExecutor struct {
	credentials map[string]nodeCredential
	// hostKeyCallback is injectable so a deployment that provisions
	// known_hosts material can enforce host-key trust without changing the
	// executor. It defaults to InsecureIgnoreHostKey to keep parity with the
	// existing OpenFuyao SSH installation path; preflight only proves that
	// some endpoint accepts the credentials, not that the host identity is
	// trusted (see the design doc's read-only boundary section).
	hostKeyCallback ssh.HostKeyCallback
}

// newNodeSSHExecutor creates the default executor. Host-key trust is not a
// preflight gate in this release; use WithHostKeyCallback to inject a
// verifier when a deployment provides known_hosts anchors.
func newNodeSSHExecutor(nodes []configv1beta1.BKENode) *nodeSSHExecutor {
	return newNodeSSHExecutorWithHostKey(nodes, ssh.InsecureIgnoreHostKey())
}

// newNodeSSHExecutorWithHostKey creates an executor with an explicit
// host-key callback. A nil callback falls back to the insecure default so a
// misconfiguration can never silently disable the SSH path.
func newNodeSSHExecutorWithHostKey(
	nodes []configv1beta1.BKENode,
	hostKeyCallback ssh.HostKeyCallback,
) *nodeSSHExecutor {
	if hostKeyCallback == nil {
		hostKeyCallback = ssh.InsecureIgnoreHostKey()
	}
	credentials := make(map[string]nodeCredential, len(nodes))
	for _, node := range nodes {
		user, port := node.Spec.Username, node.Spec.Port
		if user == "" {
			user = "root"
		}
		if port == "" {
			port = "22"
		}
		credentials[node.Name] = nodeCredential{user: user, port: port, password: node.Spec.Password}
	}
	return &nodeSSHExecutor{credentials: credentials, hostKeyCallback: hostKeyCallback}
}

func (e *nodeSSHExecutor) RunRead(
	ctx context.Context,
	subject api.Subject,
	command api.SSHReadCommand,
) (api.SSHReadResult, error) {
	if !sshcollector.Allowed(command.Name) {
		return api.SSHReadResult{}, fmt.Errorf("command is not allowlisted")
	}
	return e.runCommand(ctx, subject, command)
}

// RunCustom executes one validated absolute script without a shell. It is
// intentionally separate from the built-in SSH command allowlist.
func (e *nodeSSHExecutor) RunCustom(
	ctx context.Context,
	subject api.Subject,
	path string,
	args []string,
) (api.SSHReadResult, error) {
	if !validCustomPath(path) {
		return api.SSHReadResult{}, fmt.Errorf("custom script path must be a clean absolute path")
	}
	for _, argument := range args {
		if strings.ContainsRune(argument, 0) {
			return api.SSHReadResult{}, fmt.Errorf("custom script argument contains NUL")
		}
	}
	return e.runCommand(ctx, subject, api.SSHReadCommand{Name: path, Args: args})
}

func (e *nodeSSHExecutor) runCommand(
	ctx context.Context,
	subject api.Subject,
	command api.SSHReadCommand,
) (api.SSHReadResult, error) {
	client, closeClient, err := e.openClient(ctx, subject)
	if err != nil {
		return api.SSHReadResult{}, err
	}
	defer closeClient()
	session, err := client.NewSession()
	if err != nil {
		return api.SSHReadResult{}, fmt.Errorf("open SSH read session failed")
	}
	defer closeSSHResource("session", session)
	var stdout, stderr bytes.Buffer
	session.Stdout = &limitWriter{w: &stdout, remaining: maxSSHOutput}
	session.Stderr = &limitWriter{w: &stderr, remaining: maxSSHOutput}
	runErr := make(chan error, 1)
	go func() { runErr <- session.Run(shellCommand(command)) }()
	select {
	case err = <-runErr:
	case <-ctx.Done():
		closeSSHResource("timed-out session", session)
		return api.SSHReadResult{}, fmt.Errorf("SSH read probe timed out")
	}
	if err == nil {
		return api.SSHReadResult{Stdout: stdout.String(), Stderr: stderr.String()}, nil
	}
	if exitErr, ok := err.(*ssh.ExitError); ok {
		return api.SSHReadResult{
			Stdout: stdout.String(), Stderr: stderr.String(), ExitCode: exitErr.ExitStatus(),
		}, nil
	}
	return api.SSHReadResult{}, fmt.Errorf("execute SSH read probe failed")
}

func (e *nodeSSHExecutor) openClient(ctx context.Context, subject api.Subject) (*ssh.Client, func(), error) {
	credential, ok := e.credentials[subject.ID]
	if !ok {
		return nil, noSSHCleanup, fmt.Errorf("SSH credential reference not found for target")
	}
	// Host-key trust is injectable (see nodeSSHExecutor). The default keeps
	// parity with the existing OpenFuyao SSH path: authentication and
	// privilege remain explicit checks, while OpenSSH-compatible
	// password/agent authentication can proceed even when the operator has
	// no known_hosts entry or has rotated a node key.
	config := &ssh.ClientConfig{
		User:            credential.user,
		HostKeyCallback: e.hostKeyCallback,
		Timeout:         sshConnectionTimeout,
	}
	closeAuth, err := configureSSHAuth(config, credential.password)
	if err != nil {
		return nil, noSSHCleanup, err
	}
	address := net.JoinHostPort(subject.Address, credential.port)
	dialer := net.Dialer{Timeout: sshConnectionTimeout}
	conn, err := dialer.DialContext(ctx, "tcp", address)
	if err != nil {
		closeAuth()
		return nil, noSSHCleanup, fmt.Errorf("SSH transport unavailable: %v", err)
	}
	// ssh.NewClientConn has no context parameter. Run the handshake separately
	// and close the underlying connection when the fact deadline is canceled;
	// otherwise a half-open SSH handshake can hold the whole preflight until
	// the overall timeout, regardless of --fact-timeout.
	type handshakeResult struct {
		clientConn ssh.Conn
		channels   <-chan ssh.NewChannel
		requests   <-chan *ssh.Request
		err        error
	}
	handshake := make(chan handshakeResult, 1)
	go func() {
		clientConn, channels, requests, err := ssh.NewClientConn(conn, address, config)
		handshake <- handshakeResult{clientConn: clientConn, channels: channels, requests: requests, err: err}
	}()
	var result handshakeResult
	select {
	case result = <-handshake:
	case <-ctx.Done():
		closeSSHResource("timed-out connection", conn)
		closeAuth()
		return nil, noSSHCleanup, fmt.Errorf("SSH handshake timed out")
	}
	if result.err != nil {
		closeSSHResource("failed connection", conn)
		closeAuth()
		return nil, noSSHCleanup, fmt.Errorf("SSH authentication failed: %v", result.err)
	}
	client := ssh.NewClient(result.clientConn, result.channels, result.requests)
	return client, func() {
		closeSSHResource("client", client)
		closeSSHResource("connection", conn)
		closeAuth()
	}, nil
}

func (e *nodeSSHExecutor) DialTransport(ctx context.Context, subject api.Subject) error {
	credential, ok := e.credentials[subject.ID]
	if !ok {
		return fmt.Errorf("SSH credential reference not found for target")
	}
	address := net.JoinHostPort(subject.Address, credential.port)
	dialer := net.Dialer{Timeout: sshConnectionTimeout}
	conn, err := dialer.DialContext(ctx, "tcp", address)
	if err != nil {
		return fmt.Errorf("SSH transport unavailable: %v", err)
	}
	return conn.Close()
}

// configureSSHAuth preserves the existing password path and adds the same
// passwordless sources that an operator's local ssh command normally uses:
// SSH_AUTH_SOCK first, then unencrypted default identity files. Encrypted
// identity files are intentionally skipped; they must be unlocked in an
// ssh-agent so preflight never prompts or reads a passphrase.
func configureSSHAuth(config *ssh.ClientConfig, password string) (func(), error) {
	if password != "" {
		config.Auth = []ssh.AuthMethod{ssh.Password(password)}
		return noSSHCleanup, nil
	}

	var closers []io.Closer
	var signers []ssh.Signer
	if socket := strings.TrimSpace(os.Getenv("SSH_AUTH_SOCK")); socket != "" {
		agentConn, err := net.DialTimeout("unix", socket, sshAgentTimeout)
		if err == nil {
			agentClient := agent.NewClient(agentConn)
			agentSigners, signerErr := agentClient.Signers()
			if signerErr == nil {
				signers = append(signers, agentSigners...)
				closers = append(closers, agentConn)
			} else {
				closeSSHResource("SSH agent", agentConn)
			}
		}
	}

	for _, path := range defaultSSHIdentityFiles() {
		key, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		signer, err := ssh.ParsePrivateKey(key)
		if err != nil {
			// Encrypted keys and unsupported formats are handled by the agent
			// path above; do not prompt for or expose a passphrase here.
			continue
		}
		signers = append(signers, signer)
	}
	// Combine agent and default identities into one public-key method. The SSH
	// protocol identifies both as "publickey"; registering separate auth
	// methods can cause a server rejection of an unrelated forwarded-agent key
	// to prevent the local default identity from being attempted.
	if len(signers) > 0 {
		config.Auth = append(config.Auth, ssh.PublicKeys(signers...))
	}
	if len(config.Auth) == 0 {
		return noSSHCleanup, fmt.Errorf(
			"SSH credentials are not configured: set nodes.yaml spec.password or use SSH_AUTH_SOCK/default SSH identity",
		)
	}
	return func() {
		for _, closer := range closers {
			closeSSHResource("authentication resource", closer)
		}
	}, nil
}

func noSSHCleanup() {
	// No SSH resources were acquired, so cleanup intentionally does nothing.
}

// closeSSHResource closes an SSH resource and logs any unexpected error to stderr. Benign errors are silently ignored.
func closeSSHResource(name string, closer io.Closer) {
	if err := closer.Close(); err != nil && !benignSSHCloseError(err) {
		fmt.Fprintf(os.Stderr, "preflight cleanup: close %s: %v\n", name, err)
	}
}

func benignSSHCloseError(err error) bool {
	if errors.Is(err, io.EOF) || errors.Is(err, net.ErrClosed) ||
		errors.Is(err, syscall.ECONNRESET) {
		return true
	}
	message := err.Error()
	return strings.Contains(message, "use of closed network connection") ||
		strings.Contains(message, "connection reset by peer")
}

func defaultSSHIdentityFiles() []string {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil
	}
	return []string{
		filepath.Join(home, ".ssh", "id_ed25519"),
		filepath.Join(home, ".ssh", "id_ecdsa"),
		filepath.Join(home, ".ssh", "id_rsa"),
	}
}

func shellCommand(command api.SSHReadCommand) string {
	parts := []string{quoteShellArg(command.Name)}
	for _, arg := range command.Args {
		parts = append(parts, quoteShellArg(arg))
	}
	return strings.Join(parts, " ")
}

func quoteShellArg(value string) string {
	return "'" + strings.ReplaceAll(value, "'", `'"'"'`) + "'"
}

type limitWriter struct {
	w         io.Writer
	remaining int64
}

func (w *limitWriter) Write(p []byte) (int, error) {
	if int64(len(p)) > w.remaining {
		p = p[:w.remaining]
	}
	n, err := w.w.Write(p)
	w.remaining -= int64(n)
	if w.remaining == 0 && err == nil {
		return n, io.ErrShortWrite
	}
	return n, err
}

var _ sshcollector.ReadExecutor = (*nodeSSHExecutor)(nil)
var _ sshcollector.TransportDialer = (*nodeSSHExecutor)(nil)
