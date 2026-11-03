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

// Package ssh collects read-only facts from remote nodes over SSH.
//
// It executes ONLY allowlisted commands passed as structured argv arrays —
// never arbitrary shell strings. Command names and path arguments are
// desensitized when recorded as evidence. Host-key policy is owned by the
// injected host executor; this package only maps facts to read-only probes.
package ssh

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	neturl "net/url"
	"strconv"
	"strings"
	"time"

	"gopkg.openfuyao.cn/preflight/api"
	"gopkg.openfuyao.cn/preflight/normalize"
)

const (
	minimumDiskFields   = 2
	minimumPortKeyParts = 3
	osReleaseParts      = 2
)

// ReadExecutor runs an allowlisted read-only command on a target. The host
// injects a real SSH executor; tests inject a fake. It has NO write methods.
type ReadExecutor interface {
	// RunRead executes a whitelisted read-only command and returns stdout
	// (size-bounded), stderr and exit status. It MUST NOT accept a raw shell
	// string.
	RunRead(ctx context.Context, subject api.Subject, cmd api.SSHReadCommand) (api.SSHReadResult, error)
}

// TransportDialer probes only the node's TCP SSH endpoint. It deliberately
// does not perform an SSH handshake, so transport and credential failures are
// reported at the correct dependency level.
type TransportDialer interface {
	DialTransport(ctx context.Context, subject api.Subject) error
}

// HostKeyVerifier checks an observed host key against the host's trust anchors.
// When no anchor exists for a target, the host decides (WARN, not silent accept).
type HostKeyVerifier interface {
	Verify(ctx context.Context, subject api.Subject, observedFingerprint string) (matches bool, hadAnchor bool)
}

// Allowed reports whether a binary is whitelisted.
func Allowed(name string) bool {
	switch name {
	case "uname", "cat", "systemctl", "getent", "ip", "ss", "stat", "test",
		"hostname", "timedatectl", "df", "nproc", "free", "awk", "grep", "tr",
		"date", "sudo", "true", "findmnt", "getenforce", "curl", "nc", "find":
		return true
	default:
		return false
	}
}

// Collector is the SSH fact collector.
type Collector struct {
	executor          ReadExecutor
	verifier          HostKeyVerifier
	timeout           time.Duration
	verifiedPlatforms []api.VerifiedPlatform
}

// New returns an SSH Collector. executor and verifier are host-injected.
func New(executor ReadExecutor, verifier HostKeyVerifier, timeout time.Duration) *Collector {
	return &Collector{executor: executor, verifier: verifier, timeout: timeout}
}

// NewWithVerifiedPlatforms returns an SSH collector with release-verified platform metadata.
func NewWithVerifiedPlatforms(
	executor ReadExecutor,
	verifier HostKeyVerifier,
	timeout time.Duration,
	platforms []api.VerifiedPlatform,
) *Collector {
	return &Collector{executor: executor, verifier: verifier, timeout: timeout,
		verifiedPlatforms: append([]api.VerifiedPlatform(nil), platforms...)}
}

// ID returns the collector identifier.
func (c *Collector) ID() string { return "ssh-node" }

// Collect returns facts for the requested keys, scoped to subjects. Each
// subject is a remote node. A single fact failure is returned as Available=false,
// not a Go error, so independent rules are not canceled.
func (c *Collector) Collect(ctx context.Context, req api.FactRequest) ([]api.Fact, error) {
	if c.executor == nil {
		return errorFacts(req, "UNSUPPORTED", "no ssh executor configured"), nil
	}
	// Local subject facts are collected per-subject; each subject gets its own
	// fact set keyed by (subject, key).
	out := make([]api.Fact, 0, len(req.Keys)*len(req.Subjects))
	for _, sub := range req.Subjects {
		out = append(out, c.collectSubject(ctx, sub, req.Keys)...)
	}
	return out, nil
}

func (c *Collector) collectSubject(ctx context.Context, sub api.Subject, keys []string) []api.Fact {
	facts := make([]api.Fact, 0, len(keys))
	for _, key := range keys {
		select {
		case <-ctx.Done():
			ctxErr := ctx.Err()
			if ctxErr == nil {
				ctxErr = context.Canceled
			}
			facts = append(facts, api.Fact{Key: key, Subject: sub.ID, Available: false,
				Error: &api.FactError{Code: api.FactErrTimeout, Message: ctxErr.Error()}})
		default:
			facts = append(facts, c.collectKey(ctx, sub, key))
		}
	}
	return facts
}

// collectKey maps a fact key to an allowlisted command on the node.
func (c *Collector) collectKey(ctx context.Context, sub api.Subject, key string) api.Fact {
	if strings.HasPrefix(key, "platform.kernel.") && strings.HasSuffix(key, ".matched") {
		return c.verifiedKernelFact(ctx, sub, key)
	}
	if strings.HasPrefix(key, "endpoint.tcp.") {
		return c.tcpFact(ctx, sub, key)
	}
	if strings.HasPrefix(key, "ssh.transport.") {
		return c.transportFact(ctx, sub, key)
	}
	cmd, ok := commandFor(sub, key)
	if !ok {
		return api.Fact{Key: key, Subject: sub.ID, Available: false,
			Error: &api.FactError{Code: api.FactErrUnsupported, Message: "no command for key: " + key}}
	}
	if !Allowed(cmd.Name) {
		return api.Fact{Key: key, Subject: sub.ID, Available: false,
			Error: &api.FactError{Code: api.FactErrUnsupported, Message: "command not whitelisted: " + cmd.Name}}
	}
	readResult, err := c.executor.RunRead(ctx, sub, cmd)
	if err != nil {
		code := api.FactErrUnreachable
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			code = api.FactErrTimeout
		}
		return api.Fact{Key: key, Subject: sub.ID, Available: false,
			Error: &api.FactError{Code: code, Message: err.Error()}}
	}
	if readResult.ExitCode != 0 {
		return api.Fact{Key: key, Subject: sub.ID, Available: false,
			Error: &api.FactError{Code: api.FactErrUnreachable, Message: readResult.Stderr}}
	}
	value, parseErr := parseValue(key, strings.TrimSpace(readResult.Stdout), readResult.ExitCode)
	if parseErr != nil {
		return api.Fact{Key: key, Subject: sub.ID, Available: false,
			Error: &api.FactError{Code: api.FactErrParseError, Message: parseErr.Error()}}
	}
	return api.Fact{
		Key: key, Subject: sub.ID, Available: true, Value: value,
		Evidence: []api.Evidence{{Kind: "command", Source: "ssh-node",
			Locator: sub.ID, Observed: boundedEvidence(normalize.RedactEndpoint(fmt.Sprint(value)))}},
	}
}

func (c *Collector) transportFact(ctx context.Context, sub api.Subject, key string) api.Fact {
	dialer, ok := c.executor.(TransportDialer)
	if !ok {
		return api.Fact{Key: key, Subject: sub.ID, Available: false,
			Error: &api.FactError{
				Code:    api.FactErrUnsupported,
				Message: "SSH executor does not support transport probing",
			}}
	}
	if err := dialer.DialTransport(ctx, sub); err != nil {
		return unavailable(ctx, key, sub.ID, err)
	}
	return api.Fact{Key: key, Subject: sub.ID, Available: true, Value: true,
		Evidence: []api.Evidence{
			{
				Kind:    "tcp",
				Source:  "ssh-node",
				Locator: normalize.RedactEndpoint(net.JoinHostPort(sub.Address, sub.Metadata["sshPort"])),
			},
		}}
}

func (c *Collector) tcpFact(ctx context.Context, sub api.Subject, key string) api.Fact {
	target, resolve := endpointTargetWithResolve(key)
	host, port, ok := endpointHostPort(target)
	if !ok {
		return api.Fact{Key: key, Subject: sub.ID, Available: false,
			Error: &api.FactError{Code: api.FactErrParseError, Message: "invalid TCP endpoint"}}
	}
	// Probe through the node's own curl, not an SSH-forwarded TCP channel:
	// the node must reach the repository itself. A configured IP rides along
	// as --resolve so the domain stays the HTTP Host and TLS SNI.
	command := tcpProbeCommand(target, host, port, resolve)
	if !Allowed(command.Name) {
		return api.Fact{Key: key, Subject: sub.ID, Available: false,
			Error: &api.FactError{Code: api.FactErrUnsupported,
				Message: "command not whitelisted: " + command.Name}}
	}
	readResult, err := c.executor.RunRead(ctx, sub, command)
	if err != nil {
		return nodeTransportFact(ctx, key, sub.ID, err)
	}
	// curl exits non-zero for real transport failures (DNS, connect refused,
	// timeout). HTTP error statuses are NOT failures here: --fail is omitted
	// so 401/403/404 still prove the endpoint is reachable.
	if readResult.ExitCode != 0 {
		return api.Fact{Key: key, Subject: sub.ID, Available: false,
			Error: &api.FactError{Code: api.FactErrUnreachable,
				Message: strings.TrimSpace(readResult.Stderr)}}
	}
	locator := net.JoinHostPort(host, port)
	if resolve != "" {
		locator += " via " + normalize.RedactEndpoint(resolve)
	}
	return api.Fact{Key: key, Subject: sub.ID, Available: true, Value: true,
		Evidence: []api.Evidence{{Kind: "command", Source: "ssh-node",
			Locator: normalize.RedactEndpoint(locator)}}}
}

// sshFailureMarkers are error-message fragments from the bridge's SSH
// executor that identify node access failures. Keep in sync with the error
// strings produced by pkg/preflightbridge/ssh_executor.go.
var sshFailureMarkers = []string{
	"SSH transport unavailable",
	"SSH authentication failed",
	"SSH credential reference not found",
	"open SSH read session failed",
}

// nodeTransportFact reports an SSH-layer failure as a node access failure,
// not as an unreachable endpoint. Disguising transport or credential
// failures as domain/endpoint failures misleads remediation.
func nodeTransportFact(ctx context.Context, key, subject string, err error) api.Fact {
	code := api.FactErrUnreachable
	message := err.Error()
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		code = api.FactErrTimeout
	}
	for _, marker := range sshFailureMarkers {
		if strings.Contains(message, marker) {
			return api.Fact{Key: key, Subject: subject, Available: false,
				Error: &api.FactError{Code: code,
					Message: "node SSH access failed; " + message}}
		}
	}
	return api.Fact{Key: key, Subject: subject, Available: false,
		Error: &api.FactError{Code: code, Message: message}}
}

// tcpProbeCommand builds the node-side reachability probe. curl is used
// because it is already allowlisted and reports connect failures precisely.
// The probe only decides reachability: --insecure keeps self-signed or
// untrusted certificates from failing the TLS handshake, and --fail is
// deliberately omitted so HTTP error statuses (401/403/404) still count as
// "endpoint reachable".
func tcpProbeCommand(target, host, port, resolve string) api.SSHReadCommand {
	scheme := "https"
	if parsed, err := neturl.Parse(target); err == nil && parsed.Scheme != "" {
		scheme = parsed.Scheme
	}
	args := []string{
		"--head", "--silent", "--show-error", "--insecure",
		"--connect-timeout", "5", "--max-time", "10",
		"--output", "/dev/null", "--write-out", "%{http_code}",
	}
	if resolve != "" {
		resolveHost, resolvePort, ok := splitHostPort(resolve)
		if ok {
			args = append(args, "--resolve", host+":"+resolvePort+":"+resolveHost)
		}
	}
	args = append(args, scheme+"://"+net.JoinHostPort(host, port))
	return cmd("curl", args...)
}

func splitHostPort(address string) (string, string, bool) {
	host, port, err := net.SplitHostPort(address)
	if err != nil || host == "" || port == "" {
		return "", "", false
	}
	return host, port, true
}

// endpointTargetWithResolve splits an "@ip:port" resolve override off the
// endpoint target embedded in an endpoint.* fact key.
func endpointTargetWithResolve(key string) (target string, resolve string) {
	target = endpointTarget(key)
	if at := strings.LastIndex(target, "@"); at >= 0 {
		resolve = target[at+1:]
		target = target[:at]
		if _, _, ok := splitHostPort(resolve); !ok {
			return target, ""
		}
	}
	return target, resolve
}

func boundedEvidence(value string) string {
	value = strings.TrimSpace(value)
	const limit = 512
	if len(value) > limit {
		return value[:limit] + "..."
	}
	return value
}

func (c *Collector) verifiedKernelFact(ctx context.Context, sub api.Subject, key string) api.Fact {
	read := func(command api.SSHReadCommand) (string, error) {
		readResult, err := c.executor.RunRead(ctx, sub, command)
		if err != nil {
			return "", err
		}
		if readResult.ExitCode != 0 {
			return "", fmt.Errorf("read-only platform probe failed: %s", readResult.Stderr)
		}
		return strings.TrimSpace(readResult.Stdout), nil
	}
	kernel, err := read(cmd("uname", "-r"))
	if err != nil {
		return unavailable(ctx, key, sub.ID, err)
	}
	osRelease, err := read(cmd("cat", "/etc/os-release"))
	if err != nil {
		return unavailable(ctx, key, sub.ID, err)
	}
	architecture, err := read(cmd("uname", "-m"))
	if err != nil {
		return unavailable(ctx, key, sub.ID, err)
	}
	id, version := parseOSRelease(osRelease)
	architecture = normalize.Architecture(architecture)
	matched := false
	for _, platform := range c.verifiedPlatforms {
		if platform.Distribution != strings.ToLower(id) || platform.Version != version ||
			!stringIn(platform.Architectures, architecture) {
			continue
		}
		for _, prefix := range platform.KernelPrefixes {
			if strings.HasPrefix(kernel, prefix) {
				matched = true
			}
		}
	}
	return api.Fact{Key: key, Subject: sub.ID, Available: true, Value: matched,
		Evidence: []api.Evidence{{Kind: "command", Source: "ssh-node", Locator: sub.ID,
			Observed: fmt.Sprintf("distribution=%s version=%s arch=%s kernel=%s", id, version, architecture, kernel)}}}
}

func unavailable(ctx context.Context, key, subject string, err error) api.Fact {
	code := api.FactErrUnreachable
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		code = api.FactErrTimeout
	}
	return api.Fact{Key: key, Subject: subject, Available: false,
		Error: &api.FactError{Code: code, Message: err.Error()}}
}

func stringIn(values []string, expected string) bool {
	for _, value := range values {
		if value == expected {
			return true
		}
	}
	return false
}

// commandFor maps a fact key to an allowlisted read-only command. Only the
// keys the SDK rules actually consume are mapped; unknown keys are unsupported.
func commandFor(sub api.Subject, key string) (api.SSHReadCommand, bool) {
	if strings.HasPrefix(key, "endpoint.") {
		return endpointCommandFor(key)
	}
	return nodeCommandFor(sub, key)
}

func endpointCommandFor(key string) (api.SSHReadCommand, bool) {
	switch {
	case strings.HasPrefix(key, "endpoint.dns."):
		host := strings.TrimPrefix(key, "endpoint.dns.")
		if parsedHost, _, err := net.SplitHostPort(host); err == nil {
			host = parsedHost
		}
		return cmd("getent", "hosts", host), true
	case strings.HasPrefix(key, "endpoint.tls."), strings.HasPrefix(key, "endpoint.auth."):
		target := endpointTarget(key)
		if target == "" {
			return api.SSHReadCommand{}, false
		}
		return curlProbeCommand(target, false), true
	case strings.HasPrefix(key, "endpoint.object."):
		target := endpointTarget(key)
		if target == "" {
			return api.SSHReadCommand{}, false
		}
		// Some OCI/chart servers implement GET reliably but reject or
		// mishandle HEAD. A one-byte ranged GET keeps the probe read-only
		// while validating the actual object endpoint.
		return curlProbeCommand(target, true), true
	default:
		return api.SSHReadCommand{}, false
	}
}

func curlProbeCommand(target string, ranged bool) api.SSHReadCommand {
	args := []string{"--head"}
	if ranged {
		args = []string{"--get", "--range", "0-0"}
	}
	args = append(args,
		"--silent", "--show-error", "--location", "--max-redirs", "5",
		"--connect-timeout", "5", "--max-time", "10", "--output", "/dev/null",
		"--write-out", "%{http_code}", target,
	)
	return cmd("curl", args...)
}

func nodeCommandFor(sub api.Subject, key string) (api.SSHReadCommand, bool) {
	switch {
	case strings.HasPrefix(key, "ssh.auth."), strings.HasPrefix(key, "ssh.hostkey."):
		return cmd("uname", "-s"), true
	case strings.HasPrefix(key, "ssh.privilege."):
		if sub.Metadata["sshUser"] == "root" {
			return cmd("true"), true
		}
		return cmd("sudo", "-n", "true"), true
	case strings.HasPrefix(key, "systemd."):
		return cmd("systemctl", "--version"), true
	case key == "node.cpu.logical", key == "resources.cpu.logical":
		return cmd("nproc"), true
	case key == "node.memory.totalBytes", key == "resources.memory.totalBytes":
		return cmd("free", "-b"), true
	case key == "node.systemDisk.totalBytes", key == "resources.systemDisk.totalBytes":
		return cmd("df", "-B1", "/"), true
	case key == "platform.os", strings.HasPrefix(key, "platform.os."):
		return cmd("uname", "-s"), true
	case key == "platform.distribution", strings.HasPrefix(key, "platform.distribution."):
		return cmd("cat", "/etc/os-release"), true
	case key == "platform.kernel", strings.HasPrefix(key, "platform.kernel."):
		return cmd("uname", "-r"), true
	case strings.HasPrefix(key, "port."):
		return cmd("ss", "-H", "-lntup"), true
	default:
		return api.SSHReadCommand{}, false
	}
}

func parseValue(key, stdout string, exit int) (any, error) {
	if strings.HasPrefix(key, "endpoint.") {
		return parseEndpointValue(key, stdout, exit)
	}
	switch {
	case strings.HasPrefix(key, "ssh."), strings.HasPrefix(key, "systemd."):
		return exit == 0, nil
	case key == "node.cpu.logical", key == "resources.cpu.logical":
		return strconv.Atoi(strings.TrimSpace(stdout))
	case key == "node.memory.totalBytes", key == "resources.memory.totalBytes":
		return parseMemoryBytes(stdout)
	case key == "node.systemDisk.totalBytes", key == "resources.systemDisk.totalBytes":
		return parseDiskBytes(stdout)
	case key == "platform.os", strings.HasPrefix(key, "platform.os."):
		return strings.ToLower(stdout), nil
	case key == "platform.distribution", strings.HasPrefix(key, "platform.distribution."):
		id, version := parseOSRelease(stdout)
		return []string{id, version}, nil
	case key == "platform.kernel", strings.HasPrefix(key, "platform.kernel."):
		return stdout != "", nil
	case strings.HasPrefix(key, "port."):
		return parsePortAvailable(key, stdout)
	default:
		return strings.TrimSpace(stdout), nil
	}
}

func parsePortAvailable(key, stdout string) (any, error) {
	parts := strings.Split(key, ".")
	if len(parts) < minimumPortKeyParts {
		return nil, fmt.Errorf("port key malformed")
	}
	needle := ":" + parts[1]
	for _, line := range strings.Split(stdout, "\n") {
		if lineContainsSuffix(line, needle) {
			return false, nil
		}
	}
	return true, nil
}

func lineContainsSuffix(line, suffix string) bool {
	for _, field := range strings.Fields(line) {
		if strings.HasSuffix(field, suffix) {
			return true
		}
	}
	return false
}

func parseEndpointValue(key, stdout string, exit int) (any, error) {
	switch {
	case strings.HasPrefix(key, "endpoint.dns."):
		return exit == 0 && strings.TrimSpace(stdout) != "", nil
	case strings.HasPrefix(key, "endpoint.tcp."), strings.HasPrefix(key, "endpoint.tls."):
		return exit == 0, nil
	case strings.HasPrefix(key, "endpoint.auth."), strings.HasPrefix(key, "endpoint.object."):
		status, err := strconv.Atoi(strings.TrimSpace(stdout))
		if err != nil {
			return nil, err
		}
		if strings.HasPrefix(key, "endpoint.auth.") {
			accepted := status >= http.StatusOK && status != http.StatusUnauthorized && status != http.StatusForbidden
			return accepted, nil
		}
		return status >= http.StatusOK && status < http.StatusBadRequest, nil
	default:
		return nil, fmt.Errorf("unsupported endpoint fact")
	}
}

func parseMemoryBytes(stdout string) (any, error) {
	for _, line := range strings.Split(stdout, "\n") {
		fields := strings.Fields(line)
		if len(fields) > 1 && strings.HasPrefix(strings.ToLower(fields[0]), "mem") {
			return strconv.ParseInt(fields[1], 10, 64)
		}
	}
	return nil, fmt.Errorf("mem row missing")
}

func parseDiskBytes(stdout string) (any, error) {
	lastLine := strings.TrimSpace(stdout)
	if lastLine == "" {
		return nil, fmt.Errorf("df output missing")
	}
	if newline := strings.LastIndex(lastLine, "\n"); newline >= 0 {
		lastLine = lastLine[newline+1:]
	}
	fields := strings.Fields(lastLine)
	if len(fields) < minimumDiskFields {
		return nil, fmt.Errorf("df output malformed")
	}
	return strconv.ParseInt(fields[1], 10, 64)
}

func endpointTarget(key string) string {
	for _, prefix := range []string{"endpoint.tcp.", "endpoint.tls.", "endpoint.auth.", "endpoint.object."} {
		if strings.HasPrefix(key, prefix) {
			target := strings.TrimPrefix(key, prefix)
			if !strings.Contains(target, "://") {
				target = "https://" + target
			}
			return target
		}
	}
	return ""
}

func endpointHostPort(target string) (string, string, bool) {
	u, err := neturl.Parse(target)
	if err != nil || u.Host == "" {
		return "", "", false
	}
	host := u.Hostname()
	port := u.Port()
	if port == "" {
		switch strings.ToLower(u.Scheme) {
		case "http":
			port = "80"
		case "https", "":
			port = "443"
		default:
			return "", "", false
		}
	}
	return host, port, host != ""
}

func parseOSRelease(content string) (string, string) {
	values := map[string]string{}
	for _, line := range strings.Split(content, "\n") {
		parts := strings.SplitN(line, "=", osReleaseParts)
		if len(parts) == osReleaseParts {
			values[parts[0]] = strings.Trim(parts[1], "\"")
		}
	}
	return strings.ToLower(values["ID"]), values["VERSION_ID"]
}

// cmd builds an SSHReadCommand from a name + args.
func cmd(name string, args ...string) api.SSHReadCommand {
	return api.SSHReadCommand{Name: name, Args: args}
}

func errorFacts(req api.FactRequest, code, msg string) []api.Fact {
	out := make([]api.Fact, 0, len(req.Keys)*len(req.Subjects))
	for _, sub := range req.Subjects {
		for _, k := range req.Keys {
			out = append(out, api.Fact{Key: k, Subject: sub.ID, Available: false,
				Error: &api.FactError{Code: code, Message: msg}})
		}
	}
	if len(req.Subjects) == 0 {
		for _, k := range req.Keys {
			out = append(out, api.Fact{Key: k, Available: false,
				Error: &api.FactError{Code: code, Message: msg}})
		}
	}
	return out
}
