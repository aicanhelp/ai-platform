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

package ssh

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gopkg.openfuyao.cn/preflight/api"
)

type fakeExec struct {
	stdout       string
	stderr       string
	exitCode     int
	err          error
	last         api.SSHReadCommand
	transportErr error
}

func (f *fakeExec) DialTransport(_ context.Context, _ api.Subject) error { return f.transportErr }

func (f *fakeExec) RunRead(
	_ context.Context,
	_ api.Subject,
	command api.SSHReadCommand,
) (api.SSHReadResult, error) {
	f.last = command
	return api.SSHReadResult{Stdout: f.stdout, Stderr: f.stderr, ExitCode: f.exitCode}, f.err
}

type scriptedExec struct {
	responses map[string]api.SSHReadResult
	err       error
}

func (e *scriptedExec) RunRead(
	_ context.Context,
	_ api.Subject,
	command api.SSHReadCommand,
) (api.SSHReadResult, error) {
	return e.responses[fmt.Sprint(append([]string{command.Name}, command.Args...))], e.err
}

func TestCollectPlatformFacts(t *testing.T) {
	c := New(&fakeExec{stdout: "Linux host 5.10.0 #1 ... x86_64 GNU/Linux"}, nil, time.Second)
	facts, _ := c.Collect(context.Background(), api.FactRequest{
		Keys:     []string{"platform.os"},
		Subjects: []api.Subject{{ID: "master-01"}},
	})
	assert.Len(t, facts, 1)
	assert.True(t, facts[0].Available)
	assert.Equal(t, "linux host 5.10.0 #1 ... x86_64 gnu/linux", facts[0].Value)
	assert.Equal(t, "master-01", facts[0].Subject)
}

func TestCollectNilExecutor(t *testing.T) {
	c := New(nil, nil, time.Second)
	facts, _ := c.Collect(context.Background(), api.FactRequest{
		Keys: []string{"platform.os"}, Subjects: []api.Subject{{ID: "n"}}})
	assert.False(t, facts[0].Available)
	assert.Equal(t, "UNSUPPORTED", facts[0].Error.Code)
}

func TestCollectExecutorError(t *testing.T) {
	c := New(&fakeExec{err: assertErr("connection refused")}, nil, time.Second)
	facts, _ := c.Collect(context.Background(), api.FactRequest{
		Keys: []string{"resources.cpu.logical"}, Subjects: []api.Subject{{ID: "n"}}})
	assert.False(t, facts[0].Available)
	assert.Equal(t, api.FactErrUnreachable, facts[0].Error.Code)
}

func TestCollectNonZeroExit(t *testing.T) {
	c := New(&fakeExec{exitCode: 1, stderr: "command not found"}, nil, time.Second)
	facts, _ := c.Collect(context.Background(), api.FactRequest{
		Keys: []string{"resources.cpu.logical"}, Subjects: []api.Subject{{ID: "n"}}})
	assert.False(t, facts[0].Available)
	assert.Equal(t, api.FactErrUnreachable, facts[0].Error.Code)
}

func TestCollectUnknownKey(t *testing.T) {
	c := New(&fakeExec{}, nil, time.Second)
	facts, _ := c.Collect(context.Background(), api.FactRequest{
		Keys: []string{"bogus.key"}, Subjects: []api.Subject{{ID: "n"}}})
	assert.False(t, facts[0].Available)
	assert.Equal(t, api.FactErrUnsupported, facts[0].Error.Code)
}

func TestCollectContextCanceled(t *testing.T) {
	c := New(&fakeExec{}, nil, time.Second)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	facts, _ := c.Collect(ctx, api.FactRequest{
		Keys: []string{"platform.os"}, Subjects: []api.Subject{{ID: "n"}}})
	assert.False(t, facts[0].Available)
	assert.Equal(t, api.FactErrTimeout, facts[0].Error.Code)
}

func TestAllowedWhitelist(t *testing.T) {
	assert.True(t, Allowed("uname"))
	assert.True(t, Allowed("systemctl"))
	assert.False(t, Allowed("rm"))
	assert.False(t, Allowed("sh"))
}

func TestEndpointTCPProbesThroughNodeCurl(t *testing.T) {
	executor := &fakeExec{}
	collector := New(executor, nil, time.Second)
	facts, err := collector.Collect(context.Background(), api.FactRequest{
		Keys:     []string{"endpoint.tcp.https://registry.example.com:5443/v2/"},
		Subjects: []api.Subject{{ID: "node-1"}},
	})
	require.NoError(t, err)
	require.Len(t, facts, 1)
	assert.True(t, facts[0].Available)
	// The probe must run on the node itself (curl), not through an
	// SSH-forwarded TCP channel, so the node's own network path is verified.
	assert.Equal(t, "curl", executor.last.Name)
	assert.Contains(t, executor.last.Args, "https://registry.example.com:5443")
	assert.NotContains(t, executor.last.Args[0], "registry.example.com")
	// Reachability only: tolerate certificate failures, never use --fail so
	// HTTP error statuses (401/403/404) still count as reachable.
	assert.Contains(t, executor.last.Args, "--insecure")
	assert.NotContains(t, executor.last.Args, "--fail")
}

func TestEndpointTCPResolvesConfiguredIPKeepingDomainHostAndSNI(t *testing.T) {
	executor := &fakeExec{}
	collector := New(executor, nil, time.Second)
	facts, err := collector.Collect(context.Background(), api.FactRequest{
		Keys:     []string{"endpoint.tcp.https://registry.example.com:5443@192.0.2.10:5443"},
		Subjects: []api.Subject{{ID: "node-1"}},
	})
	require.NoError(t, err)
	require.Len(t, facts, 1)
	assert.True(t, facts[0].Available)
	assert.Equal(t, "curl", executor.last.Name)
	// --resolve connects to the configured IP while the URL keeps the domain,
	// preserving HTTP Host and TLS SNI.
	assert.Contains(t, executor.last.Args, "--resolve")
	resolveIndex := indexOf(executor.last.Args, "--resolve")
	assert.Equal(t, "registry.example.com:5443:192.0.2.10", executor.last.Args[resolveIndex+1])
	assert.Contains(t, executor.last.Args, "https://registry.example.com:5443")
}

func TestEndpointTCPReportsSSHFailureAsNodeAccessFailure(t *testing.T) {
	executor := &fakeExec{err: errors.New("SSH authentication failed: handshakex")}
	collector := New(executor, nil, time.Second)
	facts, err := collector.Collect(context.Background(), api.FactRequest{
		Keys:     []string{"endpoint.tcp.https://registry.example.com:5443"},
		Subjects: []api.Subject{{ID: "node-1"}},
	})
	require.NoError(t, err)
	require.Len(t, facts, 1)
	assert.False(t, facts[0].Available)
	require.NotNil(t, facts[0].Error)
	assert.Contains(t, facts[0].Error.Message, "node SSH access failed")
	// The message must not present the failure as a repository DNS problem.
	assert.NotContains(t, facts[0].Error.Message, "resolve")
}

func indexOf(values []string, wanted string) int {
	for i, value := range values {
		if value == wanted {
			return i
		}
	}
	return -1
}

func TestSSHTransportDoesNotPerformAuthentication(t *testing.T) {
	executor := &fakeExec{err: assertErr("authentication failed")}
	collector := New(executor, nil, time.Second)
	facts, err := collector.Collect(context.Background(), api.FactRequest{
		Keys:     []string{"ssh.transport.node-1"},
		Subjects: []api.Subject{{ID: "node-1", Address: "192.0.2.10", Metadata: map[string]string{"sshPort": "22"}}},
	})
	require.NoError(t, err)
	require.Len(t, facts, 1)
	assert.True(t, facts[0].Available)
	assert.Equal(t, true, facts[0].Value)
}

func TestSSHAuthenticationFailureIsSeparateFromTransport(t *testing.T) {
	executor := &fakeExec{err: assertErr("authentication failed")}
	collector := New(executor, nil, time.Second)
	subject := api.Subject{ID: "node-1", Address: "192.0.2.10", Metadata: map[string]string{"sshPort": "22"}}
	transport, err := collector.Collect(context.Background(), api.FactRequest{
		Keys: []string{"ssh.transport.node-1"}, Subjects: []api.Subject{subject},
	})
	require.NoError(t, err)
	auth, err := collector.Collect(context.Background(), api.FactRequest{
		Keys: []string{"ssh.auth.node-1"}, Subjects: []api.Subject{subject},
	})
	require.NoError(t, err)
	assert.True(t, transport[0].Available)
	assert.False(t, auth[0].Available)
	assert.Equal(t, api.FactErrUnreachable, auth[0].Error.Code)
}

func TestParseValueFormats(t *testing.T) {
	tests := []struct {
		name   string
		key    string
		stdout string
		want   any
	}{
		{name: "cpu", key: "resources.cpu.logical", stdout: "8", want: 8},
		{name: "memory", key: "resources.memory.totalBytes", stdout: "Mem: 17179869184 0 0", want: int64(17179869184)},
		{name: "disk", key: "resources.systemDisk.totalBytes",
			stdout: "Filesystem 1B-blocks\n/dev/vda 107374182400", want: int64(107374182400)},
		{name: "distribution", key: "platform.distribution.node-1",
			stdout: "ID=\"openEuler\"\nVERSION_ID=\"22.03\"", want: []string{"openeuler", "22.03"}},
		{name: "kernel", key: "platform.kernel.node-1", stdout: "5.10.0", want: true},
		{name: "port free", key: "port.6443.free", stdout: "LISTEN 0 128 0.0.0.0:22", want: true},
		{name: "port occupied", key: "port.6443.free", stdout: "LISTEN 0 128 [::]:6443", want: false},
		{name: "auth accepted", key: "endpoint.auth.registry.example", stdout: "302", want: true},
		{name: "auth rejected", key: "endpoint.auth.registry.example", stdout: "401", want: false},
		{name: "object exists", key: "endpoint.object.registry.example", stdout: "206", want: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseValue(tt.key, tt.stdout, 0)
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestParseValueRejectsMalformedOutput(t *testing.T) {
	tests := []struct{ key, output string }{
		{key: "resources.cpu.logical", output: "many"},
		{key: "resources.memory.totalBytes", output: "Swap: 0"},
		{key: "resources.systemDisk.totalBytes", output: "missing"},
		{key: "port.", output: ""},
		{key: "endpoint.auth.registry.example", output: "not-a-status"},
	}
	for _, tt := range tests {
		_, err := parseValue(tt.key, tt.output, 0)
		assert.Error(t, err, tt.key)
	}
	_, err := parseEndpointValue("endpoint.unsupported.value", "", 0)
	assert.ErrorContains(t, err, "unsupported endpoint")
}

func TestCommandMappingUsesReadOnlyStructuredArguments(t *testing.T) {
	subject := api.Subject{Metadata: map[string]string{"sshUser": "operator"}}
	tests := []struct {
		key      string
		name     string
		firstArg string
	}{
		{key: "endpoint.dns.registry.example.com:443", name: "getent", firstArg: "hosts"},
		{key: "endpoint.tls.registry.example.com", name: "curl", firstArg: "--head"},
		{key: "endpoint.object.registry.example.com/image", name: "curl", firstArg: "--get"},
		{key: "ssh.privilege.node-1", name: "sudo", firstArg: "-n"},
		{key: "systemd.available", name: "systemctl", firstArg: "--version"},
		{key: "resources.cpu.logical", name: "nproc"},
		{key: "resources.memory.totalBytes", name: "free", firstArg: "-b"},
		{key: "resources.systemDisk.totalBytes", name: "df", firstArg: "-B1"},
	}
	for _, tt := range tests {
		command, ok := commandFor(subject, tt.key)
		require.True(t, ok, tt.key)
		assert.Equal(t, tt.name, command.Name)
		if tt.firstArg != "" {
			require.NotEmpty(t, command.Args)
			assert.Equal(t, tt.firstArg, command.Args[0])
		}
	}
	_, ok := commandFor(subject, "unknown")
	assert.False(t, ok)
}

func TestEndpointTargetAndHostPort(t *testing.T) {
	tests := []struct {
		key, host, port string
		ok              bool
	}{
		{key: "endpoint.tcp.registry.example.com", host: "registry.example.com", port: "443", ok: true},
		{key: "endpoint.tcp.http://repo.example.com/path", host: "repo.example.com", port: "80", ok: true},
		{key: "endpoint.tcp.https://[2001:db8::1]:5443/v2", host: "2001:db8::1", port: "5443", ok: true},
		{key: "unknown", ok: false},
	}
	for _, tt := range tests {
		target := endpointTarget(tt.key)
		host, port, ok := endpointHostPort(target)
		assert.Equal(t, tt.ok, ok, tt.key)
		assert.Equal(t, tt.host, host, tt.key)
		assert.Equal(t, tt.port, port, tt.key)
	}
}

func TestVerifiedKernelFactMatchesReleaseMatrix(t *testing.T) {
	executor := &scriptedExec{responses: map[string]api.SSHReadResult{
		"[uname -r]":            {Stdout: "5.10.0-200"},
		"[cat /etc/os-release]": {Stdout: "ID=openeuler\nVERSION_ID=22.03"},
		"[uname -m]":            {Stdout: "x86_64"},
	}}
	collector := NewWithVerifiedPlatforms(executor, nil, time.Second, []api.VerifiedPlatform{{
		Distribution: "openeuler", Version: "22.03", Architectures: []string{"amd64"},
		KernelPrefixes: []string{"5.10."},
	}})
	facts, err := collector.Collect(context.Background(), api.FactRequest{
		Keys: []string{"platform.kernel.node-1.matched"}, Subjects: []api.Subject{{ID: "node-1"}},
	})
	require.NoError(t, err)
	require.Len(t, facts, 1)
	assert.True(t, facts[0].Available)
	assert.Equal(t, true, facts[0].Value)
	assert.Contains(t, facts[0].Evidence[0].Observed, "arch=amd64")
}

func TestTransportAndTCPFailuresClassifyTimeout(t *testing.T) {
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancel()
	executor := &fakeExec{transportErr: context.DeadlineExceeded, err: context.DeadlineExceeded}
	collector := New(executor, nil, time.Second)
	for _, key := range []string{"ssh.transport.node-1", "endpoint.tcp.registry.example.com:443"} {
		fact := collector.collectKey(ctx, api.Subject{
			ID: "node-1", Address: "192.0.2.10", Metadata: map[string]string{"sshPort": "22"},
		}, key)
		assert.False(t, fact.Available)
		assert.Equal(t, api.FactErrTimeout, fact.Error.Code)
	}
}

func TestBoundedEvidence(t *testing.T) {
	assert.Equal(t, "value", boundedEvidence(" value "))
	assert.Len(t, boundedEvidence(string(make([]byte, 600))), 515)
}

func assertErr(msg string) error { return &errStr{msg} }

type errStr struct{ s string }

func (e *errStr) Error() string { return e.s }

func TestCollectorID(t *testing.T) {
	c := New(&fakeExec{}, nil, time.Second)
	assert.Equal(t, "ssh-node", c.ID())
}
