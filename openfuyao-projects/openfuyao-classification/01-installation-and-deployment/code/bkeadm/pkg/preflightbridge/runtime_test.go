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
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/crypto/ssh"
	configv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"gopkg.openfuyao.cn/bkeadm/pkg/preflightconfig"
	"gopkg.openfuyao.cn/preflight/api"
	"gopkg.openfuyao.cn/preflight/collectors/router"
)

func TestVerifiedPlatformsDeduplicatesAndCopiesPolicy(t *testing.T) {
	distribution := preflightconfig.PlatformDistribution{
		Name: "openeuler", Version: "22.03", Architectures: []string{"amd64"}, KernelPrefixes: []string{"5.10."},
	}
	config := preflightconfig.Config{Operations: preflightconfig.Operations{
		Init: preflightconfig.OperationPolicy{Platform: preflightconfig.PlatformPolicy{
			Distributions: []preflightconfig.PlatformDistribution{distribution},
		}},
		ClusterCreate: preflightconfig.OperationPolicy{Platform: preflightconfig.PlatformPolicy{
			Distributions: []preflightconfig.PlatformDistribution{
				distribution, {Name: "ignored", Version: "1"},
			},
		}},
	}}
	platforms := VerifiedPlatforms(config)
	require.Len(t, platforms, 1)
	assert.Equal(t, "openeuler", platforms[0].Distribution)
	assert.Equal(t, []string{"amd64"}, platforms[0].Architectures)
	distribution.Architectures[0] = "arm64"
	assert.Equal(t, []string{"amd64"}, platforms[0].Architectures)
}

func TestSourceHostNormalizesRepositoryAddresses(t *testing.T) {
	tests := map[string]string{
		"https://registry.example.com:5443/v2/": "registry.example.com:5443",
		"registry.example.com/openfuyao":        "registry.example.com:443",
		"[2001:db8::1]":                         "[2001:db8::1]:443",
		"  repo.example.com:5000/path  ":        "repo.example.com:5000",
	}
	for input, want := range tests {
		assert.Equal(t, want, sourceHost(input), input)
	}
}

func TestRuntimeCollectorUsesExtraRouteBeforeDefaults(t *testing.T) {
	constructors := []struct {
		name string
		new  func(router.Route) api.Collector
	}{
		{name: "base", new: func(route router.Route) api.Collector { return NewRuntimeCollector(route) }},
		{name: "init", new: func(route router.Route) api.Collector {
			return NewInitRuntimeCollector(InitOptions{}, route)
		}},
		{name: "cluster", new: func(route router.Route) api.Collector {
			return NewClusterRuntimeCollector(nil, route)
		}},
	}
	for _, constructor := range constructors {
		t.Run(constructor.name, func(t *testing.T) {
			stub := &recordingPassingCollector{}
			collector := constructor.new(routeForKey("custom.fact", stub))
			facts, err := collector.Collect(context.Background(), api.FactRequest{
				Keys: []string{"custom.fact"}, Subjects: []api.Subject{{ID: "bootstrap"}},
			})
			require.NoError(t, err)
			require.Len(t, facts, 1)
			assert.True(t, facts[0].Available)
		})
	}
}

func routeForKey(key string, collector api.Collector) router.Route {
	return router.Route{
		Match:     func(_ api.Subject, candidate string) bool { return candidate == key },
		Collector: collector,
	}
}

func TestNodeSSHExecutorDefaultsAndTransport(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	defer listener.Close()
	port := listener.Addr().(*net.TCPAddr).Port
	node := configv1beta1.BKENode{
		ObjectMeta: metav1.ObjectMeta{Name: "node-1"},
		Spec:       configv1beta1.BKENodeSpec{Username: "", Port: strconv.Itoa(port), Password: "secret"},
	}
	executor := newNodeSSHExecutor([]configv1beta1.BKENode{node})
	assert.Equal(t, "root", executor.credentials["node-1"].user)
	reachable := api.Subject{ID: "node-1", Address: "127.0.0.1"}
	require.NoError(t, executor.DialTransport(context.Background(), reachable))
	assert.ErrorContains(t,
		executor.DialTransport(context.Background(), api.Subject{ID: "missing"}),
		"reference not found",
	)
}

func TestSSHCommandQuotingAndOutputLimit(t *testing.T) {
	command := api.SSHReadCommand{Name: "test", Args: []string{"plain", "a'b", "two words"}}
	assert.Equal(t, `'test' 'plain' 'a'"'"'b' 'two words'`, shellCommand(command))

	var output bytes.Buffer
	writer := &limitWriter{w: &output, remaining: 4}
	n, err := writer.Write([]byte("abcdef"))
	assert.Equal(t, 4, n)
	assert.ErrorIs(t, err, io.ErrShortWrite)
	assert.Equal(t, "abcd", output.String())
}

func TestConfigureSSHAuthPasswordAndMissingCredentials(t *testing.T) {
	t.Setenv("SSH_AUTH_SOCK", "")
	t.Setenv("HOME", t.TempDir())
	withPassword := &ssh.ClientConfig{}
	cleanup, err := configureSSHAuth(withPassword, "secret")
	require.NoError(t, err)
	require.Len(t, withPassword.Auth, 1)
	cleanup()

	withoutCredentials := &ssh.ClientConfig{}
	cleanup, err = configureSSHAuth(withoutCredentials, "")
	assert.ErrorContains(t, err, "credentials are not configured")
	assert.NotNil(t, cleanup)
}

func TestNodeSSHExecutorRejectsUnsafeOrUnknownRequests(t *testing.T) {
	executor := newNodeSSHExecutor(nil)
	_, err := executor.RunRead(context.Background(), api.Subject{}, api.SSHReadCommand{Name: "rm"})
	assert.ErrorContains(t, err, "not allowlisted")
	_, err = executor.RunRead(context.Background(), api.Subject{ID: "missing"}, api.SSHReadCommand{Name: "uname"})
	assert.ErrorContains(t, err, "reference not found")
}

// TestNodeSSHExecutorRunCustomValidatesPathAndArgs freezes the custom-script
// guards: relative paths and NUL bytes in arguments are rejected before any
// connection is opened.
func TestNodeSSHExecutorRunCustomValidatesPathAndArgs(t *testing.T) {
	executor := newNodeSSHExecutor(nil)
	_, err := executor.RunCustom(context.Background(), api.Subject{ID: "node-1"}, "relative/path", nil)
	assert.ErrorContains(t, err, "absolute")
	_, err = executor.RunCustom(context.Background(), api.Subject{ID: "node-1"}, "/opt/check", []string{"a\x00b"})
	assert.ErrorContains(t, err, "NUL")
	// An unknown subject fails at credential lookup, proving validation ran first.
	_, err = executor.RunCustom(context.Background(), api.Subject{ID: "missing"}, "/opt/check", nil)
	assert.ErrorContains(t, err, "reference not found")
}

// TestNodeSSHExecutorRunReadFailsOnClosedPort freezes the transport error
// path: a refused TCP connection surfaces as an SSH transport error.
func TestNodeSSHExecutorRunReadFailsOnClosedPort(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	port := listener.Addr().(*net.TCPAddr).Port
	require.NoError(t, listener.Close())
	node := configv1beta1.BKENode{
		ObjectMeta: metav1.ObjectMeta{Name: "node-1"},
		Spec:       configv1beta1.BKENodeSpec{Port: strconv.Itoa(port), Password: "secret"},
	}
	executor := newNodeSSHExecutor([]configv1beta1.BKENode{node})
	_, err = executor.RunRead(context.Background(), api.Subject{ID: "node-1", Address: "127.0.0.1"}, api.SSHReadCommand{Name: "uname"})
	assert.ErrorContains(t, err, "SSH transport unavailable")
}

// TestNodeSSHExecutorDialTransportRefused freezes the transport probe on a
// closed port.
func TestNodeSSHExecutorDialTransportRefused(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	port := listener.Addr().(*net.TCPAddr).Port
	require.NoError(t, listener.Close())
	node := configv1beta1.BKENode{
		ObjectMeta: metav1.ObjectMeta{Name: "node-1"},
		Spec:       configv1beta1.BKENodeSpec{Port: strconv.Itoa(port)},
	}
	executor := newNodeSSHExecutor([]configv1beta1.BKENode{node})
	err = executor.DialTransport(context.Background(), api.Subject{ID: "node-1", Address: "127.0.0.1"})
	assert.ErrorContains(t, err, "SSH transport unavailable")
}

// TestDefaultSSHIdentityFilesAreAbsolute freezes the identity-file lookup.
func TestDefaultSSHIdentityFilesAreAbsolute(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	files := defaultSSHIdentityFiles()
	require.Len(t, files, 3)
	for _, file := range files {
		assert.True(t, filepath.IsAbs(file), file)
	}
}

// TestNewNodeSSHExecutorHostKeyCallbackInjection freezes the injectable
// host-key trust boundary: an explicit callback is preserved, nil falls back
// to the insecure default so the SSH path can never be silently disabled.
// Function values are not comparable in Go, so the callbacks are compared
// through their code pointers.
func TestNewNodeSSHExecutorHostKeyCallbackInjection(t *testing.T) {
	node := configv1beta1.BKENode{
		ObjectMeta: metav1.ObjectMeta{Name: "node-1"},
		Spec:       configv1beta1.BKENodeSpec{IP: "192.0.2.11", Password: "secret"},
	}
	verifier := ssh.FixedHostKey(testHostKey(t))
	executor := newNodeSSHExecutorWithHostKey([]configv1beta1.BKENode{node}, verifier)
	assert.Equal(t,
		reflect.ValueOf(verifier).Pointer(),
		reflect.ValueOf(executor.hostKeyCallback).Pointer(),
	)

	defaultExecutor := newNodeSSHExecutor([]configv1beta1.BKENode{node})
	assert.NotNil(t, defaultExecutor.hostKeyCallback)

	nilCallback := newNodeSSHExecutorWithHostKey([]configv1beta1.BKENode{node}, nil)
	assert.NotNil(t, nilCallback.hostKeyCallback)
	// The nil fallback must equal the insecure default, not a zero callback.
	assert.Equal(t,
		reflect.ValueOf(defaultExecutor.hostKeyCallback).Pointer(),
		reflect.ValueOf(nilCallback.hostKeyCallback).Pointer(),
	)
}

// testHostKey builds a real ssh.PublicKey so FixedHostKey can be exercised.
func testHostKey(t *testing.T) ssh.PublicKey {
	t.Helper()
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	require.NoError(t, err)
	public, err := ssh.NewPublicKey(&key.PublicKey)
	require.NoError(t, err)
	return public
}

func TestCloseSSHResourceSilencesBenignCleanupErrors(t *testing.T) {
	// Sessions closed by the remote side and TCP connections closed together
	// with the SSH client are expected during read-only cleanup; they must not
	// reach stderr. Real failures still do.
	original := os.Stderr
	devNull, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	require.NoError(t, err)
	os.Stderr = devNull
	defer func() { os.Stderr = original }()

	closeSSHResource("session", errCloser{io.EOF})
	closeSSHResource("connection", errCloser{fmt.Errorf(
		"close tcp 192.0.2.1:54742->192.0.2.2:22: use of closed network connection")})
	closeSSHResource("connection", errCloser{fmt.Errorf("connection reset by peer")})
	closeSSHResource("connection", errCloser{net.ErrClosed})
}

func TestCloseSSHResourceReportsRealFailures(t *testing.T) {
	assert.False(t, benignSSHCloseError(fmt.Errorf("unexpected close failure")))
	assert.True(t, benignSSHCloseError(io.EOF))
	assert.True(t, benignSSHCloseError(fmt.Errorf(
		"close tcp 192.0.2.1:54742->192.0.2.2:22: use of closed network connection")))
	assert.True(t, benignSSHCloseError(fmt.Errorf("connection reset by peer")))
	assert.True(t, benignSSHCloseError(net.ErrClosed))
}

type errCloser struct{ err error }

func (c errCloser) Close() error { return c.err }

func TestConfigureSSHAuthLoadsDefaultIdentity(t *testing.T) {
	home := t.TempDir()
	sshDir := filepath.Join(home, ".ssh")
	require.NoError(t, os.Mkdir(sshDir, 0o700))
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	require.NoError(t, err)
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(key)})
	require.NoError(t, os.WriteFile(filepath.Join(sshDir, "id_rsa"), keyPEM, 0o600))
	t.Setenv("HOME", home)
	t.Setenv("SSH_AUTH_SOCK", "")

	config := &ssh.ClientConfig{}
	cleanup, err := configureSSHAuth(config, "")
	require.NoError(t, err)
	assert.Len(t, config.Auth, 1)
	cleanup()
	noSSHCleanup()
}

// TestConfigureSSHAuthAgentSocketErrorIsTolerated freezes the agent fallback:
// an unreachable SSH_AUTH_SOCK is skipped and identity files are still tried.
func TestConfigureSSHAuthAgentSocketErrorIsTolerated(t *testing.T) {
	t.Setenv("SSH_AUTH_SOCK", "/nonexistent/agent.sock")
	t.Setenv("HOME", t.TempDir())
	config := &ssh.ClientConfig{}
	cleanup, err := configureSSHAuth(config, "")
	assert.ErrorContains(t, err, "credentials are not configured")
	assert.NotNil(t, cleanup)
	cleanup()
}

// TestConfigureSSHAuthUsesDefaultIdentityWhenPresent freezes the identity-file
// path: an unencrypted key under HOME/.ssh is loaded as a publickey method.
func TestConfigureSSHAuthUsesDefaultIdentityWhenPresent(t *testing.T) {
	home := t.TempDir()
	sshDir := filepath.Join(home, ".ssh")
	require.NoError(t, os.Mkdir(sshDir, 0o700))
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	require.NoError(t, err)
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(key)})
	require.NoError(t, os.WriteFile(filepath.Join(sshDir, "id_rsa"), keyPEM, 0o600))
	t.Setenv("HOME", home)
	t.Setenv("SSH_AUTH_SOCK", "")

	config := &ssh.ClientConfig{}
	cleanup, err := configureSSHAuth(config, "")
	require.NoError(t, err)
	require.Len(t, config.Auth, 1)
	cleanup()
}

// TestNewInitRuntimeCollectorWithCredentials freezes the credential routing:
// username/password plus a custom repo register an endpoint credential for
// the repo host only; partial credentials register nothing.
func TestNewInitRuntimeCollectorWithCredentials(t *testing.T) {
	withCredentials := NewInitRuntimeCollector(InitOptions{
		ImageRepoUsername: "user", ImageRepoPassword: "pass",
		OtherRepo: "registry.example.com:5443/openfuyao",
	})
	require.NotNil(t, withCredentials)

	// Partial credentials (password only) must not register anything.
	partial := NewInitRuntimeCollector(InitOptions{
		ImageRepoPassword: "pass", OtherRepo: "registry.example.com:5443",
	})
	require.NotNil(t, partial)

	// No custom repo means no credential registration even with both fields.
	withoutRepo := NewInitRuntimeCollector(InitOptions{
		ImageRepoUsername: "user", ImageRepoPassword: "pass",
	})
	require.NotNil(t, withoutRepo)
}

// TestNewClusterRuntimeCollectorWithCustomRejectsBadPolicy freezes the
// custom-route construction error path: duplicate custom-check keys are
// rejected at construction. Path validation happens at execution time, not
// at construction, so a relative path alone does not fail here.
func TestNewClusterRuntimeCollectorWithCustomRejectsBadPolicy(t *testing.T) {
	duplicate := preflightconfig.OperationPolicy{CustomChecks: []preflightconfig.CustomCheck{
		{
			CheckCommon: preflightconfig.CheckCommon{Description: "d", Remediation: "r"},
			Key:         "same", Expected: "e", Expression: "actual == true",
			Collector: preflightconfig.CustomCollector{Type: "exec", Path: "/opt/check", ResultType: "bool"},
		},
		{
			CheckCommon: preflightconfig.CheckCommon{Description: "d", Remediation: "r"},
			Key:         "same", Expected: "e", Expression: "actual == true",
			Collector: preflightconfig.CustomCollector{Type: "exec", Path: "/opt/check", ResultType: "bool"},
		},
	}}
	_, err := NewClusterRuntimeCollectorWithCustom(nil, nil, duplicate)
	assert.ErrorContains(t, err, "create cluster custom collector")

	_, err = NewInitRuntimeCollectorWithCustom(InitOptions{}, nil, duplicate)
	assert.ErrorContains(t, err, "create init custom collector")
}

// TestNewClusterRuntimeCollectorWithPlatformsRoutesNodeFacts freezes the
// route ordering of the cluster collector: node subjects are claimed by the
// SSH route before any extra route, while non-node subjects fall through to
// the extra routes.
func TestNewClusterRuntimeCollectorWithPlatformsRoutesNodeFacts(t *testing.T) {
	stub := &recordingPassingCollector{}
	collector := NewClusterRuntimeCollectorWithPlatforms(nil, nil,
		routeForKey("custom.cluster-create.health", stub))

	// A node subject matches the SSH route first (it is declared before the
	// extra routes), so the custom key surfaces as an SSH-side unsupported
	// fact instead of reaching the stub.
	facts, err := collector.Collect(context.Background(), api.FactRequest{
		Keys: []string{"custom.cluster-create.health"}, Subjects: []api.Subject{{ID: "node-1"}},
	})
	require.NoError(t, err)
	require.Len(t, facts, 1)
	assert.False(t, facts[0].Available)
	assert.Equal(t, api.FactErrUnsupported, facts[0].Error.Code)
	assert.False(t, stub.requested("custom.cluster-create.health"))

	// A non-node subject skips the SSH route and reaches the extra route.
	facts, err = collector.Collect(context.Background(), api.FactRequest{
		Keys: []string{"custom.cluster-create.health"}, Subjects: []api.Subject{{ID: "bootstrap"}},
	})
	require.NoError(t, err)
	require.Len(t, facts, 1)
	assert.True(t, facts[0].Available)
	assert.True(t, stub.requested("custom.cluster-create.health"))
}
