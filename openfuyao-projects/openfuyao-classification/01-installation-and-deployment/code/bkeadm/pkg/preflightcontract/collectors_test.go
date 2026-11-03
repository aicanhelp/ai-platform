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

package preflightcontract_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gopkg.openfuyao.cn/preflight/api"
	"gopkg.openfuyao.cn/preflight/collectors/artifact"
	"gopkg.openfuyao.cn/preflight/collectors/endpoint"
	"gopkg.openfuyao.cn/preflight/collectors/kubernetes"
	"gopkg.openfuyao.cn/preflight/collectors/local"
	"gopkg.openfuyao.cn/preflight/collectors/router"
	"gopkg.openfuyao.cn/preflight/collectors/ssh"
)

const contractTimeout = time.Second

func TestArtifactCollectorContract(t *testing.T) {
	content := []byte("release artifact")
	path := filepath.Join(t.TempDir(), "openfuyao-arm64.tar.gz")
	require.NoError(t, os.WriteFile(path, content, 0o600))
	collector := artifact.New()
	assert.Equal(t, "file-artifact", collector.ID())

	expectations := map[string]any{
		"exists": true, "readable": true, "format": "tar.gz",
	}
	for kind, expected := range expectations {
		fact := collectOne(t, collector, context.Background(), "artifact."+path+"."+kind, api.Subject{})
		assert.True(t, fact.Available, kind)
		assert.Equal(t, expected, fact.Value, kind)
	}

	unsupported := collectOne(t, collector, context.Background(), "artifact."+path+".unknown", api.Subject{})
	assert.Equal(t, api.FactErrUnsupported, unsupported.Error.Code)
	canceled := canceledContext()
	timedOut := collectOne(t, collector, canceled, "artifact."+path+".exists", api.Subject{})
	assert.Equal(t, api.FactErrTimeout, timedOut.Error.Code)
}

func TestEndpointCollectorContract(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, listener.Close()) })

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		username, password, ok := request.BasicAuth()
		if !ok || username != "operator" || password != "secret" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(server.Close)
	parsed, err := url.Parse(server.URL)
	require.NoError(t, err)
	collector := endpoint.NewWithCredentials(contractTimeout, map[string]endpoint.Credential{
		parsed.Host: {Username: "operator", Password: "secret"},
	})
	assert.Equal(t, "endpoint", collector.ID())

	for _, key := range []string{
		"endpoint.dns.localhost",
		"endpoint.tcp." + listener.Addr().String(),
		"endpoint.auth." + server.URL,
		"endpoint.object." + server.URL,
	} {
		fact := collectOne(t, collector, context.Background(), key, api.Subject{ID: "bootstrap"})
		assert.True(t, fact.Available, key)
		assert.Equal(t, true, fact.Value, key)
		assert.Equal(t, "bootstrap", fact.Subject, key)
	}

	badTarget := collectOne(t, collector, context.Background(), "endpoint.tcp.://bad", api.Subject{})
	assert.Equal(t, api.FactErrParseError, badTarget.Error.Code)
	unknown := collectOne(t, collector, context.Background(), "endpoint.unknown.value", api.Subject{})
	assert.Equal(t, api.FactErrUnsupported, unknown.Error.Code)
	timedOut := collectOne(t, collector, canceledContext(), "endpoint.tcp.127.0.0.1:1", api.Subject{})
	assert.Equal(t, api.FactErrTimeout, timedOut.Error.Code)
}

func TestEndpointCollectorFailureContract(t *testing.T) {
	statusServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	t.Cleanup(statusServer.Close)
	collector := endpoint.New(time.Second)
	auth := collectOne(t, collector, context.Background(), "endpoint.auth."+statusServer.URL, api.Subject{})
	object := collectOne(t, collector, context.Background(), "endpoint.object."+statusServer.URL, api.Subject{})
	assert.Equal(t, true, auth.Value)
	assert.Equal(t, false, object.Value)

	unauthorized := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	t.Cleanup(unauthorized.Close)
	unauthorizedFact := collectOne(t, collector, context.Background(), "endpoint.auth."+unauthorized.URL, api.Subject{})
	assert.True(t, unauthorizedFact.Available)
	assert.Equal(t, false, unauthorizedFact.Value)

	tlsServer := httptest.NewTLSServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	tlsServer.Config.ErrorLog = log.New(io.Discard, "", 0)
	t.Cleanup(tlsServer.Close)
	untrusted := collectOne(t, collector, context.Background(), "endpoint.tls."+tlsServer.URL, api.Subject{})
	assert.Equal(t, api.FactErrUnreachable, untrusted.Error.Code)
	invalid := collectOne(t, collector, context.Background(), "endpoint.tls.://bad", api.Subject{})
	assert.Equal(t, api.FactErrParseError, invalid.Error.Code)
}

func TestEndpointCollectorNetworkFailureContract(t *testing.T) {
	collector := endpoint.New(time.Second)
	assert.Equal(t, "endpoint", endpoint.New(0).ID())

	dnsFailure := collectOne(t, collector, context.Background(),
		"endpoint.dns.invalid.invalid.invalid", api.Subject{})
	assert.Contains(t, []string{api.FactErrUnreachable, api.FactErrTimeout}, dnsFailure.Error.Code)

	timeoutServer := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, request *http.Request) {
		<-request.Context().Done()
	}))
	t.Cleanup(timeoutServer.Close)
	timedOut := collectOne(t, endpoint.New(20*time.Millisecond), context.Background(),
		"endpoint.object."+timeoutServer.URL, api.Subject{})
	assert.Equal(t, api.FactErrTimeout, timedOut.Error.Code)

	redirectServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		http.Redirect(w, request, request.URL.String(), http.StatusTemporaryRedirect)
	}))
	t.Cleanup(redirectServer.Close)
	redirected := collectOne(t, collector, context.Background(),
		"endpoint.object."+redirectServer.URL, api.Subject{})
	assert.Equal(t, api.FactErrUnreachable, redirected.Error.Code)
}

func TestEndpointCollectorBearerChallengeContract(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("WWW-Authenticate",
			`Bearer realm="https://127.0.0.1:1/token",service="registry",scope="repository:test:pull"`)
		w.WriteHeader(http.StatusUnauthorized)
	}))
	t.Cleanup(server.Close)

	fact := collectOne(t, endpoint.New(time.Second), context.Background(),
		"endpoint.auth."+server.URL, api.Subject{})
	assert.True(t, fact.Available)
	assert.Equal(t, false, fact.Value)

	authFact := endpoint.AuthFact("endpoint.auth.registry.example.com", true,
		"https://operator:secret@registry.example.com:443/v2")
	assert.True(t, authFact.Available)
	assert.NotContains(t, authFact.Evidence[0].Locator, "secret")
}

func TestLocalCollectorContract(t *testing.T) {
	workspace := t.TempDir()
	artifactPath := filepath.Join(workspace, "bundle.tar.gz")
	require.NoError(t, os.WriteFile(artifactPath, []byte("bundle"), 0o600))
	probe := &fakeHostProbe{files: map[string]bool{artifactPath: true}}
	collector := local.NewWithVerifiedPlatforms(probe, []api.VerifiedPlatform{{
		Distribution: "openeuler", Version: "24.03", Architectures: []string{"arm64"},
		KernelPrefixes: []string{"6.6."},
	}})
	subject := api.Subject{ID: "bootstrap", Metadata: map[string]string{
		"workspace": workspace, "outputDir": workspace, "runtimeStorage": workspace,
		"expectedOutputs": "cluster.yaml,nodes.yaml", "imageRepoCAFile": artifactPath,
	}}
	keys := []string{
		"platform.os", "platform.distribution", "platform.kernel", "platform.kernel.matched",
		"platform.architecture", "resources.cpu.logical", "resources.memory.totalBytes",
		"resources.systemDisk.totalBytes", "path." + artifactPath, "path.local.accessible",
		"path.local.output.noUnknown", "path.local.output.type", "path.local.output.noSymlink",
		"path.workspace.exists", "path.workspace.availableBytes", "path.workspace.availableInodes",
		"runtime.dataRoot.conflict", "artifact.runc.exists", "artifact.runc.readable",
		"artifact.runc.arch", "port.6443.tcp.free", "tool.tar",
	}
	facts, err := collector.Collect(context.Background(), api.FactRequest{Keys: keys, Subjects: []api.Subject{subject}})
	require.NoError(t, err)
	require.Len(t, facts, len(keys))
	for _, fact := range facts {
		assert.True(t, fact.Available, fact.Key)
		assert.Equal(t, "bootstrap", fact.Subject, fact.Key)
	}
	unknown := collectOne(t, collector, context.Background(), "unknown.fact", subject)
	assert.Equal(t, api.FactErrUnsupported, unknown.Error.Code)
	nilProbe := collectOne(t, local.New(nil), context.Background(), "platform.os", subject)
	assert.Equal(t, "UNSUPPORTED", nilProbe.Error.Code)
	timedOut := collectOne(t, collector, canceledContext(), "platform.os", subject)
	assert.Equal(t, api.FactErrTimeout, timedOut.Error.Code)
}

func TestLocalCollectorFilesystemContract(t *testing.T) {
	workspace := t.TempDir()
	pemPath := filepath.Join(workspace, "repo-ca.pem")
	require.NoError(t, os.WriteFile(pemPath, []byte("-----BEGIN CERTIFICATE-----\nAQID\n"+
		"-----END CERTIFICATE-----\n"), 0o400))
	probe := &fakeHostProbe{files: map[string]bool{workspace: true, pemPath: true}}
	collector := local.New(probe)
	subject := api.Subject{ID: "bootstrap", Metadata: map[string]string{
		"workspace": workspace, "outputDir": workspace, "imageRepoCAFile": pemPath,
	}}
	keys := []string{
		"path.workspace.creatable", "path.workspace.writable", "path.workspace.mountReadOnly",
		"path.workspace.writeProtected", "runtime.socket.conflict", "certificate.repo.parseable",
		"privateKey.repo.format", "sysctl.preflight.missing", "port.invalid.tcp.free",
	}
	facts, err := collector.Collect(context.Background(), api.FactRequest{Keys: keys, Subjects: []api.Subject{subject}})
	require.NoError(t, err)
	require.Len(t, facts, len(keys))
	assert.Equal(t, false, facts[5].Value)
	assert.Equal(t, "CERTIFICATE", facts[6].Value)
	assert.Equal(t, api.FactErrNotFound, facts[7].Error.Code)
	assert.Equal(t, api.FactErrParseError, facts[8].Error.Code)
}

func TestLocalDefaultProbeContract(t *testing.T) {
	probe := local.DefaultProbe()
	path := filepath.Join(t.TempDir(), "readable")
	require.NoError(t, os.WriteFile(path, []byte("data"), 0o600))
	assert.NotEmpty(t, probe.OSFamily())
	probe.Distribution()
	probe.Kernel()
	assert.NotEmpty(t, probe.Architecture())
	assert.Positive(t, probe.CPUCount())
	probe.MemoryTotalBytes()
	probe.SystemDiskTotalBytes()
	assert.True(t, probe.FileExists(path))
	assert.True(t, probe.FileReadable(path))
	assert.True(t, probe.PortFree("tcp", 0))
	assert.True(t, probe.ToolExists("sh"))
}

func TestSSHCollectorContract(t *testing.T) {
	executor := &fakeSSHExecutor{}
	collector := ssh.NewWithVerifiedPlatforms(executor, nil, contractTimeout, []api.VerifiedPlatform{{
		Distribution: "openeuler", Version: "24.03", Architectures: []string{"arm64"},
		KernelPrefixes: []string{"6.6."},
	}})
	assert.Equal(t, "ssh-node", collector.ID())
	assert.True(t, ssh.Allowed("uname"))
	assert.False(t, ssh.Allowed("rm"))
	subject := api.Subject{ID: "node-1", Address: "192.0.2.10", Metadata: map[string]string{
		"sshPort": "22", "sshUser": "operator",
	}}
	keys := []string{
		"ssh.transport.node-1", "ssh.auth.node-1", "ssh.privilege.node-1", "systemd.available",
		"platform.os", "platform.distribution", "platform.kernel", "platform.kernel.node-1.matched",
		"resources.cpu.logical", "resources.memory.totalBytes", "resources.systemDisk.totalBytes",
		"port.6443.free", "endpoint.dns.registry.example.com",
		"endpoint.tcp.https://registry.example.com:5443/v2", "endpoint.auth.registry.example.com",
		"endpoint.object.registry.example.com/image",
	}
	facts, err := collector.Collect(context.Background(), api.FactRequest{Keys: keys, Subjects: []api.Subject{subject}})
	require.NoError(t, err)
	require.Len(t, facts, len(keys))
	for _, fact := range facts {
		assert.True(t, fact.Available, fact.Key)
	}
	assert.Greater(t, executor.readCalls, 0)

	failed := ssh.New(&fakeSSHExecutor{err: errors.New("authentication failed")}, nil, contractTimeout)
	auth := collectOne(t, failed, context.Background(), "ssh.auth.node-1", subject)
	assert.Equal(t, api.FactErrUnreachable, auth.Error.Code)
	unsupported := collectOne(t, ssh.New(nil, nil, contractTimeout), context.Background(), "platform.os", subject)
	assert.Equal(t, "UNSUPPORTED", unsupported.Error.Code)
	unknown := collectOne(t, collector, context.Background(), "unknown.fact", subject)
	assert.Equal(t, api.FactErrUnsupported, unknown.Error.Code)
}

func TestKubernetesAndRouterCollectorContracts(t *testing.T) {
	snapshot := &fakeSnapshot{discovery: api.Discovery{ServerVersion: "v1.30.0", Resources: []api.DiscoveryResource{{
		Resource: "bkeclusters", Group: "bke.openfuyao.io", Version: "v1beta1", Kind: "BKECluster",
	}}}, object: api.Object{Exists: true}}
	collector := kubernetes.New(snapshot)
	assert.Equal(t, "kubernetes-readonly", collector.ID())
	for _, key := range []string{
		"kubernetes.api.reachable", "kubernetes.discovery", "crd.bkeclusters", "crd.bkenodes",
		"object.exists.v1|Secret|kube-system|localkubeconfig",
	} {
		fact := collectOne(t, collector, context.Background(), key, api.Subject{})
		assert.True(t, fact.Available, key)
	}
	badObject := collectOne(t, collector, context.Background(), "object.exists.bad", api.Subject{})
	assert.Equal(t, api.FactErrParseError, badObject.Error.Code)
	nilSnapshot := collectOne(t, kubernetes.New(nil), context.Background(), "kubernetes.discovery", api.Subject{})
	assert.Equal(t, "UNSUPPORTED", nilSnapshot.Error.Code)

	routed := router.New(router.Route{Match: func(api.Subject, string) bool { return true }, Collector: collector})
	assert.Equal(t, "collector-router", routed.ID())
	fact := collectOne(t, routed, context.Background(), "kubernetes.api.reachable", api.Subject{ID: "cluster"})
	assert.True(t, fact.Available)
	_, err := routed.Collect(context.Background(), api.FactRequest{Keys: []string{"one", "two"}})
	assert.Error(t, err)
	unmatched := router.New()
	fact = collectOne(t, unmatched, context.Background(), "unknown", api.Subject{ID: "cluster"})
	assert.Equal(t, api.FactErrUnsupported, fact.Error.Code)

	failingSnapshot := kubernetes.New(&fakeSnapshot{err: errors.New("API unavailable")})
	for _, key := range []string{
		"kubernetes.api.reachable", "kubernetes.discovery", "crd.bkeclusters",
		"object.exists.v1|Secret|default|sample",
	} {
		fact = collectOne(t, failingSnapshot, context.Background(), key, api.Subject{})
		assert.Equal(t, api.FactErrUnreachable, fact.Error.Code, key)
	}
}

func collectOne(
	t *testing.T,
	collector api.Collector,
	ctx context.Context,
	key string,
	subject api.Subject,
) api.Fact {
	t.Helper()
	request := api.FactRequest{Keys: []string{key}}
	if subject.ID != "" {
		request.Subjects = []api.Subject{subject}
	}
	facts, err := collector.Collect(ctx, request)
	require.NoError(t, err)
	require.Len(t, facts, 1)
	return facts[0]
}

func canceledContext() context.Context {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	return ctx
}

type fakeHostProbe struct{ files map[string]bool }

func (*fakeHostProbe) OSFamily() string                { return "Linux" }
func (*fakeHostProbe) Distribution() (string, string)  { return "openEuler", "24.03" }
func (*fakeHostProbe) Kernel() string                  { return "6.6.0-100" }
func (*fakeHostProbe) Architecture() string            { return "aarch64" }
func (*fakeHostProbe) CPUCount() int                   { return 8 }
func (*fakeHostProbe) MemoryTotalBytes() uint64        { return 16 << 30 }
func (*fakeHostProbe) SystemDiskTotalBytes() uint64    { return 100 << 30 }
func (p *fakeHostProbe) FileExists(path string) bool   { return p.files[path] }
func (p *fakeHostProbe) FileReadable(path string) bool { return p.files[path] || path != "" }
func (*fakeHostProbe) PortFree(string, int) bool       { return true }
func (*fakeHostProbe) ToolExists(name string) bool     { return name == "tar" }

type fakeSSHExecutor struct {
	err          error
	transportErr error
	readCalls    int
}

func (e *fakeSSHExecutor) DialTransport(context.Context, api.Subject) error { return e.transportErr }

func (e *fakeSSHExecutor) RunRead(
	_ context.Context,
	_ api.Subject,
	command api.SSHReadCommand,
) (api.SSHReadResult, error) {
	e.readCalls++
	if e.err != nil {
		return api.SSHReadResult{}, e.err
	}
	output := map[string]string{
		"uname -s": "Linux", "uname -r": "6.6.0-100", "uname -m": "aarch64",
		"cat /etc/os-release": "ID=openeuler\nVERSION_ID=24.03", "nproc": "8",
		"free -b": "Mem: 17179869184 0 0", "df -B1 /": "Filesystem 1B-blocks\n/dev/vda 107374182400",
		"ss -H -lntup": "LISTEN 0 128 0.0.0.0:22", "systemctl --version": "systemd 255",
		"sudo -n true": "", "getent hosts registry.example.com": "192.0.2.20 registry.example.com",
	}
	key := fmt.Sprint(command.Name)
	if len(command.Args) > 0 {
		key += " " + fmt.Sprint(command.Args[0])
		for _, argument := range command.Args[1:] {
			key += " " + argument
		}
	}
	if command.Name == "curl" {
		return api.SSHReadResult{Stdout: "200"}, nil
	}
	return api.SSHReadResult{Stdout: output[key]}, nil
}

type fakeSnapshot struct {
	discovery api.Discovery
	object    api.Object
	err       error
}

func (s *fakeSnapshot) Get(context.Context, api.ObjectKey) (api.Object, error) {
	return s.object, s.err
}

func (*fakeSnapshot) List(context.Context, api.ListQuery) ([]api.Object, error) { return nil, nil }

func (s *fakeSnapshot) Discovery(context.Context) (api.Discovery, error) {
	return s.discovery, s.err
}
