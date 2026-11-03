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

package local

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gopkg.openfuyao.cn/preflight/api"
)

// fakeProbe is a stub HostProbe for deterministic tests.
type fakeProbe struct {
	os        string
	distID    string
	distVer   string
	kernel    string
	arch      string
	cpu       int
	mem       uint64
	disk      uint64
	files     map[string]bool
	readable  map[string]bool
	portsFree map[string]bool // keyed "tcp:36443"
	tools     map[string]bool
}

func (f *fakeProbe) OSFamily() string                     { return f.os }
func (f *fakeProbe) Distribution() (string, string)       { return f.distID, f.distVer }
func (f *fakeProbe) Kernel() string                       { return f.kernel }
func (f *fakeProbe) Architecture() string                 { return f.arch }
func (f *fakeProbe) CPUCount() int                        { return f.cpu }
func (f *fakeProbe) MemoryTotalBytes() uint64             { return f.mem }
func (f *fakeProbe) SystemDiskTotalBytes() uint64         { return f.disk }
func (f *fakeProbe) FileExists(p string) bool             { return f.files[p] }
func (f *fakeProbe) FileReadable(p string) bool           { return f.readable[p] }
func (f *fakeProbe) PortFree(proto string, port int) bool { return f.portsFree[proto+":"+itoa(port)] }
func (f *fakeProbe) ToolExists(name string) bool          { return f.tools[name] }

func mkProbe() *fakeProbe {
	return &fakeProbe{
		os: "linux", distID: "openEuler", distVer: "22.03", kernel: "5.10.0",
		arch: "x86_64", cpu: 8, mem: 16 * 1024 * 1024 * 1024, disk: 100 * 1024 * 1024 * 1024,
		files: map[string]bool{"/bke": true}, readable: map[string]bool{"/bke": true},
		portsFree: map[string]bool{"tcp:36443": true, "tcp:6443": false},
		tools:     map[string]bool{"tar": true, "iptables": true},
	}
}

func TestCollectPlatformFacts(t *testing.T) {
	c := New(mkProbe())
	facts, err := c.Collect(context.Background(), api.FactRequest{
		Keys: []string{"platform.os", "platform.distribution", "platform.kernel", "platform.architecture"},
	})
	assert.NoError(t, err)
	byKey := indexFacts(facts)
	assert.Equal(t, "linux", byKey["platform.os"].Value)
	assert.Equal(t, []string{"openeuler", "22.03"}, byKey["platform.distribution"].Value)
	assert.Equal(t, "5.10.0", byKey["platform.kernel"].Value)
	assert.Equal(t, "amd64", byKey["platform.architecture"].Value)
}

func TestVerifiedKernelMatrix(t *testing.T) {
	collector := NewWithVerifiedPlatforms(mkProbe(), []api.VerifiedPlatform{{
		Distribution: "openeuler", Version: "22.03", Architectures: []string{"amd64"},
		KernelPrefixes: []string{"5.10."},
	}})
	facts, err := collector.Collect(context.Background(), api.FactRequest{Keys: []string{"platform.kernel.matched"}})
	assert.NoError(t, err)
	assert.True(t, facts[0].Available)
	assert.Equal(t, true, facts[0].Value)
}

func TestCollectResourceFacts(t *testing.T) {
	c := New(mkProbe())
	facts, _ := c.Collect(context.Background(), api.FactRequest{
		Keys: []string{"resources.cpu.logical", "resources.memory.totalBytes", "resources.systemDisk.totalBytes"},
	})
	byKey := indexFacts(facts)
	assert.Equal(t, 8, byKey["resources.cpu.logical"].Value)
	assert.Equal(t, uint64(16*1024*1024*1024), byKey["resources.memory.totalBytes"].Value)
	assert.Equal(t, uint64(100*1024*1024*1024), byKey["resources.systemDisk.totalBytes"].Value)
}

func TestCollectPathFacts(t *testing.T) {
	c := New(mkProbe())
	facts, _ := c.Collect(context.Background(), api.FactRequest{Keys: []string{"path./bke", "path./missing"}})
	byKey := indexFacts(facts)
	assert.True(t, byKey["path./bke"].Available)
	assert.Equal(t, true, byKey["path./bke"].Value.(map[string]any)["exists"])
	assert.Equal(t, true, byKey["path./bke"].Value.(map[string]any)["readable"])
	// Missing path: exists=false.
	assert.Equal(t, false, byKey["path./missing"].Value.(map[string]any)["exists"])
}

func TestExpectedOutputSafety(t *testing.T) {
	dir := t.TempDir()
	const outputs = "bke-demo.yaml,bke-node-node.yaml"
	assert.True(t, expectedOutputsAbsent(dir, outputs))
	assert.True(t, expectedOutputsRegular(dir, outputs))
	assert.True(t, expectedOutputsNoSymlink(dir, outputs))

	require.NoError(t, os.WriteFile(filepath.Join(dir, "bke-demo.yaml"), []byte("kind: BKECluster\n"), 0o600))
	assert.False(t, expectedOutputsAbsent(dir, outputs))
	assert.True(t, expectedOutputsRegular(dir, outputs))
	assert.True(t, expectedOutputsNoSymlink(dir, outputs))

	require.NoError(t, os.Symlink("bke-demo.yaml", filepath.Join(dir, "bke-node-node.yaml")))
	assert.False(t, expectedOutputsRegular(dir, outputs))
	assert.False(t, expectedOutputsNoSymlink(dir, outputs))
}

func TestCollectPortFacts(t *testing.T) {
	c := New(mkProbe())
	facts, _ := c.Collect(context.Background(), api.FactRequest{
		Keys: []string{"port.36443.free", "port.6443.free", "port.123.udp.free"},
	})
	byKey := indexFacts(facts)
	assert.True(t, byKey["port.36443.free"].Available)
	assert.Equal(t, true, byKey["port.36443.free"].Value) // free
	assert.Equal(t, false, byKey["port.6443.free"].Value) // occupied
	// udp 123 not in map -> false (occupied)
	assert.Equal(t, false, byKey["port.123.udp.free"].Value)
}

func TestCollectToolFacts(t *testing.T) {
	c := New(mkProbe())
	facts, _ := c.Collect(context.Background(), api.FactRequest{Keys: []string{"tool.tar", "tool.missing"}})
	byKey := indexFacts(facts)
	assert.True(t, byKey["tool.tar"].Available)
	assert.Equal(t, true, byKey["tool.tar"].Value)
	assert.Equal(t, false, byKey["tool.missing"].Value)
}

func TestCollectUnknownKeyIsUnavailable(t *testing.T) {
	c := New(mkProbe())
	facts, _ := c.Collect(context.Background(), api.FactRequest{Keys: []string{"bogus.key"}})
	assert.Len(t, facts, 1)
	assert.False(t, facts[0].Available)
	assert.Equal(t, api.FactErrUnsupported, facts[0].Error.Code)
}

func TestCollectNilProbeIsUnavailable(t *testing.T) {
	c := New(nil)
	facts, _ := c.Collect(context.Background(), api.FactRequest{Keys: []string{"platform.os"}})
	assert.False(t, facts[0].Available)
	assert.Equal(t, "UNSUPPORTED", facts[0].Error.Code)
}

func TestCollectContextCanceledIsTimeout(t *testing.T) {
	c := New(mkProbe())
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	facts, _ := c.Collect(ctx, api.FactRequest{Keys: []string{"platform.os"}})
	assert.False(t, facts[0].Available)
	assert.Equal(t, api.FactErrTimeout, facts[0].Error.Code)
}

func TestParseOSRelease(t *testing.T) {
	content := `NAME="openEuler"
ID="openEuler"
VERSION_ID="22.03"`
	id, ver := parseOSRelease(content)
	assert.Equal(t, "openEuler", id)
	assert.Equal(t, "22.03", ver)
}

func TestCollectSemanticPathFacts(t *testing.T) {
	dir := t.TempDir()
	outputDir := filepath.Join(dir, "output")
	require.NoError(t, os.Mkdir(outputDir, 0o755))
	protected := filepath.Join(dir, "protected")
	require.NoError(t, os.WriteFile(protected, []byte("data"), 0o444))
	probe := mkProbe()
	probe.readable[outputDir] = true
	probe.files[protected] = true
	probe.readable[protected] = true
	collector := New(probe)
	subject := api.Subject{ID: "bootstrap", Metadata: map[string]string{
		"workspace": dir, "outputDir": outputDir, "imageRepoCAFile": protected,
		"expectedOutputs": "cluster.yaml,nodes.yaml", "runtimeStorage": outputDir,
	}}

	facts, err := collector.Collect(context.Background(), api.FactRequest{
		Subjects: []api.Subject{subject},
		Keys: []string{
			"path.local.accessible", "path.local.output.noUnknown", "path.local.output.type",
			"path.local.output.noSymlink", "path.caCert.exists", "path.caCert.readable",
			"path.workspace.creatable", "path.workspace.availableBytes", "path.workspace.availableInodes",
			"path.workspace.mountReadOnly", "path.caCert.writeProtected", "runtime.dataRoot.conflict",
		},
	})
	require.NoError(t, err)
	byKey := indexFacts(facts)
	for _, key := range []string{
		"path.local.accessible", "path.local.output.noUnknown", "path.local.output.type",
		"path.local.output.noSymlink", "path.caCert.exists", "path.caCert.readable",
		"path.workspace.creatable", "path.caCert.writeProtected",
	} {
		assert.Equal(t, true, byKey[key].Value, key)
	}
	assert.Equal(t, false, byKey["path.workspace.mountReadOnly"].Value)
	assert.Greater(t, byKey["path.workspace.availableBytes"].Value.(int64), int64(0))
	assert.Greater(t, byKey["path.workspace.availableInodes"].Value.(int64), int64(0))
	assert.Equal(t, false, byKey["runtime.dataRoot.conflict"].Value)
}

func TestCollectSemanticPathRejectsUnknownSuffix(t *testing.T) {
	collector := New(mkProbe())
	facts, err := collector.Collect(context.Background(), api.FactRequest{
		Keys: []string{"path.workspace.unknown"}, Subjects: []api.Subject{{Metadata: map[string]string{}}},
	})
	require.NoError(t, err)
	assert.False(t, facts[0].Available)
	assert.Equal(t, api.FactErrUnsupported, facts[0].Error.Code)
}

func TestCertificateAndPrivateKeyFacts(t *testing.T) {
	dir := t.TempDir()
	caPath := filepath.Join(dir, "ca.pem")
	keyPath := filepath.Join(dir, "key.pem")
	privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
	require.NoError(t, err)
	template := &x509.Certificate{
		SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: "preflight-test"},
		NotBefore: time.Now().Add(-time.Minute), NotAfter: time.Now().Add(time.Hour),
		IsCA: true, BasicConstraintsValid: true, KeyUsage: x509.KeyUsageCertSign,
	}
	certificate, err := x509.CreateCertificate(rand.Reader, template, template, &privateKey.PublicKey, privateKey)
	require.NoError(t, err)
	certificatePEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certificate})
	require.NoError(t, os.WriteFile(caPath, certificatePEM, 0o600))
	require.NoError(t, os.WriteFile(keyPath, pem.EncodeToMemory(&pem.Block{
		Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(privateKey),
	}), 0o600))
	collector := New(mkProbe())

	facts, err := collector.Collect(context.Background(), api.FactRequest{
		Keys:     []string{"certificate.repo.parseable", "certificate.repo.isCA"},
		Subjects: []api.Subject{{Metadata: map[string]string{"imageRepoCAFile": caPath}}},
	})
	require.NoError(t, err)
	for _, fact := range facts {
		assert.Equal(t, true, fact.Value, fact.Key)
	}
	facts, err = collector.Collect(context.Background(), api.FactRequest{
		Keys:     []string{"privateKey.repo.format"},
		Subjects: []api.Subject{{Metadata: map[string]string{"imageRepoCAFile": keyPath}}},
	})
	require.NoError(t, err)
	assert.Equal(t, "RSA PRIVATE KEY", facts[0].Value)

	facts, err = collector.Collect(context.Background(), api.FactRequest{
		Keys:     []string{"certificate.repo.parseable"},
		Subjects: []api.Subject{{Metadata: map[string]string{"imageRepoCAFile": filepath.Join(dir, "missing")}}},
	})
	require.NoError(t, err)
	assert.Equal(t, api.FactErrNotFound, facts[0].Error.Code)
}

func TestExecutableAndSysctlFacts(t *testing.T) {
	collector := New(mkProbe())
	facts, err := collector.Collect(context.Background(), api.FactRequest{
		Keys: []string{
			"artifact.runc.exists", "artifact.runc.readable", "artifact.runc.arch",
			"artifact.runc.unknown", "sysctl.not.present",
		},
	})
	require.NoError(t, err)
	byKey := indexFacts(facts)
	assert.NotEmpty(t, byKey["artifact.runc.arch"].Value)
	assert.Equal(t, api.FactErrUnsupported, byKey["artifact.runc.unknown"].Error.Code)
	assert.Equal(t, api.FactErrNotFound, byKey["sysctl.not.present"].Error.Code)
}

func TestStdlibProbeReadOnlyOperations(t *testing.T) {
	probe := DefaultProbe()
	dir := t.TempDir()
	file := filepath.Join(dir, "readable")
	require.NoError(t, os.WriteFile(file, []byte("value"), 0o600))

	assert.NotEmpty(t, probe.OSFamily())
	assert.NotEmpty(t, probe.Architecture())
	assert.Positive(t, probe.CPUCount())
	assert.True(t, probe.FileExists(file))
	assert.True(t, probe.FileReadable(file))
	assert.False(t, probe.FileExists(filepath.Join(dir, "missing")))
	assert.False(t, probe.FileReadable(filepath.Join(dir, "missing")))
	assert.True(t, probe.PortFree("tcp", 0))
	assert.True(t, probe.PortFree("udp", 0))
	assert.True(t, probe.ToolExists("go"))
	assert.False(t, probe.ToolExists("definitely-not-a-preflight-tool"))
	assert.Greater(t, diskTotalBytes(dir), uint64(0))
}

func TestParsersHandleMissingAndValidValues(t *testing.T) {
	assert.Equal(t, uint64(16*1024), parseMemTotal("MemTotal: 16 kB\n"))
	assert.Zero(t, parseMemTotal("MemTotal:\n"))
	assert.Zero(t, parseMemTotal("SwapTotal: 16 kB\n"))
	id, version := parseOSRelease("ID='openeuler'\nVERSION_ID=\"22.03\"\n")
	assert.Equal(t, "openeuler", id)
	assert.Equal(t, "22.03", version)
	assert.Equal(t, "[2001:db8::1]:22", joinHostPort("2001:db8::1", 22))
}

func indexFacts(facts []api.Fact) map[string]api.Fact {
	m := make(map[string]api.Fact, len(facts))
	for _, f := range facts {
		m[f.Key] = f
	}
	return m
}
