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

package normalize

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gopkg.openfuyao.cn/preflight/api"
)

// TestPlanHashDeterministic freezes the core PlanHash contract: the same
// semantic input always produces the same digest, formatted as sha256:hex.
func TestPlanHashDeterministic(t *testing.T) {
	plan := &api.OperationPlan{
		Operation: api.OperationInit,
		Subjects:  []api.Subject{{ID: "local", Role: []string{"bootstrap"}}},
		Values: map[string]api.Value{
			"platform.os":           {Key: "platform.os", Type: api.ValueTypeEnum, Value: "linux"},
			"resources.cpu.logical": {Key: "resources.cpu.logical", Type: api.ValueTypeInteger, Value: 4},
		},
	}
	h1, err := ComputePlanHash(plan)
	require.NoError(t, err)
	assert.True(t, strings.HasPrefix(h1.String(), "sha256:"))
	assert.False(t, h1.IsZero())

	// Re-compute: identical.
	h2, err := ComputePlanHash(plan)
	require.NoError(t, err)
	assert.Equal(t, h1, h2)
}

// TestEquivalentInputsSamePlanHash freezes the equivalence contract: two
// OperationPlans with the same semantics but built from different entry kinds
// (CLI vs request YAML) MUST hash identically, because Origin/entryKind are
// excluded from the projection.
func TestEquivalentInputsSamePlanHash(t *testing.T) {
	base := func() *api.OperationPlan {
		return &api.OperationPlan{
			Operation: api.OperationClusterCreate,
			Subjects: []api.Subject{
				{ID: "master-01", Role: []string{"master", "etcd", "worker"}, Address: "192.0.2.10"},
			},
			Values: map[string]api.Value{
				"cluster.apiPort": {Key: "cluster.apiPort", Type: api.ValueTypeInteger, Value: 6443},
			},
			Sources: []api.Source{{ID: "image", Kind: "image", Endpoint: "registry.example.com/openfuyao"}},
		}
	}

	// "CLI flags" entry: Origin from a file list.
	cli := base()
	cli.Origin = api.PlanOrigin{Files: []string{"/abs/cluster.yaml"}}

	// "Request YAML" entry: same semantics, different origin path.
	req := base()
	req.Origin = api.PlanOrigin{Files: []string{"/different/path/request.yaml"}}

	hCLI, err := ComputePlanHash(cli)
	require.NoError(t, err)
	hReq, err := ComputePlanHash(req)
	require.NoError(t, err)
	assert.Equal(t, hCLI, hReq, "equivalent CLI/request inputs must hash identically")
}

// TestNonEquivalentInputsDifferentPlanHash freezes the converse: a change in
// resolved semantics (a different port, subject or source) MUST change the
// hash, so stale reports cannot masquerade as current.
func TestNonEquivalentInputsDifferentPlanHash(t *testing.T) {
	mk := func(port int) *api.OperationPlan {
		return &api.OperationPlan{
			Operation: api.OperationInit,
			Values: map[string]api.Value{
				"port.k8s": {Key: "port.k8s", Type: api.ValueTypeInteger, Value: port},
			},
		}
	}
	h1, err := ComputePlanHash(mk(36443))
	require.NoError(t, err)
	h2, err := ComputePlanHash(mk(36444))
	require.NoError(t, err)
	assert.NotEqual(t, h1, h2)

	// A type or semantic key change must also invalidate the plan hash even
	// when the concrete value is identical.
	typed := mk(36443)
	typed.Values["port.k8s"] = api.Value{Key: "port.k8s", Type: api.ValueTypeBytes, Value: 36443}
	h2, err = ComputePlanHash(typed)
	require.NoError(t, err)
	assert.NotEqual(t, h1, h2)

	// Different subjects also change the hash.
	p1 := &api.OperationPlan{Operation: api.OperationInit, Subjects: []api.Subject{{ID: "a"}}}
	p2 := &api.OperationPlan{Operation: api.OperationInit, Subjects: []api.Subject{{ID: "b"}}}
	h1, _ = ComputePlanHash(p1)
	h2, _ = ComputePlanHash(p2)
	assert.NotEqual(t, h1, h2)
}

// TestSubjectAndSourceOrderIndependence freezes that the projection sorts
// subjects and sources by stable ID, so input order does not affect the hash.
func TestSubjectAndSourceOrderIndependence(t *testing.T) {
	mk := func(order []string) *api.OperationPlan {
		subs := make([]api.Subject, 0, len(order))
		for _, id := range order {
			subs = append(subs, api.Subject{ID: id, Address: "10.0.0.1"})
		}
		return &api.OperationPlan{Operation: api.OperationInit, Subjects: subs}
	}
	h1, err := ComputePlanHash(mk([]string{"master-01", "master-02"}))
	require.NoError(t, err)
	h2, err := ComputePlanHash(mk([]string{"master-02", "master-01"}))
	require.NoError(t, err)
	assert.Equal(t, h1, h2)
}

// TestSensitiveRejected freezes that a credential/secret key MUST NOT enter
// the hash. Canonicalize rejects it rather than silently dropping or hashing.
func TestSensitiveRejected(t *testing.T) {
	plan := &api.OperationPlan{
		Operation: api.OperationInit,
		Values: map[string]api.Value{
			"repo.password": {Key: "repo.password", Type: api.ValueTypeString, Value: "s3cr3t"},
		},
	}
	_, err := ComputePlanHash(plan)
	assert.ErrorIs(t, err, ErrSensitiveRejected)
}

func TestCredentialedURLRejected(t *testing.T) {
	_, err := Canonicalize(map[string]any{"endpoint": "https://user:secret@example.com/v2/"})
	assert.ErrorIs(t, err, ErrSensitiveRejected)
}

func TestSourceSchemeChangesPlanHash(t *testing.T) {
	base := &api.OperationPlan{Operation: api.OperationInit, Sources: []api.Source{{ID: "repo", Endpoint: "example.com:443", Scheme: "https"}}}
	other := &api.OperationPlan{Operation: api.OperationInit, Sources: []api.Source{{ID: "repo", Endpoint: "example.com:443", Scheme: "http"}}}
	httpsHash, err := ComputePlanHash(base)
	require.NoError(t, err)
	httpHash, err := ComputePlanHash(other)
	require.NoError(t, err)
	assert.NotEqual(t, httpsHash, httpHash)
}

func TestSourceObjectPathChangesPlanHash(t *testing.T) {
	base := &api.OperationPlan{Operation: api.OperationInit, Sources: []api.Source{{ID: "repo", Endpoint: "example.com:443", Scheme: "https", ObjectPath: "/v2/a/manifests/v1"}}}
	other := &api.OperationPlan{Operation: api.OperationInit, Sources: []api.Source{{ID: "repo", Endpoint: "example.com:443", Scheme: "https", ObjectPath: "/v2/a/manifests/v2"}}}
	first, err := ComputePlanHash(base)
	require.NoError(t, err)
	second, err := ComputePlanHash(other)
	require.NoError(t, err)
	assert.NotEqual(t, first, second)
}

// TestSensitiveKeyFragments enumerates the keys that must be rejected.
func TestSensitiveKeyFragments(t *testing.T) {
	for _, k := range []string{"password", "Password", "token", "api_key", "privateKey", "kubeconfig", "credential"} {
		assert.True(t, isSensitiveKey(k), "%q should be sensitive", k)
	}
	for _, k := range []string{"port", "cpu", "memory", "os", "distribution", "endpoint"} {
		assert.False(t, isSensitiveKey(k), "%q should NOT be sensitive", k)
	}
}

// TestNormalizeArchitecture freezes arch alias folding.
func TestNormalizeArchitecture(t *testing.T) {
	for in, want := range map[string]string{
		"amd64": "amd64", "x86_64": "amd64", "x64": "amd64", "AMD64": "amd64",
		"arm64": "arm64", "aarch64": "arm64", "ARM64": "arm64",
		"mips": "other", "": "other",
	} {
		assert.Equal(t, want, Architecture(in), in)
	}
}

// TestNormalizeOSFamily freezes OS family folding.
func TestNormalizeOSFamily(t *testing.T) {
	assert.Equal(t, "linux", OSFamily("Linux"))
	assert.Equal(t, "linux", OSFamily("GNU/Linux"))
	assert.Equal(t, "windows", OSFamily("Windows"))
	assert.Equal(t, "darwin", OSFamily("Darwin"))
	assert.Equal(t, "other", OSFamily(""))
}

// TestNormalizeDistribution freezes distribution alias folding.
func TestNormalizeDistribution(t *testing.T) {
	assert.Equal(t, "openeuler", Distribution("openEuler"))
	assert.Equal(t, "openeuler", Distribution("OPENEULER"))
	assert.Equal(t, "ubuntu", Distribution("Ubuntu"))
	assert.Equal(t, "uos", Distribution("UOS"))
	assert.Equal(t, "uos", Distribution("UnionTech OS"))
}

// TestNormalizeAddress freezes that equivalent IP representations hash the
// same and hostnames are lowercased.
func TestNormalizeAddress(t *testing.T) {
	assert.Equal(t, "192.0.2.1", Address("192.0.2.1"))
	assert.Equal(t, "192.0.2.1", Address(" 192.0.2.1 "))
	assert.Equal(t, "registry.example.com", Address("Registry.Example.COM"))
}

// TestParsePort freezes port validation bounds.
func TestParsePort(t *testing.T) {
	p, err := ParsePort("6443")
	require.NoError(t, err)
	assert.Equal(t, 6443, p)

	_, err = ParsePort("")
	assert.Error(t, err)
	_, err = ParsePort("abc")
	assert.Error(t, err)
	_, err = ParsePort("0")
	assert.Error(t, err)
	_, err = ParsePort("70000")
	assert.Error(t, err)
}

// TestRedactEndpoint freezes that credentials are stripped from endpoints
// before they can reach Evidence or a Report.
func TestRedactEndpoint(t *testing.T) {
	assert.Equal(t, "https://registry.example.com/openfuyao",
		RedactEndpoint("https://user:s3cr3t@registry.example.com/openfuyao"))
	assert.Equal(t, "http://host:8080", RedactEndpoint("http://host:8080"))
	assert.Equal(t, "", RedactEndpoint(""))
	// No scheme, no userinfo.
	assert.Equal(t, "host.example.com", RedactEndpoint("host.example.com"))
}

// TestCanonicalMapKeyOrder freezes that map keys are emitted in sorted order,
// making canonical output byte-stable regardless of map iteration order.
func TestCanonicalMapKeyOrder(t *testing.T) {
	m1 := map[string]any{"z": 1, "a": 2, "m": 3}
	m2 := map[string]any{"a": 2, "m": 3, "z": 1}
	c1, err := Canonicalize(m1)
	require.NoError(t, err)
	c2, err := Canonicalize(m2)
	require.NoError(t, err)
	assert.Equal(t, string(c1), string(c2))
	// Keys appear in sorted order.
	assert.True(t, strings.Index(string(c1), "\"a\"") < strings.Index(string(c1), "\"m\""))
	assert.True(t, strings.Index(string(c1), "\"m\"") < strings.Index(string(c1), "\"z\""))
}

func TestSortedKeysIsDeterministic(t *testing.T) {
keys := SortedKeys(map[string]any{"b": 1, "a": 2, "c": 3})
assert.Equal(t, []string{"a", "b", "c"}, keys)
assert.Empty(t, SortedKeys(nil))
}
