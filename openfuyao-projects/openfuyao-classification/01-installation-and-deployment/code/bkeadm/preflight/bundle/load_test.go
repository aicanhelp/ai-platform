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

package bundle

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"

	"gopkg.openfuyao.cn/preflight/api"
)

// validBundle returns a contract-conforming bundle used as the baseline.
func validBundle() *api.PreflightBundle {
	return &api.PreflightBundle{
		Operations: []api.Operation{api.OperationInit, api.OperationClusterCreate},
		Rules: []api.RuleTemplate{
			{
				ID:        "INIT-PORT-36443",
				Operation: api.OperationInit,
				Fact:      "port.36443.free",
				Op:        api.OpEq,
				Expected:  true,
				Level:     api.SeverityError,
				Unknown:   api.UnknownBlock,
				Fix:       "stop the process using port 36443",
			},
		},
	}
}

// marshalBundle is a test helper that YAML-encodes a bundle for Load input.
func marshalBundle(t *testing.T, b *api.PreflightBundle) []byte {
	out, err := yaml.Marshal(b)
	require.NoError(t, err)
	return out
}

// TestLoadAcceptsValidBundle freezes that a well-formed bundle loads.
func TestLoadAcceptsValidBundle(t *testing.T) {
	b := validBundle()
	loaded, err := Load(marshalBundle(t, b))
	require.NoError(t, err)
	assert.Len(t, loaded.Rules, 1)
	assert.Equal(t, "INIT-PORT-36443", loaded.Rules[0].ID)
}

// TestLoadRejectsEmpty freezes that an empty/nil bundle is a BUNDLE_ERROR.
func TestLoadRejectsEmpty(t *testing.T) {
	_, err := Load(nil)
	require.Error(t, err)
	assert.True(t, IsLoadError(err))
	assert.Equal(t, "bundle is empty", err.(*LoadError).Reason)

	_, err = Load([]byte{})
	require.Error(t, err)
	assert.True(t, IsLoadError(err))
}

// TestLoadErrorFormat freezes both message shapes of LoadError.
func TestLoadErrorFormat(t *testing.T) {
	assert.Equal(t, "bundle load error: boom", (&LoadError{Reason: "boom"}).Error())
	assert.Equal(t, "bundle load error: rules: boom", (&LoadError{Field: "rules", Reason: "boom"}).Error())
}

// TestDecodeRejectsMultiDocument freezes the single-document rule.
func TestDecodeRejectsMultiDocument(t *testing.T) {
	single := marshalBundle(t, validBundle())
	_, err := decode(append(append([]byte{}, single...), append([]byte("\n---\n"), single...)...))
	require.Error(t, err)
	assert.ErrorContains(t, err, "single YAML document")
}

// TestLooksSensitiveAndContainsFold freeze the sensitive-fragment scan.
func TestLooksSensitiveAndContainsFold(t *testing.T) {
	assert.True(t, looksSensitive("set password=abc first"))
	assert.True(t, looksSensitive("Token=xyz"))
	assert.False(t, looksSensitive("free the port and retry"))
	assert.True(t, containsFold("Password=x", "password="))
	assert.False(t, containsFold("short", "longer-substring"))
	assert.False(t, containsFold("anything", ""))
	assert.True(t, containsFold("aBc", "abc"))
}

// TestLoadRejectsBadYAML freezes that unparseable input is a BUNDLE_ERROR.
func TestLoadRejectsBadYAML(t *testing.T) {
	_, err := Load([]byte("::: not [valid yaml"))
	require.Error(t, err)
	assert.True(t, IsLoadError(err))
}

// TestLoadRejectsUnknownFields freezes that strict decoding rejects unknown
// keys, so a typo or stale field cannot silently survive into a release.
func TestLoadRejectsUnknownFields(t *testing.T) {
	raw := []byte(`
schemaVersion: v1
release: v26.06
operations: [init]
rules: []
bogusField: should-not-be-allowed
`)
	_, err := Load(raw)
	require.Error(t, err)
	assert.True(t, IsLoadError(err))
}

// TestLoadRejectsMultiDocument freezes that a bundle is a single document.
func TestLoadRejectsMultiDocument(t *testing.T) {
	b := validBundle()
	doc := marshalBundle(t, b)
	// Two concatenated documents.
	_, err := Load(append(doc, append([]byte("\n---\n"), doc...)...))
	require.Error(t, err)
	assert.True(t, IsLoadError(err))
}

// TestValidateRejectsUnsupportedOperation freezes the operation allowlist.
func TestValidateRejectsUnsupportedOperation(t *testing.T) {
	b := validBundle()
	b.Operations = []api.Operation{api.Operation("build")}
	err := Validate(b)
	require.Error(t, err)
	assert.Equal(t, "operations", err.(*LoadError).Field)
}

// TestValidateRejectsDuplicateRuleIDs freezes that duplicate IDs are detected.
func TestValidateRejectsDuplicateRuleIDs(t *testing.T) {
	b := validBundle()
	b.Rules = append(b.Rules, api.RuleTemplate{
		ID: "INIT-PORT-36443", Operation: api.OperationInit, Fact: "x", Op: api.OpEq,
		Expected: true, Level: api.SeverityError, Unknown: api.UnknownBlock, Fix: "f",
	})
	err := Validate(b)
	require.Error(t, err)
	assert.Contains(t, err.(*LoadError).Reason, "duplicate rule ID INIT-PORT-36443")
}

// TestValidateRejectsRuleWithUndeclaredOperation freezes that a rule's
// operation must be declared in the bundle's operations list.
func TestValidateRejectsRuleWithUndeclaredOperation(t *testing.T) {
	b := validBundle()
	b.Operations = []api.Operation{api.OperationInit} // cluster-create not declared
	b.Rules[0].Operation = api.OperationClusterCreate
	err := Validate(b)
	require.Error(t, err)
	assert.Contains(t, err.(*LoadError).Reason, "not in bundle operations")
}

// TestValidateRejectsBadRule freezes that an invalid rule surfaces as BUNDLE_ERROR.
func TestValidateRejectsBadRule(t *testing.T) {
	b := validBundle()
	b.Rules[0].Op = api.Op("matches") // unsupported operator
	err := Validate(b)
	require.Error(t, err)
	assert.True(t, IsLoadError(err))
}

func TestValidateRejectsMissingDependency(t *testing.T) {
	b := validBundle()
	b.Rules[0].DependsOn = []string{"MISSING"}
	err := Validate(b)
	require.Error(t, err)
	assert.Contains(t, err.(*LoadError).Reason, "missing dependency MISSING")
}

func TestValidateRejectsDependencyCycle(t *testing.T) {
	b := validBundle()
	b.Rules = append(b.Rules, api.RuleTemplate{
		ID: "INIT-SECOND", Operation: api.OperationInit, Fact: "second", Op: api.OpEq,
		Expected: true, Level: api.SeverityError, Unknown: api.UnknownBlock,
		Fix: "fix", DependsOn: []string{b.Rules[0].ID},
	})
	b.Rules[0].DependsOn = []string{"INIT-SECOND"}
	err := Validate(b)
	require.Error(t, err)
	assert.Contains(t, err.(*LoadError).Reason, "dependency cycle")
}

func TestValidateRejectsCrossOperationDependency(t *testing.T) {
	b := validBundle()
	b.Operations = []api.Operation{api.OperationInit, api.OperationClusterCreate}
	b.Rules = append(b.Rules, api.RuleTemplate{
		ID: "CLUSTER-SECOND", Operation: api.OperationClusterCreate, Fact: "second", Op: api.OpEq,
		Expected: true, Level: api.SeverityError, Unknown: api.UnknownBlock,
		Fix: "fix", DependsOn: []string{b.Rules[0].ID},
	})
	err := Validate(b)
	require.Error(t, err)
	assert.Contains(t, err.(*LoadError).Reason, "belongs to operation")
}

func TestValidateRejectsInvalidVerifiedPlatforms(t *testing.T) {
	valid := api.VerifiedPlatform{
		Distribution: "openeuler", Version: "22.03", Architectures: []string{"amd64"},
		KernelPrefixes: []string{"5.10."},
	}
	tests := []struct {
		name      string
		platforms []api.VerifiedPlatform
		reason    string
	}{
		{name: "missing identity", platforms: []api.VerifiedPlatform{{}}, reason: "required"},
		{name: "duplicate", platforms: []api.VerifiedPlatform{valid, valid}, reason: "duplicate"},
		{name: "unsupported architecture", platforms: []api.VerifiedPlatform{{
			Distribution: "openeuler", Version: "22.03", Architectures: []string{"ppc64le"},
			KernelPrefixes: []string{"5.10."},
		}}, reason: "unsupported architecture"},
		{name: "invalid kernel prefix", platforms: []api.VerifiedPlatform{{
			Distribution: "openeuler", Version: "22.03", Architectures: []string{"amd64"},
			KernelPrefixes: []string{"not a kernel"},
		}}, reason: "invalid kernel prefix"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assertInvalidBundleMetadata(t, tt.platforms, tt.reason)
		})
	}
}

func assertInvalidBundleMetadata(
	t *testing.T,
	platforms []api.VerifiedPlatform,
	reason string,
) {
	t.Helper()
	bundle := validBundle()
	bundle.VerifiedPlatforms = platforms
	err := Validate(bundle)
	require.Error(t, err)
	assert.Contains(t, err.(*LoadError).Reason, reason)
}

// TestRejectSensitiveFix freezes that a remediation string embedding a
// password literal is rejected.
func TestRejectSensitiveFix(t *testing.T) {
	b := validBundle()
	b.Rules[0].Fix = "set password=hunter2 in /etc/foo"
	err := Validate(b)
	require.Error(t, err)
	assert.True(t, IsLoadError(err))
}
