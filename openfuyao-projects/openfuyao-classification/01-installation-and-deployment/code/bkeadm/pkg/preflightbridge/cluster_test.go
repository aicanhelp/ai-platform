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
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	configv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"gopkg.openfuyao.cn/preflight/api"
)

func TestLoadBKENodesStrictAcceptsNodeListMetadata(t *testing.T) {
	file := filepath.Join(t.TempDir(), "nodes.yaml")
	content := []byte(`apiVersion: bke.bocloud.com/v1beta1
kind: BKENodeList
metadata:
  name: demo-cluster-nodes
  namespace: default
items:
  - apiVersion: bke.bocloud.com/v1beta1
    kind: BKENode
    metadata:
      name: master-1
      namespace: default
    spec:
      role: [master, etcd]
      ip: 192.0.2.10
      port: "22"
      username: root
`)
	require.NoError(t, os.WriteFile(file, content, 0o600))

	nodes, err := loadBKENodesStrict(file)
	require.NoError(t, err)
	require.Len(t, nodes, 1)
	assert.Equal(t, "master-1", nodes[0].Name)
	assert.Equal(t, "192.0.2.10", nodes[0].Spec.IP)
}

func TestResolveClusterCreateReportsMissingRequiredFiles(t *testing.T) {
	_, issues, err := ResolveClusterCreate("", "")
	require.Error(t, err)
	require.Len(t, issues, 1)
	assert.Equal(t, "CLUSTER-INPUT-REQUIRED", issues[0].Code)
	assert.Equal(t, api.IssueBlockStart, issues[0].Effect)
}

func TestParseClusterResourcesAcceptsStrictClusterAndMultiDocumentNodes(t *testing.T) {
	dir := t.TempDir()
	clusterFile := filepath.Join(dir, "cluster.yaml")
	nodesFile := filepath.Join(dir, "nodes.yaml")
	clusterYAML := []byte(`apiVersion: bke.bocloud.com/v1beta1
kind: BKECluster
metadata:
  name: demo
spec:
  clusterConfig: {}
`)
	nodesYAML := []byte(`apiVersion: bke.bocloud.com/v1beta1
kind: BKENode
metadata:
  name: master-1
spec:
  ip: 192.0.2.10
  role: [master, etcd]
---
apiVersion: bke.bocloud.com/v1beta1
kind: BKENode
metadata:
  name: worker-1
spec:
  ip: 192.0.2.11
  role: [worker]
`)
	require.NoError(t, os.WriteFile(clusterFile, clusterYAML, 0o600))
	require.NoError(t, os.WriteFile(nodesFile, nodesYAML, 0o600))

	resources, issues, err := ResolveClusterCreate(clusterFile, nodesFile)
	require.NoError(t, err)
	assert.Empty(t, issues)
	require.Len(t, resources.BKENodes, 2)
	assert.Equal(t, "master-1", resources.BKENodes[0].Name)
	assert.Empty(t, ResolveClusterResources(resources))
}

func TestParseClusterResourcesClassifiesSchemaAndFileErrors(t *testing.T) {
	dir := t.TempDir()
	clusterFile := filepath.Join(dir, "cluster.yaml")
	nodesFile := filepath.Join(dir, "nodes.yaml")
	require.NoError(t, os.WriteFile(clusterFile, []byte("unknownField: true\n"), 0o600))
	require.NoError(t, os.WriteFile(nodesFile, []byte("items: []\n"), 0o600))

	_, issues, err := ParseClusterResources(clusterFile, nodesFile)
	require.Error(t, err)
	require.Len(t, issues, 1)
	assert.Equal(t, "cluster.unknownField", issues[0].Field)
	assert.Equal(t, "CLUSTER-INPUT-SCHEMA", issues[0].Code)

	validCluster := []byte(`apiVersion: bke.bocloud.com/v1beta1
kind: BKECluster
metadata:
  name: demo
spec:
  clusterConfig: {}
`)
	require.NoError(t, os.WriteFile(clusterFile, validCluster, 0o600))
	_, issues, err = ParseClusterResources(clusterFile, nodesFile)
	require.Error(t, err)
	require.Len(t, issues, 1)
	assert.Equal(t, "nodes", issues[0].Field)

	_, err = readClusterInput(dir)
	assert.ErrorContains(t, err, "regular file")
	_, err = readClusterInput(filepath.Join(dir, "missing"))
	assert.Error(t, err)
}

func TestParseClusterResourcesReportsNestedSchemaFieldPaths(t *testing.T) {
	dir := t.TempDir()
	clusterFile := filepath.Join(dir, "cluster.yaml")
	nodesFile := filepath.Join(dir, "nodes.yaml")
	nestedUnknownCluster := []byte(`apiVersion: bke.bocloud.com/v1beta1
kind: BKECluster
metadata:
  name: demo
spec:
  clusterConfig:
    cluster:
      networking:
        unexpectedCIDR: 192.0.2.0/24
`)
	require.NoError(t, os.WriteFile(clusterFile, nestedUnknownCluster, 0o600))
	require.NoError(t, os.WriteFile(nodesFile, []byte("items: []\n"), 0o600))

	_, issues, err := ParseClusterResources(clusterFile, nodesFile)
	require.Error(t, err)
	require.Len(t, issues, 1)
	assert.Equal(t, "cluster.spec.clusterConfig.cluster.networking.unexpectedCIDR", issues[0].Field)

	validCluster := []byte(`apiVersion: bke.bocloud.com/v1beta1
kind: BKECluster
metadata:
  name: demo
spec:
  clusterConfig: {}
`)
	nestedUnknownNode := []byte(`apiVersion: bke.bocloud.com/v1beta1
kind: BKENodeList
items:
  - metadata:
      name: master-1
    spec:
      ip: 192.0.2.10
      unexpectedPort: 22
`)
	require.NoError(t, os.WriteFile(clusterFile, validCluster, 0o600))
	require.NoError(t, os.WriteFile(nodesFile, nestedUnknownNode, 0o600))

	_, issues, err = ParseClusterResources(clusterFile, nodesFile)
	require.Error(t, err)
	require.Len(t, issues, 1)
	assert.Equal(t, "nodes[0].spec.unexpectedPort", issues[0].Field)

	wrongClusterType := []byte(`apiVersion: bke.bocloud.com/v1beta1
kind: BKECluster
metadata:
  name: demo
spec:
  controlPlaneEndpoint:
    port: invalid
  clusterConfig: {}
`)
	require.NoError(t, os.WriteFile(clusterFile, wrongClusterType, 0o600))
	_, issues, err = ParseClusterResources(clusterFile, nodesFile)
	require.Error(t, err)
	require.Len(t, issues, 1)
	assert.Equal(t, "cluster.spec.controlPlaneEndpoint.port", issues[0].Field)

	wrongNodeType := []byte(`apiVersion: bke.bocloud.com/v1beta1
kind: BKENode
metadata:
  name: master-1
spec:
  ip: 192.0.2.10
  role: master
`)
	require.NoError(t, os.WriteFile(clusterFile, validCluster, 0o600))
	require.NoError(t, os.WriteFile(nodesFile, wrongNodeType, 0o600))
	_, issues, err = ParseClusterResources(clusterFile, nodesFile)
	require.Error(t, err)
	require.Len(t, issues, 1)
	assert.Equal(t, "nodes[0].spec.role", issues[0].Field)
}

func TestClusterPlanFactsAllowEmptyHostnamesAndRejectDuplicateNonEmptyHostnames(t *testing.T) {
	nodes := []configv1beta1.BKENode{
		{Spec: configv1beta1.BKENodeSpec{IP: "192.0.2.10"}},
		{Spec: configv1beta1.BKENodeSpec{IP: "192.0.2.11"}},
	}
	facts := clusterPlanFacts(nodes, ResolveResult{})
	assert.Equal(t, true, factValue(facts, "unique.hostname"))
	nodes[0].Spec.Hostname = "same"
	nodes[1].Spec.Hostname = "same"
	facts = clusterPlanFacts(nodes, ResolveResult{})
	assert.Equal(t, false, factValue(facts, "unique.hostname"))
}

func TestProvenBootstrapHAMasterProducesColocationFact(t *testing.T) {
	facts := clusterPlanFacts(nil, ResolveResult{
		BootstrapNodeID: "master-01", MasterCount: 3,
		NodeRoles: map[string][]string{"master-01": {"master", "worker"}},
	})
	assert.Equal(t, true, factValue(facts, "cluster.ha.colocated"))
}

func factValue(facts []api.Fact, key string) any {
	for _, fact := range facts {
		if fact.Key == key {
			return fact.Value
		}
	}
	return nil
}

func TestClusterAdapterInputDoesNotCarrySSHPassword(t *testing.T) {
	cluster := &configv1beta1.BKECluster{}
	cluster.Spec.ClusterConfig = &configv1beta1.BKEConfig{}
	node := configv1beta1.BKENode{Spec: configv1beta1.BKENodeSpec{
		IP: "192.0.2.10", Username: "root", Password: "do-not-leak", Role: []string{"master"},
	}}
	node.Name = "master-01"
	input, err := (ClusterResources{BKECluster: cluster, BKENodes: []configv1beta1.BKENode{node}}).ToAdapterInput()
	require.NoError(t, err)
	require.NotNil(t, input.ClusterCreate)
	assert.NotContains(t, string(input.ClusterCreate.NodesYAML), "do-not-leak")
	assert.False(t, strings.Contains(strings.ToLower(string(input.ClusterCreate.NodesYAML)), "password:"))
}

func TestSanitizeSensitiveLeavesKeepsSecretReferences(t *testing.T) {
	document := map[string]any{
		"password": "secret", "identityToken": "token",
		"authSecretRef": map[string]any{"name": "repo-auth", "passwordKey": "password"},
	}
	sanitizeSensitiveLeaves(document)
	assert.NotContains(t, document, "password")
	assert.NotContains(t, document, "identityToken")
	assert.Contains(t, document, "authSecretRef")
}

func TestResolveClusterResourcesKeepsBusinessFailuresForReport(t *testing.T) {
	cluster := &configv1beta1.BKECluster{}
	cluster.Spec.ClusterConfig = &configv1beta1.BKEConfig{}
	node := configv1beta1.BKENode{}
	node.Name = "invalid-node"
	issues := ResolveClusterResources(&ClusterResources{BKECluster: cluster, BKENodes: []configv1beta1.BKENode{node}})
	assert.Empty(t, issues, "schema-valid business failures must be evaluated as Report findings")
}

func TestResolveClusterResourcesReportsStructuralPrerequisites(t *testing.T) {
	tests := []struct {
		name, field string
		resources   *ClusterResources
	}{
		{name: "missing cluster", field: "cluster"},
		{name: "missing cluster config", field: "clusterConfig", resources: &ClusterResources{
			BKECluster: &configv1beta1.BKECluster{},
		}},
		{name: "missing nodes", field: "nodes", resources: &ClusterResources{
			BKECluster: &configv1beta1.BKECluster{Spec: configv1beta1.BKEClusterSpec{
				ClusterConfig: &configv1beta1.BKEConfig{},
			}},
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			issues := ResolveClusterResources(tt.resources)
			require.Len(t, issues, 1)
			assert.Equal(t, tt.field, issues[0].Field)
			assert.Equal(t, api.IssueBlockStart, issues[0].Effect)
		})
	}
}

func TestApplyProviderManagementDefaultsAffectsResolvedSources(t *testing.T) {
	cluster := &configv1beta1.BKECluster{}
	cluster.Spec.ClusterConfig = &configv1beta1.BKEConfig{}
	resources := &ClusterResources{BKECluster: cluster}
	require.NoError(t, ApplyProviderManagementDefaults(resources, map[string]string{
		"host": "192.0.2.20", "imageRepoPort": "5443", "yumRepoPort": "8080", "imageRepoPrefix": "openfuyao",
	}))
	config := resources.BKECluster.Spec.ClusterConfig.Cluster
	assert.Equal(t, "192.0.2.20", config.ImageRepo.Ip)
	assert.Equal(t, "5443", config.ImageRepo.Port)
	assert.Equal(t, "openfuyao", config.ImageRepo.Prefix)
	assert.Equal(t, "192.0.2.20", config.HTTPRepo.Ip)
	assert.Equal(t, "8080", config.HTTPRepo.Port)
}

// TestDecodeBKENodeDocumentsEdgeCases freezes the multi-document node loader:
// empty files, empty documents between markers and per-document field paths.
func TestDecodeBKENodeDocumentsEdgeCases(t *testing.T) {
// Empty input is rejected.
_, err := decodeBKENodeDocuments([]byte(""))
assert.ErrorContains(t, err, "no BKENode resources")

// Documents that are only whitespace are skipped, so a file of blank
// separators is still empty.
_, err = decodeBKENodeDocuments([]byte("\n---\n\n---\n"))
assert.ErrorContains(t, err, "no BKENode resources")

// A multi-document file with an invalid field reports the document index.
_, err = decodeBKENodeDocuments([]byte(`apiVersion: bke.bocloud.com/v1beta1
kind: BKENode
metadata: {name: node-1}
spec: {ip: "192.0.2.11"}
---
apiVersion: bke.bocloud.com/v1beta1
kind: BKENode
metadata: {name: node-2}
spec: {ip: "192.0.2.12", unknownField: true}
`))
require.Error(t, err)
assert.Contains(t, inputErrorField("nodes", err), "nodes[1]")

// A valid multi-document file decodes every node in order.
nodes, err := decodeBKENodeDocuments([]byte(`apiVersion: bke.bocloud.com/v1beta1
kind: BKENode
metadata: {name: node-1}
spec: {ip: "192.0.2.11"}
---
apiVersion: bke.bocloud.com/v1beta1
kind: BKENode
metadata: {name: node-2}
spec: {ip: "192.0.2.12"}
`))
require.NoError(t, err)
require.Len(t, nodes, 2)
assert.Equal(t, "node-1", nodes[0].Name)
assert.Equal(t, "node-2", nodes[1].Name)
}

// TestDecodeBKENodeDocumentKindBranches freezes the single-document forms:
// list, slice and single node, plus their empty-rejection rules.
func TestDecodeBKENodeDocumentKindBranches(t *testing.T) {
// List form with empty items is rejected.
_, err := decodeBKENodeDocument([]byte(`apiVersion: bke.bocloud.com/v1beta1
kind: BKENodeList
items: []
`))
assert.ErrorContains(t, err, "at least one BKENode")

// Slice form with zero entries is rejected.
_, err = decodeBKENodeDocument([]byte("[]\n"))
assert.ErrorContains(t, err, "at least one BKENode")

// Single-node form decodes one node.
nodes, err := decodeBKENodeDocument([]byte(`apiVersion: bke.bocloud.com/v1beta1
kind: BKENode
metadata: {name: node-1}
spec: {ip: "192.0.2.11"}
`))
require.NoError(t, err)
require.Len(t, nodes, 1)
assert.Equal(t, "node-1", nodes[0].Name)
}

// TestJoinInputFieldShapes freezes the field-path joining rules.
func TestJoinInputFieldShapes(t *testing.T) {
assert.Equal(t, "nodes[0]", joinInputField("nodes", "[0]"))
assert.Equal(t, "nodes[1].spec.ip", joinInputField("nodes", "items[1].spec.ip"))
assert.Equal(t, "cluster.spec.ip", joinInputField("cluster", "spec.ip"))
}
