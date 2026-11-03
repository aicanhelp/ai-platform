/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package installer

import (
	"io"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	yamlv3 "gopkg.in/yaml.v3"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

func TestBuildCreateClusterYaml_Basic(t *testing.T) {
	req := CreateClusterRequest{
		Cluster: CreateClusterCluster{
			Name:             "bke-cluster",
			OpenFuyaoVersion: "latest",
			ImageRepo: CreateClusterImageRepo{
				Url: "example.com",
				Ip:  "1.1.1.1",
			},
		},
		Nodes: []CreateClusterNode{
			{
				Hostname: "master1",
				Ip:       "192.168.0.10",
				Port:     "",
				Username: "root",
				Password: "pwd",
				Role:     []string{"master", "etcd"},
			},
		},
	}
	defaults := DefaultResp{
		KubernetesVersion: "v1.2.3",
		ContainerRuntime:  "containerd",
		AgentHealthPort:   "58080",
		Ip:                "10.0.0.1",
	}

	yamlString, err := BuildCreateClusterYaml(req, defaults)
	require.NoError(t, err)

	docs := decodeClusterYamlDocs(t, yamlString)
	require.Len(t, docs, 2)

	clusterDoc := findDocByKind(t, docs, "BKECluster")
	nodeDoc := findDocByKind(t, docs, "BKENode")

	assert.Equal(t, "bke-cluster", clusterDoc.GetName())
	assert.Equal(t, "bke-cluster", clusterDoc.GetNamespace())

	openVer, found, err := unstructured.NestedString(clusterDoc.Object, "spec", "clusterConfig", "cluster", "openFuyaoVersion")
	require.NoError(t, err)
	require.True(t, found)
	assert.Equal(t, "latest", openVer)

	cri, found, err := unstructured.NestedString(clusterDoc.Object, "spec", "clusterConfig", "cluster", "containerRuntime", "cri")
	require.NoError(t, err)
	require.True(t, found)
	assert.Equal(t, "containerd", cri)

	httpIP, _, _ := unstructured.NestedString(clusterDoc.Object, "spec", "clusterConfig", "cluster", "httpRepo", "ip")
	customHost, _, _ := unstructured.NestedString(clusterDoc.Object, "spec", "clusterConfig", "customExtra", "host")
	assert.Equal(t, "10.0.0.1", httpIP)
	assert.Equal(t, "10.0.0.1", customHost)

	imageDomain, _, _ := unstructured.NestedString(clusterDoc.Object, "spec", "clusterConfig", "cluster", "imageRepo", "domain")
	imageIP, _, _ := unstructured.NestedString(clusterDoc.Object, "spec", "clusterConfig", "cluster", "imageRepo", "ip")
	assert.Equal(t, "example.com", imageDomain)
	assert.Equal(t, "1.1.1.1", imageIP)

	addons, _, err := unstructured.NestedSlice(clusterDoc.Object, "spec", "clusterConfig", "addons")
	require.NoError(t, err)
	assert.False(t, containsAddon(addons, openFuyaoAddonName))

	nodePort, _, _ := unstructured.NestedString(nodeDoc.Object, "spec", "port")
	assert.Equal(t, "22", nodePort)
	nodeRole, _, _ := unstructured.NestedStringSlice(nodeDoc.Object, "spec", "role")
	assert.Equal(t, []string{"master/node", "etcd"}, nodeRole)
}

func TestBuildCreateClusterYaml_OpenFuyaoAddonIncluded(t *testing.T) {
	req := CreateClusterRequest{
		Cluster: CreateClusterCluster{
			Name:             "cluster-addon",
			OpenFuyaoVersion: "latest",
		},
		Addons: []CreateClusterAddon{
			{
				Name:   openFuyaoAddonAlias,
				Params: map[string]string{"enabled": "true"},
			},
		},
		Nodes: []CreateClusterNode{
			{
				Hostname: "master1",
				Ip:       "192.168.0.10",
				Port:     "22",
				Username: "root",
				Password: "pwd",
				Role:     []string{"master"},
			},
		},
	}

	yamlString, err := BuildCreateClusterYaml(req, DefaultResp{})
	require.NoError(t, err)

	docs := decodeClusterYamlDocs(t, yamlString)
	clusterDoc := findDocByKind(t, docs, "BKECluster")

	addons, _, err := unstructured.NestedSlice(clusterDoc.Object, "spec", "clusterConfig", "addons")
	require.NoError(t, err)
	assert.True(t, containsAddon(addons, openFuyaoAddonName))
}

func TestMergeAddons_DisabledAndAppend(t *testing.T) {
	defaultAddons := []any{
		map[string]any{"name": openFuyaoAddonName},
		map[string]any{"name": "calico", "param": map[string]any{"mode": "vxlan"}},
	}
	reqAddons := []CreateClusterAddon{
		{Name: openFuyaoAddonName, Params: map[string]string{"enabled": "false"}},
		{Name: "new-addon", Params: map[string]string{"k": "v"}},
	}

	result := mergeAddons(defaultAddons, reqAddons)

	assert.False(t, containsAddon(result, openFuyaoAddonName))
	assert.True(t, containsAddon(result, "calico"))
	assert.True(t, containsAddon(result, "new-addon"))
}

func TestNormalizeNodeRoles(t *testing.T) {
	roles := normalizeNodeRoles([]string{"master", "worker", "etcd"})
	assert.Equal(t, []string{"master/node", "node", "etcd"}, roles)
}

func TestSetIfNotEmpty(t *testing.T) {
	obj := map[string]any{}
	setIfNotEmpty(&obj, "", "a", "b")
	_, found, _ := unstructured.NestedFieldNoCopy(obj, "a", "b")
	assert.False(t, found)

	setIfNotEmpty(&obj, "v", "a", "b")
	value, found, _ := unstructured.NestedString(obj, "a", "b")
	assert.True(t, found)
	assert.Equal(t, "v", value)
}

func decodeClusterYamlDocs(t *testing.T, yamlString string) []*unstructured.Unstructured {
	t.Helper()
	dec := yamlv3.NewDecoder(strings.NewReader(yamlString))
	var docs []*unstructured.Unstructured
	for {
		var obj map[string]any
		if err := dec.Decode(&obj); err != nil {
			if err == io.EOF {
				break
			}
			require.NoError(t, err)
		}
		if len(obj) == 0 {
			continue
		}
		docs = append(docs, &unstructured.Unstructured{Object: obj})
	}
	return docs
}

func findDocByKind(t *testing.T, docs []*unstructured.Unstructured, kind string) *unstructured.Unstructured {
	t.Helper()
	for _, doc := range docs {
		if doc.GetKind() == kind {
			return doc
		}
	}
	require.FailNow(t, "doc not found", "kind: %s", kind)
	return nil
}

func containsAddon(addons []any, name string) bool {
	for _, raw := range addons {
		addonMap, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if addonName, ok := addonMap["name"].(string); ok && addonName == name {
			return true
		}
	}
	return false
}
