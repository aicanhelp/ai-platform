/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package cluster

import (
	"fmt"
	"os"
	"path"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	configv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/securityinit"
	"gopkg.openfuyao.cn/bkeadm/utils"
)
func TestSplitLines(t *testing.T) {
	lines := splitLines([]byte("a\r\nb\nc"))
	require.Len(t, lines, 3)
	assert.Equal(t, "a", string(lines[0]))
	assert.Equal(t, "b", string(lines[1]))
	assert.Equal(t, "c", string(lines[2]))
}

func TestSplitYAMLDocuments(t *testing.T) {
	data := []byte(`apiVersion: v1
kind: BKENode
spec:
  ip: 10.0.0.1
---
apiVersion: v1
kind: BKENode
spec:
  ip: 10.0.0.2
`)
	docs := splitYAMLDocuments(data)
	require.Len(t, docs, 2)
	assert.Contains(t, string(docs[0]), "10.0.0.1")
	assert.Contains(t, string(docs[1]), "10.0.0.2")
}

func TestLoadBKENodesFromFile(t *testing.T) {
	dir := t.TempDir()

	singleNode := `apiVersion: bke.bocloud.com/v1beta1
kind: BKENode
metadata:
  name: node-1
spec:
  hostname: node-1
  ip: 10.0.0.1
`
	singleFile := path.Join(dir, "single.yaml")
	require.NoError(t, os.WriteFile(singleFile, []byte(singleNode), utils.DefaultFilePermission))
	nodes, err := loadBKENodesFromFile(singleFile)
	require.NoError(t, err)
	require.Len(t, nodes, 1)
	assert.Equal(t, "10.0.0.1", nodes[0].Spec.IP)

	multiDoc := `apiVersion: bke.bocloud.com/v1beta1
kind: BKENode
metadata:
  name: node-a
spec:
  hostname: node-a
  ip: 10.0.0.10
---
apiVersion: bke.bocloud.com/v1beta1
kind: BKENode
metadata:
  name: node-b
spec:
  hostname: node-b
  ip: 10.0.0.11
`
	multiFile := path.Join(dir, "multi.yaml")
	require.NoError(t, os.WriteFile(multiFile, []byte(multiDoc), utils.DefaultFilePermission))
	nodes, err = loadBKENodesFromFile(multiFile)
	require.NoError(t, err)
	require.Len(t, nodes, 2)

	invalidFile := path.Join(dir, "invalid.yaml")
	require.NoError(t, os.WriteFile(invalidFile, []byte("not: yaml: ["), utils.DefaultFilePermission))
	_, err = loadBKENodesFromFile(invalidFile)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no valid BKENode")
}

func TestLoadBKENodesFromFileMissing(t *testing.T) {
	_, err := loadBKENodesFromFile(path.Join(t.TempDir(), "missing.yaml"))
	assert.Error(t, err)
}

func TestApplyImmutableOSAnnotation(t *testing.T) {
	cluster := &configv1beta1.BKECluster{
		Spec: configv1beta1.BKEClusterSpec{
			ClusterConfig: &configv1beta1.BKEConfig{},
		},
	}

	applyImmutableOSAnnotation(cluster, "")
	assert.Nil(t, cluster.Annotations)

	applyImmutableOSAnnotation(cluster, utils.ImmutableOSAnnotationValueKubeOS)
	require.NotNil(t, cluster.Annotations)
	assert.Equal(t, utils.ImmutableOSAnnotationValueKubeOS, cluster.Annotations[utils.ImmutableOSAnnotation])
	assert.Equal(t, utils.ImmutableOSAnnotationValueKubeOS, cluster.Spec.ClusterConfig.CustomExtra["immutable-os"])

	clusterNoExtra := &configv1beta1.BKECluster{}
	applyImmutableOSAnnotation(clusterNoExtra, utils.ImmutableOSAnnotationValueKubeOS)
	require.NotNil(t, clusterNoExtra.Annotations)
	assert.Nil(t, clusterNoExtra.Spec.ClusterConfig)
}

func TestMarshalAndWriteNodeYAMLs(t *testing.T) {
	origWorkspace := global.Workspace
	global.Workspace = t.TempDir()
	t.Cleanup(func() { global.Workspace = origWorkspace })

	t.Run("empty namespace filled with cluster namespace", func(t *testing.T) {
		nodes := []configv1beta1.BKENode{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "node-1"},
				Spec:       configv1beta1.BKENodeSpec{IP: "10.0.0.1", Hostname: "node-1"},
			},
		}

		files, err := marshalAndWriteNodeYAMLs("demo-cluster", "demo", nodes)
		require.NoError(t, err)
		require.Len(t, files, 1)
		assert.FileExists(t, files[0])
		assert.Equal(t, "demo", nodes[0].Namespace)

		content, readErr := os.ReadFile(files[0])
		require.NoError(t, readErr)
		assert.Contains(t, string(content), "10.0.0.1")
		assert.Contains(t, string(content), "demo")
	})

	t.Run("mismatched namespace returns error", func(t *testing.T) {
		nodes := []configv1beta1.BKENode{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "node-2", Namespace: "other"},
				Spec:       configv1beta1.BKENodeSpec{IP: "10.0.0.2", Hostname: "node-2"},
			},
		}

		files, err := marshalAndWriteNodeYAMLs("demo-cluster", "demo", nodes)
		require.Error(t, err)
		assert.Nil(t, files)
		assert.Contains(t, err.Error(), "does not match BKECluster")
		assert.Contains(t, err.Error(), "demo-cluster")
		assert.Equal(t, "other", nodes[0].Namespace)
	})
}

func TestAlignBKENodesNamespace(t *testing.T) {
	t.Run("fill empty and keep matching", func(t *testing.T) {
		nodes := []configv1beta1.BKENode{
			{ObjectMeta: metav1.ObjectMeta{Name: "b"}},
			{ObjectMeta: metav1.ObjectMeta{Name: "c", Namespace: "demo"}},
		}
		require.NoError(t, alignBKENodesNamespace(nodes, "demo-cluster", "demo"))
		assert.Equal(t, "demo", nodes[0].Namespace)
		assert.Equal(t, "demo", nodes[1].Namespace)
	})

	t.Run("mismatch fails without rewrite", func(t *testing.T) {
		nodes := []configv1beta1.BKENode{
			{ObjectMeta: metav1.ObjectMeta{Name: "a", Namespace: "other"}},
		}
		err := alignBKENodesNamespace(nodes, "demo-cluster", "demo")
		require.Error(t, err)
		assert.Contains(t, err.Error(), `namespace="other"`)
		assert.Contains(t, err.Error(), `BKECluster "demo-cluster"`)
		assert.Equal(t, "other", nodes[0].Namespace)
	})
}

func TestProcessClusterResourcesErrors(t *testing.T) {
	t.Run("requires at least one node", func(t *testing.T) {
		resources := &ClusterResources{
			BKECluster: &configv1beta1.BKECluster{
				ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "bke-demo"},
				Spec: configv1beta1.BKEClusterSpec{
					ClusterConfig: minimalValidClusterConfig(),
				},
			},
		}
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(securityinit.Init, func() error { return nil })

		err := processClusterResources(resources)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "at least one BKENode")
	})

	t.Run("invalid cluster config", func(t *testing.T) {
		resources := &ClusterResources{
			BKECluster: &configv1beta1.BKECluster{
				Spec: configv1beta1.BKEClusterSpec{
					ClusterConfig: &configv1beta1.BKEConfig{},
				},
			},
			BKENodes: []configv1beta1.BKENode{
				{Spec: configv1beta1.BKENodeSpec{IP: "10.0.0.1", Hostname: "n1"}},
			},
		}
		err := processClusterResources(resources)
		assert.Error(t, err)
	})
}

func TestClusterDeployLoadFailure(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(NewClusterResourcesFromFiles, func(string, string) (*ClusterResources, error) {
		return nil, fmt.Errorf("load failed")
	})

	op := &Options{File: "cluster.yaml", NodesFile: "nodes.yaml"}
	op.Cluster()
}

func TestNewClusterResourcesFromFilesClusterError(t *testing.T) {
	_, err := NewClusterResourcesFromFiles(path.Join(t.TempDir(), "missing.yaml"), "nodes.yaml")
	assert.Error(t, err)
}

func minimalValidClusterConfig() *configv1beta1.BKEConfig {
	return &configv1beta1.BKEConfig{
		CustomExtra: map[string]string{
			"containerd": "containerd-1.7.0-linux-amd64.tar.gz",
		},
		Cluster: configv1beta1.Cluster{
			ImageRepo: configv1beta1.Repo{
				Domain: testRegistryDomain,
				Ip:     testIP,
				Port:   testRegistryPort,
				Prefix: "k8s",
			},
			HTTPRepo: configv1beta1.Repo{
				Domain: testYumDomain,
				Ip:     testYumIP,
				Port:   testYumPort,
			},
			ChartRepo: configv1beta1.Repo{
				Domain: testChartDomain,
				Ip:     testChartIP,
				Port:   testChartPort,
				Prefix: "charts",
			},
			NTPServer: "pool.ntp.org",
		},
	}
}
