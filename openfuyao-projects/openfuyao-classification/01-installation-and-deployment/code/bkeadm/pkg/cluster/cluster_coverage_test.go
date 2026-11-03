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
	"errors"
	"fmt"
	"os"
	"path"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	configv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/security"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	k8sexec "gopkg.openfuyao.cn/bkeadm/pkg/executor/k8s"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/securityinit"
	"gopkg.openfuyao.cn/bkeadm/utils"
)

func TestLoadBKEClusterFromFile(t *testing.T) {
	dir := t.TempDir()
	validFile := path.Join(dir, "cluster.yaml")
	validYAML := fmt.Sprintf(`apiVersion: cluster.bocloud.com/v1beta1
kind: BKECluster
metadata:
  name: demo
  namespace: bke-demo
spec:
  clusterConfig:
    cluster:
      imageRepo:
        domain: %s
        ip: %s
        port: %s
        prefix: k8s
      httpRepo:
        domain: %s
        ip: %s
        port: %s
      chartRepo:
        domain: %s
        ip: %s
        port: %s
        prefix: charts
      ntpServer: pool.ntp.org
`, testRegistryDomain, testRegistryIP, testRegistryPort, testYumDomain, testYumIP, testYumPort, testChartDomain, testChartIP, testChartPort)
	require.NoError(t, os.WriteFile(validFile, []byte(validYAML), utils.DefaultFilePermission))

	cluster, err := loadBKEClusterFromFile(validFile)
	require.NoError(t, err)
	assert.Equal(t, "demo", cluster.Name)

	emptyCfgFile := path.Join(dir, "empty.yaml")
	require.NoError(t, os.WriteFile(emptyCfgFile, []byte(`apiVersion: v1
kind: BKECluster
metadata:
  name: demo
spec: {}
`), utils.DefaultFilePermission))
	_, err = loadBKEClusterFromFile(emptyCfgFile)
	assert.Error(t, err)

	_, err = loadBKEClusterFromFile(path.Join(dir, "missing.yaml"))
	assert.Error(t, err)
}

func TestRemoveInvalidArgument(t *testing.T) {
	op := &Options{Args: []string{"invalid-format"}}
	op.Remove()
}

func TestLogInvalidArgument(t *testing.T) {
	mockK8s := &k8sexec.MockK8sClient{}
	global.K8s = mockK8s
	t.Cleanup(func() { global.K8s = nil })

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*k8sexec.MockK8sClient).WatchEventByAnnotation, func(*k8sexec.MockK8sClient, string) {})

	op := &Options{Args: []string{"invalid-format"}}
	op.Log()
}

func TestScaleWithClusterResources(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	resources := &ClusterResources{
		BKECluster: &configv1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "bke-demo"},
		},
		BKENodes: []configv1beta1.BKENode{
			{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}},
		},
	}
	patches.ApplyFunc(NewClusterResourcesFromFiles, func(string, string) (*ClusterResources, error) {
		return resources, nil
	})
	patches.ApplyFunc(marshalAndWriteClusterYAML, func(*configv1beta1.BKECluster) (string, error) {
		return t.TempDir() + "/cluster.yaml", nil
	})
	patches.ApplyFunc(marshalAndWriteNodeYAMLs, func(string, string, []configv1beta1.BKENode) ([]string, error) {
		return []string{t.TempDir() + "/node.yaml"}, nil
	})

	mockK8s := &k8sexec.MockK8sClient{}
	global.K8s = mockK8s
	t.Cleanup(func() { global.K8s = nil })
	patches.ApplyFunc((*k8sexec.MockK8sClient).PatchYaml, func(*k8sexec.MockK8sClient, string, map[string]string) error {
		return nil
	})
	patches.ApplyFunc((*k8sexec.MockK8sClient).WatchEventByAnnotation, func(*k8sexec.MockK8sClient, string) {})

	op := &Options{File: "cluster.yaml", NodesFile: "nodes.yaml"}
	op.Scale()
}

func TestScaleNodeWriteFails(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(NewClusterResourcesFromFiles, func(string, string) (*ClusterResources, error) {
		return &ClusterResources{
			BKECluster: &configv1beta1.BKECluster{
				ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "bke-demo"},
			},
			BKENodes: []configv1beta1.BKENode{{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}}},
		}, nil
	})
	patches.ApplyFunc(marshalAndWriteClusterYAML, func(*configv1beta1.BKECluster) (string, error) {
		return "/tmp/cluster.yaml", nil
	})
	patches.ApplyFunc(marshalAndWriteNodeYAMLs, func(string, string, []configv1beta1.BKENode) ([]string, error) {
		return nil, errors.New("write nodes failed")
	})

	op := &Options{File: "cluster.yaml", NodesFile: "nodes.yaml"}
	op.Scale()
}

func TestProcessClusterResourcesSuccess(t *testing.T) {
	resources := &ClusterResources{
		BKECluster: &configv1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "bke-demo"},
			Spec: configv1beta1.BKEClusterSpec{
				ClusterConfig: minimalValidClusterConfig(),
			},
		},
		BKENodes: []configv1beta1.BKENode{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "node-1"},
				Spec: configv1beta1.BKENodeSpec{
					IP:       "10.0.0.1",
					Hostname: "node-1",
					Port:     "22",
					Username: "root",
					Password: "plain-pass",
					Role:     []string{"master/node", "etcd"},
				},
			},
		},
	}

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(securityinit.Init, func() error { return nil })
	patches.ApplyFunc(security.AesDecrypt, func(string) (string, error) {
		return "", errors.New("not encrypted")
	})
	patches.ApplyFunc(security.AesEncrypt, func(string) (string, error) {
		return "encrypted-pass", nil
	})

	err := processClusterResources(resources)
	require.NoError(t, err)
	assert.Equal(t, "encrypted-pass", resources.BKENodes[0].Spec.Password)
	assert.Equal(t, "demo", resources.BKENodes[0].Labels["cluster.x-k8s.io/cluster-name"])
	assert.Equal(t, "bke-demo", resources.BKENodes[0].Namespace)
}

func TestNewClusterResourcesFromFilesNodesMissing(t *testing.T) {
	dir := t.TempDir()
	clusterFile := path.Join(dir, "cluster.yaml")
	clusterYAML := fmt.Sprintf(`apiVersion: cluster.bocloud.com/v1beta1
kind: BKECluster
metadata:
  name: demo
  namespace: bke-demo
spec:
  clusterConfig:
    cluster:
      imageRepo:
        domain: %s
        ip: %s
        port: %s
        prefix: k8s
      httpRepo:
        domain: %s
        ip: %s
        port: %s
      chartRepo:
        domain: %s
        ip: %s
        port: %s
        prefix: charts
      ntpServer: pool.ntp.org
`, testRegistryDomain, testRegistryIP, testRegistryPort, testYumDomain, testYumIP, testYumPort, testChartDomain, testChartIP, testChartPort)
	require.NoError(t, os.WriteFile(clusterFile, []byte(clusterYAML), utils.DefaultFilePermission))

	_, err := NewClusterResourcesFromFiles(clusterFile, path.Join(dir, "missing-nodes.yaml"))
	assert.Error(t, err)
}

func TestLoadClusterConfigEmpty(t *testing.T) {
	dir := t.TempDir()
	file := path.Join(dir, "empty.yaml")
	require.NoError(t, os.WriteFile(file, []byte(`apiVersion: v1
kind: BKECluster
metadata:
  name: demo
spec: {}
`), utils.DefaultFilePermission))

	_, err := loadClusterConfig(file)
	assert.Error(t, err)
}

func TestClusterDeploySuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	resources := &ClusterResources{
		BKECluster: &configv1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "demo"},
			Spec: configv1beta1.BKEClusterSpec{
				ClusterConfig: minimalValidClusterConfig(),
			},
		},
		BKENodes: []configv1beta1.BKENode{
			{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}},
		},
	}
	patches.ApplyFunc(NewClusterResourcesFromFiles, func(string, string) (*ClusterResources, error) {
		return resources, nil
	})
	patches.ApplyFunc(marshalAndWriteClusterYAML, func(*configv1beta1.BKECluster) (string, error) {
		return t.TempDir() + "/cluster.yaml", nil
	})
	patches.ApplyFunc(marshalAndWriteNodeYAMLs, func(string, string, []configv1beta1.BKENode) ([]string, error) {
		return []string{t.TempDir() + "/node.yaml"}, nil
	})

	mockK8s := &k8sexec.MockK8sClient{}
	global.K8s = mockK8s
	t.Cleanup(func() { global.K8s = nil })
	patches.ApplyFunc((*k8sexec.MockK8sClient).CreateNamespace, func(*k8sexec.MockK8sClient, *corev1.Namespace) error {
		return nil
	})
	patches.ApplyFunc((*k8sexec.MockK8sClient).InstallYaml, func(*k8sexec.MockK8sClient, string, map[string]string, string) error {
		return nil
	})
	patches.ApplyFunc((*k8sexec.MockK8sClient).WatchEventByAnnotation, func(*k8sexec.MockK8sClient, string) {})

	op := &Options{
		File:        "cluster.yaml",
		NodesFile:   "nodes.yaml",
		NtpServer:   "ntp.example.com",
		ImmutableOS: utils.ImmutableOSAnnotationValueKubeOS,
	}
	op.Cluster()
	assert.Contains(t, resources.BKECluster.Annotations, utils.ImmutableOSAnnotation)
}
