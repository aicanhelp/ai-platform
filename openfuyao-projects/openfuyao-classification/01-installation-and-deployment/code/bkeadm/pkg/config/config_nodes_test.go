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

package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	configv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/security"

	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/utils"
)

func TestNewMasterEtcdNodeDef(t *testing.T) {
	def := newMasterEtcdNodeDef("master-1", "10.0.0.1", "root", "secret")
	assert.Equal(t, "master-1", def.Hostname)
	assert.Equal(t, "10.0.0.1", def.IP)
	assert.Equal(t, "22", def.Port)
	assert.Contains(t, def.Role, "master/node")
	assert.Contains(t, def.Role, "etcd")
}

func TestCreateBKENodes(t *testing.T) {
	nodeDefs := []nodeDef{
		newMasterEtcdNodeDef("m1", "127.0.0.1", "root", "pass"),
		{
			Hostname: "n1",
			IP:       "127.0.0.2",
			Username: "root",
			Password: "pass",
			Port:     "22",
			Role:     []string{"node"},
		},
	}
	nodes := createBKENodes("bke-cluster", "bke-cluster", nodeDefs)
	require.Len(t, nodes, 2)
	assert.Equal(t, "bke-cluster-m1", nodes[0].Name)
	assert.Equal(t, "bke-cluster", nodes[0].Namespace)
	assert.Equal(t, "bke-cluster", nodes[0].Labels["cluster.x-k8s.io/cluster-name"])
	assert.Equal(t, "n1", nodes[1].Spec.Hostname)
}

func TestInitBaseConfig(t *testing.T) {
	opts := &Options{AgentHealthPort: "8088"}
	cfg := opts.initBaseConfig(
		map[string]string{"key": "value"},
		confv1beta1.Repo{Domain: "img.local", Ip: "10.0.0.1", Port: "443"},
		confv1beta1.Repo{Domain: "yum.local", Ip: "10.0.0.2", Port: "8080"},
		confv1beta1.Repo{Domain: "chart.local", Ip: "10.0.0.3", Port: "8443"},
		"ntp.local",
	)
	require.NotNil(t, cfg)
	assert.Equal(t, "value", cfg.CustomExtra["key"])
	assert.Equal(t, "img.local", cfg.Cluster.ImageRepo.Domain)
	assert.Equal(t, "8088", cfg.Cluster.AgentHealthPort)
}

func TestGenerateControllerParams(t *testing.T) {
	orig := global.CustomExtra["otherRepo"]
	global.CustomExtra["otherRepo"] = ""
	defer func() { global.CustomExtra["otherRepo"] = orig }()

	opts := &Options{Domain: "registry.example.com", ImageRepoPort: "5000"}
	sandbox, offline := opts.generateControllerParams()
	assert.Contains(t, sandbox, "registry.example.com:5000")
	assert.Equal(t, "true", offline)
}

func TestApplyFuyaoCommonConfig(t *testing.T) {
	opts := &Options{NtpServer: "ntp.test:123", AgentHealthPort: "9091"}
	cfg := &confv1beta1.BKEConfig{Addons: []confv1beta1.Product{{Name: "coredns"}}}

	opts.applyFuyaoCommonConfig(cfg, "sandbox-img", "false")

	require.GreaterOrEqual(t, len(cfg.Addons), 3)
	var hasClusterAPI, hasSystemController bool
	for _, addon := range cfg.Addons {
		if addon.Name == "cluster-api" {
			hasClusterAPI = true
			assert.Equal(t, "sandbox-img", addon.Param["sandbox"])
			assert.Equal(t, "9091", addon.Param["healthPort"])
		}
		if addon.Name == "openfuyao-system-controller" {
			hasSystemController = true
		}
	}
	assert.True(t, hasClusterAPI)
	assert.True(t, hasSystemController)
}

func TestApplyFuyaoPortalAndAllInOneConfig(t *testing.T) {
	opts := &Options{}
	cfg := &confv1beta1.BKEConfig{}

	opts.applyFuyaoPortalConfig(cfg, "sb", "true")
	assert.NotEmpty(t, cfg.Addons)

	cfg = &confv1beta1.BKEConfig{}
	opts.applyFuyaoAllInOneConfig(cfg, "sb", "true")
	assert.NotEmpty(t, cfg.Addons)
}

func TestProductAppliesAddons(t *testing.T) {
	opts := &Options{Product: "fuyao-allinone", Domain: "d.local", ImageRepoPort: "443"}
	cfg := &confv1beta1.BKEConfig{}
	result := opts.product(cfg, confv1beta1.Repo{})
	require.NotNil(t, result)
	assert.NotEmpty(t, result.Addons)
}

func TestWriteClusterAndNodesConfig(t *testing.T) {
	dir := t.TempDir()
	cluster := configv1beta1.BKECluster{}
	cluster.Name = "bke-cluster"
	cluster.Namespace = "bke-cluster"
	cluster.APIVersion = "bke.bocloud.com/v1beta1"
	cluster.Kind = "BKECluster"

	nodes := createBKENodes(cluster.Name, cluster.Namespace, []nodeDef{
		newMasterEtcdNodeDef("m1", "127.0.0.1", "root", "******"),
	})

	writeClusterAndNodesConfig(dir, "1master", cluster, nodes, true)

	clusterFile := filepath.Join(dir, "1master-cluster.yaml")
	nodeFile := filepath.Join(dir, "1master-node.yaml")
	assert.FileExists(t, clusterFile)
	assert.FileExists(t, nodeFile)

	clusterBytes, err := os.ReadFile(clusterFile)
	require.NoError(t, err)
	assert.Contains(t, string(clusterBytes), "BKECluster")

	nodeBytes, err := os.ReadFile(nodeFile)
	require.NoError(t, err)
	assert.Contains(t, string(nodeBytes), "BKENode")
}

func TestMasterLayoutGenerators(t *testing.T) {
	dir := t.TempDir()
	cluster := configv1beta1.BKECluster{}
	cluster.Name = "bke-cluster"
	cluster.Namespace = "bke-cluster"
	cfg := &confv1beta1.BKEConfig{
		Addons: []confv1beta1.Product{{Name: "coredns"}},
	}

	master1(dir, cluster, cfg)
	assert.FileExists(t, filepath.Join(dir, "1master-cluster.yaml"))
	assert.FileExists(t, filepath.Join(dir, "1master-node.yaml"))

	master1node1(dir, cluster, cfg)
	assert.FileExists(t, filepath.Join(dir, "1master1node-cluster.yaml"))
	assert.FileExists(t, filepath.Join(dir, "1master1node-nodes.yaml"))

	master3(dir, cluster, cfg)
	assert.FileExists(t, filepath.Join(dir, "3master-cluster.yaml"))
	assert.FileExists(t, filepath.Join(dir, "3master-nodes.yaml"))
}

func TestLoadBKENodes(t *testing.T) {
	dir := t.TempDir()
	nodesFile := filepath.Join(dir, "nodes.yaml")
	content := `apiVersion: bke.bocloud.com/v1beta1
kind: BKENode
metadata:
  name: bke-cluster-m1
spec:
  hostname: m1
  ip: 127.0.0.1
---
apiVersion: bke.bocloud.com/v1beta1
kind: BKENode
metadata:
  name: bke-cluster-n1
spec:
  hostname: n1
  ip: 127.0.0.2
`
	require.NoError(t, os.WriteFile(nodesFile, []byte(content), utils.DefaultFilePermission))

	opts := &Options{}
	nodes, err := opts.loadBKENodes(nodesFile)
	require.NoError(t, err)
	require.Len(t, nodes, 2)
	assert.Equal(t, "m1", nodes[0].Spec.Hostname)
}

func TestProcessNodesFileEncryptDecrypt(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)

	nodesFile := "nodes.yaml"
	require.NoError(t, os.WriteFile(nodesFile, []byte(`apiVersion: bke.bocloud.com/v1beta1
kind: BKENode
metadata:
  name: bke-cluster-m1
spec:
  hostname: m1
  password: plain-password
`), utils.DefaultFilePermission))

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(security.AesDecrypt, func(s string) (string, error) {
		return "", os.ErrInvalid
	})
	patches.ApplyFunc(security.AesEncrypt, func(s string) (string, error) {
		return "enc-" + s, nil
	})

	opts := &Options{}
	require.NoError(t, opts.processNodesFile(nodesFile, true))

	out, err := os.ReadFile("nodes-encrypt.yaml")
	require.NoError(t, err)
	assert.Contains(t, string(out), "enc-plain-password")
}

func TestEncryptAndDecryptNodesFileEntrypoints(t *testing.T) {
	dir := t.TempDir()
	nodesFile := filepath.Join(dir, "nodes.yaml")
	require.NoError(t, os.WriteFile(nodesFile, []byte(`apiVersion: bke.bocloud.com/v1beta1
kind: BKENode
metadata:
  name: node-1
spec:
  password: plain
`), utils.DefaultFilePermission))

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(security.AesDecrypt, func(s string) (string, error) {
		if strings.HasPrefix(s, "enc-") {
			return strings.TrimPrefix(s, "enc-"), nil
		}
		return "", os.ErrInvalid
	})
	patches.ApplyFunc(security.AesEncrypt, func(s string) (string, error) {
		return "enc-" + s, nil
	})

	opts := &Options{}
	require.NoError(t, opts.EncryptNodesFile(nodesFile))
	require.NoError(t, opts.DecryptNodesFile(nodesFile))
}
