/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package initialize

import (
	"encoding/json"
	"net"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize/versions"
)

func TestSetDefaultBKEConfig(t *testing.T) {
	cfg := &BkeConfig{}
	SetDefaultBKEConfig(cfg)

	assert.Equal(t, cfg.Cluster.CertificatesDir, DefaultCertificatesDir)
	assert.Equal(t, versions.KubernetesVersion(), cfg.Cluster.KubernetesVersion)
	assert.Equal(t, versions.EtcdVersion(), cfg.Cluster.EtcdVersion)
	assert.Equal(t, versions.ContainerdVersion(), cfg.Cluster.ContainerdVersion)
	assert.Equal(t, versions.OpenFuyaoVersion(), cfg.Cluster.OpenFuyaoVersion)
	assert.Equal(t, cfg.Cluster.HTTPRepo.Domain, DefaultYumRepo)
	assert.Equal(t, cfg.Cluster.HTTPRepo.Port, DefaultYumRepoPort)
	assert.Equal(t, cfg.Cluster.ImageRepo.Domain, DefaultImageRepo)
	assert.Equal(t, cfg.Cluster.ImageRepo.Port, DefaultImageRepoPort)
	assert.Equal(t, cfg.Cluster.ImageRepo.Prefix, ImageRegistryKubernetes)
	assert.Equal(t, cfg.Cluster.Networking.DNSDomain, DefaultServiceDNSDomain)
	assert.Equal(t, cfg.Cluster.Networking.ServiceSubnet, DefaultServicesSubnet)
	assert.Equal(t, cfg.Cluster.Networking.PodSubnet, DefaultPodSubnet)
	assert.Equal(t, cfg.Cluster.Etcd.DataDir, DefaultEtcdDataDir)
	assert.Equal(t, cfg.Cluster.APIServer.Port, int32(DefaultAPIBindPort))
	assert.EqualValues(t, cfg.Cluster.APIServer.ExtraArgs, map[string]string{"authorization-mode": "Node,RBAC"})
	assert.Equal(t, cfg.Cluster.Kubelet.ManifestsDir, DefaultManifestsDir)
	assert.Equal(t, cfg.Cluster.ContainerRuntime.CRI, CRIContainerd)
	assert.Equal(t, cfg.Cluster.ContainerRuntime.Runtime, DefaultRuntime)
}

func TestSetDefaultBKEConfigPreservesExplicitVersions(t *testing.T) {
	cfg := &BkeConfig{}
	cfg.Cluster.KubernetesVersion = "v1.30.0"
	cfg.Cluster.EtcdVersion = "v3.5.0"
	cfg.Cluster.ContainerdVersion = "v1.7.0"
	cfg.Cluster.OpenFuyaoVersion = "v1.0.0"

	SetDefaultBKEConfig(cfg)

	assert.Equal(t, "v1.30.0", cfg.Cluster.KubernetesVersion)
	assert.Equal(t, "v3.5.0", cfg.Cluster.EtcdVersion)
	assert.Equal(t, "v1.7.0", cfg.Cluster.ContainerdVersion)
	assert.Equal(t, "v1.0.0", cfg.Cluster.OpenFuyaoVersion)
}

// TestSetDefaultBKENodes tests the default node configuration
func TestSetDefaultBKENodes(t *testing.T) {
	nodes := GetDefaultNodes()
	assert.NotEmpty(t, nodes)
	// Verify first node has default values
	assert.Equal(t, nodes[0].Port, DefaultNodeSSHPort)
	assert.Equal(t, nodes[0].Username, DefaultNodeUserRoot)
}

func TestExportDefaults(t *testing.T) {
	cfg := &BkeConfig{}
	SetDefaultBKEConfig(cfg)

	marshal, err := json.Marshal(cfg)
	if err != nil {
		return
	}
	err = os.WriteFile("bkeconfig-default.json", marshal, DefaultFileMode)
	if err != nil {
		t.Error(err)
	}
	err = os.Remove("bkeconfig-default.json")
	if err != nil {
		t.Error(err)
	}
}

func TestGetDefaultEtcdK8sVersionImageMap(t *testing.T) {
	imageMap := GetDefaultEtcdK8sVersionImageMap()
	assert.NotNil(t, imageMap)
	assert.NotEmpty(t, imageMap)
}

func TestGetDefaultPauseK8sVersionImageMap(t *testing.T) {
	imageMap := GetDefaultPauseK8sVersionImageMap()
	assert.NotNil(t, imageMap)
	assert.NotEmpty(t, imageMap)
}

func TestGetIndexedIP(t *testing.T) {
	_, ipnet, _ := net.ParseCIDR("10.96.0.0/12")
	ip, err := getIndexedIP(ipnet, 10)
	assert.NoError(t, err)
	assert.NotNil(t, ip)
}

func TestGetDefaultBKENodes(t *testing.T) {
	nodes := GetDefaultBKENodes("test-cluster", "default")
	assert.NotEmpty(t, nodes)
	assert.Equal(t, 6, len(nodes))
}
