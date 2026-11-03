/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package validation

import (
	"strings"
	"testing"

	"bou.ke/monkey"

	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/addon"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
)

func TestValidateChartRepo(t *testing.T) {
	addons := addon.Addons{v1beta1.Product{Type: addon.ChartAddon}}

	// ip异常
	chartRepo := v1beta1.Repo{
		Domain: "",
		Ip:     "172.158.0.x",
		Port:   "8080",
		Prefix: "chart",
	}
	err := ValidateChartRepo(chartRepo, addons)
	assert.Error(t, err)

	// 无ip domain
	chartRepo = v1beta1.Repo{
		Domain: "",
		Ip:     "",
		Port:   "8080",
		Prefix: "chart",
	}
	err = ValidateChartRepo(chartRepo, addons)
	assert.Error(t, err)

	// 有ip domain
	chartRepo = v1beta1.Repo{
		Domain: "chart.domain",
		Ip:     "172.158.0.1",
		Port:   "8080",
		Prefix: "chart",
	}
	monkey.Patch(checkReachable, func(addr string) bool {
		return true
	})
	defer monkey.UnpatchAll()
	err = ValidateChartRepo(chartRepo, addons)
	assert.NoError(t, err)

	// 无ip domain 无addon
	chartRepo = v1beta1.Repo{}
	addons = addon.Addons{}
	err = ValidateChartRepo(chartRepo, addons)
	assert.NoError(t, err)

	// 无ip domain 有yaml addon
	chartRepo = v1beta1.Repo{}
	addons = addon.Addons{v1beta1.Product{Type: addon.YamlAddon}}
	err = ValidateChartRepo(chartRepo, addons)
	assert.NoError(t, err)

	// 无ip domain 有chart addon
	chartRepo = v1beta1.Repo{}
	addons = addon.Addons{v1beta1.Product{Type: addon.ChartAddon}}
	err = ValidateChartRepo(chartRepo, addons)
	assert.Error(t, err)
}

func TestResolveReachableHTTPRepoBaseURL(t *testing.T) {
	monkey.Patch(checkReachable, func(addr string) bool {
		return strings.HasPrefix(addr, "http.bocloud.k8s:") || strings.HasPrefix(addr, "reachable.example.com:")
	})
	defer monkey.UnpatchAll()

	url := ResolveReachableHTTPRepoBaseURL(v1beta1.Repo{
		Domain: "http.bocloud.k8s",
		Ip:     "192.168.1.10",
		Port:   "40080",
	})
	assert.Equal(t, "http://http.bocloud.k8s:40080", url)

	url = ResolveReachableHTTPRepoBaseURL(v1beta1.Repo{
		Domain: "unreachable.example.com",
		Ip:     "192.168.1.10",
		Port:   "40080",
	})
	assert.Equal(t, "http://192.168.1.10:40080", url)

	url = ResolveReachableHTTPRepoBaseURL(v1beta1.Repo{
		Domain: "reachable.example.com",
		Port:   "8080",
	})
	assert.Equal(t, "http://reachable.example.com:8080", url)
}

func TestResolveReachableRepoAddress(t *testing.T) {
	monkey.Patch(checkReachable, func(addr string) bool {
		return true
	})
	defer monkey.UnpatchAll()

	// 无domain
	chartRepo := v1beta1.Repo{
		Domain: "",
		Ip:     "192.168.100.20",
		Port:   "8080",
		Prefix: "chart",
	}
	url, _ := ResolveReachableRepoAddress(chartRepo)
	assert.Contains(t, url, chartRepo.Ip)

	// 无ip
	chartRepo = v1beta1.Repo{
		Domain: "chart.domain",
		Ip:     "",
		Port:   "8080",
		Prefix: "chart",
	}
	url, _ = ResolveReachableRepoAddress(chartRepo)
	assert.Contains(t, url, chartRepo.Domain)
}

func validNode(ip, hostname string, roles ...string) node.Node {
	return node.Node{
		Role:     roles,
		IP:       ip,
		Port:     "22",
		Username: "root",
		Password: "password",
		Hostname: hostname,
	}
}

func TestValidateNodesFields(t *testing.T) {
	valid := node.Nodes{
		v1beta1.Node(validNode("192.168.1.10", "master-1", node.MasterWorkerNodeRole, node.EtcdNodeRole)),
	}
	assert.NoError(t, ValidateNodesFields(valid))

	tests := []struct {
		name  string
		nodes node.Nodes
	}{
		{name: "empty", nodes: nil},
		{name: "no worker", nodes: node.Nodes{v1beta1.Node(validNode("192.168.1.10", "master-1", node.MasterNodeRole, node.EtcdNodeRole))}},
		{name: "duplicate hostname", nodes: node.Nodes{
			v1beta1.Node(validNode("192.168.1.10", "same", node.MasterWorkerNodeRole, node.EtcdNodeRole)),
			v1beta1.Node(validNode("192.168.1.11", "same", node.WorkerNodeRole)),
		}},
		{name: "duplicate ip", nodes: node.Nodes{
			v1beta1.Node(validNode("192.168.1.10", "master-1", node.MasterWorkerNodeRole, node.EtcdNodeRole)),
			v1beta1.Node(validNode("192.168.1.10", "worker-1", node.WorkerNodeRole)),
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Error(t, ValidateNodesFields(tt.nodes))
		})
	}
}

func TestValidateNonStandardNodesFields(t *testing.T) {
	assert.NoError(t, ValidateNonStandardNodesFields(nil))
	assert.NoError(t, ValidateNonStandardNodesFields(node.Nodes{
		v1beta1.Node(validNode("192.168.1.10", "master-1", node.MasterWorkerNodeRole, node.EtcdNodeRole)),
	}))
	assert.Error(t, ValidateNonStandardNodesFields(node.Nodes{
		v1beta1.Node(validNode("192.168.1.10", "master-1", node.MasterNodeRole, node.WorkerNodeRole)),
	}))
	assert.Error(t, ValidateNonStandardNodesFields(node.Nodes{
		v1beta1.Node(validNode("192.168.1.10", "same", node.MasterWorkerNodeRole, node.EtcdNodeRole)),
		v1beta1.Node(validNode("192.168.1.11", "same", node.WorkerNodeRole)),
	}))
	assert.Error(t, ValidateNonStandardNodesFields(node.Nodes{
		v1beta1.Node(validNode("", "master-1", node.MasterWorkerNodeRole, node.EtcdNodeRole)),
	}))
}

func TestValidateBKENodes(t *testing.T) {
	assert.Error(t, ValidateBKENodes(nil))
	assert.NoError(t, ValidateBKENodes([]v1beta1.BKENode{{
		Spec: v1beta1.BKENodeSpec(validNode("192.168.1.10", "master-1", node.MasterWorkerNodeRole, node.EtcdNodeRole)),
	}}))
	assert.NoError(t, ValidateBKENodesNonStandard(nil))
}

func TestValidateBasicClusterFields(t *testing.T) {
	assert.Error(t, ValidateK8sVersion(""))
	assert.Error(t, ValidateK8sVersion("v1.20.0"))
	assert.NoError(t, ValidateK8sVersion("v1.24.0"))

	assert.Error(t, ValidateControlPlaneComponents(v1beta1.ControlPlane{}))
	assert.Error(t, ValidateControlPlaneComponents(v1beta1.ControlPlane{Etcd: &v1beta1.Etcd{}}))
	assert.NoError(t, ValidateControlPlaneComponents(v1beta1.ControlPlane{Etcd: &v1beta1.Etcd{DataDir: "/var/lib/etcd"}}))

	assert.NoError(t, ValidateKubeletComponent(nil))
	assert.Error(t, ValidateKubeletComponent(&v1beta1.Kubelet{}))
	assert.NoError(t, ValidateKubeletComponent(&v1beta1.Kubelet{ManifestsDir: "/etc/kubernetes/manifests"}))

	assert.Error(t, ValidateNetworking(v1beta1.Networking{}))
	assert.Error(t, ValidateNetworking(v1beta1.Networking{PodSubnet: "bad"}))
	assert.Error(t, ValidateNetworking(v1beta1.Networking{PodSubnet: "10.244.0.0/16"}))
	assert.Error(t, ValidateNetworking(v1beta1.Networking{PodSubnet: "10.244.0.0/16", ServiceSubnet: "bad"}))
	assert.Error(t, ValidateNetworking(v1beta1.Networking{PodSubnet: "10.244.0.0/16", ServiceSubnet: "10.96.0.0/12"}))
	assert.NoError(t, ValidateNetworking(v1beta1.Networking{
		PodSubnet:     "10.244.0.0/16",
		ServiceSubnet: "10.96.0.0/12",
		DNSDomain:     "cluster.local",
	}))

	assert.Error(t, ValidateContainerRuntime(v1beta1.ContainerRuntime{CRI: "bad"}))
	assert.Error(t, ValidateContainerRuntime(v1beta1.ContainerRuntime{Runtime: "bad"}))
	assert.NoError(t, ValidateContainerRuntime(v1beta1.ContainerRuntime{CRI: "containerd", Runtime: "runc"}))
}

func TestRepoAndAddonValidation(t *testing.T) {
	assert.Equal(t, "repo.example.com:5000/openfuyao", BuildRepoURL("repo.example.com", "5000", "openfuyao"))
	assert.Equal(t, "repo.example.com/openfuyao", BuildRepoURL("repo.example.com", "", "openfuyao"))
	assert.Equal(t, "http://", BuildHTTPRepoBaseURL("", ""))
	assert.Equal(t, "http://:8080", BuildHTTPRepoBaseURL("", "8080"))
	assert.Equal(t, "http://repo.example.com", BuildHTTPRepoBaseURL("repo.example.com", ""))
	assert.Equal(t, "http://repo.example.com:8080", BuildHTTPRepoBaseURL("repo.example.com", "8080"))

	assert.Error(t, ValidateRepo(v1beta1.Repo{}))
	assert.Error(t, ValidateRepo(v1beta1.Repo{Domain: "Bad_Domain"}))
	assert.Error(t, ValidateRepo(v1beta1.Repo{Ip: "bad-ip"}))
	assert.Error(t, ValidateRepo(v1beta1.Repo{Ip: "192.168.1.10", Port: "bad"}))
	assert.Error(t, ValidateRepo(v1beta1.Repo{Ip: "192.168.1.10", Port: "0"}))
	assert.NoError(t, ValidateRepo(v1beta1.Repo{Ip: "192.168.1.10", Port: "5000"}))

	assert.False(t, IsContainChartAddon(nil))
	assert.True(t, IsContainChartAddon(addon.Addons{{Type: addon.ChartAddon}}))
	assert.NoError(t, ValidateAddons(nil))
	assert.Error(t, ValidateAddons(addon.Addons{{Name: "a", Version: "v1"}, {Name: "a", Version: "v1"}}))
	assert.Error(t, ValidateAddons(addon.Addons{{Name: "a", Type: "bad"}}))
	assert.NoError(t, ValidateAddons(addon.Addons{{Name: "a", Type: addon.YamlAddon}}))
}

func TestCustomExtraValidation(t *testing.T) {
	assert.Error(t, ValidateCustomExtra(nil))
	assert.Error(t, ValidateCustomExtra(map[string]string{"other": "value"}))
	assert.Error(t, ValidateCustomExtra(map[string]string{"containerd": "bad.tar.gz"}))
	assert.NoError(t, ValidateCustomExtra(map[string]string{
		"containerd": "containerd-1.7.0-linux-amd64.tar.gz",
	}))
}

func TestImageRepoAddress(t *testing.T) {
	assert.Equal(t, "repo.example.com:5000", GetImageRepoAddress(v1beta1.Repo{Domain: "repo.example.com", Port: "5000"}))
	assert.Equal(t, "repo.example.com", GetImageRepoAddress(v1beta1.Repo{Domain: "repo.example.com", Port: "443"}))
	assert.Equal(t, "192.168.1.10:5000", GetImageRepoAddress(v1beta1.Repo{Ip: "192.168.1.10", Port: "5000"}))
	assert.Equal(t, "192.168.1.10", GetImageRepoAddress(v1beta1.Repo{Ip: "192.168.1.10", Port: "443"}))
	assert.Equal(t, "repo.example.com:5000", GetImageRepoAddress(v1beta1.Repo{Domain: "repo.example.com", Ip: "192.168.1.10", Port: "5000"}))
	assert.Equal(t, "repo.example.com", GetImageRepoAddress(v1beta1.Repo{Domain: "repo.example.com"}))
	assert.Equal(t, "192.168.1.10", GetImageRepoAddress(v1beta1.Repo{Ip: "192.168.1.10"}))
}

func TestValidateLoadBalanceType(t *testing.T) {
	assert.NoError(t, ValidateLoadBalanceType(""))
	assert.NoError(t, ValidateLoadBalanceType("ha"))
	assert.NoError(t, ValidateLoadBalanceType("nginx"))
	assert.Error(t, ValidateLoadBalanceType("slb"))
}
