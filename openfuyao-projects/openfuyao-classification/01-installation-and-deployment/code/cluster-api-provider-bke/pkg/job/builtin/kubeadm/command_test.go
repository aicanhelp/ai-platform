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

package kubeadm

import (
	"context"
	"strings"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	bkevalidte "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/validation"
	certPlugin "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/kubeadm/certs"
	envPlugin "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/kubeadm/env"
	manifestsPlugin "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/kubeadm/manifests"
	nginxPlugin "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/kubeadm/nginx"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/cluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/mfutil"
)

func TestGenerateProviderID(t *testing.T) {
	result := generateProviderID("test-cluster", "192.168.1.10")
	assert.Contains(t, result, "bke://test-cluster/")
}

func TestGetKubeletCgroupDriver(t *testing.T) {
	boot := &mfutil.BootScope{
		BkeConfig: &bkev1beta1.BKEConfig{
			Cluster: bkev1beta1.Cluster{
				ContainerRuntime: bkev1beta1.ContainerRuntime{
					Param: map[string]string{"cgroupDriver": "systemd"},
				},
			},
		},
	}
	result := getKubeletCgroupDriver(boot)
	assert.Equal(t, "systemd", result)
}

func TestGetKubeletCgroupDriverDefault(t *testing.T) {
	boot := &mfutil.BootScope{
		BkeConfig: &bkev1beta1.BKEConfig{
			Cluster: bkev1beta1.Cluster{
				ContainerRuntime: bkev1beta1.ContainerRuntime{},
			},
		},
	}
	result := getKubeletCgroupDriver(boot)
	assert.NotEmpty(t, result)
}

func TestGetKubeletDataRootDir(t *testing.T) {
	boot := &mfutil.BootScope{
		BkeConfig: &bkev1beta1.BKEConfig{
			Cluster: bkev1beta1.Cluster{
				Kubelet: &bkev1beta1.Kubelet{
					ControlPlaneComponent: bkev1beta1.ControlPlaneComponent{
						ExtraVolumes: []bkev1beta1.HostPathMount{
							{Name: "kubelet-root-dir", HostPath: "/var/lib/kubelet"},
						},
					},
				},
			},
		},
	}
	result := getKubeletDataRootDir(boot)
	assert.Equal(t, "/var/lib/kubelet", result)
}

func TestGetKubeletDataRootDirEmpty(t *testing.T) {
	boot := &mfutil.BootScope{
		BkeConfig: &bkev1beta1.BKEConfig{
			Cluster: bkev1beta1.Cluster{
				Kubelet: &bkev1beta1.Kubelet{},
			},
		},
	}
	result := getKubeletDataRootDir(boot)
	assert.NotEmpty(t, result)
}

func TestProcessVolumeNotExists(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	var extraVolumes []string
	processVolume("/host/path", "/mount/path", "test-volume", &extraVolumes)
	assert.Len(t, extraVolumes, 1)
}

func TestProcessVolumeKubeletRootDir(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	var extraVolumes []string
	processVolume("/var/lib/kubelet", "/var/lib/kubelet", "kubelet-root-dir", &extraVolumes)
	assert.Len(t, extraVolumes, 0)
}

func TestBuildKubeletCommand(t *testing.T) {
	k := &KubeadmPlugin{
		boot: &mfutil.BootScope{
			BkeConfig: &bkev1beta1.BKEConfig{
				Cluster: bkev1beta1.Cluster{
					KubernetesVersion: "v1.28.0",
					Networking:        bkev1beta1.Networking{ServiceSubnet: "10.96.0.0/12", DNSDomain: "cluster.local"},
					CertificatesDir:   "/etc/kubernetes/pki",
					Kubelet:           &bkev1beta1.Kubelet{ManifestsDir: "/etc/kubernetes/manifests"},
				},
			},
			HostIP:   "192.168.1.10",
			HostName: "test-node",
		},
		clusterName: "test-cluster",
	}

	cfg := bkeinit.BkeConfig(*k.boot.BkeConfig)
	command, err := k.buildKubeletCommand(cfg, "http://test.com/kubelet")
	assert.NoError(t, err)
	assert.NotEmpty(t, command)
}

func newNginxKubeadmPlugin(lbType string) *KubeadmPlugin {
	return &KubeadmPlugin{
		boot: &mfutil.BootScope{
			BkeConfig: &bkev1beta1.BKEConfig{
				Cluster: bkev1beta1.Cluster{LoadBalanceType: lbType},
			},
		},
		clusterName:    "test-cluster",
		GableNameSpace: "default",
	}
}

func TestJoinWorkerNginxCommand_NonNginxSkips(t *testing.T) {
	k := newNginxKubeadmPlugin("ha")
	assert.NoError(t, k.joinWorkerNginxCommand())
}

func TestJoinWorkerNginxCommand_GetNodesError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(plugin.GetNodesDataFromNs, func(_, _ string) (bkenode.Nodes, error) {
		return nil, assert.AnError
	})
	err := newNginxKubeadmPlugin("nginx").joinWorkerNginxCommand()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to get nodes data")
}

func TestJoinWorkerNginxCommand_NoMasterNodes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(plugin.GetNodesDataFromNs, func(_, _ string) (bkenode.Nodes, error) {
		return bkenode.Nodes{{IP: "10.0.0.5", Role: []string{bkenode.WorkerNodeRole}}}, nil
	})
	err := newNginxKubeadmPlugin("nginx").joinWorkerNginxCommand()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no master nodes found")
}

func TestJoinWorkerNginxCommand_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(plugin.GetNodesDataFromNs, func(_, _ string) (bkenode.Nodes, error) {
		return bkenode.Nodes{{IP: "10.0.0.1", Role: []string{bkenode.MasterNodeRole}}}, nil
	})
	patches.ApplyFunc((*nginxPlugin.Plugin).Execute, func(_ *nginxPlugin.Plugin, _ context.Context, _ []string) ([]string, error) {
		return nil, nil
	})
	k := newNginxKubeadmPlugin("nginx")
	k.boot.BkeConfig.Cluster.NginxProxyVersion = "1.29.0"
	assert.NoError(t, k.joinWorkerNginxCommand())
}

func TestReconfigureNginx_NonNginxSkips(t *testing.T) {
	assert.NoError(t, newNginxKubeadmPlugin("ha").reconfigureNginx())
}

func TestGetKubeletExtraArgsError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(cluster.GetNodesData, func(namespace, clusterName string) ([]bkev1beta1.Node, error) {
		return nil, assert.AnError
	})

	boot := &mfutil.BootScope{
		ClusterNamespace: "default",
		ClusterName:      "test",
		BkeConfig:        &bkev1beta1.BKEConfig{},
	}
	result := getKubeletExtraArgs(boot)
	assert.Empty(t, result)
}

func TestGetKubeletExtraVolumesError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(cluster.GetNodesData, func(namespace, clusterName string) ([]bkev1beta1.Node, error) {
		return nil, assert.AnError
	})

	boot := &mfutil.BootScope{
		ClusterNamespace: "default",
		ClusterName:      "test",
		BkeConfig:        &bkev1beta1.BKEConfig{},
	}
	result := getKubeletExtraVolumes(boot)
	assert.Empty(t, result)
}

func TestGetKubeletDataRootDirFromCurrentNode(t *testing.T) {
	boot := &mfutil.BootScope{
		CurrentNode: bkenode.Node{
			Kubelet: &bkev1beta1.Kubelet{
				ControlPlaneComponent: bkev1beta1.ControlPlaneComponent{
					ExtraVolumes: []bkev1beta1.HostPathMount{
						{Name: "kubelet-root-dir", HostPath: "/custom/kubelet"},
					},
				},
			},
		},
		BkeConfig: &bkev1beta1.BKEConfig{
			Cluster: bkev1beta1.Cluster{},
		},
	}
	result := getKubeletDataRootDir(boot)
	assert.Equal(t, "/custom/kubelet", result)
}

func TestInstallContainerdCommandValidationError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(bkevalidte.ValidateCustomExtra, func(extra map[string]string) error {
		return assert.AnError
	})

	k := &KubeadmPlugin{
		boot: &mfutil.BootScope{
			BkeConfig: &bkev1beta1.BKEConfig{
				CustomExtra: map[string]string{"containerd": "test"},
			},
		},
	}

	err := k.installContainerdCommand()
	assert.Error(t, err)
}

func TestInstallKubeletCommandError(t *testing.T) {
	k := &KubeadmPlugin{
		boot: &mfutil.BootScope{
			BkeConfig: &bkev1beta1.BKEConfig{
				Cluster: bkev1beta1.Cluster{},
			},
		},
	}

	err := k.installKubeletCommand(false)
	assert.Error(t, err)
}

func TestInstallKubectlCommand(t *testing.T) {
	k := &KubeadmPlugin{
		boot: &mfutil.BootScope{
			BkeConfig: &bkev1beta1.BKEConfig{
				Cluster: bkev1beta1.Cluster{
					KubernetesVersion: "v1.28.0",
				},
			},
		},
	}
	err := k.installKubectlCommand()
	assert.NoError(t, err)
}

func TestGetKubeletCgroupDriverWithParam(t *testing.T) {
	boot := &mfutil.BootScope{
		BkeConfig: &bkev1beta1.BKEConfig{
			Cluster: bkev1beta1.Cluster{
				ContainerRuntime: bkev1beta1.ContainerRuntime{
					Param: map[string]string{"cgroupDriver": "cgroupfs"},
				},
			},
		},
	}
	result := getKubeletCgroupDriver(boot)
	assert.Equal(t, "cgroupfs", result)
}

func TestProcessVolumeExists(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	var extraVolumes []string
	processVolume("/host/path", "/mount/path", "test-volume", &extraVolumes)
	assert.Len(t, extraVolumes, 1)
	assert.Contains(t, extraVolumes[0], "/host/path:/mount/path")
}

func TestGetKubeletExtraArgsWithNodes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(cluster.GetNodesData, func(namespace, clusterName string) ([]bkev1beta1.Node, error) {
		return []bkev1beta1.Node{
			{Kubelet: &bkev1beta1.Kubelet{ControlPlaneComponent: bkev1beta1.ControlPlaneComponent{ExtraArgs: map[string]string{"key": "value"}}}},
		}, nil
	})

	boot := &mfutil.BootScope{
		ClusterNamespace: "default",
		ClusterName:      "test",
		HostName:         "node1",
		BkeConfig:        &bkev1beta1.BKEConfig{},
		CurrentNode:      bkenode.Node{Hostname: "node1"},
	}
	getKubeletExtraArgs(boot)
}

func TestGetKubeletExtraVolumesWithNodes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(cluster.GetNodesData, func(namespace, clusterName string) ([]bkev1beta1.Node, error) {
		return []bkev1beta1.Node{
			{Kubelet: &bkev1beta1.Kubelet{
				ControlPlaneComponent: bkev1beta1.ControlPlaneComponent{
					ExtraVolumes: []bkev1beta1.HostPathMount{{Name: "vol", HostPath: "/path", MountPath: "/mount"}},
				},
			}},
		}, nil
	})
	patches.ApplyFunc(utils.Exists, func(path string) bool { return true })

	boot := &mfutil.BootScope{
		ClusterNamespace: "default",
		ClusterName:      "test",
		HostName:         "node1",
		BkeConfig:        &bkev1beta1.BKEConfig{},
		CurrentNode:      bkenode.Node{Hostname: "node1"},
	}
	getKubeletExtraVolumes(boot)
}

func TestBuildKubeletCommandWithNodeLocalDNS(t *testing.T) {
	k := &KubeadmPlugin{
		boot: &mfutil.BootScope{
			BkeConfig: &bkev1beta1.BKEConfig{
				Cluster: bkev1beta1.Cluster{
					KubernetesVersion: "v1.28.0",
					Networking:        bkev1beta1.Networking{ServiceSubnet: "10.96.0.0/12", DNSDomain: "cluster.local"},
					CertificatesDir:   "/etc/kubernetes/pki",
					Kubelet:           &bkev1beta1.Kubelet{ManifestsDir: "/etc/kubernetes/manifests"},
				},
				Addons:      []bkev1beta1.Product{{Name: "nodelocaldns", Param: map[string]string{"localdns": "169.254.20.10"}}},
				CustomExtra: map[string]string{"proxyMode": "ipvs"},
			},
			HostIP:   "192.168.1.10",
			HostName: "test-node",
		},
		clusterName: "test-cluster",
	}

	cfg := bkeinit.BkeConfig(*k.boot.BkeConfig)
	command, err := k.buildKubeletCommand(cfg, "http://test.com/kubelet")
	assert.NoError(t, err)
	assert.Contains(t, strings.Join(command, " "), "169.254.20.10")
}

func TestBuildKubeletCommandWithKubeletConfigRef(t *testing.T) {
	k := &KubeadmPlugin{
		boot: &mfutil.BootScope{
			BkeConfig: &bkev1beta1.BKEConfig{
				Cluster: bkev1beta1.Cluster{
					KubernetesVersion: "v1.28.0",
					Networking:        bkev1beta1.Networking{ServiceSubnet: "10.96.0.0/12", DNSDomain: "cluster.local"},
					CertificatesDir:   "/etc/kubernetes/pki",
					Kubelet:           &bkev1beta1.Kubelet{ManifestsDir: "/etc/kubernetes/manifests"},
				},
			},
			KubeletConfigRef: &bkev1beta1.KubeletConfigRef{Name: "kubelet-config", Namespace: "kube-system"},
			HostIP:           "192.168.1.10",
			HostName:         "test-node",
		},
		clusterName:    "test-cluster",
		GableNameSpace: "default",
	}

	cfg := bkeinit.BkeConfig(*k.boot.BkeConfig)
	command, err := k.buildKubeletCommand(cfg, "http://test.com/kubelet")
	assert.NoError(t, err)
	assert.Contains(t, strings.Join(command, " "), "useDeliveredConfig=true")
	assert.Contains(t, strings.Join(command, " "), "kubeletConfigName=kubelet-config")
}

func TestRunControlPlaneCertCommand(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockExec := &mockExecutor{}
	k := &KubeadmPlugin{
		exec:           mockExec,
		k8sClient:      &fakeClient{},
		clusterName:    "test",
		GableNameSpace: "default",
		boot: &mfutil.BootScope{
			BkeConfig: &bkev1beta1.BKEConfig{
				Cluster: bkev1beta1.Cluster{CertificatesDir: "/etc/kubernetes/pki"},
			},
		},
	}

	patches.ApplyFunc((*certPlugin.CertPlugin).Execute, func(_ *certPlugin.CertPlugin, _ context.Context, _ []string) ([]string, error) {
		return nil, nil
	})

	err := k.runControlPlaneCertCommand()
	assert.NoError(t, err)
}

func TestRunControlPlaneManifestCommand(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockExec := &mockExecutor{}
	k := &KubeadmPlugin{
		exec: mockExec,
		boot: &mfutil.BootScope{
			BkeConfig: &bkev1beta1.BKEConfig{
				Cluster: bkev1beta1.Cluster{
					ControlPlane: bkev1beta1.ControlPlane{
						Etcd: &bkev1beta1.Etcd{DataDir: "/var/lib/etcd"},
					},
					Kubelet: &bkev1beta1.Kubelet{ManifestsDir: "/etc/kubernetes/manifests"},
				},
			},
			CurrentNode: bkenode.Node{},
		},
	}

	patches.ApplyFunc((*manifestsPlugin.ManifestPlugin).Execute, func(_ *manifestsPlugin.ManifestPlugin, _ context.Context, _ []string) ([]string, error) {
		return nil, nil
	})

	err := k.runControlPlaneManifestCommand()
	assert.NoError(t, err)
}

func TestUpgradeControlPlaneManifestCommand(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockExec := &mockExecutor{}
	k := &KubeadmPlugin{
		exec: mockExec,
		boot: &mfutil.BootScope{
			BkeConfig: &bkev1beta1.BKEConfig{},
		},
	}

	patches.ApplyFunc((*manifestsPlugin.ManifestPlugin).Execute, func(_ *manifestsPlugin.ManifestPlugin, _ context.Context, _ []string) ([]string, error) {
		return nil, nil
	})

	err := k.upgradeControlPlaneManifestCommand("kube-apiserver")
	assert.NoError(t, err)
}

func TestUpgradeControlPlaneManifestCommandEmpty(t *testing.T) {
	k := &KubeadmPlugin{}
	err := k.upgradeControlPlaneManifestCommand()
	assert.NoError(t, err)
}

func TestUpgradePrePullImageCommand(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockExec := &mockExecutor{}
	k := &KubeadmPlugin{
		exec: mockExec,
		boot: &mfutil.BootScope{
			BkeConfig: &bkev1beta1.BKEConfig{},
		},
	}

	patches.ApplyFunc((*envPlugin.EnvPlugin).Execute, func(_ *envPlugin.EnvPlugin, _ context.Context, _ []string) ([]string, error) {
		return nil, nil
	})

	err := k.upgradePrePullImageCommand()
	assert.NoError(t, err)
}

func TestJoinWorkerCertCommand(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockExec := &mockExecutor{}
	k := &KubeadmPlugin{
		exec:           mockExec,
		k8sClient:      &fakeClient{},
		clusterName:    "test",
		GableNameSpace: "default",
		boot: &mfutil.BootScope{
			BkeConfig: &bkev1beta1.BKEConfig{
				Cluster: bkev1beta1.Cluster{CertificatesDir: "/etc/kubernetes/pki"},
			},
		},
	}

	patches.ApplyFunc((*certPlugin.CertPlugin).Execute, func(_ *certPlugin.CertPlugin, _ context.Context, _ []string) ([]string, error) {
		return nil, nil
	})

	err := k.joinWorkerCertCommand()
	assert.NoError(t, err)
}

func TestInitControlPlaneCertCommand(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockExec := &mockExecutor{}
	k := &KubeadmPlugin{
		exec:           mockExec,
		k8sClient:      &fakeClient{},
		clusterName:    "test",
		GableNameSpace: "default",
		boot: &mfutil.BootScope{
			BkeConfig: &bkev1beta1.BKEConfig{
				Cluster: bkev1beta1.Cluster{CertificatesDir: "/etc/kubernetes/pki"},
			},
		},
	}

	patches.ApplyFunc((*certPlugin.CertPlugin).Execute, func(_ *certPlugin.CertPlugin, _ context.Context, _ []string) ([]string, error) {
		return nil, nil
	})

	err := k.initControlPlaneCertCommand()
	assert.NoError(t, err)
}

func TestInitControlPlaneManifestCommand(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockExec := &mockExecutor{}
	k := &KubeadmPlugin{
		exec: mockExec,
		boot: &mfutil.BootScope{
			BkeConfig: &bkev1beta1.BKEConfig{
				Cluster: bkev1beta1.Cluster{
					ControlPlane: bkev1beta1.ControlPlane{
						Etcd: &bkev1beta1.Etcd{DataDir: "/var/lib/etcd"},
					},
					Kubelet: &bkev1beta1.Kubelet{ManifestsDir: "/etc/kubernetes/manifests"},
				},
			},
			CurrentNode: bkenode.Node{},
		},
	}

	patches.ApplyFunc((*manifestsPlugin.ManifestPlugin).Execute, func(_ *manifestsPlugin.ManifestPlugin, _ context.Context, _ []string) ([]string, error) {
		return nil, nil
	})

	err := k.initControlPlaneManifestCommand()
	assert.NoError(t, err)
}

func TestJoinControlPlaneCertCommand(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockExec := &mockExecutor{}
	k := &KubeadmPlugin{
		exec:           mockExec,
		k8sClient:      &fakeClient{},
		clusterName:    "test",
		GableNameSpace: "default",
		boot: &mfutil.BootScope{
			BkeConfig: &bkev1beta1.BKEConfig{
				Cluster: bkev1beta1.Cluster{CertificatesDir: "/etc/kubernetes/pki"},
			},
		},
	}

	patches.ApplyFunc((*certPlugin.CertPlugin).Execute, func(_ *certPlugin.CertPlugin, _ context.Context, _ []string) ([]string, error) {
		return nil, nil
	})

	err := k.joinControlPlaneCertCommand()
	assert.NoError(t, err)
}

func TestJoinControlPlaneManifestCommand(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockExec := &mockExecutor{}
	k := &KubeadmPlugin{
		exec: mockExec,
		boot: &mfutil.BootScope{
			BkeConfig: &bkev1beta1.BKEConfig{
				Cluster: bkev1beta1.Cluster{
					ControlPlane: bkev1beta1.ControlPlane{
						Etcd: &bkev1beta1.Etcd{DataDir: "/var/lib/etcd"},
					},
					Kubelet: &bkev1beta1.Kubelet{ManifestsDir: "/etc/kubernetes/manifests"},
				},
			},
			CurrentNode: bkenode.Node{},
		},
	}

	patches.ApplyFunc((*manifestsPlugin.ManifestPlugin).Execute, func(_ *manifestsPlugin.ManifestPlugin, _ context.Context, _ []string) ([]string, error) {
		return nil, nil
	})

	err := k.joinControlPlaneManifestCommand()
	assert.NoError(t, err)
}

func TestRunControlPlaneManifestCommandWithNodeConfig(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockExec := &mockExecutor{}
	k := &KubeadmPlugin{
		exec: mockExec,
		boot: &mfutil.BootScope{
			BkeConfig: &bkev1beta1.BKEConfig{
				Cluster: bkev1beta1.Cluster{
					ControlPlane: bkev1beta1.ControlPlane{
						Etcd: &bkev1beta1.Etcd{DataDir: "/var/lib/etcd"},
					},
					Kubelet: &bkev1beta1.Kubelet{ManifestsDir: "/etc/kubernetes/manifests"},
				},
			},
			CurrentNode: bkenode.Node{
				ControlPlane: bkev1beta1.ControlPlane{
					Etcd: &bkev1beta1.Etcd{DataDir: "/custom/etcd"},
				},
				Kubelet: &bkev1beta1.Kubelet{ManifestsDir: "/custom/manifests"},
			},
		},
	}

	patches.ApplyFunc((*manifestsPlugin.ManifestPlugin).Execute, func(_ *manifestsPlugin.ManifestPlugin, _ context.Context, _ []string) ([]string, error) {
		return nil, nil
	})

	err := k.runControlPlaneManifestCommand()
	assert.NoError(t, err)
}
