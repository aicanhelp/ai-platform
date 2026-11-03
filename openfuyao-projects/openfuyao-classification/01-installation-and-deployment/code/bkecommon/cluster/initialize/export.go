/******************************************************************
 * Copyright (c) 2024 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package initialize

import (
	"gopkg.openfuyao.cn/bkecommon/cluster/api/v1beta1"
	"gopkg.openfuyao.cn/bkecommon/cluster/node"
)

const (
	// defaultMaster1IP is a placeholder IP address for the first master node in default configuration
	defaultMaster1IP = "[master-1-ip]"
	// defaultMaster2IP is a placeholder IP address for the second master node in default configuration
	defaultMaster2IP = "[master-2-ip]"
	// defaultMaster3IP is a placeholder IP address for the third master node in default configuration
	defaultMaster3IP = "[master-3-ip]"
	// defaultWorker1IP is a placeholder IP address for the first worker node in default configuration
	defaultWorker1IP = "[worker-1-ip]"
	// defaultWorker2IP is a placeholder IP address for the second worker node in default configuration
	defaultWorker2IP = "[worker-2-ip]"
	// defaultWorker3IP is a placeholder IP address for the third worker node in default configuration
	defaultWorker3IP = "[worker-3-ip]"
	defaultSSHPort   = DefaultNodeSSHPort
	defaultPassword  = "*********"
)

func GetDefaultBKEConfig() *BkeConfig {
	cfg := &BkeConfig{
		Cluster:     defaultCluster(),
		Addons:      defaultAddons(),
		CustomExtra: defaultCustomExtra(),
	}

	SetDefaultBKEConfig(cfg)
	return cfg
}

func defaultCluster() v1beta1.Cluster {
	return v1beta1.Cluster{
		ControlPlane:      defaultControlPlane(),
		Kubelet:           defaultKubelet(),
		Networking:        v1beta1.Networking{},
		KubernetesVersion: "",
		CertificatesDir:   "",
		NTPServer:         DefaultNTPServer,
		ImageRepo: v1beta1.Repo{
			Domain: "",
			Ip:     "",
			Port:   "",
			Prefix: "",
		},
	}
}

func defaultControlPlane() v1beta1.ControlPlane {
	return v1beta1.ControlPlane{
		ControllerManager: &v1beta1.ControlPlaneComponent{},
		Scheduler: &v1beta1.ControlPlaneComponent{
			ExtraVolumes: []v1beta1.HostPathMount{
				{
					Name:      "example",
					HostPath:  "/host/exam",
					MountPath: "/host/exam",
					ReadOnly:  true,
					PathType:  "Directory",
				},
			},
		},
		APIServer: &v1beta1.APIServer{
			APIEndpoint: v1beta1.APIEndpoint{
				Host: "",
				Port: int32(DefaultAPIBindPort),
			},
			ControlPlaneComponent: v1beta1.ControlPlaneComponent{
				ExtraArgs:    nil,
				ExtraVolumes: nil,
			},
			CertSANs: nil,
		},
		Etcd: &v1beta1.Etcd{
			DataDir: "",
			ControlPlaneComponent: v1beta1.ControlPlaneComponent{
				ExtraArgs:    map[string]string{},
				ExtraVolumes: nil,
			},
			ServerCertSANs: nil,
			PeerCertSANs:   nil,
		},
	}
}

func defaultKubelet() *v1beta1.Kubelet {
	return &v1beta1.Kubelet{
		ControlPlaneComponent: v1beta1.ControlPlaneComponent{
			ExtraArgs:    map[string]string{},
			ExtraVolumes: nil,
		},
		ManifestsDir: "",
	}
}

// GetDefaultBKENodes returns default BKENode configurations for a cluster
// This is used for generating example configurations
func GetDefaultBKENodes(clusterName, namespace string) []v1beta1.BKENode {
	defaultNodeSpecs := []v1beta1.Node{
		newNode(defaultMaster1IP, "master-1", []string{node.MasterNodeRole, node.EtcdNodeRole}),
		newNode(defaultMaster2IP, "master-2", []string{node.MasterNodeRole, node.EtcdNodeRole}),
		newNode(defaultMaster3IP, "master-3", []string{node.MasterNodeRole, node.EtcdNodeRole}),
		newNode(defaultWorker1IP, "worker-1", []string{node.WorkerNodeRole}),
		newNode(defaultWorker2IP, "worker-2", []string{node.WorkerNodeRole}),
		newNode(defaultWorker3IP, "worker-3", []string{node.WorkerNodeRole}),
	}
	return node.ConvertNodesToBKENodes(node.Nodes(defaultNodeSpecs), namespace, clusterName)
}

// GetDefaultNodes returns default Node configurations (legacy format)
// Deprecated: Use GetDefaultBKENodes instead for new implementations
func GetDefaultNodes() []v1beta1.Node {
	return []v1beta1.Node{
		newNode(defaultMaster1IP, "master-1", []string{node.MasterNodeRole, node.EtcdNodeRole}),
		newNode(defaultMaster2IP, "master-2", []string{node.MasterNodeRole, node.EtcdNodeRole}),
		newNode(defaultMaster3IP, "master-3", []string{node.MasterNodeRole, node.EtcdNodeRole}),
		newNode(defaultWorker1IP, "worker-1", []string{node.WorkerNodeRole}),
		newNode(defaultWorker2IP, "worker-2", []string{node.WorkerNodeRole}),
		newNode(defaultWorker3IP, "worker-3", []string{node.WorkerNodeRole}),
	}
}

func defaultAddons() []v1beta1.Product {
	return []v1beta1.Product{
		{
			Name:    "kubeproxy",
			Version: "1.25.6",
			Param: map[string]string{
				"clusterNetworkMode": "calico",
			},
		},
		{
			Name:    "calico",
			Version: "v3.4.1",
			Param: map[string]string{
				"calicoMode": "bgp",
			},
		},
		{
			Name:    "coredns",
			Version: "v1.8.0",
		},
		{
			Name:    "nfs-csi",
			Version: "v4.1.0",
			Param: map[string]string{
				"nfsServer": "[your nfs server ip]",
			},
		},
		{
			Name:    "bocoperator",
			Version: "latest",
			Param: map[string]string{
				"boc.version":   "v4.0",
				"boc.nfsServer": "[your nfs server ip]",
			},
		},
		{
			Name:    "cluster-api",
			Version: "v1.3.2",
		},
	}
}

func defaultCustomExtra() map[string]string {
	return map[string]string{
		"containerd": "containerd-1.6.16-linux-{.arch}.tar.gz",
	}
}

func newNode(ip, hostname string, roles []string) v1beta1.Node {
	return v1beta1.Node{
		IP:       ip,
		Port:     defaultSSHPort,
		Username: "root",
		Password: defaultPassword,
		Hostname: hostname,
		Role:     roles,
	}
}
