/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package client

import (
	"reflect"
	"testing"
)

func TestWithClusterLevel(t *testing.T) {
	opts := &QueryOptions{}
	WithClusterLevel()(opts)

	if opts.Level != LevelCluster {
		t.Errorf("WithClusterLevel() Level = %v, want %v", opts.Level, LevelCluster)
	}
}

func TestWithNodeName(t *testing.T) {
	opts := &QueryOptions{}
	WithNodeName("host-192-168-100-5")(opts)

	if opts.Level != LevelNode {
		t.Errorf("WithNodeName() Level = %v, want %v", opts.Level, LevelNode)
	}
	if opts.NodeName != "host-192-168-100-5" {
		t.Errorf("WithNodeName() NodeName = %v, want %v", opts.NodeName, "host-192-168-100-5")
	}
}

func TestWithDeploymentDetails(t *testing.T) {
	opts := &QueryOptions{}
	WithDeploymentDetails(".*", "monitoring", "blackbox-exporter")(opts)

	if opts.Level != LevelDeployment {
		t.Errorf("WithDeploymentDetails() Level = %v, want %v", opts.Level, LevelDeployment)
	}
	if opts.ResourceFilter != ".*" {
		t.Errorf("WithDeploymentDetails() ResourceFilter = %v, want %v", opts.ResourceFilter, ".*")
	}
	if opts.NamespaceName != "monitoring" {
		t.Errorf("WithDeploymentDetails() NamespaceName = %v, want %v", opts.NamespaceName, "monitoring")
	}
	if opts.DeploymentName != "blackbox-exporter" {
		t.Errorf("WithDeploymentDetails() DeploymentName = %v, want %v", opts.DeploymentName, "blackbox-exporter")
	}
}

func TestWithStatefulsetDetails(t *testing.T) {
	opts := &QueryOptions{}
	WithStatefulsetDetails(".*", "monitoring", "alertmanager-main")(opts)

	if opts.Level != LevelStatefulset {
		t.Errorf("WithStatefulsetDetails() Level = %v, want %v", opts.Level, LevelStatefulset)
	}
	if opts.StatefulsetName != "alertmanager-main" {
		t.Errorf("WithStatefulsetDetails() StatefulsetName = %v, want %v", opts.StatefulsetName, "alertmanager-main")
	}
}

func TestWithDaemonsetDetails(t *testing.T) {
	opts := &QueryOptions{}
	WithDaemonsetDetails(".*", "monitoring", "node-exporter")(opts)

	if opts.Level != LevelDaemonset {
		t.Errorf("WithDaemonsetDetails() Level = %v, want %v", opts.Level, LevelDaemonset)
	}
	if opts.DaemonsetName != "node-exporter" {
		t.Errorf("WithDaemonsetDetails() DaemonsetName = %v, want %v", opts.DaemonsetName, "node-exporter")
	}
}

func TestWithPodDetails(t *testing.T) {
	opts := &QueryOptions{}
	WithPodDetails("monitoring", "alertmanager-main-0")(opts)

	if opts.Level != LevelPod {
		t.Errorf("WithPodDetails() Level = %v, want %v", opts.Level, LevelPod)
	}
	if opts.NamespaceName != "monitoring" {
		t.Errorf("WithPodDetails() NamespaceName = %v, want %v", opts.NamespaceName, "monitoring")
	}
	if opts.PodName != "alertmanager-main-0" {
		t.Errorf("WithPodDetails() PodName = %v, want %v", opts.PodName, "alertmanager-main-0")
	}
}

func TestWithContainerDetails(t *testing.T) {
	opts := &QueryOptions{}
	WithContainerDetails("monitoring", "alertmanager-main-0", "alertmanager")(opts)

	if opts.Level != LevelContainer {
		t.Errorf("WithContainerDetails() Level = %v, want %v", opts.Level, LevelContainer)
	}
	if opts.NamespaceName != "monitoring" {
		t.Errorf("WithContainerDetails() NamespaceName = %v, want %v", opts.NamespaceName, "monitoring")
	}
	if opts.PodName != "alertmanager-main-0" {
		t.Errorf("WithContainerDetails() PodName = %v, want %v", opts.PodName, "alertmanager-main-0")
	}
	if opts.ContainerName != "alertmanager" {
		t.Errorf("WithContainerDetails() ContainerName = %v, want %v", opts.ContainerName, "alertmanager")
	}
}

func TestWithControlPlaneComponentsKubeApiServerDetails(t *testing.T) {
	opts := &QueryOptions{}
	WithControlPlaneComponentsKubeApiServerDetails("kube-apiserver", "LIST", "pods")(opts)

	if opts.Level != LevelControlPlaneComponents {
		t.Errorf("WithControlPlaneComponentsKubeApiServerDetails() Level = %v, want %v", opts.Level, LevelControlPlaneComponents)
	}
	if opts.Component != "kube-apiserver" {
		t.Errorf("WithControlPlaneComponentsKubeApiServerDetails() Component = %v, want %v", opts.Component, "kube-apiserver")
	}
	if opts.Verb != "LIST" {
		t.Errorf("WithControlPlaneComponentsKubeApiServerDetails() Verb = %v, want %v", opts.Verb, "LIST")
	}
	if opts.Resource != "pods" {
		t.Errorf("WithControlPlaneComponentsKubeApiServerDetails() Resource = %v, want %v", opts.Resource, "pods")
	}
}

func TestWithControlPlaneComponentsETCD(t *testing.T) {
	opts := &QueryOptions{}
	WithControlPlaneComponentsETCD("etcd", "192.168.100.5:2379")(opts)

	if opts.Level != LevelControlPlaneComponents {
		t.Errorf("WithControlPlaneComponentsETCD() Level = %v, want %v", opts.Level, LevelControlPlaneComponents)
	}
	if opts.Component != "etcd" {
		t.Errorf("WithControlPlaneComponentsETCD() Component = %v, want %v", opts.Component, "etcd")
	}
	if opts.Instance != "192.168.100.5:2379" {
		t.Errorf("WithControlPlaneComponentsETCD() Instance = %v, want %v", opts.Instance, "192.168.100.5:2379")
	}
}

func TestWithControlPlaneComponentsKubeScheduler(t *testing.T) {
	opts := &QueryOptions{}
	WithControlPlaneComponentsKubeScheduler("kube-scheduler", "192.168.100.5:10259")(opts)

	if opts.Level != LevelControlPlaneComponents {
		t.Errorf("WithControlPlaneComponentsKubeScheduler() Level = %v, want %v", opts.Level, LevelControlPlaneComponents)
	}
	if opts.Component != "kube-scheduler" {
		t.Errorf("WithControlPlaneComponentsKubeScheduler() Component = %v, want %v", opts.Component, "kube-scheduler")
	}
	if opts.Instance != "192.168.100.5:10259" {
		t.Errorf("WithControlPlaneComponentsKubeScheduler() Instance = %v, want %v", opts.Instance, "192.168.100.5:10259")
	}
}

func TestWithControlPlaneComponentsKubeControllerManager(t *testing.T) {
	opts := &QueryOptions{}
	WithControlPlaneComponentsKubeControllerManager("kube-controller-manager", "192.168.100.5:10257")(opts)

	if opts.Level != LevelControlPlaneComponents {
		t.Errorf("WithControlPlaneComponentsKubeControllerManager() Level = %v, want %v", opts.Level, LevelControlPlaneComponents)
	}
	if opts.Component != "kube-controller-manager" {
		t.Errorf("WithControlPlaneComponentsKubeControllerManager() Component = %v, want %v", opts.Component, "kube-controller-manager")
	}
	if opts.Instance != "192.168.100.5:10257" {
		t.Errorf("WithControlPlaneComponentsKubeControllerManager() Instance = %v, want %v", opts.Instance, "192.168.100.5:10257")
	}
}

func TestWithNodeComponentKubelet(t *testing.T) {
	opts := &QueryOptions{}
	WithNodeComponentKubelet("kubelet", "192.168.100.5:10250")(opts)

	if opts.Level != LevelNodeComponents {
		t.Errorf("WithNodeComponentKubelet() Level = %v, want %v", opts.Level, LevelNodeComponents)
	}
	if opts.Component != "kubelet" {
		t.Errorf("WithNodeComponentKubelet() Component = %v, want %v", opts.Component, "kubelet")
	}
	if opts.Instance != "192.168.100.5:10250" {
		t.Errorf("WithNodeComponentKubelet() Instance = %v, want %v", opts.Instance, "192.168.100.5:10250")
	}
}

func TestWithNodeComponentKubeProxy(t *testing.T) {
	opts := &QueryOptions{}
	WithNodeComponentKubeProxy("kube-proxy", "host-192-168-100-5")(opts)

	if opts.Level != LevelNodeComponents {
		t.Errorf("WithNodeComponentKubeProxy() Level = %v, want %v", opts.Level, LevelNodeComponents)
	}
	if opts.Component != "kube-proxy" {
		t.Errorf("WithNodeComponentKubeProxy() Component = %v, want %v", opts.Component, "kube-proxy")
	}
	if opts.Instance != "host-192-168-100-5" {
		t.Errorf("WithNodeComponentKubeProxy() Instance = %v, want %v", opts.Instance, "host-192-168-100-5")
	}
}

func TestWithAddonsComponentCoredns(t *testing.T) {
	opts := &QueryOptions{}
	WithAddonsComponentCoredns("coredns", "172.17.177.104:9153")(opts)

	if opts.Level != LevelAddonsComponents {
		t.Errorf("WithAddonsComponentCoredns() Level = %v, want %v", opts.Level, LevelAddonsComponents)
	}
	if opts.Component != "coredns" {
		t.Errorf("WithAddonsComponentCoredns() Component = %v, want %v", opts.Component, "coredns")
	}
	if opts.Instance != "172.17.177.104:9153" {
		t.Errorf("WithAddonsComponentCoredns() Instance = %v, want %v", opts.Instance, "172.17.177.104:9153")
	}
}

func TestNewQueryOptions(t *testing.T) {
	tests := []struct {
		name   string
		option QueryOption
		want   *QueryOptions
	}{
		{
			name:   "Cluster Level",
			option: WithClusterLevel(),
			want: &QueryOptions{
				Level: LevelCluster,
			},
		},
		{
			name:   "Node Level",
			option: WithNodeName("test-node"),
			want: &QueryOptions{
				Level:    LevelNode,
				NodeName: "test-node",
			},
		},
		{
			name:   "Pod Level",
			option: WithPodDetails("namespace", "pod-name"),
			want: &QueryOptions{
				Level:         LevelPod,
				NamespaceName: "namespace",
				PodName:       "pod-name",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NewQueryOptions(tt.option)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewQueryOptions() = %v, want %v", got, tt.want)
			}
		})
	}
}
