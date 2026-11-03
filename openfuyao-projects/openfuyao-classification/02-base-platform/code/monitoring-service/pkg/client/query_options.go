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

// Level defines various levels of monitoring context.
type Level int

// Constants for various monitoring levels.
const (
	LevelQuery                  = iota // LevelQuery represents the query level.
	LevelCluster                       // LevelCluster represents the cluster level.
	LevelNode                          // LevelNode represents the node level.
	LevelDeployment                    // LevelDeployment represents the deployment level.
	LevelStatefulset                   // LevelStatefulset represents the statefulset level.
	LevelDaemonset                     // LevelDaemonset represents the daemonset level.
	LevelPod                           // LevelPod represents the pod level.
	LevelContainer                     // LevelContainer represents the container level.
	LevelControlPlaneComponents        // LevelControlPlaneComponents represents the control plane components level.
	LevelNodeComponents                // LevelNodeComponents represents the node components level.
	LevelAddonsComponents              // LevelAddonsComponents represents the addons components level.
)

// Constants for Kubernetes control plane components.
const (
	KubeAPIServer         = "kube-apiserver"
	ETCD                  = "etcd"
	KubeScheduler         = "kube-scheduler"
	KubeControllerManager = "kube-controller-manager"
	Kubelet               = "kubelet"
	KubeProxy             = "kube-proxy"
	CoreDNS               = "coredns"
)

// QueryOptions encapsulates various settings that can be adjusted to refine a monitoring query.
type QueryOptions struct {
	Level           Level  `json:"level"`
	ResourceFilter  string `json:"resourceFilter"`
	NodeName        string `json:"nodeName"`
	NamespaceName   string `json:"namespaceName"`
	DeploymentName  string `json:"deploymentName"`
	StatefulsetName string `json:"statefulsetName"`
	DaemonsetName   string `json:"daemonsetName"`
	PodName         string `json:"podName"`
	ContainerName   string `json:"containerName"`
	Component       string `json:"component"`
	Verb            string `json:"verb"`
	Resource        string `json:"resource"`
	Instance        string `json:"instance"`
}

// WithClusterLevel sets the query level to cluster.
func WithClusterLevel() QueryOption {
	return func(o *QueryOptions) {
		o.Level = LevelCluster
	}
}

// WithNodeName configures the query to filter results based on the specified node name.
func WithNodeName(name string) QueryOption {
	return func(o *QueryOptions) {
		o.Level = LevelNode
		o.NodeName = name
	}
}

// WithDeploymentDetails configures the query to filter results for a specific deployment.
func WithDeploymentDetails(resourceFilter, namespaceName, deploymentName string) QueryOption {
	return func(o *QueryOptions) {
		o.Level = LevelDeployment
		o.ResourceFilter = resourceFilter
		o.NamespaceName = namespaceName
		o.DeploymentName = deploymentName
	}
}

// WithStatefulsetDetails configures the query to filter results for a specific statefulset.
func WithStatefulsetDetails(resourceFilter, namespaceName, statefulsetName string) QueryOption {
	return func(o *QueryOptions) {
		o.Level = LevelStatefulset
		o.ResourceFilter = resourceFilter
		o.NamespaceName = namespaceName
		o.StatefulsetName = statefulsetName
	}
}

// WithDaemonsetDetails configures the query to filter results for a specific daemonset.
func WithDaemonsetDetails(resourceFilter, namespaceName, daemonsetName string) QueryOption {
	return func(o *QueryOptions) {
		o.Level = LevelDaemonset
		o.ResourceFilter = resourceFilter
		o.NamespaceName = namespaceName
		o.DaemonsetName = daemonsetName
	}
}

// WithPodDetails configures the query to filter results for a specific pod.
func WithPodDetails(namespaceName, podName string) QueryOption {
	return func(o *QueryOptions) {
		o.Level = LevelPod
		o.NamespaceName = namespaceName
		o.PodName = podName
	}
}

// WithContainerDetails configures the query to filter results for a specific container.
func WithContainerDetails(namespaceName, podName string, containerName string) QueryOption {
	return func(o *QueryOptions) {
		o.Level = LevelContainer
		o.NamespaceName = namespaceName
		o.PodName = podName
		o.ContainerName = containerName
	}
}

// WithControlPlaneComponentsKubeApiServerDetails configures the query to filter results for kube apiserver.
func WithControlPlaneComponentsKubeApiServerDetails(component, verb, resource string) QueryOption {
	return func(o *QueryOptions) {
		o.Level = LevelControlPlaneComponents
		o.Component = component
		o.Verb = verb
		o.Resource = resource
	}
}

// WithControlPlaneComponentsETCD configures the query to filter results for etcd.
func WithControlPlaneComponentsETCD(component string, instance string) QueryOption {
	return func(o *QueryOptions) {
		o.Level = LevelControlPlaneComponents
		o.Component = component
		o.Instance = instance
	}
}

// WithControlPlaneComponentsKubeScheduler configures the query to filter results for kube scheduler.
func WithControlPlaneComponentsKubeScheduler(component string, instance string) QueryOption {
	return func(o *QueryOptions) {
		o.Level = LevelControlPlaneComponents
		o.Component = component
		o.Instance = instance
	}
}

// WithControlPlaneComponentsKubeControllerManager configures the query to filter results for kube controller manager.
func WithControlPlaneComponentsKubeControllerManager(component string, instance string) QueryOption {
	return func(o *QueryOptions) {
		o.Level = LevelControlPlaneComponents
		o.Component = component
		o.Instance = instance
	}
}

// WithNodeComponentKubelet configures the query to filter results for kubelet.
func WithNodeComponentKubelet(component string, instance string) QueryOption {
	return func(o *QueryOptions) {
		o.Level = LevelNodeComponents
		o.Component = component
		o.Instance = instance
	}
}

// WithNodeComponentKubeProxy configures the query to filter results for kube proxy.
func WithNodeComponentKubeProxy(component string, instance string) QueryOption {
	return func(o *QueryOptions) {
		o.Level = LevelNodeComponents
		o.Component = component
		o.Instance = instance
	}
}

// WithAddonsComponentCoredns configures the query to filter results for coredns.
func WithAddonsComponentCoredns(component string, instance string) QueryOption {
	return func(o *QueryOptions) {
		o.Level = LevelAddonsComponents
		o.Component = component
		o.Instance = instance
	}
}

// NewQueryOptions creates a new instance of QueryOptions and applies a given configuration function to it.
func NewQueryOptions(option QueryOption) *QueryOptions {
	opts := &QueryOptions{}
	option(opts)
	return opts
}

// QueryOption defines a type of function that applies configurations to a QueryOptions instance.
type QueryOption func(*QueryOptions)
