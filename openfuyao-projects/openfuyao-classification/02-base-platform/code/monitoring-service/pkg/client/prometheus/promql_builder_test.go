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

package prometheus

import (
	"fmt"
	"testing"

	"openfuyao.com/monitoring-service/pkg/client"
)

func TestMakeExpr(t *testing.T) {
	type args struct {
		metric string
		opts   client.QueryOptions
	}
	tests := []struct {
		name string
		args args
		want string
	}{
		{
			name: "Cluster level CPU usage",
			args: args{
				metric: "cluster_cpu_usage",
				opts:   client.QueryOptions{Level: client.LevelCluster},
			},
			want: `sum(rate(node_cpu_seconds_total{mode!="idle", mode!="iowait", ` +
				`mode!="steal"}[5m])) by (cluster)`,
		},

		{
			name: "Node level CPU usage",
			args: args{
				metric: "node_cpu_usage",
				opts: client.QueryOptions{
					Level:    client.LevelNode,
					NodeName: "host-192-168-100-5",
				},
			},
			want: `sum(label_replace(sum(machine_cpu_cores{node=~"host-192-168-100-5"}) by (cluster, node), ` +
				`"instance", "$1", "node", "(.*)")) by (cluster, instance) * (1 - ` +
				`(avg(irate(node_cpu_seconds_total{mode="idle",` +
				`instance=~"host-192-168-100-5"}[5m])) by (cluster, instance)))`,
		},

		{
			name: "Workload (Statefulsets) level CPU usage",
			args: args{
				metric: "workload_cpu_usage",
				opts: client.QueryOptions{
					Level:           client.LevelStatefulset,
					NamespaceName:   "monitoring",
					StatefulsetName: "alertmanager-main",
				},
			},
			want: `sum by (cluster, namespace, pod) (irate(container_cpu_usage_seconds_total{job="kubelet", ` +
				`metrics_path="/metrics/cadvisor", container!="", image!=""}[5m])) * on (cluster, namespace, pod) ` +
				`group_left(owner_kind, owner_name) kube_pod_owner{owner_kind="StatefulSet", ` +
				`owner_name=~"alertmanager-main", ` +
				`job="kube-state-metrics"} * on (cluster, namespace, pod) group_left(node) kube_pod_info{pod=~".*().*", ` +
				`namespace=~"monitoring", job="kube-state-metrics"}`,
		},

		{
			name: "Pod level CPU usage",
			args: args{
				metric: "pod_cpu_usage",
				opts: client.QueryOptions{
					Level:         client.LevelPod,
					NamespaceName: "monitoring",
					PodName:       "alertmanager-main-0",
				},
			},
			want: `sum(irate(container_cpu_usage_seconds_total{pod=~"alertmanager-main-0", namespace=~"monitoring", ` +
				`image!="", job="kubelet", ` +
				`metrics_path="/metrics/cadvisor"}[5m])) by (cluster, namespace, pod, container)`,
		},

		{
			name: "Container level CPU usage",
			args: args{
				metric: "pod_cpu_usage",
				opts: client.QueryOptions{
					Level:         client.LevelContainer,
					NamespaceName: "monitoring",
					PodName:       "alertmanager-main-0",
					ContainerName: "alertmanager",
				},
			},
			want: `sum(irate(container_cpu_usage_seconds_total{namespace=~"monitoring", pod=~"alertmanager-main-0", ` +
				`container=~"alertmanager", image!="", job="kubelet", ` +
				`metrics_path="/metrics/cadvisor"}[5m])) by (cluster, namespace, pod, container)`,
		},

		{
			name: "Kube apiserver QPS by verb and resource",
			args: args{
				metric: "kube_apiserver_qps_by_verb_resource",
				opts: client.QueryOptions{
					Level:     client.LevelControlPlaneComponents,
					Component: client.KubeAPIServer,
					Verb:      "LIST",
					Resource:  "customresourcedefinitions",
				},
			},
			want: `sum(rate(apiserver_request_total{job="apiserver", verb=~"LIST", ` +
				`resource=~"customresourcedefinitions"}[5m])) by (cluster, verb, resource)`,
		},

		{
			name: "ETCD CPU usage",
			args: args{
				metric: "etcd_cpu_usage",
				opts: client.QueryOptions{
					Level:     client.LevelControlPlaneComponents,
					Component: client.ETCD,
					Instance:  "192.168.100.5:2379",
				},
			},
			want: `sum(rate(process_cpu_seconds_total{job="etcd", ` +
				`instance=~"192.168.100.5:2379"}[5m])) by (cluster, instance)`,
		},

		{
			name: "Kube scheduler CPU usage",
			args: args{
				metric: "kube_scheduler_cpu_usage",
				opts: client.QueryOptions{
					Level:     client.LevelControlPlaneComponents,
					Component: client.KubeScheduler,
					Instance:  "192.168.100.5:10259",
				},
			},
			want: `sum(rate(process_cpu_seconds_total{job="kube-scheduler", ` +
				`instance=~"192.168.100.5:10259"}[5m])) by (cluster, instance)`,
		},

		{
			name: "Kube controller manager CPU usage",
			args: args{
				metric: "kube_controller_manager_cpu_usage",
				opts: client.QueryOptions{
					Level:     client.LevelControlPlaneComponents,
					Component: client.KubeControllerManager,
					Instance:  "192.168.100.5:10257",
				},
			},
			want: `sum(rate(process_cpu_seconds_total{job="kube-controller-manager", ` +
				`instance=~"192.168.100.5:10257"}[5m])) by (cluster, instance)`,
		},

		{
			name: "Kubelet CPU usage",
			args: args{
				metric: "kubelet_cpu_usage",
				opts: client.QueryOptions{
					Level:     client.LevelNodeComponents,
					Component: client.Kubelet,
					Instance:  "192.168.100.5:10250",
				},
			},
			want: `sum(rate(process_cpu_seconds_total{job="kubelet", metrics_path="/metrics", ` +
				`instance=~"192.168.100.5:10250"}[5m])) by (cluster, instance)`,
		},

		{
			name: "Kube proxy CPU usage",
			args: args{
				metric: "kube_proxy_cpu_usage",
				opts: client.QueryOptions{
					Level:     client.LevelNodeComponents,
					Component: client.Kubelet,
					Instance:  "host-192-168-100-5",
				},
			},
			want: `sum(rate(process_cpu_seconds_total{job="kube-proxy", ` +
				`instance=~"host-192-168-100-5"}[5m])) by (cluster, instance)`,
		},

		{
			name: "CoreDNS cache size",
			args: args{
				metric: "coredns_cache_size",
				opts: client.QueryOptions{
					Level:     client.LevelAddonsComponents,
					Component: client.CoreDNS,
					Instance:  "172.17.177.104:9153",
				},
			},
			want: `sum(coredns_cache_entries{job="kube-dns", instance=~"172.17.177.104:9153"}) by (cluster, ` +
				`instance, type)`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := makeExpr(tt.args.metric, tt.args.opts); got != tt.want {
				t.Errorf("makeExpr() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMakeNodeMetricExpr(t *testing.T) {
	tmpl := `sum(machine_cpu_cores) by (cluster, node)`
	opts := client.QueryOptions{
		Level:    client.LevelNode,
		NodeName: "host-192-168-100-5",
	}

	result := makeNodeMetricExpr(tmpl, opts)
	fmt.Println(result)

	// Check that node selector was added
	if !contains(result, `node=~"host-192-168-100-5"`) {
		t.Errorf("makeNodeMetricExpr() missing node selector")
	}
}

func TestMakeDeploymentMetricExpr(t *testing.T) {
	tmpl := `sum by (cluster, namespace, pod) (irate(container_cpu_usage_seconds_total{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor", container!="", image!=""}[5m])) * on (cluster, namespace, pod) ` +
		`group_left(owner_kind, owner_name) kube_pod_owner{job="kube-state-metrics"} * on (cluster, namespace, ` +
		`pod) group_left(node) kube_pod_info{job="kube-state-metrics"}`
	opts := client.QueryOptions{
		Level:          client.LevelDeployment,
		ResourceFilter: ".*",
		NamespaceName:  "monitoring",
		DeploymentName: "blackbox-exporter",
	}

	result := makeDeploymentMetricExpr(tmpl, opts)

	// Check that deployment selector was added
	if !contains(result, `owner_kind="ReplicaSet"`) {
		t.Errorf("makeDeploymentMetricExpr() missing ReplicaSet selector")
	}
	// Check that pod selector was added
	if !contains(result, `pod=~".*(.*).*"`) || !contains(result, `namespace=~"monitoring"`) {
		t.Errorf("makeDeploymentMetricExpr() missing pod/namespace selector")
	}
}

func TestMakeStatefulsetMetricExpr(t *testing.T) {
	tmpl := `sum by (cluster, namespace, pod) (irate(container_cpu_usage_seconds_total{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor", container!="", image!=""}[5m])) * on (cluster, namespace, pod) ` +
		`group_left(owner_kind, owner_name) kube_pod_owner{job="kube-state-metrics"} * on (cluster, namespace, ` +
		`pod) group_left(node) kube_pod_info{job="kube-state-metrics"}`
	opts := client.QueryOptions{
		Level:           client.LevelStatefulset,
		ResourceFilter:  ".*",
		NamespaceName:   "monitoring",
		StatefulsetName: "alertmanager-main",
	}

	result := makeStatefulsetMetricExpr(tmpl, opts)

	// Check that statefulset selector was added
	if !contains(result, `owner_kind="StatefulSet"`) {
		t.Errorf("makeStatefulsetMetricExpr() missing StatefulSet selector")
	}
}

func TestMakeDaemonsetMetricExpr(t *testing.T) {
	tmpl := `sum by (cluster, namespace, pod) (irate(container_cpu_usage_seconds_total{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor", container!="", image!=""}[5m])) * on (cluster, namespace, pod) ` +
		`group_left(owner_kind, owner_name) kube_pod_owner{job="kube-state-metrics"} * on (cluster, namespace, ` +
		`pod) group_left(node) kube_pod_info{job="kube-state-metrics"}`
	opts := client.QueryOptions{
		Level:          client.LevelDaemonset,
		ResourceFilter: ".*",
		NamespaceName:  "monitoring",
		DaemonsetName:  "node-exporter",
	}

	result := makeDaemonsetMetricExpr(tmpl, opts)

	// Check that daemonset selector was added
	if !contains(result, `owner_kind="DaemonSet"`) {
		t.Errorf("makeDaemonsetMetricExpr() missing DaemonSet selector")
	}
}

func TestMakePodMetricExpr(t *testing.T) {
	tmpl := `sum(irate(container_cpu_usage_seconds_total{image!="", job="kubelet", ` +
		`metrics_path="/metrics/cadvisor"}[5m])) by (cluster, namespace, pod, container)`
	opts := client.QueryOptions{
		Level:         client.LevelPod,
		NamespaceName: "monitoring",
		PodName:       "alertmanager-main-0",
	}

	result := makePodMetricExpr(tmpl, opts)

	// Check that pod selector was added
	if !contains(result, `pod=~"alertmanager-main-0"`) {
		t.Errorf("makePodMetricExpr() missing pod selector")
	}
	if !contains(result, `namespace=~"monitoring"`) {
		t.Errorf("makePodMetricExpr() missing namespace selector")
	}
}

func TestMakeContainerMetricExpr(t *testing.T) {
	tmpl := `sum(irate(container_cpu_usage_seconds_total{image!="", job="kubelet", ` +
		`metrics_path="/metrics/cadvisor"}[5m])) by (cluster, namespace, pod, container)`
	opts := client.QueryOptions{
		Level:         client.LevelContainer,
		NamespaceName: "monitoring",
		PodName:       "alertmanager-main-0",
		ContainerName: "alertmanager",
	}

	result := makeContainerMetricExpr(tmpl, opts)

	// Check that container selector was added
	if !contains(result, `container=~"alertmanager"`) {
		t.Errorf("makeContainerMetricExpr() missing container selector")
	}
	if !contains(result, `pod=~"alertmanager-main-0"`) {
		t.Errorf("makeContainerMetricExpr() missing pod selector")
	}
	if !contains(result, `namespace=~"monitoring"`) {
		t.Errorf("makeContainerMetricExpr() missing namespace selector")
	}
}

func TestMakeComponentsExpr(t *testing.T) {
	tests := []struct {
		name     string
		tmpl     string
		opts     client.QueryOptions
		contains string
	}{
		{
			name:     "KubeAPIServer with verb and resource",
			tmpl:     `sum(rate(apiserver_request_total{job="apiserver", verb=~"$verb", resource=~"$resource"}[5m]))`,
			opts:     client.QueryOptions{Component: client.KubeAPIServer, Verb: "LIST", Resource: "pods"},
			contains: `verb=~"LIST"`,
		},
		{
			name:     "ETCD with instance",
			tmpl:     `sum(rate(process_cpu_seconds_total{job="etcd", instance=~"$instance"}[5m]))`,
			opts:     client.QueryOptions{Component: client.ETCD, Instance: "192.168.100.5:2379"},
			contains: `instance=~"192.168.100.5:2379"`,
		},
		{
			name:     "Kubelet with instance",
			tmpl:     `sum(rate(process_cpu_seconds_total{job="kubelet", instance=~"$instance"}[5m]))`,
			opts:     client.QueryOptions{Component: client.Kubelet, Instance: "192.168.100.5:10250"},
			contains: `instance=~"192.168.100.5:10250"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := makeComponentsExpr(tt.tmpl, tt.opts)
			if !contains(result, tt.contains) {
				t.Errorf("makeComponentsExpr() = %v, should contain %v", result, tt.contains)
			}
		})
	}
}

func TestReplaceTemplatePlaceholders(t *testing.T) {
	tests := []struct {
		name         string
		template     string
		replacements map[string]string
		want         string
	}{
		{
			name:         "Single placeholder",
			template:     `verb=~"$verb"`,
			replacements: map[string]string{"verb": "LIST"},
			want:         `verb=~"LIST"`,
		},
		{
			name:         "Multiple placeholders",
			template:     `verb=~"$verb", resource=~"$resource"`,
			replacements: map[string]string{"verb": "GET", "resource": "pods"},
			want:         `verb=~"GET", resource=~"pods"`,
		},
		{
			name:         "No placeholders",
			template:     `job="apiserver"`,
			replacements: map[string]string{"verb": "LIST"},
			want:         `job="apiserver"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := replaceTemplatePlaceholders(tt.template, tt.replacements)
			if got != tt.want {
				t.Errorf("replaceTemplatePlaceholders() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMakeWorkloadSelector(t *testing.T) {
	tests := []struct {
		name         string
		workloadKind string
		workloadName string
		wantContains string
	}{
		{
			name:         "ReplicaSet",
			workloadKind: "ReplicaSet",
			workloadName: "my-app",
			wantContains: `owner_kind="ReplicaSet"`,
		},
		{
			name:         "StatefulSet",
			workloadKind: "StatefulSet",
			workloadName: "alertmanager",
			wantContains: `owner_kind="StatefulSet"`,
		},
		{
			name:         "DaemonSet",
			workloadKind: "DaemonSet",
			workloadName: "node-exporter",
			wantContains: `owner_kind="DaemonSet"`,
		},
		{
			name:         "Unknown",
			workloadKind: "Unknown",
			workloadName: "test",
			wantContains: `owner_kind="Unknown"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := makeWorkloadSelector(tt.workloadKind, tt.workloadName)
			if !contains(got, tt.wantContains) {
				t.Errorf("makeWorkloadSelector() = %v, should contain %v", got, tt.wantContains)
			}
			if !contains(got, tt.workloadName) {
				t.Errorf("makeWorkloadSelector() = %v, should contain workload name %v", got, tt.workloadName)
			}
		})
	}
}

func TestRemoveDuplicates(t *testing.T) {
	tests := []struct {
		name  string
		input []string
		want  []string
	}{
		{
			name:  "No duplicates",
			input: []string{"a", "b", "c"},
			want:  []string{"a", "b", "c"},
		},
		{
			name:  "With duplicates",
			input: []string{"a", "b", "a", "c", "b"},
			want:  []string{"a", "b", "c"},
		},
		{
			name:  "All duplicates",
			input: []string{"a", "a", "a"},
			want:  []string{"a"},
		},
		{
			name:  "Empty slice",
			input: []string{},
			want:  []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := removeDuplicates(tt.input)
			if len(got) != len(tt.want) {
				t.Errorf("removeDuplicates() length = %v, want %v", len(got), len(tt.want))
			}
			for i, v := range tt.want {
				if i < len(got) && got[i] != v {
					// Check if value exists in result (order may differ)
					found := false
					for _, g := range got {
						if g == v {
							found = true
							break
						}
					}
					if !found {
						t.Errorf("removeDuplicates() missing value %v", v)
					}
				}
			}
		})
	}
}

func TestFindMatches(t *testing.T) {
	tests := []struct {
		name    string
		tmpl    string
		pattern string
		want    []string
	}{
		{
			name:    "Find container patterns",
			tmpl:    `container_cpu_usage{job="kubelet"} container_memory_usage{job="kubelet"}`,
			pattern: `\bcontainer_\w+`,
			want:    []string{"container_cpu_usage", "container_memory_usage"},
		},
		{
			name:    "Find node patterns",
			tmpl:    `node_cpu_seconds_total{instance="host"} node_memory_bytes{instance="host"}`,
			pattern: `\bnode_\w+`,
			want:    []string{"node_cpu_seconds_total", "node_memory_bytes"},
		},
		{
			name:    "No matches",
			tmpl:    `sum(rate(metric_total[5m]))`,
			pattern: `\bpod_\w+`,
			want:    []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := findMatches(tt.tmpl, tt.pattern)
			if len(got) != len(tt.want) {
				t.Errorf("findMatches() length = %v, want %v", len(got), len(tt.want))
			}
			for _, w := range tt.want {
				found := false
				for _, g := range got {
					if g == w {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("findMatches() missing match %v", w)
				}
			}
		})
	}
}

// Helper function to check if string contains substring
func contains(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
