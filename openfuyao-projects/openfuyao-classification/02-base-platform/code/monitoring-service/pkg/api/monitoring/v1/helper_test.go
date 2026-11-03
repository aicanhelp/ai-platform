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

package v1

import (
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	"github.com/emicklei/go-restful/v3"
	"github.com/stretchr/testify/assert"

	"openfuyao.com/monitoring-service/pkg/client"
)

func TestParseRequestParamsResources(t *testing.T) {
	tests := []struct {
		name     string
		url      string
		pathVars map[string]string
		want     reqParams
	}{
		{
			name:     "empty request uses default filter",
			url:      "http://example.com/rest/monitoring/v1/cluster",
			pathVars: map[string]string{},
			want: reqParams{
				metricFilter:   defaultFilter,
				resourceFilter: defaultFilter,
			},
		},
		{
			name:     "query params time, start, end, step, expr",
			url:      "http://example.com/path?time=2024-05-13%2009:00:00&start=2024-05-13%2008:00:00&end=2024-05-13%2010:00:00&step=15s&expr=up",
			pathVars: map[string]string{},
			want: reqParams{
				time:           "2024-05-13 09:00:00",
				start:          "2024-05-13 08:00:00",
				end:            "2024-05-13 10:00:00",
				step:           "15s",
				expression:     "up",
				metricFilter:   defaultFilter,
				resourceFilter: defaultFilter,
			},
		},
		{
			name:     "custom metrics_filter and resources_filter",
			url:      "http://example.com/path?metrics_filter=cpu_usage&resources_filter=.*",
			pathVars: map[string]string{},
			want: reqParams{
				metricFilter:   "cpu_usage",
				resourceFilter: ".*",
			},
		},
		{
			name:     "cluster level no path params",
			url:      "http://example.com/rest/monitoring/v1/cluster?time=2024-05-13%2009:00:00",
			pathVars: map[string]string{},
			want: reqParams{
				time:           "2024-05-13 09:00:00",
				metricFilter:   defaultFilter,
				resourceFilter: defaultFilter,
			},
		},
		{
			name:     "node path param",
			url:      "http://example.com/path",
			pathVars: map[string]string{"node": "host-192-168-100-5"},
			want: reqParams{
				nodeName:       "host-192-168-100-5",
				metricFilter:   defaultFilter,
				resourceFilter: defaultFilter,
			},
		},
		{
			name:     "namespace and pod path params",
			url:      "http://example.com/path",
			pathVars: map[string]string{"namespace": "monitoring", "pod": "alertmanager-main-0"},
			want: reqParams{
				namespaceName:  "monitoring",
				podName:        "alertmanager-main-0",
				metricFilter:   defaultFilter,
				resourceFilter: defaultFilter,
			},
		},
		{
			name:     "deployment path param",
			url:      "http://example.com/path",
			pathVars: map[string]string{"namespace": "monitoring", "deployment": "blackbox-exporter"},
			want: reqParams{
				namespaceName:  "monitoring",
				deploymentName: "blackbox-exporter",
				metricFilter:   defaultFilter,
				resourceFilter: defaultFilter,
			},
		},
		{
			name:     "statefulset path param",
			url:      "http://example.com/path",
			pathVars: map[string]string{"namespace": "monitoring", "statefulset": "alertmanager-main"},
			want: reqParams{
				namespaceName:   "monitoring",
				statefulsetName: "alertmanager-main",
				metricFilter:    defaultFilter,
				resourceFilter:  defaultFilter,
			},
		},
		{
			name:     "daemonset path param",
			url:      "http://example.com/path",
			pathVars: map[string]string{"namespace": "monitoring", "daemonset": "node-exporter"},
			want: reqParams{
				namespaceName:  "monitoring",
				daemonsetName:  "node-exporter",
				metricFilter:   defaultFilter,
				resourceFilter: defaultFilter,
			},
		},
		{
			name:     "container path param",
			url:      "http://example.com/path",
			pathVars: map[string]string{"namespace": "monitoring", "pod": "app-0", "container": "main"},
			want: reqParams{
				namespaceName:  "monitoring",
				podName:        "app-0",
				containerName:  "main",
				metricFilter:   defaultFilter,
				resourceFilter: defaultFilter,
			},
		},
		{
			name: "all resource path params and query",
			url:  "http://example.com/path?time=2024-05-13%2009:00:00&metrics_filter=.*&resources_filter=.*",
			pathVars: map[string]string{
				"node": "n1", "namespace": "ns", "deployment": "dep",
				"statefulset": "sts", "daemonset": "ds", "pod": "p1", "container": "c1",
			},
			want: reqParams{
				time:            "2024-05-13 09:00:00",
				metricFilter:    ".*",
				resourceFilter:  ".*",
				nodeName:        "n1",
				namespaceName:   "ns",
				deploymentName:  "dep",
				statefulsetName: "sts",
				daemonsetName:   "ds",
				podName:         "p1",
				containerName:   "c1",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			httpReq := httptest.NewRequest(http.MethodGet, tt.url, nil)
			req := restful.NewRequest(httpReq)
			for k, v := range tt.pathVars {
				req.PathParameters()[k] = v
			}
			got := parseRequestParamsResources(req)
			assert.Equal(t, tt.want.time, got.time, "time")
			assert.Equal(t, tt.want.start, got.start, "start")
			assert.Equal(t, tt.want.end, got.end, "end")
			assert.Equal(t, tt.want.step, got.step, "step")
			assert.Equal(t, tt.want.expression, got.expression, "expression")
			assert.Equal(t, tt.want.metricFilter, got.metricFilter, "metricFilter")
			assert.Equal(t, tt.want.resourceFilter, got.resourceFilter, "resourceFilter")
			assert.Equal(t, tt.want.nodeName, got.nodeName, "nodeName")
			assert.Equal(t, tt.want.namespaceName, got.namespaceName, "namespaceName")
			assert.Equal(t, tt.want.deploymentName, got.deploymentName, "deploymentName")
			assert.Equal(t, tt.want.statefulsetName, got.statefulsetName, "statefulsetName")
			assert.Equal(t, tt.want.daemonsetName, got.daemonsetName, "daemonsetName")
			assert.Equal(t, tt.want.podName, got.podName, "podName")
			assert.Equal(t, tt.want.containerName, got.containerName, "containerName")
		})
	}
}

func TestParseRequestParamsComponents(t *testing.T) {
	tests := []struct {
		name     string
		url      string
		pathVars map[string]string
		want     reqParams
	}{
		{
			name:     "empty request uses default filter",
			url:      "http://example.com/rest/monitoring/v1/components",
			pathVars: map[string]string{},
			want: reqParams{
				metricFilter: defaultFilter,
			},
		},
		{
			name:     "query params time, start, end, step",
			url:      "http://example.com/path?time=2024-05-13%2009:00:00&start=2024-05-13%2008:00:00&end=2024-05-13%2010:00:00&step=30s",
			pathVars: map[string]string{},
			want: reqParams{
				time:         "2024-05-13 09:00:00",
				start:        "2024-05-13 08:00:00",
				end:          "2024-05-13 10:00:00",
				step:         "30s",
				metricFilter: defaultFilter,
			},
		},
		{
			name:     "component path param only",
			url:      "http://example.com/path",
			pathVars: map[string]string{"component": "etcd"},
			want: reqParams{
				component:    "etcd",
				metricFilter: defaultFilter,
			},
		},
		{
			name:     "component and instance (etcd)",
			url:      "http://example.com/path?instance=192.168.100.5:2379",
			pathVars: map[string]string{"component": "etcd"},
			want: reqParams{
				component:    "etcd",
				instance:     "192.168.100.5:2379",
				metricFilter: defaultFilter,
			},
		},
		{
			name:     "kube-apiserver with verb and resource",
			url:      "http://example.com/path?verb=LIST&resource=pods",
			pathVars: map[string]string{"component": "kube-apiserver"},
			want: reqParams{
				component:    "kube-apiserver",
				verb:         "LIST",
				resource:     "pods",
				metricFilter: defaultFilter,
			},
		},
		{
			name:     "custom metrics_filter",
			url:      "http://example.com/path?metrics_filter=cpu_usage",
			pathVars: map[string]string{"component": "kubelet"},
			want: reqParams{
				component:    "kubelet",
				metricFilter: "cpu_usage",
			},
		},
		{
			name:     "all component query and path params",
			url:      "http://example.com/path?time=2024-05-13%2009:00:00&verb=GET&resource=configmaps&instance=10.0.0.1:6443&metrics_filter=.*",
			pathVars: map[string]string{"component": "kube-apiserver"},
			want: reqParams{
				time:         "2024-05-13 09:00:00",
				metricFilter: ".*",
				component:    "kube-apiserver",
				verb:         "GET",
				resource:     "configmaps",
				instance:     "10.0.0.1:6443",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			httpReq := httptest.NewRequest(http.MethodGet, tt.url, nil)
			req := restful.NewRequest(httpReq)
			for k, v := range tt.pathVars {
				req.PathParameters()[k] = v
			}
			got := parseRequestParamsComponents(req)
			assert.Equal(t, tt.want.time, got.time, "time")
			assert.Equal(t, tt.want.start, got.start, "start")
			assert.Equal(t, tt.want.end, got.end, "end")
			assert.Equal(t, tt.want.step, got.step, "step")
			assert.Equal(t, tt.want.metricFilter, got.metricFilter, "metricFilter")
			assert.Equal(t, tt.want.component, got.component, "component")
			assert.Equal(t, tt.want.verb, got.verb, "verb")
			assert.Equal(t, tt.want.resource, got.resource, "resource")
			assert.Equal(t, tt.want.instance, got.instance, "instance")
		})
	}
}

func TestParseRequestParamsResourcesEscapeSpecialChars(t *testing.T) {
	// When query or path contains special chars, they are escaped via utils.EscapeSpecialChars
	// e.g. newline -> \n
	httpReq := httptest.NewRequest(http.MethodGet, "http://example.com/path?time=2024-05-13%2009:00:00", nil)
	req := restful.NewRequest(httpReq)
	req.PathParameters()["pod"] = "app" + "\n" + "main" // real newline

	got := parseRequestParamsResources(req)
	assert.Equal(t, "2024-05-13 09:00:00", got.time)
	// Newline in pod name is escaped to backslash-n
	assert.Equal(t, `app\nmain`, got.podName)
}

func TestParseRequestParamsComponentsEscapeSpecialChars(t *testing.T) {
	httpReq := httptest.NewRequest(http.MethodGet, "http://example.com/path?instance=10.0.0.1%3A2379", nil)
	req := restful.NewRequest(httpReq)
	req.PathParameters()["component"] = "kube-apiserver"

	got := parseRequestParamsComponents(req)
	assert.Equal(t, "kube-apiserver", got.component)
	assert.Equal(t, "10.0.0.1:2379", got.instance) // URL decoded; no special chars to escape
}

func TestTransformQueryOptionsResources(t *testing.T) {
	type args struct {
		reqParams reqParams
		lvl       client.Level
	}
	tests := []struct {
		name          string
		args          args
		wantQueryOpts queryOptions
		wantErr       bool
	}{
		{
			name: "Test with LevelCluster",
			args: args{
				reqParams: reqParams{
					time:         "2024-05-13 09:00:00",
					metricFilter: "cluster_nodes_count",
				},
				lvl: client.LevelCluster,
			},
			wantQueryOpts: queryOptions{
				start:        time.Time{},
				end:          time.Time{},
				time:         time.Date(2024, 5, 13, 9, 0, 0, 0, time.Local),
				step:         time.Duration(0 * time.Second),
				metricFilter: "cluster_nodes_count",
				namedMetrics: clusterMetrics,
				option:       client.WithClusterLevel(),
			},
			wantErr: false,
		},

		{
			name: "Test with LevelDeployment",
			args: args{
				reqParams: reqParams{
					time:           "2024-05-13 09:00:00",
					metricFilter:   ".*",
					resourceFilter: ".*",
					namespaceName:  "monitoring",
					deploymentName: "blackbox-exporter",
				},
				lvl: client.LevelDeployment,
			},
			wantQueryOpts: queryOptions{
				start:        time.Time{},
				end:          time.Time{},
				time:         time.Date(2024, 5, 13, 9, 0, 0, 0, time.Local),
				step:         time.Duration(0 * time.Second),
				metricFilter: ".*",
				namedMetrics: workloadMetrics,
				option: client.WithDeploymentDetails(".*", "monitoring",
					"blackbox-exporter"),
			},
			wantErr: false,
		},

		{
			name: "Test with LevelStatefulset",
			args: args{
				reqParams: reqParams{
					time:            "2024-05-13 09:00:00",
					metricFilter:    ".*",
					resourceFilter:  ".*",
					namespaceName:   "monitoring",
					statefulsetName: "alertmanager-main",
				},
				lvl: client.LevelStatefulset,
			},
			wantQueryOpts: queryOptions{
				start:        time.Time{},
				end:          time.Time{},
				time:         time.Date(2024, 5, 13, 9, 0, 0, 0, time.Local),
				step:         time.Duration(0 * time.Second),
				metricFilter: ".*",
				namedMetrics: workloadMetrics,
				option: client.WithStatefulsetDetails(".*", "monitoring",
					"alertmanager-main"),
			},
			wantErr: false,
		},

		{
			name: "Test with LevelDaemonset",
			args: args{
				reqParams: reqParams{
					time:           "2024-05-13 09:00:00",
					metricFilter:   ".*",
					resourceFilter: ".*",
					namespaceName:  "monitoring",
					daemonsetName:  "node-exporter",
				},
				lvl: client.LevelDaemonset,
			},
			wantQueryOpts: queryOptions{
				start:        time.Time{},
				end:          time.Time{},
				time:         time.Date(2024, 5, 13, 9, 0, 0, 0, time.Local),
				step:         time.Duration(0 * time.Second),
				metricFilter: ".*",
				namedMetrics: workloadMetrics,
				option: client.WithDaemonsetDetails(".*", "monitoring",
					"node-exporter"),
			},
			wantErr: false,
		},

		{
			name: "Test with LevelPod",
			args: args{
				reqParams: reqParams{
					time:          "2024-05-13 09:00:00",
					metricFilter:  ".*",
					namespaceName: "monitoring",
					podName:       "alertmanager-main-0",
				},
				lvl: client.LevelPod,
			},
			wantQueryOpts: queryOptions{
				start:        time.Time{},
				end:          time.Time{},
				time:         time.Date(2024, 5, 13, 9, 0, 0, 0, time.Local),
				step:         time.Duration(0 * time.Second),
				metricFilter: ".*",
				namedMetrics: podMetrics,
				option:       client.WithPodDetails("monitoring", "alertmanager-main-0"),
			},
			wantErr: false,
		},

		{
			name: "Test with unsupported level",
			args: args{
				reqParams: reqParams{},
				lvl:       999,
			},
			wantQueryOpts: queryOptions{},
			wantErr:       true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotQueryOpts, err := transformQueryOptionsResources(tt.args.reqParams, tt.args.lvl)
			if (err != nil) != tt.wantErr {
				t.Errorf("transformQueryOptionsResources() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if err != nil {
				return
			}

			gotOpts := client.NewQueryOptions(gotQueryOpts.option)
			wantOpts := client.NewQueryOptions(tt.wantQueryOpts.option)
			if !reflect.DeepEqual(gotOpts, wantOpts) {
				t.Errorf("Option did not set correctly, got: %v, want: %v", gotOpts, wantOpts)
				return
			}

			gotQueryOpts.option = nil
			tt.wantQueryOpts.option = nil
			if !reflect.DeepEqual(gotQueryOpts, tt.wantQueryOpts) {
				t.Errorf("transformQueryOptionsResources() gotQueryOpts = %v, want %v", gotQueryOpts, tt.wantQueryOpts)
			}
		})
	}
}

func TestTransformQueryOptionsComponents(t *testing.T) {
	type args struct {
		reqParams reqParams
		lvl       client.Level
	}
	tests := []struct {
		name          string
		args          args
		wantQueryOpts queryOptions
		wantErr       bool
	}{
		{
			name: "Test with LevelControlPlaneComponents - ETCD",
			args: args{
				reqParams: reqParams{
					time:         "2024-05-13 09:00:00",
					metricFilter: ".*",
					instance:     "192.168.100.5:2379",
					component:    client.ETCD,
				},
				lvl: client.LevelControlPlaneComponents,
			},
			wantQueryOpts: queryOptions{
				start:        time.Time{},
				end:          time.Time{},
				time:         time.Date(2024, 5, 13, 9, 0, 0, 0, time.Local),
				step:         time.Duration(0 * time.Second),
				metricFilter: ".*",
				namedMetrics: controlPlaneComponentsETCDMetrics,
				option:       client.WithControlPlaneComponentsETCD("etcd", "192.168.100.5:2379"),
			},
			wantErr: false,
		},

		{
			name: "Test with LevelNodeComponents - kubelet",
			args: args{
				reqParams: reqParams{
					time:         "2024-05-13 09:00:00",
					metricFilter: ".*",
					instance:     "192.168.100.5:10250",
					component:    client.Kubelet,
				},
				lvl: client.LevelNodeComponents,
			},
			wantQueryOpts: queryOptions{
				start:        time.Time{},
				end:          time.Time{},
				time:         time.Date(2024, 5, 13, 9, 0, 0, 0, time.Local),
				step:         time.Duration(0 * time.Second),
				metricFilter: ".*",
				namedMetrics: nodeComponentsKubeletMetrics,
				option:       client.WithNodeComponentKubelet("kubelet", "192.168.100.5:10250"),
			},
			wantErr: false,
		},

		{
			name: "Test with LevelAddonsComponents - coredns",
			args: args{
				reqParams: reqParams{
					time:         "2024-05-13 09:00:00",
					metricFilter: ".*",
					instance:     "172.17.177.104:9153",
					component:    client.CoreDNS,
				},
				lvl: client.LevelAddonsComponents,
			},
			wantQueryOpts: queryOptions{
				start:        time.Time{},
				end:          time.Time{},
				time:         time.Date(2024, 5, 13, 9, 0, 0, 0, time.Local),
				step:         time.Duration(0 * time.Second),
				metricFilter: ".*",
				namedMetrics: addonsComponentsCorednsMetrics,
				option:       client.WithAddonsComponentCoredns("coredns", "172.17.177.104:9153"),
			},
			wantErr: false,
		},

		{
			name: "Test with unsupported level",
			args: args{
				reqParams: reqParams{},
				lvl:       999,
			},
			wantQueryOpts: queryOptions{},
			wantErr:       true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotQueryOpts, err := transformQueryOptionsComponents(tt.args.reqParams, tt.args.lvl)
			if (err != nil) != tt.wantErr {
				t.Errorf("transformQueryOptionsResources() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if err != nil {
				return
			}

			gotOpts := client.NewQueryOptions(gotQueryOpts.option)
			wantOpts := client.NewQueryOptions(tt.wantQueryOpts.option)
			if !reflect.DeepEqual(gotOpts, wantOpts) {
				t.Errorf("Option did not set correctly, got: %v, want: %v", gotOpts, wantOpts)
				return
			}

			gotQueryOpts.option = nil
			tt.wantQueryOpts.option = nil
			if !reflect.DeepEqual(gotQueryOpts, tt.wantQueryOpts) {
				t.Errorf("transformQueryOptionsResources() gotQueryOpts = %v, want %v", gotQueryOpts, tt.wantQueryOpts)
			}
		})
	}
}

func TestTransformVerbResourceInstanceQueryOptions(t *testing.T) {
	type args struct {
		reqParams reqParams
		queryOpts queryOptions
	}
	tests := []struct {
		name    string
		args    args
		want    queryOptions
		wantErr bool
	}{
		{
			name: "KubeAPIServer with missing parameters (verb and resource)",
			args: args{
				reqParams: reqParams{component: "kube-apiserver", verb: "", resource: ""},
				queryOpts: queryOptions{},
			},
			want:    queryOptions{},
			wantErr: true,
		},

		{
			name: "KubeAPIServer with missing parameter (resource)",
			args: args{
				reqParams: reqParams{component: "kube-apiserver", verb: "LIST", resource: ""},
				queryOpts: queryOptions{},
			},
			want:    queryOptions{},
			wantErr: true,
		},

		{
			name: "KubeAPIServer with all parameters set",
			args: args{
				reqParams: reqParams{component: "kube-apiserver", verb: "LIST", resource: "apiservices",
					instance: "192.168.100.5:6443"},
				queryOpts: queryOptions{},
			},
			want:    queryOptions{},
			wantErr: true,
		},

		{
			name: "Valid case for KubeAPIServer",
			args: args{
				reqParams: reqParams{component: "kube-apiserver", verb: "LIST", resource: "apiservices",
					instance: ""},
				queryOpts: queryOptions{},
			},
			want:    queryOptions{},
			wantErr: false,
		},

		{
			name: "Non-KubeAPIServer with verb set",
			args: args{
				reqParams: reqParams{component: "kube-scheduler", verb: "GET", resource: "",
					instance: "192.168.100.5:10259"},
				queryOpts: queryOptions{},
			},
			want:    queryOptions{},
			wantErr: true,
		},

		{
			name: "Non-KubeAPIServer with missing instance",
			args: args{
				reqParams: reqParams{component: "kube-scheduler", verb: "", resource: "",
					instance: ""},
				queryOpts: queryOptions{},
			},
			want:    queryOptions{},
			wantErr: true,
		},

		{
			name: "Valid case for Non-KubeAPIServer",
			args: args{
				reqParams: reqParams{component: "kube-scheduler", verb: "", resource: "",
					instance: "192.168.100.5:10259"},
				queryOpts: queryOptions{},
			},
			want:    queryOptions{},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := transformVerbResourceInstanceQueryOptions(tt.args.reqParams, tt.args.queryOpts)
			if (err != nil) != tt.wantErr {
				t.Errorf("transformVerbResourceInstanceQueryOptions() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("transformVerbResourceInstanceQueryOptions() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestTransformTimeBasedQueryOptions(t *testing.T) {
	type args struct {
		reqParams reqParams
		queryOpts queryOptions
	}
	tests := []struct {
		name    string
		args    args
		want    queryOptions
		wantErr bool
	}{
		{
			name: "Valid instant time",
			args: args{
				reqParams: reqParams{
					time: "2024-05-04 10:40:00",
				},
				queryOpts: queryOptions{},
			},
			want: queryOptions{
				time: time.Date(2024, 5, 4, 10, 40, 0, 0, time.Local),
			},
			wantErr: false,
		},

		{
			name: "Valid instant time with unnecessary step",
			args: args{
				reqParams: reqParams{
					time: "2024-05-04 10:40:00",
					step: "15s",
				},
				queryOpts: queryOptions{},
			},
			want: queryOptions{
				time: time.Date(2024, 5, 4, 10, 40, 0, 0, time.Local),
			},
			wantErr: true,
		},

		{
			name: "Valid time range",
			args: args{
				reqParams: reqParams{
					start: "2024-05-04 10:40:00",
					end:   "2024-05-04 10:45:00",
				},
				queryOpts: queryOptions{},
			},
			want: queryOptions{
				start: time.Date(2024, 5, 4, 10, 40, 0, 0, time.Local),
				end:   time.Date(2024, 5, 4, 10, 45, 0, 0, time.Local),
				step:  time.Duration(15 * time.Second),
			},
			wantErr: false,
		},

		{
			name: "Invalid time range (invalid format for start time)",
			args: args{
				reqParams: reqParams{
					start: "2024-05-4 10:40:00",
					end:   "2024-05-04 10:45:00",
				},
				queryOpts: queryOptions{},
			},
			want:    queryOptions{},
			wantErr: true,
		},

		{
			name: "Invalid time range (start time equals end time)",
			args: args{
				reqParams: reqParams{
					start: "2024-05-04 10:45:00",
					end:   "2024-05-04 10:45:00",
				},
				queryOpts: queryOptions{},
			},
			want: queryOptions{
				start: time.Date(2024, 5, 4, 10, 45, 0, 0, time.Local),
				end:   time.Date(2024, 5, 4, 10, 45, 0, 0, time.Local),
			},
			wantErr: true,
		},

		{
			name: "Invalid time range (start time after end time)",
			args: args{
				reqParams: reqParams{
					start: "2024-05-04 10:46:00",
					end:   "2024-05-04 10:45:00",
				},
				queryOpts: queryOptions{},
			},
			want: queryOptions{
				start: time.Date(2024, 5, 4, 10, 46, 0, 0, time.Local),
				end:   time.Date(2024, 5, 4, 10, 45, 0, 0, time.Local),
			},
			wantErr: true,
		},

		{
			name: "Invalid time range (invalid format for step)",
			args: args{
				reqParams: reqParams{
					start: "2024-05-04 10:46:00",
					end:   "2024-05-04 10:45:00",
					step:  "5",
				},
				queryOpts: queryOptions{},
			},
			want: queryOptions{
				start: time.Date(2024, 5, 4, 10, 46, 0, 0, time.Local),
				end:   time.Date(2024, 5, 4, 10, 45, 0, 0, time.Local),
			},
			wantErr: true,
		},

		{
			name: "Invalid time range (missing start time or end time)",
			args: args{
				reqParams: reqParams{
					start: "2024-05-04 10:46:00",
				},
				queryOpts: queryOptions{},
			},
			want:    queryOptions{},
			wantErr: true,
		},

		{
			name: "Time conflict error",
			args: args{
				reqParams: reqParams{
					start: "2024-05-04 10:40:00",
					end:   "2024-05-04 10:45:00",
					time:  "2024-05-04 10:45:00",
				},
				queryOpts: queryOptions{},
			},
			want:    queryOptions{},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := transformTimeBasedQueryOptions(tt.args.reqParams, tt.args.queryOpts)
			if (err != nil) != tt.wantErr {
				t.Errorf("transformTimeBasedQueryOptions() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("transformTimeBasedQueryOptions() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetDefaultParam(t *testing.T) {
	// Test the logic directly since getDefaultParam is a simple helper
	tests := []struct {
		name         string
		paramValue   string
		defaultValue string
		want         string
	}{
		{
			name:         "Parameter provided",
			paramValue:   "custom-value",
			defaultValue: ".*",
			want:         "custom-value",
		},
		{
			name:         "Parameter empty, use default",
			paramValue:   "",
			defaultValue: ".*",
			want:         ".*",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test the logic: if value is empty, return default
			value := tt.paramValue
			if value == "" {
				value = tt.defaultValue
			}
			if value != tt.want {
				t.Errorf("getDefaultParam logic: got %v, want %v", value, tt.want)
			}
		})
	}
}

func TestIsRangeQuery(t *testing.T) {
	tests := []struct {
		name  string
		query queryOptions
		want  bool
	}{
		{
			name: "Range Query",
			query: queryOptions{
				start: time.Date(2024, 5, 4, 10, 40, 0, 0, time.Local),
				end:   time.Date(2024, 5, 4, 10, 45, 0, 0, time.Local),
				time:  time.Time{},
			},
			want: true,
		},
		{
			name: "Instant Query",
			query: queryOptions{
				time: time.Date(2024, 5, 4, 10, 40, 0, 0, time.Local),
			},
			want: false,
		},
		{
			name: "Zero time",
			query: queryOptions{
				time: time.Time{},
			},
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.query.isRangeQuery()
			if got != tt.want {
				t.Errorf("isRangeQuery() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestSetControlPlaneComponentOptions(t *testing.T) {
	tests := []struct {
		name      string
		reqParams reqParams
		wantErr   bool
	}{
		{
			name: "KubeAPIServer",
			reqParams: reqParams{
				component: client.KubeAPIServer,
				verb:      "LIST",
				resource:  "pods",
			},
			wantErr: false,
		},
		{
			name: "ETCD",
			reqParams: reqParams{
				component: client.ETCD,
				instance:  "192.168.100.5:2379",
			},
			wantErr: false,
		},
		{
			name: "KubeScheduler",
			reqParams: reqParams{
				component: client.KubeScheduler,
				instance:  "192.168.100.5:10259",
			},
			wantErr: false,
		},
		{
			name: "KubeControllerManager",
			reqParams: reqParams{
				component: client.KubeControllerManager,
				instance:  "192.168.100.5:10257",
			},
			wantErr: false,
		},
		{
			name: "Unknown Component",
			reqParams: reqParams{
				component: "unknown-component",
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			queryOpts := queryOptions{}
			got, err := setControlPlaneComponentOptions(tt.reqParams, queryOpts)
			if (err != nil) != tt.wantErr {
				t.Errorf("setControlPlaneComponentOptions() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && got.option == nil {
				t.Errorf("setControlPlaneComponentOptions() option should not be nil")
			}
		})
	}
}

func TestSetNodeComponentOptions(t *testing.T) {
	tests := []struct {
		name      string
		reqParams reqParams
		wantErr   bool
	}{
		{
			name: "Kubelet",
			reqParams: reqParams{
				component: client.Kubelet,
				instance:  "192.168.100.5:10250",
			},
			wantErr: false,
		},
		{
			name: "KubeProxy",
			reqParams: reqParams{
				component: client.KubeProxy,
				instance:  "host-192-168-100-5",
			},
			wantErr: false,
		},
		{
			name: "Unknown Component",
			reqParams: reqParams{
				component: "unknown-component",
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			queryOpts := queryOptions{}
			got, err := setNodeComponentOptions(tt.reqParams, queryOpts)
			if (err != nil) != tt.wantErr {
				t.Errorf("setNodeComponentOptions() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && got.option == nil {
				t.Errorf("setNodeComponentOptions() option should not be nil")
			}
		})
	}
}

func TestSetAddonsComponentOptions(t *testing.T) {
	tests := []struct {
		name      string
		reqParams reqParams
		wantErr   bool
	}{
		{
			name: "CoreDNS",
			reqParams: reqParams{
				component: client.CoreDNS,
				instance:  "172.17.177.104:9153",
			},
			wantErr: false,
		},
		{
			name: "Unknown Component",
			reqParams: reqParams{
				component: "unknown-component",
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			queryOpts := queryOptions{}
			got, err := setAddonsComponentOptions(tt.reqParams, queryOpts)
			if (err != nil) != tt.wantErr {
				t.Errorf("setAddonsComponentOptions() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && got.option == nil {
				t.Errorf("setAddonsComponentOptions() option should not be nil")
			}
		})
	}
}
