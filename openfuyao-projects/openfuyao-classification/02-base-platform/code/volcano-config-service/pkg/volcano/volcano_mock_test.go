//go:build linux

/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package volcano

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	"volcano-config-service/pkg/constant"
	"volcano-config-service/pkg/controller"
)

type mockAPIServer struct {
	mu          sync.Mutex
	configMaps  map[string]*v1.ConfigMap
	nodeConfig  *controller.NodeConfig
	oenuma      *controller.Oenuma
	nodes       *v1.NodeList
	pods        *v1.PodList
}

func newMockAPIServer() *mockAPIServer {
	return &mockAPIServer{
		configMaps: map[string]*v1.ConfigMap{
			constant.VolcanoConfigServiceDefaultNamespace + "/" + constant.VolcanoConfigServiceConfigmap: {
				ObjectMeta: metav1.ObjectMeta{
					Name:      constant.VolcanoConfigServiceConfigmap,
					Namespace: constant.VolcanoConfigServiceDefaultNamespace,
				},
				Data: map[string]string{
					constant.VolcanoConfigServiceConfigmapName: `actions: "enqueue, allocate, backfill"
tiers:
- plugins:
  - name: priority
  - name: binpack`,
				},
			},
		},
		nodeConfig: &controller.NodeConfig{
			ObjectMeta: metav1.ObjectMeta{Name: "kubelet-config", Namespace: "default"},
			Spec: controller.NodeConfigSpec{
				Nodes: []controller.NodeConfiguration{
					{NodeName: "test-node", EnableStaticCPUManager: true, EnableNUMADistance: false},
				},
			},
		},
		oenuma: &controller.Oenuma{
			ObjectMeta: metav1.ObjectMeta{Name: "podafi", Namespace: "tcs"},
			Spec:       controller.OenumaSpec{Updateenable: "enable"},
		},
		nodes: &v1.NodeList{
			Items: []v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "node1"},
					Status: v1.NodeStatus{
						NodeInfo: v1.NodeSystemInfo{KubeletVersion: "v1.28.0"},
					},
				},
			},
		},
		pods: &v1.PodList{
			Items: []v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "volcano-scheduler-0",
						Namespace: constant.VolcanoConfigServiceDefaultNamespace,
						UID:       types.UID("test-uid"),
					},
					Spec: v1.PodSpec{SchedulerName: "volcano"},
				},
			},
		},
	}
}

func (s *mockAPIServer) handler() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("/api/v1/", func(w http.ResponseWriter, r *http.Request) {
		s.mu.Lock()
		defer s.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		path := strings.TrimPrefix(r.URL.Path, "/api/v1/")
		parts := strings.Split(path, "/")

		if len(parts) == 1 && parts[0] == "nodes" && r.Method == "GET" {
			json.NewEncoder(w).Encode(s.nodes)
			return
		}
		if len(parts) == 1 && parts[0] == "pods" && r.Method == "GET" {
			json.NewEncoder(w).Encode(s.pods)
			return
		}
		if len(parts) >= 4 && parts[2] == "configmaps" {
			key := parts[1] + "/" + parts[3]
			switch r.Method {
			case "GET":
				if cm, ok := s.configMaps[key]; ok {
					json.NewEncoder(w).Encode(cm)
				} else {
					w.WriteHeader(http.StatusNotFound)
				}
			case "POST":
				var cm v1.ConfigMap
				json.NewDecoder(r.Body).Decode(&cm)
				s.configMaps[parts[1]+"/"+cm.Name] = &cm
				json.NewEncoder(w).Encode(cm)
			case "PUT":
				var cm v1.ConfigMap
				json.NewDecoder(r.Body).Decode(&cm)
				s.configMaps[key] = &cm
				json.NewEncoder(w).Encode(cm)
			default:
				w.WriteHeader(http.StatusMethodNotAllowed)
			}
			return
		}
		if len(parts) >= 4 && parts[2] == "pods" {
			switch r.Method {
			case "GET":
				json.NewEncoder(w).Encode(s.pods)
			case "DELETE":
				w.WriteHeader(http.StatusOK)
				w.Write([]byte(`{"status":"Success"}`))
			default:
				w.WriteHeader(http.StatusMethodNotAllowed)
			}
			return
		}
		w.WriteHeader(http.StatusNotFound)
	})

	mux.HandleFunc("/apis/", func(w http.ResponseWriter, r *http.Request) {
		s.mu.Lock()
		defer s.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		path := r.URL.Path

		if strings.Contains(path, "nodeconfigs") && r.Method == "GET" {
			json.NewEncoder(w).Encode(s.nodeConfig)
			return
		}
		if strings.Contains(path, "nodeconfigs") && r.Method == "PUT" {
			var nc controller.NodeConfig
			json.NewDecoder(r.Body).Decode(&nc)
			s.nodeConfig = &nc
			json.NewEncoder(w).Encode(nc)
			return
		}
		if strings.Contains(path, "oenumas") && r.Method == "GET" {
			json.NewEncoder(w).Encode(s.oenuma)
			return
		}
		if strings.Contains(path, "oenumas") && r.Method == "PUT" {
			var oe controller.Oenuma
			json.NewDecoder(r.Body).Decode(&oe)
			s.oenuma = &oe
			json.NewEncoder(w).Encode(oe)
			return
		}
		w.WriteHeader(http.StatusNotFound)
	})

	return mux
}

func newMockVolcanoClient(t *testing.T) *volcanoClient {
	t.Helper()
	apiServer := newMockAPIServer()
	server := httptest.NewServer(apiServer.handler())
	t.Cleanup(server.Close)

	config := &rest.Config{Host: server.URL}
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		t.Fatalf("Failed to create clientset: %v", err)
	}
	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		t.Fatalf("Failed to create dynamic client: %v", err)
	}
	return &volcanoClient{
		kubeConfig:    config,
		dynamicClient: dynamicClient,
		clientset:     clientset,
	}
}

func TestMockMethodNumaAwareGet(t *testing.T) {
	v := newMockVolcanoClient(t)
	resp, code := v.MethodNumaAwareGet()
	assert.NotNil(t, resp)
	assert.NotEqual(t, 0, code)
}

func TestMockMethodNumaAwarePut(t *testing.T) {
	v := newMockVolcanoClient(t)
	resp, code := v.MethodNumaAwarePut("open")
	assert.NotNil(t, resp)
	_ = code
}

func TestMockMethodNumaAwarePut_Close(t *testing.T) {
	v := newMockVolcanoClient(t)
	resp, code := v.MethodNumaAwarePut("close")
	assert.NotNil(t, resp)
	_ = code
}

func TestMockMethodNumaDistanceGet(t *testing.T) {
	v := newMockVolcanoClient(t)
	resp, code := v.MethodNumaDistanceGet()
	assert.NotNil(t, resp)
	_ = code
}

func TestMockMethodNumaDistancePut_Open(t *testing.T) {
	v := newMockVolcanoClient(t)
	resp, code := v.MethodNumaDistancePut("open")
	assert.NotNil(t, resp)
	_ = code
}

func TestMockMethodNumaDistancePut_Close(t *testing.T) {
	v := newMockVolcanoClient(t)
	resp, code := v.MethodNumaDistancePut("close")
	assert.NotNil(t, resp)
	_ = code
}

func TestMockMethodNumaFastGet(t *testing.T) {
	v := newMockVolcanoClient(t)
	resp, code := v.MethodNumaFastGet()
	assert.NotNil(t, resp)
	assert.Equal(t, 200, code)
}

func TestMockMethodNumaFastPut(t *testing.T) {
	v := newMockVolcanoClient(t)
	resp, code := v.MethodNumaFastPut("open")
	assert.NotNil(t, resp)
	_ = code
}

func TestMockMethodNumaFastPut_Close(t *testing.T) {
	v := newMockVolcanoClient(t)
	resp, code := v.MethodNumaFastPut("close")
	assert.NotNil(t, resp)
	_ = code
}

func TestMockMethodBinpackGet(t *testing.T) {
	v := newMockVolcanoClient(t)
	resp, code := v.MethodBinpackGet()
	assert.NotNil(t, resp)
	_ = code
}

func TestMockMethodBinpackPut(t *testing.T) {
	v := newMockVolcanoClient(t)
	resp, code := v.MethodBinpackPut("open")
	assert.NotNil(t, resp)
	_ = code
}

func TestMockMethodDrfGet(t *testing.T) {
	v := newMockVolcanoClient(t)
	resp, code := v.MethodDrfGet()
	assert.NotNil(t, resp)
	_ = code
}

func TestMockMethodGangGet(t *testing.T) {
	v := newMockVolcanoClient(t)
	resp, code := v.MethodGangGet()
	assert.NotNil(t, resp)
	_ = code
}

func TestMockMethodPriorityGet(t *testing.T) {
	v := newMockVolcanoClient(t)
	resp, code := v.MethodPriorityGet()
	assert.NotNil(t, resp)
	_ = code
}

func TestMockMethodBalanceGet(t *testing.T) {
	v := newMockVolcanoClient(t)
	resp, code := v.MethodBalanceGet()
	assert.NotNil(t, resp)
	_ = code
}

func TestMockMethodDefault(t *testing.T) {
	v := newMockVolcanoClient(t)
	resp, code := v.MethodDefault()
	assert.NotNil(t, resp)
	_ = code
}

func TestMockMethodNodeList(t *testing.T) {
	v := newMockVolcanoClient(t)
	resp, code := v.MethodNodeList()
	assert.NotNil(t, resp)
	_ = code
}

func TestMockMethodNodeListByName(t *testing.T) {
	v := newMockVolcanoClient(t)
	resp, code := v.MethodNodeListByName("node1")
	assert.NotNil(t, resp)
	_ = code
}

func TestMockMethodCRInfo(t *testing.T) {
	v := newMockVolcanoClient(t)
	resp, code := v.MethodCRInfo("test-cr")
	assert.NotNil(t, resp)
	_ = code
}

func TestMockMethodUuidToName(t *testing.T) {
	v := newMockVolcanoClient(t)
	resp, code := v.MethodUuidToName("test-uid")
	assert.NotNil(t, resp)
	_ = code
}

func TestMockMethodSchedulerGet(t *testing.T) {
	v := newMockVolcanoClient(t)
	resp, code := v.MethodSchedulerGet()
	assert.NotNil(t, resp)
	_ = code
}

func TestMockCheckKubeletSupport(t *testing.T) {
	v := newMockVolcanoClient(t)
	version, supported, err := checkKubeletSupport(v.clientset)
	_ = err
	_ = version
	_ = supported
}
