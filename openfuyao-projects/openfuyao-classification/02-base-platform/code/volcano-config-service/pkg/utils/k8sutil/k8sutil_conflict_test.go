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

package k8sutil

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	"volcano-config-service/pkg/controller"
)

// nodeConfigGroupResource 是 NodeConfig 的 GroupResource，用于构造 API 错误
var nodeConfigGroupResource = schema.GroupResource{Group: "volcano.sh", Resource: "nodeconfigs"}

// nodeConfigTestServer 模拟支持 resourceVersion 冲突的 NodeConfig API server。
// conflictsRemaining 表示前 N 次 PUT 返回 409，之后成功。
type nodeConfigTestServer struct {
	mu                 sync.Mutex
	nodeConfig         *controller.NodeConfig
	conflictsRemaining int
	putCalls           int
	server             *httptest.Server
}

func newNodeConfigTestServer(t *testing.T, initial *controller.NodeConfig, conflicts int) (*nodeConfigTestServer, *kubernetes.Clientset) {
	t.Helper()
	s := &nodeConfigTestServer{
		nodeConfig:         initial,
		conflictsRemaining: conflicts,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/apis/volcano.sh/v1/nodeconfigs/kubelet-config", func(w http.ResponseWriter, r *http.Request) {
		s.mu.Lock()
		defer s.mu.Unlock()

		switch r.Method {
		case http.MethodGet:
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(s.nodeConfig)
		case http.MethodPut:
			s.putCalls++
			if s.conflictsRemaining > 0 {
				s.conflictsRemaining--
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusConflict)
				json.NewEncoder(w).Encode(errors.NewConflict(
					nodeConfigGroupResource, "kubelet-config",
					fmt.Errorf("the object has been modified; please apply your changes to the latest version and try again")))
				return
			}
			var nc controller.NodeConfig
			if err := json.NewDecoder(r.Body).Decode(&nc); err != nil {
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			s.nodeConfig = &nc
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(nc)
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
	})

	s.server = httptest.NewServer(mux)
	t.Cleanup(s.server.Close)

	config := &rest.Config{Host: s.server.URL}
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		t.Fatalf("failed to create clientset: %v", err)
	}
	return s, clientset
}

// TestPutNodeConfigWithRetry_ConflictRetries 验证 409 冲突时按预期重试并最终成功
func TestPutNodeConfigWithRetry_ConflictRetries(t *testing.T) {
	initial := &controller.NodeConfig{
		TypeMeta: metav1.TypeMeta{Kind: "NodeConfig", APIVersion: "volcano.sh/v1"},
		ObjectMeta: metav1.ObjectMeta{
			Name:            "kubelet-config",
			Namespace:       "default",
			ResourceVersion: "1",
		},
		Spec: controller.NodeConfigSpec{
			Nodes: []controller.NodeConfiguration{
				{NodeName: "node-a", EnableStaticCPUManager: false},
			},
		},
	}

	// 前 2 次 PUT 返回 409，第 3 次成功
	s, clientset := newNodeConfigTestServer(t, initial, 2)

	err := PutNodeConfigWithRetry(clientset, func(latest *controller.NodeConfig) (*controller.NodeConfig, error) {
		for i := range latest.Spec.Nodes {
			latest.Spec.Nodes[i].EnableStaticCPUManager = true
		}
		return latest, nil
	})
	if err != nil {
		t.Fatalf("expected retry to succeed, got error: %v", err)
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.putCalls != 3 {
		t.Errorf("expected 3 PUT calls (2 conflicts + 1 success), got %d", s.putCalls)
	}
	if !s.nodeConfig.Spec.Nodes[0].EnableStaticCPUManager {
		t.Errorf("expected EnableStaticCPUManager=true after retry, got false")
	}
}

// TestPutNodeConfigWithRetry_NonConflictErrorReturnsImmediately 验证非冲突错误直接返回不重试
func TestPutNodeConfigWithRetry_NonConflictErrorReturnsImmediately(t *testing.T) {
	initial := &controller.NodeConfig{
		TypeMeta: metav1.TypeMeta{Kind: "NodeConfig", APIVersion: "volcano.sh/v1"},
		ObjectMeta: metav1.ObjectMeta{
			Name:            "kubelet-config",
			Namespace:       "default",
			ResourceVersion: "1",
		},
	}

	// 通过直接返回错误的 applyFunc 验证立即返回路径
	s, clientset := newNodeConfigTestServer(t, initial, 0)

	expectedErr := fmt.Errorf("apply failed: simulated non-conflict error")
	err := PutNodeConfigWithRetry(clientset, func(latest *controller.NodeConfig) (*controller.NodeConfig, error) {
		return nil, expectedErr
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if err.Error() != expectedErr.Error() {
		t.Errorf("expected error %v, got %v", expectedErr, err)
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.putCalls != 0 {
		t.Errorf("expected 0 PUT calls for apply error, got %d", s.putCalls)
	}
}

// TestPutNodeConfigWithRetry_ExhaustedRetries 验证持续冲突时耗尽重试后返回错误
func TestPutNodeConfigWithRetry_ExhaustedRetries(t *testing.T) {
	initial := &controller.NodeConfig{
		TypeMeta: metav1.TypeMeta{Kind: "NodeConfig", APIVersion: "volcano.sh/v1"},
		ObjectMeta: metav1.ObjectMeta{
			Name:            "kubelet-config",
			Namespace:       "default",
			ResourceVersion: "1",
		},
	}

	// 永远冲突（次数设为远超最大重试数）
	s, clientset := newNodeConfigTestServer(t, initial, 1000)

	err := PutNodeConfigWithRetry(clientset, func(latest *controller.NodeConfig) (*controller.NodeConfig, error) {
		return latest, nil
	})
	if err == nil {
		t.Fatal("expected error after exhausted retries, got nil")
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.putCalls != nodeConfigMaxRetries {
		t.Errorf("expected %d PUT calls (all conflicted), got %d", nodeConfigMaxRetries, s.putCalls)
	}
}

// TestIsConflictError 验证冲突错误识别
func TestIsConflictError(t *testing.T) {
	conflictErr := errors.NewConflict(
		nodeConfigGroupResource, "kubelet-config",
		fmt.Errorf("the object has been modified"))
	if !isConflictError(conflictErr) {
		t.Errorf("expected conflict error to be detected as conflict")
	}

	notFoundErr := errors.NewNotFound(nodeConfigGroupResource, "kubelet-config")
	if isConflictError(notFoundErr) {
		t.Errorf("expected NotFound error to NOT be detected as conflict")
	}

	serverErr := errors.NewInternalError(fmt.Errorf("boom"))
	if isConflictError(serverErr) {
		t.Errorf("expected InternalError to NOT be detected as conflict")
	}

	if isConflictError(nil) {
		t.Errorf("expected nil error to NOT be detected as conflict")
	}
}
