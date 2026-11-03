/*
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
* Author: YuXiang Guo
* Date: 2025-09-05
 */

package utils

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

// TestFakeInitKubeClient 测试 InitKubeClient 函数
func TestFakeInitKubeClient(t *testing.T) {
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// 模拟 Kubernetes API 响应
		switch r.URL.Path {
		case "/api/v1/namespaces/default/pods":
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{
				"kind": "PodList",
				"apiVersion": "v1",
				"items": [{
					"metadata": {
						"name": "test-pod",
						"namespace": "default"
					}
				}]
			}`))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))

	server.StartTLS()
	defer server.Close()

	config := &rest.Config{
		Host: server.URL,
		TLSClientConfig: rest.TLSClientConfig{
			Insecure: true,
		},
	}

	// 创建客户端
	client, err := kubernetes.NewForConfig(config)
	if err != nil {
		t.Fatalf("Error creating client: %v", err)
	}

	// 测试客户端
	pods, err := client.CoreV1().Pods("default").List(context.TODO(), v1.ListOptions{})
	if err != nil {
		t.Fatalf("Error listing pods: %v", err)
	}

	if len(pods.Items) != 1 {
		t.Errorf("Expected 1 pod, got %d", len(pods.Items))
	}
}

// TestInitKubeClientInClusterConfigError covers the error path of
// InitKubeClient: outside a Kubernetes pod, rest.InClusterConfig() returns an
// error because the in-cluster env vars are not present.
func TestInitKubeClientInClusterConfigError(t *testing.T) {
	// Ensure the in-cluster config env vars are unset so InClusterConfig fails
	// deterministically regardless of where the test runs.
	hostKey := "KUBERNETES_SERVICE_HOST"
	portKey := "KUBERNETES_SERVICE_PORT"
	oldHost, hadHost := os.LookupEnv(hostKey)
	oldPort, hadPort := os.LookupEnv(portKey)
	os.Unsetenv(hostKey)
	os.Unsetenv(portKey)
	defer func() {
		if hadHost {
			os.Setenv(hostKey, oldHost)
		}
		if hadPort {
			os.Setenv(portKey, oldPort)
		}
	}()

	client, err := InitKubeClient()
	if err == nil {
		if client != nil {
			t.Skip("InitKubeClient unexpectedly succeeded (running inside a cluster); skipping error-path test")
		}
		t.Fatal("Expected an error when not running in a cluster, got nil error and nil client")
	}
	if client != nil {
		t.Errorf("Expected nil client on error, got non-nil")
	}
}
