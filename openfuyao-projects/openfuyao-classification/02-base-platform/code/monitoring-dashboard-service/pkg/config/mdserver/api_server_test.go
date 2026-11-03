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

package mdserver

import (
	"context"
	"errors"
	"net/http"
	"testing"
	"time"

	"k8s.io/client-go/rest"

	"monitoring-dashboard-service/pkg/config/k8s"
	"monitoring-dashboard-service/pkg/config/runtime"
)

func newTestRunConfig() *runtime.RunConfig {
	return &runtime.RunConfig{
		Server: &runtime.ServerConfig{
			BindAddress:  "0.0.0.0",
			InsecurePort: 0, // 随机空闲端口
		},
		KubernetesCfg: &k8s.KubernetesCfg{
			KubeConfig: &rest.Config{Host: "http://127.0.0.1:6443"},
			QPS:        100,
			Burst:      200,
		},
	}
}

// TestNewAPIServerSuccess 正常创建APIServer
func TestNewAPIServerSuccess(t *testing.T) {
	apiServer, err := NewAPIServer(newTestRunConfig())
	if err != nil {
		t.Fatalf("NewAPIServer failed: %v", err)
	}
	if apiServer.Server == nil {
		t.Error("expected non-nil http server")
	}
	if apiServer.KubernetesClient == nil {
		t.Error("expected non-nil kubernetes client")
	}
	if apiServer.Server.Addr != ":0" {
		t.Errorf("unexpected server addr: %s", apiServer.Server.Addr)
	}
}

// TestNewAPIServerNilKubeConfig 空k8s配置创建失败
func TestNewAPIServerNilKubeConfig(t *testing.T) {
	cfg := newTestRunConfig()
	cfg.KubernetesCfg = &k8s.KubernetesCfg{}
	if _, err := NewAPIServer(cfg); err == nil {
		t.Error("expected error for nil kube config, got nil")
	}
}

// TestAPIServerRunAndShutdown 启动并优雅关闭APIServer
func TestAPIServerRunAndShutdown(t *testing.T) {
	apiServer, err := NewAPIServer(newTestRunConfig())
	if err != nil {
		t.Fatalf("NewAPIServer failed: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	result := make(chan error, 1)
	go func() {
		result <- apiServer.Run(ctx)
	}()

	// 等待server启动后取消context触发优雅关闭
	time.Sleep(200 * time.Millisecond)
	cancel()

	select {
	case err := <-result:
		if !errors.Is(err, http.ErrServerClosed) {
			t.Errorf("expected http.ErrServerClosed, got %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("APIServer.Run did not return after context cancel")
	}
}
