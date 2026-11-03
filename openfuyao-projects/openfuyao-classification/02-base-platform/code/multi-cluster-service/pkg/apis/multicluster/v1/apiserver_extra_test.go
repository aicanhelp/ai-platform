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
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/emicklei/go-restful/v3"
	"k8s.io/client-go/rest"

	k8sclient "openfuyao.com/multi-cluster-service/pkg/apis/multicluster/v1/client/k8s"
	"openfuyao.com/multi-cluster-service/pkg/apis/multicluster/v1/config"
	"openfuyao.com/multi-cluster-service/pkg/apis/multicluster/v1/runtime"
)

func TestAddSecurityHeader(t *testing.T) {
	var got http.Header
	handler := addSecurityHeader(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		got = w.Header()
	}))

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest("GET", "/", nil))

	want := map[string]string{
		"Content-Security-Policy":   "connect-src 'self' https:;frame-ancestors 'none';object-src 'none'",
		"Cache-Control":             "no-cache, no-store, must-revalidate",
		"X-Frame-Options":           "DENY",
		"X-Content-Type-Options":    "nosniff",
		"X-XSS-Protection":          "1",
		"Strict-Transport-Security": "max-age=31536000",
	}
	for key, value := range want {
		if got.Get(key) != value {
			t.Errorf("header %s = %q, want %q", key, got.Get(key), value)
		}
	}
}

func TestSendHellowithRetryExtra(t *testing.T) {
	patches := gomonkey.ApplyMethod(reflect.TypeOf(&APIClient{}), "Post",
		func(_ *APIClient, _ string, _ interface{}, _ string) ([]byte, error) {
			return nil, nil
		})
	defer patches.Reset()

	s := &APIServer{ApiClient: NewAPIClient()}
	done := make(chan struct{})
	go func() {
		s.sendHellowithRetry()
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("sendHellowithRetry did not return after successful post")
	}
}

func TestAPIServerRun(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	_ = listener.Close()

	s := &APIServer{
		Server:    &http.Server{Addr: fmt.Sprintf("127.0.0.1:%d", port)},
		container: restful.NewContainer(),
		MgrClient: newMcsMgrClient(t),
		ApiClient: NewAPIClient(),
	}

	ctx, cancel := context.WithCancel(context.Background())
	errCh := make(chan error, 1)
	go func() {
		errCh <- s.Run(ctx)
	}()

	// wait until the server accepts connections
	deadline := time.Now().Add(5 * time.Second)
	ready := false
	for !ready && time.Now().Before(deadline) {
		conn, err := net.DialTimeout("tcp", fmt.Sprintf("127.0.0.1:%d", port), 200*time.Millisecond)
		if err == nil {
			_ = conn.Close()
			ready = true
		}
	}
	if !ready {
		t.Fatal("api server did not start listening")
	}

	cancel()
	select {
	case err := <-errCh:
		if !errors.Is(err, http.ErrServerClosed) {
			t.Errorf("Run() error = %v, want http.ErrServerClosed", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Run() did not return after context cancel")
	}
}

func TestNewServerExtra(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		cfg := &config.RunConfig{
			Server:        &runtime.ServerConfig{InsecurePort: 18080},
			KubernetesCfg: &k8sclient.KubernetesCfg{KubeConfig: &rest.Config{Host: "127.0.0.1:1"}},
		}

		server, err := NewServer(cfg, context.TODO(), newMcsMgrClient(t))
		if err != nil {
			t.Fatalf("NewServer() error = %v", err)
		}
		if server == nil || server.Server == nil || server.KubernetesClient == nil {
			t.Errorf("server incomplete: %+v", server)
		}
	})

	t.Run("missing kubernetes configuration", func(t *testing.T) {
		// NOTE: passing KubernetesCfg == nil panics inside k8s.NewKubernetesClient
		// (nil parameter deref) - recorded as a product defect, not fixed here.
		cfg := &config.RunConfig{
			Server:        &runtime.ServerConfig{InsecurePort: 18080},
			KubernetesCfg: &k8sclient.KubernetesCfg{KubeConfig: nil},
		}

		if _, err := NewServer(cfg, context.TODO(), newMcsMgrClient(t)); err == nil {
			t.Error("NewServer() expected error when kubernetes config is missing")
		}
	})
}
