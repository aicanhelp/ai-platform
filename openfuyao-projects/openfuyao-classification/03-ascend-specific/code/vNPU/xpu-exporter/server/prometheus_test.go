/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package server

import (
	"context"
	"crypto/tls"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"

	"huawei.com/vxpu-device-plugin/pkg/log"
	"huawei.com/xpu-exporter/common/limiter"
)

type mockCollectorService struct{}

func (m *mockCollectorService) CreateCollector(_, _ time.Duration) prometheus.Collector {
	return nil
}

func (m *mockCollectorService) Start(ctx context.Context, _ context.CancelFunc) { _ = ctx }
func (m *mockCollectorService) GetName() string                                 { return "mock-collector" }

func newTestExporterServer() *ExporterServer {
	return &ExporterServer{
		Port:           9119,
		Ip:             "127.0.0.1",
		Concurrency:    100,
		LimitIPReq:     "20/1",
		LimitIPConn:    10,
		LimitTotalConn: 200,
	}
}

func setupNewServerPatches() *gomonkey.Patches {
	patches := gomonkey.NewPatches()
	patches.ApplyFunc(net.Listen, func(network, address string) (net.Listener, error) {
		return &mockListener{}, nil
	})
	patches.ApplyFunc(limiter.LimitListener, func(ln net.Listener, _ int, _ int, _ int) (net.Listener, error) {
		return ln, nil
	})
	patches.ApplyFunc(limiter.NewLimitHandler, func(_ http.Handler, _ *limiter.HandlerConfig) (http.Handler, error) {
		return http.DefaultServeMux, nil
	})
	return patches
}

func setupHTTPSVerifyPatches(getCertsFunc func(suffix string) (string, error), loadX509Func func(certFile, keyFile string) (tls.Certificate, error)) *gomonkey.Patches {
	patches := gomonkey.NewPatches()
	patches.ApplyFunc(os.Getenv, func(string) string { return "on" })
	if getCertsFunc != nil {
		patches.ApplyFunc(getCertsFile, getCertsFunc)
	}
	if loadX509Func != nil {
		patches.ApplyFunc(tls.LoadX509KeyPair, loadX509Func)
	}
	return patches
}

func TestVerifyServerParams(t *testing.T) {
	tests := getTestCases()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.setup != nil {
				patches := tt.setup()
				defer patches.Reset()
			}

			err := tt.server.VerifyServerParams()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}

}

func getTestCases() []struct {
	name    string
	server  *ExporterServer
	wantErr bool
	setup   func() *gomonkey.Patches
} {
	return []struct {
		name    string
		server  *ExporterServer
		wantErr bool
		setup   func() *gomonkey.Patches
	}{
		{
			name:   "valid http config",
			server: newTestExporterServer(),
			setup:  func() *gomonkey.Patches { return setupEnv("") },
		},
		{
			name: "invalid port",
			server: func() *ExporterServer {
				s := newTestExporterServer()
				s.Port = 1024
				return s
			}(),
			wantErr: true,
		},
		{
			name: "invalid ip",
			server: func() *ExporterServer {
				s := newTestExporterServer()
				s.Ip = "invalid.ip"
				return s
			}(),
			wantErr: true,
		},
		{
			name:    "https with invalid certs",
			server:  newTestExporterServer(),
			wantErr: true,
			setup:   func() *gomonkey.Patches { return setupHTTPSFailure() },
		},
	}
}

func setupEnv(envValue string) *gomonkey.Patches {
	patches := gomonkey.NewPatches()
	patches.ApplyFunc(os.Getenv, func(string) string { return envValue })
	return patches
}

func setupHTTPSFailure() *gomonkey.Patches {
	patches := setupEnv("on")
	patches.ApplyFunc(os.ReadDir, func(string) ([]os.DirEntry, error) {
		return nil, errors.New("dir error")
	})
	return patches
}

func TestStartServe(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(net.Listen, func(network, address string) (net.Listener, error) {
		return &mockListener{}, nil
	})

	patches.ApplyMethodFunc(&http.Server{}, "Serve", func(l net.Listener) error {
		return nil
	})

	patches.ApplyMethodFunc(&http.Server{}, "ServeTLS", func(l net.Listener, certFile, keyFile string) error {
		return nil
	})

	patches.ApplyFunc(log.Infof, func(format string, args ...interface{}) { _ = format })
	patches.ApplyFunc(log.Errorf, func(format string, args ...interface{}) { _ = format })

	t.Run("start http server", func(t *testing.T) {
		http.DefaultServeMux = http.NewServeMux()
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		s := &ExporterServer{
			Port:         9119,
			Ip:           "127.0.0.1",
			ProtocolType: HTTP,
		}
		reg := prometheus.NewRegistry()
		s.StartServe(ctx, cancel, reg)
	})

	t.Run("start https server", func(t *testing.T) {
		http.DefaultServeMux = http.NewServeMux()
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		patches.ApplyFunc(getCertsFile, func(suffix string) (string, error) {
			return "test.crt", nil
		})

		patches.ApplyFunc(tls.LoadX509KeyPair, func(certFile, keyFile string) (tls.Certificate, error) {
			return tls.Certificate{}, nil
		})

		s := &ExporterServer{
			Port:         9119,
			Ip:           "127.0.0.1",
			ProtocolType: HTTPS,
		}
		reg := prometheus.NewRegistry()
		s.StartServe(ctx, cancel, reg)

	})
}

func TestRegisterCollectorService(t *testing.T) {
	s := &ExporterServer{}
	mockCollector := &mockCollectorService{}

	t.Run("valid registration", func(t *testing.T) {
		err := s.RegisterCollectorService(mockCollector)
		assert.NoError(t, err)
		assert.Equal(t, mockCollector, s.collectService)
	})

	t.Run("nil collector", func(t *testing.T) {
		err := s.RegisterCollectorService(nil)
		assert.Error(t, err)
	})
}

type mockListener struct{}

func (m *mockListener) Accept() (net.Conn, error) { return nil, nil }
func (m *mockListener) Close() error              { return nil }
func (m *mockListener) Addr() net.Addr            { return &net.TCPAddr{} }

func TestIndexHandler_HTTP(t *testing.T) {
	s := &ExporterServer{Port: 9119, ProtocolType: HTTP}
	handler := indexHandler(s)
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	rec := httptest.NewRecorder()
	handler(rec, req)
	body := rec.Body.String()
	assert.Contains(t, body, "http://IP:9119/metrics")
	assert.Contains(t, body, "XPU-Exporter")
}

func TestIndexHandler_HTTPS(t *testing.T) {
	s := &ExporterServer{Port: 9119, ProtocolType: HTTPS}
	handler := indexHandler(s)
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	rec := httptest.NewRecorder()
	handler(rec, req)
	body := rec.Body.String()
	assert.Contains(t, body, "https://IP:9119/metrics")
}

func TestGetTLSConfig(t *testing.T) {
	cfg := getTLSConfig()
	assert.Equal(t, uint16(tls.VersionTLS13), cfg.MinVersion)
	assert.False(t, cfg.InsecureSkipVerify)
}

func TestGetCertsFile_DirReadError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(os.ReadDir, func(string) ([]os.DirEntry, error) {
		return nil, errors.New("dir error")
	})
	_, err := getCertsFile(".crt")
	assert.Error(t, err)
}

func TestGetCertsFile_NoMatchingSuffix(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(os.ReadDir, func(string) ([]os.DirEntry, error) {
		return []os.DirEntry{
			dirEntry{name: "readme.txt", isDir: false},
			dirEntry{name: "subdir", isDir: true},
		}, nil
	})
	_, err := getCertsFile(".crt")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no files with the suffix")
}

func TestGetCertsFile_FoundMatch(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(os.ReadDir, func(string) ([]os.DirEntry, error) {
		return []os.DirEntry{
			dirEntry{name: "server.crt", isDir: false},
			dirEntry{name: "subdir", isDir: true},
		}, nil
	})
	result, err := getCertsFile(".crt")
	assert.NoError(t, err)
	assert.Contains(t, result, "server.crt")
}

type dirEntry struct {
	name  string
	isDir bool
}

func (d dirEntry) Name() string               { return d.name }
func (d dirEntry) IsDir() bool                { return d.isDir }
func (d dirEntry) Type() os.FileMode          { return 0 }
func (d dirEntry) Info() (os.FileInfo, error) { return nil, nil }

func TestInitConfig(t *testing.T) {
	s := &ExporterServer{
		Concurrency: 100,
		LimitIPReq:  "20/1",
	}
	conf := s.initConfig()
	assert.True(t, conf.PrintLog)
	assert.Equal(t, http.MethodGet, conf.Method)
	assert.Equal(t, 100, conf.TotalConCurrency)
	assert.Equal(t, "20/1", conf.IPConCurrency)
}

func TestNewServer_LimiterHandlerError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(limiter.NewLimitHandler, func(_ http.Handler, _ *limiter.HandlerConfig) (http.Handler, error) {
		return nil, errors.New("handler error")
	})
	s := newTestExporterServer()
	conf := s.initConfig()
	_, _, err := s.newServer(conf)
	assert.Error(t, err)
}

func TestNewServer_ListenError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(net.Listen, func(network, address string) (net.Listener, error) {
		return nil, errors.New("listen error")
	})
	s := newTestExporterServer()
	conf := s.initConfig()
	_, _, err := s.newServer(conf)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "listen IP and port error")
}

func TestNewServer_LimitListenerError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(net.Listen, func(network, address string) (net.Listener, error) {
		return &mockListener{}, nil
	})
	patches.ApplyFunc(limiter.LimitListener, func(_ net.Listener, _ int, _ int, _ int) (net.Listener, error) {
		return nil, errors.New("limit error")
	})
	s := newTestExporterServer()
	conf := s.initConfig()
	_, _, err := s.newServer(conf)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "limit listener error")
}

func TestVerifyServerParams_InvalidParams(t *testing.T) {
	tests := []struct {
		name   string
		modify func(s *ExporterServer)
	}{
		{name: "port too high", modify: func(s *ExporterServer) { s.Port = 40001 }},
		{name: "port too low", modify: func(s *ExporterServer) { s.Port = 0 }},
		{name: "limitIPReq invalid format", modify: func(s *ExporterServer) { s.LimitIPReq = "abc" }},
		{name: "limitIPConn zero", modify: func(s *ExporterServer) { s.LimitIPConn = 0 }},
		{name: "limitIPConn exceeds max", modify: func(s *ExporterServer) { s.LimitIPConn = 129 }},
		{name: "limitTotalConn zero", modify: func(s *ExporterServer) { s.LimitTotalConn = 0 }},
		{name: "limitTotalConn exceeds max", modify: func(s *ExporterServer) { s.LimitTotalConn = 513 }},
		{name: "concurrency zero", modify: func(s *ExporterServer) { s.Concurrency = 0 }},
		{name: "concurrency exceeds max", modify: func(s *ExporterServer) { s.Concurrency = 513 }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := setupEnv("")
			defer patches.Reset()
			s := newTestExporterServer()
			tt.modify(s)
			err := s.VerifyServerParams()
			assert.Error(t, err)
		})
	}
}

func TestVerifyServerParams_ValidBoundaryValues(t *testing.T) {
	tests := []struct {
		name   string
		modify func(s *ExporterServer)
	}{
		{name: "valid limitIPConn boundary", modify: func(s *ExporterServer) { s.LimitIPConn = 1 }},
		{name: "valid concurrency boundary", modify: func(s *ExporterServer) { s.Concurrency = 1 }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := setupEnv("")
			defer patches.Reset()
			s := newTestExporterServer()
			tt.modify(s)
			err := s.VerifyServerParams()
			assert.NoError(t, err)
		})
	}
}

func TestCreateCollector(t *testing.T) {
	s := &ExporterServer{collectService: &mockCollectorService{}}
	collector := s.CreateCollector(time.Second, time.Second)
	assert.Nil(t, collector)
}

func TestStartCollect(t *testing.T) {
	s := &ExporterServer{collectService: &mockCollectorService{}}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	s.StartCollect(ctx, cancel)
}

func TestNewServer_HTTPS(t *testing.T) {
	patches := setupNewServerPatches()
	defer patches.Reset()

	s := newTestExporterServer()
	s.ProtocolType = HTTPS
	s.cert = &tls.Certificate{}
	conf := s.initConfig()
	srv, ln, err := s.newServer(conf)
	assert.NoError(t, err)
	assert.NotNil(t, srv)
	assert.NotNil(t, srv.TLSConfig)
	assert.NotNil(t, ln)
}

func TestNewServer_HTTP(t *testing.T) {
	patches := setupNewServerPatches()
	defer patches.Reset()

	s := newTestExporterServer()
	s.ProtocolType = HTTP
	conf := s.initConfig()
	srv, ln, err := s.newServer(conf)
	assert.NoError(t, err)
	assert.NotNil(t, srv)
	assert.Nil(t, srv.TLSConfig)
	assert.NotNil(t, ln)
}

func TestVerifyServerParams_HTTPSSuccess(t *testing.T) {
	patches := setupHTTPSVerifyPatches(
		func(suffix string) (string, error) {
			if suffix == ".crt" {
				return "/opt/xpu/certs/server.crt", nil
			}
			return "/opt/xpu/certs/server.key", nil
		},
		func(certFile, keyFile string) (tls.Certificate, error) {
			return tls.Certificate{}, nil
		},
	)
	defer patches.Reset()

	s := newTestExporterServer()
	err := s.VerifyServerParams()
	assert.NoError(t, err)
	assert.Equal(t, HTTPS, s.ProtocolType)
	assert.NotNil(t, s.cert)
}

func TestVerifyServerParams_HTTPSCertFileError(t *testing.T) {
	patches := setupHTTPSVerifyPatches(
		func(suffix string) (string, error) {
			return "", errors.New("cert not found")
		},
		nil,
	)
	defer patches.Reset()

	s := newTestExporterServer()
	err := s.VerifyServerParams()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "HTTPS certificate error")
}

func TestVerifyServerParams_HTTPSKeyFileError(t *testing.T) {
	patches := setupHTTPSVerifyPatches(
		func(suffix string) (string, error) {
			if suffix == ".crt" {
				return "/opt/xpu/certs/server.crt", nil
			}
			return "", errors.New("key not found")
		},
		nil,
	)
	defer patches.Reset()

	s := newTestExporterServer()
	err := s.VerifyServerParams()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "HTTPS key error")
}

func TestVerifyServerParams_HTTPSKeyPairError(t *testing.T) {
	patches := setupHTTPSVerifyPatches(
		func(suffix string) (string, error) {
			if suffix == ".crt" {
				return "/opt/xpu/certs/server.crt", nil
			}
			return "/opt/xpu/certs/server.key", nil
		},
		func(certFile, keyFile string) (tls.Certificate, error) {
			return tls.Certificate{}, errors.New("key pair error")
		},
	)
	defer patches.Reset()

	s := newTestExporterServer()
	err := s.VerifyServerParams()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "HTTPS key pair error")
}

func TestStartServe_ServerCreateError(t *testing.T) {
	http.DefaultServeMux = http.NewServeMux()
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(limiter.NewLimitHandler, func(_ http.Handler, _ *limiter.HandlerConfig) (http.Handler, error) {
		return nil, errors.New("handler error")
	})
	patches.ApplyFunc(log.Errorf, func(format string, args ...interface{}) { _ = format })

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	s := newTestExporterServer()
	reg := prometheus.NewRegistry()
	s.StartServe(ctx, cancel, reg)
}

func TestIndexHandler_WriteError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(log.Errorf, func(format string, args ...interface{}) { _ = format })

	s := &ExporterServer{Port: 9119, ProtocolType: HTTP}
	handler := s.indexHandler
	rec := &errorRecorder{}
	handler(rec, nil)
	assert.True(t, rec.writeCalled)
}

type errorRecorder struct {
	writeCalled bool
}

func (e *errorRecorder) Header() http.Header { return http.Header{} }
func (e *errorRecorder) Write([]byte) (int, error) {
	e.writeCalled = true
	return 0, errors.New("write error")
}
func (e *errorRecorder) WriteHeader(code int) { _ = code }

func TestNewServer_GetCertificateNil(t *testing.T) {
	patches := setupNewServerPatches()
	defer patches.Reset()
	s := newTestExporterServer()
	s.ProtocolType = HTTPS
	conf := s.initConfig()
	srv, _, err := s.newServer(conf)
	assert.NoError(t, err)
	cert, err := srv.TLSConfig.GetCertificate(nil)
	assert.Error(t, err)
	assert.Nil(t, cert)
}

func TestNewServer_GetCertificateLoaded(t *testing.T) {
	patches := setupNewServerPatches()
	defer patches.Reset()
	s := newTestExporterServer()
	s.ProtocolType = HTTPS
	s.cert = &tls.Certificate{}
	conf := s.initConfig()
	srv, _, err := s.newServer(conf)
	assert.NoError(t, err)
	cert, err := srv.TLSConfig.GetCertificate(nil)
	assert.NoError(t, err)
	assert.NotNil(t, cert)
}

func TestRegisterCollectorService_NilServer(t *testing.T) {
	var s *ExporterServer
	err := s.RegisterCollectorService(&mockCollectorService{})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "s is nil")
}

func runStartServe(t *testing.T, s *ExporterServer) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	reg := prometheus.NewRegistry()
	done := make(chan struct{})
	go func() {
		s.StartServe(ctx, cancel, reg)
		close(done)
	}()
	time.Sleep(100 * time.Millisecond)
	cancel()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("StartServe did not finish within timeout")
	}
}

func TestStartServe_HTTPGoroutineShutdown(t *testing.T) {
	http.DefaultServeMux = http.NewServeMux()
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(net.Listen, func(network, address string) (net.Listener, error) {
		return &mockListener{}, nil
	})
	patches.ApplyMethodFunc(&http.Server{}, "Serve", func(l net.Listener) error {
		return http.ErrServerClosed
	})
	patches.ApplyFunc(log.Infof, func(format string, args ...interface{}) { _ = format })
	patches.ApplyFunc(log.Errorf, func(format string, args ...interface{}) { _ = format })

	s := newTestExporterServer()
	s.ProtocolType = HTTP
	runStartServe(t, s)
}

func TestStartServe_HTTPSGoroutineShutdown(t *testing.T) {
	http.DefaultServeMux = http.NewServeMux()
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(net.Listen, func(network, address string) (net.Listener, error) {
		return &mockListener{}, nil
	})
	patches.ApplyMethodFunc(&http.Server{}, "ServeTLS", func(l net.Listener, certFile, keyFile string) error {
		return http.ErrServerClosed
	})
	patches.ApplyFunc(log.Infof, func(format string, args ...interface{}) { _ = format })
	patches.ApplyFunc(log.Errorf, func(format string, args ...interface{}) { _ = format })

	s := newTestExporterServer()
	s.ProtocolType = HTTPS
	runStartServe(t, s)
}

func TestStartServe_ServerErrorCancel(t *testing.T) {
	http.DefaultServeMux = http.NewServeMux()
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(net.Listen, func(network, address string) (net.Listener, error) {
		return &mockListener{}, nil
	})
	patches.ApplyMethodFunc(&http.Server{}, "Serve", func(l net.Listener) error {
		return errors.New("serve error")
	})
	patches.ApplyFunc(log.Infof, func(format string, args ...interface{}) { _ = format })
	patches.ApplyFunc(log.Errorf, func(format string, args ...interface{}) { _ = format })

	s := newTestExporterServer()
	s.ProtocolType = HTTP
	runStartServe(t, s)
}

func TestStartServe_ShutdownError(t *testing.T) {
	http.DefaultServeMux = http.NewServeMux()
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(net.Listen, func(network, address string) (net.Listener, error) {
		return &mockListener{}, nil
	})
	patches.ApplyMethodFunc(&http.Server{}, "Serve", func(l net.Listener) error {
		return http.ErrServerClosed
	})
	patches.ApplyMethodFunc(&http.Server{}, "Shutdown", func(ctx context.Context) error {
		return errors.New("shutdown error")
	})
	patches.ApplyFunc(log.Infof, func(format string, args ...interface{}) { _ = format })
	patches.ApplyFunc(log.Errorf, func(format string, args ...interface{}) { _ = format })

	s := newTestExporterServer()
	s.ProtocolType = HTTP
	runStartServe(t, s)
}
