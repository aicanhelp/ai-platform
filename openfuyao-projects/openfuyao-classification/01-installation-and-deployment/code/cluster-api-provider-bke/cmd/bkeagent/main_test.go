/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package main

import (
	"context"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/manager"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/testutils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/option"
)

func init() {
}

func TestParseFlags(t *testing.T) {
	tests := []struct {
		name                string
		args                []string
		expectedNtpServer   string
		expectedNtpcron     string
		expectedHealthPort  string
		expectedLogLevel    string
		expectedPlatform    string
		expectedVersion     string
		expectedShowVersion bool
	}{
		{
			name:            "default values",
			args:            []string{},
			expectedNtpcron: "0 */1 * * *",
		},
		{
			name:              "with ntp server",
			args:              []string{"-ntpserver", "ntp.example.com"},
			expectedNtpServer: "ntp.example.com",
			expectedNtpcron:   "0 */1 * * *",
		},
		{
			name:            "with ntp cron",
			args:            []string{"-ntpcron", "0 */2 * * *"},
			expectedNtpcron: "0 */2 * * *",
		},
		{
			name:               "with health port",
			args:               []string{"-health-port", "8080"},
			expectedHealthPort: "8080",
			expectedNtpcron:    "0 */1 * * *",
		},
		{
			name:             "with log level",
			args:             []string{"-log-level", "DEBUG"},
			expectedLogLevel: "DEBUG",
			expectedNtpcron:  "0 */1 * * *",
		},
		{
			name:             "with platform",
			args:             []string{"-platform", "linux"},
			expectedPlatform: "linux",
			expectedNtpcron:  "0 */1 * * *",
		},
		{
			name:            "with version",
			args:            []string{"-version", "v1.0.0"},
			expectedVersion: "v1.0.0",
			expectedNtpcron: "0 */1 * * *",
		},
		{
			name:                "show version",
			args:                []string{"-v"},
			expectedShowVersion: true,
			expectedNtpcron:     "0 */1 * * *",
		},
		{
			name:               "all flags",
			args:               []string{"-ntpserver", "ntp.example.com", "-ntpcron", "0 */3 * * *", "-health-port", "9090", "-log-level", "TRACE", "-platform", "linux", "-version", "v1.0.0"},
			expectedNtpServer:  "ntp.example.com",
			expectedNtpcron:    "0 */3 * * *",
			expectedHealthPort: "9090",
			expectedLogLevel:   "TRACE",
			expectedPlatform:   "linux",
			expectedVersion:    "v1.0.0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original args
			oldArgs := os.Args
			defer func() {
				os.Args = oldArgs
				flag.CommandLine = flag.NewFlagSet(os.Args[0], flag.ExitOnError)
			}()

			// Reset flag.CommandLine to avoid flag redefinition errors
			flag.CommandLine = flag.NewFlagSet(os.Args[0], flag.ContinueOnError)

			// Set up args for this test
			os.Args = append([]string{os.Args[0]}, tt.args...)

			cfg := parseFlags()

			assert.Equal(t, tt.expectedNtpServer, cfg.ntpServer)
			assert.Equal(t, tt.expectedNtpcron, cfg.ntpcron)
			assert.Equal(t, tt.expectedHealthPort, cfg.healthPort)
			assert.Equal(t, tt.expectedLogLevel, cfg.logLevel)
			assert.Equal(t, tt.expectedShowVersion, cfg.showVersion)
			if tt.expectedPlatform != "" {
				assert.Equal(t, tt.expectedPlatform, option.Platform)
			}
			if tt.expectedVersion != "" {
				assert.Equal(t, tt.expectedVersion, option.Version)
			}
		})
	}
}

func TestApplyLogLevel(t *testing.T) {
	// Empty keeps current level; must not panic.
	applyLogLevel("")
	applyLogLevel("   ")

	// Invalid level falls back to default instead of failing.
	applyLogLevel("NOT_A_LEVEL")
	applyLogLevel("WARN")
	applyLogLevel("DEBUG")
	applyLogLevel("warning")
}

func TestValidatePort(t *testing.T) {
	tests := []struct {
		name    string
		port    int
		wantErr bool
	}{
		{
			name:    "valid port in middle range",
			port:    8080,
			wantErr: false,
		},
		{
			name:    "valid port at minimum",
			port:    MinValidPort,
			wantErr: false,
		},
		{
			name:    "valid port at maximum",
			port:    MaxValidPort,
			wantErr: false,
		},
		{
			name:    "port below minimum",
			port:    0,
			wantErr: true,
		},
		{
			name:    "port above maximum",
			port:    65536,
			wantErr: true,
		},
		{
			name:    "negative port",
			port:    -1,
			wantErr: true,
		},
		{
			name:    "reserved port (should warn but not error)",
			port:    80,
			wantErr: false,
		},
		{
			name:    "port at reserved threshold",
			port:    ReservedPortThreshold,
			wantErr: false,
		},
		{
			name:    "port just below reserved threshold",
			port:    ReservedPortThreshold - 1,
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validatePort(tt.port)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestContainsAny(t *testing.T) {
	tests := []struct {
		name       string
		s          string
		substrings []string
		want       bool
	}{
		{
			name:       "contains first substring",
			s:          "address already in use",
			substrings: []string{"address already in use", "bind: address already in use"},
			want:       true,
		},
		{
			name:       "contains second substring",
			s:          "bind: address already in use",
			substrings: []string{"address already in use", "bind: address already in use"},
			want:       true,
		},
		{
			name:       "contains third substring",
			s:          "Only one usage of each socket address",
			substrings: []string{"address already in use", "bind: address already in use", "Only one usage of each socket address"},
			want:       true,
		},
		{
			name:       "does not contain any",
			s:          "some other error",
			substrings: []string{"address already in use", "bind: address already in use"},
			want:       false,
		},
		{
			name:       "empty string",
			s:          "",
			substrings: []string{"address already in use"},
			want:       false,
		},
		{
			name:       "empty substrings",
			s:          "some error",
			substrings: []string{},
			want:       false,
		},
		{
			name:       "case sensitive match",
			s:          "Address Already In Use",
			substrings: []string{"address already in use"},
			want:       false,
		},
		{
			name:       "partial match",
			s:          "address already in use error",
			substrings: []string{"address already in use"},
			want:       true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := containsAny(tt.s, tt.substrings)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestIsPortInUse(t *testing.T) {
	tests := []struct {
		name    string
		setup   func() (int, func())
		want    bool
		wantErr bool
	}{
		{
			name: "port not in use",
			setup: func() (int, func()) {
				// Find an available port
				listener, err := net.Listen("tcp", ":0")
				require.NoError(t, err)
				port := listener.Addr().(*net.TCPAddr).Port
				listener.Close()
				// Wait a bit for port to be released
				time.Sleep(100 * time.Millisecond)
				return port, func() {}
			},
			want:    false,
			wantErr: false,
		},
		{
			name: "port in use by listener",
			setup: func() (int, func()) {
				listener, err := net.Listen("tcp", ":0")
				require.NoError(t, err)
				port := listener.Addr().(*net.TCPAddr).Port
				return port, func() {
					listener.Close()
				}
			},
			want:    true,
			wantErr: false,
		},
		{
			name: "port in use by HTTP server",
			setup: func() (int, func()) {
				listener, err := net.Listen("tcp", ":0")
				require.NoError(t, err)
				port := listener.Addr().(*net.TCPAddr).Port
				listener.Close()

				server := &http.Server{
					Addr: fmt.Sprintf(":%d", port),
				}
				go server.ListenAndServe()
				// Wait for server to start
				time.Sleep(100 * time.Millisecond)

				return port, func() {
					server.Close()
				}
			},
			want:    true,
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			port, cleanup := tt.setup()
			defer cleanup()

			got, err := isPortInUse(port)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

func TestHealthChecker_CheckHealth(t *testing.T) {
	tests := []struct {
		name    string
		setup   func() *healthChecker
		want    bool
		wantMsg string
		wantErr bool
	}{
		{
			name: "context cancelled",
			setup: func() *healthChecker {
				ctx, cancel := context.WithCancel(context.Background())
				cancel()
				return &healthChecker{
					ctx: ctx,
				}
			},
			want:    false,
			wantMsg: "service is shutting down",
		},
		{
			name: "manager not initialized",
			setup: func() *healthChecker {
				return &healthChecker{
					ctx:     context.Background(),
					manager: nil,
				}
			},
			want:    false,
			wantMsg: "manager not initialized",
		},
		{
			name: "rest config not initialized",
			setup: func() *healthChecker {
				// Create a mock manager using testutils to bypass the manager nil check
				// so we can test the restConfig nil case
				mockMgr, _ := testutils.TestGetManagerClient(nil)
				return &healthChecker{
					ctx:        context.Background(),
					manager:    mockMgr, // Use non-nil mock to bypass manager check
					restConfig: nil,
				}
			},
			want:    false,
			wantMsg: "rest config not initialized",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hc := tt.setup()
			healthy, msg := hc.checkHealth()
			assert.Equal(t, tt.want, healthy)
			assert.Contains(t, msg, tt.wantMsg)
		})
	}
}

func TestStartHealthServer(t *testing.T) {
	t.Run("port 0 should not start server", func(t *testing.T) {
		ctx := context.Background()
		hc := &healthChecker{ctx: ctx}
		err := startHealthServer(0, hc)
		assert.NoError(t, err) // port 0 is handled specially, no error
	})

	t.Run("invalid port above maximum", func(t *testing.T) {
		ctx := context.Background()
		hc := &healthChecker{ctx: ctx}
		err := startHealthServer(65536, hc)
		assert.Error(t, err)
	})

	t.Run("port already in use", func(t *testing.T) {
		// Start a server on a random port
		listener, err := net.Listen("tcp", ":0")
		require.NoError(t, err)
		port := listener.Addr().(*net.TCPAddr).Port
		listener.Close()

		server := &http.Server{
			Addr: fmt.Sprintf("127.0.0.1:%d", port),
		}
		go server.ListenAndServe()
		time.Sleep(100 * time.Millisecond)
		defer server.Close()

		ctx := context.Background()
		hc := &healthChecker{ctx: ctx}
		err = startHealthServer(port, hc)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "already in use")
	})

	t.Run("successful start", func(t *testing.T) {
		// Find an available port
		listener, err := net.Listen("tcp", ":0")
		require.NoError(t, err)
		port := listener.Addr().(*net.TCPAddr).Port
		listener.Close()
		time.Sleep(100 * time.Millisecond)

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		hc := &healthChecker{ctx: ctx}
		err = startHealthServer(port, hc)
		require.NoError(t, err)

		// Wait for server to start
		time.Sleep(200 * time.Millisecond)

		// Verify server is running
		resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/healthz", port))
		require.NoError(t, err)
		defer resp.Body.Close()

		// Should return service unavailable since manager is not initialized
		assert.Equal(t, http.StatusServiceUnavailable, resp.StatusCode)
	})
}

func TestHealthServerEndpoint(t *testing.T) {
	// Test that the health check endpoint works correctly
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	hc := &healthChecker{
		ctx: ctx,
	}

	// Find an available port
	listener, err := net.Listen("tcp", ":0")
	require.NoError(t, err)
	port := listener.Addr().(*net.TCPAddr).Port
	listener.Close()
	time.Sleep(100 * time.Millisecond)

	err = startHealthServer(port, hc)
	require.NoError(t, err)

	// Wait for server to start
	time.Sleep(200 * time.Millisecond)

	// Test health endpoint
	resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/healthz", port))
	require.NoError(t, err)
	defer resp.Body.Close()

	// Should return service unavailable since manager is not initialized
	assert.Equal(t, http.StatusServiceUnavailable, resp.StatusCode)
}

func TestNewManager(t *testing.T) {
	patches := gomonkey.ApplyFunc(ctrl.GetConfigOrDie, func() *rest.Config {
		return &rest.Config{Host: "https://test-cluster"}
	})
	defer patches.Reset()

	patches.ApplyFunc(ctrl.NewManager, func(config *rest.Config, options ctrl.Options) (manager.Manager, error) {
		return nil, fmt.Errorf("mock error")
	})

	_, err := newManager()
	assert.Error(t, err)
}

func TestSetupController(t *testing.T) {
	patches := gomonkey.ApplyFunc(utils.HostName, func() string {
		return "test-node"
	})
	defer patches.Reset()

	mockMgr, _ := testutils.TestGetManagerClient(nil)
	j := job.Job{}
	err := setupController(mockMgr, j, context.Background())
	assert.Error(t, err)
}

func TestRun(t *testing.T) {
	patches := gomonkey.ApplyFunc(enableCrdHasInstalled, func() error {
		return fmt.Errorf("crd install error")
	})
	defer patches.Reset()

	cfg := config{
		ntpServer:  "ntp.example.com",
		ntpcron:    "0 */1 * * *",
		healthPort: "8080",
	}

	run(cfg)
}

func TestRunWithHealthPortError(t *testing.T) {
	patches := gomonkey.ApplyFunc(enableCrdHasInstalled, func() error {
		return nil
	})
	defer patches.Reset()

	patches.ApplyFunc(ctrl.SetupSignalHandler, func() context.Context {
		return context.Background()
	})

	patches.ApplyFunc(os.Exit, func(code int) {
		panic(fmt.Sprintf("os.Exit(%d)", code))
	})

	cfg := config{
		healthPort: "invalid",
	}

	assert.Panics(t, func() {
		run(cfg)
	})
}

func TestRunWithManagerError(t *testing.T) {
	patches := gomonkey.ApplyFunc(enableCrdHasInstalled, func() error {
		return nil
	})
	defer patches.Reset()

	patches.ApplyFunc(ctrl.SetupSignalHandler, func() context.Context {
		return context.Background()
	})

	patches.ApplyFunc(newManager, func() (ctrl.Manager, error) {
		return nil, fmt.Errorf("manager error")
	})

	patches.ApplyFunc(os.Exit, func(code int) {
		panic(fmt.Sprintf("os.Exit(%d)", code))
	})

	cfg := config{
		healthPort: "",
	}

	assert.Panics(t, func() {
		run(cfg)
	})
}

func TestRunWithJobError(t *testing.T) {
	patches := gomonkey.ApplyFunc(enableCrdHasInstalled, func() error {
		return nil
	})
	defer patches.Reset()

	patches.ApplyFunc(ctrl.SetupSignalHandler, func() context.Context {
		return context.Background()
	})

	mockMgr, _ := testutils.TestGetManagerClient(nil)
	patches.ApplyFunc(newManager, func() (ctrl.Manager, error) {
		return mockMgr, nil
	})

	patches.ApplyFunc(job.NewJob, func(client interface{}) (job.Job, error) {
		return job.Job{}, fmt.Errorf("job error")
	})

	patches.ApplyFunc(os.Exit, func(code int) {
		panic(fmt.Sprintf("os.Exit(%d)", code))
	})

	cfg := config{
		healthPort: "",
	}

	assert.Panics(t, func() {
		run(cfg)
	})
}

func TestRunWithControllerError(t *testing.T) {
	patches := gomonkey.ApplyFunc(enableCrdHasInstalled, func() error {
		return nil
	})
	defer patches.Reset()

	patches.ApplyFunc(ctrl.SetupSignalHandler, func() context.Context {
		return context.Background()
	})

	mockMgr, _ := testutils.TestGetManagerClient(nil)
	patches.ApplyFunc(newManager, func() (ctrl.Manager, error) {
		return mockMgr, nil
	})

	patches.ApplyFunc(job.NewJob, func(client interface{}) (job.Job, error) {
		return job.Job{}, nil
	})

	patches.ApplyFunc(setupController, func(mgr ctrl.Manager, j job.Job, ctx context.Context) error {
		return fmt.Errorf("controller error")
	})

	patches.ApplyFunc(os.Exit, func(code int) {
		panic(fmt.Sprintf("os.Exit(%d)", code))
	})

	cfg := config{
		healthPort: "",
	}

	assert.Panics(t, func() {
		run(cfg)
	})
}

func TestMain_ShowVersion(t *testing.T) {
	oldArgs := os.Args
	defer func() { os.Args = oldArgs }()

	os.Args = []string{"cmd", "-v"}
	flag.CommandLine = flag.NewFlagSet(os.Args[0], flag.ExitOnError)

	patches := gomonkey.ApplyFunc(os.Exit, func(code int) {
		panic(fmt.Sprintf("os.Exit(%d)", code))
	})
	defer patches.Reset()

	assert.Panics(t, func() {
		main()
	})
}

func TestHealthChecker_CheckHealthWithClient(t *testing.T) {
	mockMgr, _ := testutils.TestGetManagerClient(nil)
	restConfig := &rest.Config{Host: "https://test-cluster"}

	hc := &healthChecker{
		ctx:        context.Background(),
		manager:    mockMgr,
		restConfig: restConfig,
	}

	healthy, msg := hc.checkHealth()
	assert.False(t, healthy)
	assert.Contains(t, msg, "failed")
}

func TestStartHealthServerWithPort0(t *testing.T) {
	ctx := context.Background()
	hc := &healthChecker{ctx: ctx}
	err := startHealthServer(-1, hc)
	assert.NoError(t, err)
}

func TestIsPortInUseWithInvalidPort(t *testing.T) {
	inUse, err := isPortInUse(99999)
	assert.Error(t, err)
	assert.False(t, inUse)
}

func TestRunWithHealthPortZero(t *testing.T) {
	patches := gomonkey.ApplyFunc(enableCrdHasInstalled, func() error {
		return nil
	})
	defer patches.Reset()

	patches.ApplyFunc(ctrl.SetupSignalHandler, func() context.Context {
		return context.Background()
	})

	mockMgr, _ := testutils.TestGetManagerClient(nil)
	patches.ApplyFunc(newManager, func() (ctrl.Manager, error) {
		return mockMgr, nil
	})

	patches.ApplyFunc(job.NewJob, func(client interface{}) (job.Job, error) {
		return job.Job{}, nil
	})

	patches.ApplyFunc(setupController, func(mgr ctrl.Manager, j job.Job, ctx context.Context) error {
		return nil
	})

	patches.ApplyMethod(mockMgr, "Start", func(_ interface{}, _ context.Context) error {
		return fmt.Errorf("manager start error")
	})

	patches.ApplyFunc(os.Exit, func(code int) {
		panic(fmt.Sprintf("os.Exit(%d)", code))
	})

	cfg := config{
		healthPort: "0",
	}

	assert.Panics(t, func() {
		run(cfg)
	})
}

func TestRunWithStartHealthServerError(t *testing.T) {
	patches := gomonkey.ApplyFunc(enableCrdHasInstalled, func() error {
		return nil
	})
	defer patches.Reset()

	patches.ApplyFunc(ctrl.SetupSignalHandler, func() context.Context {
		return context.Background()
	})

	patches.ApplyFunc(startHealthServer, func(port int, hc *healthChecker) error {
		return fmt.Errorf("port in use")
	})

	patches.ApplyFunc(os.Exit, func(code int) {
		panic(fmt.Sprintf("os.Exit(%d)", code))
	})

	cfg := config{
		healthPort: "8080",
	}

	assert.Panics(t, func() {
		run(cfg)
	})
}
