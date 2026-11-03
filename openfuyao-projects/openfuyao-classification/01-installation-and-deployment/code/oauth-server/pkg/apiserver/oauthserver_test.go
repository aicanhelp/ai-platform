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

package apiserver

import (
	"context"
	"crypto/tls"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/dynamic"
	dynamicfake "k8s.io/client-go/dynamic/fake"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"

	overallconfigs "openfuyao/oauth-server/cmd/oauth-server/app/config"
	"openfuyao/oauth-server/pkg/config"
	"openfuyao/oauth-server/pkg/constants"
	"openfuyao/oauth-server/pkg/fuyaoerrors"
	"openfuyao/oauth-server/pkg/httpserver"
	"openfuyao/oauth-server/pkg/idp/fuyaopassword"
	"openfuyao/oauth-server/pkg/oauth2"
)

func createTestConfig() *overallconfigs.OAuthServerAPIServerConfig {
	return &overallconfigs.OAuthServerAPIServerConfig{
		HttpServerConfig: &httpserver.ServerOptions{
			HttpPort:  9096,
			HttpsPort: 0,
		},
		K8sConfig: config.NewKubernetesConfig(),
		IDPLoginStoreConfig: &overallconfigs.IDPLoginStoreConfig{
			SessionName:    "idpLogin",
			SessionMaxAge:  300,
			CsrfCookieName: "csrf",
			SigningKey:     []byte("signing-key"),
			EncryptionKey:  []byte("encryption-key-123456789012345678901234567890"),
		},
		IPProtectorConfig: overallconfigs.NewDefaultOAuthAPIServerServerConfig().IPProtectorConfig,
		LoginConfig:       overallconfigs.NewDefaultOAuthAPIServerServerConfig().LoginConfig,
		OAuthServerConfig: &overallconfigs.OAuthServerConfig{
			CodeTokenNamespace: "oauth-code-token",
			AuthCodeExp:        time.Minute * 5,
			AccessTokenExp:     time.Hour * 2,
			RefreshTokenExp:    time.Hour * 2,
			IsGenerateRefresh:  false,
			JWTKeyID:           "access_token_sign_key",
			JWTPrivateKey:      []byte("jwt-private-key"),
			ClientMapper: map[string]string{
				"console": "console-password",
			},
		},
	}
}

func TestNewOAuthServerAPIServer(t *testing.T) {
	tests := []struct {
		name    string
		cfg     *overallconfigs.OAuthServerAPIServerConfig
		wantErr bool
		setup   func() *gomonkey.Patches
	}{
		{
			name:    "successfully create OAuthServerAPIServer",
			cfg:     createTestConfig(),
			wantErr: false,
			setup: func() *gomonkey.Patches {
				patches := gomonkey.NewPatches()
				scheme := runtime.NewScheme()
				fakeK8sClient := fake.NewSimpleClientset()
				fakeDynamicClient := dynamicfake.NewSimpleDynamicClient(scheme)

				patches.ApplyFunc(httpserver.NewHttpServer, func(_ *httpserver.ServerOptions) (*http.Server, error) {
					return &http.Server{Addr: ":9096"}, nil
				})

				patches.ApplyFunc(config.GetKubernetesClient, func(_ *config.KubernetesConfig) kubernetes.Interface {
					return fakeK8sClient
				})

				patches.ApplyFunc(config.GetDynamicClient, func(_ *config.KubernetesConfig) dynamic.Interface {
					return fakeDynamicClient
				})

				return patches
			},
		},
		{
			name:    "fail to create http server",
			cfg:     createTestConfig(),
			wantErr: true,
			setup: func() *gomonkey.Patches {
				patches := gomonkey.NewPatches()
				patches.ApplyFunc(httpserver.NewHttpServer, func(_ *httpserver.ServerOptions) (*http.Server, error) {
					return nil, fuyaoerrors.ErrFailToLoadCert
				})
				return patches
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var patches *gomonkey.Patches
			if tt.setup != nil {
				patches = tt.setup()
				defer patches.Reset()
			}

			stopCh := make(chan struct{})
			server, err := NewOAuthServerAPIServer(tt.cfg, stopCh)

			if tt.wantErr {
				assert.Error(t, err)
				assert.Nil(t, server)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, server)
				assert.NotNil(t, server.Server)
				assert.NotNil(t, server.Router)
				assert.NotNil(t, server.Login)
				assert.NotNil(t, server.OAuthServer)
				assert.Equal(t, tt.cfg, server.Cfg)
			}
		})
	}
}

func TestOAuthServerAPIServer_PrepareRun(t *testing.T) {
	tests := []struct {
		name    string
		cfg     *overallconfigs.OAuthServerAPIServerConfig
		wantErr bool
		setup   func() *gomonkey.Patches
	}{
		{
			name:    "successfully prepare run with default csrf cookie name",
			cfg:     createTestConfig(),
			wantErr: false,
			setup: func() *gomonkey.Patches {
				patches := gomonkey.NewPatches()
				scheme := runtime.NewScheme()
				fakeK8sClient := fake.NewSimpleClientset()
				fakeDynamicClient := dynamicfake.NewSimpleDynamicClient(scheme)

				patches.ApplyFunc(httpserver.NewHttpServer, func(_ *httpserver.ServerOptions) (*http.Server, error) {
					return &http.Server{Addr: ":9096"}, nil
				})

				patches.ApplyFunc(config.GetKubernetesClient, func(_ *config.KubernetesConfig) kubernetes.Interface {
					return fakeK8sClient
				})

				patches.ApplyFunc(config.GetDynamicClient, func(_ *config.KubernetesConfig) dynamic.Interface {
					return fakeDynamicClient
				})

				return patches
			},
		},
		{
			name: "successfully prepare run with custom csrf cookie name",
			cfg: func() *overallconfigs.OAuthServerAPIServerConfig {
				cfg := createTestConfig()
				cfg.IDPLoginStoreConfig.CsrfCookieName = "custom-csrf"
				return cfg
			}(),
			wantErr: false,
			setup: func() *gomonkey.Patches {
				patches := gomonkey.NewPatches()
				scheme := runtime.NewScheme()
				fakeK8sClient := fake.NewSimpleClientset()
				fakeDynamicClient := dynamicfake.NewSimpleDynamicClient(scheme)

				patches.ApplyFunc(httpserver.NewHttpServer, func(_ *httpserver.ServerOptions) (*http.Server, error) {
					return &http.Server{Addr: ":9096"}, nil
				})

				patches.ApplyFunc(config.GetKubernetesClient, func(_ *config.KubernetesConfig) kubernetes.Interface {
					return fakeK8sClient
				})

				patches.ApplyFunc(config.GetDynamicClient, func(_ *config.KubernetesConfig) dynamic.Interface {
					return fakeDynamicClient
				})

				return patches
			},
		},
		{
			name: "successfully prepare run with empty csrf cookie name",
			cfg: func() *overallconfigs.OAuthServerAPIServerConfig {
				cfg := createTestConfig()
				cfg.IDPLoginStoreConfig.CsrfCookieName = ""
				return cfg
			}(),
			wantErr: false,
			setup: func() *gomonkey.Patches {
				patches := gomonkey.NewPatches()
				scheme := runtime.NewScheme()
				fakeK8sClient := fake.NewSimpleClientset()
				fakeDynamicClient := dynamicfake.NewSimpleDynamicClient(scheme)

				patches.ApplyFunc(httpserver.NewHttpServer, func(_ *httpserver.ServerOptions) (*http.Server, error) {
					return &http.Server{Addr: ":9096"}, nil
				})

				patches.ApplyFunc(config.GetKubernetesClient, func(_ *config.KubernetesConfig) kubernetes.Interface {
					return fakeK8sClient
				})

				patches.ApplyFunc(config.GetDynamicClient, func(_ *config.KubernetesConfig) dynamic.Interface {
					return fakeDynamicClient
				})

				return patches
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var patches *gomonkey.Patches
			if tt.setup != nil {
				patches = tt.setup()
				defer patches.Reset()
			}

			stopCh := make(chan struct{})
			server, err := NewOAuthServerAPIServer(tt.cfg, stopCh)
			if err != nil {
				t.Fatalf("Failed to create server: %v", err)
			}

			err = server.PrepareRun(stopCh)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)

				// Verify routes are registered
				testRouter := server.Router
				assert.NotNil(t, testRouter)

				// Test login endpoint
				req := httptest.NewRequest("GET", constants.FuyaoLoginEndpoint, nil)
				rr := httptest.NewRecorder()
				testRouter.ServeHTTP(rr, req)
				// Should not return 404, meaning route is registered
				assert.NotEqual(t, http.StatusNotFound, rr.Code, "Login endpoint should be registered")

				// Test logout endpoint
				req = httptest.NewRequest("POST", constants.FuyaoLogoutEndpoint, nil)
				rr = httptest.NewRecorder()
				testRouter.ServeHTTP(rr, req)
				assert.NotEqual(t, http.StatusNotFound, rr.Code, "Logout endpoint should be registered")

				// Test password confirm endpoint
				req = httptest.NewRequest("POST", constants.FuyaoPasswordConfirmEndpoint, nil)
				rr = httptest.NewRecorder()
				testRouter.ServeHTTP(rr, req)
				assert.NotEqual(t, http.StatusNotFound, rr.Code, "Password confirm endpoint should be registered")

				// Test password modify endpoint
				req = httptest.NewRequest("POST", constants.FuyaoPasswordModifyEndpoint, nil)
				rr = httptest.NewRecorder()
				testRouter.ServeHTTP(rr, req)
				assert.NotEqual(t, http.StatusNotFound, rr.Code, "Password modify endpoint should be registered")

				// Test oauth authorize endpoint
				req = httptest.NewRequest("GET", constants.FuyaoOAuthAuthorizeEndpoint, nil)
				rr = httptest.NewRecorder()
				testRouter.ServeHTTP(rr, req)
				assert.NotEqual(t, http.StatusNotFound, rr.Code, "OAuth authorize endpoint should be registered")

				// Test oauth token endpoint
				req = httptest.NewRequest("POST", constants.FuyaoOAuthTokenEndpoint, nil)
				rr = httptest.NewRecorder()
				testRouter.ServeHTTP(rr, req)
				assert.NotEqual(t, http.StatusNotFound, rr.Code, "OAuth token endpoint should be registered")
			}
		})
	}
}

func TestOAuthServerAPIServer_PrepareRun_CSRFErrorHandler(t *testing.T) {
	cfg := createTestConfig()
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	scheme := runtime.NewScheme()
	fakeK8sClient := fake.NewSimpleClientset()
	fakeDynamicClient := dynamicfake.NewSimpleDynamicClient(scheme)

	patches.ApplyFunc(httpserver.NewHttpServer, func(_ *httpserver.ServerOptions) (*http.Server, error) {
		return &http.Server{Addr: ":9096"}, nil
	})

	patches.ApplyFunc(config.GetKubernetesClient, func(_ *config.KubernetesConfig) kubernetes.Interface {
		return fakeK8sClient
	})

	patches.ApplyFunc(config.GetDynamicClient, func(_ *config.KubernetesConfig) dynamic.Interface {
		return fakeDynamicClient
	})

	stopCh := make(chan struct{})
	server, err := NewOAuthServerAPIServer(cfg, stopCh)
	if err != nil {
		t.Fatalf("Failed to create server: %v", err)
	}

	err = server.PrepareRun(stopCh)
	assert.NoError(t, err)

	// Test CSRF error handler by sending a request with invalid CSRF token
	req := httptest.NewRequest("POST", constants.FuyaoLoginEndpoint, nil)
	rr := httptest.NewRecorder()

	// The CSRF middleware should handle this, but we can't easily test it without a valid CSRF token
	// This test verifies the route is set up correctly
	server.Router.ServeHTTP(rr, req)
	// The response should not be 404, indicating the route is registered
	assert.NotEqual(t, http.StatusNotFound, rr.Code)
}

func TestOAuthServerAPIServer_Run(t *testing.T) {
	tests := []struct {
		name        string
		cfg         *overallconfigs.OAuthServerAPIServerConfig
		ctxCancel   bool
		hasTLS      bool
		setup       func() *gomonkey.Patches
		expectError bool
	}{
		{
			name: "run http server successfully",
			cfg:  createTestConfig(),
			setup: func() *gomonkey.Patches {
				patches := gomonkey.NewPatches()
				scheme := runtime.NewScheme()
				fakeK8sClient := fake.NewSimpleClientset()
				fakeDynamicClient := dynamicfake.NewSimpleDynamicClient(scheme)

				patches.ApplyFunc(httpserver.NewHttpServer, func(_ *httpserver.ServerOptions) (*http.Server, error) {
					return &http.Server{Addr: ":0"}, nil // Use :0 to get a free port
				})

				patches.ApplyFunc(config.GetKubernetesClient, func(_ *config.KubernetesConfig) kubernetes.Interface {
					return fakeK8sClient
				})

				patches.ApplyFunc(config.GetDynamicClient, func(_ *config.KubernetesConfig) dynamic.Interface {
					return fakeDynamicClient
				})

				return patches
			},
			ctxCancel:   true,
			expectError: false,
		},
		{
			name: "run https server successfully",
			cfg: func() *overallconfigs.OAuthServerAPIServerConfig {
				cfg := createTestConfig()
				cfg.HttpServerConfig.HttpsPort = 9443
				return cfg
			}(),
			hasTLS: true,
			setup: func() *gomonkey.Patches {
				patches := gomonkey.NewPatches()
				scheme := runtime.NewScheme()
				fakeK8sClient := fake.NewSimpleClientset()
				fakeDynamicClient := dynamicfake.NewSimpleDynamicClient(scheme)

				patches.ApplyFunc(httpserver.NewHttpServer, func(_ *httpserver.ServerOptions) (*http.Server, error) {
					server := &http.Server{Addr: ":0"}
					server.TLSConfig = &tls.Config{}
					return server, nil
				})

				patches.ApplyFunc(config.GetKubernetesClient, func(_ *config.KubernetesConfig) kubernetes.Interface {
					return fakeK8sClient
				})

				patches.ApplyFunc(config.GetDynamicClient, func(_ *config.KubernetesConfig) dynamic.Interface {
					return fakeDynamicClient
				})

				return patches
			},
			ctxCancel:   true,
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var patches *gomonkey.Patches
			if tt.setup != nil {
				patches = tt.setup()
				defer patches.Reset()
			}

			stopCh := make(chan struct{})
			server, err := NewOAuthServerAPIServer(tt.cfg, stopCh)
			if err != nil {
				t.Fatalf("Failed to create server: %v", err)
			}

			// Create a context that will be cancelled
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			// Start server in a goroutine
			errCh := make(chan error, 1)
			go func() {
				errCh <- server.Run(ctx)
			}()

			// Cancel context after a short delay to trigger shutdown
			if tt.ctxCancel {
				time.Sleep(100 * time.Millisecond)
				cancel()
			}

			// Wait for server to finish or timeout
			select {
			case err := <-errCh:
				if tt.expectError {
					assert.Error(t, err)
				} else {
					// http.ErrServerClosed is expected when server shuts down gracefully
					if err != nil && err != http.ErrServerClosed {
						t.Logf("Unexpected error: %v", err)
					}
				}
			case <-time.After(2 * time.Second):
				cancel()
				t.Fatal("Server did not shut down within timeout")
			}
		})
	}
}

func TestOAuthServerAPIServer_Run_Shutdown(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	scheme := runtime.NewScheme()
	fakeK8sClient := fake.NewSimpleClientset()
	fakeDynamicClient := dynamicfake.NewSimpleDynamicClient(scheme)

	patches.ApplyFunc(httpserver.NewHttpServer, func(_ *httpserver.ServerOptions) (*http.Server, error) {
		return &http.Server{Addr: ":0"}, nil
	})

	patches.ApplyFunc(config.GetKubernetesClient, func(_ *config.KubernetesConfig) kubernetes.Interface {
		return fakeK8sClient
	})

	patches.ApplyFunc(config.GetDynamicClient, func(_ *config.KubernetesConfig) dynamic.Interface {
		return fakeDynamicClient
	})

	cfg := createTestConfig()
	stopCh := make(chan struct{})
	server, err := NewOAuthServerAPIServer(cfg, stopCh)
	if err != nil {
		t.Fatalf("Failed to create server: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start server
	errCh := make(chan error, 1)
	go func() {
		errCh <- server.Run(ctx)
	}()

	// Give server time to start
	time.Sleep(50 * time.Millisecond)

	// Cancel context to trigger shutdown
	cancel()

	// Wait for shutdown
	select {
	case err := <-errCh:
		// http.ErrServerClosed is expected
		if err != nil && err != http.ErrServerClosed {
			t.Logf("Server shutdown error: %v", err)
		}
	case <-time.After(1 * time.Second):
		t.Fatal("Server did not shut down within timeout")
	}
}

func TestOAuthServerAPIServer_Structure(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	scheme := runtime.NewScheme()
	fakeK8sClient := fake.NewSimpleClientset()
	fakeDynamicClient := dynamicfake.NewSimpleDynamicClient(scheme)

	patches.ApplyFunc(httpserver.NewHttpServer, func(_ *httpserver.ServerOptions) (*http.Server, error) {
		return &http.Server{Addr: ":9096"}, nil
	})

	patches.ApplyFunc(config.GetKubernetesClient, func(_ *config.KubernetesConfig) kubernetes.Interface {
		return fakeK8sClient
	})

	patches.ApplyFunc(config.GetDynamicClient, func(_ *config.KubernetesConfig) dynamic.Interface {
		return fakeDynamicClient
	})

	cfg := createTestConfig()
	stopCh := make(chan struct{})
	server, err := NewOAuthServerAPIServer(cfg, stopCh)
	if err != nil {
		t.Fatalf("Failed to create server: %v", err)
	}

	// Verify structure
	assert.NotNil(t, server.Server)
	assert.NotNil(t, server.Router)
	assert.NotNil(t, server.Login)
	assert.NotNil(t, server.OAuthServer)
	assert.Equal(t, cfg, server.Cfg)

	// Verify types
	assert.IsType(t, &http.Server{}, server.Server)
	assert.IsType(t, &fuyaopassword.Login{}, server.Login)
	assert.IsType(t, &oauth2.FuyaoAuthorizeServer{}, server.OAuthServer)
}

func TestOAuthServerAPIServer_ComponentInitialization(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	scheme := runtime.NewScheme()
	fakeK8sClient := fake.NewSimpleClientset()
	fakeDynamicClient := dynamicfake.NewSimpleDynamicClient(scheme)

	var capturedK8sConfig *config.KubernetesConfig
	var capturedDynamicConfig *config.KubernetesConfig

	patches.ApplyFunc(httpserver.NewHttpServer, func(_ *httpserver.ServerOptions) (*http.Server, error) {
		return &http.Server{Addr: ":9096"}, nil
	})

	patches.ApplyFunc(config.GetKubernetesClient, func(k8sConfig *config.KubernetesConfig) kubernetes.Interface {
		capturedK8sConfig = k8sConfig
		return fakeK8sClient
	})

	patches.ApplyFunc(config.GetDynamicClient, func(k8sConfig *config.KubernetesConfig) dynamic.Interface {
		capturedDynamicConfig = k8sConfig
		return fakeDynamicClient
	})

	cfg := createTestConfig()
	stopCh := make(chan struct{})
	server, err := NewOAuthServerAPIServer(cfg, stopCh)
	if err != nil {
		t.Fatalf("Failed to create server: %v", err)
	}

	// Verify that components were initialized with correct config
	assert.NotNil(t, server)
	assert.Equal(t, cfg.K8sConfig, capturedK8sConfig)
	assert.Equal(t, cfg.K8sConfig, capturedDynamicConfig)

	// Verify session store was created
	assert.NotNil(t, server.Login)
	loginType := reflect.TypeOf(server.Login).Elem()
	assert.Equal(t, "Login", loginType.Name())

	// Verify OAuth server was created
	assert.NotNil(t, server.OAuthServer)
	oauthType := reflect.TypeOf(server.OAuthServer).Elem()
	assert.Equal(t, "FuyaoAuthorizeServer", oauthType.Name())
}
