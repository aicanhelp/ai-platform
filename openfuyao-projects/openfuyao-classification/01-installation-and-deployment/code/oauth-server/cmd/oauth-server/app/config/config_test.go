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

package config

import (
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"openfuyao/oauth-server/pkg/config"
	"openfuyao/oauth-server/pkg/fuyaoerrors"
	"openfuyao/oauth-server/pkg/httpserver"
	"openfuyao/oauth-server/pkg/zlog"
)

// TestMain sets up the test environment by initializing the logger to only output to console,
// avoiding the need for external config files or log directories.
func TestMain(m *testing.M) {
	// Override the global logger with a console-only logger for tests
	zlog.Logger = zlog.GetLogger(&zlog.LogConfig{
		Level:       "info",
		EncoderType: "console",
		OutMod:      "console", // Console only - no file writing
	})
	os.Exit(m.Run())
}

func TestNewDefaultOAuthAPIServerServerConfig(t *testing.T) {
	cfg := NewDefaultOAuthAPIServerServerConfig()

	assert.NotNil(t, cfg)
	assert.NotNil(t, cfg.HttpServerConfig)
	assert.NotNil(t, cfg.K8sConfig)
	assert.NotNil(t, cfg.LoginConfig)
	assert.NotNil(t, cfg.IPProtectorConfig)
	assert.NotNil(t, cfg.IDPLoginStoreConfig)
	assert.NotNil(t, cfg.OAuthServerConfig)
}

func TestOAuthServerAPIServerConfigValidate(t *testing.T) {
	tests := []struct {
		name          string
		config        *OAuthServerAPIServerConfig
		expectErrors  bool
		errorContains error
	}{
		{
			name: "valid config",
			config: &OAuthServerAPIServerConfig{
				HttpServerConfig: &httpserver.ServerOptions{
					HttpPort:  9096,
					HttpsPort: 0,
				},
				K8sConfig:   config.NewKubernetesConfig(),
				LoginConfig: &LoginConfig{Provider: "fuyaoPaswordProvider"},
				IPProtectorConfig: &IPProtectorConfig{
					FailTimes:    5,
					FailDuration: time.Minute * 5,
					LockDuration: time.Minute * 30,
				},
				IDPLoginStoreConfig: &IDPLoginStoreConfig{
					SessionName:   "idpLogin",
					SessionMaxAge: 300,
				},
				OAuthServerConfig: &OAuthServerConfig{
					CodeTokenNamespace: "oauth-code-token",
					JWTPrivateKey:      []byte("test-key"),
					ClientMapper:       map[string]string{"console": "console-password"},
				},
			},
			expectErrors: false,
		},
		{
			name: "invalid login config",
			config: &OAuthServerAPIServerConfig{
				HttpServerConfig: &httpserver.ServerOptions{
					HttpPort:  9096,
					HttpsPort: 0,
				},
				K8sConfig:           config.NewKubernetesConfig(),
				LoginConfig:         &LoginConfig{Provider: ""},
				IPProtectorConfig:   newDefaultIPProtectorConfig(),
				IDPLoginStoreConfig: newIDPLoginStoreConfig(),
				OAuthServerConfig: &OAuthServerConfig{
					JWTPrivateKey: []byte("test-key"),
					ClientMapper:  map[string]string{"console": "console-password"},
				},
			},
			expectErrors:  true,
			errorContains: fuyaoerrors.ErrLoginConfigMissing,
		},
		{
			name: "invalid oauth server config - missing JWT key",
			config: &OAuthServerAPIServerConfig{
				HttpServerConfig: &httpserver.ServerOptions{
					HttpPort:  9096,
					HttpsPort: 0,
				},
				K8sConfig:           config.NewKubernetesConfig(),
				LoginConfig:         newDefaultLoginConfig(),
				IPProtectorConfig:   newDefaultIPProtectorConfig(),
				IDPLoginStoreConfig: newIDPLoginStoreConfig(),
				OAuthServerConfig: &OAuthServerConfig{
					ClientMapper: map[string]string{"console": "console-password"},
				},
			},
			expectErrors:  true,
			errorContains: fuyaoerrors.ErrJWTPrivateKeyMissing,
		},
		{
			name: "invalid oauth server config - missing client mapper",
			config: &OAuthServerAPIServerConfig{
				HttpServerConfig: &httpserver.ServerOptions{
					HttpPort:  9096,
					HttpsPort: 0,
				},
				K8sConfig:           config.NewKubernetesConfig(),
				LoginConfig:         newDefaultLoginConfig(),
				IPProtectorConfig:   newDefaultIPProtectorConfig(),
				IDPLoginStoreConfig: newIDPLoginStoreConfig(),
				OAuthServerConfig: &OAuthServerConfig{
					JWTPrivateKey: []byte("test-key"),
				},
			},
			expectErrors:  true,
			errorContains: fuyaoerrors.ErrClientInfoMissing,
		},
		{
			name: "invalid idp login store config",
			config: &OAuthServerAPIServerConfig{
				HttpServerConfig: &httpserver.ServerOptions{
					HttpPort:  9096,
					HttpsPort: 0,
				},
				K8sConfig:         config.NewKubernetesConfig(),
				LoginConfig:       newDefaultLoginConfig(),
				IPProtectorConfig: newDefaultIPProtectorConfig(),
				IDPLoginStoreConfig: &IDPLoginStoreConfig{
					SessionName: "",
				},
				OAuthServerConfig: &OAuthServerConfig{
					JWTPrivateKey: []byte("test-key"),
					ClientMapper:  map[string]string{"console": "console-password"},
				},
			},
			expectErrors:  true,
			errorContains: fuyaoerrors.ErrIdpLoginStoreConfigMissing,
		},
		{
			name: "invalid ip protector config - partial fields",
			config: &OAuthServerAPIServerConfig{
				HttpServerConfig: &httpserver.ServerOptions{
					HttpPort:  9096,
					HttpsPort: 0,
				},
				K8sConfig:   config.NewKubernetesConfig(),
				LoginConfig: newDefaultLoginConfig(),
				IPProtectorConfig: &IPProtectorConfig{
					FailTimes:    5,
					FailDuration: 0,
					LockDuration: time.Minute * 30,
				},
				IDPLoginStoreConfig: newIDPLoginStoreConfig(),
				OAuthServerConfig: &OAuthServerConfig{
					JWTPrivateKey: []byte("test-key"),
					ClientMapper:  map[string]string{"console": "console-password"},
				},
			},
			expectErrors:  true,
			errorContains: fuyaoerrors.ErrIPProtectorConfigMissing,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			errs := tt.config.Validate()
			if tt.expectErrors {
				assert.NotEmpty(t, errs)
				if tt.errorContains != nil {
					assert.Contains(t, errs, tt.errorContains)
				}
			} else {
				assert.Empty(t, errs)
			}
		})
	}
}

func TestOAuthServerAPIServerConfigComplete(t *testing.T) {
	tests := []struct {
		name   string
		config *OAuthServerAPIServerConfig
	}{
		{
			name: "complete all nil fields",
			config: &OAuthServerAPIServerConfig{
				HttpServerConfig:    nil,
				K8sConfig:           nil,
				LoginConfig:         nil,
				IPProtectorConfig:   nil,
				IDPLoginStoreConfig: nil,
				OAuthServerConfig:   nil,
			},
		},
		{
			name: "complete partial nil fields",
			config: &OAuthServerAPIServerConfig{
				HttpServerConfig:    httpserver.NewDefaultHttpServerOptions(),
				K8sConfig:           nil,
				LoginConfig:         nil,
				IPProtectorConfig:   newDefaultIPProtectorConfig(),
				IDPLoginStoreConfig: nil,
				OAuthServerConfig:   nil,
			},
		},
		{
			name: "all fields already set",
			config: &OAuthServerAPIServerConfig{
				HttpServerConfig:    httpserver.NewDefaultHttpServerOptions(),
				K8sConfig:           config.NewKubernetesConfig(),
				LoginConfig:         newDefaultLoginConfig(),
				IPProtectorConfig:   newDefaultIPProtectorConfig(),
				IDPLoginStoreConfig: newIDPLoginStoreConfig(),
				OAuthServerConfig:   newOAuthServerConfig(),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.config.Complete()
			assert.NotNil(t, result)
			assert.NotNil(t, result.HttpServerConfig)
			assert.NotNil(t, result.K8sConfig)
			assert.NotNil(t, result.LoginConfig)
			assert.NotNil(t, result.IPProtectorConfig)
			assert.NotNil(t, result.IDPLoginStoreConfig)
			assert.NotNil(t, result.OAuthServerConfig)
		})
	}
}

func TestLoginConfigValidate(t *testing.T) {
	tests := []struct {
		name         string
		config       *LoginConfig
		expectErrors bool
	}{
		{
			name: "valid config",
			config: &LoginConfig{
				Provider: "fuyaoPaswordProvider",
			},
			expectErrors: false,
		},
		{
			name: "empty provider",
			config: &LoginConfig{
				Provider: "",
			},
			expectErrors: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			errs := tt.config.Validate()
			if tt.expectErrors {
				assert.NotEmpty(t, errs)
				assert.Contains(t, errs, fuyaoerrors.ErrLoginConfigMissing)
			} else {
				assert.Empty(t, errs)
			}
		})
	}
}

func TestNewDefaultLoginConfig(t *testing.T) {
	cfg := newDefaultLoginConfig()
	assert.NotNil(t, cfg)
	assert.Equal(t, "fuyaoPaswordProvider", cfg.Provider)
}

func TestIPProtectorConfigValidate(t *testing.T) {
	tests := []struct {
		name         string
		config       *IPProtectorConfig
		expectErrors bool
	}{
		{
			name: "valid config",
			config: &IPProtectorConfig{
				FailTimes:    5,
				FailDuration: time.Minute * 5,
				LockDuration: time.Minute * 30,
			},
			expectErrors: false,
		},
		{
			name: "all fields zero - valid",
			config: &IPProtectorConfig{
				FailTimes:    0,
				FailDuration: 0,
				LockDuration: 0,
			},
			expectErrors: false,
		},
		{
			name: "partial fields - invalid",
			config: &IPProtectorConfig{
				FailTimes:    5,
				FailDuration: 0,
				LockDuration: time.Minute * 30,
			},
			expectErrors: true,
		},
		{
			name: "missing fail times",
			config: &IPProtectorConfig{
				FailTimes:    0,
				FailDuration: time.Minute * 5,
				LockDuration: time.Minute * 30,
			},
			expectErrors: true,
		},
		{
			name: "missing lock duration - valid due to code logic",
			config: &IPProtectorConfig{
				FailTimes:    5,
				FailDuration: time.Minute * 5,
				LockDuration: 0,
			},
			// When LockDuration == 0, the first if condition returns nil (considered as disabled)
			// Condition: (FailTimes == 0 && FailDuration == 0) || LockDuration == 0
			// Result: (false && false) || true = true -> returns nil
			expectErrors: false,
		},
		{
			name: "missing fail duration only - invalid",
			config: &IPProtectorConfig{
				FailTimes:    5,
				FailDuration: 0,
				LockDuration: time.Minute * 30,
			},
			// Condition: (FailTimes == 0 && FailDuration == 0) || LockDuration == 0
			// Result: (false && true) || false = false -> proceeds to second check
			// Second check: FailTimes == 0 || FailDuration == 0 || LockDuration == 0
			// Result: false || true || false = true -> returns error
			expectErrors: true,
		},
		{
			name: "missing both fail duration and lock duration",
			config: &IPProtectorConfig{
				FailTimes:    5,
				FailDuration: 0,
				LockDuration: 0,
			},
			expectErrors: false, // Returns nil when LockDuration == 0
		},
		{
			name: "missing fail times only",
			config: &IPProtectorConfig{
				FailTimes:    0,
				FailDuration: time.Minute * 5,
				LockDuration: 0,
			},
			expectErrors: false, // Returns nil when LockDuration == 0
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			errs := tt.config.Validate()
			if tt.expectErrors {
				assert.NotEmpty(t, errs)
				assert.Contains(t, errs, fuyaoerrors.ErrIPProtectorConfigMissing)
			} else {
				assert.Empty(t, errs)
			}
		})
	}
}

func TestNewDefaultIPProtectorConfig(t *testing.T) {
	cfg := newDefaultIPProtectorConfig()
	assert.NotNil(t, cfg)
	assert.Equal(t, 5, cfg.FailTimes)
	assert.Equal(t, time.Minute*5, cfg.FailDuration)
	assert.Equal(t, time.Minute*30, cfg.LockDuration)
}

func TestIDPLoginStoreConfigValidate(t *testing.T) {
	tests := []struct {
		name         string
		config       *IDPLoginStoreConfig
		expectErrors bool
	}{
		{
			name: "valid config",
			config: &IDPLoginStoreConfig{
				SessionName:   "idpLogin",
				SessionMaxAge: 300,
			},
			expectErrors: false,
		},
		{
			name: "missing session name",
			config: &IDPLoginStoreConfig{
				SessionName:   "",
				SessionMaxAge: 300,
			},
			expectErrors: true,
		},
		{
			name: "zero session max age - valid but logs warning",
			config: &IDPLoginStoreConfig{
				SessionName:   "idpLogin",
				SessionMaxAge: 0,
			},
			expectErrors: false,
		},
		{
			name: "negative session max age - valid but logs warning",
			config: &IDPLoginStoreConfig{
				SessionName:   "idpLogin",
				SessionMaxAge: -1,
			},
			expectErrors: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			errs := tt.config.Validate()
			if tt.expectErrors {
				assert.NotEmpty(t, errs)
				assert.Contains(t, errs, fuyaoerrors.ErrIdpLoginStoreConfigMissing)
			} else {
				assert.Empty(t, errs)
			}
		})
	}
}

func TestNewIDPLoginStoreConfig(t *testing.T) {
	cfg := newIDPLoginStoreConfig()
	assert.NotNil(t, cfg)
	assert.Equal(t, "idpLogin", cfg.SessionName)
	assert.Equal(t, 300, cfg.SessionMaxAge)
	assert.Equal(t, "csrf", cfg.CsrfCookieName)
}

func TestOAuthServerConfigValidate(t *testing.T) {
	tests := []struct {
		name         string
		config       *OAuthServerConfig
		expectErrors bool
	}{
		{
			name: "valid config",
			config: &OAuthServerConfig{
				CodeTokenNamespace: "oauth-code-token",
				AuthCodeExp:        time.Minute * 5,
				AccessTokenExp:     time.Hour * 2,
				RefreshTokenExp:    time.Hour * 2,
				IsGenerateRefresh:  false,
				JWTKeyID:           "access_token_sign_key",
				JWTPrivateKey:      []byte("test-key"),
				ClientMapper:       map[string]string{"console": "console-password"},
			},
			expectErrors: false,
		},
		{
			name: "missing JWT private key",
			config: &OAuthServerConfig{
				CodeTokenNamespace: "oauth-code-token",
				JWTPrivateKey:      nil,
				ClientMapper:       map[string]string{"console": "console-password"},
			},
			expectErrors: true,
		},
		{
			name: "missing client mapper",
			config: &OAuthServerConfig{
				CodeTokenNamespace: "oauth-code-token",
				JWTPrivateKey:      []byte("test-key"),
				ClientMapper:       nil,
			},
			expectErrors: true,
		},
		{
			name: "empty client mapper",
			config: &OAuthServerConfig{
				CodeTokenNamespace: "oauth-code-token",
				JWTPrivateKey:      []byte("test-key"),
				ClientMapper:       map[string]string{},
			},
			expectErrors: true,
		},
		{
			name: "empty namespace - uses default",
			config: &OAuthServerConfig{
				CodeTokenNamespace: "",
				JWTPrivateKey:      []byte("test-key"),
				ClientMapper:       map[string]string{"console": "console-password"},
			},
			expectErrors: false,
		},
		{
			name: "zero expiration times - valid but logs warnings",
			config: &OAuthServerConfig{
				CodeTokenNamespace: "oauth-code-token",
				AuthCodeExp:        0,
				AccessTokenExp:     0,
				JWTPrivateKey:      []byte("test-key"),
				ClientMapper:       map[string]string{"console": "console-password"},
			},
			expectErrors: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			errs := tt.config.Validate()
			if tt.expectErrors {
				assert.NotEmpty(t, errs)
			} else {
				assert.Empty(t, errs)
			}
		})
	}
}

func TestNewOAuthServerConfig(t *testing.T) {
	cfg := newOAuthServerConfig()
	assert.NotNil(t, cfg)
	assert.Equal(t, "oauth-code-token", cfg.CodeTokenNamespace)
	assert.Equal(t, time.Minute*5, cfg.AuthCodeExp)
	assert.Equal(t, time.Hour*2, cfg.AccessTokenExp)
	assert.Equal(t, time.Hour*2, cfg.RefreshTokenExp)
	assert.Equal(t, false, cfg.IsGenerateRefresh)
	assert.Equal(t, "access_token_sign_key", cfg.JWTKeyID)
	assert.NotNil(t, cfg.ClientMapper)
	assert.Equal(t, "console-password", cfg.ClientMapper["console"])
}
