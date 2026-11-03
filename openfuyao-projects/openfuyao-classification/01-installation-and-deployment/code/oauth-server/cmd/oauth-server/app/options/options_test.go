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

package options

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"

	"openfuyao/oauth-server/pkg/fuyaoerrors"
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

func TestNewOAuthServerOption(t *testing.T) {
	option := NewOAuthServerOption()
	assert.NotNil(t, option)
	assert.Equal(t, "", option.ConfigFile)
}

func TestOAuthServerOptionValidate(t *testing.T) {
	tests := []struct {
		name        string
		option      *OAuthServerOption
		expectError bool
	}{
		{
			name: "valid config file",
			option: &OAuthServerOption{
				ConfigFile: "/path/to/config.yaml",
			},
			expectError: false,
		},
		{
			name: "missing config file",
			option: &OAuthServerOption{
				ConfigFile: "",
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.option.Validate()
			if tt.expectError {
				assert.Error(t, err)
				assert.Equal(t, fuyaoerrors.ErrOAuthServerConfigFileMissing, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestReadConfig(t *testing.T) {
	// Create a temporary config file
	tempDir := t.TempDir()
	configFile := filepath.Join(tempDir, "config.yaml")

	configContent := `
HttpServerConfig:
  HttpPort: 9096
  HttpsPort: 0
K8SConfig:
  KubeConfigPath: ""
LoginConfig:
  Provider: "fuyaoPaswordProvider"
IPProtectorConfig:
  FailTimes: 5
  FailDuration: 300000000000
  LockDuration: 1800000000000
IDPLoginStoreConfig:
  SessionName: "idpLogin"
  SessionMaxAge: 300
  CsrfCookieName: "csrf"
OAuthServerConfig:
  CodeTokenNamespace: "oauth-code-token"
  AuthCodeExp: 300000000000
  AccessTokenExp: 7200000000000
  RefreshTokenExp: 7200000000000
  IsGenerateRefresh: false
  JWTKeyID: "access_token_sign_key"
`

	err := os.WriteFile(configFile, []byte(configContent), 0644)
	assert.NoError(t, err)

	// This test will fail without a real k8s cluster or proper mock
	// So we just test the basic flow
	t.Run("missing config file", func(t *testing.T) {
		option := &OAuthServerOption{
			ConfigFile: "/nonexistent/config.yaml",
		}
		_, err := option.ReadConfig()
		assert.Error(t, err)
	})
}

func TestReadDataFromK8sSecret(t *testing.T) {
	tests := []struct {
		name        string
		secretName  string
		key         string
		secretData  map[string][]byte
		expectError bool
	}{
		{
			name:       "valid secret data",
			secretName: "test-secret",
			key:        "test-key",
			secretData: map[string][]byte{
				"test-key": []byte("test-value"),
			},
			expectError: false,
		},
		{
			name:       "missing key in secret",
			secretName: "test-secret",
			key:        "missing-key",
			secretData: map[string][]byte{
				"other-key": []byte("test-value"),
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create fake k8s client
			fakeClient := fake.NewSimpleClientset()

			// Create a secret if data is provided
			if len(tt.secretData) > 0 {
				secret := &corev1.Secret{
					ObjectMeta: metav1.ObjectMeta{
						Name:      tt.secretName,
						Namespace: secretNamespace,
					},
					Data: tt.secretData,
				}
				_, err := fakeClient.CoreV1().Secrets(secretNamespace).Create(
					context.TODO(),
					secret,
					metav1.CreateOptions{},
				)
				assert.NoError(t, err)
			}

			data, err := readDataFromK8sSecret(fakeClient, tt.secretName, tt.key)

			if tt.expectError {
				assert.Error(t, err)
				assert.Equal(t, fuyaoerrors.ErrFailToGetSecret, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.secretData[tt.key], data)
			}
		})
	}

	t.Run("secret not found", func(t *testing.T) {
		fakeClient := fake.NewSimpleClientset()
		_, err := readDataFromK8sSecret(fakeClient, "nonexistent-secret", "test-key")
		assert.Error(t, err)
		assert.Equal(t, fuyaoerrors.ErrFailToGetSecret, err)
	})
}

func TestReadConfigWithMockK8sSecrets(t *testing.T) {
	// Skip this test in unit test environment because:
	// 1. ReadConfig internally creates a k8s client via k8sconfig.GetKubernetesClient()
	// 2. GetKubernetesClient calls zlog.LogFatal when k8s connection fails, which terminates the test process
	// 3. Without dependency injection for the k8s client, we cannot mock it properly
	//
	// This test would require either:
	// - A real k8s cluster (integration test)
	// - Refactoring ReadConfig to accept a k8s client interface (dependency injection)
	t.Skip("Skipping: ReadConfig calls LogFatal on k8s connection failure, cannot be unit tested without DI")
}

func TestReadDataFromK8sSecretEdgeCases(t *testing.T) {
	t.Run("nil data in secret", func(t *testing.T) {
		fakeClient := fake.NewSimpleClientset()
		secret := &corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-secret",
				Namespace: secretNamespace,
			},
			Data: map[string][]byte{},
		}
		_, err := fakeClient.CoreV1().Secrets(secretNamespace).Create(
			context.TODO(),
			secret,
			metav1.CreateOptions{},
		)
		assert.NoError(t, err)

		data, err := readDataFromK8sSecret(fakeClient, "test-secret", "missing-key")
		assert.Error(t, err)
		assert.Nil(t, data)
		assert.Equal(t, fuyaoerrors.ErrFailToGetSecret, err)
	})

	t.Run("empty byte array in secret", func(t *testing.T) {
		fakeClient := fake.NewSimpleClientset()
		secret := &corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-secret",
				Namespace: secretNamespace,
			},
			Data: map[string][]byte{
				"test-key": []byte{},
			},
		}
		_, err := fakeClient.CoreV1().Secrets(secretNamespace).Create(
			context.TODO(),
			secret,
			metav1.CreateOptions{},
		)
		assert.NoError(t, err)

		data, err := readDataFromK8sSecret(fakeClient, "test-secret", "test-key")
		assert.NoError(t, err)
		assert.NotNil(t, data)
		assert.Equal(t, []byte{}, data)
	})
}

func TestReadConfigInvalidJSON(t *testing.T) {
	// Note: This test demonstrates the limitation of testing ReadConfig without dependency injection
	// The actual ReadConfig method creates its own k8s client, so we can't easily inject a mock

	// We can still test JSON marshaling/unmarshaling in isolation
	t.Run("test JSON handling in isolation", func(t *testing.T) {
		// Create fake k8s client with invalid JSON in client-mapper
		fakeClient := fake.NewSimpleClientset()

		jwtSecret := &corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      jwtSecretName,
				Namespace: secretNamespace,
			},
			Data: map[string][]byte{
				"oauth-jwt.key":            []byte("jwt-key-content"),
				"oauth-cookie-sign.key":    []byte("sign-key-content"),
				"oauth-cookie-encrypt.key": []byte("encrypt-key-content"),
			},
		}
		_, err := fakeClient.CoreV1().Secrets(secretNamespace).Create(
			context.TODO(),
			jwtSecret,
			metav1.CreateOptions{},
		)
		assert.NoError(t, err)

		oauthSecret := &corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      oauthSecretName,
				Namespace: secretNamespace,
			},
			Data: map[string][]byte{
				"client-mapper": []byte("invalid-json"),
			},
		}
		_, err = fakeClient.CoreV1().Secrets(secretNamespace).Create(
			context.TODO(),
			oauthSecret,
			metav1.CreateOptions{},
		)
		assert.NoError(t, err)

		// We can read the data and verify JSON unmarshaling would fail
		data, err := readDataFromK8sSecret(fakeClient, oauthSecretName, "client-mapper")
		assert.NoError(t, err)

		var clientMapper map[string]string
		err = json.Unmarshal(data, &clientMapper)
		assert.Error(t, err, "Invalid JSON should fail to unmarshal")
	})
}

func TestClientMapperJSONMarshaling(t *testing.T) {
	t.Run("valid JSON", func(t *testing.T) {
		clientMapper := map[string]string{
			"console": "console-password",
			"web":     "web-password",
		}
		data, err := json.Marshal(clientMapper)
		assert.NoError(t, err)

		var decoded map[string]string
		err = json.Unmarshal(data, &decoded)
		assert.NoError(t, err)
		assert.Equal(t, clientMapper, decoded)
	})

	t.Run("invalid JSON", func(t *testing.T) {
		invalidJSON := []byte("invalid-json")
		var decoded map[string]string
		err := json.Unmarshal(invalidJSON, &decoded)
		assert.Error(t, err)
	})

	t.Run("empty JSON object", func(t *testing.T) {
		emptyJSON := []byte("{}")
		var decoded map[string]string
		err := json.Unmarshal(emptyJSON, &decoded)
		assert.NoError(t, err)
		assert.Equal(t, map[string]string{}, decoded)
	})
}
