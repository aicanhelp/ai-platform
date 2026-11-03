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

package options

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"

	"openfuyao/oauth-webhook/pkg/fuyaoerrors"
	"openfuyao/oauth-webhook/pkg/k8sconfig"
)

func TestNewOAuthWebhookOptionAndValidate(t *testing.T) {
	option := NewOAuthWebhookOption()
	require.NotNil(t, option)
	require.Equal(t, "", option.ConfigFile)
	require.ErrorIs(t, option.Validate(), fuyaoerrors.ErrOAuthWebhookConfigFileMissing)

	option.ConfigFile = "config.yaml"
	require.NoError(t, option.Validate())
}

func TestReadConfigSuccess(t *testing.T) {
	testDir := newTestDir(t, "options-config-")
	configFile := writeFile(t, testDir, "oauth-webhook.yaml", `
HttpServerConfig:
  HttpPort: 19095
  HttpsPort: 0
`)

	patches := gomonkey.ApplyFunc(k8sconfig.GetKubernetesClient, func(*k8sconfig.KubernetesConfig) kubernetes.Interface {
		return fake.NewSimpleClientset(&corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      secretName,
				Namespace: secretNamespace,
			},
			Data: map[string][]byte{
				"oauth-jwt.key": []byte("real-signing-key"),
			},
		})
	})
	defer patches.Reset()

	option := &OAuthWebhookOption{ConfigFile: configFile}
	cfg, err := option.ReadConfig()
	require.NoError(t, err)
	require.NotNil(t, cfg.HttpServerConfig)
	require.Equal(t, 19095, cfg.HttpServerConfig.HttpPort)
	require.Equal(t, []byte("real-signing-key"), cfg.JWTPrivateKey)
}

func TestReadConfigFailsWhenSecretMissing(t *testing.T) {
	testDir := newTestDir(t, "options-config-")
	configFile := writeFile(t, testDir, "oauth-webhook.yaml", `
HttpServerConfig:
  HttpPort: 19095
`)

	patches := gomonkey.ApplyFunc(k8sconfig.GetKubernetesClient, func(*k8sconfig.KubernetesConfig) kubernetes.Interface {
		return fake.NewSimpleClientset()
	})
	defer patches.Reset()

	option := &OAuthWebhookOption{ConfigFile: configFile}
	_, err := option.ReadConfig()
	require.EqualError(t, err, "fail to load data from jwt secret")
}

func TestReadConfigFailsWhenSecretKeyMissing(t *testing.T) {
	testDir := newTestDir(t, "options-config-")
	configFile := writeFile(t, testDir, "oauth-webhook.yaml", `
HttpServerConfig:
  HttpPort: 19095
`)

	patches := gomonkey.ApplyFunc(k8sconfig.GetKubernetesClient, func(*k8sconfig.KubernetesConfig) kubernetes.Interface {
		return fake.NewSimpleClientset(&corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      secretName,
				Namespace: secretNamespace,
			},
			Data: map[string][]byte{
				"another.key": []byte("unexpected"),
			},
		})
	})
	defer patches.Reset()

	option := &OAuthWebhookOption{ConfigFile: configFile}
	_, err := option.ReadConfig()
	require.EqualError(t, err, "fail to load data from jwt secret")
}

func newTestDir(t *testing.T, prefix string) string {
	t.Helper()

	dir, err := os.MkdirTemp(".", prefix)
	require.NoError(t, err)
	t.Cleanup(func() {
		_ = os.RemoveAll(dir)
	})

	return dir
}

func writeFile(t *testing.T, dir, name, content string) string {
	t.Helper()

	filePath := filepath.Join(dir, name)
	require.NoError(t, os.WriteFile(filePath, []byte(content), 0o600))
	return filePath
}
