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

package k8sconfig

import (
	"errors"
	"os"
	"os/user"
	"path/filepath"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/client-go/util/homedir"
)

const (
	qps   = 1000000
	burst = 1000000
)

func TestNewKubernetesConfig(t *testing.T) {
	t.Run("create new kubernetes config", func(t *testing.T) {
		config := NewKubernetesConfig()
		assert.NotNil(t, config)
		assert.Equal(t, float32(qps), config.QPS)
		assert.Equal(t, burst, config.Burst)
	})
}

func TestKubernetesConfigValidate(t *testing.T) {
	testValidateWithConfig(t)
	testValidateEmptyConfig(t)
}

func testValidateWithConfig(t *testing.T) {
	t.Run("validate kubernetes config", func(t *testing.T) {
		config := &KubernetesConfig{
			KubeConfigFile: "/path/to/kubeconfig",
			QPS:            100,
			Burst:          200,
		}

		errs := config.Validate()
		assert.Empty(t, errs)
	})
}

func testValidateEmptyConfig(t *testing.T) {
	t.Run("validate empty kubernetes config", func(t *testing.T) {
		config := &KubernetesConfig{}

		errs := config.Validate()
		assert.Empty(t, errs)
	})
}

func TestGetDefaultKubeConfigFile(t *testing.T) {
	testGetDefaultWithHomeDir(t)
	testGetDefaultWithUserCurrent(t)
	testGetDefaultNoHomeAvailable(t)
	testGetDefaultFileNotExists(t)
}

func testGetDefaultWithHomeDir(t *testing.T) {
	t.Run("get default kubeconfig file with home dir", func(t *testing.T) {
		setupHomeEnv(t, "/home/testuser")

		patches := gomonkey.NewPatches()
		defer patches.Reset()

		patches.ApplyFunc(homedir.HomeDir, func() string {
			return "/home/testuser"
		})

		patches.ApplyFunc(os.Stat, func(name string) (os.FileInfo, error) {
			if name == "/home/testuser/.kube/config" {
				return nil, nil
			}
			return nil, os.ErrNotExist
		})

		result := getDefaultKubeConfigFile()
		assert.Equal(t, "/home/testuser/.kube/config", result)
	})
}

func testGetDefaultWithUserCurrent(t *testing.T) {
	t.Run("get default kubeconfig file with user current", func(t *testing.T) {
		setupHomeEnv(t, "/home/testuser")

		patches := gomonkey.ApplyFunc(homedir.HomeDir, func() string {
			return ""
		})

		patches.ApplyFunc(user.Current, func() (*user.User, error) {
			return &user.User{HomeDir: "/home/testuser"}, nil
		})

		patches.ApplyFunc(os.Stat, func(name string) (os.FileInfo, error) {
			if name == "/home/testuser/.kube/config" {
				return nil, nil
			}
			return nil, os.ErrNotExist
		})

		defer patches.Reset()

		result := getDefaultKubeConfigFile()
		assert.Equal(t, "/home/testuser/.kube/config", result)
	})
}

func testGetDefaultNoHomeAvailable(t *testing.T) {
	t.Run("no home directory available", func(t *testing.T) {
		setupHomeEnv(t, "/home/testuser")

		patches := gomonkey.ApplyFunc(homedir.HomeDir, func() string {
			return ""
		})

		patches.ApplyFunc(user.Current, func() (*user.User, error) {
			return nil, errors.New("user error")
		})

		defer patches.Reset()

		result := getDefaultKubeConfigFile()
		assert.Empty(t, result)
	})
}

func testGetDefaultFileNotExists(t *testing.T) {
	t.Run("kubeconfig file does not exist", func(t *testing.T) {
		patches := gomonkey.ApplyFunc(homedir.HomeDir, func() string {
			return "/home/testuser"
		})

		patches.ApplyFunc(os.Stat, func(name string) (os.FileInfo, error) {
			return nil, os.ErrNotExist
		})

		defer patches.Reset()

		result := getDefaultKubeConfigFile()
		assert.Empty(t, result)
	})
}

func setupHomeEnv(t *testing.T, homePath string) {
	originalHome := os.Getenv("HOME")
	t.Setenv("HOME", homePath)

	t.Cleanup(func() {
		if originalHome != "" {
			err := os.Setenv("HOME", originalHome)
			if err != nil {
				return
			}
		}
	})
}

func TestGetKubeConfigOrInClusterConfigFromFile(t *testing.T) {
	testDir := newTestDir(t, "kubeconfig-")
	kubeConfigFile := writeFile(t, testDir, "config", `
apiVersion: v1
kind: Config
clusters:
- cluster:
    server: https://127.0.0.1:6443
    insecure-skip-tls-verify: true
  name: test-cluster
contexts:
- context:
    cluster: test-cluster
    user: test-user
  name: test-context
current-context: test-context
users:
- name: test-user
  user:
    token: test-token
`)

	cfg := GetKubeConfigOrInClusterConfig(&KubernetesConfig{
		KubeConfigFile: kubeConfigFile,
	})

	require.Equal(t, "https://127.0.0.1:6443", cfg.Host)
}

func TestGetKubernetesClientFromConfig(t *testing.T) {
	testDir := newTestDir(t, "kubeclient-")
	kubeConfigFile := writeFile(t, testDir, "config", `
apiVersion: v1
kind: Config
clusters:
- cluster:
    server: https://127.0.0.1:6443
    insecure-skip-tls-verify: true
  name: test-cluster
contexts:
- context:
    cluster: test-cluster
    user: test-user
  name: test-context
current-context: test-context
users:
- name: test-user
  user:
    token: test-token
`)

	client := GetKubernetesClient(&KubernetesConfig{KubeConfigFile: kubeConfigFile})
	require.NotNil(t, client)
	require.NotNil(t, client.CoreV1())
	require.NotNil(t, client.Discovery())
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
