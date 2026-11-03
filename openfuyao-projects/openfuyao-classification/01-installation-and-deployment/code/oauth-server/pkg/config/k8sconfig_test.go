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
	"errors"
	"io/fs"
	"os"
	"os/user"
	"path/filepath"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"

	"openfuyao/oauth-server/pkg/zlog"
)

const (
	qps   = 1000000
	burst = 1000000
)

var (
	newQps   = float32(qps) + 1
	newBurst = burst + 1
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

func TestGetKubeConfigOrInClusterConfig(t *testing.T) {
	t.Run("use provided kubeconfig file", func(t *testing.T) {
		tempDir := t.TempDir()
		kubeConfigFile := filepath.Join(tempDir, "kubeconfig")

		// 创建一个简单的 kubeconfig 文件
		kubeConfigContent := `
apiVersion: v1
kind: Config
clusters:
- cluster:
   server: https://test-server
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
`
		const fileMode = 0644
		fm := fs.FileMode(uint32(fileMode))
		err := os.WriteFile(kubeConfigFile, []byte(kubeConfigContent), fm)
		assert.NoError(t, err)

		config := &KubernetesConfig{
			KubeConfigFile: kubeConfigFile,
			QPS:            float32(qps),
			Burst:          burst,
		}

		// Mock clientcmd.BuildConfigFromFlags 成功
		patches := gomonkey.ApplyFunc(clientcmd.BuildConfigFromFlags, func(
			masterUrl, kubeconfigPath string) (*rest.Config, error) {
			return &rest.Config{
				Host:  "https://test-server",
				QPS:   newQps,   // 原始值
				Burst: newBurst, // 原始值
			}, nil
		})
		defer patches.Reset()

		result := GetKubeConfigOrInClusterConfig(config)
		assert.NotNil(t, result)
		// 验证 QPS 和 Burst 被正确覆盖
		assert.Equal(t, newQps, result.QPS)
		assert.Equal(t, newBurst, result.Burst)
	})
}

func TestGetKubeConfigOrInClusterConfigBuildFail(t *testing.T) {
	t.Run("kubeconfig file build fails, use in-cluster config", func(t *testing.T) {
		config := &KubernetesConfig{
			KubeConfigFile: "/invalid/path",
			QPS:            newQps,
			Burst:          newBurst,
		}

		// Mock clientcmd.BuildConfigFromFlags 失败
		patches := gomonkey.ApplyFunc(clientcmd.BuildConfigFromFlags, func(masterUrl, kubeconfigPath string) (
			*rest.Config, error) {
			if kubeconfigPath == "/invalid/path" {
				return nil, errors.New("build config error")
			}
			// 对于默认配置文件，模拟成功
			return &rest.Config{}, nil
		})

		// Mock rest.InClusterConfig 成功
		patches.ApplyFunc(rest.InClusterConfig, func() (*rest.Config, error) {
			return &rest.Config{
				Host:  "https://in-cluster-server",
				QPS:   float32(qps), // 原始值
				Burst: burst,        // 原始值
			}, nil
		})

		// Mock getDefaultKubeConfigFile 返回有效的配置文件路径
		patches.ApplyFunc(getDefaultKubeConfigFile, func() string {
			return "/valid/kubeconfig"
		})
		defer patches.Reset()

		result := GetKubeConfigOrInClusterConfig(config)
		assert.NotNil(t, result)
		// 验证 QPS 和 Burst 被正确覆盖
		assert.Equal(t, newQps, result.QPS)
		assert.Equal(t, newBurst, result.Burst)
	})
}

func TestGetKubeConfigOrInClusterConfigNilConfig(t *testing.T) {
	t.Run("nil config parameter", func(t *testing.T) {
		// Mock rest.InClusterConfig 成功
		patches := gomonkey.ApplyFunc(rest.InClusterConfig, func() (*rest.Config, error) {
			return &rest.Config{
				Host:  "https://in-cluster-server",
				QPS:   float32(qps) + 1,
				Burst: burst + 1,
			}, nil
		})
		defer patches.Reset()

		result := GetKubeConfigOrInClusterConfig(nil)
		assert.NotNil(t, result)
		// 验证使用了默认的 QPS 和 Burst 值
		assert.Equal(t, float32(qps), result.QPS)
		assert.Equal(t, burst, result.Burst)
	})
}

func TestGetKubernetesClientCreationFails(t *testing.T) {
	t.Run("kubernetes client creation fails", func(t *testing.T) {
		// Mock GetKubeConfigOrInClusterConfig
		patches := gomonkey.ApplyFunc(GetKubeConfigOrInClusterConfig, func(k8sConfig *KubernetesConfig) *rest.Config {
			return &rest.Config{}
		})

		// Mock kubernetes.NewForConfig 返回错误
		patches.ApplyFunc(kubernetes.NewForConfig, func(c *rest.Config) (kubernetes.Interface, error) {
			return nil, errors.New("client creation error")
		})

		// Mock zlog.LogFatalf 来避免测试退出
		patches.ApplyFunc(zlog.LogFatalf, func(format string, args ...interface{}) {
			// 不执行任何操作
		})
		defer patches.Reset()

		assert.NotPanics(t, func() {
			client := GetKubernetesClient(&KubernetesConfig{})
			assert.Nil(t, client)
		})
	})
}
