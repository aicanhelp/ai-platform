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

package k8s

import (
	"os"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/util/homedir"
)

func TestGetKubeConfig(t *testing.T) {
	// 模拟 InClusterConfig 成功
	t.Run("集群内配置成功", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		// 打桩 InClusterConfig 返回成功
		patches.ApplyFunc(rest.InClusterConfig, func() (*rest.Config, error) {
			return &rest.Config{Host: "cluster-host"}, nil
		})

		config := GetKubeConfig()
		assert.Equal(t, "cluster-host", config.Host)
	})
}

func TestGetKubeConfigFile(t *testing.T) {
	t.Run("所有方法失败", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		patches.ApplyFunc(homedir.HomeDir, func() string {
			return ""
		})

		result := getKubeConfigFile()
		assert.Equal(t, "", result)
	})

	t.Run("文件不存在", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		patches.ApplyFunc(homedir.HomeDir, func() string {
			return "/home/user"
		})

		patches.ApplyFunc(os.Stat, func(name string) (os.FileInfo, error) {
			return nil, os.ErrNotExist
		})

		result := getKubeConfigFile()
		assert.Equal(t, "", result)
	})
}

func TestKubernetesCfgValidate(t *testing.T) {
	t.Run("所有验证通过", testValidateAllPass)
	t.Run("文件不存在", testValidateFileNotExist)
	t.Run("KubeConfig为空", testValidateKubeConfigNil)
	t.Run("文件不存在且KubeConfig为空", testValidateFileAndKubeConfigNil)
	t.Run("KubeConfigFile为空", testValidateKubeConfigFileEmpty)
}

func testValidateAllPass(t *testing.T) {
	cfg := &KubernetesCfg{
		KubeConfigFile: "/path/to/valid/kubeconfig",
		KubeConfig:     &rest.Config{},
	}

	// 打桩 os.Stat 返回文件存在
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(os.Stat, func(name string) (os.FileInfo, error) {
		return nil, nil
	})

	errs := cfg.Validate()
	assert.Empty(t, errs)
}

func testValidateFileNotExist(t *testing.T) {
	cfg := &KubernetesCfg{
		KubeConfigFile: "/path/to/invalid/kubeconfig",
		KubeConfig:     &rest.Config{},
	}

	// 打桩 os.Stat 返回文件不存在
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(os.Stat, func(name string) (os.FileInfo, error) {
		return nil, os.ErrNotExist
	})

	errs := cfg.Validate()
	assert.Len(t, errs, 1)
	assert.ErrorIs(t, errs[0], os.ErrNotExist)
}

func testValidateKubeConfigNil(t *testing.T) {
	cfg := &KubernetesCfg{
		KubeConfigFile: "/path/to/valid/kubeconfig",
		KubeConfig:     nil,
	}

	// 打桩 os.Stat 返回文件存在
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(os.Stat, func(name string) (os.FileInfo, error) {
		return nil, nil
	})

	errs := cfg.Validate()
	assert.Len(t, errs, 1)
	assert.EqualError(t, errs[0], "k8s config get nil")
}

const length = 2

func testValidateFileAndKubeConfigNil(t *testing.T) {
	cfg := &KubernetesCfg{
		KubeConfigFile: "/path/to/invalid/kubeconfig",
		KubeConfig:     nil,
	}

	// 打桩 os.Stat 返回文件不存在
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(os.Stat, func(name string) (os.FileInfo, error) {
		return nil, os.ErrNotExist
	})

	errs := cfg.Validate()
	assert.Len(t, errs, length)
	assert.ErrorIs(t, errs[0], os.ErrNotExist)
	assert.EqualError(t, errs[1], "k8s config get nil")
}

func testValidateKubeConfigFileEmpty(t *testing.T) {
	cfg := &KubernetesCfg{
		KubeConfigFile: "",
		KubeConfig:     &rest.Config{},
	}

	errs := cfg.Validate()
	assert.Empty(t, errs)
}

func TestNewKubernetesCfg(t *testing.T) {
	t.Run("空kubeconfig文件路径", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		patches.ApplyFunc(getKubeConfigFile, func() string {
			return ""
		})

		patches.ApplyFunc(GetKubeConfig, func() *rest.Config {
			return &rest.Config{}
		})

		cfg := NewKubernetesCfg()

		assert.Equal(t, "", cfg.KubeConfigFile)
		assert.NotNil(t, cfg.KubeConfig)
	})

	t.Run("空kubeconfig对象", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		patches.ApplyFunc(getKubeConfigFile, func() string {
			return "/path/to/kubeconfig"
		})

		patches.ApplyFunc(GetKubeConfig, func() *rest.Config {
			return nil
		})

		cfg := NewKubernetesCfg()

		assert.Equal(t, "/path/to/kubeconfig", cfg.KubeConfigFile)
		assert.Nil(t, cfg.KubeConfig)
	})

}
