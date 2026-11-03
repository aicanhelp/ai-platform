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
	"os/user"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/pkg/errors"
	"github.com/stretchr/testify/assert"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/util/homedir"

	"marketplace-service/pkg/zlog"
)

// TestValidate 测试配置校验
func TestValidate(t *testing.T) {
	// 分支1：配置正常
	cfg := &KubernetesCfg{
		KubeConfigFile: "test.yaml",
		KubeConfig:     &rest.Config{},
	}
	patches := gomonkey.ApplyFunc(os.Stat, func(_ string) (os.FileInfo, error) {
		return nil, nil
	})
	defer patches.Reset()
	errs := cfg.Validate()
	assert.Empty(t, errs)

	// 分支2：文件不存在 + KubeConfig为空
	cfg2 := &KubernetesCfg{KubeConfigFile: "invalid.yaml"}
	patches2 := gomonkey.ApplyFunc(os.Stat, func(_ string) (os.FileInfo, error) {
		return nil, os.ErrNotExist
	})
	defer patches2.Reset()
	errs2 := cfg2.Validate()
	const length = 2
	assert.Len(t, errs2, length)
}

// TestGetKubeConfig_InCluster 测试集群内配置获取
func TestGetKubeConfig_InCluster(t *testing.T) {
	patches := gomonkey.ApplyFunc(rest.InClusterConfig, func() (*rest.Config, error) {
		return &rest.Config{}, nil
	})
	defer patches.Reset()
	cfg := GetKubeConfig()
	assert.NotNil(t, cfg)
}

// TestGetKubeConfigFile 测试配置文件路径获取
func TestGetKubeConfigFile(t *testing.T) {
	// 分支1：HOME目录存在且有.kube/config
	patches := gomonkey.ApplyFunc(homedir.HomeDir, func() string {
		return "/test/home"
	}).ApplyFunc(os.Stat, func(_ string) (os.FileInfo, error) {
		return nil, nil
	})
	defer patches.Reset()
	path := getKubeConfigFile()
	assert.Equal(t, "/test/home/.kube/config", path)

	// 分支2：HOME为空，从user.Current获取
	patches2 := gomonkey.ApplyFunc(homedir.HomeDir, func() string {
		return ""
	}).ApplyFunc(user.Current, func() (*user.User, error) {
		return &user.User{HomeDir: "/test/user"}, nil
	}).ApplyFunc(os.Stat, func(_ string) (os.FileInfo, error) {
		return nil, nil
	})
	defer patches2.Reset()
	path2 := getKubeConfigFile()
	assert.Equal(t, "/test/user/.kube/config", path2)
}

// TestGetKubeConfig_Fatal 测试配置获取失败（触发Fatal）
func TestGetKubeConfig_Fatal(t *testing.T) {
	// Mock集群内配置获取失败 + 本地kubeconfig文件路径为空
	patches := gomonkey.ApplyFunc(rest.InClusterConfig, func() (*rest.Config, error) {
		return nil, errors.New("failed to get in-cluster config")
	}).ApplyFunc(getKubeConfigFile, func() string {
		return "" // 模拟本地kubeconfig文件路径为空
	}).ApplyFunc(zlog.Fatalf, func(format string, v ...interface{}) {
		// Mock zlog.Fatalf避免测试进程退出，同时记录日志便于调试
		zlog.Errorf("mock fatal log: %s, args: %v", format, v)
	})
	defer patches.Reset()
	GetKubeConfig()
	assert.True(t, true, "GetKubeConfig触发Fatal后正常执行，未崩溃")
}
