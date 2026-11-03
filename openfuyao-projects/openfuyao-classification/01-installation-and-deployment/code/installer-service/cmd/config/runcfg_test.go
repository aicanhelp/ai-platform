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

package config

import (
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"installer-service/pkg/client/k8s"
	"installer-service/pkg/server/runtime"
)

func TestNewRunConfig(t *testing.T) {
	// 提前mock k8s和runtime的构造函数，防止执行真实逻辑
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	// 1. 完全mock k8s.NewKubernetesCfg，返回空对象避免配置加载
	mockK8sCfg := &k8s.KubernetesCfg{}
	patches.ApplyFunc(k8s.NewKubernetesCfg, func() *k8s.KubernetesCfg {
		return mockK8sCfg
	})

	// 2. mock runtime.NewServerConfig，返回其真实默认值
	expectedServerCfg := runtime.NewServerConfig()
	patches.ApplyFunc(runtime.NewServerConfig, func() *runtime.ServerConfig {
		return expectedServerCfg
	})

	cfg := NewRunConfig()
	assert.NotNil(t, cfg)
	assert.Equal(t, expectedServerCfg, cfg.Server)
	assert.Equal(t, mockK8sCfg, cfg.KubernetesCfg)
}

func TestRunConfigValidate(t *testing.T) {
	tests := []struct {
		name       string
		serverErrs []error
		k8sErrs    []error
		wantLen    int
	}{
		{name: "no errors", serverErrs: []error{}, k8sErrs: []error{}, wantLen: 0},
		{name: "server errors", serverErrs: []error{assert.AnError}, k8sErrs: []error{}, wantLen: 1},
		{name: "k8s errors", serverErrs: []error{}, k8sErrs: []error{assert.AnError, assert.AnError}, wantLen: 2},
		{name: "both errors", serverErrs: []error{assert.AnError}, k8sErrs: []error{assert.AnError}, wantLen: 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			serverCfg := &runtime.ServerConfig{}
			patches.ApplyMethod(serverCfg, "Validate", func() []error { return tt.serverErrs })

			k8sCfg := &k8s.KubernetesCfg{}
			patches.ApplyMethod(k8sCfg, "Validate", func() []error { return tt.k8sErrs })

			errs := (&RunConfig{Server: serverCfg, KubernetesCfg: k8sCfg}).Validate()
			assert.Len(t, errs, tt.wantLen)
		})
	}
}
