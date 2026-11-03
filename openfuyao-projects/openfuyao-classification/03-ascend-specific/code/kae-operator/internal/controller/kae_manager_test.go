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

package controller

import (
	"context"
	"errors"
	"os"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	v1 "gopkg.openfuyao.cn/kae-operator/api/v1"
	"gopkg.openfuyao.cn/kae-operator/internal/constants"
)

// TestDeployComponent 部署测试，不启用特性时，返回 Disabled 状态
func TestDeployComponentDisabled(t *testing.T) {
	m := &KAEPolicyManager{}

	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.Getenv, func(key string) string {
		return "test"
	})
	kaePolicy := &v1.KAEPolicy{}
	reconciler := &KAEPolicyReconciler{
		Client: fake.NewClientBuilder().Build(),
		Scheme: runtime.NewScheme(),
	}
	patches.ApplyGlobalVar(&componentPaths, []string{
		"../../assets/component-driver",
		"../../assets/component-device-plugin-hpre",
	})
	err := m.init(context.TODO(), reconciler, kaePolicy)
	assert.Nil(t, err)
	state, err := m.deployComponent()
	assert.Nil(t, err)
	assert.Equal(t, v1.Disabled, state)
}

// TestDeployComponentFailed 部署组件 NotReady
func TestDeployComponentFailed(t *testing.T) {
	m := &KAEPolicyManager{}
	reconciler := &KAEPolicyReconciler{
		Client: fake.NewClientBuilder().Build(),
		Scheme: runtime.NewScheme(),
	}
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	// 测试用例2：处理函数返回错误
	patches.Reset()
	patches.ApplyFunc(os.Getenv, func(key string) string {
		return "test"
	})
	kaePolicy := &v1.KAEPolicy{
		Spec: v1.KAEPolicySpec{
			Driver:       v1.DriverSpec{Enabled: BoolPtr(true)},
			DevicePlugin: v1.DevicePluginSpec{EnabledHPRE: BoolPtr(true)},
		},
	}

	patches.ApplyGlobalVar(&componentPaths, []string{
		"../../assets/component-driver",
		"../../assets/component-device-plugin-hpre",
	})

	patches.ApplyFunc(HandlerServiceAccount, func(manager KAEPolicyManager) (v1.State, error) {
		return v1.NotReady, errors.New("test")
	})
	err := m.init(context.TODO(), reconciler, kaePolicy)
	assert.Nil(t, err)
	state, err := m.deployComponent()
	assert.NotNil(t, err)
	assert.Equal(t, v1.NotReady, state)
}

// BoolPtr 返回bool类型指针
func BoolPtr(b bool) *bool {
	return &b
}

func TestKAEPolicyManagerIsComponentEnabled(t *testing.T) {
	testKaePolicy := &v1.KAEPolicy{
		Spec: v1.KAEPolicySpec{
			Driver: v1.DriverSpec{
				Enabled: BoolPtr(true),
			},
			DevicePlugin: v1.DevicePluginSpec{
				EnabledHPRE: BoolPtr(true),
				EnabledSEC:  BoolPtr(true),
				EnabledZIP:  BoolPtr(true),
			},
		},
	}
	tests := []struct {
		name          string
		componentName string
		want          bool
	}{
		{
			"DriverDaemonSetName",
			constants.ComponentDriver,
			true,
		},
		{
			"HpreDevicePluginDaemonSetName",
			constants.ComponentDriverPluginHpre,
			true,
		},
		{
			"SecDevicePluginDaemonSetName",
			constants.ComponentDriverPluginSec,
			true,
		},
		{
			"ZipDevicePluginDaemonSetName",
			constants.ComponentDriverPluginZip,
			true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &KAEPolicyManager{
				kaeInstance: testKaePolicy,
			}
			assert.Equal(t, tt.want, m.isComponentEnabled(tt.componentName))
		})
	}
}
