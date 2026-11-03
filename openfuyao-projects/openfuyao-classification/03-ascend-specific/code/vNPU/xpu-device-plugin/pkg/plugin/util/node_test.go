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

package util

import (
	"fmt"
	"os/exec"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"

	"huawei.com/vxpu-device-plugin/pkg/lock"
	"huawei.com/vxpu-device-plugin/pkg/plugin/config"
	"huawei.com/vxpu-device-plugin/pkg/plugin/xpu"
)

func newNodeConfigMap(yamlContent string) *v1.ConfigMap {
	return &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      nodeConfigCmName,
			Namespace: "xpu",
		},
		Data: map[string]string{
			nodeConfigYamlName: yamlContent,
		},
	}
}

func newTemplateConfigMap(yamlContent string) *v1.ConfigMap {
	return &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      templateConfigCmName,
			Namespace: "xpu",
		},
		Data: map[string]string{
			templateConfigYamlName: yamlContent,
		},
	}
}

func TestParseNodeModeFromConfigMap(t *testing.T) {
	yamlContent := `
nodeConfigs:
- name: node1
  vnpu-mode: soft
- name: node2
  vnpu-mode: hard
`
	configMap := newNodeConfigMap(yamlContent)
	lock.SetClient(fake.NewClientset(configMap))

	t.Setenv("POD_NAMESPACE", "xpu")
	config.NodeName = "node1"
	config.NodeMode = ""
	parseNodeModeFromConfigMap()
	if config.NodeMode != "soft" {
		t.Error("parse node mode from configmap failed")
	}

	config.NodeName = "node2"
	config.NodeMode = ""
	parseNodeModeFromConfigMap()
	if config.NodeMode != "hard" {
		t.Error("parse node mode from configmap failed")
	}

	config.NodeName = "node3"
	config.NodeMode = ""
	parseNodeModeFromConfigMap()
	if config.NodeMode != "soft" {
		t.Error("parse node mode from configmap failed")
	}
}

func TestParseEmptyNodeModeFromConfigMap(t *testing.T) {
	yamlContent := `
nodeConfigs:
`
	configMap := newNodeConfigMap(yamlContent)
	lock.SetClient(fake.NewClientset(configMap))

	t.Setenv("POD_NAMESPACE", "xpu")
	config.NodeName = "node"
	config.NodeMode = ""
	parseNodeModeFromConfigMap()
	if config.NodeMode != "soft" {
		t.Error("parse node mode from configmap failed")
	}
}

func TestParseNodeTemplateFromConfigMap(t *testing.T) {
	yamlContent := `
vnpus:
- chipName: Ascend910B
  aiCore: 20
  aiCPU: 8
  templates:
  - name: vir01
    aiCore: 1
    aiCPU: 1
`
	configMap := newTemplateConfigMap(yamlContent)
	lock.SetClient(fake.NewClientset(configMap))
	t.Setenv("POD_NAMESPACE", "xpu")

	parseNodeTemplateFromConfigMap()
	if config.NodeTemplateInfos == nil || len(config.NodeTemplateInfos) != 1 {
		t.Error("parse node template from configmap failed")
	}
}

func TestParseNodeTemplateFromConfigMap_Error(t *testing.T) {
	lock.SetClient(fake.NewClientset())
	t.Setenv("POD_NAMESPACE", "xpu")
	parseNodeTemplateFromConfigMap()
}

func TestSetNodeConfig_SoftMode(t *testing.T) {
	yamlContent := `
nodeConfigs:
- name: node-soft
  vnpu-mode: soft
`
	configMap := newNodeConfigMap(yamlContent)
	templateCm := newTemplateConfigMap("vnpus: []")
	lock.SetClient(fake.NewClientset(configMap, templateCm))
	t.Setenv("POD_NAMESPACE", "xpu")
	config.NodeName = "node-soft"

	patch := gomonkey.ApplyFunc(xpu.GetCardIDs, func() ([]int32, error) {
		return []int32{0}, nil
	})
	defer patch.Reset()
	patchExec := gomonkey.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		return &exec.Cmd{Path: "/bin/true", Args: append([]string{"true"}, arg...)}
	})
	defer patchExec.Reset()

	SetNodeConfig()
}

func TestEnableDeviceShareMode_Error(t *testing.T) {
	patch := gomonkey.ApplyFunc(xpu.GetCardIDs, func() ([]int32, error) {
		return nil, fmt.Errorf("get card error")
	})
	defer patch.Reset()
	enableDeviceShareMode()
}

func TestEnableDeviceShareMode_NoCards(t *testing.T) {
	patch := gomonkey.ApplyFunc(xpu.GetCardIDs, func() ([]int32, error) {
		return []int32{}, nil
	})
	defer patch.Reset()
	enableDeviceShareMode()
}

func TestDisablePresetVirtualDevice(t *testing.T) {
	patchExec := gomonkey.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		return &exec.Cmd{Path: "/bin/true", Args: append([]string{"true"}, arg...)}
	})
	defer patchExec.Reset()
	disablePresetVirtualDevice()
}
