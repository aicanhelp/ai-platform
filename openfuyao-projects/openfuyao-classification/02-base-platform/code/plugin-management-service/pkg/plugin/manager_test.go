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

package plugin

import (
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/dynamic/fake"
)

func TestCheckPluginInstallment(t *testing.T) {
	scheme := runtime.NewScheme()
	
	pluginName := "test-plugin"
	cp := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "console.openfuyao.com/v1beta1",
			"kind":       "ConsolePlugin",
			"metadata": map[string]interface{}{
				"name": pluginName,
			},
			"spec": map[string]interface{}{
				"enabled": true,
			},
		},
	}

	tests := []struct {
		name       string
		pluginName string
		existing   []runtime.Object
		want       bool
	}{
		{
			name:       "plugin exists",
			pluginName: pluginName,
			existing:   []runtime.Object{cp},
			want:       true,
		},
		{
			name:       "plugin does not exist",
			pluginName: "non-existent",
			existing:   []runtime.Object{},
			want:       false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := fake.NewSimpleDynamicClient(scheme, tt.existing...)
			cm := &ConsolePluginManager{Client: client}
			if got := cm.CheckPluginInstallment(tt.pluginName); got != tt.want {
				t.Errorf("CheckPluginInstallment() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestListConsolePlugins(t *testing.T) {
	scheme := runtime.NewScheme()
	cp1 := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "console.openfuyao.com/v1beta1",
			"kind":       "ConsolePlugin",
			"metadata": map[string]interface{}{
				"name": "plugin-1",
			},
		},
	}
	cp2 := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "console.openfuyao.com/v1beta1",
			"kind":       "ConsolePlugin",
			"metadata": map[string]interface{}{
				"name": "plugin-2",
			},
		},
	}

	client := fake.NewSimpleDynamicClient(scheme, cp1, cp2)
	cm := &ConsolePluginManager{Client: client}

	plugins, err := cm.ListConsolePlugins()
	if err != nil {
		t.Fatalf("ListConsolePlugins() error = %v", err)
	}

	if len(plugins) != 2 {
		t.Errorf("Expected 2 plugins, got %d", len(plugins))
	}
}

func TestSetPluginEnablementIfInstalled(t *testing.T) {
	scheme := runtime.NewScheme()
	pluginName := "test-plugin"
	cp := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "console.openfuyao.com/v1beta1",
			"kind":       "ConsolePlugin",
			"metadata": map[string]interface{}{
				"name": pluginName,
			},
			"spec": map[string]interface{}{
				"enabled": false,
			},
		},
	}

	client := fake.NewSimpleDynamicClient(scheme, cp)
	cm := &ConsolePluginManager{Client: client}

	// Test enabling
	err := cm.SetPluginEnablementIfInstalled(pluginName, true)
	if err != nil {
		t.Errorf("SetPluginEnablementIfInstalled() error = %v", err)
	}

	// Test skip patching if already enabled
	err = cm.SetPluginEnablementIfInstalled(pluginName, true)
	if err != nil {
		t.Errorf("SetPluginEnablementIfInstalled() error = %v", err)
	}

	// Test plugin not found
	err = cm.SetPluginEnablementIfInstalled("non-existent", true)
	if err == nil {
		t.Errorf("Expected error for non-existent plugin, got nil")
	}
}

func TestCheckPluginEnablementIfInstalled(t *testing.T) {
	scheme := runtime.NewScheme()
	pluginName := "test-plugin"
	cp := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "console.openfuyao.com/v1beta1",
			"kind":       "ConsolePlugin",
			"metadata": map[string]interface{}{
				"name": pluginName,
			},
			"spec": map[string]interface{}{
				"enabled": true,
			},
		},
	}

	client := fake.NewSimpleDynamicClient(scheme, cp)
	cm := &ConsolePluginManager{Client: client}

	enabled, err := cm.CheckPluginEnablementIfInstalled(pluginName)
	if err != nil {
		t.Fatalf("CheckPluginEnablementIfInstalled() error = %v", err)
	}
	if !enabled {
		t.Errorf("Expected plugin to be enabled")
	}

	_, err = cm.CheckPluginEnablementIfInstalled("non-existent")
	if err == nil {
		t.Errorf("Expected error for non-existent plugin")
	}
}
