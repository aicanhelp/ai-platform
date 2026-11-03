/*
 *
 *  * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  * openFuyao is licensed under Mulan PSL v2.
 *  * You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  * You may obtain a copy of Mulan PSL v2 at:
 *  *          http://license.coscl.org.cn/MulanPSL2
 *  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *  * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *  * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *  * See the Mulan PSL v2 for more details.
 *
 */

package controller

import (
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestNodeConfigDeepCopy(t *testing.T) {
	// 创建一个测试用的 NodeConfig 实例
	originalTime := metav1.Now()
	originalConfig := &NodeConfig{
		TypeMeta: metav1.TypeMeta{
			Kind:       "NodeConfig",
			APIVersion: "volcano.sh/v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-node-config",
			Namespace: "default",
			Labels: map[string]string{
				"test": "true",
			},
		},
		Spec: NodeConfigSpec{
			Nodes: []NodeConfiguration{
				{
					NodeName:               "node-1",
					EnableStaticCPUManager: true,
					EnableNUMADistance:     false,
					KubeletConfigPath:      "/etc/kubernetes/kubelet-config.yaml",
				},
				{
					NodeName:               "node-2",
					EnableStaticCPUManager: false,
					EnableNUMADistance:     true,
					KubeletConfigPath:      "/etc/kubernetes/kubelet-config-2.yaml",
				},
			},
		},
		Status: NodeConfigStatus{
			NodeStatuses: []NodeStatus{
				{
					NodeName:                "node-1",
					StaticCPUManagerEnabled: true,
					NUMADistanceEnabled:     false,
					State:                   NodeConfigStateConfigured,
					LastUpdateTime:          originalTime,
				},
				{
					NodeName:                "node-2",
					StaticCPUManagerEnabled: false,
					NUMADistanceEnabled:     true,
					State:                   NodeConfigStateConfiguring,
					Error:                   "配置中...",
					LastUpdateTime:          originalTime,
				},
			},
			LastUpdateTime: originalTime,
		},
	}

	// 执行深拷贝
	copiedConfig := originalConfig.DeepCopy()

	// 测试拷贝是否正确
	if !reflect.DeepEqual(originalConfig, copiedConfig) {
		t.Errorf("DeepCopy() 结果与原对象不匹配")
	}

	// 修改原对象，确保拷贝的对象不受影响
	originalConfig.Name = "modified-name"
	originalConfig.Spec.Nodes[0].NodeName = "modified-node-1"
	originalConfig.Status.NodeStatuses[0].State = NodeConfigStateFailed

	// 验证修改不影响拷贝对象
	if copiedConfig.Name == "modified-name" {
		t.Errorf("修改原对象的 Name 影响了拷贝对象")
	}

	if copiedConfig.Spec.Nodes[0].NodeName == "modified-node-1" {
		t.Errorf("修改原对象的 NodeName 影响了拷贝对象")
	}

	if copiedConfig.Status.NodeStatuses[0].State == NodeConfigStateFailed {
		t.Errorf("修改原对象的 State 影响了拷贝对象")
	}

	// 测试 DeepCopyObject 方法
	objCopy := originalConfig.DeepCopyObject()
	copiedFromObj, ok := objCopy.(*NodeConfig)
	if !ok {
		t.Errorf("DeepCopyObject() 返回的类型不正确")
	}

	if !reflect.DeepEqual(originalConfig, copiedFromObj) {
		t.Errorf("DeepCopyObject() 结果与原对象不匹配")
	}
}

func TestNodeConfigListDeepCopy(t *testing.T) {
	// 创建测试用的 NodeConfigList
	originalTime := metav1.Now()
	originalList := &NodeConfigList{
		TypeMeta: metav1.TypeMeta{
			Kind:       "NodeConfigList",
			APIVersion: "volcano.sh/v1",
		},
		ListMeta: metav1.ListMeta{
			ResourceVersion: "1",
		},
		Items: []NodeConfig{
			{
				TypeMeta: metav1.TypeMeta{
					Kind:       "NodeConfig",
					APIVersion: "volcano.sh/v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "config-1",
					Namespace: "default",
				},
				Spec: NodeConfigSpec{
					Nodes: []NodeConfiguration{
						{
							NodeName:               "node-1",
							EnableStaticCPUManager: true,
							EnableNUMADistance:     false,
						},
					},
				},
				Status: NodeConfigStatus{
					NodeStatuses: []NodeStatus{
						{
							NodeName:                "node-1",
							StaticCPUManagerEnabled: true,
							NUMADistanceEnabled:     false,
							State:                   NodeConfigStateConfigured,
							LastUpdateTime:          originalTime,
						},
					},
					LastUpdateTime: originalTime,
				},
			},
		},
	}

	// 执行深拷贝
	copiedList := originalList.DeepCopy()

	// 测试拷贝是否正确
	if !reflect.DeepEqual(originalList, copiedList) {
		t.Errorf("DeepCopy() 结果与原对象不匹配")
	}

	// 修改原对象，确保拷贝的对象不受影响
	originalList.ResourceVersion = "2"
	originalList.Items[0].Name = "modified-config-1"
	originalList.Items[0].Spec.Nodes[0].NodeName = "modified-node-1"

	// 验证修改不影响拷贝对象
	if copiedList.ResourceVersion == "2" {
		t.Errorf("修改原对象的 ResourceVersion 影响了拷贝对象")
	}

	if copiedList.Items[0].Name == "modified-config-1" {
		t.Errorf("修改原对象的 Item Name 影响了拷贝对象")
	}

	if copiedList.Items[0].Spec.Nodes[0].NodeName == "modified-node-1" {
		t.Errorf("修改原对象的 NodeName 影响了拷贝对象")
	}

	// 测试 DeepCopyObject 方法
	objCopy := originalList.DeepCopyObject()
	copiedFromObj, ok := objCopy.(*NodeConfigList)
	if !ok {
		t.Errorf("DeepCopyObject() 返回的类型不正确")
	}

	if !reflect.DeepEqual(originalList, copiedFromObj) {
		t.Errorf("DeepCopyObject() 结果与原对象不匹配")
	}
}

func TestAddKnownTypes(t *testing.T) {
	// 创建测试用的 Scheme
	testScheme := runtime.NewScheme()

	// 添加已知类型到 Scheme
	err := addKnownTypes(testScheme)
	if err != nil {
		t.Fatalf("添加已知类型到 Scheme 失败: %v", err)
	}

	// 验证类型是否已正确添加
	gv := schema.GroupVersion{Group: "volcano.sh", Version: "v1"}

	// 检查 NodeConfig 类型
	obj, err := testScheme.New(gv.WithKind("NodeConfig"))
	if err != nil {
		t.Errorf("无法通过 Scheme 创建 NodeConfig: %v", err)
	}

	_, ok := obj.(*NodeConfig)
	if !ok {
		t.Errorf("从 Scheme 创建的对象不是 NodeConfig 类型")
	}

	// 检查 NodeConfigList 类型
	listObj, err := testScheme.New(gv.WithKind("NodeConfigList"))
	if err != nil {
		t.Errorf("无法通过 Scheme 创建 NodeConfigList: %v", err)
	}

	_, ok = listObj.(*NodeConfigList)
	if !ok {
		t.Errorf("从 Scheme 创建的对象不是 NodeConfigList 类型")
	}
}

func TestNodeConfigState(t *testing.T) {
	// 测试常量定义是否正确
	if NodeConfigStateConfiguring != "Configuring" {
		t.Errorf("NodeConfigStateConfiguring 常量值不正确")
	}

	if NodeConfigStateConfigured != "Configured" {
		t.Errorf("NodeConfigStateConfigured 常量值不正确")
	}

	if NodeConfigStateFailed != "Failed" {
		t.Errorf("NodeConfigStateFailed 常量值不正确")
	}
}

func TestSetupScheme(t *testing.T) {
	// 测试 builder 和 addToScheme 变量
	if builder == nil {
		t.Errorf("builder 不应为 nil")
	}

	if addToScheme == nil {
		t.Errorf("addToScheme 不应为 nil")
	}

	// 测试 GroupVersion
	expectedGV := schema.GroupVersion{Group: "volcano.sh", Version: "v1"}
	if groupVersion != expectedGV {
		t.Errorf("groupVersion 不正确，期望 %v，实际 %v", expectedGV, groupVersion)
	}

	// 创建新的 Scheme 并应用 addToScheme
	testScheme := runtime.NewScheme()
	err := addToScheme(testScheme)
	if err != nil {
		t.Fatalf("通过 addToScheme 添加类型失败: %v", err)
	}

	// 验证类型是否已添加
	gvk := schema.GroupVersionKind{
		Group:   "volcano.sh",
		Version: "v1",
		Kind:    "NodeConfig",
	}

	if !testScheme.Recognizes(gvk) {
		t.Errorf("Scheme 无法识别 NodeConfig 类型")
	}
}
