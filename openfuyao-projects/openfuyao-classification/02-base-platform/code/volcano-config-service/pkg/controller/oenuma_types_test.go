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

const (
	updatedReplicaCount = 10
	updatedNumaCount    = 10
)

// TestSchemeGroupVersion 测试 GroupVersion 是否正确
func TestSchemeGroupVersion(t *testing.T) {
	expected := schema.GroupVersion{Group: "resource.sop.huawei.com", Version: "v1alpha1"}
	if schemeGroupVersion != expected {
		t.Errorf("期望 schemeGroupVersion 为 %v，实际为 %v", expected, schemeGroupVersion)
	}
}

// TestAddOenumaKnownTypes 测试将自定义类型添加到 scheme
func TestAddOenumaKnownTypes(t *testing.T) {
	scheme := runtime.NewScheme()
	err := addOenumaKnownTypes(scheme)
	if err != nil {
		t.Fatalf("添加类型到 scheme 失败: %v", err)
	}

	// 验证类型是否已添加
	gvk := schema.GroupVersionKind{
		Group:   "resource.sop.huawei.com",
		Version: "v1alpha1",
		Kind:    "Oenuma",
	}

	obj, err := scheme.New(gvk)
	if err != nil {
		t.Errorf("无法创建 Oenuma 对象: %v", err)
	}

	_, ok := obj.(*Oenuma)
	if !ok {
		t.Errorf("创建的对象不是 *Oenuma 类型")
	}

	// 测试 OenumaList 类型
	gvkList := schema.GroupVersionKind{
		Group:   "resource.sop.huawei.com",
		Version: "v1alpha1",
		Kind:    "OenumaList",
	}

	listObj, err := scheme.New(gvkList)
	if err != nil {
		t.Errorf("无法创建 OenumaList 对象: %v", err)
	}

	_, ok = listObj.(*OenumaList)
	if !ok {
		t.Errorf("创建的对象不是 *OenumaList 类型")
	}
}

// TestOenumaDeepCopy 测试 Oenuma 的 DeepCopy 方法
func TestOenumaDeepCopy(t *testing.T) {
	original := createTestOenuma()
	copied := original.DeepCopy()

	// 验证拷贝是否成功
	if !reflect.DeepEqual(original, copied) {
		t.Errorf("DeepCopy 结果与原对象不匹配")
	}

	// 修改原对象，确保不影响拷贝对象
	original.Name = "changed-name"
	original.Spec.Replicas = updatedReplicaCount
	original.Spec.Node[0].Name = "changed-node"
	original.Spec.Node[0].Numa[0].CPUSet = "changed-cpuset"
	original.Spec.Node[0].PodAffinity[0].Containers[0].Name = "changed-container"

	// 验证修改不影响拷贝对象
	if copied.Name == "changed-name" {
		t.Errorf("修改原对象的 Name 影响了拷贝对象")
	}
	if copied.Spec.Replicas == updatedReplicaCount {
		t.Errorf("修改原对象的 Replicas 影响了拷贝对象")
	}
	if copied.Spec.Node[0].Name == "changed-node" {
		t.Errorf("修改原对象的 Node Name 影响了拷贝对象")
	}
	if copied.Spec.Node[0].Numa[0].CPUSet == "changed-cpuset" {
		t.Errorf("修改原对象的 CPUSet 影响了拷贝对象")
	}
	if copied.Spec.Node[0].PodAffinity[0].Containers[0].Name == "changed-container" {
		t.Errorf("修改原对象的 Container Name 影响了拷贝对象")
	}

	// 测试 nil 情况
	var nilOenuma *Oenuma
	nilCopy := (*Oenuma).DeepCopy(nilOenuma)
	if nilCopy == nil {
		t.Log("nil 对象的 DeepCopy 正确返回 nil")
	} else {
		t.Errorf("nil 对象的 DeepCopy 应返回 nil，但返回了 %v", nilCopy)
	}
}

// TestOenumaDeepCopyObject 测试 Oenuma 的 DeepCopyObject 方法
func TestOenumaDeepCopyObject(t *testing.T) {
	original := createTestOenuma()
	objCopy := original.DeepCopyObject()

	// 验证类型转换
	copied, ok := objCopy.(*Oenuma)
	if !ok {
		t.Fatalf("DeepCopyObject 返回的不是 *Oenuma 类型")
	}

	// 验证内容一致
	if !reflect.DeepEqual(original, copied) {
		t.Errorf("DeepCopyObject 结果与原对象不匹配")
	}

	// 测试 nil 情况
	var nilOenuma *Oenuma
	nilCopy := (*Oenuma).DeepCopyObject(nilOenuma)
	if nilCopy == nil {
		t.Log("nil 对象的 DeepCopyObject 正确返回 nil")
	} else {
		t.Errorf("nil 对象的 DeepCopyObject 应返回 nil，但返回了 %v", nilCopy)
	}
}

// TestOenumaListDeepCopy 测试 OenumaList 的 DeepCopy 方法
func TestOenumaListDeepCopy(t *testing.T) {
	original := createTestOenumaList()
	copied := original.DeepCopy()

	// 验证拷贝是否成功
	if !reflect.DeepEqual(original, copied) {
		t.Errorf("DeepCopy 结果与原对象不匹配")
	}

	// 修改原对象，确保不影响拷贝对象
	original.Items[0].Name = "changed-name"
	original.Items[0].Spec.Replicas = updatedReplicaCount

	// 验证修改不影响拷贝对象
	if copied.Items[0].Name == "changed-name" {
		t.Errorf("修改原对象的 Item Name 影响了拷贝对象")
	}
	if copied.Items[0].Spec.Replicas == updatedReplicaCount {
		t.Errorf("修改原对象的 Item Replicas 影响了拷贝对象")
	}

	// 测试 nil 情况
	var nilList *OenumaList
	nilCopy := (*OenumaList).DeepCopy(nilList)
	if nilCopy == nil {
		t.Log("nil 对象的 DeepCopy 正确返回 nil")
	} else {
		t.Errorf("nil 对象的 DeepCopy 应返回 nil，但返回了 %v", nilCopy)
	}
}

// TestOenumaListDeepCopyObject 测试 OenumaList 的 DeepCopyObject 方法
func TestOenumaListDeepCopyObject(t *testing.T) {
	original := createTestOenumaList()
	objCopy := original.DeepCopyObject()

	// 验证类型转换
	copied, ok := objCopy.(*OenumaList)
	if !ok {
		t.Fatalf("DeepCopyObject 返回的不是 *OenumaList 类型")
	}

	// 验证内容一致
	if !reflect.DeepEqual(original, copied) {
		t.Errorf("DeepCopyObject 结果与原对象不匹配")
	}

	// 测试 nil 情况
	var nilList *OenumaList
	nilCopy := (*OenumaList).DeepCopyObject(nilList)
	if nilCopy == nil {
		t.Log("nil 对象的 DeepCopyObject 正确返回 nil")
	} else {
		t.Errorf("nil 对象的 DeepCopyObject 应返回 nil，但返回了 %v", nilCopy)
	}
}

// TestOenumaSpecDeepCopy 测试 OenumaSpec 的 DeepCopy 方法
func TestOenumaSpecDeepCopy(t *testing.T) {
	original := createTestOenumaSpec()
	copied := original.DeepCopy()

	// 验证拷贝是否成功
	if !reflect.DeepEqual(original, copied) {
		t.Errorf("DeepCopy 结果与原对象不匹配")
	}

	// 修改原对象，确保不影响拷贝对象
	original.Replicas = updatedReplicaCount
	original.Node[0].Name = "changed-node"

	// 验证修改不影响拷贝对象
	if copied.Replicas == updatedReplicaCount {
		t.Errorf("修改原对象的 Replicas 影响了拷贝对象")
	}
	if copied.Node[0].Name == "changed-node" {
		t.Errorf("修改原对象的 Node Name 影响了拷贝对象")
	}

	// 测试 nil 情况
	var nilSpec *OenumaSpec
	nilCopy := (*OenumaSpec).DeepCopy(nilSpec)
	if nilCopy == nil {
		t.Log("nil 对象的 DeepCopy 正确返回 nil")
	} else {
		t.Errorf("nil 对象的 DeepCopy 应返回 nil，但返回了 %v", nilCopy)
	}
}

// TestOenumaNodeDeepCopy 测试 OenumaNode 的 DeepCopy 方法
func TestOenumaNodeDeepCopy(t *testing.T) {
	original := createTestOenumaNode()
	copied := original.DeepCopy()

	// 使用解引用操作符将指针转换为值
	if !reflect.DeepEqual(original, *copied) {
		t.Errorf("DeepCopy 结果与原对象不匹配")
	}

	// 修改原对象，确保不影响拷贝对象
	original.Name = "changed-node"
	original.Numa[0].CPUSet = "changed-cpuset"
	original.PodAffinity[0].PodName = "changed-pod"

	// 验证修改不影响拷贝对象
	if copied.Name == "changed-node" {
		t.Errorf("修改原对象的 Name 影响了拷贝对象")
	}
	if copied.Numa[0].CPUSet == "changed-cpuset" {
		t.Errorf("修改原对象的 CPUSet 影响了拷贝对象")
	}
	if copied.PodAffinity[0].PodName == "changed-pod" {
		t.Errorf("修改原对象的 PodName 影响了拷贝对象")
	}

	// 测试 nil 情况
	var nilNode *OenumaNode
	nilCopy := (*OenumaNode).DeepCopy(nilNode)
	if nilCopy == nil {
		t.Log("nil 对象的 DeepCopy 正确返回 nil")
	} else {
		t.Errorf("nil 对象的 DeepCopy 应返回 nil，但返回了 %v", nilCopy)
	}
}

// TestNumaConfigDeepCopy 测试 NumaConfig 的 DeepCopy 方法
func TestNumaConfigDeepCopy(t *testing.T) {
	original := createTestNumaConfig()
	copied := original.DeepCopy()

	// 验证拷贝是否成功
	if !reflect.DeepEqual(original, *copied) {
		t.Errorf("DeepCopy 结果与原对象不匹配")
	}

	// 修改原对象，确保不影响拷贝对象
	original.NumaNum = updatedNumaCount
	original.CPUSet = "changed-cpuset"

	// 验证修改不影响拷贝对象
	if copied.NumaNum == updatedNumaCount {
		t.Errorf("修改原对象的 NumaNum 影响了拷贝对象")
	}
	if copied.CPUSet == "changed-cpuset" {
		t.Errorf("修改原对象的 CPUSet 影响了拷贝对象")
	}

	// 测试 nil 情况
	var nilNumaConfig *NumaConfig
	nilCopy := (*NumaConfig).DeepCopy(nilNumaConfig)
	if nilCopy == nil {
		t.Log("nil 对象的 DeepCopy 正确返回 nil")
	} else {
		t.Errorf("nil 对象的 DeepCopy 应返回 nil，但返回了 %v", nilCopy)
	}
}

// TestPodAffinityConfigDeepCopy 测试 PodAffinityConfig 的 DeepCopy 方法
func TestPodAffinityConfigDeepCopy(t *testing.T) {
	original := createTestPodAffinityConfig()
	copied := original.DeepCopy()

	// 验证拷贝是否成功
	if !reflect.DeepEqual(original, *copied) {
		t.Errorf("DeepCopy 结果与原对象不匹配")
	}

	// 修改原对象，确保不影响拷贝对象
	original.NumaNum = updatedNumaCount
	original.PodName = "changed-pod"
	original.Containers[0].Name = "changed-container"

	// 验证修改不影响拷贝对象
	if copied.NumaNum == updatedNumaCount {
		t.Errorf("修改原对象的 NumaNum 影响了拷贝对象")
	}
	if copied.PodName == "changed-pod" {
		t.Errorf("修改原对象的 PodName 影响了拷贝对象")
	}
	if copied.Containers[0].Name == "changed-container" {
		t.Errorf("修改原对象的 Container Name 影响了拷贝对象")
	}

	// 测试 nil 情况
	var nilPodAffinity *PodAffinityConfig
	nilCopy := (*PodAffinityConfig).DeepCopy(nilPodAffinity)
	if nilCopy == nil {
		t.Log("nil 对象的 DeepCopy 正确返回 nil")
	} else {
		t.Errorf("nil 对象的 DeepCopy 应返回 nil，但返回了 %v", nilCopy)
	}
}

// TestContainerAffinityConfigDeepCopy 测试 ContainerAffinityConfig 的 DeepCopy 方法
func TestContainerAffinityConfigDeepCopy(t *testing.T) {
	original := createTestContainerAffinityConfig()
	copied := original.DeepCopy()

	// 验证拷贝是否成功
	if !reflect.DeepEqual(original, *copied) {
		t.Errorf("DeepCopy 结果与原对象不匹配")
	}

	// 修改原对象，确保不影响拷贝对象
	original.Name = "changed-container"
	original.CPUSet = "changed-cpuset"

	// 验证修改不影响拷贝对象
	if copied.Name == "changed-container" {
		t.Errorf("修改原对象的 Name 影响了拷贝对象")
	}
	if copied.CPUSet == "changed-cpuset" {
		t.Errorf("修改原对象的 CPUSet 影响了拷贝对象")
	}

	// 测试 nil 情况
	var nilContainerAffinity *ContainerAffinityConfig
	nilCopy := (*ContainerAffinityConfig).DeepCopy(nilContainerAffinity)
	if nilCopy == nil {
		t.Log("nil 对象的 DeepCopy 正确返回 nil")
	} else {
		t.Errorf("nil 对象的 DeepCopy 应返回 nil，但返回了 %v", nilCopy)
	}
}

// 创建测试辅助函数

// 创建用于测试的 Oenuma 对象
func createTestOenuma() *Oenuma {
	return &Oenuma{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Oenuma",
			APIVersion: "resource.sop.huawei.com/v1alpha1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-oenuma",
			Namespace: "default",
			Labels: map[string]string{
				"app": "test",
			},
		},
		Spec: OenumaSpec{
			Name:         "test-oenuma",
			Replicas:     3,
			Updateenable: "enable",
			Node: []OenumaNode{
				createTestOenumaNode(),
			},
		},
		Status: OenumaStatus{},
	}
}

// 创建用于测试的 OenumaList 对象
func createTestOenumaList() *OenumaList {
	return &OenumaList{
		TypeMeta: metav1.TypeMeta{
			Kind:       "OenumaList",
			APIVersion: "resource.sop.huawei.com/v1alpha1",
		},
		ListMeta: metav1.ListMeta{},
		Items: []Oenuma{
			*createTestOenuma(),
		},
	}
}

// 创建用于测试的 OenumaSpec 对象
func createTestOenumaSpec() *OenumaSpec {
	return &OenumaSpec{
		Name:         "test-oenuma",
		Replicas:     3,
		Updateenable: "enable",
		Node: []OenumaNode{
			createTestOenumaNode(),
		},
	}
}

// 创建用于测试的 OenumaNode 对象
func createTestOenumaNode() OenumaNode {
	return OenumaNode{
		Name: "test-node",
		Numa: []NumaConfig{
			createTestNumaConfig(),
		},
		PodAffinity: []PodAffinityConfig{
			createTestPodAffinityConfig(),
		},
	}
}

// 创建用于测试的 NumaConfig 对象
func createTestNumaConfig() NumaConfig {
	return NumaConfig{
		NumaNum: 0,
		CPUSet:  "0-3",
		MemSet:  "0",
	}
}

// 创建用于测试的 PodAffinityConfig 对象
func createTestPodAffinityConfig() PodAffinityConfig {
	return PodAffinityConfig{
		NumaNum:    0,
		Namespace:  "default",
		PodName:    "test-pod",
		Containers: []ContainerAffinityConfig{createTestContainerAffinityConfig()},
	}
}

// 创建用于测试的 ContainerAffinityConfig 对象
func createTestContainerAffinityConfig() ContainerAffinityConfig {
	return ContainerAffinityConfig{
		Name:        "test-container",
		ContainerId: "container-123",
		CPUSet:      "0-1",
		MemSet:      "0",
	}
}
