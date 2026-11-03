/*
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
* Author: YuXiang Guo
* Date: 2024-11-25
 */

package helper

import (
	"encoding/json"
	"fmt"
	"reflect"
	"testing"

	admissionv1 "k8s.io/api/admission/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"

	"openfuyao.com/colocation-management/pkg/common"
)

func TestToAdmissionErrorResponse(t *testing.T) {
	tests := []struct {
		name     string
		uid      types.UID
		err      error
		expected *admissionv1.AdmissionResponse
	}{
		{
			name: "basic error response",
			uid:  "111",
			err:  fmt.Errorf("test error"),
			expected: &admissionv1.AdmissionResponse{
				UID: "111",
				Result: &metav1.Status{
					Message: "test error",
					Status:  metav1.StatusFailure,
				},
			},
		},
		{
			name: "empty uid",
			uid:  "",
			err:  fmt.Errorf("empty uid error"),
			expected: &admissionv1.AdmissionResponse{
				UID: "",
				Result: &metav1.Status{
					Message: "empty uid error",
					Status:  metav1.StatusFailure,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			response := ToAdmissionErrorResponse(tt.uid, tt.err)
			if response.UID != tt.expected.UID {
				t.Errorf("Expected UID %s, got %s", tt.expected.UID, response.UID)
			}
			if response.Result.Message != tt.expected.Result.Message {
				t.Errorf("Expected message %s, got %s", tt.expected.Result.Message, response.Result.Message)
			}
			if response.Result.Status != tt.expected.Result.Status {
				t.Errorf("Expected status %s, got %s", tt.expected.Result.Status, response.Result.Status)
			}
		})
	}
}

func TestGetK8sObjType(t *testing.T) {
	pod := CreateTestPod()

	objType := extractK8sObjectType(pod)

	if objType.Name() != "Pod" {
		t.Errorf("Expected type name Pod, got %s", objType.Name())
	}

	if objType.Kind() != reflect.Struct {
		t.Errorf("Expected struct kind, got %v", objType.Kind())
	}
}

func TestNewK8sObj(t *testing.T) {
	pod := CreateTestPod()
	objType := extractK8sObjectType(pod)

	newObj := NewK8sObj(objType)

	if newObj == nil {
		t.Error("Expected non-nil object")
	}

	// 验证返回的对象是 metav1.Object 类型
	if _, ok := newObj.(metav1.Object); !ok {
		t.Error("Expected object to implement metav1.Object")
	}
}

func TestStaticObjectCreator(t *testing.T) {
	testPod := CreateTestPod()
	rawPod, err := ToRawPod(testPod)
	if err != nil {
		t.Fatalf("Failed to marshal pod: %v", err)
	}

	creator := NewStaticObjectCreator(testPod)

	obj, err := creator.NewObject(rawPod)
	if err != nil {
		t.Fatalf("Failed to create object: %v", err)
	}

	if obj == nil {
		t.Error("Expected non-nil object")
	}

	if pod, ok := obj.(*v1.Pod); !ok {
		t.Errorf("Expected *corev1.Pod, got %T", obj)
	} else if pod.Name != testPod.Name {
		t.Errorf("Expected pod name %s, got %s", testPod.Name, pod.Name)
	}

	_, err = creator.NewObject([]byte("invalid json"))
	if err == nil {
		t.Error("Expected error for invalid JSON")
	}
}

func TestDynamicObjectCreator(t *testing.T) {
	testPod := CreateTestPod()
	rawPod, err := ToRawPod(testPod)
	if err != nil {
		t.Fatalf("Failed to marshal pod: %v", err)
	}

	creator := NewDynamicObjectCreator()

	obj, err := creator.NewObject(rawPod)
	if err != nil {
		t.Fatalf("Failed to create object: %v", err)
	}

	if obj == nil {
		t.Error("Expected non-nil object")
	}

	if pod, ok := obj.(*v1.Pod); !ok {
		t.Errorf("Expected *corev1.Pod, got %T", obj)
	} else if pod.Name != testPod.Name {
		t.Errorf("Expected pod name %s, got %s", testPod.Name, pod.Name)
	}

	invalidJSON := []byte(`{"apiVersion": "invalid.group/v1", "kind": "InvalidKind", "metadata": {"name": "test"}}`)
	obj, err = creator.NewObject(invalidJSON)
	if err != nil {
		t.Fatalf("Failed to create unstructured object: %v", err)
	}

	if unstructuredObj, ok := obj.(*unstructured.Unstructured); !ok {
		t.Errorf("Expected *unstructured.Unstructured, got %T", obj)
	} else if unstructuredObj.GetName() != "test" {
		t.Errorf("Expected unstructured object name 'test', got '%s'", unstructuredObj.GetName())
	}

	_, err = creator.NewObject([]byte("completely invalid json"))
	if err == nil {
		t.Error("Expected error for completely invalid JSON")
	}
}

func TestNewStaticObjectCreator(t *testing.T) {
	testPod := CreateTestPod()

	creator := NewStaticObjectCreator(testPod)

	if creator == nil {
		t.Error("Expected non-nil creator")
	}

	if staticCreator, ok := creator.(staticObjectCreator); !ok {
		t.Errorf("Expected staticObjectCreator, got %T", creator)
	} else if staticCreator.objType.Name() != "Pod" {
		t.Errorf("Expected objType Pod, got %s", staticCreator.objType.Name())
	}
}

func TestNewDynamicObjectCreator(t *testing.T) {
	creator := NewDynamicObjectCreator()

	if creator == nil {
		t.Error("Expected non-nil creator")
	}

	if _, ok := creator.(dynamicObjectCreator); !ok {
		t.Errorf("Expected dynamicObjectCreator, got %T", creator)
	}
}

func TestEdgeCases(t *testing.T) {
	staticCreator := NewStaticObjectCreator(CreateTestPod())
	_, err := staticCreator.NewObject(nil)
	if err == nil {
		t.Error("Expected error for nil input")
	}

	_, err = staticCreator.NewObject([]byte{})
	if err == nil {
		t.Error("Expected error for empty input")
	}

	_, err = staticCreator.NewObject([]byte("   "))
	if err == nil {
		t.Error("Expected error for whitespace-only input")
	}
}

func ToRawPod(pod *v1.Pod) ([]byte, error) {
	RawPodJson, err := json.Marshal(pod)
	return RawPodJson, err
}

func CreateTestPod() *v1.Pod {
	container := &v1.Container{
		Name: "container1",
		Resources: v1.ResourceRequirements{
			Requests: map[v1.ResourceName]resource.Quantity{
				common.ExtenderResourceMemory: resource.MustParse("2Gi"),
				common.ExtenderResourceCPU:    resource.MustParse("2"),
			},
		},
	}

	initContainer := &v1.Container{
		Name: "containerInit",
		Resources: v1.ResourceRequirements{
			Requests: map[v1.ResourceName]resource.Quantity{
				common.ExtenderResourceMemory: resource.MustParse("2Gi"),
				common.ExtenderResourceCPU:    resource.MustParse("2"),
			},
			Limits: map[v1.ResourceName]resource.Quantity{
				common.ExtenderResourceMemory: resource.MustParse("2Gi"),
				common.ExtenderResourceCPU:    resource.MustParse("2"),
			},
		},
	}

	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "pod",
			Annotations: map[string]string{
				common.PriorityAnnotationKey: "true",
				common.QosClassAnnotationKey: "BE",
			},
		},
		Spec: v1.PodSpec{
			Containers:     []v1.Container{*container},
			InitContainers: []v1.Container{*initContainer},
		},
		Status: v1.PodStatus{QOSClass: v1.PodQOSBestEffort},
	}

	return pod
}
