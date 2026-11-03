/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package annotation

import (
	"testing"

	"sigs.k8s.io/controller-runtime/pkg/client"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/common"
)

type mockAnnotObject struct {
	client.Object
	annotations map[string]string
}

func (m *mockAnnotObject) GetAnnotations() map[string]string {
	return m.annotations
}

func (m *mockAnnotObject) SetAnnotations(annotations map[string]string) {
	m.annotations = annotations
}

func TestSetBKEClusterDefaultAnnotation(t *testing.T) {
	obj := &mockAnnotObject{}
	SetBKEClusterDefaultAnnotation(obj)

	if _, ok := obj.annotations[DeleteIgnoreTargetClusterAnnotationKey]; !ok {
		t.Error("expected DeleteIgnoreTargetClusterAnnotationKey to be set")
	}
	if _, ok := obj.annotations[DeleteIgnoreNamespaceAnnotationKey]; !ok {
		t.Error("expected DeleteIgnoreNamespaceAnnotationKey to be set")
	}
	if _, ok := obj.annotations[common.BKEAgentListenerAnnotationKey]; !ok {
		t.Error("expected BKEAgentListenerAnnotationKey to be set")
	}
	if _, ok := obj.annotations[common.BKEClusterFromAnnotationKey]; !ok {
		t.Error("expected BKEClusterFromAnnotationKey to be set")
	}
	if _, ok := obj.annotations[DeepRestoreNodeAnnotationKey]; !ok {
		t.Error("expected DeepRestoreNodeAnnotationKey to be set")
	}
	if _, ok := obj.annotations[MasterSchedulableAnnotationKey]; !ok {
		t.Error("expected MasterSchedulableAnnotationKey to be set")
	}
	if _, ok := obj.annotations[NodeBootWaitTimeOutAnnotationKey]; !ok {
		t.Error("expected NodeBootWaitTimeOutAnnotationKey to be set")
	}
}

func TestSetBKEClusterDefaultAnnotation_ExistingAnnotation(t *testing.T) {
	obj := &mockAnnotObject{
		annotations: map[string]string{
			DeleteIgnoreTargetClusterAnnotationKey: "false",
			MasterSchedulableAnnotationKey:         "true",
		},
	}
	SetBKEClusterDefaultAnnotation(obj)

	if obj.annotations[DeleteIgnoreTargetClusterAnnotationKey] != "false" {
		t.Error("existing annotation should not be overwritten")
	}
	if obj.annotations[MasterSchedulableAnnotationKey] != "true" {
		t.Error("existing annotation should not be overwritten")
	}
}

func TestHasAnnotation(t *testing.T) {
	obj := &mockAnnotObject{
		annotations: map[string]string{"test-key": "test-value"},
	}
	value, ok := HasAnnotation(obj, "test-key")
	if !ok {
		t.Error("expected true, got false")
	}
	if value != "test-value" {
		t.Errorf("expected test-value, got %s", value)
	}
}

func TestHasAnnotation_NotFound(t *testing.T) {
	obj := &mockAnnotObject{annotations: map[string]string{}}
	_, ok := HasAnnotation(obj, "test-key")
	if ok {
		t.Error("expected false, got true")
	}
}

func TestHasAnnotation_NilAnnotations(t *testing.T) {
	obj := &mockAnnotObject{annotations: nil}
	_, ok := HasAnnotation(obj, "test-key")
	if ok {
		t.Error("expected false, got true")
	}
}

func TestSetAnnotation(t *testing.T) {
	obj := &mockAnnotObject{}
	SetAnnotation(obj, "test-key", "test-value")
	if obj.annotations["test-key"] != "test-value" {
		t.Errorf("expected test-value, got %s", obj.annotations["test-key"])
	}
}

func TestSetAnnotation_NilAnnotations(t *testing.T) {
	obj := &mockAnnotObject{annotations: nil}
	SetAnnotation(obj, "test-key", "test-value")
	if obj.annotations == nil {
		t.Error("annotations should not be nil after SetAnnotation")
	}
	if obj.annotations["test-key"] != "test-value" {
		t.Errorf("expected test-value, got %s", obj.annotations["test-key"])
	}
}

func TestRemoveAnnotation(t *testing.T) {
	obj := &mockAnnotObject{
		annotations: map[string]string{"test-key": "test-value"},
	}
	RemoveAnnotation(obj, "test-key")
	if _, ok := obj.annotations["test-key"]; ok {
		t.Error("annotation should be removed")
	}
}

func TestRemoveAnnotation_NilAnnotations(t *testing.T) {
	obj := &mockAnnotObject{annotations: nil}
	RemoveAnnotation(obj, "test-key")
}

func TestBKENormalEventAnnotation(t *testing.T) {
	result := BKENormalEventAnnotation()
	if _, ok := result[common.BKEEventAnnotationKey]; !ok {
		t.Error("expected BKEEventAnnotationKey to be set")
	}
}

func TestBKEFinishEventAnnotation(t *testing.T) {
	result := BKEFinishEventAnnotation()
	if _, ok := result[common.BKEEventAnnotationKey]; !ok {
		t.Error("expected BKEEventAnnotationKey to be set")
	}
	if _, ok := result[common.BKEFinishEventAnnotationKey]; !ok {
		t.Error("expected BKEFinishEventAnnotationKey to be set")
	}
}
