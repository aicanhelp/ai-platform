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

package addonutil

import (
	"testing"

	"helm.sh/helm/v3/pkg/releaseutil"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

func TestGetListKinds(t *testing.T) {
	kinds := GetListKinds()
	if len(kinds) != 6 {
		t.Errorf("expected 6 kinds, got %d", len(kinds))
	}
	if kinds[0] != "ConfigMapList" {
		t.Errorf("expected ConfigMapList, got %s", kinds[0])
	}
}

func TestIsListKind(t *testing.T) {
	tests := []struct {
		kind   string
		expect bool
	}{
		{"ConfigMapList", true},
		{"SecretList", true},
		{"ClusterRoleList", true},
		{"ClusterRoleBindingList", true},
		{"RoleList", true},
		{"RoleBindingList", true},
		{"Pod", false},
		{"Service", false},
		{"Deployment", false},
	}

	for _, tt := range tests {
		result := IsListKind(tt.kind)
		if result != tt.expect {
			t.Errorf("IsListKind(%s) = %v, expected %v", tt.kind, result, tt.expect)
		}
	}
}

func TestSortInstallUnstructuredByKind(t *testing.T) {
	resources := []unstructured.Unstructured{
		{Object: map[string]interface{}{"kind": "Service"}},
		{Object: map[string]interface{}{"kind": "ConfigMap"}},
		{Object: map[string]interface{}{"kind": "Deployment"}},
		{Object: map[string]interface{}{"kind": "Namespace"}},
	}

	sorted := SortInstallUnstructuredByKind(resources)
	if len(sorted) != 4 {
		t.Errorf("expected 4, got %d", len(sorted))
	}
}

func TestSortUninstallUnstructuredByKind(t *testing.T) {
	resources := []unstructured.Unstructured{
		{Object: map[string]interface{}{"kind": "Service"}},
		{Object: map[string]interface{}{"kind": "ConfigMap"}},
		{Object: map[string]interface{}{"kind": "Deployment"}},
		{Object: map[string]interface{}{"kind": "Namespace"}},
	}

	sorted := SortUninstallUnstructuredByKind(resources)
	if len(sorted) != 4 {
		t.Errorf("expected 4, got %d", len(sorted))
	}
}

func TestUnwrapList_NotList(t *testing.T) {
	unstruct := unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind": "Pod",
			"metadata": map[string]interface{}{
				"name": "test-pod",
			},
		},
	}

	result, err := UnwrapList(unstruct)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(result) != 1 {
		t.Errorf("expected 1, got %d", len(result))
	}
	if result[0].GetKind() != "Pod" {
		t.Errorf("expected Pod, got %s", result[0].GetKind())
	}
}

func TestUnwrapList_ListKind(t *testing.T) {
	unstruct := unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind": "ConfigMapList",
			"items": []interface{}{
				map[string]interface{}{
					"kind": "ConfigMap",
					"metadata": map[string]interface{}{
						"name": "config1",
					},
				},
				map[string]interface{}{
					"kind": "ConfigMap",
					"metadata": map[string]interface{}{
						"name": "config2",
					},
				},
			},
		},
	}

	result, err := UnwrapList(unstruct)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(result) != 2 {
		t.Errorf("expected 2, got %d", len(result))
	}
}

func TestUnwrapList_ListKind_NoItems(t *testing.T) {
	unstruct := unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind": "ConfigMapList",
		},
	}

	_, err := UnwrapList(unstruct)
	if err == nil {
		t.Error("expected error, got nil")
	}
}

func TestCreateOrderingMap(t *testing.T) {
	ordering := releaseutil.KindSortOrder{
		"Namespace",
		"ResourceQuota",
		"ServiceAccount",
	}
	result := createOrderingMap(ordering)
	if len(result) != 3 {
		t.Errorf("expected 3, got %d", len(result))
	}
}

func TestLessByKind_BothUnknown(t *testing.T) {
	result := lessByKind("CustomResource", "AnotherCustom", releaseutil.InstallOrder, false)
	if result {
		t.Error("expected false, got true")
	}
}

func TestLessByKind_OneUnknown(t *testing.T) {
	result := lessByKind("CustomResource", "Namespace", releaseutil.InstallOrder, true)
	if !result {
		t.Error("expected true, got false")
	}
}

func TestLessByKind_BothKnown(t *testing.T) {
	result := lessByKind("Namespace", "Service", releaseutil.InstallOrder, false)
	if !result {
		t.Error("expected true, got false")
	}
}
