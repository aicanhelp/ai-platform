/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package addonutil

import (
	"fmt"
	"sort"

	"helm.sh/helm/v3/pkg/releaseutil"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

// GetListKinds defines additional Kubernetes list kinds to be preserved.
func GetListKinds() []string {
	return []string{
		"ConfigMapList",
		"SecretList",
		"ClusterRoleList",
		"ClusterRoleBindingList",
		"RoleList",
		"RoleBindingList",
	}
}

// SortInstallUnstructuredByKind sorts unstructured Kubernetes resources by kind in installation order.
func SortInstallUnstructuredByKind(unstructs []unstructured.Unstructured) []unstructured.Unstructured {
	sort.SliceStable(unstructs, func(i, j int) bool {
		return lessByKind(unstructs[i].GetKind(), unstructs[j].GetKind(), releaseutil.InstallOrder, false)
	})
	return unstructs
}

// SortUninstallUnstructuredByKind sorts unstructured Kubernetes resources by kind in uninstallation order.
func SortUninstallUnstructuredByKind(unstructs []unstructured.Unstructured) []unstructured.Unstructured {
	sort.SliceStable(unstructs, func(i, j int) bool {
		return lessByKind(unstructs[i].GetKind(), unstructs[j].GetKind(), releaseutil.UninstallOrder, true)
	})
	return unstructs
}

// createOrderingMap creates a mapping from kind to ordering index based on the provided sort order.
func createOrderingMap(o releaseutil.KindSortOrder) map[string]int {
	ordering := make(map[string]int, len(o))
	for v, k := range o {
		ordering[k] = v
	}
	return ordering
}

// lessByKind compares two kinds based on the provided ordering and unknown kind handling.
func lessByKind(kindA, kindB string, o releaseutil.KindSortOrder, unknownFirst bool) bool {
	ordering := createOrderingMap(o)
	first, aok := ordering[kindA]
	second, bok := ordering[kindB]

	if !aok && !bok {
		if kindA != kindB {
			return kindA < kindB
		}
		return first < second
	}

	if !aok {
		return unknownFirst
	}
	if !bok {
		return !unknownFirst
	}
	return first < second
}

// IsListKind checks if the given kind is a Kubernetes list kind.
func IsListKind(kind string) bool {
	for _, listKind := range GetListKinds() {
		if listKind == kind {
			return true
		}
	}
	return false
}

// UnwrapList unwraps a list-type Kubernetes resource into individual resources.
func UnwrapList(unstruct unstructured.Unstructured) ([]unstructured.Unstructured, error) {
	if !IsListKind(unstruct.GetKind()) {
		return []unstructured.Unstructured{unstruct}, nil
	}

	items, found, err := unstructured.NestedSlice(unstruct.Object, "items")
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, fmt.Errorf("no items found in list")
	}

	var unwrapped []unstructured.Unstructured
	for _, item := range items {
		unwrapped = append(unwrapped, unstructured.Unstructured{Object: item.(map[string]interface{})})
	}
	return unwrapped, nil
}
