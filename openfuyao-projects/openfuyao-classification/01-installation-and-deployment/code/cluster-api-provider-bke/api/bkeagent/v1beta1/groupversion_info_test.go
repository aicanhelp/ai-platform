/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package v1beta1

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestGroupVersion(t *testing.T) {
	expectedGroup := "bkeagent.bocloud.com"
	expectedVersion := "v1beta1"

	if GroupVersion.Group != expectedGroup {
		t.Errorf("expected group %s, got %s", expectedGroup, GroupVersion.Group)
	}
	if GroupVersion.Version != expectedVersion {
		t.Errorf("expected version %s, got %s", expectedVersion, GroupVersion.Version)
	}
}

func TestSchemeBuilderGroupVersion(t *testing.T) {
	if SchemeBuilder == nil {
		t.Skip("SchemeBuilder is nil")
	}

	expectedGV := schema.GroupVersion{Group: "bkeagent.bocloud.com", Version: "v1beta1"}
	builderGV := SchemeBuilder.GroupVersion

	if !reflect.DeepEqual(builderGV, expectedGV) {
		t.Errorf("expected GroupVersion %v, got %v", expectedGV, builderGV)
	}
}

func TestGroupVersionKind(t *testing.T) {
	gvk := GroupVersion.WithKind("Command")

	if gvk.Group != "bkeagent.bocloud.com" {
		t.Errorf("expected group bkeagent.bocloud.com, got %s", gvk.Group)
	}
	if gvk.Version != "v1beta1" {
		t.Errorf("expected version v1beta1, got %s", gvk.Version)
	}
	if gvk.Kind != "Command" {
		t.Errorf("expected kind Command, got %s", gvk.Kind)
	}
}

func TestGroupVersionResource(t *testing.T) {
	resource := GroupVersion.WithResource("commands")

	if resource.Group != "bkeagent.bocloud.com" {
		t.Errorf("expected group bkeagent.bocloud.com, got %s", resource.Group)
	}
	if resource.Version != "v1beta1" {
		t.Errorf("expected version v1beta1, got %s", resource.Version)
	}
	if resource.Resource != "commands" {
		t.Errorf("expected resource commands, got %s", resource.Resource)
	}
}
