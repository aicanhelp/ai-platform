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

package kube

import (
	"bytes"
	"io"
	"os"
	"testing"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	yamlutil "k8s.io/apimachinery/pkg/util/yaml"
	fakedynamic "k8s.io/client-go/dynamic/fake"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/addon"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

func TestConstants(t *testing.T) {
	if DefaultFilePermission != 0644 {
		t.Errorf("Expected DefaultFilePermission to be 0644, got %o", DefaultFilePermission)
	}
	if DefaultYamlDecoderBufferSize != 4096 {
		t.Errorf("Expected DefaultYamlDecoderBufferSize to be 4096, got %d", DefaultYamlDecoderBufferSize)
	}
}

func TestUnStructYaml(t *testing.T) {
	yamlContent := `---
apiVersion: v1
kind: Pod
metadata:
  name: test-pod
  namespace: default
spec:
  containers:
  - name: test
    image: nginx:latest
`
	decoder := yamlutil.NewYAMLOrJSONDecoder(bytes.NewReader([]byte(yamlContent)), 4096)
	unstruct, gvk, err := UnStructYaml(decoder)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if unstruct == nil {
		t.Fatal("Expected unstruct to be not nil")
	}
	if gvk == nil {
		t.Fatal("Expected gvk to be not nil")
	}
	if gvk.Kind != "Pod" {
		t.Errorf("Expected Kind to be Pod, got %s", gvk.Kind)
	}
	if gvk.Group != "" {
		t.Errorf("Expected Group to be empty, got %s", gvk.Group)
	}
	if gvk.Version != "v1" {
		t.Errorf("Expected Version to be v1, got %s", gvk.Version)
	}
}

func TestUnStructYamlInvalidYAML(t *testing.T) {
	yamlContent := `---
invalid: yaml: content: [`
	decoder := yamlutil.NewYAMLOrJSONDecoder(bytes.NewReader([]byte(yamlContent)), 4096)
	unstruct, gvk, err := UnStructYaml(decoder)
	if err == nil {
		t.Fatal("Expected error for invalid YAML")
	}
	if unstruct != nil {
		t.Error("Expected unstruct to be nil")
	}
	if gvk != nil {
		t.Error("Expected gvk to be nil")
	}
}

func TestUnStructYamlEmpty(t *testing.T) {
	decoder := yamlutil.NewYAMLOrJSONDecoder(bytes.NewReader([]byte("")), 4096)
	unstruct, gvk, err := UnStructYaml(decoder)
	if err != io.EOF {
		t.Errorf("Expected EOF error, got %v", err)
	}
	if unstruct != nil {
		t.Error("Expected unstruct to be nil for empty input")
	}
	if gvk != nil {
		t.Error("Expected gvk to be nil for empty input")
	}
}

func TestGetUnStructListFromDecoder(t *testing.T) {
	yamlContent := `---
apiVersion: v1
kind: Pod
metadata:
  name: pod1
---
apiVersion: v1
kind: Service
metadata:
  name: svc1
`
	decoder := yamlutil.NewYAMLOrJSONDecoder(bytes.NewReader([]byte(yamlContent)), 4096)
	list, err := GetUnStructListFromDecoder(decoder)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if len(list) != 2 {
		t.Errorf("Expected 2 items in list, got %d", len(list))
	}
	if list[0].GetKind() != "Pod" {
		t.Errorf("Expected first item to be Pod, got %s", list[0].GetKind())
	}
	if list[1].GetKind() != "Service" {
		t.Errorf("Expected second item to be Service, got %s", list[1].GetKind())
	}
}

func TestGetUnStructListFromDecoderWithError(t *testing.T) {
	yamlContent := `---
invalid: yaml: content: [`
	decoder := yamlutil.NewYAMLOrJSONDecoder(bytes.NewReader([]byte(yamlContent)), 4096)
	list, err := GetUnStructListFromDecoder(decoder)
	if err == nil {
		t.Fatal("Expected error for invalid YAML")
	}
	if list != nil {
		t.Error("Expected list to be nil")
	}
}

func TestProcessUnstructuredList(t *testing.T) {
	c := &Client{}

	unstructList := []*unstructured.Unstructured{
		{
			Object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata": map[string]interface{}{
					"name": "pod1",
				},
			},
		},
		{
			Object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Service",
				"metadata": map[string]interface{}{
					"name": "svc1",
				},
			},
		},
	}

	result := c.processUnstructuredList(unstructList)
	if len(result) != 2 {
		t.Errorf("Expected 2 items, got %d", len(result))
	}
}

func TestSortUnstructuredListInstall(t *testing.T) {
	c := &Client{}

	unstructList := []unstructured.Unstructured{
		{
			Object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
			},
		},
	}

	task := &Task{
		Operate: addon.CreateAddon,
	}

	result := c.sortUnstructuredList(unstructList, task)
	if len(result) != 1 {
		t.Errorf("Expected 1 item, got %d", len(result))
	}
}

func TestSortUnstructuredListUpgrade(t *testing.T) {
	c := &Client{}

	unstructList := []unstructured.Unstructured{
		{
			Object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
			},
		},
	}

	task := &Task{
		Operate: addon.UpgradeAddon,
	}

	result := c.sortUnstructuredList(unstructList, task)
	if len(result) != 1 {
		t.Errorf("Expected 1 item, got %d", len(result))
	}
}

func TestSortUnstructuredListRemove(t *testing.T) {
	c := &Client{}

	unstructList := []unstructured.Unstructured{
		{
			Object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
			},
		},
	}

	task := &Task{
		Operate: addon.RemoveAddon,
	}

	result := c.sortUnstructuredList(unstructList, task)
	if len(result) != 1 {
		t.Errorf("Expected 1 item, got %d", len(result))
	}
}

func TestTrueVar(t *testing.T) {
	if !trueVar {
		t.Error("Expected trueVar to be true")
	}
}

func TestGetResourceInterface_Namespaced(t *testing.T) {
	scheme := runtime.NewScheme()
	dynamicClient := fakedynamic.NewSimpleDynamicClient(scheme)
	c := &Client{DynamicClient: dynamicClient}

	mapping := &meta.RESTMapping{
		Resource: schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"},
		Scope:    meta.RESTScopeNamespace,
	}

	unstruct := unstructured.Unstructured{}
	unstruct.SetNamespace("default")

	ri, err := c.getResourceInterface(mapping, unstruct)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if ri == nil {
		t.Fatal("Expected resource interface to be not nil")
	}
}

func TestGetResourceInterface_ClusterScoped(t *testing.T) {
	scheme := runtime.NewScheme()
	dynamicClient := fakedynamic.NewSimpleDynamicClient(scheme)
	c := &Client{DynamicClient: dynamicClient}

	mapping := &meta.RESTMapping{
		Resource: schema.GroupVersionResource{Group: "", Version: "v1", Resource: "nodes"},
		Scope:    meta.RESTScopeRoot,
	}

	unstruct := unstructured.Unstructured{}

	ri, err := c.getResourceInterface(mapping, unstruct)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if ri == nil {
		t.Fatal("Expected resource interface to be not nil")
	}
}

func TestHandleWaitAndLogging_NoBlock(t *testing.T) {
	logger := (*log.Logger)(nil)
	c := &Client{Log: logger}

	obj := &unstructured.Unstructured{}
	obj.SetKind("Pod")
	obj.SetName("test-pod")

	task := &Task{Block: false, Operate: addon.CreateAddon}

	err := c.handleWaitAndLogging(obj, *obj, task)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
}

func TestHandleWaitAndLogging_RemoveAddon(t *testing.T) {
	logger := (*log.Logger)(nil)
	c := &Client{Log: logger}

	obj := &unstructured.Unstructured{}
	obj.SetKind("Pod")
	obj.SetName("test-pod")

	task := &Task{Block: true, Operate: addon.RemoveAddon}

	err := c.handleWaitAndLogging(obj, *obj, task)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
}

func TestHandleWaitAndLogging_NilObj(t *testing.T) {
	logger := (*log.Logger)(nil)
	c := &Client{Log: logger}

	unstruct := unstructured.Unstructured{}
	unstruct.SetKind("Pod")
	unstruct.SetName("test-pod")

	task := &Task{Block: false, Operate: addon.CreateAddon}

	err := c.handleWaitAndLogging(nil, unstruct, task)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
}

func TestProcessUnstructuredList_WithList(t *testing.T) {
	c := &Client{}

	unstructList := []*unstructured.Unstructured{
		{
			Object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "List",
				"items": []interface{}{
					map[string]interface{}{
						"apiVersion": "v1",
						"kind":       "Pod",
						"metadata":   map[string]interface{}{"name": "pod1"},
					},
				},
			},
		},
	}

	result := c.processUnstructuredList(unstructList)
	if len(result) == 0 {
		t.Error("Expected at least one item")
	}
}

func TestSortUnstructuredList_UpdateAddon(t *testing.T) {
	c := &Client{}
	unstructList := []unstructured.Unstructured{
		{Object: map[string]interface{}{"apiVersion": "v1", "kind": "Pod"}},
	}
	task := &Task{Operate: addon.UpdateAddon}

	result := c.sortUnstructuredList(unstructList, task)
	if len(result) != 1 {
		t.Errorf("Expected 1 item, got %d", len(result))
	}
}

func TestGetUnStructListFromDecoder_Success(t *testing.T) {
	yamlContent := "apiVersion: v1\nkind: Pod\nmetadata:\n  name: test"
	decoder := yamlutil.NewYAMLOrJSONDecoder(bytes.NewReader([]byte(yamlContent)), 4096)
	list, err := GetUnStructListFromDecoder(decoder)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if len(list) == 0 {
		t.Error("Expected at least one item")
	}
}

func TestGetUnStructListFromDecoder_EmptyYaml(t *testing.T) {
	decoder := yamlutil.NewYAMLOrJSONDecoder(bytes.NewReader([]byte("")), 4096)
	list, err := GetUnStructListFromDecoder(decoder)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if len(list) != 0 {
		t.Errorf("Expected empty list, got %d items", len(list))
	}
}

func TestSortUnstructuredList_RemoveAddon(t *testing.T) {
	c := &Client{}
	unstructList := []unstructured.Unstructured{
		{Object: map[string]interface{}{"apiVersion": "v1", "kind": "Pod"}},
		{Object: map[string]interface{}{"apiVersion": "v1", "kind": "Service"}},
	}
	task := &Task{Operate: addon.RemoveAddon}
	result := c.sortUnstructuredList(unstructList, task)
	if len(result) != 2 {
		t.Errorf("Expected 2 items, got %d", len(result))
	}
}

func TestSortUnstructuredList_InstallAddon(t *testing.T) {
	c := &Client{}
	unstructList := []unstructured.Unstructured{
		{Object: map[string]interface{}{"apiVersion": "v1", "kind": "Pod"}},
	}
	task := &Task{Operate: addon.CreateAddon}
	result := c.sortUnstructuredList(unstructList, task)
	if len(result) != 1 {
		t.Errorf("Expected 1 item, got %d", len(result))
	}
}

func TestHandleOperation_UnknownOperation(t *testing.T) {
	logger := (*log.Logger)(nil)
	c := &Client{Log: logger}

	task := &Task{Operate: "unknown"}
	unstruct := unstructured.Unstructured{}
	gvk := schema.GroupVersionKind{Kind: "Pod"}

	obj, err := c.handleOperation(nil, unstruct, task, gvk)
	if err == nil {
		t.Error("Expected error for unknown operation")
	}
	if obj != nil {
		t.Error("Expected nil obj")
	}
}

func TestHandleRemoveOperation_NotFound(t *testing.T) {
	logger := (*log.Logger)(nil)
	scheme := runtime.NewScheme()
	dynamicClient := fakedynamic.NewSimpleDynamicClient(scheme)
	c := &Client{Log: logger, DynamicClient: dynamicClient}

	unstruct := unstructured.Unstructured{}
	unstruct.SetName("test")
	unstruct.SetNamespace("default")

	mapping := &meta.RESTMapping{
		Resource: schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"},
		Scope:    meta.RESTScopeNamespace,
	}

	dr, _ := c.getResourceInterface(mapping, unstruct)
	task := &Task{Operate: addon.RemoveAddon}

	err := c.handleRemoveOperation(dr, unstruct, task)
	if err != nil {
		t.Errorf("Expected no error for not found, got %v", err)
	}
}

func TestHandleCreateOnlyOperation_SkipWhenExists(t *testing.T) {
	scheme := runtime.NewScheme()
	existing := &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "v1",
		"kind":       "ConfigMap",
		"metadata": map[string]interface{}{
			"name":      "test-cm",
			"namespace": "default",
		},
		"data": map[string]interface{}{
			"k": "old",
		},
	}}
	dynamicClient := fakedynamic.NewSimpleDynamicClient(scheme, existing)
	c := &Client{Log: (*log.Logger)(nil), DynamicClient: dynamicClient}

	desired := unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "v1",
		"kind":       "ConfigMap",
		"metadata": map[string]interface{}{
			"name":      "test-cm",
			"namespace": "default",
		},
		"data": map[string]interface{}{
			"k": "new",
		},
	}}
	mapping := &meta.RESTMapping{
		Resource: schema.GroupVersionResource{Group: "", Version: "v1", Resource: "configmaps"},
		Scope:    meta.RESTScopeNamespace,
	}
	dr, err := c.getResourceInterface(mapping, desired)
	if err != nil {
		t.Fatalf("getResourceInterface: %v", err)
	}

	obj, err := c.handleCreateOnlyOperation(dr, desired, &Task{Operate: addon.UpgradeAddon, ApplyStrategy: ApplyStrategyCreateOnly})
	if err != nil {
		t.Fatalf("create-only: %v", err)
	}
	if obj == nil {
		t.Fatal("expected existing object")
	}
	data, _, _ := unstructured.NestedStringMap(obj.Object, "data")
	if data["k"] != "old" {
		t.Fatalf("expected existing data preserved, got %#v", data)
	}
}

func TestHandleCreateOnlyOperation_CreateWhenMissing(t *testing.T) {
	scheme := runtime.NewScheme()
	dynamicClient := fakedynamic.NewSimpleDynamicClient(scheme)
	c := &Client{Log: (*log.Logger)(nil), DynamicClient: dynamicClient}

	desired := unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "v1",
		"kind":       "ConfigMap",
		"metadata": map[string]interface{}{
			"name":      "test-cm",
			"namespace": "default",
		},
		"data": map[string]interface{}{
			"k": "new",
		},
	}}
	mapping := &meta.RESTMapping{
		Resource: schema.GroupVersionResource{Group: "", Version: "v1", Resource: "configmaps"},
		Scope:    meta.RESTScopeNamespace,
	}
	dr, err := c.getResourceInterface(mapping, desired)
	if err != nil {
		t.Fatalf("getResourceInterface: %v", err)
	}

	obj, err := c.handleCreateOnlyOperation(dr, desired, &Task{Operate: addon.UpgradeAddon, ApplyStrategy: ApplyStrategyCreateOnly})
	if err != nil {
		t.Fatalf("create-only: %v", err)
	}
	if obj == nil || obj.GetName() != "test-cm" {
		t.Fatalf("expected created object, got %#v", obj)
	}
}

func TestHandleOperation_CreateOnlyStrategy(t *testing.T) {
	scheme := runtime.NewScheme()
	existing := &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "v1",
		"kind":       "ConfigMap",
		"metadata": map[string]interface{}{
			"name":      "test-cm",
			"namespace": "default",
		},
	}}
	dynamicClient := fakedynamic.NewSimpleDynamicClient(scheme, existing)
	c := &Client{Log: (*log.Logger)(nil), DynamicClient: dynamicClient}

	desired := unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "v1",
		"kind":       "ConfigMap",
		"metadata": map[string]interface{}{
			"name":      "test-cm",
			"namespace": "default",
		},
	}}
	mapping := &meta.RESTMapping{
		Resource: schema.GroupVersionResource{Group: "", Version: "v1", Resource: "configmaps"},
		Scope:    meta.RESTScopeNamespace,
	}
	dr, err := c.getResourceInterface(mapping, desired)
	if err != nil {
		t.Fatalf("getResourceInterface: %v", err)
	}

	obj, err := c.handleOperation(dr, desired, &Task{
		Operate:       addon.UpgradeAddon,
		ApplyStrategy: ApplyStrategyCreateOnly,
	}, schema.GroupVersionKind{Group: "", Version: "v1", Kind: "ConfigMap"})
	if err != nil {
		t.Fatalf("handleOperation: %v", err)
	}
	if obj == nil || obj.GetName() != "test-cm" {
		t.Fatalf("expected existing object returned, got %#v", obj)
	}
}

func TestRenderYamlToDecoder_FileNotFound(t *testing.T) {
	task := &Task{
		Name:     "test",
		FilePath: "/nonexistent/file.yaml",
		Param:    map[string]interface{}{},
	}

	_, err := RenderYamlToDecoder(task)
	if err == nil {
		t.Error("Expected error for non-existent file")
	}
}

func TestHandleUpdateOperation_CRD(t *testing.T) {
	logger := (*log.Logger)(nil)
	c := &Client{Log: logger}

	unstruct := unstructured.Unstructured{}
	unstruct.SetName("test-crd")
	gvk := schema.GroupVersionKind{Kind: "CustomResourceDefinition"}

	task := &Task{Operate: addon.UpdateAddon}

	obj, err := c.handleUpdateOperation(nil, unstruct, task, gvk)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if obj != nil {
		t.Error("Expected obj to be nil for CRD")
	}
}

func TestRenderManifestAndTaskYAMLBranches(t *testing.T) {
	empty, err := RenderManifest("empty", nil, nil)
	if err != nil || empty != nil {
		t.Fatalf("RenderManifest empty = %q, %v; want nil, nil", string(empty), err)
	}

	rendered, err := RenderManifest("cm", []byte("kind: ConfigMap\nmetadata:\n  name: {{ .name }}\n"), map[string]interface{}{"name": "rendered-cm"})
	if err != nil {
		t.Fatalf("RenderManifest() error = %v", err)
	}
	if !bytes.Contains(rendered, []byte("name: rendered-cm")) {
		t.Fatalf("RenderManifest() = %q", string(rendered))
	}

	if _, err := renderTaskYAML(nil); err == nil {
		t.Fatal("renderTaskYAML(nil) expected error")
	}
	if _, err := renderTaskYAML(&Task{}); err == nil {
		t.Fatal("renderTaskYAML(empty task) expected error")
	}

	file, err := os.CreateTemp(t.TempDir(), "noexecute-*.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := file.WriteString("kind: ConfigMap\nmetadata:\n  name: {{ .name }}\n"); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	raw, err := renderTaskYAML(&Task{Name: "noexecute", FilePath: file.Name(), Param: map[string]interface{}{"name": "ignored"}})
	if err != nil {
		t.Fatalf("renderTaskYAML(noexecute) error = %v", err)
	}
	if !bytes.Contains(raw, []byte("{{ .name }}")) {
		t.Fatalf("noexecute manifest should not be rendered: %q", string(raw))
	}
}

func TestClientGetRestMapperNil(t *testing.T) {
	if _, err := (&Client{}).getRestMapper(); err == nil {
		t.Fatal("getRestMapper() expected error when mapper is nil")
	}
}
