/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package k8sutil

import (
	"errors"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	dynamicfake "k8s.io/client-go/dynamic/fake"
	k8sfake "k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"

	"monitoring-dashboard-service/pkg/config/global"
)

const validKubeConfig = `apiVersion: v1
kind: Config
clusters:
- name: test-cluster
  cluster:
    server: https://192.168.0.1:6443
users:
- name: test-user
  user:
    token: test-token
contexts:
- name: test-context
  context:
    cluster: test-cluster
    user: test-user
current-context: test-context
`

// TestLoadKubeConfigFromBytesValid 合法kubeconfig字节流
func TestLoadKubeConfigFromBytesValid(t *testing.T) {
	config, err := LoadKubeConfigFromBytes([]byte(validKubeConfig))
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if config.Host != "https://192.168.0.1:6443" {
		t.Errorf("unexpected host: %s", config.Host)
	}
}

// TestLoadKubeConfigFromBytesInvalid 非法kubeconfig字节流
func TestLoadKubeConfigFromBytesInvalid(t *testing.T) {
	if _, err := LoadKubeConfigFromBytes([]byte("this is not yaml config")); err == nil {
		t.Error("expected error for invalid kubeconfig bytes, got nil")
	}
}

// TestLoadKubeConfigFromBytesMissingCurrentContext 缺失current-context
func TestLoadKubeConfigFromBytesMissingCurrentContext(t *testing.T) {
	noContext := strings.Replace(validKubeConfig, "current-context: test-context", "", 1)
	if _, err := LoadKubeConfigFromBytes([]byte(noContext)); err == nil {
		t.Error("expected error for missing current-context, got nil")
	}
}

func newTestSecret() *v1.Secret {
	return &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-secret",
			Namespace: global.MonitorServiceDefaultNamespace,
		},
		Data: map[string][]byte{"key": []byte("value")},
	}
}

// TestGetSecretAndUpdateAndDelete Secret的查改删
func TestGetSecretAndUpdateAndDelete(t *testing.T) {
	clientset := k8sfake.NewSimpleClientset(newTestSecret())

	secret, err := GetSecret(clientset, "test-secret", global.MonitorServiceDefaultNamespace)
	if err != nil {
		t.Fatalf("GetSecret failed: %v", err)
	}
	if secret.Name != "test-secret" {
		t.Errorf("unexpected secret name: %s", secret.Name)
	}

	if _, err := GetSecret(clientset, "not-exist", global.MonitorServiceDefaultNamespace); err == nil {
		t.Error("expected error for missing secret, got nil")
	}

	secret.Data["key"] = []byte("new-value")
	updated, err := UpdateSecret(clientset, secret)
	if err != nil {
		t.Fatalf("UpdateSecret failed: %v", err)
	}
	if string(updated.Data["key"]) != "new-value" {
		t.Errorf("secret not updated: %v", updated.Data)
	}

	if err := DeleteSecret(clientset, "test-secret", global.MonitorServiceDefaultNamespace); err != nil {
		t.Fatalf("DeleteSecret failed: %v", err)
	}
	if err := DeleteSecret(clientset, "test-secret", global.MonitorServiceDefaultNamespace); err == nil {
		t.Error("expected error for deleting missing secret, got nil")
	}
}

// TestListSecret 列出Secret
func TestListSecret(t *testing.T) {
	clientset := k8sfake.NewSimpleClientset(newTestSecret())
	secretList, err := ListSecret(clientset, global.MonitorServiceDefaultNamespace)
	if err != nil {
		t.Fatalf("ListSecret failed: %v", err)
	}
	if len(secretList.Items) != 1 {
		t.Errorf("expected 1 secret, got %d", len(secretList.Items))
	}
}

// TestCreateSecret 创建Secret成功与重名失败
func TestCreateSecret(t *testing.T) {
	clientset := k8sfake.NewSimpleClientset()
	created, err := CreateSecret(clientset, newTestSecret())
	if err != nil {
		t.Fatalf("CreateSecret failed: %v", err)
	}
	if created.Name != "test-secret" {
		t.Errorf("unexpected created name: %s", created.Name)
	}
	if _, err := CreateSecret(clientset, newTestSecret()); err == nil {
		t.Error("expected error for duplicate secret, got nil")
	}
}

func newTestConfigMap() *v1.ConfigMap {
	return &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-configmap",
			Namespace: global.MonitorServiceDefaultNamespace,
		},
		Data: map[string]string{"key": "value"},
	}
}

// TestConfigMapCRUD ConfigMap增删改查
func TestConfigMapCRUD(t *testing.T) {
	clientset := k8sfake.NewSimpleClientset()

	created, err := CreateConfigMap(clientset, newTestConfigMap())
	if err != nil {
		t.Fatalf("CreateConfigMap failed: %v", err)
	}
	if created.Name != "test-configmap" {
		t.Errorf("unexpected created name: %s", created.Name)
	}

	got, err := GetConfigMap(clientset, "test-configmap", global.MonitorServiceDefaultNamespace)
	if err != nil {
		t.Fatalf("GetConfigMap failed: %v", err)
	}
	if got.Data["key"] != "value" {
		t.Errorf("unexpected configmap data: %v", got.Data)
	}

	if _, err := GetConfigMap(clientset, "not-exist", global.MonitorServiceDefaultNamespace); err == nil {
		t.Error("expected error for missing configmap, got nil")
	}

	list, err := ListConfigmap(clientset, global.MonitorServiceDefaultNamespace)
	if err != nil {
		t.Fatalf("ListConfigmap failed: %v", err)
	}
	if len(list.Items) != 1 {
		t.Errorf("expected 1 configmap, got %d", len(list.Items))
	}

	got.Data["key"] = "new-value"
	updated, err := UpdateConfigMap(clientset, got)
	if err != nil {
		t.Fatalf("UpdateConfigMap failed: %v", err)
	}
	if updated.Data["key"] != "new-value" {
		t.Errorf("configmap not updated: %v", updated.Data)
	}

	if err := DeleteConfigMap(clientset, "test-configmap", global.MonitorServiceDefaultNamespace); err != nil {
		t.Fatalf("DeleteConfigMap failed: %v", err)
	}
	if err := DeleteConfigMap(clientset, "test-configmap", global.MonitorServiceDefaultNamespace); err == nil {
		t.Error("expected error for deleting missing configmap, got nil")
	}
}

func newFakeDynamicClient(objects ...runtime.Object) *dynamicfake.FakeDynamicClient {
	scheme := runtime.NewScheme()
	return dynamicfake.NewSimpleDynamicClientWithCustomListKinds(
		scheme,
		map[schema.GroupVersionResource]string{
			{
				Group:    global.CRDRepoGroup,
				Version:  global.CRDRepoVersion,
				Resource: global.DashboardRepoResource,
			}: global.DashboardRepoListKind,
		},
		objects...,
	)
}

func newDashboardUnstructured(name string) *unstructured.Unstructured {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       global.DashboardRepoKind,
			"apiVersion": global.CRDRepoGroup + "/" + global.CRDRepoVersion,
			"metadata": map[string]interface{}{
				"namespace": global.MonitorServiceDefaultNamespace,
				"name":      name,
			},
			"spec": map[string]interface{}{
				"dashboardID": name,
			},
		},
	}
}

// TestCrExists CR存在性检查
func TestCrExists(t *testing.T) {
	gvr := schema.GroupVersionResource{
		Group:    global.CRDRepoGroup,
		Version:  global.CRDRepoVersion,
		Resource: global.DashboardRepoResource,
	}
	client := newFakeDynamicClient(newDashboardUnstructured("dashboard-exist"))

	exists, err := CrExists(client, "dashboard-exist", global.MonitorServiceDefaultNamespace, gvr)
	if err != nil {
		t.Fatalf("CrExists failed: %v", err)
	}
	if !exists {
		t.Error("expected CR to exist")
	}

	exists, err = CrExists(client, "dashboard-missing", global.MonitorServiceDefaultNamespace, gvr)
	if err != nil {
		t.Fatalf("CrExists failed for missing CR: %v", err)
	}
	if exists {
		t.Error("expected CR to not exist")
	}

	client.PrependReactor("get", gvr.Resource, func(action k8stesting.Action) (bool, runtime.Object, error) {
		return true, nil, errors.New("get cr error")
	})
	if _, err := CrExists(client, "dashboard-exist", global.MonitorServiceDefaultNamespace, gvr); err == nil {
		t.Error("expected error from reactor, got nil")
	}
}

// TestStructToUnstructured 结构体转unstructured
func TestStructToUnstructured(t *testing.T) {
	type normalStruct struct {
		Name string `json:"name"`
	}
	result, err := StructToUnstructured(&normalStruct{Name: "test"})
	if err != nil {
		t.Fatalf("StructToUnstructured failed: %v", err)
	}
	name, _, _ := unstructured.NestedString(result.Object, "name")
	if name != "test" {
		t.Errorf("unexpected name: %s", name)
	}

	type badStruct struct {
		Ch chan int `json:"ch"`
	}
	if _, err := StructToUnstructured(&badStruct{}); err == nil {
		t.Error("expected error for unsupported chan field, got nil")
	}
}

// TestResourceMetadataRegexValid metadata.name正则校验
func TestResourceMetadataRegexValid(t *testing.T) {
	valid, err := ResourceMetadataRegexValid("abc-123.def")
	if err != nil || !valid {
		t.Errorf("expected valid for lowercase name, got valid=%v err=%v", valid, err)
	}

	valid, err = ResourceMetadataRegexValid("ABC")
	if valid || err == nil {
		t.Errorf("expected invalid for uppercase name, got valid=%v err=%v", valid, err)
	}
	var regexErr *RegexError
	if !errors.As(err, &regexErr) {
		t.Errorf("expected RegexError, got %T", err)
	}
	if err.Error() != "RegexError: Input does not match metadata.name regex pattern" {
		t.Errorf("unexpected error message: %s", err.Error())
	}
}
