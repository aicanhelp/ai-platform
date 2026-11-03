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

// Package k8sutil contains public function for the application-management-service project
package k8sutil

import (
	"context"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	dynamicfake "k8s.io/client-go/dynamic/fake"
	"k8s.io/client-go/kubernetes/fake"
)

// TestGetSecret tests the GetSecret function
func TestGetSecret(t *testing.T) {
	clientset := fake.NewSimpleClientset(&v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-secret",
			Namespace: "default",
		},
	})

	secret, err := GetSecret(clientset, "test-secret", "default")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if secret.Name != "test-secret" {
		t.Errorf("Expected secret name %s, got %s", "test-secret", secret.Name)
	}
}

func TestGetSecret_NotFound(t *testing.T) {
	clientset := fake.NewSimpleClientset()

	_, err := GetSecret(clientset, "missing-secret", "default")
	if err == nil {
		t.Fatalf("Expected error, got nil")
	}
	if !errors.IsNotFound(err) {
		t.Fatalf("Expected NotFound error, got %v", err)
	}
}

func TestListSecret(t *testing.T) {
	clientset := fake.NewSimpleClientset(
		&v1.Secret{ObjectMeta: metav1.ObjectMeta{Name: "s1", Namespace: "default"}},
		&v1.Secret{ObjectMeta: metav1.ObjectMeta{Name: "s2", Namespace: "default"}},
	)

	list, err := ListSecret(clientset, "default")
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if list == nil {
		t.Fatalf("Expected non-nil list")
	}
	if got := len(list.Items); got != 2 {
		t.Fatalf("Expected 2 secrets, got %d", got)
	}
}

// TestCreateSecret tests the CreateSecret function
func TestCreateSecret(t *testing.T) {
	newSecret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "new-secret",
			Namespace: "default",
		},
	}

	clientset := fake.NewSimpleClientset()
	createdSecret, err := CreateSecret(clientset, newSecret)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if createdSecret.Name != "new-secret" {
		t.Errorf("Expected secret name %s, got %s", "new-secret", createdSecret.Name)
	}
}

func TestCreateSecret_AlreadyExists(t *testing.T) {
	existing := &v1.Secret{ObjectMeta: metav1.ObjectMeta{Name: "dup", Namespace: "default"}}
	clientset := fake.NewSimpleClientset(existing)

	_, err := CreateSecret(clientset, existing)
	if err == nil {
		t.Fatalf("Expected error, got nil")
	}
	if !errors.IsAlreadyExists(err) {
		t.Fatalf("Expected AlreadyExists error, got %v", err)
	}
}

// TestUpdateSecret tests the UpdateSecret function
func TestUpdateSecret(t *testing.T) {
	initialSecret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "update-secret",
			Namespace: "default",
		},
		Data: map[string][]byte{
			"key": []byte("old-value"),
		},
	}
	clientset := fake.NewSimpleClientset(initialSecret)

	// Update the secret
	initialSecret.Data["key"] = []byte("new-value")
	updatedSecret, err := UpdateSecret(clientset, initialSecret)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if string(updatedSecret.Data["key"]) != "new-value" {
		t.Errorf("Expected updated secret value %s, got %s", "new-value", string(updatedSecret.Data["key"]))
	}
}

func TestUpdateSecret_NotFound(t *testing.T) {
	clientset := fake.NewSimpleClientset()

	missing := &v1.Secret{ObjectMeta: metav1.ObjectMeta{Name: "missing", Namespace: "default"}}
	_, err := UpdateSecret(clientset, missing)
	if err == nil {
		t.Fatalf("Expected error, got nil")
	}
}

// TestDeleteSecret tests the DeleteSecret function
func TestDeleteSecret(t *testing.T) {
	secretName := "delete-secret"
	namespace := "default"
	clientset := fake.NewSimpleClientset(&v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      secretName,
			Namespace: namespace,
		},
	})

	err := DeleteSecret(clientset, secretName, namespace)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}

	// Attempt to get the deleted secret to confirm deletion
	_, err = clientset.CoreV1().Secrets(namespace).Get(context.Background(), secretName, metav1.GetOptions{})
	if !errors.IsNotFound(err) {
		t.Errorf("Expected NotFound error, got %v", err)
	}
}

// TestGetConfigMap tests the GetConfigMap function
func TestGetConfigMap(t *testing.T) {
	clientset := fake.NewSimpleClientset(&v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-configmap",
			Namespace: "default",
		},
	})

	configMap, err := GetConfigMap(clientset, "test-configmap", "default")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if configMap.Name != "test-configmap" {
		t.Errorf("Expected configMap name %s, got %s", "test-configmap", configMap.Name)
	}
}

func TestGetConfigMap_NotFound(t *testing.T) {
	clientset := fake.NewSimpleClientset()

	_, err := GetConfigMap(clientset, "missing-configmap", "default")
	if err == nil {
		t.Fatalf("Expected error, got nil")
	}
	if !errors.IsNotFound(err) {
		t.Fatalf("Expected NotFound error, got %v", err)
	}
}

func TestListConfigmap(t *testing.T) {
	clientset := fake.NewSimpleClientset(
		&v1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "default"}},
		&v1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "c2", Namespace: "default"}},
	)

	list, err := ListConfigmap(clientset, "default")
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if list == nil {
		t.Fatalf("Expected non-nil list")
	}
	if got := len(list.Items); got != 2 {
		t.Fatalf("Expected 2 configmaps, got %d", got)
	}
}

// TestCreateConfigMap tests the CreateConfigMap function
func TestCreateConfigMap(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	configMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "create-configmap",
			Namespace: "default",
		},
	}

	createdConfigMap, err := CreateConfigMap(clientset, configMap)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if createdConfigMap.Name != "create-configmap" {
		t.Errorf("Expected configMap name %s, got %s", "create-configmap", createdConfigMap.Name)
	}
}

func TestCreateConfigMap_AlreadyExists(t *testing.T) {
	existing := &v1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "dup", Namespace: "default"}}
	clientset := fake.NewSimpleClientset(existing)

	_, err := CreateConfigMap(clientset, existing)
	if err == nil {
		t.Fatalf("Expected error, got nil")
	}
	if !errors.IsAlreadyExists(err) {
		t.Fatalf("Expected AlreadyExists error, got %v", err)
	}
}

// TestUpdateConfigMap tests the UpdateConfigMap function
func TestUpdateConfigMap(t *testing.T) {
	configMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "update-configmap",
			Namespace: "default",
		},
	}
	clientset := fake.NewSimpleClientset(configMap)

	configMap.Data = map[string]string{"key": "new-value"}
	updatedConfigMap, err := UpdateConfigMap(clientset, configMap)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if updatedConfigMap.Data["key"] != "new-value" {
		t.Errorf("Expected updated configMap value %s, got %s", "new-value", updatedConfigMap.Data["key"])
	}
}

func TestUpdateConfigMap_NotFound(t *testing.T) {
	clientset := fake.NewSimpleClientset()

	missing := &v1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "missing", Namespace: "default"}}
	_, err := UpdateConfigMap(clientset, missing)
	if err == nil {
		t.Fatalf("Expected error, got nil")
	}
}

// TestDeleteConfigMap tests the DeleteConfigMap function
func TestDeleteConfigMap(t *testing.T) {
	clientset := fake.NewSimpleClientset(&v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "delete-configmap",
			Namespace: "default",
		},
	})

	err := DeleteConfigMap(clientset, "delete-configmap", "default")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}

	_, err = clientset.CoreV1().ConfigMaps("default").Get(context.Background(), "delete-configmap",
		metav1.GetOptions{})
	if !errors.IsNotFound(err) {
		t.Errorf("Expected NotFound error, got %v", err)
	}
}

// TestStructToUnstructured tests the StructToUnstructured function
func TestStructToUnstructured(t *testing.T) {
	// Define a simple struct that should be easy to convert
	type PodSpec struct {
		Containers []string `json:"containers"`
	}
	podSpec := &PodSpec{
		Containers: []string{"nginx", "redis"},
	}

	// Variables
	var expectedContainerNum = 2

	// Successful conversion test
	unstructuredObj, err := StructToUnstructured(podSpec)
	if err != nil {
		t.Errorf("Expected no error in conversion, got %v", err)
	}
	if unstructuredObj == nil {
		t.Errorf("Expected a non-nil unstructured object")
	}

	// Verify the content of the unstructured object
	containers, found, err := unstructured.NestedStringSlice(unstructuredObj.Object, "containers")
	if err != nil || !found {
		t.Errorf("Failed to find 'containers' in the unstructured object: %v", err)
	}
	if len(containers) != expectedContainerNum || containers[0] != "nginx" || containers[1] != "redis" {
		t.Errorf("Containers did not match expected values, got: %v", containers)
	}

	// Test with a type that cannot be converted to unstructured (like channels, functions)
	invalidType := make(chan int)               // This inherently is a pointer
	_, err = StructToUnstructured(&invalidType) // Invalid use case, not typically possible but demonstrates error handling
	if err == nil {
		t.Error("Expected an error when trying to convert a non-convertible type to unstructured")
	}
}

func TestResourceMetadataRegexValid(t *testing.T) {
	ok, err := ResourceMetadataRegexValid("valid-name")
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if !ok {
		t.Fatalf("Expected ok=true")
	}

	ok, err = ResourceMetadataRegexValid("k8s_invalid(")
	if err != nil {
		t.Fatalf("Expected no error due to unanchored regex, got %v", err)
	}
	if !ok {
		t.Fatalf("Expected ok=true due to unanchored regex")
	}
}

func TestCrExists(t *testing.T) {
	gvr := schema.GroupVersionResource{Group: "example.com", Version: "v1", Resource: "widgets"}

	s := runtime.NewScheme()
	obj := &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "example.com/v1",
		"kind":       "Widget",
		"metadata": map[string]interface{}{
			"name":      "w1",
			"namespace": "default",
		},
	}}

	dyn := dynamicfake.NewSimpleDynamicClient(s, obj)

	exists, err := CrExists(dyn, "w1", "default", gvr)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if !exists {
		t.Fatalf("Expected exists=true")
	}

	exists, err = CrExists(dyn, "missing", "default", gvr)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if exists {
		t.Fatalf("Expected exists=false")
	}
}

func TestLoadKubeConfigFromBytes_Invalid(t *testing.T) {
	_, err := LoadKubeConfigFromBytes([]byte("not yaml"))
	if err == nil {
		t.Fatalf("Expected error, got nil")
	}
}
