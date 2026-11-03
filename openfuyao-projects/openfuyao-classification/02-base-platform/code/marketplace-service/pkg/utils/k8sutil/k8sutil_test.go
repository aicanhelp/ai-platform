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
	"context"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
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
	if len(containers) != 2 || containers[0] != "nginx" || containers[1] != "redis" {
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
	tests := []struct {
		metadataName string
		expected     bool
	}{
		// Valid cases
		{"validname", true},
		{"valid-name", true},
		{"valid.name", true},
		{"valid-name123", true},
		{"valid.name-123", true},
		{"v", true},
		{"123", true},
		{"valid123.name", true},

		// Invalid cases
		{"InvalidName", false},   // Contains uppercase letters
		{"invalid_name", false},  // Contains underscore
		{"invalid~name", false},  // Contains invalid character
		{"", false},              // Empty string
		{"-invalid", false},      // Starts with hyphen
		{"invalid-", false},      // Ends with hyphen
		{".invalid", false},      // Starts with dot
		{"invalid.", false},      // Ends with dot
		{"invalid..name", false}, // Contains consecutive dots
		{"invalid/name", false},  // Contains slash
	}

	for _, test := range tests {
		result, err := ResourceMetadataRegexValid(test.metadataName)
		if err != nil && test.expected {
			t.Errorf("Unexpected error for input %q: %v", test.metadataName, err)
		}
		if result != test.expected {
			t.Errorf("ResourceMetadataRegexValid(%q) = %v; want %v", test.metadataName, result, test.expected)
		}
	}
}
