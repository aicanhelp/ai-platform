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
	"k8s.io/apimachinery/pkg/runtime/schema"
	dynamicfake "k8s.io/client-go/dynamic/fake"
	k8sfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/kubernetes/scheme"
)

func TestLoadKubeConfigFromBytes(t *testing.T) {
	_, err := LoadKubeConfigFromBytes([]byte("invalid-content"))
	if err == nil {
		t.Errorf("Expected error for invalid kubeconfig, got nil")
	}

	kubeConfigYaml := []byte(`
apiVersion: v1
kind: Config
clusters:
- cluster:
    server: xxxxx
  name: test
contexts:
- context:
    cluster: test
    user: test
  name: test
current-context: test
users:
- name: test
  user:
    token: dummytoken
`)
	cfg, err := LoadKubeConfigFromBytes(kubeConfigYaml)
	if err != nil {
		t.Errorf("Expected no error for valid kubeconfig, got %v", err)
	}
	if cfg == nil {
		t.Errorf("Expected non-nil config for valid kubeconfig")
	}
}

// TestGetSecret tests the GetSecret function
func TestGetSecret(t *testing.T) {
	clientset := k8sfake.NewSimpleClientset(&v1.Secret{
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

	secret, err = GetSecret(clientset, "not-existing-secret", "default")
	if err == nil {
		t.Errorf("Expected error, got nil")
	}
	if secret != nil {
		t.Errorf("Expected not getting secret, got %v", secret)
	}
}

func TestListSecret(t *testing.T) {
	clientset := k8sfake.NewSimpleClientset(&v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-secret",
			Namespace: "default",
		},
	})

	secretList, err := ListSecret(clientset, "default")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if len(secretList.Items) != 1 {
		t.Errorf("Expected 1 secret, got %v", len(secretList.Items))
	}
	secret := secretList.Items[0]
	if secretList.Items[0].Name != "test-secret" {
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

	clientset := k8sfake.NewSimpleClientset()
	createdSecret, err := CreateSecret(clientset, newSecret)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if createdSecret.Name != "new-secret" {
		t.Errorf("Expected secret name %s, got %s", "new-secret", createdSecret.Name)
	}

	// 失败场景：已存在同名Secret
	_, _ = CreateSecret(clientset, newSecret) // 再次创建
	_, err = CreateSecret(clientset, newSecret)
	if err == nil {
		t.Errorf("Expected error when creating duplicate secret, got nil")
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
	clientset := k8sfake.NewSimpleClientset(initialSecret)

	// Update the secret
	initialSecret.Data["key"] = []byte("new-value")
	updatedSecret, err := UpdateSecret(clientset, initialSecret)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if string(updatedSecret.Data["key"]) != "new-value" {
		t.Errorf("Expected updated secret value %s, got %s", "new-value", string(updatedSecret.Data["key"]))
	}

	// 失败场景：更新不存在的Secret
	nonExistSecret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "not-exist-secret",
			Namespace: "default",
		},
	}
	_, err = UpdateSecret(clientset, nonExistSecret)
	if err == nil {
		t.Errorf("Expected error when updating non-exist secret, got nil")
	}
}

// TestDeleteSecret tests the DeleteSecret function
func TestDeleteSecret(t *testing.T) {
	secretName := "delete-secret"
	namespace := "default"
	clientset := k8sfake.NewSimpleClientset(&v1.Secret{
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

	// 失败场景：删除不存在的Secret
	err = DeleteSecret(clientset, "not-exist-secret", namespace)
	if err == nil {
		t.Errorf("Expected error when deleting non-exist secret, got nil")
	}
}

// TestGetConfigMap tests the GetConfigMap function
func TestGetConfigMap(t *testing.T) {
	clientset := k8sfake.NewSimpleClientset(&v1.ConfigMap{
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

	// 失败场景：获取不存在的ConfigMap
	_, err = GetConfigMap(clientset, "not-exist-configmap", "default")
	if err == nil {
		t.Errorf("Expected error when getting non-exist configmap, got nil")
	}
}

func TestListConfigmap(t *testing.T) {
	clientset := k8sfake.NewSimpleClientset(&v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-configmap",
			Namespace: "default",
		},
	})

	configMapList, err := ListConfigMap(clientset, "default")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if len(configMapList.Items) != 1 {
		t.Errorf("Expected 1 configmap, got %v", len(configMapList.Items))
	}
	configMap := configMapList.Items[0]
	if configMap.Name != "test-configmap" {
		t.Errorf("Expected configMap name %s, got %s", "test-configmap", configMap.Name)
	}
}

// TestCreateConfigMap tests the CreateConfigMap function
func TestCreateConfigMap(t *testing.T) {
	clientset := k8sfake.NewSimpleClientset()
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

	// 失败场景：已存在同名ConfigMap
	_, err = CreateConfigMap(clientset, configMap)
	if err == nil {
		t.Errorf("Expected error when creating duplicate configmap, got nil")
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
	clientset := k8sfake.NewSimpleClientset(configMap)

	configMap.Data = map[string]string{"key": "new-value"}
	updatedConfigMap, err := UpdateConfigMap(clientset, configMap)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if updatedConfigMap.Data["key"] != "new-value" {
		t.Errorf("Expected updated configMap value %s, got %s", "new-value", updatedConfigMap.Data["key"])
	}

	// 失败场景：更新不存在的ConfigMap
	nonExistConfigMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "not-exist-configmap",
			Namespace: "default",
		},
	}
	_, err = UpdateConfigMap(clientset, nonExistConfigMap)
	if err == nil {
		t.Errorf("Expected error when updating non-exist configmap, got nil")
	}
}

// TestDeleteConfigMap tests the DeleteConfigMap function
func TestDeleteConfigMap(t *testing.T) {
	clientset := k8sfake.NewSimpleClientset(&v1.ConfigMap{
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

	// 失败场景：删除不存在的ConfigMap
	err = DeleteConfigMap(clientset, "not-exist-configmap", "default")
	if err == nil {
		t.Errorf("Expected error when deleting non-exist configmap, got nil")
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
	var notExpectedContainer = 2
	if len(containers) != notExpectedContainer || containers[0] != "nginx" || containers[1] != "redis" {
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
	type args struct {
		metadataName string
	}
	tests := []struct {
		name    string
		args    args
		want    bool
		wantErr bool
	}{
		{
			name: "RegexNotMatch",
			args: args{
				metadataName: "----",
			},
			want:    false,
			wantErr: true,
		},
		{
			name: "RegexMatch",
			args: args{
				metadataName: "example.com",
			},
			want:    true,
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ResourceMetadataRegexValid(tt.args.metadataName)
			if (err != nil) != tt.wantErr {
				t.Errorf("ResourceMetadataRegexValid() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("ResourceMetadataRegexValid() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestRegexError(t *testing.T) {
	err := &RegexError{Message: "invalid pattern"}
	if got := err.Error(); got != "RegexError: invalid pattern" {
		t.Errorf("RegexError.Error() = %q, want RegexError: invalid pattern", got)
	}
}

func TestCrExists(t *testing.T) {
	gvr := schema.GroupVersionResource{Group: "test.io", Version: "v1", Resource: "widgets"}

	t.Run("resource_exists", func(t *testing.T) {
		dynamicClient := dynamicfake.NewSimpleDynamicClientWithCustomListKinds(
			scheme.Scheme,
			map[schema.GroupVersionResource]string{
				gvr: "WidgetList",
			},
			&unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "test.io/v1",
					"kind":       "Widget",
					"metadata": map[string]interface{}{
						"name":      "widget-a",
						"namespace": "default",
					},
				},
			},
		)
		exists, err := CrExists(dynamicClient, "widget-a", "default", gvr)
		if err != nil {
			t.Fatalf("CrExists() error = %v", err)
		}
		if !exists {
			t.Error("CrExists() = false, want true")
		}
	})

	t.Run("resource_not_found", func(t *testing.T) {
		dynamicClient := dynamicfake.NewSimpleDynamicClient(scheme.Scheme)
		exists, err := CrExists(dynamicClient, "missing", "default", gvr)
		if err != nil {
			t.Fatalf("CrExists() error = %v", err)
		}
		if exists {
			t.Error("CrExists() = true, want false")
		}
	})
}
