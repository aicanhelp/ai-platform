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

// Package k8sutil contains public function for the console-service project
package k8sutil

import (
	"context"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"

	"volcano-config-service/pkg/constant"
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

func TestDeleteVolcanoPod(t *testing.T) {
	clientset := fake.NewSimpleClientset()

	err := DeleteVolcanoPod(clientset, constant.VolcanoConfigServiceDefaultNamespace)
	if err == nil {
		t.Errorf("Expected error, got %v", err)
	}
}

func TestGetNodeList(t *testing.T) {
	clientset := fake.NewSimpleClientset()

	err, _ := GetNodeList(clientset)
	if err == nil {
		t.Errorf("Expected error, got %v", err)
	}
}

func TestGetNodeListByName(t *testing.T) {
	clientset := fake.NewSimpleClientset()

	err, _ := GetNodeListByName(clientset, "name")
	if err == nil {
		t.Errorf("Expected error, got %v", err)
	}
}

func TestGetScheduler(t *testing.T) {
	clientset := fake.NewSimpleClientset()

	err, _ := GetScheduler(clientset)
	if err != nil {
		t.Errorf("Expected error, got %v", err)
	}
}

func TestPodUuidToName(t *testing.T) {
	clientset := fake.NewSimpleClientset()

	_, err := PodUuidToName(clientset, "1111")
	if err != nil {
		t.Errorf("Expected error, got %v", err)
	}
}

func TestExtractSchedulerName(t *testing.T) {
	strings := extractSchedulerName("volcano-scheduler-111")
	if strings != "volcano-scheduler" {
		t.Errorf("Expected error, got volcano-scheduler")
	}
}
