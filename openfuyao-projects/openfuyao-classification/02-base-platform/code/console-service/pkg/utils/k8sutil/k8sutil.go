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
	"fmt"
	"regexp"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"

	"console-service/pkg/constant"
	"console-service/pkg/zlog"
)

const (
	requestLastTime = 10
)

// LoadKubeConfigFromBytes parses the kubeconfig yaml data to the rest.Config struct.
func LoadKubeConfigFromBytes(kubeconfig []byte) (*rest.Config, error) {

	// 验证 kubeconfig 的结构和数据的合法性
	clientConfig, err := clientcmd.NewClientConfigFromBytes(kubeconfig)
	if err != nil {
		return nil, err
	}

	// kubeconfig 中包含的数据如服务器地址、认证信息等有问题，如不可访问的服务器地址或无效的认证信息
	config, err := clientConfig.ClientConfig()
	if err != nil {
		return nil, err
	}
	return config, nil
}

// GetSecret looks up secret by its name and namespace
func GetSecret(clientset kubernetes.Interface, secretName, namespace string) (*v1.Secret, error) {
	secret, err := clientset.CoreV1().Secrets(namespace).
		Get(context.Background(), secretName, metav1.GetOptions{})
	if err != nil {
		zlog.Errorf("Secret %s lookup failed, err: %v", secretName, err)
		return nil, err
	}
	zlog.Infof("Secret %s found in namespace %s", secret.Name, secret.Namespace)
	return secret, nil
}

// ListSecret lists all secrets by namespace
func ListSecret(clientset kubernetes.Interface, namespace string) (*v1.SecretList, error) {
	secretList, err := clientset.CoreV1().Secrets(namespace).
		List(context.Background(), metav1.ListOptions{})
	if err != nil {
		zlog.Errorf("Listing Secret failed, err: %v", err)
		return nil, err
	}
	zlog.Infof("List all Secrets in namespace %s", namespace)
	return secretList, nil
}

// CreateSecret create secret
func CreateSecret(clientset kubernetes.Interface, secret *v1.Secret) (*v1.Secret, error) {
	secretName := secret.Name
	secret, err := clientset.CoreV1().Secrets(secret.Namespace).
		Create(context.Background(), secret, metav1.CreateOptions{})
	if err != nil {
		zlog.Infof("Failed to create secret %s, err: %v", secretName, err)
		return nil, err
	}
	zlog.Infof("Secret %s created in namespace %s", secretName, secret.Namespace)
	return secret, err
}

// UpdateSecret update secret
func UpdateSecret(clientset kubernetes.Interface, secret *v1.Secret) (*v1.Secret, error) {
	secretName := secret.Name
	secret, err := clientset.CoreV1().Secrets(secret.Namespace).
		Update(context.Background(), secret, metav1.UpdateOptions{})
	if err != nil {
		zlog.Infof("Failed to update secret %s, err: %v", secretName, err)
		return nil, err
	}
	zlog.Infof("Secret %s updated in namespace %s", secret.Name, secret.Namespace)
	return secret, err
}

// DeleteSecret delete secret
func DeleteSecret(clientset kubernetes.Interface, secretName, namespace string) error {
	ctx, cancel := context.WithTimeout(context.Background(), requestLastTime*time.Second)
	defer cancel()

	err := clientset.CoreV1().Secrets(namespace).Delete(ctx, secretName, metav1.DeleteOptions{})
	if err != nil {
		zlog.Infof("Failed to delete secret %s, err: %v", secretName, err)
		return err
	}
	zlog.Infof("Secret %s deleted in namespace %s\n", secretName, namespace)
	return err
}

// GetConfigMap looks up configMap by its name and namespace
func GetConfigMap(clientset kubernetes.Interface, configmapName, namespace string) (*v1.ConfigMap, error) {
	configMap, err := clientset.CoreV1().ConfigMaps(namespace).
		Get(context.Background(), configmapName, metav1.GetOptions{})
	if err != nil {
		zlog.Errorf("ConfigMap %s lookup failed, err: %v", configmapName, err)
		return nil, err
	}
	zlog.Infof("ConfigMap %s found in namespace %s", configMap.Name, configMap.Namespace)
	return configMap, nil
}

// ListConfigMap lists all secrets by namespace
func ListConfigMap(clientset kubernetes.Interface, namespace string) (*v1.ConfigMapList, error) {
	configmapList, err := clientset.CoreV1().ConfigMaps(namespace).
		List(context.Background(), metav1.ListOptions{})
	if err != nil {
		zlog.Errorf("Listing Configmap failed, err: %v", err)
		return nil, err
	}
	zlog.Infof("List all Configmap in namespace %s", namespace)
	return configmapList, nil
}

// CreateConfigMap create secret
func CreateConfigMap(clientset kubernetes.Interface, configMap *v1.ConfigMap) (*v1.ConfigMap, error) {
	configMapName := configMap.Name
	configMap, err := clientset.CoreV1().ConfigMaps(configMap.Namespace).
		Create(context.Background(), configMap, metav1.CreateOptions{})
	if err != nil {
		zlog.Infof("Failed to create configMap %s, err: %v", configMapName, err)
		return nil, err
	}
	zlog.Infof("ConfigMap %s created in namespace %s", configMap.Name, configMap.Namespace)
	return configMap, nil
}

// UpdateConfigMap update config map
func UpdateConfigMap(clientset kubernetes.Interface, configMap *v1.ConfigMap) (*v1.ConfigMap, error) {
	configMapName := configMap.Name
	configMap, err := clientset.CoreV1().ConfigMaps(configMap.Namespace).
		Update(context.Background(), configMap, metav1.UpdateOptions{})
	if err != nil {
		zlog.Infof("Failed to update configMap %s, err: %v", configMapName, err)
		return nil, err
	}

	zlog.Infof("ConfigMap %s updated in namespace %s", configMap.Name, configMap.Namespace)
	return configMap, nil
}

// DeleteConfigMap delete config map
func DeleteConfigMap(clientset kubernetes.Interface, configmapName, namespace string) error {
	ctx, cancel := context.WithTimeout(context.Background(), requestLastTime*time.Second)
	defer cancel()

	err := clientset.CoreV1().ConfigMaps(namespace).Delete(ctx, configmapName, metav1.DeleteOptions{})
	if err != nil {
		zlog.Infof("Failed to delete secret %s, err: %v", configmapName, err)
		return err
	}
	zlog.Infof("Configmap %s deleted in namespace %s\n", configmapName, namespace)
	return err
}

// CrExists check cr existence by metadata.name
// for cluster level resource, parameter namespace should be empty
func CrExists(dynamicClient dynamic.Interface, resourceName, namespace string,
	gvr schema.GroupVersionResource) (bool, error) {
	_, err := dynamicClient.Resource(gvr).Namespace(namespace).
		Get(context.Background(), resourceName, metav1.GetOptions{})
	if err != nil {
		if errors.IsNotFound(err) {
			return false, nil
		}
		zlog.Errorf("check CR existence failed: %v", err)
		return false, err
	}
	return true, nil
}

// StructToUnstructured convert from any struct to Unstructured struct
// mainly for kubernetes usage
func StructToUnstructured(v interface{}) (*unstructured.Unstructured, error) {
	UnstructuredResult, err := runtime.DefaultUnstructuredConverter.ToUnstructured(v)
	if err != nil {
		return nil, err
	}
	return &unstructured.Unstructured{
		Object: UnstructuredResult,
	}, nil
}

// ResourceMetadataRegexValid sanitize metadata.name
// return lowercase for the parameter
// return error if regular expression failed, same regex used for k8s
func ResourceMetadataRegexValid(metadataName string) (bool, error) {
	re, err := regexp.Compile(constant.MetadataNameRegExPattern)
	if err != nil {
		return false, err
	}
	if !re.MatchString(metadataName) {
		return false, &RegexError{
			Message: "Input does not match metadata.name regex pattern",
		}
	}
	return true, nil
}

// RegexError error struct for regular expression error
type RegexError struct {
	Message string
}

func (e *RegexError) Error() string {
	return fmt.Sprintf("RegexError: %s", e.Message)
}
