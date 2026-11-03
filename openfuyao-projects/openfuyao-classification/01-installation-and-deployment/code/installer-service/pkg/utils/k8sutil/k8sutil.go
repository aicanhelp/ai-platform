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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"

	"installer-service/pkg/zlog"
)

const (
	requestLastTime = 10
)

// GetSecret 通过name 和 namespace 获取 secret
func GetSecret(clientset kubernetes.Interface, secretName, namespace string) (*v1.Secret, error) {
	secret, err := clientset.CoreV1().Secrets(namespace).
		Get(context.Background(), secretName, metav1.GetOptions{})
	if err != nil {
		zlog.Errorf("Secret %s lookup failed, err: %v", secretName, err)
		return nil, err
	}
	zlog.Debugf("Secret %s found in namespace %s", secret.Name, secret.Namespace)
	return secret, nil
}

// GetConfigMap 通过name 和 namespace 获取 configMap
func GetConfigMap(clientset kubernetes.Interface, configmapName, namespace string) (*v1.ConfigMap, error) {
	configMap, err := clientset.CoreV1().ConfigMaps(namespace).
		Get(context.Background(), configmapName, metav1.GetOptions{})
	if err != nil {
		zlog.Errorf("ConfigMap %s lookup failed, err: %v", configmapName, err)
		return nil, err
	}
	zlog.Debugf("ConfigMap %s found in namespace %s", configMap.Name, configMap.Namespace)
	return configMap, nil
}
