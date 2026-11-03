/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package utils

import (
	"context"
	"fmt"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
)

// ResourceManager 资源管理器
type ResourceManager struct {
	executor      *executor.SSHExecutor
	dynamicClient dynamic.Interface
	clientSet     *kubernetes.Clientset
}

// NewResourceManager 创建资源管理器
func NewResourceManager(exec *executor.SSHExecutor, dynamicClient dynamic.Interface, clientSet *kubernetes.Clientset) *ResourceManager {
	return &ResourceManager{
		executor:      exec,
		dynamicClient: dynamicClient,
		clientSet:     clientSet,
	}
}

// CheckOrCreateNamespace 检查命名空间是否存在，如果不存在则创建
func (r *ResourceManager) CheckOrCreateNamespace(namespace string) error {
	_, err := r.clientSet.CoreV1().Namespaces().Get(context.TODO(), namespace, metav1.GetOptions{})
	if err != nil {
		if errors.IsNotFound(err) {
			_, err := r.clientSet.CoreV1().Namespaces().Create(context.TODO(), &corev1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: namespace,
				},
			}, metav1.CreateOptions{})
			if err != nil {
				return fmt.Errorf("创建命名空间 %s 时发生错误: %w", namespace, err)
			}
			return nil
		}
		return fmt.Errorf("检查命名空间 %s 时发生错误: %w", namespace, err)
	}
	return nil
}

// CreateConfigMap 创建 ConfigMap
func (r *ResourceManager) CreateConfigMap(name, namespace, key, data string) error {
	if err := r.CheckOrCreateNamespace(namespace); err != nil {
		return err
	}

	// 检查 ConfigMap 是否已存在
	_, err := r.clientSet.CoreV1().ConfigMaps(namespace).Get(context.TODO(), name, metav1.GetOptions{})
	if err == nil {
		configMap, err := r.clientSet.CoreV1().ConfigMaps(namespace).Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("获取现有 ConfigMap %s 时发生错误: %w", name, err)
		}

		// 更新 ConfigMap 的数据
		if configMap.Data == nil {
			configMap.Data = make(map[string]string)
		}
		configMap.Data[key] = data

		_, err = r.clientSet.CoreV1().ConfigMaps(namespace).Update(context.TODO(), configMap, metav1.UpdateOptions{})
		if err != nil {
			return fmt.Errorf("更新 ConfigMap %s 时发生错误: %w", name, err)
		}
		return nil
	} else if !errors.IsNotFound(err) {
		return fmt.Errorf("检查 ConfigMap %s 时发生错误: %w", name, err)
	}

	// ConfigMap不存在，创建新的
	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Data: map[string]string{
			key: data,
		},
	}

	_, err = r.clientSet.CoreV1().ConfigMaps(namespace).Create(context.TODO(), configMap, metav1.CreateOptions{})
	if err != nil {
		return fmt.Errorf("创建 ConfigMap %s 时发生错误: %w", name, err)
	}
	return nil
}

// BatchCreateConfigMaps 批量创建多个 ConfigMap
func (r *ResourceManager) BatchCreateConfigMaps(configMaps []map[string]string) error {
	for _, cm := range configMaps {
		name := cm["name"]
		namespace := cm["namespace"]
		key := cm["key"]
		data := cm["data"]

		err := r.CreateConfigMap(name, namespace, key, data)
		if err != nil {
			return fmt.Errorf("批量创建 ConfigMap 时发生错误，ConfigMap %s: %w", name, err)
		}
	}

	return nil
}

// CheckConfigMapExists 检查 ConfigMap 是否存在
func (r *ResourceManager) CheckConfigMapExists(name, namespace string) (bool, error) {
	_, err := r.clientSet.CoreV1().ConfigMaps(namespace).Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		if errors.IsNotFound(err) {
			return false, nil
		}
		return false, err
	}
	return true, nil
}
