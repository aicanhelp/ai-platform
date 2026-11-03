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
	"encoding/json"
	"fmt"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	"volcano-config-service/pkg/controller"
	"volcano-config-service/pkg/zlog"
)

const (
	requestLastTime = 10

	// nodeConfigMaxRetries 是 PutNodeConfigWithRetry 的最大冲突重试次数
	nodeConfigMaxRetries = 5
	// nodeConfigRetryBaseInterval 是冲突重试线性退避的基础间隔
	nodeConfigRetryBaseInterval = 100 * time.Millisecond
)

// GetSecret looks up secret by its name and namespace
func GetSecret(clientset kubernetes.Interface, secretName, namespace string) (*v1.Secret, error) {
	secret, err := clientset.CoreV1().Secrets(namespace).
		Get(context.Background(), secretName, metav1.GetOptions{})
	if err != nil {
		zlog.Error("Secret %s lookup failed, err: %v", secretName, err)
		return nil, err
	}
	zlog.Debugf("Secret %s found in namespace %s", secret.Name, secret.Namespace)
	return secret, nil
}

// CreateSecret create secret
func CreateSecret(clientset kubernetes.Interface, secret *v1.Secret) (*v1.Secret, error) {
	secretName := secret.Name
	secret, err := clientset.CoreV1().Secrets(secret.Namespace).
		Create(context.Background(), secret, metav1.CreateOptions{})
	if err != nil {
		zlog.Debugf("Failed to create secret %s, err: %v", secretName, err)
		return nil, err
	}
	zlog.Debugf("Secret %s created in namespace %s", secretName, secret.Namespace)
	return secret, err
}

// UpdateSecret update secret
func UpdateSecret(clientset kubernetes.Interface, secret *v1.Secret) (*v1.Secret, error) {
	secretName := secret.Name
	secret, err := clientset.CoreV1().Secrets(secret.Namespace).
		Update(context.Background(), secret, metav1.UpdateOptions{})
	if err != nil {
		zlog.Debugf("Failed to update secret %s, err: %v", secretName, err)
		return nil, err
	}
	zlog.Debugf("Secret %s updated in namespace %s", secret.Name, secret.Namespace)
	return secret, err
}

// DeleteSecret delete secret
func DeleteSecret(clientset kubernetes.Interface, secretName, namespace string) error {
	ctx, cancel := context.WithTimeout(context.Background(), requestLastTime*time.Second)
	defer cancel()

	err := clientset.CoreV1().Secrets(namespace).Delete(ctx, secretName, metav1.DeleteOptions{})
	if err != nil {
		zlog.Debugf("Failed to delete secret %s, err: %v", secretName, err)
		return err
	}
	zlog.Debugf("Secret %s deleted in namespace %s\n", secretName, namespace)
	return err
}

// GetConfigMap looks up configMap by its name and namespace
func GetConfigMap(clientset kubernetes.Interface, configmapName, namespace string) (*v1.ConfigMap, error) {
	configMap, err := clientset.CoreV1().ConfigMaps(namespace).
		Get(context.Background(), configmapName, metav1.GetOptions{})
	if err != nil {
		zlog.Error("ConfigMap %s lookup failed, err: %v", configmapName, err)
		return nil, err
	}
	zlog.Debugf("ConfigMap %s found in namespace %s", configMap.Name, configMap.Namespace)
	return configMap, nil
}

// CreateConfigMap create secret
func CreateConfigMap(clientset kubernetes.Interface, configMap *v1.ConfigMap) (*v1.ConfigMap, error) {
	configMapName := configMap.Name
	configMap, err := clientset.CoreV1().ConfigMaps(configMap.Namespace).
		Create(context.Background(), configMap, metav1.CreateOptions{})
	if err != nil {
		zlog.Debugf("Failed to create configMap %s, err: %v", configMapName, err)
		return nil, err
	}
	zlog.Debugf("ConfigMap %s created in namespace %s", configMap.Name, configMap.Namespace)
	return configMap, nil
}

// UpdateConfigMap update config map
func UpdateConfigMap(clientset kubernetes.Interface, configMap *v1.ConfigMap) (*v1.ConfigMap, error) {
	configMapName := configMap.Name
	configMap, err := clientset.CoreV1().ConfigMaps(configMap.Namespace).
		Update(context.Background(), configMap, metav1.UpdateOptions{})
	if err != nil {
		zlog.Debugf("Failed to update configMap %s, err: %v", configMapName, err)
		return nil, err
	}

	zlog.Debugf("ConfigMap %s updated in namespace %s", configMap.Name, configMap.Namespace)
	return configMap, nil
}

// DeleteConfigMap delete config map
func DeleteConfigMap(clientset kubernetes.Interface, configmapName, namespace string) error {
	ctx, cancel := context.WithTimeout(context.Background(), requestLastTime*time.Second)
	defer cancel()

	err := clientset.CoreV1().ConfigMaps(namespace).Delete(ctx, configmapName, metav1.DeleteOptions{})
	if err != nil {
		zlog.Debugf("Failed to delete configmap %s, err: %v", configmapName, err)
		return err
	}
	zlog.Debugf("Configmap %s deleted in namespace %s\n", configmapName, namespace)
	return err
}

// GetNodeList Get Node List
func GetNodeList(clientset kubernetes.Interface) (*v1.NodeList, error) {
	NodeList, err := clientset.CoreV1().Nodes().List(context.Background(), metav1.ListOptions{})
	if err != nil {
		zlog.Error("Nodes lookup failed, err: %v", err)
		return nil, err
	}

	return NodeList, nil
}

// extractSchedulerName extract Scheduler Name
func extractSchedulerName(podName string) string {
	index := strings.Index(podName, "-scheduler")
	if index != -1 {
		return podName[:index+len("-scheduler")]
	}

	return ""
}

// GetScheduler Get Scheduler Name
func GetScheduler(clientset kubernetes.Interface) ([]string, error) {
	PodList, err := clientset.CoreV1().Pods("").List(context.Background(), metav1.ListOptions{})
	if err != nil {
		zlog.Error("PodList lookup failed, err: %v", err)
		return nil, err
	}

	schedulerSet := map[string]bool{}

	for _, pod := range PodList.Items {
		if !strings.Contains(pod.Name, "scheduler") {
			continue
		}
		if pod.Spec.SchedulerName == "" {
			continue
		}
		schedulerName := extractSchedulerName(pod.Name)
		if schedulerName != "" {
			schedulerSet[schedulerName] = true
		}
	}

	var availableSchedulers []string
	for scheduler := range schedulerSet {
		availableSchedulers = append(availableSchedulers, scheduler)
	}

	return availableSchedulers, nil
}

// GetNodeListByName Get Node List By Name
func GetNodeListByName(clientset kubernetes.Interface, name string) (*v1.NodeList, error) {
	NodeList, err := clientset.CoreV1().Nodes().List(context.Background(), metav1.ListOptions{})
	if err != nil {
		zlog.Error("Nodes lookup failed, err: %v", err)
		return nil, err
	}

	var filteredNodes v1.NodeList

	for _, node := range NodeList.Items {
		if strings.Contains(node.ObjectMeta.Name, name) {
			filteredNodes.Items = append(filteredNodes.Items, node)
		}
	}

	return &filteredNodes, nil
}

// GetCR Get CR
func GetCR(dynamicClient *dynamic.DynamicClient, name string) (*unstructured.Unstructured, error) {
	gvr := schema.GroupVersionResource{
		Group:    "nodeinfo.volcano.sh",
		Version:  "v1alpha1",
		Resource: "numatopologies",
	}

	CR, err := dynamicClient.Resource(gvr).Get(context.Background(), name, metav1.GetOptions{})
	if err != nil {
		zlog.Error("CRList lookup failed, err: %v", err)
		return nil, err
	}

	return CR, nil
}

// PodUuidToName get pod list and get pod name by uuid
func PodUuidToName(clientset kubernetes.Interface, uuid string) (string, error) {
	PodList, err := clientset.CoreV1().Pods("").List(context.Background(), metav1.ListOptions{})
	if err != nil {
		zlog.Error("PodList lookup failed, err: %v", err)
		return "", err
	}

	podUuid := types.UID(uuid)
	var podName string
	for _, pod := range PodList.Items {
		if pod.ObjectMeta.UID == podUuid {
			podName = pod.ObjectMeta.Name
			return podName, nil
		}
	}

	zlog.Error("uuid not find, err: %v", err)
	return "", err
}

// DeleteVolcanoPod restart volcano scheduler
func DeleteVolcanoPod(clientset kubernetes.Interface, namespace string) error {
	ctx, cancel := context.WithTimeout(context.Background(), requestLastTime*time.Second)
	defer cancel()

	podName, err := GetPodNameByPrefix(clientset, namespace, "volcano-scheduler")
	if err != nil {
		zlog.Debugf("Failed to get pod %s, err: %v", podName, err)
		return err
	}

	err = clientset.CoreV1().Pods(namespace).Delete(ctx, podName, metav1.DeleteOptions{})
	if err != nil {
		zlog.Debugf("Failed to delete pod %s, err: %v", podName, err)
		return err
	}
	zlog.Debugf("Pod %s deleted in namespace %s\n", podName, namespace)
	return err
}

// GetPodNameByPrefix get Pod Name By Prefix
func GetPodNameByPrefix(clientset kubernetes.Interface, namespace string, prefix string) (string, error) {
	pods, err := clientset.CoreV1().Pods(namespace).List(context.Background(), metav1.ListOptions{})
	if err != nil {
		return "", err
	}

	for _, pod := range pods.Items {
		if strings.HasPrefix(pod.Name, prefix) {
			return pod.Name, nil
		}
	}

	return "", fmt.Errorf("no pods found with prefix: %s", prefix)
}

// GetNodeConfig 获取 NodeConfig CRD
func GetNodeConfig(clientset kubernetes.Interface) (*controller.NodeConfig, error) {
	nodeConfig := &controller.NodeConfig{}
	err := clientset.CoreV1().RESTClient().Get().
		AbsPath("/apis/volcano.sh/v1").
		Resource("nodeconfigs").
		Name("kubelet-config").
		Do(context.TODO()).
		Into(nodeConfig)

	return nodeConfig, err
}

// PutNodeConfig 修改 NodeConfig CRD
func PutNodeConfig(clientset kubernetes.Interface, data []byte) error {
	err := clientset.CoreV1().RESTClient().Put().
		AbsPath("/apis/volcano.sh/v1").
		Resource("nodeconfigs").
		Name("kubelet-config").
		Body(data).
		Do(context.TODO()).
		Error()

	return err
}

// PutNodeConfigWithRetry 以冲突重试方式修改 NodeConfig CRD。
// applyFunc 接收最新的 NodeConfig 并返回需要序列化提交的修改后对象。
// 当 PUT 因 resourceVersion 冲突失败时（通常因为 numaadj DaemonSet
// 同时在写入节点注册信息），会重新获取最新版本并重新应用修改后重试，
// 避免覆盖其他实例刚写入的节点数据。
func PutNodeConfigWithRetry(clientset kubernetes.Interface,
	applyFunc func(latest *controller.NodeConfig) (*controller.NodeConfig, error)) error {

	var lastErr error

	for i := 0; i < nodeConfigMaxRetries; i++ {
		// 获取最新的 NodeConfig
		latest, err := GetNodeConfig(clientset)
		if err != nil {
			return fmt.Errorf("failed to get NodeConfig: %v", err)
		}

		// 应用修改
		modified, err := applyFunc(latest)
		if err != nil {
			return err
		}

		// 设置必要的元数据字段
		modified.Kind = "NodeConfig"
		modified.APIVersion = "volcano.sh/v1"

		data, err := json.Marshal(modified)
		if err != nil {
			return fmt.Errorf("failed to marshal NodeConfig: %v", err)
		}

		// 提交修改
		err = PutNodeConfig(clientset, data)
		if err == nil {
			return nil
		}

		lastErr = err
		// 判断是否为冲突错误（HTTP 409 Conflict）
		if !isConflictError(err) {
			// 非冲突错误，直接返回
			return err
		}

		zlog.FormatInfo("NodeConfig conflict detected (attempt %d/%d), retrying...",
			i+1, nodeConfigMaxRetries)
		time.Sleep(time.Duration(i+1) * nodeConfigRetryBaseInterval)
	}

	return fmt.Errorf("failed to update NodeConfig after %d retries: %v", nodeConfigMaxRetries, lastErr)
}

// isConflictError 判断错误是否为 HTTP 409 Conflict。
// REST client 的 Do().Error() 返回 *apierrors.StatusError，
// apierrors.IsConflict 基于结构化 Status 判断，可靠且无误判。
func isConflictError(err error) bool {
	return apierrors.IsConflict(err)
}

// GetOenuma 获取 Oenuma CRD
func GetOenuma(clientset kubernetes.Interface) (*controller.Oenuma, error) {
	oenuma := &controller.Oenuma{}
	err := clientset.CoreV1().RESTClient().Get().
		AbsPath("/apis/resource.sop.huawei.com/v1alpha1").
		Namespace("tcs").
		Resource("oenumas").
		Name("podafi").
		Do(context.TODO()).
		Into(oenuma)

	return oenuma, err
}

// PutOenuma 修改 Oenuma CRD
func PutOenuma(clientset kubernetes.Interface, data []byte) error {
	err := clientset.CoreV1().RESTClient().Put().
		AbsPath("/apis/resource.sop.huawei.com/v1alpha1").
		Namespace("tcs").
		Resource("oenumas").
		Name("podafi").
		Body(data).
		Do(context.TODO()).
		Error()

	return err
}

// InitClient initializes kubeClient
func InitClient() (*kubernetes.Clientset, error) {
	config, err := rest.InClusterConfig()
	if err != nil {
		return nil, err
	}

	Client, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, err
	}

	return Client, nil
}
