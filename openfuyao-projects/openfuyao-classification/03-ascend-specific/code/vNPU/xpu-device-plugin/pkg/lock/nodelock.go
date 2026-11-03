/*
Copyright(C)2020-2023. Huawei Technologies Co.,Ltd. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package lock

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"

	"huawei.com/vxpu-device-plugin/pkg/log"
)

const (
	maxLockRetry      = 5
	lockRetryInterval = 100 * time.Millisecond
	lockExpiredTime   = 5 * time.Minute
)

var kubeClient kubernetes.Interface

func GetClient() kubernetes.Interface {
	return kubeClient
}

func SetClient(client kubernetes.Interface) {
	kubeClient = client
}

func NewClient() error {
	kubeConfig := os.Getenv("KUBECONFIG")
	if kubeConfig == "" {
		kubeConfig = filepath.Join(os.Getenv("HOME"), ".kube", "config")
	}
	config, err := rest.InClusterConfig()
	if err != nil {
		config, err = clientcmd.BuildConfigFromFlags("", kubeConfig)
		if err != nil {
			return err
		}
	}
	client, err := kubernetes.NewForConfig(config)
	kubeClient = client
	return err
}

func retryUpdateNode(ctx context.Context, nodeName string, modify func(*corev1.Node)) error {
	var err error
	for i := 1; i <= maxLockRetry; i++ {
		time.Sleep(lockRetryInterval)
		node, getErr := kubeClient.CoreV1().Nodes().Get(ctx, nodeName, v1.GetOptions{})
		if getErr != nil {
			log.Errorf("Failed to get node when retry to update: %v, node: %s, retry: %d", getErr, nodeName, i)
			err = getErr
			continue
		}
		newNode := node.DeepCopy()
		modify(newNode)
		_, err = kubeClient.CoreV1().Nodes().Update(ctx, newNode, v1.UpdateOptions{})
		if err == nil {
			return nil
		}
		log.Errorf("Failed to update node: %v, node: %s, retry: %d", err, nodeName, i)
	}
	return err
}

func setNodeLock(nodeName string, lockName string, namespace string, podName string) error {
	ctx := context.Background()
	node, err := kubeClient.CoreV1().Nodes().Get(ctx, nodeName, v1.GetOptions{})
	if err != nil {
		log.Errorln("get node failed", err.Error())
		return err
	}
	if lockName == "" {
		return fmt.Errorf("lockName is empty")
	}
	if _, ok := node.ObjectMeta.Annotations[lockName]; ok {
		return fmt.Errorf("node %s is locked", nodeName)
	}
	newNode := node.DeepCopy()
	newNode.ObjectMeta.Annotations[lockName] = formatLockValue(namespace, podName)
	_, err = kubeClient.CoreV1().Nodes().Update(ctx, newNode, v1.UpdateOptions{})
	if err != nil {
		err = retryUpdateNode(ctx, nodeName, func(n *corev1.Node) {
			n.ObjectMeta.Annotations[lockName] = formatLockValue(namespace, podName)
		})
	}
	if err != nil {
		return fmt.Errorf("setNodeLock exceeds retry count %d", maxLockRetry)
	}
	log.Infof("Node lock set, node: %s, pod: %s/%s", nodeName, namespace, podName)
	return nil
}

func ReleaseNodeLock(nodeName string, lockName string) error {
	ctx := context.Background()
	node, err := kubeClient.CoreV1().Nodes().Get(ctx, nodeName, v1.GetOptions{})
	if err != nil {
		return err
	}
	if _, ok := node.ObjectMeta.Annotations[lockName]; !ok {
		log.Infof("Node lock not set, node: %s", nodeName)
		return nil
	}
	newNode := node.DeepCopy()
	delete(newNode.ObjectMeta.Annotations, lockName)
	_, err = kubeClient.CoreV1().Nodes().Update(ctx, newNode, v1.UpdateOptions{})
	if err != nil {
		err = retryUpdateNode(ctx, nodeName, func(n *corev1.Node) {
			delete(n.ObjectMeta.Annotations, lockName)
		})
	}
	if err != nil {
		return fmt.Errorf("releaseNodeLock exceeds retry count %d", maxLockRetry)
	}
	log.Infof("Node lock released, node: %s", nodeName)
	return nil
}

// ReleaseNodeLockWithPod releases a node lock only if the pod identity matches
func ReleaseNodeLockWithPod(nodeName string, lockName string, namespace string, podName string) error {
	ctx := context.Background()
	node, err := kubeClient.CoreV1().Nodes().Get(ctx, nodeName, v1.GetOptions{})
	if err != nil {
		return err
	}
	value, ok := node.ObjectMeta.Annotations[lockName]
	if !ok {
		log.Infof("Node lock not set, node: %s", nodeName)
		return nil
	}
	state, parseErr := ParseNodeLock(value)
	if parseErr == nil && state.Namespace != "" && state.PodName != "" {
		if state.Namespace != namespace || state.PodName != podName {
			log.Infof("Node lock held by different pod, node: %s, lockPod: %s/%s, requestPod: %s/%s",
				nodeName, state.Namespace, state.PodName, namespace, podName)
			return fmt.Errorf("node %s lock held by %s/%s, not %s/%s", nodeName, state.Namespace, state.PodName, namespace, podName)
		}
	}
	newNode := node.DeepCopy()
	delete(newNode.ObjectMeta.Annotations, lockName)
	_, err = kubeClient.CoreV1().Nodes().Update(ctx, newNode, v1.UpdateOptions{})
	if err != nil {
		err = retryUpdateNode(ctx, nodeName, func(n *corev1.Node) {
			delete(n.ObjectMeta.Annotations, lockName)
		})
	}
	if err != nil {
		return fmt.Errorf("releaseNodeLockWithPod exceeds retry count %d", maxLockRetry)
	}
	log.Infof("Node lock released, node: %s, pod: %s/%s", nodeName, namespace, podName)
	return nil
}

// ObtainLockNode tries to acquire a node lock (without pod identity, for backward compatibility)
func ObtainLockNode(nodeName string, lockName string) error {
	return ObtainLockNodeWithPod(nodeName, lockName, "", "")
}

// ObtainLockNodeWithPod tries to acquire a node lock with pod identity
func ObtainLockNodeWithPod(nodeName string, lockName string, namespace string, podName string) error {
	ctx := context.Background()
	node, err := kubeClient.CoreV1().Nodes().Get(ctx, nodeName, v1.GetOptions{})
	if err != nil {
		return err
	}
	value, ok := node.ObjectMeta.Annotations[lockName]
	if !ok {
		return setNodeLock(nodeName, lockName, namespace, podName)
	}
	state, parseErr := ParseNodeLock(value)
	if parseErr != nil {
		log.Errorf("Failed to parse lock value: %v, node: %s, value: %s", parseErr, nodeName, value)
		return parseErr
	}
	if time.Since(state.Time) > lockExpiredTime {
		log.Infof("Node lock expired, node: %s, lockTime: %v", nodeName, state.Time)
		err = ReleaseNodeLock(nodeName, lockName)
		if err != nil {
			log.Errorf("Failed to release node lock: %v, node: %s", err, nodeName)
			return err
		}
		return setNodeLock(nodeName, lockName, namespace, podName)
	}
	return fmt.Errorf("node %s has been locked within %v", nodeName, lockExpiredTime)
}

// LockState represents the parsed state of a node lock annotation.
type LockState struct {
	Time      time.Time
	Namespace string
	PodName   string
}

// ParseNodeLock parses lock annotation value into a LockState.
func ParseNodeLock(value string) (*LockState, error) {
	parts := strings.SplitN(value, ",", 3)
	lockTime, err := time.Parse(time.RFC3339, parts[0])
	if err != nil {
		return nil, err
	}
	if len(parts) < 3 {
		return &LockState{Time: lockTime}, nil
	}
	return &LockState{Time: lockTime, Namespace: parts[1], PodName: parts[2]}, nil
}

// GetLockPodIdentity returns the namespace and pod name from a node's lock annotation
func GetLockPodIdentity(nodeName string, lockName string) (string, string, error) {
	ctx := context.Background()
	node, err := kubeClient.CoreV1().Nodes().Get(ctx, nodeName, v1.GetOptions{})
	if err != nil {
		return "", "", err
	}
	value, ok := node.ObjectMeta.Annotations[lockName]
	if !ok {
		return "", "", nil
	}
	state, parseErr := ParseNodeLock(value)
	if parseErr != nil {
		return "", "", parseErr
	}
	return state.Namespace, state.PodName, nil
}

func formatLockValue(namespace string, podName string) string {
	if namespace == "" || podName == "" {
		return time.Now().Format(time.RFC3339)
	}
	return fmt.Sprintf("%s,%s,%s", time.Now().Format(time.RFC3339), namespace, podName)
}
