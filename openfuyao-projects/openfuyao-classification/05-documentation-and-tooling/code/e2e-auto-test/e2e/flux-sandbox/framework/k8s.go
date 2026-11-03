/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package framework

import (
	"context"
	"fmt"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	"sigs.k8s.io/controller-runtime/pkg/client"

	apiv1alpha1 "gitcode.com/openFuyao/e2e-auto-test/e2e/flux-sandbox/api/v1alpha1"
)

// AgentPodLabel selects the e2e agent DaemonSet pods. The watcher recognizes
// this label as an agent pod for its fast cleanup path.
const AgentPodLabel = "app=flux-sandbox-agent"

// K8s wraps the typed clients used by the e2e framework.
type K8s struct {
	// Client is the controller-runtime client (CRD-aware).
	Client client.Client
	// Kube is the clientset for core operations (pods, nodes).
	Kube *kubernetes.Clientset
	// Cfg is the e2e configuration.
	Cfg *Config
}

// NewK8s builds the clients from the kubeconfig in cfg.
func NewK8s(cfg *Config) (*K8s, error) {
	restCfg, err := clientcmd.BuildConfigFromFlags("", cfg.Kubeconfig)
	if err != nil {
		return nil, fmt.Errorf("build kubeconfig: %w", err)
	}

	scheme := runtime.NewScheme()
	for _, add := range []func(*runtime.Scheme) error{
		corev1.AddToScheme,
		appsv1.AddToScheme,
		rbacv1.AddToScheme,
		apiv1alpha1.AddToScheme,
	} {
		if err := add(scheme); err != nil {
			return nil, err
		}
	}

	c, err := client.New(restCfg, client.Options{Scheme: scheme})
	if err != nil {
		return nil, fmt.Errorf("create controller-runtime client: %w", err)
	}
	kube, err := kubernetes.NewForConfig(restCfg)
	if err != nil {
		return nil, fmt.Errorf("create clientset: %w", err)
	}
	return &K8s{Client: c, Kube: kube, Cfg: cfg}, nil
}

// ApplySandboxCR creates the Sandbox CR named name (recreating it if it
// already exists) and returns its UID.
func (k *K8s) ApplySandboxCR(name string) (string, error) {
	ctx := context.Background()
	if err := k.DeleteSandboxCR(name); err != nil {
		return "", err
	}
	sb := &apiv1alpha1.Sandbox{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: k.Cfg.Namespace,
			Labels:    map[string]string{"sandbox.flux.io/sandbox-group": "e2e-test"},
		},
		Spec: apiv1alpha1.SandboxSpec{
			SandboxGroup: "e2e-test",
			Image:        "docker.io/library/busybox:latest",
		},
	}
	if err := k.Client.Create(ctx, sb); err != nil {
		return "", fmt.Errorf("create Sandbox CR %s: %w", name, err)
	}
	created := &apiv1alpha1.Sandbox{}
	if err := k.Client.Get(ctx, types.NamespacedName{Name: name, Namespace: k.Cfg.Namespace}, created); err != nil {
		return "", fmt.Errorf("get Sandbox CR %s: %w", name, err)
	}
	return string(created.UID), nil
}

// DeleteSandboxCR deletes the Sandbox CR, tolerating NotFound.
func (k *K8s) DeleteSandboxCR(name string) error {
	sb := &apiv1alpha1.Sandbox{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: k.Cfg.Namespace}}
	if err := k.Client.Delete(context.Background(), sb); err != nil && !apierrors.IsNotFound(err) {
		return fmt.Errorf("delete Sandbox CR %s: %w", name, err)
	}
	return nil
}

// SandboxCRExists reports whether the Sandbox CR currently exists.
func (k *K8s) SandboxCRExists(name string) bool {
	sb := &apiv1alpha1.Sandbox{}
	err := k.Client.Get(context.Background(), types.NamespacedName{Name: name, Namespace: k.Cfg.Namespace}, sb)
	return err == nil
}

// AgentPodInfo identifies a ready agent pod.
type AgentPodInfo struct {
	Name string
	UID  string
	IP   string
	Node string
}

// WaitForAgentPod waits until an agent pod is Running with its agent
// container Ready and returns its identity.
func (k *K8s) WaitForAgentPod() (*AgentPodInfo, error) {
	return k.WaitForReadyPod(AgentPodLabel)
}

// WaitForReadyPod waits until a pod matching labelSelector is Running with a
// ready "agent" container and returns its identity.
func (k *K8s) WaitForReadyPod(labelSelector string) (*AgentPodInfo, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()

	for {
		if info := k.findReadyPod(ctx, labelSelector); info != nil {
			return info, nil
		}
		if ctx.Err() != nil {
			return nil, fmt.Errorf("timeout waiting for ready pod (label %s)", labelSelector)
		}
		time.Sleep(pollInterval)
	}
}

// findReadyPod lists pods matching labelSelector and returns the first ready
// one, or nil while none is ready yet (transient listing errors are retried).
func (k *K8s) findReadyPod(ctx context.Context, labelSelector string) *AgentPodInfo {
	pods, err := k.Kube.CoreV1().Pods(k.Cfg.Namespace).List(ctx, metav1.ListOptions{LabelSelector: labelSelector})
	if err != nil {
		return nil
	}
	for i := range pods.Items {
		if info := readyAgentPodInfo(&pods.Items[i]); info != nil {
			return info
		}
	}
	return nil
}

// readyAgentPodInfo returns the pod identity when it is Running with a ready
// agent container, nil otherwise.
func readyAgentPodInfo(pod *corev1.Pod) *AgentPodInfo {
	if pod.Status.Phase != corev1.PodRunning {
		return nil
	}
	for _, cs := range pod.Status.ContainerStatuses {
		if cs.Name == "agent" && cs.Ready {
			return &AgentPodInfo{
				Name: pod.Name,
				UID:  string(pod.UID),
				IP:   pod.Status.PodIP,
				Node: pod.Spec.NodeName,
			}
		}
	}
	return nil
}

// DeleteAgentPod deletes the given agent pod (gracefully).
func (k *K8s) DeleteAgentPod(name string) error {
	return k.Kube.CoreV1().Pods(k.Cfg.Namespace).Delete(context.Background(), name, metav1.DeleteOptions{})
}

// CreateAgentPod creates the fixed-name agent pod. It is used by Setup
// deployment and by RestartAgentPod to recreate the pod under the same name
// (keeping the name stable is what the agent's name-based reconcile relies
// on). AlreadyExists is tolerated.
func (k *K8s) CreateAgentPod() error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := k.Client.Create(ctx, buildAgentPod(k.Cfg)); err != nil && !apierrors.IsAlreadyExists(err) {
		return fmt.Errorf("create agent pod %s: %w", AgentPodName, err)
	}
	return nil
}

// GetPod fetches a pod in the e2e namespace.
func (k *K8s) GetPod(name string) (*corev1.Pod, error) {
	return k.Kube.CoreV1().Pods(k.Cfg.Namespace).Get(context.Background(), name, metav1.GetOptions{})
}

// NodeIP returns the InternalIP of the given node.
func (k *K8s) NodeIP(nodeName string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	node, err := k.Kube.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	for _, addr := range node.Status.Addresses {
		if addr.Type == corev1.NodeInternalIP {
			return addr.Address, nil
		}
	}
	return "", fmt.Errorf("node %s has no InternalIP", nodeName)
}
