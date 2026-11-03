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

package k8s

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	discoveryv1 "k8s.io/api/discovery/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
)

type K8SClient struct {
	RestConfig    *rest.Config
	Clientset     *kubernetes.Clientset
	DynamicClient dynamic.Interface
}

func localKubeconfigPath() (string, error) {
	if v := strings.TrimSpace(os.Getenv("KUBECONFIG")); v != "" {
		return v, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("get home dir failed: %w", err)
	}
	return filepath.Join(home, ".kube", "config"), nil
}

// NewK8SClientFromLocalKubeconfig 从本机 kubeconfig 创建 Kubernetes clientset（优先使用 $KUBECONFIG，否则 ~/.kube/config）
func NewK8SClientFromLocalKubeconfig() (*K8SClient, error) {
	cfgPath, err := localKubeconfigPath()
	if err != nil {
		return nil, err
	}
	restCfg, err := clientcmd.BuildConfigFromFlags("", cfgPath)
	if err != nil {
		return nil, fmt.Errorf("failed to build kubeconfig (%s): %w", cfgPath, err)
	}
	// E2E 用例（如 hermes-router）会高频轮询 ListPods/ListServices/ListEndpointSlices
	// （CheckAllContainersReady 每 10s 一次、ReadCompletedJSONL 每 5s 一次），Go client
	// 默认 QPS=5/Burst=10 易被打满导致 "client rate limiter Wait ... context deadline exceeded"。
	// 调高到 QPS=50/Burst=100，消除限流引发的用例超时。
	applyE2EClientRateLimits(restCfg)
	clientset, err := kubernetes.NewForConfig(restCfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create clientset: %w", err)
	}
	dynamicClient, err := dynamic.NewForConfig(restCfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create dynamic client: %w", err)
	}
	return &K8SClient{RestConfig: restCfg, Clientset: clientset, DynamicClient: dynamicClient}, nil
}

// applyE2EClientRateLimits sets a generous QPS/Burst on the rest config to avoid
// client-side rate limiting during high-frequency polling in E2E tests.
func applyE2EClientRateLimits(cfg *rest.Config) {
	if cfg == nil {
		return
	}
	if cfg.QPS <= 0 {
		cfg.QPS = 50
	}
	if cfg.Burst <= 0 {
		cfg.Burst = 100
	}
}

func NewK8SClientViaSSH(sshClient *executor.SSHExecutor) (*K8SClient, error) {
	// 通过 SSH 获取 kubeconfig
	kubeconfigPath, err := localKubeconfigPath()
	if err != nil {
		return nil, fmt.Errorf("failed to get kubeconfig path: %w", err)
	}
	getKubeconfigCmd := fmt.Sprintf("cat %s", kubeconfigPath)

	execResult, err := sshClient.Exec(getKubeconfigCmd)
	if err != nil {
		return nil, fmt.Errorf("failed to get kubeconfig via SSH: %w", err)
	}
	kubeconfig := execResult.Stdout

	// 创建临时的 kubeconfig 文件
	tmpFile, err := os.CreateTemp("", "kubeconfig-*")
	if err != nil {
		return nil, fmt.Errorf("failed to create temp file: %w", err)
	}
	defer os.Remove(tmpFile.Name())

	if _, err := tmpFile.WriteString(kubeconfig); err != nil {
		return nil, fmt.Errorf("failed to write kubeconfig: %w", err)
	}
	tmpFile.Close()

	// 创建 Kubernetes 客户端
	config, err := clientcmd.BuildConfigFromFlags("", tmpFile.Name())
	if err != nil {
		return nil, fmt.Errorf("failed to build config: %w", err)
	}
	applyE2EClientRateLimits(config)

	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create clientset: %w", err)
	}

	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create dynamic client: %w", err)
	}

	return &K8SClient{RestConfig: config, Clientset: clientset, DynamicClient: dynamicClient}, nil
}

func NewK8SDynamicClientViaSSH(sshClient *executor.SSHExecutor) (dynamic.Interface, error) {
	// 通过 SSH 获取 kubeconfig
	kubeconfigPath, err := localKubeconfigPath()
	if err != nil {
		return nil, fmt.Errorf("failed to get kubeconfig path: %w", err)
	}
	getKubeconfigCmd := fmt.Sprintf("cat %s", kubeconfigPath)

	execResult, err := sshClient.Exec(getKubeconfigCmd)
	if err != nil {
		return nil, fmt.Errorf("failed to get kubeconfig via SSH: %w", err)
	}
	kubeconfig := execResult.Stdout

	// 创建临时文件
	tmpFile, err := os.CreateTemp("", "kubeconfig-*")
	if err != nil {
		return nil, fmt.Errorf("failed to create temp file: %w", err)
	}
	defer os.Remove(tmpFile.Name())

	if _, err := tmpFile.WriteString(kubeconfig); err != nil {
		return nil, fmt.Errorf("failed to write kubeconfig: %w", err)
	}
	tmpFile.Close()

	// 创建 dynamic client
	config, err := clientcmd.BuildConfigFromFlags("", tmpFile.Name())
	if err != nil {
		return nil, fmt.Errorf("failed to build config: %w", err)
	}

	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create dynamic client: %w", err)
	}

	return dynamicClient, nil
}

func (c *K8SClient) ListPods(ctx context.Context, namespace string, opts metav1.ListOptions) (*corev1.PodList, error) {
	return c.Clientset.CoreV1().Pods(namespace).List(ctx, opts)
}

func (c *K8SClient) ListServices(ctx context.Context, namespace string, opts metav1.ListOptions) (*corev1.ServiceList, error) {
	return c.Clientset.CoreV1().Services(namespace).List(ctx, opts)
}

func (c *K8SClient) ListEndpointSlices(ctx context.Context, namespace string, opts metav1.ListOptions) (*discoveryv1.EndpointSliceList, error) {
	return c.Clientset.DiscoveryV1().EndpointSlices(namespace).List(ctx, opts)
}

func (c *K8SClient) ListNodes(ctx context.Context, opts metav1.ListOptions) (*corev1.NodeList, error) {
	return c.Clientset.CoreV1().Nodes().List(ctx, opts)
}

func (c *K8SClient) ListServiceAccounts(ctx context.Context, namespace string, opts metav1.ListOptions) (*corev1.ServiceAccountList, error) {
	return c.Clientset.CoreV1().ServiceAccounts(namespace).List(ctx, opts)
}

func (c *K8SClient) DeleteServiceAccount(ctx context.Context, namespace, name string) error {
	return c.Clientset.CoreV1().ServiceAccounts(namespace).Delete(ctx, name, metav1.DeleteOptions{})
}

func (c *K8SClient) ListNamespaces(ctx context.Context, opts metav1.ListOptions) (*corev1.NamespaceList, error) {
	return c.Clientset.CoreV1().Namespaces().List(ctx, opts)
}

func (c *K8SClient) GetNamespace(ctx context.Context, name string) (*corev1.Namespace, error) {
	return c.Clientset.CoreV1().Namespaces().Get(ctx, name, metav1.GetOptions{})
}

func (c *K8SClient) CreateNamespaceIfNotExists(ctx context.Context, namespace string) error {
	_, err := c.Clientset.CoreV1().Namespaces().Get(ctx, namespace, metav1.GetOptions{})
	if err == nil {
		return nil // 命名空间已存在
	}

	ns := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: namespace,
		},
	}

	_, err = c.Clientset.CoreV1().Namespaces().Create(ctx, ns, metav1.CreateOptions{})
	return err
}

func (c *K8SClient) DeleteNamespace(ctx context.Context, namespace string) error {
	_, err := c.Clientset.CoreV1().Namespaces().Get(ctx, namespace, metav1.GetOptions{})
	if errors.IsNotFound(err) {
		return nil // 命名空间不存在
	}
	if err != nil {
		return fmt.Errorf("failed to check namespace: %v", err)
	}

	return c.Clientset.CoreV1().Namespaces().Delete(ctx, namespace, metav1.DeleteOptions{})
}
