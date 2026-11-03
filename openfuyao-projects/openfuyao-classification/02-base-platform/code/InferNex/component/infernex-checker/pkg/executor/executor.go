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

// Package executor provides two node access clients: SSHClient for executing
// commands on remote nodes over SSH, and K8sClient for querying cluster state
// via the Kubernetes API.
package executor

import (
	"bytes"
	"context"
	"fmt"
	"net"
	"os"
	"strconv"
	"time"

	"golang.org/x/crypto/ssh"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"

	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/types"
)

const sshTimeout = 30 * time.Second

// CommandRunner is an interface for executing shell commands on a remote node.
// SSHClient implements this interface; tests can inject a fake implementation.
type CommandRunner interface {
	Run(cmd string) (stdout, stderr string, err error)
	NodeName() string
}

// SSHClient wraps a single-node SSH connection
type SSHClient struct {
	node   types.NodeInfo
	client *ssh.Client
}

// NewSSHClient establishes an SSH connection
func NewSSHClient(node types.NodeInfo) (*SSHClient, error) {
	var authMethods []ssh.AuthMethod

	if node.KeyFile != "" {
		key, err := os.ReadFile(node.KeyFile)
		if err != nil {
			return nil, fmt.Errorf("failed to read key file: %w", err)
		}
		signer, err := ssh.ParsePrivateKey(key)
		if err != nil {
			return nil, fmt.Errorf("failed to parse private key: %w", err)
		}
		authMethods = append(authMethods, ssh.PublicKeys(signer))
	}

	if node.Password != "" {
		authMethods = append(authMethods, ssh.Password(node.Password))
	}

	cfg := &ssh.ClientConfig{
		User:            node.User,
		Auth:            authMethods,
		HostKeyCallback: ssh.InsecureIgnoreHostKey(), // internal network, skip host key verification
		Timeout:         sshTimeout,
	}

	addr := net.JoinHostPort(node.IP, strconv.Itoa(node.Port))
	conn, err := net.DialTimeout("tcp", addr, sshTimeout)
	if err != nil {
		return nil, fmt.Errorf("TCP connection failed %s: %w", addr, err)
	}

	sshConn, chans, reqs, err := ssh.NewClientConn(conn, addr, cfg)
	if err != nil {
		if closeErr := conn.Close(); closeErr != nil {
			return nil, fmt.Errorf("SSH handshake failed %s: %w (also failed to close connection: %v)",
				addr, err, closeErr)
		}
		return nil, fmt.Errorf("SSH handshake failed %s: %w", addr, err)
	}

	return &SSHClient{
		node:   node,
		client: ssh.NewClient(sshConn, chans, reqs),
	}, nil
}

// Run executes a command on the remote node and returns stdout and stderr
func (c *SSHClient) Run(cmd string) (stdout, stderr string, err error) {
	session, err := c.client.NewSession()
	if err != nil {
		return "", "", fmt.Errorf("failed to create SSH session: %w", err)
	}
	defer session.Close()

	var outBuf, errBuf bytes.Buffer
	session.Stdout = &outBuf
	session.Stderr = &errBuf

	runErr := session.Run(cmd)
	return outBuf.String(), errBuf.String(), runErr
}

// Close closes the SSH connection
func (c *SSHClient) Close() {
	if c.client != nil {
		c.client.Close()
	}
}

// NodeName returns the node name
func (c *SSHClient) NodeName() string {
	return c.node.Name
}

// K8sClient wraps a Kubernetes API client
type K8sClient struct {
	client kubernetes.Interface
}

// NewSSHRunner is the standard factory for creating a CommandRunner from a NodeInfo.
// Both configenv and hardware packages expose a package-level var of this type so tests
// can override it to inject a fake runner without a real SSH connection.
func NewSSHRunner(node types.NodeInfo) (CommandRunner, func(), error) {
	client, err := NewSSHClient(node)
	if err != nil {
		return nil, func() { /* no connection established, nothing to clean up */ }, err
	}
	if client == nil {
		return nil, func() { /* no connection established, nothing to clean up */ },
			fmt.Errorf("newSSHClient returned nil client")
	}
	return client, func() { client.Close() }, nil
}

// NewK8sClientForTest creates a K8sClient from an existing clientset, intended for unit tests only.
func NewK8sClientForTest(clientset kubernetes.Interface) *K8sClient {
	return &K8sClient{client: clientset}
}

// NewK8sClient creates a K8s client from kubeconfig
func NewK8sClient(kubeconfig string) (*K8sClient, error) {
	config, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
	if err != nil {
		return nil, fmt.Errorf("failed to load kubeconfig: %w", err)
	}

	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create K8s client: %w", err)
	}

	return &K8sClient{client: clientset}, nil
}

// GetPodsInNamespace returns pods matching the label selector in the given namespace
func (k *K8sClient) GetPodsInNamespace(ctx context.Context, namespace, labelSelector string) ([]corev1.Pod, error) {
	list, err := k.client.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get pod list: %w", err)
	}
	return list.Items, nil
}

// GetPodsInAllNamespaces returns pods matching the label selector across all namespaces
func (k *K8sClient) GetPodsInAllNamespaces(ctx context.Context, labelSelector string) ([]corev1.Pod, error) {
	list, err := k.client.CoreV1().Pods("").List(ctx, metav1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get pod list: %w", err)
	}
	return list.Items, nil
}

// GetAllPodsOnNode returns all non-terminated pods on the specified node
func (k *K8sClient) GetAllPodsOnNode(ctx context.Context, nodeName string) ([]corev1.Pod, error) {
	list, err := k.client.CoreV1().Pods("").List(ctx, metav1.ListOptions{
		FieldSelector: fmt.Sprintf("spec.nodeName=%s", nodeName),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get pod list on node %s: %w", nodeName, err)
	}

	var result []corev1.Pod
	for _, pod := range list.Items {
		if pod.Status.Phase != corev1.PodSucceeded && pod.Status.Phase != corev1.PodFailed {
			result = append(result, pod)
		}
	}
	return result, nil
}

// GetNode returns the specified node object
func (k *K8sClient) GetNode(ctx context.Context, nodeName string) (*corev1.Node, error) {
	node, err := k.client.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to get node %s: %w", nodeName, err)
	}
	return node, nil
}

// GetNodes returns all nodes
func (k *K8sClient) GetNodes(ctx context.Context) ([]corev1.Node, error) {
	list, err := k.client.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to get node list: %w", err)
	}
	return list.Items, nil
}

// CreatePod creates a temporary pod
func (k *K8sClient) CreatePod(ctx context.Context, pod *corev1.Pod) (*corev1.Pod, error) {
	return k.client.CoreV1().Pods(pod.Namespace).Create(ctx, pod, metav1.CreateOptions{})
}

// DeletePod deletes a pod
func (k *K8sClient) DeletePod(ctx context.Context, namespace, name string) error {
	return k.client.CoreV1().Pods(namespace).Delete(ctx, name, metav1.DeleteOptions{})
}

// GetPod returns a single pod
func (k *K8sClient) GetPod(ctx context.Context, namespace, name string) (*corev1.Pod, error) {
	return k.client.CoreV1().Pods(namespace).Get(ctx, name, metav1.GetOptions{})
}

// Clientset returns the raw clientset for advanced use cases
func (k *K8sClient) Clientset() kubernetes.Interface {
	return k.client
}
