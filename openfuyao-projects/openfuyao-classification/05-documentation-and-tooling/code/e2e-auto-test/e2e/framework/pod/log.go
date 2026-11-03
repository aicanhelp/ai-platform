// package pod offer pod related functions
package pod

import (
	"context"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

// LogResult stores the log results
type LogResult struct {
	Namespace     string
	PodName       string
	ContainerName string
	Log           string
}

// GetLogsByPod retrieves logs from a specific pod and container
func GetLogsByPod(ctx context.Context, client kubernetes.Interface, namespace, podName, containerName string) (*LogResult, error) {
	if client == nil {
		return nil, fmt.Errorf("kubernetes client is nil")
	}
	if podName == "" {
		return nil, fmt.Errorf("pod name is required")
	}
	if containerName == "" {
		return nil, fmt.Errorf("container name is required")
	}

	pod, err := client.CoreV1().Pods(namespace).Get(ctx, podName, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to get pod %s/%s: %w", namespace, podName, err)
	}

	containerName, err = checkontainerByName(pod, containerName)
	if err != nil {
		return nil, err
	}

	res, err := collectPodLog(ctx, client, pod, containerName)
	if err != nil {
		return nil, err
	}

	return &res, nil
}

// GetLogsByPodLabel retrieves logs from all pods matching a label selector
func GetLogsByPodLabel(ctx context.Context, client kubernetes.Interface, namespace, labelSelector, containerName string) ([]LogResult, error) {
	if client == nil {
		return nil, fmt.Errorf("kubernetes client is nil")
	}
	if containerName == "" {
		return nil, fmt.Errorf("container name is required")
	}

	pods, err := client.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{LabelSelector: labelSelector})
	if err != nil {
		return nil, fmt.Errorf("list pods in namespace %s: %w", namespace, err)
	}

	if len(pods.Items) == 0 {
		return nil, fmt.Errorf("no matched pod found")
	}

	results := make([]LogResult, 0, len(pods.Items))
	for i := range pods.Items {
		pod := &pods.Items[i]

		logRes, err := collectPodLog(ctx, client, pod, containerName)
		if err != nil {
			return nil, err
		}

		results = append(results, logRes)
	}

	return results, nil
}

// collectPodLog collects logs from a specific pod container
func collectPodLog(ctx context.Context, client kubernetes.Interface, pod *corev1.Pod, containerName string) (LogResult, error) {
	cName, err := checkontainerByName(pod, containerName)
	if err != nil {
		return LogResult{}, err
	}

	req := client.CoreV1().Pods(pod.Namespace).GetLogs(pod.Name, &corev1.PodLogOptions{
		Container: cName,
	})

	raw, err := req.Do(ctx).Raw()
	if err != nil {
		return LogResult{}, fmt.Errorf("get log for pod %s/%s: %w", pod.Namespace, pod.Name, err)
	}

	return LogResult{
		ContainerName: containerName,
		PodName:       pod.Name,
		Namespace:     pod.Namespace,
		Log:           string(raw),
	}, nil
}

// checkontainerByName validates and returns the container name for log collection
func checkontainerByName(pod *corev1.Pod, containerName string) (string, error) {
	if pod == nil {
		return "", fmt.Errorf("pod is nil")
	}
	if len(pod.Spec.Containers) == 0 {
		return "", fmt.Errorf("pod %s/%s has no containers", pod.Namespace, pod.Name)
	}
	if containerName == "" {
		return "", fmt.Errorf("container name is not provided")
	}

	for _, c := range pod.Spec.Containers {
		if c.Name == containerName {
			return c.Name, nil
		}
	}

	return "", fmt.Errorf("container %s not found in pod %s/%s", containerName, pod.Namespace, pod.Name)
}
