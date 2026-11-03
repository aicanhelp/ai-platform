package utils

import (
	"context"
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// CheckSocketExists verifies that a Unix domain socket file exists inside an EPP pod
// by running kubectl exec with test -S.
func CheckSocketExists(k8sClient K8SClientInterface, ctx context.Context, namespace string, socketPath string) (bool, error) {
	pods, err := k8sClient.ListPods(ctx, namespace, metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("list pods failed: %v", err)
	}

	foundEPPPod := false
	for i := range pods.Items {
		if !strings.Contains(pods.Items[i].Name, "epp") {
			continue
		}
		// 跳过正在删除（Terminating）的旧 release 残留 pod，避免对残留实例做 kubectl exec。
		if pods.Items[i].DeletionTimestamp != nil {
			continue
		}
		if pods.Items[i].Status.Phase != corev1.PodRunning {
			continue
		}
		foundEPPPod = true

		for _, containerName := range socketCheckContainerOrder(&pods.Items[i]) {
			cmd := fmt.Sprintf("kubectl exec %s -n %s -c %s -- test -S %s", pods.Items[i].Name, namespace, containerName, socketPath)
			_, _, err = execLocal(cmd)
			if err == nil {
				return true, nil
			}
		}
	}
	if !foundEPPPod {
		return false, fmt.Errorf("no epp pod found in namespace %s", namespace)
	}

	return false, nil
}

func socketCheckContainerOrder(pod *corev1.Pod) []string {
	containerNames := make([]string, 0, len(pod.Spec.Containers))
	seen := make(map[string]struct{}, len(pod.Spec.Containers))

	for _, container := range pod.Spec.Containers {
		if container.Name == "epp" {
			containerNames = append(containerNames, container.Name)
			seen[container.Name] = struct{}{}
			break
		}
	}

	for _, container := range pod.Spec.Containers {
		if _, ok := seen[container.Name]; ok {
			continue
		}
		containerNames = append(containerNames, container.Name)
	}

	return containerNames
}
