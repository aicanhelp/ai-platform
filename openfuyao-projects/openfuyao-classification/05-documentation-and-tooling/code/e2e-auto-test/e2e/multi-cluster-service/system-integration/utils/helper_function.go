package utils

import (
	"context"
	"fmt"
	"strings"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
	podFunc "gitcode.com/openFuyao/e2e-auto-test/e2e/framework/pod"
	// MultiClusterConfig "gitcode.com/openFuyao/e2e-auto-test/e2e/multi-cluster-service/system-integration/multi-cluster-config"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// system reboot
func SystemReboot(sshClient *executor.SSHExecutor, reboot_time int) error {
	cmd := fmt.Sprintf("shutdown -r +%d", reboot_time)
	_, err := sshClient.Exec(cmd)
	if err != nil {
		return fmt.Errorf("reboot command execute failed : %v", err)
	}
	return nil
}

// multi-cluster-service installation check
func MCSInstallationCheck(k8sClient *k8s.K8SClient, ctx context.Context) (bool, error) {

	mcsPodMap := map[string]int{
		"etcd":                            0,
		"karmada-aggregated-apiserver":    0,
		"karmada-apiserver":               0,
		"karmada-controller-manager":      0,
		"karmada-kube-controller-manager": 0,
		"karmada-scheduler":               0,
		"karmada-webhook":                 0,
		"multi-cluster-website":           0,
		"multicluster-service":            0,
	}

	pods, err := k8sClient.ListPods(ctx, "karmada-system", metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("get pod failed: %v", err)
	}
	if len(pods.Items) == 0 {
		return false, fmt.Errorf("no pods in namespace karmada-system")
	}

	for _, pod := range pods.Items {
		if strings.Contains(pod.Name, "etcd") {
			mcsPodMap["etcd"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, fmt.Errorf("etcd pod not ready")
			}
		}
		if strings.Contains(pod.Name, "karmada-aggregated-apiserver") {
			mcsPodMap["karmada-aggregated-apiserver"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, fmt.Errorf("karmada-aggregated-apiserver pod not ready")
			}
		}
		if strings.Contains(pod.Name, "karmada-apiserver") {
			mcsPodMap["karmada-apiserver"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, fmt.Errorf("karmada-apiserver pod not ready")
			}
		}
		if strings.Contains(pod.Name, "karmada-controller-manager") {
			mcsPodMap["karmada-controller-manager"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, fmt.Errorf("karmada-controller-manager pod not ready")
			}
		}
		if strings.Contains(pod.Name, "karmada-kube-controller-manager") {
			mcsPodMap["karmada-kube-controller-manager"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, fmt.Errorf("karmada-kube-controller-manager pod not ready")
			}
		}
		if strings.Contains(pod.Name, "karmada-scheduler") {
			mcsPodMap["karmada-scheduler"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, fmt.Errorf("karmada-scheduler pod not ready")
			}
		}
		if strings.Contains(pod.Name, "karmada-webhook") {
			mcsPodMap["karmada-webhook"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, fmt.Errorf("karmada-webhook pod not ready")
			}
		}
		if strings.Contains(pod.Name, "multi-cluster-website") {
			mcsPodMap["multi-cluster-website"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, fmt.Errorf("multi-cluster-website pod not ready")
			}
			// imageStatus := podFunc.ImageCheck(pod, []string{MultiClusterConfig.ImageMultiClusterWebsite})
			// if !imageStatus {
			// 	return false, fmt.Errorf("multi-cluster-website pod got wrong image")
			// }
		}
		if strings.Contains(pod.Name, "multicluster-service") {
			mcsPodMap["multicluster-service"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, fmt.Errorf("multicluster-service pod not ready")
			}
			// imageStatus := podFunc.ImageCheck(pod, []string{MultiClusterConfig.ImageMultiClusterService})
			// if !imageStatus {
			// 	return false, fmt.Errorf("multicluster-service pod got wrong image")
			// }
		}
	}

	for k, v := range mcsPodMap {
		if v == 0 {
			return false, fmt.Errorf("%s pod does not been created", k)
		}
	}

	return true, nil
}
