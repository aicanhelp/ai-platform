package utils

import (
	"context"
	"fmt"
	"strings"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
	podFunc "gitcode.com/openFuyao/e2e-auto-test/e2e/framework/pod"
	// rayConfig "gitcode.com/openFuyao/e2e-auto-test/e2e/ray/system-integration/ray-config"
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

// ray installation check
func RayInstallationCheck(k8sClient *k8s.K8SClient, ctx context.Context) (bool, error) {

	rayPodMap := map[string]int{
		"kuberay-operator":     0,
		"ray-package-grafana":  0,
		"ray-package-promtail": 0,
		"ray-website":          0,
		"ray-loki":             0,
		"openfuyao-rayservice": 0,
	}

	pods, err := k8sClient.ListPods(ctx, "", metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("get pod failed: %v", err)
	}
	if len(pods.Items) == 0 {
		return false, fmt.Errorf("no pods in namespace default")
	}

	for _, pod := range pods.Items {
		if strings.Contains(pod.Name, "kuberay-operator") {
			rayPodMap["kuberay-operator"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, fmt.Errorf("kuberay-operator pod not ready")
			}
		}
		if strings.Contains(pod.Name, "ray-package-grafana") {
			rayPodMap["ray-package-grafana"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, fmt.Errorf("ray-package-grafana pod not ready")
			}
			// imageStatus := podFunc.ImageCheck(pod, []string{rayConfig.ImageGrafana})
			// if !imageStatus {
			// 	return false, fmt.Errorf("ray-package-grafana pod got wrong image")
			// }
		}
		if strings.Contains(pod.Name, "ray-package-promtail") {
			rayPodMap["ray-package-promtail"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, fmt.Errorf("ray-package-promtail pod not ready")
			}
			// imageStatus := podFunc.ImageCheck(pod, []string{rayConfig.ImagePromtail})
			// if !imageStatus {
			// 	return false, fmt.Errorf("ray-package-promtail pod get wrong image")
			// }
		}
		if strings.Contains(pod.Name, "ray-website") {
			rayPodMap["ray-website"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, fmt.Errorf("ray-website pod not ready")
			}
			// imageStatus := podFunc.ImageCheck(pod, []string{rayConfig.ImageOauthProxy, rayConfig.ImageRayWebsite})
			// if !imageStatus {
			// 	return false, fmt.Errorf("karmada-controller-manager pod got wrong image")
			// }
		}
	}

	montoringPods, err := k8sClient.ListPods(ctx, "monitoring", metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("get namespace monitoring pod failed : %v", err)
	}
	if len(montoringPods.Items) == 0 {
		return false, fmt.Errorf("no pods in namespace monitoring")
	}

	for _, pod := range pods.Items {
		if strings.Contains(pod.Name, "ray-loki") {
			rayPodMap["ray-loki"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, fmt.Errorf("ray-loki pod not ready")
			}
		}
	}

	vcjobPods, err := k8sClient.ListPods(ctx, "vcjob", metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("get namespace vcjob pod failed : %v", err)
	}
	if len(vcjobPods.Items) == 0 {
		return false, fmt.Errorf("no pods in namespace default")
	}

	for _, pod := range pods.Items {
		if strings.Contains(pod.Name, "openfuyao-rayservice") {
			rayPodMap["openfuyao-rayservice"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, fmt.Errorf("openfuyao-rayservice pod not ready")
			}
			// imageStatus := podFunc.ImageCheck(pod, []string{rayConfig.ImageRayService})
			// if !imageStatus {
			// 	return false, fmt.Errorf("openfuyao-rayservice pod got wrong images")
			// }
		}
	}

	for k, v := range rayPodMap {
		if v == 0 {
			return false, fmt.Errorf("%s pod does not been created", k)
		}
	}

	return true, nil
}
