package utils

import (
	"context"
	"fmt"
	"strings"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
	podFunc "gitcode.com/openFuyao/e2e-auto-test/e2e/framework/pod"
	// loggingConfig "gitcode.com/openFuyao/e2e-auto-test/e2e/logging/system-integration/logging-config"
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

// logging installation check
func LoggingInstallationCheck(k8sClient *k8s.K8SClient, ctx context.Context) (bool, error) {

	loggingPodMap := map[string]int{
		"logging-package": 0,
		"loki":            0,
		"promtail":        0,
	}

	pods, err := k8sClient.ListPods(ctx, "", metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("get pod failed: %v", err)
	}
	if len(pods.Items) == 0 {
		return false, nil
	}

	for _, pod := range pods.Items {
		if strings.Contains(pod.Name, "logging-package") {
			loggingPodMap["logging-package"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, nil
			}
			// imageStatus := podFunc.ImageCheck(pod, []string{loggingConfig.ImageLoggingWebsite, loggingConfig.ImageOauthProxy})
			// if !imageStatus {
			// 	return false, nil
			// }
		}
	}

	lokiPods, err := k8sClient.ListPods(ctx, "loki", metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("get loki pod failed : %v", err)
	}
	if len(pods.Items) == 0 {
		return false, nil
	}

	for _, pod := range lokiPods.Items {
		if strings.Contains(pod.Name, "loki") {
			loggingPodMap["loki"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, nil
			}
		}
		if strings.Contains(pod.Name, "promtail") {
			loggingPodMap["promtail"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, nil
			}
		}
	}

	for _, v := range loggingPodMap {
		if v == 0 {
			return false, nil
		}
	}

	return true, nil
}
