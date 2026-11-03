package utils

import (
	"context"
	"fmt"
	"strings"

	// colocationConfig "gitcode.com/openFuyao/e2e-auto-test/e2e/colocation/system-integration/colocation-config"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
	podFunc "gitcode.com/openFuyao/e2e-auto-test/e2e/framework/pod"
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

// colocation installation check
func ColocationInstallationCheck(k8sClient *k8s.K8SClient, ctx context.Context) (bool, error) {

	colocationPodMap := map[string]int{
		"colocation-manager":            0,
		"node-feature-discovery-gc":     0,
		"node-feature-discovery-master": 0,
		"node-feature-discovery-worker": 0,
		"colocation-service":            0,
		"colocation-website":            0,
	}

	pods, err := k8sClient.ListPods(ctx, "openfuyao-colocation", metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("get pod failed : %v", err)
	}
	if len(pods.Items) == 0 {
		return false, nil
	}

	for _, pod := range pods.Items {
		if strings.Contains(pod.Name, "colocation-manager") {
			colocationPodMap["colocation-manager"] += 1
			podCheck := podFunc.VerifyPodStatus(pod)
			if podCheck == false {
				return false, nil
			}
			// imageCheck := podFunc.ImageCheck(pod, []string{colocationConfig.ColocationManagerImage})
			// if imageCheck == false {
			// 	return false, fmt.Errorf("colocation-manager pod got wrong image !")
			// }
		}
		if strings.Contains(pod.Name, "colocation-service") {
			colocationPodMap["colocation-service"] += 1
			podCheck := podFunc.VerifyPodStatus(pod)
			if podCheck == false {
				return false, nil
			}
			// imageCheck := podFunc.ImageCheck(pod, []string{colocationConfig.ColocationServiceImage})
			// if imageCheck == false {
			// 	return false, fmt.Errorf("colocation-service pod get wrong image !")
			// }
		}
		if strings.Contains(pod.Name, "colocation-website") {
			colocationPodMap["colocation-website"] += 1
			podCheck := podFunc.VerifyPodStatus(pod)
			if podCheck == false {
				return false, nil
			}
			// imageCheck := podFunc.ImageCheck(pod, []string{colocationConfig.ColocationWebsiteImage, colocationConfig.OauthProxyImage})
			// if imageCheck == false {
			// 	return false, fmt.Errorf("colocation-website pod got wrong image !")
			// }
		}
		if strings.Contains(pod.Name, "node-feature-discovery-gc") {
			colocationPodMap["node-feature-discovery-gc"] += 1
			podCheck := podFunc.VerifyPodStatus(pod)
			if podCheck == false {
				return false, nil
			}
		}
		if strings.Contains(pod.Name, "node-feature-discovery-master") {
			colocationPodMap["node-feature-discovery-master"] += 1
			podCheck := podFunc.VerifyPodStatus(pod)
			if podCheck == false {
				return false, nil
			}
		}
		if strings.Contains(pod.Name, "node-feature-discovery-worker") {
			colocationPodMap["node-feature-discovery-worker"] += 1
			podCheck := podFunc.VerifyPodStatus(pod)
			if podCheck == false {
				return false, nil
			}
		}
	}

	for _, v := range colocationPodMap {
		if v == 0 {
			return false, nil
		}
	}

	return true, nil
}
