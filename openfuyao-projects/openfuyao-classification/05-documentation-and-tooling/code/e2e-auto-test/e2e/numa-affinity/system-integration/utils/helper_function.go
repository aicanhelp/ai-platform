package utils

import (
	"context"
	"fmt"
	"strings"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
	podFunc "gitcode.com/openFuyao/e2e-auto-test/e2e/framework/pod"
	// numaAffinityConfig "gitcode.com/openFuyao/e2e-auto-test/e2e/numa-affinity/system-integration/numa-affinity-config"
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

// numa affinity installation check
func NumaAffinityInstallationCheck(k8sClient *k8s.K8SClient, ctx context.Context) (bool, error) {

	numaPodMap := map[string]int{
		"numa-affinity-webhook":       0,
		"numa-exporter":               0,
		"numaadj":                     0,
		"scheduler-config":            0,
		"resource-exporter-daemonset": 0,
		"volcano-admission":           0,
		"volcano-controllers":         0,
		"volcano-config-website":      0,
		"volcano-scheduler":           0,
	}

	pods, err := k8sClient.ListPods(ctx, "volcano-system", metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("get pod failed : %v", err)
	}
	if len(pods.Items) == 0 {
		return false, nil
	}

	for _, pod := range pods.Items {
		if strings.Contains(pod.Name, "numa-affinity-webhook") {
			numaPodMap["numa-affinity-webhook"] += 1
			podCheck := podFunc.VerifyPodStatus(pod)
			if podCheck == false {
				return false, nil
			}
			// imageCheck := podFunc.ImageCheck(pod, []string{numaAffinityConfig.NumaAffinityWebhookImage})
			// if imageCheck == false {
			// 	return false, fmt.Errorf("numa-affinity-webhook pod got wrong image !")
			// }
		}
		if strings.Contains(pod.Name, "numa-exporter") {
			numaPodMap["numa-exporter"] += 1
			podCheck := podFunc.VerifyPodStatus(pod)
			if podCheck == false {
				return false, nil
			}
			// imageCheck := podFunc.ImageCheck(pod, []string{numaAffinityConfig.VolcanoConfigServiceImage})
			// if imageCheck == false {
			// 	return false, fmt.Errorf("numa-exporter pod got wrong image !")
			// }
		}
		if strings.Contains(pod.Name, "numaadj") {
			numaPodMap["numaadj"] += 1
			podCheck := podFunc.VerifyPodStatus(pod)
			if podCheck == false {
				return false, nil
			}
			// imageCheck := podFunc.ImageCheck(pod, []string{numaAffinityConfig.VolcanoConfigServiceImage})
			// if imageCheck == false {
			// 	return false, fmt.Errorf("numaadj pod get wrong image !")
			// }
		}
		if strings.Contains(pod.Name, "scheduler-config") {
			numaPodMap["scheduler-config"] += 1
			podCheck := podFunc.VerifyPodStatus(pod)
			if podCheck == false {
				return false, nil
			}
			// imageCheck := podFunc.ImageCheck(pod, []string{numaAffinityConfig.VolcanoConfigServiceImage})
			// if imageCheck == false {
			// 	return false, fmt.Errorf("scheduler-config pod got wrong image !")
			// }
		}
		if strings.Contains(pod.Name, "resource-exporter-daemonset") {
			numaPodMap["resource-exporter-daemonset"] += 1
			podCheck := podFunc.VerifyPodStatus(pod)
			if podCheck == false {
				return false, nil
			}
		}
		if strings.Contains(pod.Name, "volcano-admission") {
			if strings.Contains(pod.Name, "init") {
				continue
			}
			numaPodMap["volcano-admission"] += 1
			podCheck := podFunc.VerifyPodStatus(pod)
			if podCheck == false {
				return false, nil
			}
		}
		if strings.Contains(pod.Name, "volcano-controllers") {
			numaPodMap["volcano-controllers"] += 1
			podCheck := podFunc.VerifyPodStatus(pod)
			if podCheck == false {
				return false, nil
			}
		}
		if strings.Contains(pod.Name, "volcano-config-website") {
			numaPodMap["volcano-config-website"] += 1
			podCheck := podFunc.VerifyPodStatus(pod)
			if podCheck == false {
				return false, nil
			}
			// imageCheck := podFunc.ImageCheck(pod, []string{numaAffinityConfig.VolcanoConfigWebsiteImage, numaAffinityConfig.OauthProxyImage})
			// if imageCheck == false {
			// 	return false, fmt.Errorf("volcano-config-website pod got wrong images !")
			// }
		}
		if strings.Contains(pod.Name, "volcano-scheduler") {
			numaPodMap["volcano-scheduler"] += 1
			podCheck := podFunc.VerifyPodStatus(pod)
			if podCheck == false {
				return false, nil
			}
		}
	}

	for _, v := range numaPodMap {
		if v == 0 {
			return false, nil
		}
	}

	return true, nil
}

// pre-operation of volcano-installation
func PreOpBeforeVolcanoInstallation(sshClient *executor.SSHExecutor) error {
	_, err := sshClient.Exec("helm repo add volcano-sh https://volcano-sh.github.io/helm-charts && helm repo update")
	if err != nil {
		return err
	}
	return nil
}

func PreOpBeforeVolcanoInstallationLocal(localexecutor *executor.LocalExecutor) error {
	_, err := localexecutor.Exec("helm repo add volcano-sh https://volcano-sh.github.io/helm-charts && helm repo update")
	if err != nil {
		return err
	}
	return nil
}
