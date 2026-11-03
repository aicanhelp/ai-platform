package utils

import (
	"context"
	"fmt"
	"strings"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
	podFunc "gitcode.com/openFuyao/e2e-auto-test/e2e/framework/pod"
	// infernexConfig "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/system-integration/infernex-config"
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

// infernex installation check
func InfernexInstallationCheck(k8sClient *k8s.K8SClient, ctx context.Context) (bool, error) {

	infernexPodMap := map[string]int{
		"cache-indexer":                   0,
		"epp":                             0,
		"mooncake":                        0,
		"redis":                           0,
		"proxy":                           0,
		"prefill":                         0,
		"decode":                          0,
		"gateway":                         0,
		"istiod":                          0,
		"hardware-diagnosis":              0,
		"hardware-monitor":                0,
		"network-performance-exporter":    0,
		"nats":                            0,
		"elastic-scaler-controller":       0,
		"resourcescalinggroup-controller": 0,
		"tidal-controller":                0,
	}

	pods, err := k8sClient.ListPods(ctx, "ai-inference", metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("get ai-inference pod failed: %v", err)
	}
	if len(pods.Items) == 0 {
		return false, fmt.Errorf("no pods in namespace ai-inference")
	}

	for _, pod := range pods.Items {
		if strings.Contains(pod.Name, "cache-indexer") {
			infernexPodMap["cache-indexer"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, fmt.Errorf("cache-indexer pod not ready")
			}
			// imageStatus := podFunc.ImageCheck(pod, []string{infernexConfig.ImageCacheIndexer})
			// if !imageStatus {
			// 	return false, fmt.Errorf("cache-indexer got wrong image")
			// }
		}
		if strings.Contains(pod.Name, "epp") {
			infernexPodMap["epp"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, fmt.Errorf("epp pod not ready")
			}
			// imageStatus := podFunc.ImageCheck(pod, []string{infernexConfig.ImageRouter})
			// if !imageStatus {
			// 	return false, fmt.Errorf("epp pod got wrong image")
			// }
		}
		if strings.Contains(pod.Name, "decode") {
			infernexPodMap["decode"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, fmt.Errorf("decode pod not ready")
			}
		}
		if strings.Contains(pod.Name, "prefill") {
			infernexPodMap["prefill"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, fmt.Errorf("prefill pod not ready")
			}
		}
		if strings.Contains(pod.Name, "proxy") {
			infernexPodMap["proxy"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, fmt.Errorf("proxy pod not ready")
			}
			// imageStatus := podFunc.ImageCheck(pod, []string{infernexConfig.ImageProxyServer})
			// if !imageStatus {
			// 	return false, fmt.Errorf("proxy pod got wrong image")
			// }
		}
		if strings.Contains(pod.Name, "mooncake") {
			infernexPodMap["mooncake"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, fmt.Errorf("mooncake pod not ready")
			}
		}
		if strings.Contains(pod.Name, "redis") {
			infernexPodMap["redis"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, fmt.Errorf("redis pod not ready")
			}
		}
		if strings.Contains(pod.Name, "gateway") {
			infernexPodMap["gateway"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, fmt.Errorf("gateway pod not ready")
			}
		}
	}

	istioPods, err := k8sClient.ListPods(ctx, "istio-system", metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("get istio pod failed : %v", err)
	}
	if len(istioPods.Items) == 0 {
		return false, fmt.Errorf("no pods in namespace istio-system")
	}

	for _, pod := range istioPods.Items {
		if strings.Contains(pod.Name, "istiod") {
			infernexPodMap["istiod"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, fmt.Errorf("istiod pod not ready")
			}
		}
	}

	eagleEyePods, err := k8sClient.ListPods(ctx, "eagle-eye", metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("get eagle-eye pod failed : %v", err)
	}
	if len(eagleEyePods.Items) == 0 {
		return false, fmt.Errorf("no pods in namespace eagle-eye")
	}

	for _, pod := range eagleEyePods.Items {
		if strings.Contains(pod.Name, "hardware-diagnosis") {
			infernexPodMap["hardware-diagnosis"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, fmt.Errorf("hardware-diagnosis pod not ready")
			}
			// imageStatus := podFunc.ImageCheck(pod, []string{infernexConfig.ImageEagleEyeHardwareDiagnosis})
			// if !imageStatus {
			// 	return false, fmt.Errorf("hardware-diagnosis pod got wrong image")
			// }
		}
		if strings.Contains(pod.Name, "hardware-monitor") {
			infernexPodMap["hardware-monitor"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, fmt.Errorf("hardware-monitor pod not ready")
			}
			// imageStatus := podFunc.ImageCheck(pod, []string{infernexConfig.ImageEagleEyeHardwareMonitor})
			// if !imageStatus {
			// 	return false, fmt.Errorf("hardware-monitor pod got wrong image")
			// }
		}
		if strings.Contains(pod.Name, "network-performance-exporter") {
			infernexPodMap["network-performance-exporter"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, fmt.Errorf("network-performance-exporter pod not ready")
			}
			// imageStatus := podFunc.ImageCheck(pod, []string{infernexConfig.ImageNpe})
			// if !imageStatus {
			// 	return false, fmt.Errorf("network-performance-exporter pod got wrong image")
			// }
		}
	}

	natsPods, err := k8sClient.ListPods(ctx, "nats", metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("get nats pod failed : %v", err)
	}
	if len(natsPods.Items) == 0 {
		return false, fmt.Errorf("no pods in namespace nats")
	}

	for _, pod := range natsPods.Items {
		if strings.Contains(pod.Name, "nats") {
			infernexPodMap["nats"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, fmt.Errorf("nats pod not ready")
			}
		}
	}

	scalerPods, err := k8sClient.ListPods(ctx, "scaling-system", metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("get scaler pod failed : %v", err)
	}
	if len(scalerPods.Items) == 0 {
		return false, fmt.Errorf("no pods in namespace scaling-system")
	}

	for _, pod := range scalerPods.Items {
		if strings.Contains(pod.Name, "elastic-scaler-controller") {
			infernexPodMap["elastic-scaler-controller"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, fmt.Errorf("elastic-scaler-controller pod not ready")
			}
			// imageStatus := podFunc.ImageCheck(pod, []string{infernexConfig.ImageElasticScaler})
			// if !imageStatus {
			// 	return false, fmt.Errorf("elastic-scaler-controller pod got wrong image")
			// }
		}
		if strings.Contains(pod.Name, "resourcescalinggroup-controller") {
			infernexPodMap["resourcescalinggroup-controller"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, fmt.Errorf("resourcescalinggroup-controller pod not ready")
			}
			// imageStatus := podFunc.ImageCheck(pod, []string{infernexConfig.ImageRsg})
			// if !imageStatus {
			// 	return false, fmt.Errorf("resourcescalinggroup-controller pod got wrong image")
			// }
		}
		if strings.Contains(pod.Name, "tidal-controller") {
			infernexPodMap["tidal-controller"] += 1
			podStatus := podFunc.VerifyPodStatus(pod)
			if !podStatus {
				return false, fmt.Errorf("tidal-controller pod not ready")
			}
			// imageStatus := podFunc.ImageCheck(pod, []string{infernexConfig.ImageTidal})
			// if !imageStatus {
			// 	return false, fmt.Errorf("tidal-controller pod got wrong image")
			// }
		}
	}

	for k, v := range infernexPodMap {
		if v == 0 {
			return false, fmt.Errorf("%s pod does not been created", k)
		}
	}

	return true, nil
}

// infernex anomaly check without npu or npu-operator
func InfernexAnomalyCheckWithoutNPU(k8sClient *k8s.K8SClient, ctx context.Context, namespace string) (bool, error) {
	pods, err := k8sClient.ListPods(ctx, namespace, metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("get pod failed : %v", err)
	}
	for _, pod := range pods.Items {
		// vllm
		if strings.Contains(pod.Name, "prefill") && pod.Status.Phase == "Pending" {
			if strings.Contains(pod.Status.Conditions[0].Message, "Insufficient") {
				return true, nil
			}
		}
	}
	return false, nil
}
