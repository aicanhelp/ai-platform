/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package clusterapi

import (
	"context"
	_ "embed"
	"fmt"
	"os"
	"path/filepath"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/rand"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/k8s"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

var (
	//go:embed cluster-api.yaml
	clusterAPI []byte
	//go:embed cluster-api-bke.yaml
	clusterAPIBKE []byte
)

func ensureK8sClient() error {
	if global.K8s != nil {
		return nil
	}
	var err error
	global.K8s, err = k8s.NewKubernetesClient("")
	return err
}

func writeClusterAPITemplates(tmplDir string) error {
	if err := os.MkdirAll(tmplDir, utils.DefaultDirPermission); err != nil {
		return err
	}

	files := map[string][]byte{
		filepath.Join(tmplDir, "cluster-api.yaml"):     clusterAPI,
		filepath.Join(tmplDir, "cluster-api-bke.yaml"): clusterAPIBKE,
	}

	for path, content := range files {
		if err := os.WriteFile(path, content, utils.DefaultFilePermission); err != nil {
			return fmt.Errorf("failed to write %s: %w", path, err)
		}
	}
	return nil
}

func installClusterAPIWithRetry(yamlFile, repo, caBundle string) error {
	for {
		time.Sleep(utils.DefaultMinCheckSeconds * time.Second)
		err := global.K8s.InstallYaml(yamlFile, map[string]string{
			"repo":         repo,
			"capiCaBundle": caBundle,
		}, "")
		if err == nil {
			return nil
		}
		log.Debug("Installation failed. Try again in 5 seconds")
		log.Debugf("err: %v", err)
	}
}

const waitForClusterAPIPodsTimeout = 10 * time.Minute

// WaitForClusterAPIPodsReady waits for all pods in the cluster-system namespace
// to be running and ready (readiness probe passed). It should be called after
// k3s restart to ensure CAPI and BKE controller pods are fully operational.
func WaitForClusterAPIPodsReady() error {
	if err := ensureK8sClient(); err != nil {
		return err
	}
	if err := waitForClusterAPIPodsRunning(); err != nil {
		return fmt.Errorf("wait for cluster-api pods running: %w", err)
	}
	return nil
}

func waitForClusterAPIPodsRunning() error {
	client := global.K8s.GetClient()
	namespace := "cluster-system"
	deadline := time.Now().Add(waitForClusterAPIPodsTimeout)

	for {
		if time.Now().After(deadline) {
			return fmt.Errorf("timed out waiting for cluster-api pods to be ready after %v", waitForClusterAPIPodsTimeout)
		}

		time.Sleep(time.Duration(rand.IntnRange(utils.DefaultMinCheckSeconds, utils.DefaultMaxCheckSeconds)) * time.Second)
		log.Info("Wait for the cluster-api pods to be ready...")

		pods, err := client.CoreV1().Pods(namespace).List(context.Background(), metav1.ListOptions{})
		if err != nil {
			continue
		}

		if len(pods.Items) == 0 {
			continue
		}

		if AreAllPodsReady(pods.Items) {
			log.Info("Cluster-api pods are ready")
			return nil
		}
	}
}

// IsPodReady checks if a pod's PodReady condition is True.
func IsPodReady(pod *corev1.Pod) bool {
	for _, cond := range pod.Status.Conditions {
		if cond.Type == corev1.PodReady {
			return cond.Status == corev1.ConditionTrue
		}
	}
	return false
}

// notReadyContainers returns the names of containers whose Ready field is false.
func notReadyContainers(pod *corev1.Pod) []string {
	var names []string
	for _, cs := range pod.Status.ContainerStatuses {
		if !cs.Ready {
			names = append(names, cs.Name)
		}
	}
	return names
}

// logNotRunningPod logs why a pod is not in Running phase.
func logNotRunningPod(pod *corev1.Pod) {
	if len(pod.Status.ContainerStatuses) > 0 {
		lastContainer := pod.Status.ContainerStatuses[len(pod.Status.ContainerStatuses)-1]
		if lastContainer.State.Waiting != nil {
			log.Warnf("Pod %s is not running, phase=%s, container %s waiting: %s",
				pod.Name, pod.Status.Phase, lastContainer.Name, lastContainer.State.Waiting.Reason)
			return
		}
	}
	log.Warnf("Pod %s is not running, phase=%s", pod.Name, pod.Status.Phase)
}

// AreAllPodsReady checks that every pod is in Running phase AND has its
// PodReady condition set to True. This ensures readiness probes have passed,
// which is required for Service Endpoints to be populated.
func AreAllPodsReady(pods []corev1.Pod) bool {
	for _, pod := range pods {
		if pod.Status.Phase != corev1.PodRunning {
			logNotRunningPod(&pod)
			return false
		}

		if !IsPodReady(&pod) {
			log.Warnf("Pod %s is running but not ready, not-ready containers: %v",
				pod.Name, notReadyContainers(&pod))
			return false
		}
	}
	return true
}

func DeployClusterAPI(repo, manifestsVersion, providerVersion string) error {
	if err := ensureK8sClient(); err != nil {
		return err
	}

	tmplDir := filepath.Join(global.Workspace, "tmpl")
	if err := writeClusterAPITemplates(tmplDir); err != nil {
		return err
	}

	// 1. Generate webhook TLS certificates and create K8s Secrets
	log.Info("Generate webhook TLS certificates...")
	capiCaBundle, bkeCaBundle, err := ensureWebhookSecrets()
	if err != nil {
		return fmt.Errorf("ensure webhook secrets: %w", err)
	}

	// 2. Deploy cluster-api.yaml (caBundle injected via template)
	log.Info("Install the Cluster API...")
	clusterAPIFile := filepath.Join(tmplDir, "cluster-api.yaml")
	if err := installClusterAPIWithRetry(clusterAPIFile, repo, capiCaBundle); err != nil {
		return fmt.Errorf("install cluster-api: %w", err)
	}

	// 3. Deploy cluster-api-bke.yaml (caBundle injected via template)
	// Note: the BKENode password encryption key Secret is created by the
	// bke-controller-manager itself on first startup (common/security
	// SecretKeyProvider); waitForClusterAPIPodsRunning below only returns
	// once the controller is Ready, i.e. the key exists by then.
	clusterAPIBKEFile := filepath.Join(tmplDir, "cluster-api-bke.yaml")
	params := map[string]string{
		"repo":             repo,
		"manifestsVersion": manifestsVersion,
		"providerVersion":  providerVersion,
		"bkeCaBundle":      bkeCaBundle,
	}
	if err := global.K8s.InstallYaml(clusterAPIBKEFile, params, ""); err != nil {
		return fmt.Errorf("install cluster-api-bke: %w", err)
	}

	if err := waitForClusterAPIPodsRunning(); err != nil {
		return fmt.Errorf("wait for cluster-api pods running: %w", err)
	}
	return nil
}
