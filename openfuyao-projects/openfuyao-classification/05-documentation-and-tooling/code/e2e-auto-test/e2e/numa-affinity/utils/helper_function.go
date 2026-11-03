package utils

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
)

type FunctionalConfig struct {
	SSHHost               string
	SSHPort               int
	SSHUser               string
	SSHPassword           string
	VolcanoNamespace      string
	VolcanoConfigMapName  string
	VolcanoSchedulerName  string
	KubeletConfigPath     string
	KubeletServiceName    string
	ContainerdServiceName string
	BizNamespace          string
	BizDeploymentName     string
	BizLabelSelector      string
	AutoRestore           bool
	HelmReleaseName       string
	HelmNamespace         string
	HelmURL               string
	HelmVersion           string
}

type PodSnapshot struct {
	Name         string
	UID          string
	Phase        string
	Ready        bool
	RestartCount int64
}

func LoadFunctionalConfig() (*FunctionalConfig, error) {
	host := firstNonEmpty(
		os.Getenv("FUNCTIONAL_TEST_IP"),
		os.Getenv("SYSTEM_TEST_IP"),
		os.Getenv("COMPONENTS_TEST_IP"),
	)
	user := firstNonEmpty(
		os.Getenv("FUNCTIONAL_TEST_USER"),
		os.Getenv("SYSTEM_TEST_USER"),
		os.Getenv("COMPONENTS_TEST_USER"),
	)
	password := firstNonEmpty(
		os.Getenv("FUNCTIONAL_TEST_PASSWORD"),
		os.Getenv("SYSTEM_TEST_PASSWORD"),
		os.Getenv("COMPONENTS_TEST_PASSWORD"),
	)
	portRaw := firstNonEmpty(
		os.Getenv("FUNCTIONAL_TEST_PORT"),
		os.Getenv("SYSTEM_TEST_PORT"),
		os.Getenv("COMPONENTS_TEST_PORT"),
		"22",
	)

	if host == "" || user == "" || password == "" {
		return nil, fmt.Errorf("FUNCTIONAL_TEST_IP/USER/PASSWORD (or SYSTEM_TEST_*/COMPONENTS_TEST_*) must be configured")
	}

	port, err := strconv.Atoi(portRaw)
	if err != nil {
		return nil, fmt.Errorf("invalid ssh port %q: %w", portRaw, err)
	}

	autoRestore, err := parseBoolWithDefault(os.Getenv("NUMA_FUNCTIONAL_AUTO_RESTORE"), true)
	if err != nil {
		return nil, err
	}

	selector := strings.TrimSpace(os.Getenv("NUMA_FUNCTIONAL_BIZ_LABEL_SELECTOR"))
	if selector == "" {
		labelKey := firstNonEmpty(os.Getenv("NUMA_FUNCTIONAL_BIZ_LABEL_KEY"), "app")
		labelValue := firstNonEmpty(os.Getenv("NUMA_FUNCTIONAL_BIZ_LABEL_VALUE"), "numa-functional-biz")
		selector = fmt.Sprintf("%s=%s", labelKey, labelValue)
	}

	cfg := &FunctionalConfig{
		SSHHost:               host,
		SSHPort:               port,
		SSHUser:               user,
		SSHPassword:           password,
		VolcanoNamespace:      firstNonEmpty(os.Getenv("NUMA_FUNCTIONAL_VOLCANO_NAMESPACE"), "volcano-system"),
		VolcanoConfigMapName:  firstNonEmpty(os.Getenv("NUMA_FUNCTIONAL_VOLCANO_CONFIGMAP"), "volcano-scheduler-configmap"),
		VolcanoSchedulerName:  firstNonEmpty(os.Getenv("NUMA_FUNCTIONAL_SCHEDULER_NAME"), "volcano"),
		KubeletConfigPath:     firstNonEmpty(os.Getenv("NUMA_FUNCTIONAL_KUBELET_CONFIG_PATH"), "/var/lib/kubelet/config.yaml"),
		KubeletServiceName:    firstNonEmpty(os.Getenv("NUMA_FUNCTIONAL_KUBELET_SERVICE"), "kubelet"),
		ContainerdServiceName: firstNonEmpty(os.Getenv("NUMA_FUNCTIONAL_CONTAINERD_SERVICE"), "containerd"),
		BizNamespace:          firstNonEmpty(os.Getenv("NUMA_FUNCTIONAL_BIZ_NAMESPACE"), "default"),
		BizDeploymentName:     firstNonEmpty(os.Getenv("NUMA_FUNCTIONAL_BIZ_DEPLOYMENT_NAME"), "numa-functional-biz"),
		BizLabelSelector:      selector,
		AutoRestore:           autoRestore,
		HelmReleaseName:       strings.TrimSpace(os.Getenv("NUMA_FUNCTIONAL_HELM_RELEASE")),
		HelmNamespace:         strings.TrimSpace(os.Getenv("NUMA_FUNCTIONAL_HELM_NAMESPACE")),
		HelmURL:               strings.TrimSpace(os.Getenv("NUMA_FUNCTIONAL_HELM_URL")),
		HelmVersion:           strings.TrimSpace(os.Getenv("NUMA_FUNCTIONAL_HELM_VERSION")),
	}

	return cfg, nil
}

func ExecStdout(sshClient *executor.SSHExecutor, cmd string) (string, error) {
	res, err := sshClient.Exec(cmd)
	if err != nil {
		stderr := ""
		if res != nil {
			stderr = strings.TrimSpace(res.Stderr)
			if stderr == "" {
				stderr = strings.TrimSpace(res.Stdout)
			}
		}
		return "", fmt.Errorf("run command failed: %s, stderr: %s, error: %w", cmd, stderr, err)
	}
	return strings.TrimSpace(res.Stdout), nil
}

func NormalizeOutput(raw string) string {
	text := strings.TrimSpace(raw)
	text = strings.Trim(text, "'")
	text = strings.Trim(text, "\"")
	return strings.TrimSpace(text)
}

func RestartSystemdServiceAndWaitActive(sshClient *executor.SSHExecutor, serviceName string) error {
	service := NormalizeOutput(serviceName)
	if service == "" {
		return fmt.Errorf("service name is empty")
	}

	cmd := fmt.Sprintf("systemctl restart %s && systemctl is-active %s", ShellQuote(service), ShellQuote(service))
	status, err := ExecStdout(sshClient, cmd)
	if err != nil {
		return err
	}
	if NormalizeOutput(status) != "active" {
		return fmt.Errorf("service %s is not active after restart", service)
	}
	return nil
}

func EnsureNamespace(sshClient *executor.SSHExecutor, namespace string) error {
	cmd := fmt.Sprintf("kubectl create namespace %s --dry-run=client -o yaml | kubectl apply -f -", ShellQuote(namespace))
	_, err := ExecStdout(sshClient, cmd)
	return err
}

func EnsureBusinessWorkload(sshClient *executor.SSHExecutor, cfg *FunctionalConfig) (bool, error) {
	if err := EnsureNamespace(sshClient, cfg.BizNamespace); err != nil {
		return false, err
	}

	deployCmd := fmt.Sprintf("kubectl get deployment %s -n %s --no-headers 2>/dev/null | wc -l", ShellQuote(cfg.BizDeploymentName), ShellQuote(cfg.BizNamespace))
	countRaw, err := ExecStdout(sshClient, deployCmd)
	if err != nil {
		return false, err
	}

	count, err := strconv.Atoi(strings.TrimSpace(countRaw))
	if err != nil {
		return false, fmt.Errorf("parse deployment count failed: %w", err)
	}
	if count > 0 {
		// Wait for existing deployment pods to be fully ready to avoid race conditions
		return false, nil
	}

	labelKey, labelValue, err := ParseSingleLabelSelector(cfg.BizLabelSelector)
	if err != nil {
		return false, err
	}

	yamlContent := BuildBusinessDeploymentYAML(cfg.BizNamespace, cfg.BizDeploymentName, labelKey, labelValue)
	applyCmd := fmt.Sprintf("cat <<'EOF' | kubectl apply -f -\n%s\nEOF", yamlContent)
	if _, err = ExecStdout(sshClient, applyCmd); err != nil {
		return false, err
	}

	return true, nil
}

func DeleteBusinessWorkload(sshClient *executor.SSHExecutor, cfg *FunctionalConfig) {
	_, _ = ExecStdout(sshClient, fmt.Sprintf("kubectl delete deployment %s -n %s --ignore-not-found=true", ShellQuote(cfg.BizDeploymentName), ShellQuote(cfg.BizNamespace)))
	_, _ = ExecStdout(sshClient, fmt.Sprintf("kubectl wait --for=delete pod -l %s -n %s --timeout=60s 2>/dev/null", ShellQuote(cfg.BizLabelSelector), ShellQuote(cfg.BizNamespace)))
}

func GetRunningPodSnapshots(sshClient *executor.SSHExecutor, namespace string, labelSelector string) ([]PodSnapshot, error) {
	selectorArg := ""
	if strings.TrimSpace(labelSelector) != "" {
		selectorArg = fmt.Sprintf("-l %s", ShellQuote(labelSelector))
	}

	cmd := fmt.Sprintf("kubectl get pods -n %s %s -o json", ShellQuote(namespace), selectorArg)
	rawJSON, err := ExecStdout(sshClient, cmd)
	if err != nil {
		return nil, err
	}

	var podList struct {
		Items []struct {
			Metadata struct {
				Name              string  `json:"name"`
				UID               string  `json:"uid"`
				DeletionTimestamp *string `json:"deletionTimestamp"`
			} `json:"metadata"`
			Status struct {
				Phase             string `json:"phase"`
				ContainerStatuses []struct {
					RestartCount int64 `json:"restartCount"`
				} `json:"containerStatuses"`
				Conditions []struct {
					Type   string `json:"type"`
					Status string `json:"status"`
				} `json:"conditions"`
			} `json:"status"`
		} `json:"items"`
	}

	if err = json.Unmarshal([]byte(rawJSON), &podList); err != nil {
		return nil, fmt.Errorf("parse pod json failed: %w", err)
	}

	result := make([]PodSnapshot, 0)
	nonRunning := make([]string, 0)
	for _, item := range podList.Items {
		if item.Metadata.DeletionTimestamp != nil {
			continue
		}
		if item.Status.Phase != "Running" {
			nonRunning = append(nonRunning, fmt.Sprintf("%s(phase=%s)", item.Metadata.Name, item.Status.Phase))
			continue
		}

		ready := false
		for _, condition := range item.Status.Conditions {
			if condition.Type == "Ready" && strings.EqualFold(condition.Status, "True") {
				ready = true
				break
			}
		}

		restartCount := int64(0)
		for _, status := range item.Status.ContainerStatuses {
			restartCount += status.RestartCount
		}

		result = append(result, PodSnapshot{
			Name:         item.Metadata.Name,
			UID:          item.Metadata.UID,
			Phase:        item.Status.Phase,
			Ready:        ready,
			RestartCount: restartCount,
		})
	}

	if len(result) == 0 {
		if len(nonRunning) > 0 {
			return nil, fmt.Errorf("no running pods found in namespace=%s selector=%s, non-running pods: %s", namespace, labelSelector, strings.Join(nonRunning, ", "))
		}
		return nil, fmt.Errorf("no pods found in namespace=%s selector=%s", namespace, labelSelector)
	}

	return result, nil
}

func CollectPodDiagnostics(sshClient *executor.SSHExecutor, namespace string, labelSelector string) string {
	var parts []string

	podsCmd := fmt.Sprintf("kubectl get pods -n %s", ShellQuote(namespace))
	if strings.TrimSpace(labelSelector) != "" {
		podsCmd = fmt.Sprintf("kubectl get pods -n %s -l %s", ShellQuote(namespace), ShellQuote(labelSelector))
	}
	if out, err := ExecStdout(sshClient, podsCmd); err == nil {
		parts = append(parts, "kubectl get pods:\n"+out)
	} else {
		parts = append(parts, fmt.Sprintf("kubectl get pods failed: %v", err))
	}

	eventsCmd := fmt.Sprintf("kubectl get events -n %s --sort-by=.lastTimestamp 2>/dev/null | tail -n 30", ShellQuote(namespace))
	if out, err := ExecStdout(sshClient, eventsCmd); err == nil {
		parts = append(parts, "recent events (tail 30):\n"+out)
	} else {
		parts = append(parts, fmt.Sprintf("kubectl get events failed: %v", err))
	}

	return strings.Join(parts, "\n\n")
}

func WaitForVolcanoComponentsReady(sshClient *executor.SSHExecutor, namespace string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	var lastErr error

	for time.Now().Before(deadline) {
		lastErr = nil

		// 1) Wait for volcano core pods to be Running.
		podsCmd := fmt.Sprintf(
			"kubectl get pods -n %s -o json 2>/dev/null",
			ShellQuote(namespace),
		)
		rawJSON, err := ExecStdout(sshClient, podsCmd)
		if err != nil {
			lastErr = fmt.Errorf("query volcano pods failed: %w", err)
			time.Sleep(5 * time.Second)
			continue
		}

		var podList struct {
			Items []struct {
				Metadata struct {
					Name string `json:"name"`
				} `json:"metadata"`
				Status struct {
					Phase      string `json:"phase"`
					Conditions []struct {
						Type   string `json:"type"`
						Status string `json:"status"`
					} `json:"conditions"`
				} `json:"status"`
			} `json:"items"`
		}
		if err = json.Unmarshal([]byte(rawJSON), &podList); err != nil {
			lastErr = fmt.Errorf("parse volcano pod json failed: %w", err)
			time.Sleep(5 * time.Second)
			continue
		}

		coreKeywords := []string{"volcano-scheduler", "volcano-controllers", "volcano-admission"}
		readyCore := make(map[string]bool, len(coreKeywords))
		for _, kw := range coreKeywords {
			readyCore[kw] = false
		}
		for _, item := range podList.Items {
			for _, kw := range coreKeywords {
				if strings.Contains(item.Metadata.Name, kw) && item.Status.Phase == "Running" {
					readyCore[kw] = true
				}
			}
		}

		allCoreReady := true
		for _, kw := range coreKeywords {
			if !readyCore[kw] {
				allCoreReady = false
				break
			}
		}
		if !allCoreReady {
			notReady := make([]string, 0)
			for _, kw := range coreKeywords {
				if !readyCore[kw] {
					notReady = append(notReady, kw)
				}
			}
			lastErr = fmt.Errorf("volcano core pods not all running yet, missing: %s", strings.Join(notReady, ", "))
			time.Sleep(5 * time.Second)
			continue
		}

		// 2) Wait for volcano-admission Service to have ready endpoints.
		epCmd := fmt.Sprintf(
			"kubectl get endpoints volcano-admission-service -n %s -o json 2>/dev/null",
			ShellQuote(namespace),
		)
		epJSON, err := ExecStdout(sshClient, epCmd)
		if err != nil {
			lastErr = fmt.Errorf("query volcano-admission endpoints failed: %w", err)
			time.Sleep(5 * time.Second)
			continue
		}

		var ep struct {
			Subsets []struct {
				Addresses []struct {
					IP string `json:"ip"`
				} `json:"addresses"`
				NotReadyAddresses []struct {
					IP string `json:"ip"`
				} `json:"notReadyAddresses"`
			} `json:"subsets"`
		}
		if err = json.Unmarshal([]byte(epJSON), &ep); err != nil {
			lastErr = fmt.Errorf("parse volcano-admission endpoints json failed: %w", err)
			time.Sleep(5 * time.Second)
			continue
		}

		readyCount := 0
		for _, subset := range ep.Subsets {
			readyCount += len(subset.Addresses)
		}
		if readyCount == 0 {
			lastErr = fmt.Errorf("volcano-admission-service has no ready endpoints yet")
			time.Sleep(5 * time.Second)
			continue
		}

		// All checks passed.
		return nil
	}

	if lastErr == nil {
		lastErr = fmt.Errorf("volcano components not ready within %s", timeout)
	}
	return lastErr
}

func AssertPodSnapshotsStable(before []PodSnapshot, current []PodSnapshot) error {
	currentByUID := make(map[string]PodSnapshot, len(current))
	for _, item := range current {
		currentByUID[item.UID] = item
	}

	for _, old := range before {
		newer, exists := currentByUID[old.UID]
		if !exists {
			return fmt.Errorf("pod was recreated or disappeared: name=%s uid=%s", old.Name, old.UID)
		}
		if !newer.Ready || newer.Phase != "Running" {
			return fmt.Errorf("pod is not running and ready: name=%s phase=%s ready=%v", newer.Name, newer.Phase, newer.Ready)
		}
		if newer.RestartCount != old.RestartCount {
			return fmt.Errorf("pod restart count changed: name=%s before=%d after=%d", newer.Name, old.RestartCount, newer.RestartCount)
		}
	}

	return nil
}

func BackupFile(sshClient *executor.SSHExecutor, filePath string) (string, error) {
	sanitized := strings.NewReplacer("/", "_", "\\", "_", ":", "_").Replace(filePath)
	backupPath := fmt.Sprintf("/tmp/numa-functional-%d-%s.bak", time.Now().UnixNano(), sanitized)

	cmd := fmt.Sprintf("cp %s %s", ShellQuote(filePath), ShellQuote(backupPath))
	if _, err := ExecStdout(sshClient, cmd); err != nil {
		return "", err
	}

	return backupPath, nil
}

func RestoreFile(sshClient *executor.SSHExecutor, backupPath string, targetPath string) error {
	cmd := fmt.Sprintf("cp %s %s", ShellQuote(backupPath), ShellQuote(targetPath))
	_, err := ExecStdout(sshClient, cmd)
	return err
}

func RemoveFile(sshClient *executor.SSHExecutor, filePath string) {
	_, _ = ExecStdout(sshClient, fmt.Sprintf("rm -f %s", ShellQuote(filePath)))
}

func EnsurePreferClosestNumaNodesFalse(sshClient *executor.SSHExecutor, kubeletConfigPath string) (bool, error) {
	cmd := fmt.Sprintf(
		"if grep -Eq '^[[:space:]]*prefer-closest-numa-nodes:[[:space:]]*true([[:space:]]|$)' %s; then "+
			"sed -i -E 's/^([[:space:]]*prefer-closest-numa-nodes:[[:space:]]*)true([[:space:]]|$)/\\1false\\2/' %s; "+
			"echo changed; else echo unchanged; fi",
		ShellQuote(kubeletConfigPath),
		ShellQuote(kubeletConfigPath),
	)

	out, err := ExecStdout(sshClient, cmd)
	if err != nil {
		return false, err
	}

	return strings.Contains(strings.ToLower(out), "changed"), nil
}

func ParseSingleLabelSelector(selector string) (string, string, error) {
	parts := strings.Split(selector, "=")
	if len(parts) != 2 {
		return "", "", fmt.Errorf("selector should be single key=value, got: %s", selector)
	}
	key := strings.TrimSpace(parts[0])
	value := strings.TrimSpace(parts[1])
	if key == "" || value == "" {
		return "", "", fmt.Errorf("selector key/value should not be empty: %s", selector)
	}
	return key, value, nil
}

func BuildBusinessDeploymentYAML(namespace string, deploymentName string, labelKey string, labelValue string) string {
	return fmt.Sprintf(`apiVersion: apps/v1
kind: Deployment
metadata:
  name: %s
  namespace: %s
spec:
  replicas: 1
  selector:
    matchLabels:
      %s: %s
  template:
    metadata:
      labels:
        %s: %s
    spec:
      containers:
      - name: pause
        image: alpine:latest
        imagePullPolicy: IfNotPresent
        command: ["sleep", "360000"]
        resources:
          limits:
            cpu: "100m"
            memory: "64Mi"
          requests:
            cpu: "100m"
            memory: "64Mi"`, deploymentName, namespace, labelKey, labelValue, labelKey, labelValue)
}

func ShellQuote(text string) string {
	escaped := strings.ReplaceAll(text, "'", `'"'"'`)
	return fmt.Sprintf("'%s'", escaped)
}

func GenerateCaseNamespace(caseID string) string {
	normalized := strings.ToLower(caseID)
	normalized = strings.ReplaceAll(normalized, "_", "-")
	normalized = strings.ReplaceAll(normalized, ".", "-")
	return fmt.Sprintf("%s-%d", normalized, time.Now().Unix())
}

func BuildVolcanoDeploymentYAML(namespace string, deploymentName string, podLabel string, schedulerName string) string {
	return fmt.Sprintf(`apiVersion: apps/v1
kind: Deployment
metadata:
  name: %s
  namespace: %s
spec:
  replicas: 1
  selector:
    matchLabels:
      app: %s
  template:
    metadata:
      labels:
        app: %s
      annotations:
        volcano.sh/numa-topology-policy: single-numa-node
    spec:
      schedulerName: %s
      containers:
      - name: pause
        image: alpine:latest
        imagePullPolicy: IfNotPresent
        command: ["sleep", "360000"]
        resources:
          limits:
            cpu: "100m"
            memory: "64Mi"
          requests:
            cpu: "100m"
            memory: "64Mi"`, deploymentName, namespace, podLabel, podLabel, schedulerName)
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		trimmed := strings.TrimSpace(value)
		if trimmed != "" {
			return trimmed
		}
	}
	return ""
}

func parseBoolWithDefault(raw string, defaultValue bool) (bool, error) {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return defaultValue, nil
	}

	parsed, err := strconv.ParseBool(trimmed)
	if err != nil {
		return false, fmt.Errorf("invalid bool value %q: %w", raw, err)
	}
	return parsed, nil
}
