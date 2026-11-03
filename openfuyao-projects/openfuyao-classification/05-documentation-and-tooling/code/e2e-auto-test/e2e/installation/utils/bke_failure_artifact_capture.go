package utils

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
)

const (
	bkeControllerManagerNamespace = "cluster-system"
	controllerManagerPodPrefix      = "bke-controller-manager"
	bkeFailureArtifactLogPrefix     = "[bke-failure-artifact]"
)

type bkeFailureArtifactTarget struct {
	name           string
	kubeconfigPath string
}

type controllerManagerPodInfo struct {
	name         string
	nodeName     string
	phase        string
	containers   []string
	restartCount map[string]int32
}

type logfFunc func(format string, args ...interface{})

// BKEFailureArtifactOutputDir builds the artifact directory for a failed spec.
// Prefer E2E_ARTIFACTS_DIR / ARTIFACTS_DIR; otherwise use e2e-artifacts/installation.
func BKEFailureArtifactOutputDir(specFileName string, lineNumber int) (string, error) {
	baseDir := strings.TrimSpace(os.Getenv("E2E_ARTIFACTS_DIR"))
	if baseDir == "" {
		baseDir = strings.TrimSpace(os.Getenv("ARTIFACTS_DIR"))
	}
	if baseDir == "" {
		baseDir = filepath.Join("e2e-artifacts", "installation")
	}

	fileName := filepath.Base(specFileName)
	ext := filepath.Ext(fileName)
	if ext != "" {
		fileName = strings.TrimSuffix(fileName, ext)
	}

	specDir := fmt.Sprintf("%s_%d", fileName, lineNumber)
	outputDir := filepath.Join(baseDir, specDir)
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return "", err
	}
	return outputDir, nil
}

// CaptureBKEFailureArtifacts collects BKE failure diagnostics (BC/BN CRD snapshots and
// bke-controller-manager logs) from bootstrap and discovered child clusters into outputDir
// via SSH on GUIDE_NODE_HOST. logf may be nil.
func CaptureBKEFailureArtifacts(ctx context.Context, outputDir string, logf logfFunc) {
	if logf == nil {
		logf = func(string, ...interface{}) {}
	}
	logf("%s output dir: %s\n", bkeFailureArtifactLogPrefix, outputDir)

	sshExec, cleanup, err := newGuideNodeSSHExecutor(logf)
	if err != nil {
		logf("%s connect guide node failed: %v\n", bkeFailureArtifactLogPrefix, err)
		return
	}
	defer cleanup()

	targets, err := bkeFailureArtifactTargetsViaSSH(sshExec, logf)
	if err != nil {
		logf("%s build artifact targets failed: %v\n", bkeFailureArtifactLogPrefix, err)
		return
	}
	if len(targets) == 0 {
		logf("%s no kubeconfig target available\n", bkeFailureArtifactLogPrefix)
		return
	}

	// Collect BC/BN CRD snapshots before controller-manager pod logs so cluster
	// state is captured as early as possible on failure (before slow log dumps).
	for _, target := range targets {
		if err := captureBKEClusterCRDSnapshotsViaSSH(outputDir, sshExec, target, logf); err != nil {
			logf("%s capture cluster resources on %s failed: %v\n", bkeFailureArtifactLogPrefix, target.name, err)
		}
	}

	for _, target := range targets {
		if err := captureControllerManagerLogsForTargetViaSSH(ctx, outputDir, sshExec, target, logf); err != nil {
			logf("%s capture %s failed: %v\n", bkeFailureArtifactLogPrefix, target.name, err)
		}
	}
}

func newGuideNodeSSHExecutor(logf logfFunc) (executor.Executor, func(), error) {
	guideConfig := config.LoadGuideNodeFromEnv()
	if strings.TrimSpace(guideConfig.Host) == "" {
		return nil, nil, fmt.Errorf("GUIDE_NODE_HOST is not set")
	}
	if strings.TrimSpace(guideConfig.Password) == "" {
		return nil, nil, fmt.Errorf("GUIDE_NODE_PASSWORD is not set")
	}

	sshExec, err := executor.NewSSHExecutor(
		guideConfig.Host, guideConfig.Port, guideConfig.Username, guideConfig.Password,
	)
	if err != nil {
		return nil, nil, err
	}
	logf("%s connected guide node %s:%d\n", bkeFailureArtifactLogPrefix, guideConfig.Host, guideConfig.Port)
	return sshExec, func() { _ = sshExec.Close() }, nil
}

func bkeFailureArtifactTargetsViaSSH(sshExec executor.Executor, logf logfFunc) ([]bkeFailureArtifactTarget, error) {
	bootstrapKubeconfig := DefaultBootstraoClusterKubeconfig
	bootstrapHost, err := kubeconfigServerHostViaSSH(sshExec, bootstrapKubeconfig)
	if err != nil {
		return nil, fmt.Errorf("resolve bootstrap kubeconfig server: %w", err)
	}

	seenHosts := map[string]bool{bootstrapHost: true}
	targets := []bkeFailureArtifactTarget{}

	// with-management-cluster 场景下，业务 BC/BN 在引导节点默认 kubeconfig 对应的管理集群 API 上。
	if defaultHost, err := kubeconfigServerHostViaSSH(sshExec, ""); err != nil {
		logf("%s resolve default kubeconfig server failed: %v\n", bkeFailureArtifactLogPrefix, err)
	} else if !seenHosts[defaultHost] {
		seenHosts[defaultHost] = true
		targets = append(targets, bkeFailureArtifactTarget{
			name:           "default",
			kubeconfigPath: "",
		})
	}

	targets = append(targets, bkeFailureArtifactTarget{
		name:           "bootstrap",
		kubeconfigPath: bootstrapKubeconfig,
	})

	childTargets := discoverChildClusterArtifactTargetsViaSSH(sshExec, bootstrapKubeconfig, bootstrapHost, logf)
	for _, target := range childTargets {
		host, err := kubeconfigServerHostViaSSH(sshExec, target.kubeconfigPath)
		if err != nil {
			logf("%s resolve kubeconfig server for %s failed: %v\n", bkeFailureArtifactLogPrefix, target.name, err)
			continue
		}
		if seenHosts[host] {
			continue
		}
		seenHosts[host] = true
		targets = append(targets, target)
	}
	return targets, nil
}

func discoverChildClusterArtifactTargetsViaSSH(
	sshExec executor.Executor,
	bootstrapKubeconfig, bootstrapHost string,
	logf logfFunc,
) []bkeFailureArtifactTarget {
	result, err := execKubectlOnGuide(sshExec, bootstrapKubeconfig, "get secrets -A -o json")
	if err != nil {
		logf("%s list kubeconfig secrets failed: %v\n", bkeFailureArtifactLogPrefix, err)
		return nil
	}
	if result.ExitCode != 0 {
		logf("%s list kubeconfig secrets failed: %s\n", bkeFailureArtifactLogPrefix, strings.TrimSpace(result.Stderr))
		return nil
	}

	var secretList struct {
		Items []struct {
			Metadata struct {
				Name      string `json:"name"`
				Namespace string `json:"namespace"`
			} `json:"metadata"`
			Data map[string]string `json:"data"`
		} `json:"items"`
	}
	if err := json.Unmarshal([]byte(result.Stdout), &secretList); err != nil {
		logf("%s parse kubeconfig secrets failed: %v\n", bkeFailureArtifactLogPrefix, err)
		return nil
	}

	targets := []bkeFailureArtifactTarget{}
	seenHosts := map[string]bool{bootstrapHost: true}
	for _, secret := range secretList.Items {
		if !strings.HasSuffix(secret.Metadata.Name, "-kubeconfig") {
			continue
		}

		encoded := secret.Data["value"]
		if encoded == "" {
			encoded = secret.Data["ha"]
		}
		if encoded == "" {
			continue
		}

		clusterName := strings.TrimSuffix(secret.Metadata.Name, "-kubeconfig")
		remotePath := fmt.Sprintf("/tmp/e2e-bke-failure-artifact-%s-%s.conf",
			sanitizePathSegment(secret.Metadata.Namespace),
			sanitizePathSegment(clusterName),
		)
		if err := saveRemoteKubeconfigViaSSH(sshExec, encoded, remotePath); err != nil {
			logf("%s save kubeconfig secret %s/%s failed: %v\n",
				bkeFailureArtifactLogPrefix, secret.Metadata.Namespace, secret.Metadata.Name, err)
			continue
		}

		host, err := kubeconfigServerHostViaSSH(sshExec, remotePath)
		if err != nil {
			logf("%s parse kubeconfig secret %s/%s failed: %v\n",
				bkeFailureArtifactLogPrefix, secret.Metadata.Namespace, secret.Metadata.Name, err)
			continue
		}
		if seenHosts[host] {
			continue
		}

		seenHosts[host] = true
		targets = append(targets, bkeFailureArtifactTarget{
			name:           sanitizePathSegment(fmt.Sprintf("%s-%s", secret.Metadata.Namespace, clusterName)),
			kubeconfigPath: remotePath,
		})
	}
	return targets
}

func captureControllerManagerLogsForTargetViaSSH(
	_ context.Context,
	outputDir string,
	sshExec executor.Executor,
	target bkeFailureArtifactTarget,
	logf logfFunc,
) error {
	pods, err := listControllerManagerPodsViaSSH(sshExec, target.kubeconfigPath)
	if err != nil {
		logf("%s list pods in %s on %s failed: %v\n",
			bkeFailureArtifactLogPrefix, bkeControllerManagerNamespace, target.name, err)
		return nil
	}

	matched := false
	for _, pod := range pods {
		matched = true
		if err := capturePodLogsViaSSH(outputDir, sshExec, target, pod, logf); err != nil {
			logf("%s capture pod %s/%s on %s failed: %v\n",
				bkeFailureArtifactLogPrefix, bkeControllerManagerNamespace, pod.name, target.name, err)
		}
	}

	if !matched {
		logf("%s no pod with prefix %q found on %s\n", bkeFailureArtifactLogPrefix, controllerManagerPodPrefix, target.name)
	}
	return nil
}

func captureBKEClusterCRDSnapshotsViaSSH(
	outputDir string,
	sshExec executor.Executor,
	target bkeFailureArtifactTarget,
	logf logfFunc,
) error {
	resources := []struct {
		name string
		args string
		ext  string
	}{
		{
			name: "bc-list",
			args: "get bc -A -o yaml",
			ext:  ".yaml",
		},
		{
			name: "bn-list",
			args: "get bn -A -o yaml",
			ext:  ".yaml",
		},
		{
			name: "bc-describe",
			args: "describe bc -A",
			ext:  ".txt",
		},
		{
			name: "bn-describe",
			args: "describe bn -A",
			ext:  ".txt",
		},
	}

	for _, resource := range resources {
		result, err := execKubectlOnGuide(sshExec, target.kubeconfigPath, resource.args)
		if err != nil {
			logf("%s %s on %s failed: %v\n", bkeFailureArtifactLogPrefix, resource.name, target.name, err)
			continue
		}
		if result.ExitCode != 0 {
			logf("%s %s on %s failed: %s\n",
				bkeFailureArtifactLogPrefix, resource.name, target.name, strings.TrimSpace(result.Stderr))
			continue
		}

		fileName := fmt.Sprintf("%s-%s%s",
			sanitizePathSegment(target.name),
			resource.name,
			resource.ext,
		)
		if err := os.WriteFile(filepath.Join(outputDir, fileName), []byte(result.Stdout), 0644); err != nil {
			logf("%s write %s on %s failed: %v\n", bkeFailureArtifactLogPrefix, resource.name, target.name, err)
			continue
		}
		logf("%s saved %s on %s\n", bkeFailureArtifactLogPrefix, resource.name, target.name)
	}

	return nil
}

func listControllerManagerPodsViaSSH(sshExec executor.Executor, kubeconfigPath string) ([]controllerManagerPodInfo, error) {
	result, err := execKubectlOnGuide(sshExec, kubeconfigPath,
		fmt.Sprintf("get pods -n %s -o json", bkeControllerManagerNamespace))
	if err != nil {
		return nil, err
	}
	if result.ExitCode != 0 {
		return nil, fmt.Errorf("%s", strings.TrimSpace(result.Stderr))
	}

	var podList struct {
		Items []struct {
			Metadata struct {
				Name string `json:"name"`
			} `json:"metadata"`
			Spec struct {
				NodeName   string `json:"nodeName"`
				Containers []struct {
					Name string `json:"name"`
				} `json:"containers"`
				InitContainers []struct {
					Name string `json:"name"`
				} `json:"initContainers"`
			} `json:"spec"`
			Status struct {
				Phase             string `json:"phase"`
				ContainerStatuses []struct {
					Name         string `json:"name"`
					RestartCount int32  `json:"restartCount"`
				} `json:"containerStatuses"`
				InitContainerStatuses []struct {
					Name         string `json:"name"`
					RestartCount int32  `json:"restartCount"`
				} `json:"initContainerStatuses"`
			} `json:"status"`
		} `json:"items"`
	}
	if err := json.Unmarshal([]byte(result.Stdout), &podList); err != nil {
		return nil, err
	}

	pods := make([]controllerManagerPodInfo, 0)
	for _, item := range podList.Items {
		if !strings.HasPrefix(item.Metadata.Name, controllerManagerPodPrefix) {
			continue
		}

		pod := controllerManagerPodInfo{
			name:         item.Metadata.Name,
			nodeName:     item.Spec.NodeName,
			phase:        item.Status.Phase,
			restartCount: map[string]int32{},
		}
		for _, container := range append(item.Spec.InitContainers, item.Spec.Containers...) {
			pod.containers = append(pod.containers, container.Name)
		}
		for _, status := range append(item.Status.InitContainerStatuses, item.Status.ContainerStatuses...) {
			pod.restartCount[status.Name] = status.RestartCount
		}
		pods = append(pods, pod)
	}
	return pods, nil
}

func capturePodLogsViaSSH(
	outputDir string,
	sshExec executor.Executor,
	target bkeFailureArtifactTarget,
	pod controllerManagerPodInfo,
	logf logfFunc,
) error {
	metadata := fmt.Sprintf("target=%s\nnamespace=%s\npod=%s\nnode=%s\nphase=%s\ncapturedAt=%s\n",
		target.name,
		bkeControllerManagerNamespace,
		pod.name,
		pod.nodeName,
		pod.phase,
		time.Now().Format(time.RFC3339),
	)
	metadataPath := filepath.Join(outputDir, fmt.Sprintf("%s-%s-metadata.txt",
		sanitizePathSegment(target.name), sanitizePathSegment(pod.name)))
	if err := os.WriteFile(metadataPath, []byte(metadata), 0644); err != nil {
		return err
	}

	for _, containerName := range pod.containers {
		if err := writeContainerLogViaSSH(outputDir, sshExec, target, pod.name, containerName, false, logf); err != nil {
			logf("%s write current log for %s/%s/%s failed: %v\n",
				bkeFailureArtifactLogPrefix, bkeControllerManagerNamespace, pod.name, containerName, err)
		}
		if pod.restartCount[containerName] > 0 {
			if err := writeContainerLogViaSSH(outputDir, sshExec, target, pod.name, containerName, true, logf); err != nil {
				logf("%s write previous log for %s/%s/%s failed: %v\n",
					bkeFailureArtifactLogPrefix, bkeControllerManagerNamespace, pod.name, containerName, err)
			}
		}
	}
	return nil
}

func writeContainerLogViaSSH(
	outputDir string,
	sshExec executor.Executor,
	target bkeFailureArtifactTarget,
	podName, containerName string,
	previous bool,
	logf logfFunc,
) error {
	args := fmt.Sprintf("logs -n %s %s -c %s --timestamps=true",
		bkeControllerManagerNamespace, shellQuote(podName), shellQuote(containerName))
	if previous {
		args += " --previous"
	}

	result, err := execKubectlOnGuide(sshExec, target.kubeconfigPath, args)
	if err != nil {
		return err
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("%s", strings.TrimSpace(result.Stderr))
	}

	suffix := "current"
	if previous {
		suffix = "previous"
	}
	fileName := fmt.Sprintf("%s-%s-%s-%s.log",
		sanitizePathSegment(target.name),
		sanitizePathSegment(podName),
		sanitizePathSegment(containerName),
		suffix,
	)
	if err := os.WriteFile(filepath.Join(outputDir, fileName), []byte(result.Stdout), 0644); err != nil {
		return err
	}
	logf("%s saved %s/%s/%s %s log\n",
		bkeFailureArtifactLogPrefix, bkeControllerManagerNamespace, podName, containerName, suffix)
	return nil
}

func execKubectlOnGuide(sshExec executor.Executor, kubeconfigPath, args string) (*executor.ExecResult, error) {
	prefix := ""
	if strings.TrimSpace(kubeconfigPath) != "" {
		prefix = fmt.Sprintf("KUBECONFIG=%s ", kubeconfigPath)
	}
	return sshExec.Exec(fmt.Sprintf("%skubectl %s", prefix, args))
}

func kubeconfigServerHostViaSSH(sshExec executor.Executor, kubeconfigPath string) (string, error) {
	result, err := execKubectlOnGuide(sshExec, kubeconfigPath,
		`config view --minify -o jsonpath='{.clusters[0].cluster.server}'`)
	if err != nil {
		return "", err
	}
	if result.ExitCode != 0 {
		return "", fmt.Errorf("%s", strings.TrimSpace(result.Stderr))
	}
	host := strings.TrimSpace(result.Stdout)
	if host == "" {
		return "", fmt.Errorf("empty server host in kubeconfig %s", kubeconfigPath)
	}
	return host, nil
}

func saveRemoteKubeconfigViaSSH(sshExec executor.Executor, encoded, remotePath string) error {
	decoded, err := base64.StdEncoding.DecodeString(strings.TrimSpace(encoded))
	if err != nil {
		return err
	}
	if !strings.Contains(string(decoded), "apiVersion:") {
		return fmt.Errorf("decoded kubeconfig is invalid")
	}

	encodedContent := base64.StdEncoding.EncodeToString(decoded)
	cmd := fmt.Sprintf("cat <<'EOF' | base64 -d > %s\n%s\nEOF", remotePath, encodedContent)
	result, err := sshExec.Exec(cmd)
	if err != nil {
		return err
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("%s", strings.TrimSpace(result.Stderr))
	}
	return nil
}

func shellQuote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", "'\\''") + "'"
}

func sanitizePathSegment(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return "unknown"
	}

	re := regexp.MustCompile(`[^A-Za-z0-9_.-]+`)
	value = re.ReplaceAllString(value, "-")
	value = strings.Trim(value, "-_.")
	if value == "" {
		return "unknown"
	}
	if len(value) > 120 {
		value = value[:120]
	}
	return value
}
