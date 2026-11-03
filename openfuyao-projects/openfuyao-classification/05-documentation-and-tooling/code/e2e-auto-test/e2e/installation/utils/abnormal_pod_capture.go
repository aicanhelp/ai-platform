package utils

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
)

const abnormalPodCaptureLimit = 10

type abnormalPodInfo struct {
	namespace string
	name      string
	phase     string
}

type abnormalPodContainerStatus struct {
	Ready bool `json:"ready"`
}

// CaptureAbnormalPodDescribeArtifacts collects `kubectl describe pod` output for
// abnormal pods (phase not Running/Succeeded, or with containers not ready) on
// bootstrap and discovered child clusters into outputDir via SSH on GUIDE_NODE_HOST.
// logf may be nil.
func CaptureAbnormalPodDescribeArtifacts(ctx context.Context, outputDir string, logf logfFunc) {
	if logf == nil {
		logf = func(string, ...interface{}) {}
	}

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

	for _, target := range targets {
		captureAbnormalPodDescribeForTargetViaSSH(outputDir, sshExec, target, logf)
	}
}

func captureAbnormalPodDescribeForTargetViaSSH(
	outputDir string,
	sshExec executor.Executor,
	target bkeFailureArtifactTarget,
	logf logfFunc,
) {
	pods, err := listAbnormalPodsViaSSH(sshExec, target.kubeconfigPath)
	if err != nil {
		logf("%s list abnormal pods on %s failed: %v\n", bkeFailureArtifactLogPrefix, target.name, err)
		return
	}
	if len(pods) == 0 {
		logf("%s no abnormal pod found on %s\n", bkeFailureArtifactLogPrefix, target.name)
		return
	}

	summary := make([]string, 0, len(pods))
	for _, pod := range pods {
		summary = append(summary, fmt.Sprintf("%s/%s(phase=%s)", pod.namespace, pod.name, pod.phase))
	}
	logf("%s abnormal pods on %s: %s\n",
		bkeFailureArtifactLogPrefix, target.name, strings.Join(summary, ", "))

	for _, pod := range pods {
		args := fmt.Sprintf("describe pod -n %s %s", shellQuote(pod.namespace), shellQuote(pod.name))
		result, err := execKubectlOnGuide(sshExec, target.kubeconfigPath, args)
		if err != nil {
			logf("%s describe pod %s/%s on %s failed: %v\n",
				bkeFailureArtifactLogPrefix, pod.namespace, pod.name, target.name, err)
			continue
		}
		if result.ExitCode != 0 {
			logf("%s describe pod %s/%s on %s failed: %s\n",
				bkeFailureArtifactLogPrefix, pod.namespace, pod.name, target.name, strings.TrimSpace(result.Stderr))
			continue
		}

		fileName := fmt.Sprintf("%s-pod-%s-%s-describe.txt",
			sanitizePathSegment(target.name),
			sanitizePathSegment(pod.namespace),
			sanitizePathSegment(pod.name),
		)
		if err := os.WriteFile(filepath.Join(outputDir, fileName), []byte(result.Stdout), 0644); err != nil {
			logf("%s write describe of %s/%s on %s failed: %v\n",
				bkeFailureArtifactLogPrefix, pod.namespace, pod.name, target.name, err)
			continue
		}
		logf("%s saved describe of %s/%s on %s\n",
			bkeFailureArtifactLogPrefix, pod.namespace, pod.name, target.name)
	}
}

func listAbnormalPodsViaSSH(sshExec executor.Executor, kubeconfigPath string) ([]abnormalPodInfo, error) {
	result, err := execKubectlOnGuide(sshExec, kubeconfigPath, "get pods -A -o json")
	if err != nil {
		return nil, err
	}
	if result.ExitCode != 0 {
		return nil, fmt.Errorf("%s", strings.TrimSpace(result.Stderr))
	}

	var podList struct {
		Items []struct {
			Metadata struct {
				Name      string `json:"name"`
				Namespace string `json:"namespace"`
			} `json:"metadata"`
			Status struct {
				Phase             string                       `json:"phase"`
				ContainerStatuses []abnormalPodContainerStatus `json:"containerStatuses"`
			} `json:"status"`
		} `json:"items"`
	}
	if err := json.Unmarshal([]byte(result.Stdout), &podList); err != nil {
		return nil, err
	}

	pods := make([]abnormalPodInfo, 0)
	for _, item := range podList.Items {
		if !isAbnormalPod(item.Status.Phase, item.Status.ContainerStatuses) {
			continue
		}
		pods = append(pods, abnormalPodInfo{
			namespace: item.Metadata.Namespace,
			name:      item.Metadata.Name,
			phase:     item.Status.Phase,
		})
		if len(pods) >= abnormalPodCaptureLimit {
			break
		}
	}
	return pods, nil
}

func isAbnormalPod(phase string, containerStatuses []abnormalPodContainerStatus) bool {
	if phase != "Running" && phase != "Succeeded" {
		return true
	}
	for _, status := range containerStatuses {
		if !status.Ready {
			return true
		}
	}
	return false
}
