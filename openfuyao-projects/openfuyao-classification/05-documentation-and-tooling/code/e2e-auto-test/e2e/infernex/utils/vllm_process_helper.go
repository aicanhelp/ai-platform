package utils

import (
	"bytes"
	"context"
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/remotecommand"
)

const (
	// PDRoleLabelKey 标识 prefill/decode Pod 的 label。
	PDRoleLabelKey = "openfuyao.com/pdRole"
	PrefillPDRole  = "prefill"
	DecodePDRole   = "decode"

	PrefillEngineContainer = "prefill-engine"
	DecodeEngineContainer  = "decode-engine"
)

// vllmServeProcessListShell 在容器内通过 ps 查找 vllm serve 进程完整命令行。
const vllmServeProcessListShell = `ps -ef 2>/dev/null | grep 'vllm serve' | grep -v grep`

// PDEngineTarget 步骤 1 定位结果：Pod 与 engine 容器名。
type PDEngineTarget struct {
	Pod       corev1.Pod
	Container string
}

// 在 namespace 中按 openfuyao.com/pdRole 查找 Running/Ready Pod，并确认 engine 容器存在。
func ListPDEngineTargets(clientset *kubernetes.Clientset, ctx context.Context, namespace, pdRole string) ([]PDEngineTarget, error) {
	role := strings.ToLower(strings.TrimSpace(pdRole))
	container, err := engineContainerForPDRole(role)
	if err != nil {
		return nil, err
	}

	labelSelector := fmt.Sprintf("%s=%s", PDRoleLabelKey, role)
	podList, err := clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{LabelSelector: labelSelector})
	if err != nil {
		return nil, fmt.Errorf("list pods (labelSelector=%q): %w", labelSelector, err)
	}

	var targets []PDEngineTarget
	for i := range podList.Items {
		pod := &podList.Items[i]
		if pod.Status.Phase != corev1.PodRunning {
			continue
		}
		if !podAllContainersReady(pod) {
			continue
		}
		if !podHasContainer(pod, container) {
			return nil, fmt.Errorf("pod %s 缺少容器 %q", pod.Name, container)
		}
		targets = append(targets, PDEngineTarget{Pod: *pod, Container: container})
	}
	if len(targets) == 0 {
		return nil, fmt.Errorf("namespace %s 中未找到 label %s=%s 且 Running/Ready 的 Pod",
			namespace, PDRoleLabelKey, role)
	}
	return targets, nil
}

// 在指定 Pod 的 engine 容器内执行 ps，获取 vllm serve 完整命令行。
func GetVLLMServeCommand(
	restConfig *rest.Config,
	clientset *kubernetes.Clientset,
	ctx context.Context,
	namespace string,
	target PDEngineTarget,
) (string, error) {
	out, err := execShellInPod(restConfig, clientset, ctx, namespace, target.Pod.Name, target.Container, vllmServeProcessListShell)
	if err != nil {
		return "", fmt.Errorf("pod %s container %s: exec ps: %w", target.Pod.Name, target.Container, err)
	}
	out = strings.TrimSpace(out)
	if out == "" {
		return "", fmt.Errorf("pod %s container %s: 未找到 vllm serve 进程", target.Pod.Name, target.Container)
	}

	// ps 可能返回多行，取首条含 vllm serve 的行作为完整命令。
	var cmdline string
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		if strings.Contains(line, "vllm serve") {
			cmdline = line
			break
		}
	}
	if cmdline == "" {
		return "", fmt.Errorf("pod %s container %s: ps 输出中无 vllm serve 行:\n%s",
			target.Pod.Name, target.Container, out)
	}
	return cmdline, nil
}

// GetContainerEnvVarValue 在 engine 容器内执行 echo $ENV_NAME，返回环境变量值。
func GetContainerEnvVarValue(
	restConfig *rest.Config,
	clientset *kubernetes.Clientset,
	ctx context.Context,
	namespace string,
	target PDEngineTarget,
	envName string,
) (string, error) {
	shellCmd := fmt.Sprintf("echo $%s", envName)
	out, err := execShellInPod(restConfig, clientset, ctx, namespace, target.Pod.Name, target.Container, shellCmd)
	if err != nil {
		return "", fmt.Errorf("pod %s container %s: echo %s: %w", target.Pod.Name, target.Container, envName, err)
	}
	return strings.TrimSpace(out), nil
}

// 断言 vllm 完整命令包含 expected 片段。
func VerifyVLLMCommandContainsExtraArgs(cmdline string, expected []string) error {
	for _, sub := range expected {
		if !cmdlineContains(cmdline, sub) {
			return fmt.Errorf("vllm 命令行未包含 %q，命令:\n%s", sub, cmdline)
		}
	}
	return nil
}

func engineContainerForPDRole(pdRole string) (string, error) {
	switch pdRole {
	case PrefillPDRole:
		return PrefillEngineContainer, nil
	case DecodePDRole:
		return DecodeEngineContainer, nil
	default:
		return "", fmt.Errorf("不支持的 pdRole %q，仅支持 prefill/decode", pdRole)
	}
}

func podHasContainer(pod *corev1.Pod, containerName string) bool {
	for _, c := range pod.Spec.Containers {
		if c.Name == containerName {
			return true
		}
	}
	return false
}

func podAllContainersReady(pod *corev1.Pod) bool {
	if len(pod.Status.ContainerStatuses) == 0 {
		return false
	}
	for _, cs := range pod.Status.ContainerStatuses {
		if !cs.Ready {
			return false
		}
	}
	return true
}

func execShellInPod(
	restConfig *rest.Config,
	clientset *kubernetes.Clientset,
	ctx context.Context,
	namespace, podName, container, shellCmd string,
) (string, error) {
	req := clientset.CoreV1().RESTClient().Post().
		Resource("pods").
		Name(podName).
		Namespace(namespace).
		SubResource("exec").
		VersionedParams(&corev1.PodExecOptions{
			Container: container,
			Command:   []string{"/bin/sh", "-c", shellCmd},
			Stdout:    true,
			Stderr:    true,
		}, scheme.ParameterCodec)

	exec, err := remotecommand.NewSPDYExecutor(restConfig, "POST", req.URL())
	if err != nil {
		return "", err
	}

	var stdout, stderr bytes.Buffer
	err = exec.StreamWithContext(ctx, remotecommand.StreamOptions{
		Stdout: &stdout,
		Stderr: &stderr,
	})
	out := strings.TrimSpace(stdout.String())
	if err != nil {
		return out, fmt.Errorf("%w (stderr: %s)", err, strings.TrimSpace(stderr.String()))
	}
	if out == "" && stderr.Len() > 0 {
		out = strings.TrimSpace(stderr.String())
	}
	return out, nil
}

func cmdlineContains(cmdline, sub string) bool {
	if strings.Contains(cmdline, sub) {
		return true
	}
	compact := func(s string) string {
		return strings.Map(func(r rune) rune {
			if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
				return -1
			}
			return r
		}, s)
	}
	return strings.Contains(compact(cmdline), compact(sub))
}
