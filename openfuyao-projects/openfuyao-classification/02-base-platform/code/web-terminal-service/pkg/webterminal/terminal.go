/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Package webterminal 提供Web终端相关的功能和服务。
package webterminal

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"strings"
	"time"

	"github.com/gorilla/websocket"
	v1 "k8s.io/api/core/v1"
	apierr "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/remotecommand"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	"openfuyao.com/web-terminal-service/api/v1beta1"
	"openfuyao.com/web-terminal-service/pkg/zlog"
)

// HandleInterface 定义了处理终端交互的接口
type HandleInterface interface {
	// HandleTerminal 处理与指定Pod容器的终端交互
	HandleTerminal(ctx context.Context, namespace, podName, containerName string, conn *websocket.Conn)
	// HandleCusterTerminal 定义 user Pod交互terminal的方法
	HandleCusterTerminal(ctx context.Context, username string, conn *websocket.Conn)
}

const (
	period     = 5 * time.Second
	pingPeriod = 10 * time.Second
)

type execOptions struct {
	namespace     string
	podName       string
	containerName string
	cmd           []string
	stdin         bool
	stdout        bool
	stderr        bool
	tty           bool
	persuo        Persuo
}

type terminaler struct {
	client    kubernetes.Interface
	config    *rest.Config
	MgrClient client.Client
}

// Persuo is an interface that implements io.Reader, io.Writer and remotecommand.TerminalSizeQueue
type Persuo interface {
	io.Reader
	io.Writer
	remotecommand.TerminalSizeQueue
}

// NewTerminal creates a new terminaler instance with the provided client, config, and mgrclient.
func NewTerminal(client kubernetes.Interface, config *rest.Config, mgrclient client.Client) HandleInterface {
	return &terminaler{client: client, config: config, MgrClient: mgrclient}
}

func isPodReady(pod *v1.Pod) bool {
	for _, condition := range pod.Status.Conditions {
		if condition.Type == v1.PodReady && condition.Status == v1.ConditionTrue {
			return true
		}
	}
	return false
}

func (t *terminaler) HandleTerminal(ctx context.Context, namespace, podName,
	containerName string, conn *websocket.Conn) {
	var err error
	terminalWindow := &Window{conn: conn, sizeChan: make(chan remotecommand.TerminalSize), ctx: ctx, terminaler: t}

	supportedShell := t.getShell(ctx, namespace, podName, containerName)
	if supportedShell == "" {
		zlog.LogErrorf("No valid shell found in the container")
		WriteErr := conn.WriteMessage(websocket.TextMessage, []byte("404 LogError:  No valid shell found in the container"))
		if WriteErr != nil {
			zlog.LogErrorf("Websocket write message error: %v", WriteErr)
		}
		terminalWindow.Close("No valid shell found")
		return
	}

	options := execOptions{
		namespace:     namespace,
		podName:       podName,
		containerName: containerName,
		cmd:           []string{supportedShell},
		stdin:         true,
		stdout:        true,
		stderr:        true,
		tty:           true,
		persuo:        terminalWindow,
	}

	err = t.startProcess(ctx, options)

	if err != nil && !errors.Is(err, context.Canceled) {
		zlog.LogErrorf("Shell execution failed: %v", err)
		terminalWindow.Close(err.Error())
		return
	}

	terminalWindow.Close("Process finished")
}

func (t *terminaler) startProcess(ctx context.Context, options execOptions) error {
	exec, err := t.executePodExec(options)
	if err != nil {
		return err
	}

	err = exec.StreamWithContext(ctx, remotecommand.StreamOptions{
		Stdin:             options.persuo,
		Stdout:            options.persuo,
		Stderr:            options.persuo,
		Tty:               options.tty,
		TerminalSizeQueue: options.persuo,
	})

	return err
}

func (t *terminaler) getShell(ctx context.Context, namespace, podName, containerName string) string {
	shells := []string{"bash", "sh"}
	for _, shell := range shells {
		if t.shellExists(ctx, namespace, podName, containerName, shell) {
			return shell
		}
	}
	return ""
}

func (t *terminaler) shellExists(ctx context.Context, namespace, podName, containerName, shell string) bool {
	cmd := []string{"which", shell}
	options := execOptions{
		namespace:     namespace,
		podName:       podName,
		containerName: containerName,
		cmd:           cmd,
		stdin:         false,
		stdout:        true,
		stderr:        true,
		tty:           false,
	}

	exec, err := t.executePodExec(options)
	if err != nil {
		zlog.LogWarnf("Failed to create exec executor for shell check: %v", err)
		return false
	}

	var output bytes.Buffer
	err = exec.StreamWithContext(ctx, remotecommand.StreamOptions{
		Stdout: &output,
		Stderr: os.Stderr,
	})
	if err != nil {
		zlog.LogWarnf("Failed to execute command %s: %v", shell, err)
		return false
	}
	return output.Len() > 0
}

func (t *terminaler) executePodExec(options execOptions) (remotecommand.Executor, error) {
	req := t.client.CoreV1().RESTClient().Post().
		Resource("pods").
		Name(options.podName).
		Namespace(options.namespace).
		SubResource("exec")

	req.VersionedParams(&v1.PodExecOptions{
		Container: options.containerName,
		Command:   options.cmd,
		Stdin:     options.stdin,
		Stdout:    options.stdout,
		Stderr:    options.stderr,
		TTY:       options.tty,
	}, scheme.ParameterCodec)

	exec, err := remotecommand.NewSPDYExecutor(t.config, "POST", req.URL())
	if err != nil {
		zlog.LogWarnf("Failed to create exec executor: %v", err)
		return nil, err
	}

	return exec, nil
}

func (t *terminaler) HandleCusterTerminal(ctx context.Context, username string, conn *websocket.Conn) {
	var err error
	webTerminalTemplate := &v1beta1.WebterminalTemplate{}
	user := fmt.Sprintf("%s-%s", "openfuyao", username)

	err = t.MgrClient.Get(ctx, types.NamespacedName{Name: user, Namespace: UserPodNamespace}, webTerminalTemplate)
	if err != nil {
		zlog.LogInfof("not get user pod: %v", err)
		t.CreateUserPod(ctx, user)
		t.startSessionWithPing(ctx, UserPodNamespace, user, UserContainerName, conn)
		return
	}

	if webTerminalTemplate != nil && webTerminalTemplate.ObjectMeta.DeletionTimestamp.IsZero() {
		zlog.LogInfof("CR already exists and is not being deleted. Skipping creation.")
		t.startSessionWithPing(ctx, UserPodNamespace, user, UserContainerName, conn)
		return
	}

	if !webTerminalTemplate.ObjectMeta.DeletionTimestamp.IsZero() {
		if controllerutil.ContainsFinalizer(webTerminalTemplate, Finalizer) {
			// 删除过程中
			zlog.LogInfof("Resource is being deleted, waiting for finalizer removal.")
			err = wait.PollUntilContextTimeout(ctx, time.Second, time.Minute, false,
				func(ctx context.Context) (done bool, err error) {
					GetErr := t.MgrClient.Get(ctx, types.NamespacedName{Name: user, Namespace: UserPodNamespace}, webTerminalTemplate)
					if GetErr != nil {
						return false, GetErr
					}
					if !controllerutil.ContainsFinalizer(webTerminalTemplate, Finalizer) {
						zlog.LogInfof("Resource deletion has completed.")
						return true, nil
					}
					return false, nil
				})

			if err != nil {
				zlog.LogErrorf("LogError during polling delete CR: %v", err)
				return
			}
		}
	}
	// 等待删除完后创建新的CR
	t.CreateUserPod(ctx, user)
	t.startSessionWithPing(ctx, UserPodNamespace, user, UserContainerName, conn)

}

func (t *terminaler) CreateUserPod(ctx context.Context, user string) { // 定义好user pod模板，创建CR
	webTemplate := template(user)
	kubectlPod := &v1.Pod{}
	err := wait.PollUntilContextTimeout(ctx, period, time.Minute, false,
		func(ctx context.Context) (done bool, err error) {
			err = t.MgrClient.Get(ctx, types.NamespacedName{Name: user, Namespace: "openfuyao-system"}, kubectlPod)
			if err != nil {
				if apierr.IsNotFound(err) {
					creErr := t.MgrClient.Create(ctx, webTemplate)
					if creErr != nil && !apierr.IsAlreadyExists(creErr) {
						zlog.LogInfof("create pod failed %v", err)
						return false, creErr
					}
				}
				zlog.LogInfof("get pod failed %v", err)
				return false, nil
			}
			if kubectlPod.Status.Phase != "Running" || !isPodReady(kubectlPod) {
				zlog.LogInfof("pod is not running! \n")
				return false, nil
			}

			zlog.LogInfof("get pod success!")
			return true, nil
		})

	if err != nil {
		zlog.LogErrorf("LogError creating v1beta1 cluster object : %v", err)
		return
	}

}

func (t *terminaler) startSessionWithPing(ctx context.Context, namespace, podName, containerName string, conn *websocket.Conn) {
	// 定义上下文和 Ping 定时器
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// 定期发送 Ping 消息
	go wait.UntilWithContext(ctx, func(ctx context.Context) {
		if err := conn.WriteControl(websocket.PingMessage, nil, time.Now().Add(pingPeriod)); err != nil {
			zlog.LogErrorf("Failed to send ping message: %v", err)
			cancel()
			_ = conn.Close()
		}
	}, pingSend)

	// 设置 Pong 和 Close 处理器
	conn.SetPongHandler(func(string) error {
		err := conn.SetReadDeadline(time.Now().Add(pongWait)) // 每次收到 Pong 更新读取超时
		if err != nil {
			zlog.LogWarnf("Pong read out time : %v", err)
		}
		return nil
	})
	conn.SetCloseHandler(func(code int, text string) error {
		zlog.LogInfof("WebSocket connection closed: code %d, %s", code, text)
		// 取消上下文，停止后台任务
		cancel()
		// 确保发送 Close 帧
		err := conn.WriteControl(websocket.CloseMessage,
			websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""),
			time.Now().Add(WaitWirte))
		if err != nil {
			zlog.LogErrorf("LogError sending close frame: %v", err)
		}

		return nil
	})

	t.HandleTerminal(ctx, namespace, podName, containerName, conn)
}

func getImagePath(filePath string) (string, error) {
	content, err := ioutil.ReadFile(filePath)
	if err != nil {
		return "", err
	}
	return string(content), nil
}

func template(user string) *v1beta1.WebterminalTemplate {
	imagePath, err := getImagePath(ImagePath)
	if err != nil {
		zlog.LogFatalf("Failed to read image path: %v", err)
	}
	zlog.LogInfof("Creating pod for user: %s with image: %s \n", user, imagePath)
	imagePath = strings.TrimSpace(imagePath)
	pod := createPodTemplate(user, imagePath)

	return &v1beta1.WebterminalTemplate{
		ObjectMeta: metav1.ObjectMeta{
			Name:      user,
			Namespace: UserPodNamespace,
		},
		Spec: v1beta1.WebterminalTemplateSpec{
			PodTemplate: *pod,
		},
	}
}

func createPodTemplate(user, imagePath string) *v1beta1.PodTemplate {
	podUser := int64(65532)

	return &v1beta1.PodTemplate{
		ObjectMeta: v1beta1.PodTemplateObjectMeta{
			Name:      user,
			Namespace: UserPodNamespace,
		},
		Spec: v1beta1.PodTemplateSpec{
			InitContainers: createInitContainers(imagePath),
			Containers:     createMainContainers(imagePath, podUser),
			Volumes:        createVolumes(),
		},
	}
}

func createInitContainers(imagePath string) []v1.Container {
	return []v1.Container{
		{
			Name:    "init-kubeconfig",
			Image:   imagePath,
			Command: []string{"sh", "-c"},
			Args: []string{
				`mkdir -p /mnt/.kube && cp /etc/kubernetes/config /mnt/.kube/config && 
				chmod 644 /mnt/.kube/config && export KUBECONFIG=/mnt/.kube/config && 
				kubectl config set-context --current --namespace=default`,
			},
			VolumeMounts: getInitContainerVolumeMounts(),
			SecurityContext: &v1.SecurityContext{
				RunAsUser: new(int64),
			},
		},
	}
}

func getInitContainerVolumeMounts() []v1.VolumeMount {
	return []v1.VolumeMount{
		{
			Name:      "kubeconfig",
			MountPath: "/mnt/.kube",
		},
		{
			Name:      "config-source",
			MountPath: "/etc/kubernetes",
		},
	}
}

func createMainContainers(imagePath string, podUser int64) []v1.Container {
	return []v1.Container{
		{
			Name:            UserContainerName,
			Image:           imagePath,
			ImagePullPolicy: v1.PullIfNotPresent,
			Env: []v1.EnvVar{
				{
					Name:  "KUBECONFIG",
					Value: "/mnt/.kube/config",
				},
			},
			VolumeMounts: getMainContainerVolumeMounts(),
			SecurityContext: &v1.SecurityContext{
				RunAsUser:  &podUser,
				RunAsGroup: new(int64),
			},
		},
	}
}

func getMainContainerVolumeMounts() []v1.VolumeMount {
	return []v1.VolumeMount{
		{
			Name:      "kubeconfig",
			MountPath: "/mnt/.kube",
		},
	}
}

func createVolumes() []v1.Volume {
	return []v1.Volume{
		{
			Name: "kubeconfig",
			VolumeSource: v1.VolumeSource{
				EmptyDir: &v1.EmptyDirVolumeSource{},
			},
		},
		{
			Name: "config-source",
			VolumeSource: v1.VolumeSource{
				HostPath: &v1.HostPathVolumeSource{
					Path: "/root/.kube",
				},
			},
		},
	}
}
