/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package kubelet

import (
	"bytes"
	_ "embed"
	"fmt"
	"os"
	"strings"
	"text/template"

	"github.com/blang/semver/v4"
	"github.com/shirou/gopsutil/v3/host"

	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	bkenet "gopkg.openfuyao.cn/cluster-api-provider-bke/common/utils/net"
	utils "gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
)

type RunKubeletCommand struct {
	ContainerRuntime string
	ContainerName    string
	KubeletImage     string
	PauseImage       string
	HostIP           string
	KubeConfigPath   string
	ExtraVolumes     []string
	ExtraKubeletArgs []string
	k8sVersion       semver.Version
	rootDirPath      string
}

const (
	// RwxRxRx is the permission of the directory
	RwxRxRx = 0755
)

var (
	//go:embed tmpl/kubelet.sh.tmpl
	kubeletScript string

	defaultKubeletVolumes = []string{
		//"/var/lib/etcd:/var/lib/etcd",
		"/etc/os-release:/etc/os-release",
		"/etc/kubernetes:/etc/kubernetes",
		"/etc/localtime:/etc/localtime",
		"/etc/ssl/certs:/etc/ssl/certs",
		"/etc/sysconfig/network-scripts:/etc/sysconfig/network-scripts",
		"/etc/resolv.conf:/etc/resolv.conf",
		"/run:/run",
		"/var/run:/var/run:rw",
		"/sys:/sys",
		"/proc:/proc",
		"/dev:/dev",
		"/lib/modules:/lib/modules",
		"/usr/libexec/kubernetes:/usr/libexec/kubernetes",
		//"/var/lib/docker:/var/lib/docker:rw,rslave",
		"/var/lib/calico:/var/lib/calico",
		"/var/lib/cni:/var/lib/cni",
		"/var/lib/lxc:/var/lib/lxc",
		"/var/log/pods:/var/log/pods",
		"/var/lib/kubelet:/var/lib/kubelet:shared",
		"/etc/kubernetes/kubelet-config.yaml:/var/lib/kubelet/config.yaml",
		//"/var/lib/containerd:/var/lib/containerd",
		"/var/log/containers:/var/log/containers",
		"/etc/cni:/etc/cni:rw",
		"/opt/cni:/opt/cni:rw",
		"/opt/fabric:/opt/fabric",
	}

	defaultKubeletArgs = []string{
		"--v=0",
		fmt.Sprintf("--config=%s", utils.GetKubeletConfPath()),
	}

	V12317, _ = semver.ParseTolerant("v1.23.17")
	V124, _   = semver.ParseTolerant("v1.24")
	V127, _   = semver.ParseTolerant("v1.27")
	V121, _   = semver.ParseTolerant("v1.21")
)

// NewRunKubeletCommand 改为采用二进制方式部署，这里待删除
func NewRunKubeletCommand() *RunKubeletCommand {
	return &RunKubeletCommand{
		ContainerRuntime: "containerd",
		ContainerName:    "kubelet",
		KubeletImage:     "deploy.bocloud.k8s:40443/kubernetes/kubelet:v1.23.17",
		KubeConfigPath:   "/etc/kubernetes/admin.conf",
		ExtraVolumes:     []string{},
		ExtraKubeletArgs: []string{},
		k8sVersion:       V12317,
		rootDirPath:      bkeinit.DefaultKubeletRootDir,
	}
}

func (k *RunKubeletCommand) validate() error {
	if k.KubeletImage == "" {
		return fmt.Errorf("kubelet image is empty")
	}

	if k.PauseImage == "" {
		return fmt.Errorf("pause image is empty")
	}

	if k.HostIP == "" {
		return fmt.Errorf("host ip is empty")
	}

	if k.k8sVersion.LT(V121) {
		return fmt.Errorf("kubernetes version is empty or less than v1.21")
	}

	if k.ContainerRuntime != "containerd" && k.ContainerRuntime != "docker" {
		return fmt.Errorf("container runtime %q is not supported", k.ContainerRuntime)
	}

	return nil
}

func (k *RunKubeletCommand) SetK8sVersion(version string) *RunKubeletCommand {
	v, _ := semver.ParseTolerant(version)
	k.k8sVersion = v
	return k
}

func (k *RunKubeletCommand) SetKubeletImage(image string) *RunKubeletCommand {
	k.KubeletImage = image
	return k
}

func (k *RunKubeletCommand) SetContainerRuntime(runtime string) *RunKubeletCommand {
	k.ContainerRuntime = runtime
	return k
}

func (k *RunKubeletCommand) SetKubeConfigPath(path string) *RunKubeletCommand {
	k.KubeConfigPath = path
	return k
}

func (k *RunKubeletCommand) SetContainerName(name string) *RunKubeletCommand {
	k.ContainerName = name
	return k
}

func (k *RunKubeletCommand) SetExtraVolumes(volumes []string) *RunKubeletCommand {
	k.ExtraVolumes = volumes
	return k
}

func (k *RunKubeletCommand) SetExtraKubeletArgs(args []string) *RunKubeletCommand {
	k.ExtraKubeletArgs = args
	return k
}

func (k *RunKubeletCommand) SetPauseImage(image string) *RunKubeletCommand {
	k.PauseImage = image
	return k
}

func (k *RunKubeletCommand) SetHostIP(ip string) *RunKubeletCommand {
	k.HostIP = ip
	return k
}

func (k *RunKubeletCommand) SetRootDirPath(path string) *RunKubeletCommand {
	k.rootDirPath = path
	return k
}

func (k *RunKubeletCommand) Command() (string, []string, error) {
	if err := k.validate(); err != nil {
		return "", nil, err
	}
	cmd := k.getCmd()
	command := append([]string{}, k.getCmdArgs()...)
	command = append(command, k.getVolumeArgs()...)
	command = append(command, k.KubeletImage)
	command = append(command, k.getKubeletArgs()...)
	command = utils.SliceRemoveString(command, "")
	return cmd, command, nil
}

func (k *RunKubeletCommand) getCmd() string {
	switch k.ContainerRuntime {
	case "docker":
		return "docker"
	case "containerd":
		return "nerdctl"
	default:
		return "nerdctl"
	}
}

func (k *RunKubeletCommand) getCmdArgs() []string {
	containerName := fmt.Sprintf("--name=%s", k.ContainerName)
	netHost := "--net=host"
	pidHost := "--pid=host"
	userRoot := "--user=root"
	privileged := "--privileged"
	restartAlways := "--restart=always"
	run := "run"
	detach := "--detach"
	namespace := "--namespace=k8s.io"
	insecure := "--insecure-registry"

	switch k.ContainerRuntime {
	case "docker":
		return []string{run, detach, containerName, netHost, pidHost, userRoot, privileged, restartAlways}
	case "containerd":
		return []string{namespace, insecure, run, detach, containerName, netHost, pidHost, userRoot, privileged, restartAlways}
	default:
		return []string{namespace, insecure, run, detach, containerName, netHost, pidHost, userRoot, privileged, restartAlways}
	}
}

func (k *RunKubeletCommand) getVolumeArgs() []string {

	platform := "centos"
	h, _, _, err := host.PlatformInformation()
	if err == nil {
		platform = h
	}

	preVolumes := append(defaultKubeletVolumes, k.ExtraVolumes...)
	uniqueVolumes := utils.UniqueStringSlice(preVolumes)
	var volumes []string

	rootDirMountFlag := false
	for _, volume := range uniqueVolumes {
		if platform == "kylin" && strings.HasPrefix(volume, "/proc:") {
			continue
		}
		args := strings.Split(volume, ":")
		if args[0] == k.rootDirPath && !strings.HasSuffix(volume, "shared") {
			volume = fmt.Sprintf("%s:shared", volume)
			rootDirMountFlag = true
		}
		volumes = append(volumes, fmt.Sprintf("--volume %s", volume))
	}
	if !rootDirMountFlag {
		volumes = append(volumes, fmt.Sprintf("--volume %s:%s:shared", k.rootDirPath, k.rootDirPath))
	}

	// containerd 需要额外挂载 /sys/fs/cgroup 有些操作系统直接挂载/sys 会可能看不到 cgroup里的部分内容，导致kubelet启动报错 （UBOC-16778）
	if k.ContainerRuntime == "containerd" {
		volumes = append(volumes, "--volume /sys/fs/cgroup:/sys/fs/cgroup")
	}

	return utils.UniqueStringSlice(volumes)
}

func (k *RunKubeletCommand) getKubeletArgs() []string {
	preArgs := defaultKubeletArgs

	// 固定变参
	kubeConfig := fmt.Sprintf("--kubeconfig=%s", k.KubeConfigPath)
	hostNameOverride := fmt.Sprintf("--hostname-override=%s", utils.HostName())
	NodeIP := fmt.Sprintf("--node-ip=%s", k.HostIP)
	rootDir := fmt.Sprintf("--root-dir=%s", k.rootDirPath)
	preArgs = append(preArgs, kubeConfig, hostNameOverride, NodeIP, rootDir)

	// containerd 需要额外添加如下参数 在1.27之前
	if k.ContainerRuntime == "containerd" && k.k8sVersion.LT(V127) {
		preArgs = append(preArgs, "--container-runtime=remote")
		preArgs = append(preArgs, "--container-runtime-endpoint=unix:///run/containerd/containerd.sock")
	}
	// docker 需要额外添加如下参数 在1.24以后
	if k.ContainerRuntime == "docker" && k.k8sVersion.GE(V124) {
		preArgs = append(preArgs, "--container-runtime-endpoint=unix:///var/run/cri-dockerd.sock")
	}
	// 小于1.24需要额外添加如下参数
	if k.k8sVersion.LT(V124) {
		preArgs = append(preArgs, "--network-plugin=cni")
		preArgs = append(preArgs, "--cni-bin-dir=/opt/cni/bin")
		preArgs = append(preArgs, "--cni-conf-dir=/etc/cni/net.d")
	}

	// 小于1.27需要额外添加如下参数
	if k.k8sVersion.LT(V127) {
		preArgs = append(preArgs, "--pod-infra-container-image="+k.PauseImage)
	}

	preArgs = append(preArgs, k.ExtraKubeletArgs...)

	uniqueArgs := utils.UniqueStringSlice(preArgs)

	uniqueArgs = append([]string{"kubelet"}, uniqueArgs...)
	return uniqueArgs
}

func (k *RunKubeletCommand) ExportKubeletScript(refresh bool) error {
	if !refresh && utils.Exists(utils.GetKubeletScriptPath()) {
		return nil
	}
	if err := k.validate(); err != nil {
		return fmt.Errorf("validate kubelet command failed: %w", err)
	}

	// get current container runtime and command
	currenContainerRuntime := k.ContainerRuntime
	cmd, args, err := k.Command()
	if err != nil {
		return fmt.Errorf("generate %q run kubelet command failed: %w", currenContainerRuntime, err)
	}
	currentCommand := fmt.Sprintf("%s %s", cmd, strings.Join(args, " "))

	dockerCommand, containerdCommand, isDocker := "", "", currenContainerRuntime == "docker"

	if isDocker {
		dockerCommand = currentCommand
		k.SetContainerRuntime("containerd")
		cmd, args, err = k.Command()
		if err != nil {
			return fmt.Errorf("generate %q run kubelet command failed: %w", currenContainerRuntime, err)
		}
		containerdCommand = fmt.Sprintf("%s %s", cmd, strings.Join(args, " "))
	} else {
		containerdCommand = currentCommand
		k.SetContainerRuntime("docker")
		cmd, args, err = k.Command()
		if err != nil {
			return fmt.Errorf("generate %q run kubelet command failed: %w", currenContainerRuntime, err)
		}
		dockerCommand = fmt.Sprintf("%s %s", cmd, strings.Join(args, " "))
	}

	param := map[string]string{"containerName": k.ContainerName, "dockerCommand": dockerCommand, "containerdCommand": containerdCommand}
	t, err := template.New("kubeletScript").Parse(kubeletScript)
	if err != nil {
		return err
	}
	if !utils.Exists(utils.KubernetesDir) {
		if err := os.MkdirAll(utils.KubernetesDir, RwxRxRx); err != nil {
			return fmt.Errorf("create %q directory failed: %w", utils.KubernetesDir, err)
		}
	}
	writer, err := os.OpenFile(utils.GetKubeletScriptPath(), os.O_WRONLY|os.O_CREATE|os.O_TRUNC, RwxRxRx)
	defer writer.Close()
	if err != nil {
		return fmt.Errorf("open kubelet script file failed: %w", err)
	}
	if err := t.Execute(writer, param); err != nil {
		return fmt.Errorf("execute kubelet script template failed: %w", err)
	}
	return nil
}

// buildKubeletCommand builds a RunKubeletCommand with the specified container runtime
func buildKubeletCommand(config map[string]string, runtime string) *RunKubeletCommand {
	return NewRunKubeletCommand().
		SetContainerRuntime(runtime).
		SetKubeletImage(config["kubeletImage"]).
		SetPauseImage(config["pauseImage"]).
		SetKubeConfigPath(config["kubeconfigPath"]).
		SetExtraVolumes(strings.Split(config["extraVolumes"], ";")).
		SetExtraKubeletArgs(strings.Split(config["extraArgs"], ";")).
		SetK8sVersion(config["kubernetesVersion"]).
		SetHostIP(config["hostIP"]).
		SetRootDirPath(config["dataRootDir"])
}

// newKubeletCommand creates a kubelet command with the specified container runtime
func newKubeletCommand(config map[string]string, runtime string) (string, []string, error) {
	return buildKubeletCommand(config, runtime).Command()
}

// runKubeletCommandFromTemplate executes a kubelet command from a template
func runKubeletCommandFromTemplate(config map[string]string, templateName, templateContent string) (string, []string, error) {
	externalHost, err := bkenet.GetExternalIP()
	if err != nil {
		return "", nil, err
	}

	if config != nil {
		config["externalHost"] = externalHost
		config["hostName"] = utils.HostName()
	}

	tpl, err := template.New(templateName).Parse(templateContent)
	if err != nil {
		return "", nil, err
	}
	var buf bytes.Buffer
	if err := tpl.Execute(&buf, config); err != nil {
		return "", nil, err
	}
	t := strings.ReplaceAll(buf.String(), "\n", " ")
	t = strings.ReplaceAll(t, "\\", "")
	t = strings.ReplaceAll(t, "\r", "")
	r := strings.Split(t, " ")
	var res []string
	for _, v := range r {
		if v != "" {
			res = append(res, v)
		}
	}
	return res[0], res[1:], nil
}
