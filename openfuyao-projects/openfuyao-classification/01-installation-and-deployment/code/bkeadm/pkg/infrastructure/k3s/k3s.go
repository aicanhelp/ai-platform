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

package k3s

import (
	"bytes"
	"context"
	_ "embed"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"text/template"
	"time"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/mount"
	"github.com/docker/go-connections/nat"
	bkecommon "gopkg.openfuyao.cn/cluster-api-provider-bke/common"
	configinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	econd "gopkg.openfuyao.cn/bkeadm/pkg/executor/containerd"
	"gopkg.openfuyao.cn/bkeadm/pkg/executor/docker"
	"gopkg.openfuyao.cn/bkeadm/pkg/executor/exec"
	"gopkg.openfuyao.cn/bkeadm/pkg/executor/k8s"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

var (
	//go:embed registries.yaml
	registries string
	//go:embed generate-custom-ca-certs.sh
	k3sCertScript string
	//go:embed core.conf
	k3sCoreScript string
	k3sImage      = utils.DefaultLocalK3sRegistry
	k3sPause      = utils.DefaultK3sPause
	k3sCoredns    string
)

const (
	// DefaultK3sDataDir k3s证书相关数据路径
	DefaultK3sDataDir  = "/var/lib/rancher/k3s"
	waitConfigInterval = 2
	waitInterval       = 5   // 每次检查的间隔（5秒）
	waitTimeout        = 300 // 总超时时间（5分钟）

	// k3s kubeconfig 相关常量
	k3sKubeconfigPath   = "/etc/rancher/k3s/k3s.yaml"
	kubeconfigReadRetry = 5
	kubeconfigReadDelay = 2 // 秒
	k8sClientRetryCount = 10
	k8sClientRetryDelay = 6 // 秒
	nodeReadyRetryCount = 10
	nodeReadyRetryDelay = 3 // 秒

	// 引导集群容器日志目录，与主机同路径 bind mount，
	// 使 Pod 日志在主机可直接查看、可事后审计
	K3sPodLogDir       = "/var/log/pods"
	K3sContainerLogDir = "/var/log/containers"
)

// Config represents the K3s startup configuration parameters
type Config struct {
	OnlineImage    string // 在线安装使用的镜像
	OtherRepo      string // 其他镜像仓库地址
	OtherRepoIP    string // 其他镜像仓库 IP
	HostIP         string // 主机 IP
	ImageRepo      string // 镜像仓库域名
	ImageRepoPort  string // 镜像仓库端口
	KubernetesPort string // Kubernetes API 端口
	ConsolePort    string // BKE Console 宿主机端口
}

// EnsureDirExists ensures the specified directory exists, creating it if necessary
func EnsureDirExists(dir string) error {
	if !utils.Exists(dir) {
		err := os.MkdirAll(dir, utils.DefaultDirPermission)
		if err != nil {
			return err
		}
	}
	return nil
}

// EnsureK3sLogDirs 确保主机上存在 k3s 容器日志挂载目录，
// 引导集群容器日志通过 bind mount 落盘主机，便于直接查看与事后审计
func EnsureK3sLogDirs() error {
	if err := EnsureDirExists(K3sPodLogDir); err != nil {
		return err
	}
	return EnsureDirExists(K3sContainerLogDir)
}

// readKubeconfig 读取 k3s kubeconfig 文件，支持重试
func readKubeconfig() ([]byte, error) {
	var result []byte
	var err error
	for i := 0; i < kubeconfigReadRetry; i++ {
		result, err = os.ReadFile(k3sKubeconfigPath)
		if err != nil {
			time.Sleep(kubeconfigReadDelay * time.Second)
			log.Debug("Failed to get kubeconfig, retrying...")
			continue
		}
		break
	}
	if len(result) == 0 {
		return nil, errors.New("failed to get k3s kubeconfig file")
	}
	return result, nil
}

// processKubeconfig 处理 kubeconfig 并写入到用户目录和 k3s 配置目录
func processKubeconfig(hostIP, kubernetesPort string, kubeconfigData []byte) (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("failed to get user home directory: %w", err)
	}
	kubeDir := fmt.Sprintf("%s/.kube", home)
	if err := os.MkdirAll(kubeDir, utils.DefaultDirPermission); err != nil {
		return "", fmt.Errorf("failed to create kube directory: %w", err)
	}
	kubeconfigPath := fmt.Sprintf("%s/config", kubeDir)
	kubeconfigContent := strings.Replace(
		string(kubeconfigData), "127.0.0.1:36443", fmt.Sprintf("%s:%s", hostIP, kubernetesPort), 1)
	if err := os.WriteFile(kubeconfigPath, []byte(kubeconfigContent), utils.SecureFilePermission); err != nil {
		return "", err
	}
	if err := os.Remove(k3sKubeconfigPath); err != nil {
		log.Warnf("failed to remove original k3s.yaml: %v", err)
	}
	if err := os.WriteFile(k3sKubeconfigPath, []byte(kubeconfigContent), utils.DefaultFilePermission); err != nil {
		log.Warnf("failed to write k3s.yaml: %v, please run export KUBECONFIG=%s", err, kubeconfigPath)
	}
	return kubeconfigContent, nil
}

// waitForK8sClient 等待 Kubernetes 客户端可用
func waitForK8sClient(kubeconfigPath string) error {
	log.Info("Waiting for the cluster to start...")
	var err error
	for i := 0; i < k8sClientRetryCount; i++ {
		global.K8s, err = k8s.NewKubernetesClient(kubeconfigPath)
		if err != nil {
			time.Sleep(k8sClientRetryDelay * time.Second)
			continue
		}
		break
	}
	if global.K8s == nil {
		return errors.New("failed to connect to kubernetes")
	}
	return nil
}

// waitForNodeReady 等待节点就绪
func waitForNodeReady() error {
	log.Info("Waiting for cluster Ready...")
	clientset := global.K8s.GetClient()
	for i := 0; i < nodeReadyRetryCount; i++ {
		node, err := clientset.CoreV1().Nodes().Get(context.Background(), utils.LocalKubernetesName, metav1.GetOptions{})
		if err != nil {
			time.Sleep(nodeReadyRetryDelay * time.Second)
			continue
		}
		if len(node.Spec.Taints) > 1 {
			time.Sleep(nodeReadyRetryDelay * time.Second)
			continue
		}
		_, err = clientset.CoreV1().Namespaces().Get(context.Background(), "kube-system", metav1.GetOptions{})
		if err != nil {
			time.Sleep(nodeReadyRetryDelay * time.Second)
			continue
		}
		break
	}
	return nil
}

// createKubeconfigSecret 在 Kubernetes 集群中创建 kubeconfig Secret
func createKubeconfigSecret(kubeconfigContent string) error {
	clientset := global.K8s.GetClient()
	_, err := clientset.CoreV1().Secrets(metav1.NamespaceSystem).Create(context.Background(), &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "localkubeconfig",
			Namespace: metav1.NamespaceSystem,
		},
		StringData: map[string]string{
			"config": kubeconfigContent,
		},
	}, metav1.CreateOptions{})
	return err
}

// setupKubeconfigAndWaitCluster 设置 kubeconfig 并等待集群就绪
func setupKubeconfigAndWaitCluster(hostIP, kubernetesPort string) error {
	kubeconfigData, err := readKubeconfig()
	if err != nil {
		return err
	}
	kubeconfigContent, err := processKubeconfig(hostIP, kubernetesPort, kubeconfigData)
	if err != nil {
		return err
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return fmt.Errorf("failed to get user home directory: %w", err)
	}
	kubeconfigPath := fmt.Sprintf("%s/.kube/config", home)
	if err := waitForK8sClient(kubeconfigPath); err != nil {
		return err
	}
	if err := waitForNodeReady(); err != nil {
		return err
	}
	if err := createKubeconfigSecret(kubeconfigContent); err != nil {
		return err
	}
	log.Info("The local Kubernetes startup succeeded")
	return nil
}

// prepareK3sImages 准备 k3s 镜像地址
func prepareK3sImages(onlineImage, otherRepo, imageRepoPort, imageRepo, localImage string) {
	localK3sImagePath := fmt.Sprintf("127.0.0.1:%s/%s/%s", imageRepoPort, bkecommon.ImageRegistryKubernetes, utils.DefaultLocalK3sRegistry)
	localK3sPausePath := fmt.Sprintf("%s:443/%s/%s", imageRepo, bkecommon.ImageRegistryKubernetes, utils.DefaultK3sPause)
	if localImage != "" {
		k3sImage = localK3sImagePath
		k3sPause = localK3sPausePath
	} else if otherRepo != "" {
		k3sImage = fmt.Sprintf("%s%s", otherRepo, utils.DefaultLocalK3sRegistry)
		k3sPause = fmt.Sprintf("%s%s", otherRepo, utils.DefaultK3sPause)
	} else if onlineImage == "" {
		k3sImage = localK3sImagePath
		k3sPause = localK3sPausePath
	} else {
		k3sImage = fmt.Sprintf("%s/%s", utils.DefaultThirdMirror, utils.DefaultLocalK3sRegistry)
		k3sPause = fmt.Sprintf("%s/%s", utils.DefaultThirdMirror, utils.DefaultK3sPause)
	}
}

// getImageRepoIP 获取镜像仓库 IP 地址
func getImageRepoIP(otherRepo, otherRepoIp, hostIP, imageRepo, localImagePath string) (string, string) {
	repo := fmt.Sprintf("%s:%s", imageRepo, "443")
	imageRepoIP := hostIP
	if otherRepo != "" && strings.Contains(otherRepo, imageRepo) && localImagePath == "" {
		imageRepoIP = otherRepoIp
		repo = strings.Split(otherRepo, "/")[0]
	} else {
		imageRepoInfo, err := econd.ContainerInspect(utils.LocalImageRegistryName)
		if err == nil && len(imageRepoInfo.Id) > 0 {
			imageRepoIP = imageRepoInfo.NetworkSettings.IPAddress
		}
	}
	return repo, imageRepoIP
}

// getImageRepoIPWithDocker 使用 Docker API 获取镜像仓库 IP 地址
func getImageRepoIPWithDocker(otherRepo, otherRepoIp, hostIP, imageRepo string) (string, string) {
	repo := fmt.Sprintf("%s:%s", imageRepo, "443")
	imageRepoIP := hostIP
	if otherRepo != "" && strings.Contains(otherRepo, configinit.DefaultImageRepo) {
		imageRepoIP = otherRepoIp
		repo = strings.Split(otherRepo, "/")[0]
	} else {
		client := global.Docker.GetClient()
		imageRepoInfo, err := client.ContainerInspect(context.Background(), utils.LocalImageRegistryName)
		if err == nil {
			imageRepoIP = imageRepoInfo.NetworkSettings.IPAddress
		}
	}
	return repo, imageRepoIP
}

// generateRegistriesConfig 生成 registries 配置文件
func generateRegistriesConfig(repo, k3sConfig string) error {
	tmpl0, err := template.New("registries").Parse(registries)
	if err != nil {
		return err
	}
	buf0 := new(bytes.Buffer)
	if err = tmpl0.Execute(buf0, map[string]string{"repo": repo}); err != nil {
		return err
	}
	err = os.Remove(k3sConfig + "/registries.yaml")
	if err != nil && !os.IsNotExist(err) {
		log.Warnf("failed to remove original registries.yaml: %v", err)
	}
	return os.WriteFile(k3sConfig+"/registries.yaml", buf0.Bytes(), utils.DefaultFilePermission)
}

// StartK3sWithContainerd starts K3s cluster with containerd runtime
func StartK3sWithContainerd(cfg Config, localImage string) error {
	if isKubernetesAvailable() {
		log.Info("A kubernetes cluster already exists.")
		return nil
	}
	prepareK3sImages(cfg.OnlineImage, cfg.OtherRepo, cfg.ImageRepoPort, cfg.ImageRepo, localImage)
	if err := econd.EnsureImageExists(k3sImage); err != nil {
		return err
	}
	if _, exist := econd.ContainerExists(utils.LocalKubernetesName); exist {
		if err := econd.ContainerRemove(utils.LocalKubernetesName); err != nil {
			return fmt.Errorf("remove container %s failed before start: %w", utils.LocalKubernetesName, err)
		}
	}
	k3sConfigPath := "/etc/rancher/k3s"
	if err := EnsureDirExists(k3sConfigPath); err != nil {
		return err
	}
	if err := EnsureK3sLogDirs(); err != nil {
		return err
	}
	if err := customCA(); err != nil {
		return err
	}
	repo, imageRepoIP := getImageRepoIP(cfg.OtherRepo, cfg.OtherRepoIP, cfg.HostIP, cfg.ImageRepo, localImage)
	log.Infof("params: onlineImage=%s otherRepo=%s, otherRepoIp=%s, hostIP=%s, imageRepo=%s, imageRepoPort=%s, kubernetesPort=%s",
		cfg.OnlineImage, cfg.OtherRepo, cfg.OtherRepoIP, cfg.HostIP, cfg.ImageRepo, cfg.ImageRepoPort, cfg.KubernetesPort)
	if err := generateRegistriesConfig(repo, k3sConfigPath); err != nil {
		return err
	}
	log.Info("Start the local Kubernetes cluster...")
	k3sStartScript := []string{
		"run", "-d", fmt.Sprintf("--name=%s", utils.LocalKubernetesName),
		"-p", fmt.Sprintf("%s:36443", cfg.KubernetesPort), "--privileged", "--restart=always", "-p", fmt.Sprintf("%s:30010", cfg.ConsolePort),
		"--add-host", fmt.Sprintf("%s:%s", cfg.ImageRepo, imageRepoIP),
		"-v", "/etc/rancher/k3s:/etc/rancher/k3s", "-v", "/etc/timezone:/etc/timezone", "-v", "/etc/docker:/etc/docker",
		"-v", "/etc/localtime:/etc/localtime", "-v", "/var/lib/rancher/k3s:/var/lib/rancher/k3s", "-v", "/bke:/bke",
		"-v", "/etc/openFuyao:/etc/openFuyao", "-v", fmt.Sprintf("%s:%s", utils.DefaultExtendManifestsDir, utils.DefaultExtendManifestsDir),
		"-v", fmt.Sprintf("%s:%s", K3sPodLogDir, K3sPodLogDir),
		"-v", fmt.Sprintf("%s:%s", K3sContainerLogDir, K3sContainerLogDir),
		// Do not pass --token: let k3s generate a random cluster secret on first start
		// and reuse the persisted value under /var/lib/rancher/k3s on restart.
		k3sImage, "server", "--snapshotter=native", fmt.Sprintf("--https-listen-port=%s", cfg.KubernetesPort),
		"--service-cidr=100.10.0.0/16", "--cluster-cidr=100.20.0.0/16",
		fmt.Sprintf("--advertise-address=%s", cfg.HostIP),
		fmt.Sprintf("--tls-san=%s", cfg.HostIP), fmt.Sprintf("--node-name=%s", utils.LocalKubernetesName),
		fmt.Sprintf("--pause-image=%s", k3sPause),
		"--disable=coredns,servicelb,traefik,local-storage,metrics-server"}
	if err := econd.Run(k3sStartScript); err != nil {
		return err
	}
	time.Sleep(waitConfigInterval * time.Second)
	if err := econd.CP(fmt.Sprintf("%s:/bin/k3s", utils.LocalKubernetesName), "/usr/bin/kubectl"); err != nil {
		return fmt.Errorf("copy kubectl from container: %w", err)
	}
	return setupKubeconfigAndWaitCluster(cfg.HostIP, cfg.KubernetesPort)
}

func checkCurrentCoreDNSConfig() error {
	cmd := exec.CommandExecutor{}

	// 获取当前的CoreDNS ConfigMap
	getCmd := []string{"get", "configmap", "coredns", "-n", "kube-system", "-o", "jsonpath={.data.Corefile}"}
	output, err := cmd.ExecuteCommandWithCombinedOutput(utils.KubeCtl, getCmd...)
	if err != nil {
		return fmt.Errorf("failed to get current coredns config: %v", err)
	}

	log.Info("Current CoreDNS configuration detected")

	// 检查是否已经包含/etc/resolv.conf
	if strings.Contains(output, "/etc/resolv.conf") {
		log.Info("Found /etc/resolv.conf in current config, will be replaced")
		return nil
	}

	// 检查是否已经使用固定DNS
	if containsFixedDNS(output) {
		log.Info("Fixed DNS already configured, no need to patch")
		return nil
	}

	log.Info("Current config does not contain /etc/resolv.conf")
	return nil
}

func verifyCoreDNSConfig() error {
	cmd := exec.CommandExecutor{}

	// 等待一段时间让配置生效
	time.Sleep(time.Duration(waitConfigInterval) * time.Second)

	// 获取修改后的配置进行验证
	getCmd := []string{"get", "configmap", "coredns", "-n", "kube-system", "-o", "jsonpath={.data.Corefile}"}
	output, err := cmd.ExecuteCommandWithCombinedOutput(utils.KubeCtl, getCmd...)
	if err != nil {
		return fmt.Errorf("failed to verify coredns config: %v", err)
	}

	// 检查是否仍然包含/etc/resolv.conf
	if strings.Contains(output, "/etc/resolv.conf") {
		return fmt.Errorf("config still contains /etc/resolv.conf after patch")
	}

	// 检查是否成功配置了固定DNS
	if !containsFixedDNS(output) {
		return fmt.Errorf("fixed DNS not found in config after patch")
	}

	log.Info("CoreDNS config verification passed: fixed DNS configured successfully")
	return nil
}

func verifyCoreDNSRunning() error {
	cmd := exec.CommandExecutor{}

	// 等待Pod重启
	time.Sleep(time.Duration(waitInterval) * time.Second)

	// 检查CoreDNS Pod状态
	checkCmd := []string{"get", "pods", "-n", "kube-system", "-l", "k8s-app=kube-dns", "-o", "jsonpath={.items[*].status.phase}"}
	output, err := cmd.ExecuteCommandWithCombinedOutput(utils.KubeCtl, checkCmd...)
	if err != nil {
		return fmt.Errorf("failed to check coredns pod status: %v", err)
	}

	if !strings.Contains(output, "Running") {
		return fmt.Errorf("coredns pods not in Running state: %s", output)
	}

	log.Info("CoreDNS pods are running successfully")
	return nil
}

// 检查是否包含固定DNS配置
func containsFixedDNS(config string) bool {
	fixedDNSPatterns := []string{
		"forward . 8.8.8.8",
		"forward . 8.8.4.4",
		"forward . 1.1.1.1",
		"forward . 1.0.0.1",
		"forward . 208.67.222.222",
		"forward . 208.67.220.220",
	}

	for _, pattern := range fixedDNSPatterns {
		if strings.Contains(config, pattern) {
			return true
		}
	}
	return false
}

func FixCoreDnsLoop(otherRepo, imageRepo string) error {
	if err := checkCurrentCoreDNSConfig(); err != nil {
		return fmt.Errorf("coreDNS config check failed: %v", err)
	}

	escapedCorefile := strings.ReplaceAll(k3sCoreScript, "\n", "\\n")
	escapedCorefile = strings.ReplaceAll(escapedCorefile, "\"", "\\\"")
	patchData := fmt.Sprintf(`{"data":{"Corefile":"%s"}}`, escapedCorefile)

	cmd := exec.CommandExecutor{}
	patchCmd := []string{"patch", "configmap", "coredns", "-n", "kube-system", "--type", "merge", "-p", patchData}
	output, err := cmd.ExecuteCommandWithCombinedOutput(utils.KubeCtl, patchCmd...)
	if err != nil {
		return fmt.Errorf("failed to patch coredns: %v, output: %s", err, output)
	}
	log.Infof("CoreDNS ConfigMap updated: %s", output)

	// 验证配置是否成功修改
	if err := verifyCoreDNSConfig(); err != nil {
		return fmt.Errorf("config verification failed after patch: %v", err)
	}

	if err := ModK3sCorednsImage(otherRepo, imageRepo); err != nil {
		return fmt.Errorf("mod coredns image tag failed: %v", err)
	}
	deleteCmd := []string{"delete", "pod", "-n", "kube-system", "-l", "k8s-app=kube-dns"}
	output, err = cmd.ExecuteCommandWithCombinedOutput(utils.KubeCtl, deleteCmd...)
	if err != nil {
		return fmt.Errorf("failed to delete coredns pods: %v, output: %s", err, output)
	}
	log.Infof("CoreDNS pods restarted: %s", output)

	// 最终验证
	if err := verifyCoreDNSRunning(); err != nil {
		return fmt.Errorf("coreDNS not running properly after restart: %v", err)
	}

	return nil
}

func ModCorednsConfigWithRetry(otherRepo, imageRepo string) error {
	const maxRetries = 3
	var lastError error

	for i := 0; i < maxRetries; i++ {
		log.Infof("Attempting to fix CoreDNS loop (attempt %d/%d)", i+1, maxRetries)

		if err := FixCoreDnsLoop(otherRepo, imageRepo); err != nil {
			lastError = err
			log.Debugf("Attempt %d failed: %v", i+1, err)
			time.Sleep(time.Duration(i+1) * waitConfigInterval * time.Second) // 指数退避
			continue
		}

		log.Infof("CoreDNS loop fix completed successfully on attempt %d", i+1)
		return nil
	}

	return fmt.Errorf("failed to fix CoreDNS loop after %d attempts: %v", maxRetries, lastError)
}

func ModK3sCorednsImage(otherRepo, imageRepo string) error {
	log.Infof("Waiting for coredns to be ready (timeout=%ds, interval=%ds)", waitTimeout, waitInterval)
	startTime := time.Now()
	cmdExecutor := exec.CommandExecutor{} // 复用命令执行器
	for {
		// 1. 检查coredns Deployment是否存在且就绪
		checkCmd := []string{
			"exec", utils.LocalKubernetesName, "kubectl",
			"-n", "kube-system",
			"get", "deployment/coredns",
			"-o", `jsonpath='{.status.conditions[?(@.type=="Available")].status}'`, // 只取Available状态
			"--ignore-not-found", // 不存在时不报错
		}
		output, err := cmdExecutor.ExecuteCommandWithCombinedOutput(utils.NerdCtl, checkCmd...)
		outputStr := strings.Trim(string(output), "'") // 去除jsonpath返回的单引号
		// 2. 判断是否满足条件：输出为"True"表示存在且就绪,
		if outputStr == "True" {
			log.Infof("coredns is ready (elapsed: %v)", time.Since(startTime).Round(time.Second))
			break
		}
		// 3. 判断是否超时
		if time.Since(startTime) >= time.Duration(waitTimeout)*time.Second {
			return fmt.Errorf("timeout waiting for coredns (%ds): output=%s, error=%v",
				waitTimeout, outputStr, err)
		}
		// 4. 未满足条件且未超时，继续等待
		log.Infof("coredns not ready (current status: %s), waiting...", outputStr)
		time.Sleep(time.Duration(waitInterval) * time.Second)
	}
	if otherRepo != "" {
		if imageRepo == configinit.DefaultImageRepo { //是默认在线安装
			k3sCoredns = fmt.Sprintf("%s%s", "cr.openfuyao.cn/openfuyao/", "kubernetes/coredns:v1.10.1")
		} else {
			log.Infof("coredns image registry: %s", otherRepo)
			k3sCoredns = fmt.Sprintf("%s%s", otherRepo, "kubernetes/coredns:v1.10.1")
		}
	} else {
		k3sCoredns = fmt.Sprintf("%s:443/%s/%s", imageRepo, bkecommon.ImageRegistryKubernetes, "kubernetes/coredns:v1.10.1")
	}
	// Modify coredns images
	k3sModCorednsImageScript := []string{"exec", utils.LocalKubernetesName, "kubectl",
		"-n", "kube-system", "set", "image", "deployment/coredns", fmt.Sprintf("coredns=%s", k3sCoredns)}
	log.Infof("Generated coredns image address: %s", k3sCoredns)
	var cmd2 = exec.CommandExecutor{}
	output, err := cmd2.ExecuteCommandWithCombinedOutput(utils.NerdCtl, k3sModCorednsImageScript...)
	if err != nil {
		return fmt.Errorf("set coredns image: %w, output: %s", err, string(output))
	}
	log.Info("mod k3s coredns image tag succeeded")
	return nil
}

func isKubernetesAvailable() bool {
	k8sClient, err := k8s.NewKubernetesClient("")
	if err != nil {
		return false
	}

	nodes, err := k8sClient.GetClient().CoreV1().Nodes().List(context.Background(), metav1.ListOptions{})
	if err != nil {
		return false
	}

	for _, node := range nodes.Items {
		for _, cond := range node.Status.Conditions {
			if cond.Type == corev1.NodeReady && cond.Status == corev1.ConditionTrue {
				global.K8s = k8sClient // 只在成功时赋值
				return true
			}
		}
	}
	return false
}

func buildK3sContainerConfig(hostIP string) *container.Config {
	return &container.Config{
		Hostname:     utils.LocalKubernetesName,
		AttachStdin:  false,
		AttachStdout: false,
		AttachStderr: false,
		ExposedPorts: map[nat.Port]struct{}{"6443/tcp": {}},
		Tty:          true,
		StdinOnce:    false,
		Env:          []string{"KUBECONFIG=/etc/kubernetes/admin.conf"},
		// Do not pass --token: let k3s generate/persist the cluster secret.
		Cmd: []string{"server", "--snapshotter=native", "--service-cidr=100.10.0.0/16",
			"--cluster-cidr=100.20.0.0/16",
			fmt.Sprintf("--tls-san=%s", hostIP), fmt.Sprintf("--node-name=%s", utils.LocalKubernetesName),
			fmt.Sprintf("--pause-image=%s", k3sPause), "--disable=coredns,servicelb,traefik,local-storage,metrics-server"},
		Image:      k3sImage,
		Volumes:    map[string]struct{}{"/var": {}},
		Labels:     map[string]string{"bke-local-kubernetes": "cluster-api"},
		StopSignal: "SIGRTMIN+3",
	}
}

func buildK3sHostConfig(kubernetesPort, imageRepo, imageRepoIP string) *container.HostConfig {
	initFlag := false
	return &container.HostConfig{
		PortBindings: map[nat.Port][]nat.PortBinding{
			nat.Port("6443/tcp"): {{HostIP: "0.0.0.0", HostPort: kubernetesPort}},
		},
		RestartPolicy: container.RestartPolicy{Name: "on-failure", MaximumRetryCount: 10},
		ExtraHosts:    []string{fmt.Sprintf("%s:%s", imageRepo, imageRepoIP)},
		Privileged:    true,
		SecurityOpt:   []string{"seccomp=unconfined", "apparmor=unconfined", "label=disable"},
		Tmpfs:         map[string]string{"/run": "", "/tmp": ""},
		Mounts: []mount.Mount{
			{Type: mount.TypeBind, Source: "/etc/rancher/k3s", Target: "/etc/rancher/k3s"},
			{Type: mount.TypeBind, Source: "/var/lib/rancher/k3s", Target: "/var/lib/rancher/k3s"},
			{Type: mount.TypeBind, Source: "/etc/timezone", Target: "/etc/timezone", ReadOnly: true},
			{Type: mount.TypeBind, Source: "/etc/localtime", Target: "/etc/localtime", ReadOnly: true},
			{Type: mount.TypeBind, Source: K3sPodLogDir, Target: K3sPodLogDir},
			{Type: mount.TypeBind, Source: K3sContainerLogDir, Target: K3sContainerLogDir},
		},
		Init: &initFlag,
	}
}

func prepareK3sEnvironment(cfg Config, localImage string) (string, string, error) {
	prepareK3sImages(cfg.OnlineImage, cfg.OtherRepo, cfg.ImageRepoPort, cfg.ImageRepo, localImage)

	err := global.Docker.EnsureImageExists(docker.ImageRef{Image: k3sImage}, utils.RetryOptions{MaxRetry: 3, Delay: 1})
	if err != nil {
		return "", "", err
	}

	containerRunFlag, err := global.Docker.EnsureContainerRun(utils.LocalKubernetesName)
	if err != nil {
		return "", "", err
	}
	if containerRunFlag {
		if isKubernetesAvailable() {
			return "", "", nil
		}
		if err := global.Docker.ContainerRemove(utils.LocalKubernetesName); err != nil {
			return "", "", fmt.Errorf("remove container %s failed before restart: %w", utils.LocalKubernetesName, err)
		}
	}

	k3sConfigPath := "/etc/rancher/k3s"
	if err = EnsureDirExists(k3sConfigPath); err != nil {
		return "", "", err
	}
	if err = EnsureK3sLogDirs(); err != nil {
		return "", "", err
	}

	if err = customCA(); err != nil {
		return "", "", err
	}

	repo, imageRepoIP := getImageRepoIPWithDocker(cfg.OtherRepo, cfg.OtherRepoIP, cfg.HostIP, cfg.ImageRepo)

	if err = generateRegistriesConfig(repo, k3sConfigPath); err != nil {
		return "", "", err
	}

	return repo, imageRepoIP, nil
}

// StartK3sWithDocker starts K3s cluster with Docker runtime
func StartK3sWithDocker(cfg Config, localImage string) error {
	if isKubernetesAvailable() {
		log.Info("A kubernetes cluster already exists.")
		return nil
	}

	repo, imageRepoIP, err := prepareK3sEnvironment(cfg, localImage)
	if err != nil {
		return err
	}
	if repo == "" && imageRepoIP == "" {
		log.Info("The local Kubernetes cluster is already running")
		return nil
	}

	log.Info("Start the local Kubernetes cluster...")

	containerConfig := buildK3sContainerConfig(cfg.HostIP)
	hostConfig := buildK3sHostConfig(cfg.KubernetesPort, cfg.ImageRepo, imageRepoIP)

	err = global.Docker.Run(containerConfig, hostConfig, nil, nil, utils.LocalKubernetesName)
	if err != nil {
		return err
	}

	time.Sleep(utils.DefaultSleepSeconds * time.Second)

	if err = global.Docker.CopyFromContainer(utils.LocalKubernetesName, "/bin/k3s", "/usr/bin/kubectl"); err != nil {
		return fmt.Errorf("copy kubectl from container: %w", err)
	}

	return setupKubeconfigAndWaitCluster(cfg.HostIP, cfg.KubernetesPort)
}

func customCA() error {
	var (
		output string
		err    error
	)
	// save generate-custom-ca-certs.sh to /var/lib/rancher/k3s/generate-custom-ca-certs.sh
	if err := EnsureDirExists(DefaultK3sDataDir); err != nil {
		return fmt.Errorf("create k3s certs dir failed: %w", err)
	}
	genShFile := filepath.Join(DefaultK3sDataDir, "generate-custom-ca-certs.sh")
	err = os.WriteFile(genShFile, []byte(k3sCertScript), utils.DefaultFilePermission)
	if err != nil {
		return fmt.Errorf("write generate-custom-ca-certs.sh failed: %w", err)
	}

	executor := &exec.CommandExecutor{}
	output, err = executor.ExecuteCommandWithCombinedOutput("/bin/bash", "-c",
		fmt.Sprintf("cd %s && chmod +x ./generate-custom-ca-certs.sh && ./generate-custom-ca-certs.sh &&"+
			"chmod -x ./generate-custom-ca-certs.sh", DefaultK3sDataDir))
	if err != nil {
		return fmt.Errorf("generate k3s tls cert failed, output: %s, err: %w", output, err)
	}

	return nil
}
