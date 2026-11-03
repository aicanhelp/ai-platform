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

// Package installer 实现console和oauth 相关pod的安装
package bkeconsole

import (
	"context"
	"embed"
	_ "embed"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
	bkecommon "gopkg.openfuyao.cn/cluster-api-provider-bke/common"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/client-go/kubernetes"

	"gopkg.openfuyao.cn/bkeadm/pkg/common/types"
	econd "gopkg.openfuyao.cn/bkeadm/pkg/executor/containerd"
	"gopkg.openfuyao.cn/bkeadm/pkg/executor/exec"
	"gopkg.openfuyao.cn/bkeadm/pkg/executor/k8s"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure/k3s"
	"gopkg.openfuyao.cn/bkeadm/pkg/initialize/clusterapi"
	"gopkg.openfuyao.cn/bkeadm/pkg/initialize/repository"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

var (
	//go:embed resource/*
	resourceFS embed.FS
	//go:embed consts.sh
	constScript string
	//go:embed log.sh
	logScript string
	//go:embed util.sh
	utilScript string
	//go:embed installConsole.sh
	consoleScript string
	//go:embed generateSecret.sh
	generateScript string
	//go:embed installOauthAndUser.sh
	installOauthAndUserScript string
	//go:embed dnsconfig.yaml
	dnsConfig string
	//go:embed coredns.yaml
	corednsYaml []byte
	k3sImage    = utils.DefaultLocalK3sRegistry
	k3sPause    = utils.DefaultK3sPause
	webhookFile = "/var/lib/rancher/k3s/webhook/webhook-config.yaml"
	cacheTtl    = "60s"
	k3sName     = "kubernetes"
	scriptDir   = "/var/lib/rancher/k3s/"
	resourceDir = "/var/lib/rancher/k3s/resource"
)

func copyEmbeddedFS(embeddedFS embed.FS, embedPath, dstDir string) error {
	return fs.WalkDir(embeddedFS, embedPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		// 构建目标路径
		relPath, err := filepath.Rel(embedPath, path)
		if err != nil {
			return err
		}
		dstPath := filepath.Join(dstDir, relPath)
		if d.IsDir() {
			// 创建目录
			info, err := d.Info()
			if err != nil {
				return err
			}
			return os.MkdirAll(dstPath, info.Mode())
		} else {
			// 读取嵌入文件内容
			data, err := embeddedFS.ReadFile(path)
			if err != nil {
				return err
			}
			// 确保目录存在
			if err := os.MkdirAll(filepath.Dir(dstPath), utils.DefaultDirPermission); err != nil {
				return err
			}
			// 写入目标文件
			return os.WriteFile(dstPath, data, utils.DefaultFilePermission)
		}
	})
}

func writeToDir(dir string, script string, scriptFile string) error {
	if !utils.Exists(dir) {
		err := os.MkdirAll(dir, utils.DefaultDirPermission)
		if err != nil {
			return fmt.Errorf("create dir failed: %w", err)
		}
	}
	shFile := filepath.Join(dir, script)
	err := os.WriteFile(shFile, []byte(scriptFile), utils.DefaultFilePermission)
	if err != nil {
		return fmt.Errorf("write %s fialed: %w", script, err)
	}
	return nil
}

// deployConsole 部署 console 相关 Pod
func deployConsole(restartConfig types.K3sRestartConfig, repo, openFuyaoVersion string) error {
	if !utils.Exists(resourceDir) {
		err := os.MkdirAll(resourceDir, utils.DefaultDirPermission)
		if err != nil {
			return fmt.Errorf("create dir failed: %w", err)
		}
	}
	err := copyEmbeddedFS(resourceFS, "resource", resourceDir)
	if err != nil {
		return fmt.Errorf("error copying embedded files: %w", err)
	}

	err = writeToDir(scriptDir, "installConsole.sh", consoleScript)
	if err != nil {
		return fmt.Errorf("write installConsole.sh failed: %w", err)
	}

	err = writeToDir(scriptDir, "consts.sh", constScript)
	if err != nil {
		return fmt.Errorf("write consts.sh failed: %w", err)
	}
	err = writeToDir(scriptDir, "log.sh", logScript)
	if err != nil {
		return fmt.Errorf("write log.sh failed: %w", err)
	}
	err = writeToDir(scriptDir, "utils.sh", utilScript)
	if err != nil {
		return fmt.Errorf("write utils.sh failed: %w", err)
	}

	if err := repository.EnsureConsoleTools(restartConfig.OtherSource, runtime.GOARCH); err != nil {
		return fmt.Errorf("prepare console tools failed: %w", err)
	}

	executor := &exec.CommandExecutor{}

	// 构建命令字符串
	command := fmt.Sprintf("cd %s && export REPO='%s' && export OPENFUYAO_VERSION='%s' && export BKE_FILE_PATH='%s'", scriptDir, repo, openFuyaoVersion,
		fmt.Sprintf("%s/%s/files", global.Workspace, utils.SourceDataDirectory))

	// 如果 otherRepo 为空（离线安装），添加额外的环境变量
	if restartConfig.OtherRepo == "" && restartConfig.OnlineImage == "" {
		command += " && export OFFLINE_INSTALL='true'"
		command += fmt.Sprintf(" && export HOST_IP='%s'", restartConfig.HostIP)
	}

	// 添加脚本执行部分
	command += " && chmod +x ./installConsole.sh && ./installConsole.sh && chmod -x ./installConsole.sh"

	output, err := executor.ExecuteCommandWithCombinedOutput("/bin/bash", "-c", command)
	if err != nil {
		return fmt.Errorf("installConsole failed, output: %s, err: %w", output, err)
	}
	return nil
}

func generateSecret() error {
	err := writeToDir(scriptDir, "generateSecret.sh", generateScript)
	if err != nil {
		return fmt.Errorf("write generateSecret.sh failed: %w", err)
	}

	executor := &exec.CommandExecutor{}
	output, err := executor.ExecuteCommandWithCombinedOutput("/bin/bash", "-c",
		fmt.Sprintf("cd %s && chmod +x ./generateSecret.sh && ./generateSecret.sh &&"+
			"chmod -x ./generateSecret.sh", scriptDir))
	if err != nil {
		return fmt.Errorf("generateSecret failed, output: %s, err: %w", output, err)
	}
	return nil
}

func k3sRestart(config types.K3sRestartConfig) error {
	if err := stopLocalK3sCluster(); err != nil {
		return err
	}
	configureK3sImages(config)
	if err := econd.EnsureImageExists(k3sImage); err != nil {
		return err
	}
	imageRepoIP := resolveImageRepoIP(config)
	if err := startK3sContainer(config.HostIP, config.ImageRepo, imageRepoIP, config.KubernetesPort, config.ConsolePort); err != nil {
		return err
	}
	const len2 = 2
	time.Sleep(len2 * time.Second)
	kubeconfigPath, err := processKubeconfig(config.HostIP, config.KubernetesPort)
	if err != nil {
		return err
	}
	if err := waitForKubernetesReady(kubeconfigPath); err != nil {
		return err
	}
	return waitForClusterReady()
}

func stopLocalK3sCluster() error {
	log.Info("Start to rm -f the local Kubernetes cluster...")
	if err := econd.Run([]string{"rm", "-f", k3sName}); err != nil {
		return fmt.Errorf("stop k3s cluster: %w", err)
	}
	if err := ensureK3sContainerReleased(); err != nil {
		return err
	}
	log.Info("stop the local k3s cluster success")
	return nil
}

func configureK3sImages(config types.K3sRestartConfig) {
	k3sImage = fmt.Sprintf("%s/%s", utils.DefaultThirdMirror, utils.DefaultLocalK3sRegistry)
	k3sPause = fmt.Sprintf("%s/%s", utils.DefaultThirdMirror, utils.DefaultK3sPause)
	if config.OtherRepo != "" {
		k3sImage = fmt.Sprintf("%s%s", config.OtherRepo, utils.DefaultLocalK3sRegistry)
		k3sPause = fmt.Sprintf("%s%s", config.OtherRepo, utils.DefaultK3sPause)
		return
	}
	if config.OnlineImage != "" {
		return
	}
	k3sImage = fmt.Sprintf("127.0.0.1:%s/%s/%s", config.ImageRepoPort, bkecommon.ImageRegistryKubernetes, utils.DefaultLocalK3sRegistry)
	k3sPause = fmt.Sprintf("%s:443/%s/%s", config.ImageRepo, bkecommon.ImageRegistryKubernetes, utils.DefaultK3sPause)
}

func resolveImageRepoIP(config types.K3sRestartConfig) string {
	if config.OtherRepo != "" && strings.Contains(config.OtherRepo, config.ImageRepo) {
		return config.OtherRepoIp
	}
	imageRepoInfo, err := econd.ContainerInspect(utils.LocalImageRegistryName)
	if err == nil && len(imageRepoInfo.Id) > 0 {
		return imageRepoInfo.NetworkSettings.IPAddress
	}
	return config.HostIP
}

func ensureK3sContainerReleased() error {
	const (
		maxRetry = 5
		waitTime = 2 * time.Second
	)
	var lastContainerID string
	for attempt := 1; attempt <= maxRetry; attempt++ {
		info, exist := econd.ContainerExists(k3sName)
		if !exist {
			return nil
		}
		lastContainerID = info.Id
		log.Debugf("Container %s still exists after stop, id=%s, retrying remove (%d/%d)",
			k3sName, info.Id, attempt, maxRetry)
		if err := econd.Run([]string{"rm", "-f", k3sName}); err != nil {
			log.Debugf("retry stop %s err: %v", k3sName, err)
		}
		time.Sleep(waitTime)
	}
	return fmt.Errorf("container %s is still occupying the name, id=%s", k3sName, lastContainerID)
}

// DNSConfig 结构体用于存储 DNS 配置
type DNSConfig struct {
	Servers []string `yaml:"servers"`
}

// getDNSServers 从嵌入的配置文件中获取 DNS 服务器列表
func getDNSServers() ([]string, error) {
	var config DNSConfig

	// 直接解析嵌入的 YAML 内容
	if err := yaml.Unmarshal([]byte(dnsConfig), &config); err != nil {
		return nil, fmt.Errorf("failed to parse DNS config: %v", err)
	}

	if len(config.Servers) == 0 {
		return nil, fmt.Errorf("no DNS servers configured")
	}

	return config.Servers, nil
}

func startK3sContainer(hostIP, imageRepo, imageRepoIP, kubernetesPort, consolePort string) error {
	log.Info("Restart the k3s cluster...")

	// 确保主机上存在 k3s 容器日志挂载目录，保证重启后日志挂载与初始化路径一致
	if err := k3s.EnsureK3sLogDirs(); err != nil {
		return err
	}

	// 获取 DNS 服务器配置
	dnsServers, err := getDNSServers()
	if err != nil {
		return fmt.Errorf("get DNS servers: %w", err)
	}

	// 首先构建 nerdctl run 的基础命令和参数
	k3sStartScript := []string{"run", "-d", fmt.Sprintf("--name=%s", utils.LocalKubernetesName),
		"-p", fmt.Sprintf("%s:36443", kubernetesPort), "--privileged", "--restart=always", "-p", fmt.Sprintf("%s:30010", consolePort),
		"--add-host", fmt.Sprintf("%s:%s", imageRepo, imageRepoIP),
		"-v", "/etc/rancher/k3s:/etc/rancher/k3s", "-v", "/etc/timezone:/etc/timezone", "-v", "/etc/docker:/etc/docker",
		"-v", "/etc/localtime:/etc/localtime", "-v", "/var/lib/rancher/k3s:/var/lib/rancher/k3s", "-v", "/bke:/bke",
		"-v", "/etc/openFuyao:/etc/openFuyao",
		"-v", fmt.Sprintf("%s:%s", utils.DefaultExtendManifestsDir, utils.DefaultExtendManifestsDir),
		"-v", fmt.Sprintf("%s:%s", k3s.K3sPodLogDir, k3s.K3sPodLogDir),
		"-v", fmt.Sprintf("%s:%s", k3s.K3sContainerLogDir, k3s.K3sContainerLogDir)}

	// 在 k3s server 命令之前添加 DNS 服务器配置
	for _, dnsServer := range dnsServers {
		k3sStartScript = append(k3sStartScript, "--dns", dnsServer)
	}
	log.Infof("configure k3s dns servers: %v", dnsServers)

	// 最后追加容器镜像和 k3s server 的命令及参数。
	// 不传 --token：首次由 k3s 随机生成并写入 /var/lib/rancher/k3s，重启复用已有值。
	k3sStartScript = append(k3sStartScript, k3sImage, "server", "--snapshotter=native",
		"--service-cidr=100.10.0.0/16", "--cluster-cidr=100.20.0.0/16",
		fmt.Sprintf("--https-listen-port=%s", kubernetesPort),
		fmt.Sprintf("--tls-san=%s", hostIP),
		fmt.Sprintf("--advertise-address=%s", hostIP),
		fmt.Sprintf("--node-name=%s", utils.LocalKubernetesName),
		fmt.Sprintf("--pause-image=%s", k3sPause),
		fmt.Sprintf("--kube-apiserver-arg=authentication-token-webhook-config-file=%s", webhookFile),
		fmt.Sprintf("--kube-apiserver-arg=authentication-token-webhook-cache-ttl=%s", cacheTtl),
		"--disable=coredns,servicelb,traefik,local-storage,metrics-server")

	if err := econd.Run(k3sStartScript); err != nil {
		return fmt.Errorf("start k3s container: %w", err)
	}
	return nil
}

func processKubeconfig(hostIP, kubernetesPort string) (string, error) {
	var result []byte
	var err error
	const len2 = 2
	for i := 0; i < 5; i++ {
		result, err = os.ReadFile("/etc/rancher/k3s/k3s.yaml")
		if err != nil {
			time.Sleep(len2 * time.Second)
			log.Debug("failed to get kubeconfig, retrying...")
			continue
		}
		break
	}
	if len(result) == 0 {
		return "", errors.New("failed to get /etc/rancher/k3s/k3s.yaml ")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("failed to get user home directory: %v", err)
	}

	kubeDir := fmt.Sprintf("%s/.kube", home)
	if err := os.MkdirAll(kubeDir, utils.DefaultDirPermission); err != nil {
		return "", fmt.Errorf("failed to create .kube directory: %v", err)
	}
	kubeconfigPath := fmt.Sprintf("%s/.kube/config", home)
	kubeconfigContent := strings.Replace(string(result), "127.0.0.1:36443",
		fmt.Sprintf("%s:%s", hostIP, kubernetesPort), 1)
	err = os.WriteFile(kubeconfigPath, []byte(kubeconfigContent), utils.SecureFilePermission)
	if err != nil {
		return "", err
	}
	if err := os.Remove("/etc/rancher/k3s/k3s.yaml"); err != nil && !os.IsNotExist(err) {
		log.Warnf("Failed to remove original k3s.yaml: %v", err)
	}
	err = os.WriteFile("/etc/rancher/k3s/k3s.yaml", []byte(kubeconfigContent), utils.DefaultFilePermission)
	if err != nil {
		log.Warnf("failed to write k3s.yaml: %v, please run export KUBECONFIG=%s", err, kubeconfigPath)
	}
	return kubeconfigPath, nil
}

func waitForKubernetesReady(kubeconfigPath string) error {
	log.Info("waiting for the cluster to start...")
	var err error
	const len6 = 6
	for i := 1; i < 10; i++ {
		global.K8s, err = k8s.NewKubernetesClient(kubeconfigPath)
		if err != nil {
			time.Sleep(len6 * time.Second)
			continue
		}
		break
	}
	if global.K8s == nil {
		return errors.New("failed to connect to Kubernetes. ")
	}
	return nil
}

func waitForClusterReady() error {
	log.Info("waiting for cluster Ready...")
	clientset := global.K8s.GetClient()
	const len3 = 3
	for i := 0; i < 10; i++ {
		node, err := clientset.CoreV1().Nodes().Get(context.Background(), utils.LocalKubernetesName, metav1.GetOptions{})
		if err != nil {
			time.Sleep(len3 * time.Second)
			continue
		}
		if len(node.Spec.Taints) > 1 {
			time.Sleep(len3 * time.Second)
			continue
		}
		_, err = clientset.CoreV1().Namespaces().Get(context.Background(), "kube-system", metav1.GetOptions{})
		if err != nil {
			time.Sleep(len3 * time.Second)
			continue
		}
		break
	}
	return nil
}

type oauthAndUserDeployConfig struct {
	onlineImage      string
	otherRepo        string
	hostIP           string
	repo             string
	openFuyaoVersion string
}

func deployOauthAndUser(config oauthAndUserDeployConfig, stage string) error {
	err := writeToDir(scriptDir, "installOauthAndUser.sh", installOauthAndUserScript)
	if err != nil {
		return fmt.Errorf("write installOauthAndUser.sh failed: %w", err)
	}

	executor := &exec.CommandExecutor{}
	// 构建命令字符串
	command := fmt.Sprintf("cd %s && export REPO='%s' && export OPENFUYAO_VERSION='%s'", scriptDir, config.repo, config.openFuyaoVersion)

	// 如果 otherRepo 为空（离线安装），添加额外的环境变量
	if config.otherRepo == "" && config.onlineImage == "" {
		command += " && export OFFLINE_INSTALL='true'"
		command += fmt.Sprintf(" && export HOST_IP='%s'", config.hostIP)
	}

	// 添加脚本执行部分
	command += fmt.Sprintf(" && chmod +x ./installOauthAndUser.sh && ./installOauthAndUser.sh '%s' && chmod -x ./installOauthAndUser.sh", stage)

	output, err := executor.ExecuteCommandWithCombinedOutput("/bin/bash", "-c", command)
	if err != nil {
		return fmt.Errorf("installOauthAndUser %s failed, output: %s, err: %w", stage, output, err)
	}
	return nil
}

func getPods(client kubernetes.Interface, namespace string) (*corev1.PodList, error) {
	pods, err := client.CoreV1().Pods(namespace).List(context.Background(), metav1.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("list %s pods: %w", namespace, err)
	}
	if len(pods.Items) == 0 {
		log.Info("No pods found in openfuyap-system namespace")
		return pods, fmt.Errorf("no pods found")
	}
	// 创建一个新的 PodList 来存储过滤后的 Pod
	filteredPods := &corev1.PodList{
		TypeMeta: pods.TypeMeta,
		ListMeta: pods.ListMeta,
	}
	// 过滤掉一次性任务 Pod
	for _, pod := range pods.Items {
		// 检查 Pod 是否由 Job 创建（通过检查 ownerReferences）
		isJobPod := false
		for _, ownerRef := range pod.OwnerReferences {
			if ownerRef.Kind == "Job" {
				isJobPod = true
				break
			}
		}
		// 排除一次性任务 Pod
		if !isJobPod {
			filteredPods.Items = append(filteredPods.Items, pod)
		} else {
			log.Debugf("Filtering out one-time pod: %s", pod.Name)
		}
	}
	// 检查过滤后是否有 Pod
	if len(filteredPods.Items) == 0 {
		log.Infof("No continuous pods found in %s namespace after filtering", namespace)
		return filteredPods, fmt.Errorf("no continuous pods found in %s", namespace)
	}
	return filteredPods, nil
}

// waitAllConsolePodRunning 等待所有pod处于Ready状态
func waitAllConsolePodRunning() {
	client := global.K8s.GetClient()
	for {
		time.Sleep(time.Duration(rand.IntnRange(utils.DefaultMinCheckSeconds, utils.DefaultMaxCheckSeconds)) * time.Second)
		log.Info("Waiting for Console service and cluster-system containers to be ready...")
		podList1, err1 := getPods(client, "openfuyao-system")
		podList2, err2 := getPods(client, "ingress-nginx")
		podList3, err3 := getPods(client, "cluster-system")
		if err1 != nil || err2 != nil || err3 != nil {
			continue
		}
		// 从 PodList 中提取 Pod 切片并合并
		var allPods []corev1.Pod
		allPods = append(allPods, podList1.Items...)
		allPods = append(allPods, podList2.Items...)
		allPods = append(allPods, podList3.Items...)
		if clusterapi.AreAllPodsReady(allPods) {
			log.Info("All installer service and cluster-system containers are ready")
			break
		}
	}
}

func deployCoredns(repo string) error {
	tmplDir := filepath.Join(global.Workspace, "tmpl")
	if err := os.MkdirAll(tmplDir, utils.DefaultDirPermission); err != nil {
		return fmt.Errorf("failed to create %s: %w", tmplDir, err)
	}
	corednsFile := filepath.Join(tmplDir, "coredns.yaml")
	if err := os.WriteFile(corednsFile, corednsYaml, utils.DefaultFilePermission); err != nil {
		return fmt.Errorf("failed to write %s: %w", corednsFile, err)
	}

	log.Info("Install Coredns...")
	if err := global.K8s.InstallYaml(corednsFile, map[string]string{"repo": repo}, ""); err != nil {
		return err
	}
	log.Info("Install Coredns Success")
	return nil
}

// DeployConsoleAll 部署引导节点用户管理功能
func DeployConsoleAll(restartConfig types.K3sRestartConfig, repo, openFuyaoVersion string) error {
	var err error
	if global.K8s == nil {
		global.K8s, err = k8s.NewKubernetesClient("")
		if err != nil {
			return fmt.Errorf("create kubernetes client: %w", err)
		}
	}

	err = deployCoredns(repo)
	if err != nil {
		return fmt.Errorf("deploy coredns: %w", err)
	}

	err = deployConsole(restartConfig, repo, openFuyaoVersion)
	if err != nil {
		return fmt.Errorf("deploy console: %w", err)
	}

	waitAllConsolePodRunning()

	err = generateSecret()
	if err != nil {
		return fmt.Errorf("generate console secret: %w", err)
	}
	log.Info("GenerateSecret success")
	oauthConfig := oauthAndUserDeployConfig{
		onlineImage:      restartConfig.OnlineImage,
		otherRepo:        restartConfig.OtherRepo,
		hostIP:           restartConfig.HostIP,
		repo:             repo,
		openFuyaoVersion: openFuyaoVersion,
	}

	err = deployOauthAndUser(oauthConfig, "webhook")
	if err != nil {
		return fmt.Errorf("deploy oauth webhook: %w", err)
	}
	log.Info("DeployOauthWebhook success")

	err = k3sRestart(restartConfig)
	if err != nil {
		return fmt.Errorf("restart k3s for console: %w", err)
	}
	log.Info("K3sRestart success")

	err = deployOauthAndUser(oauthConfig, "server")
	if err != nil {
		return fmt.Errorf("deploy oauth server and user management: %w", err)
	}
	log.Info("DeployOauthServerAndUser success")

	// 等pod 都重启
	waitAllConsolePodRunning()

	return nil
}
