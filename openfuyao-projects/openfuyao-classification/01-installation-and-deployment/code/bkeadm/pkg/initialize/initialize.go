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

package initialize

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"

	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/host"
	"github.com/shirou/gopsutil/v3/mem"
	"github.com/spf13/afero"
	"gopkg.in/yaml.v3"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkecommon "gopkg.openfuyao.cn/cluster-api-provider-bke/common"
	configinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	configsource "gopkg.openfuyao.cn/cluster-api-provider-bke/common/source"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"gopkg.openfuyao.cn/bkeadm/common/cluster/initialize/versions"
	"gopkg.openfuyao.cn/bkeadm/pkg/build"
	"gopkg.openfuyao.cn/bkeadm/pkg/cluster"
	"gopkg.openfuyao.cn/bkeadm/pkg/common/types"
	"gopkg.openfuyao.cn/bkeadm/pkg/config"
	econd "gopkg.openfuyao.cn/bkeadm/pkg/executor/containerd"
	"gopkg.openfuyao.cn/bkeadm/pkg/executor/k8s"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure/containerd"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure/k3s"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure/kubelet"
	"gopkg.openfuyao.cn/bkeadm/pkg/initialize/bkeagent"
	"gopkg.openfuyao.cn/bkeadm/pkg/initialize/bkeconfig"
	"gopkg.openfuyao.cn/bkeadm/pkg/initialize/bkeconsole"
	"gopkg.openfuyao.cn/bkeadm/pkg/initialize/clusterapi"
	"gopkg.openfuyao.cn/bkeadm/pkg/initialize/repository"
	"gopkg.openfuyao.cn/bkeadm/pkg/initialize/syscompat"
	"gopkg.openfuyao.cn/bkeadm/pkg/initialize/timezone"
	"gopkg.openfuyao.cn/bkeadm/pkg/root"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

type Options struct {
	root.Options
	File           string   `json:"file"`
	Args           []string `json:"args"`
	HostIP         string   `json:"hostIP"`
	Domain         string   `json:"domain"`
	KubernetesPort string   `json:"kubernetesPort"`
	ConsolePort    string   `json:"consolePort"`
	ImageRepoPort  string   `json:"imageRepoPort"`
	YumRepoPort    string   `json:"yumRepoPort"`
	ChartRepoPort  string   `json:"chartRepoPort"`
	ClusterAPI     string   `json:"clusterAPI"`
	OFVersion      string   `json:"OFVersion"`
	VersionUrl     string   `json:"versionUrl"`
	NtpServer      string   `json:"ntpServer"`
	Runtime        string   `json:"runtime"`
	RuntimeStorage string   `json:"runtimeStorage"`
	OnlineImage    string   `json:"onlineImage"`
	OtherRepo      string   `json:"otherRepo"`
	OtherSource    string   `json:"otherSource"`
	OtherChart     string   `json:"otherChart"` // TODO: helm chart 私有源地址
	InstallConsole bool     `json:"installConsole"`
	EnableNTP      bool     `json:"enableNTP"` // 是否启用NTP服务，默认为true

	ImageRepoCAFile    string `json:"imageRepoCAFile"`    // 镜像仓库CA证书文件
	ImageRepoUsername  string `json:"imageRepoUsername"`  // 镜像仓库用户名
	ImageRepoPassword  string `json:"imageRepoPassword"`  // 镜像仓库密码
	ImageRepoTLSVerify bool   `json:"imageRepoTLSVerify"` // 是否验证TLS证书

	ImageFilePath   string `json:"imageFilePath"`   // 引导节点初始化本地镜像路径
	AgentHealthPort string `json:"agentHealthPort"` // 集群节点代理的健康监听端口

	// Dependency Injection
	FS               afero.Fs
	DownloadFunc     func(url, dest string) error
	SetPatchConfigFn func(version, path, key string) error
	K8sClient        k8s.KubernetesClient
}

var oc repository.OtherRepo

// 主初始化
func (op *Options) Initialize() {
	log.Info("BKE initialize ...")
	// Explicitly materialize the workspace layout (tmpl/volumes/mount). This
	// was historically done as a package-init side effect of pkg/global; it is
	// now opt-in so read-only surfaces (preflight, help, version) stay
	// side-effect-free on load. Only the real install path creates them.
	if err := global.EnsureWorkspaceLayout(global.Workspace); err != nil {
		// Preserve the historical package-init behavior: directory creation
		// failures were warnings and did not stop Initialize at this point.
		log.Warnf("Failed to ensure workspace layout, %v", err)
	}
	op.nodeInfo()
	err := op.Validate()
	if err != nil {
		log.Errorf("Validation failure, %v", err)
		return
	}
	err = op.setTimezone()
	if err != nil {
		log.Errorf("Timezone failure, %v", err)
		return
	}
	err = op.prepareEnvironment()
	if err != nil {
		log.Errorf("Failed to prepare environment, %v", err)
		return
	}
	err = op.ensureContainerServer()
	if err != nil {
		log.Errorf("Failed to start the container service, %v", err)
		return
	}
	err = op.ensureRepository()
	if err != nil {
		log.Errorf("Failed to start warehouse, %v", err)
		return
	}
	err = op.ensureClusterAPI()
	if err != nil {
		log.Errorf("Failed to start cluster API, %v", err)
		return
	}
	// 条件性安装 bkeconsole
	if op.InstallConsole {
		err = op.ensureConsoleAll()
		if err != nil {
			log.Errorf("Failed to start Console, %v", err)
			return
		}
	} else {
		log.Info("Skipping bkeconsole installation as requested")
	}

	// Generating a Configuration File
	op.generateClusterConfig()
	// chmod /bke dir and file permission (reason: openEuler 20.03 umask 0077 need modify permission)
	op.modifyPermission()
	log.Info("BKE initialization is complete")
	op.deployCluster()
}

// 节点信息收集打印
func (op *Options) nodeInfo() {
	h, _ := host.Info()
	c, _ := cpu.Counts(false)
	v, _ := mem.VirtualMemory()

	log.Infof("HOSTNAME: %s", h.Hostname)
	log.Infof("PLATFORM: %s", h.Platform)
	log.Infof("Version:  %s", h.PlatformVersion)
	log.Infof("KERNEL:   %s", h.KernelVersion)
	log.Infof("GOOS:     %s", runtime.GOOS)
	log.Infof("ARCH:     %s", runtime.GOARCH)
	log.Infof("CPU:      %d", c)
	log.Infof("MEMORY:   %dG", v.Total/1024/1024/1024+1)

	if op.InstallConsole {
		log.Info("BKE Console: ENABLED")
	} else {
		log.Info("BKE Console: DISABLED")
	}

}

// 验证初始化参数，环境验证，验证运行环境是否满足安装要求
func (op *Options) Validate() error {
	log.Info("BKE initialize environment check...")
	var err error

	oc, err = repository.ParseOnlineConfig(op.Domain, op.OnlineImage, op.OtherRepo, op.OtherSource, op.OtherChart)

	if err != nil {
		return errors.New(fmt.Sprintf("Configuration parsing failure %v", err))
	}

	op.logAuthMode()

	if err = op.validateDiskSpace(); err != nil {
		return err
	}

	if err = op.validatePorts(); err != nil {
		return err
	}

	op.setGlobalCustomExtra()
	return nil
}

// logAuthMode 记录认证模式
func (op *Options) logAuthMode() {
	if op.ImageRepoTLSVerify {
		if op.ImageRepoUsername != "" && op.ImageRepoPassword != "" {
			log.Info("Password authentication will be used")
		} else if op.ImageRepoCAFile != "" {
			log.Info("CA certificate authentication will be used")
		} else {
			log.Info("TLS verify enabled without username/password/CA file; using default trust if available")
		}
	}
}

// validateDiskSpace 检查磁盘空间
func (op *Options) validateDiskSpace() error {
	_, free := utils.DiskUsage(global.Workspace)
	if utils.Exists(path.Join(global.Workspace, utils.ImageDataDirectory)) {
		if free/1024/1024/1024 < utils.MinDiskSpaceExisting {
			return errors.New(fmt.Sprintf("The available space of the working directory %s is less than %d GB",
				global.Workspace, utils.MinDiskSpaceExisting))
		}
	} else {
		if free/1024/1024/1024 < utils.MinDiskSpace {
			return errors.New(fmt.Sprintf("The available space of the working directory %s is less than %d GB",
				global.Workspace, utils.MinDiskSpace))
		}
	}
	return nil
}

// validatePorts 检查端口占用
func (op *Options) validatePorts() error {
	ports := []string{op.KubernetesPort, op.ConsolePort, op.ImageRepoPort, op.ChartRepoPort, op.YumRepoPort}
	ports = op.filterExistingContainerPorts(ports)
	log.Infof("prepare to validatePorts: %v", ports)

	err := utils.CheckPorts(ports)
	if err != nil {
		return errors.New(fmt.Sprintf("The port is already in use %v", err))
	}
	return nil
}

// filterExistingContainerPorts 过滤已存在容器的端口
func (op *Options) filterExistingContainerPorts(ports []string) []string {
	if infrastructure.IsDocker() {
		ports = op.filterDockerContainerPorts(ports)
	}
	if infrastructure.IsContainerd() {
		ports = op.filterContainerdPorts(ports)
	}
	return ports
}

// filterDockerContainerPorts 过滤 Docker 容器端口
func (op *Options) filterDockerContainerPorts(ports []string) []string {
	if _, ok := global.Docker.ContainerExists(utils.LocalKubernetesName); ok {
		ports = utils.RemoveStringObject(ports, op.KubernetesPort)
	}
	if _, ok := global.Docker.ContainerExists(utils.LocalImageRegistryName); ok {
		ports = utils.RemoveStringObject(ports, op.ImageRepoPort)
	}
	if _, ok := global.Docker.ContainerExists(utils.LocalChartRegistryName); ok {
		ports = utils.RemoveStringObject(ports, op.ChartRepoPort)
	}
	if _, ok := global.Docker.ContainerExists(utils.LocalYumRegistryName); ok {
		ports = utils.RemoveStringObject(ports, op.YumRepoPort)
	}
	return ports
}

// filterContainerdPorts 过滤 Containerd 容器端口
func (op *Options) filterContainerdPorts(ports []string) []string {
	if _, ok := econd.ContainerExists(utils.LocalKubernetesName); ok {
		// Kubernetes API 和 Console NodePort 都由本地 K3s 容器发布
		ports = utils.RemoveStringObject(ports, op.KubernetesPort)
		ports = utils.RemoveStringObject(ports, op.ConsolePort)
	}
	if _, ok := econd.ContainerExists(utils.LocalImageRegistryName); ok {
		ports = utils.RemoveStringObject(ports, op.ImageRepoPort)
	}
	if _, ok := econd.ContainerExists(utils.LocalChartRegistryName); ok {
		ports = utils.RemoveStringObject(ports, op.ChartRepoPort)
	}
	if _, ok := econd.ContainerExists(utils.LocalYumRegistryName); ok {
		ports = utils.RemoveStringObject(ports, op.YumRepoPort)
	}
	return ports
}

// setGlobalCustomExtra 设置全局自定义扩展配置
func (op *Options) setGlobalCustomExtra() {
	global.CustomExtra["domain"] = op.Domain
	global.CustomExtra["host"] = op.HostIP
	global.CustomExtra["imageRepoPort"] = op.ImageRepoPort
	global.CustomExtra["yumRepoPort"] = op.YumRepoPort
	global.CustomExtra["chartRepoPort"] = op.ChartRepoPort
	global.CustomExtra["clusterapi"] = op.ClusterAPI
	global.CustomExtra["nfsserverpath"] = "/"
	global.CustomExtra["onlineImage"] = op.OnlineImage
	global.CustomExtra["otherRepo"] = oc.Repo
	global.CustomExtra["otherRepoIp"] = oc.RepoIP
	global.CustomExtra["otherSource"] = oc.Source
	global.CustomExtra["otherChart"] = oc.ChartRepo
	global.CustomExtra["otherChartIp"] = oc.ChartRepoIP
	global.CustomExtra["agentLogLevel"] = "INFO"
}

// 设置时区和NTP服务器
func (op *Options) setTimezone() error {
	log.Info("set up the host machine zone")
	err := timezone.SetTimeZone()
	if err != nil {
		return err
	}
	// 如果禁用了NTP服务，则跳过NTP服务器设置
	if !op.EnableNTP {
		log.Info("NTP service is disabled, skipping NTP server setup")
		op.NtpServer = "" // 设置为空字符串，后续配置会跳过NTP设置
		return nil
	}
	log.Info("set ntp server")
	newNTPServer, err := timezone.NTPServer(op.NtpServer, op.HostIP, len(oc.Repo) > 0)
	if err != nil {
		return err
	}
	op.NtpServer = newNTPServer
	return nil
}

// 准备初始化环境，配置下载源，设置hosts，配置私有仓库CA证书等
func (op *Options) prepareEnvironment() error {
	log.Info("config local source")

	op.configLocalSource()

	hostIP, domain := op.resolveHostIPAndDomain()
	if err := syscompat.SetHosts(hostIP, domain); err != nil {
		log.Warnf("Failed to set hosts %v", err)
	}

	clientAuthConfig := op.buildClientAuthConfig()
	if err := op.configurePrivateRegistry(clientAuthConfig); err != nil {
		return err
	}

	return op.initRepositories(clientAuthConfig)
}

// configLocalSource 配置本地源
func (op *Options) configLocalSource() {
	// OtherRepo设置私有镜像源，OnlineImage指定二进制来源，两者有一个就认为是在线安装
	if op.OtherRepo == "" && op.OnlineImage == "" {
		baseurl := "file://" + path.Join(global.Workspace, utils.SourceDataDirectory)
		if len(oc.Source) > 0 {
			baseurl = oc.Source
		}
		err := configsource.SetSource(baseurl)
		if err != nil {
			log.Warnf("Failed to set download source %v", err)
		}
	}
}

// resolveHostIPAndDomain 解析写入 /etc/hosts 的 IP 与 domain。
func (op *Options) resolveHostIPAndDomain() (string, string) {
	hostIP, domain := op.HostIP, op.Domain
	if op.OtherRepo != "" || op.OnlineImage != "" {
		hostIP = op.resolveOnlineModeHostIP(domain)
	} else {
		log.Infof("Offline mode: domain=%s bound to bootstrap node IP=%s", domain, hostIP)
	}
	return applyDefaultRepoHostOverride(hostIP, domain)
}

func (op *Options) resolveOnlineModeHostIP(domain string) string {
	registryHost, _ := repository.ParseRegistryHostPort(op.OtherRepo)
	switch {
	case net.ParseIP(registryHost) != nil:
		log.Infof("Online mode: domain=%s bound to otherRepo IP=%s", domain, registryHost)
		return registryHost
	case registryHost != "" && registryHost == op.Domain && oc.RepoIP != "":
		log.Infof("Online mode: domain=%s bound to repo IP=%s", domain, oc.RepoIP)
		return oc.RepoIP
	default:
		log.Infof("Online mode: domain=%s bound to bootstrap node IP=%s", domain, op.HostIP)
		return op.HostIP
	}
}

func applyDefaultRepoHostOverride(hostIP, domain string) (string, string) {
	if !strings.Contains(oc.Repo, configinit.DefaultImageRepo) {
		return hostIP, domain
	}
	hostIP = oc.RepoIP
	domain = strings.Split(strings.Split(oc.Repo, "/")[0], ":")[0]
	return hostIP, domain
}

// buildClientAuthConfig 构建客户端认证配置
func (op *Options) buildClientAuthConfig() *repository.CertificateConfig {
	return &repository.CertificateConfig{
		TLSVerify: op.ImageRepoTLSVerify,
		Username:  op.ImageRepoUsername,
		Password:  op.ImageRepoPassword,
		CAFile:    op.ImageRepoCAFile,
	}
}

// configurePrivateRegistry 配置私有仓库
func (op *Options) configurePrivateRegistry(cfg *repository.CertificateConfig) error {
	registryHost, registryPort := repository.ParseRegistryHostPort(oc.Repo)
	if registryHost != "" && registryPort != "" {
		cfg.RegistryHost = registryHost
		cfg.RegistryPort = registryPort
		if cfg.TLSVerify && cfg.CAFile != "" {
			if err := repository.SetupCACertificate(cfg); err != nil {
				return fmt.Errorf("配置私有仓库CA证书失败：%v", err)
			}
		}
	} else {
		log.Debug("cannot resolve private registry address, skipping CA certificate configuration")
	}
	return nil
}

// initRepositories 初始化仓库
func (op *Options) initRepositories(clientAuthConfig *repository.CertificateConfig) error {
	if err := repository.RepoInit(oc, clientAuthConfig); err != nil {
		return err
	}
	if err := repository.DecompressionSystemSourceFile(); err != nil {
		return err
	}
	if op.ImageFilePath == "" {
		if err := repository.SourceInit(oc); err != nil {
			return err
		}
	}
	if err := syscompat.RepoUpdate(); err != nil {
		log.Warnf("Failed to update repo %v", err)
	}
	if err := syscompat.Compat(); err != nil {
		return errors.New(fmt.Sprintf("The system is not compatible %v", err))
	}
	if err := syscompat.SetForwardAccept(); err != nil {
		log.Warnf("Failed to set iptables FORWARD ACCEPT: %v", err)
	}
	syscompat.SetSysctl()
	return nil
}

// 容器运行时安装
func (op *Options) ensureContainerServer() error {
	err := repository.PrepareRepositoryDependOn(op.ImageFilePath)
	if err != nil {
		return err
	}
	op.modifyPermission()
	result, err := repository.VerifyContainerdFile(op.ImageFilePath)
	if err != nil {
		return err
	}
	containerdFile := result.ContainerdList[0]
	cniPluginFile := result.CniPluginList[0]
	for _, cond := range result.ContainerdList {
		if strings.Contains(cond, runtime.GOARCH) {
			containerdFile = cond
			continue
		}
	}
	for _, cni := range result.CniPluginList {
		if strings.Contains(cni, runtime.GOARCH) {
			cniPluginFile = cni
			continue
		}
	}
	err = infrastructure.RuntimeInstall(infrastructure.RuntimeConfig{
		Runtime:        op.Runtime,
		RuntimeStorage: op.RuntimeStorage,
		Domain:         op.Domain,
		ImageRepoPort:  op.ImageRepoPort,
		ContainerdFile: result.FilePath + "/" + containerdFile,
		CniPluginFile:  result.FilePath + "/" + cniPluginFile,
		DockerdFile:    result.FilePath + "/" + strings.Replace(utils.KylinDocker, "{.arch}", runtime.GOARCH, -1),
		HostIP:         op.HostIP,
		CAFile:         op.ImageRepoCAFile,
	})
	if err != nil {
		return fmt.Errorf("install container runtime: %w", err)
	}
	return nil
}

// 仓库服务启动
func (op *Options) ensureRepository() error {
	log.Info("Start the base dependency service")
	var err error
	// 新安装方式，registry镜像从本地获取
	if op.ImageFilePath != "" {
		err = repository.LoadLocalImage()
		if err != nil {
			return err
		}
	}
	// 加载本地仓库镜像
	err = repository.LoadLocalRepository()
	if err != nil {
		return err
	}
	// 启动镜像仓库服务
	err = repository.ContainerServer(op.ImageFilePath, op.ImageRepoPort, oc.Repo, oc.Image)
	if err != nil {
		return err
	}
	// sync local image to registry
	if op.ImageFilePath != "" {
		err = repository.SyncLocalImage(op.ImageRepoPort)
		if err != nil {
			return err
		}
	}
	// 启动YUM仓库服务
	err = repository.YumServer(op.ImageFilePath, op.ImageRepoPort, op.YumRepoPort, oc.Repo, oc.Image)
	if err != nil {
		return err
	}
	// 启动Chart仓库服务
	err = repository.ChartServer(op.ImageFilePath, op.ImageRepoPort, op.ChartRepoPort, oc.Repo, oc.Image)
	if err != nil {
		return err
	}
	// 注意：初始化流程不再拉起 NFS 服务（2049 端口也不再校验）。
	// 如需 NFS，可通过 `bke start nfs` 手动启动。
	log.Info("Skip starting NFS service during init")
	return nil
}

// 从文件名中提取版本号
func extractVersionFromFilename(filename string) string {
	base := strings.TrimSuffix(filename, filepath.Ext(filename))
	versionRegex := regexp.MustCompile(`^(?:.*-)?(latest|v\d+\.\d+(?:[-.]\w+)*)$`)
	matches := versionRegex.FindStringSubmatch(base)
	if len(matches) >= utils.MatchFields {
		return matches[1]
	}
	return ""
}

// 离线模式生成ConfigMap
func (op *Options) offlineGenerateDeployCM(patchesDir string) error {
	fs := op.FS
	if fs == nil {
		fs = afero.NewOsFs()
	}
	setPatchFn := op.SetPatchConfigFn
	if setPatchFn == nil {
		setPatchFn = bkeconfig.SetPatchConfig
	}

	if _, err := fs.Stat(patchesDir); os.IsNotExist(err) {
		log.Warnf("patchesDir %s not exist, use default", patchesDir)
		return err
	}

	entries, err := afero.ReadDir(fs, patchesDir)
	if err != nil {
		log.Warnf("read %s fail %s, use default", patchesDir, err)
		return err
	}

	for _, entry := range entries {
		version := extractVersionFromFilename(entry.Name())
		if op.OFVersion == version {
			log.Infof("version %s file, generate cm", op.OFVersion)
			fullPath := filepath.Join(patchesDir, entry.Name())
			cmKey := fmt.Sprintf("%s%s", utils.PatchValuePrefix, version)
			if err = setPatchFn(version, fullPath, cmKey); err != nil {
				log.Warnf("generate cm fail %s, use default", err)
				return err
			}
			return nil
		}
	}
	return fmt.Errorf("offline patch %s not exist, use default", op.OFVersion)
}

// 解析YAML字节数据到切片映射
func parseYAMLBytesToSliceMap(data []byte) ([]map[string]string, error) {
	var rawList []map[string]string
	if err := yaml.Unmarshal(data, &rawList); err != nil {
		return nil, fmt.Errorf("failed to unmarshal YAML: %w", err)
	}

	result := make([]map[string]string, 0, len(rawList))
	for i, item := range rawList {
		if _, exists := item["openFuyaoVersion"]; !exists {
			return nil, fmt.Errorf("item at index %d is missing required field 'openFuyaoVersion'", i)
		}
		if _, exists := item["filePath"]; !exists {
			return nil, fmt.Errorf("item at index %d is missing required field 'filePath'", i)
		}
		item["filePath"], _ = strings.CutPrefix(item["filePath"], "./")
		result = append(result, item)
	}
	return result, nil
}

// 在线模式生成ConfigMap
func (op *Options) onlineGenerateDeployCM() error {
	fs := op.FS
	if fs == nil {
		fs = afero.NewOsFs()
	}
	downloadFn := op.DownloadFunc
	if downloadFn == nil {
		downloadFn = utils.DownloadFile
	}
	setPatchFn := op.SetPatchConfigFn
	if setPatchFn == nil {
		setPatchFn = bkeconfig.SetPatchConfig
	}

	patchesDir := filepath.Join(global.Workspace, utils.PatchDataDirectory)
	if err := fs.MkdirAll(patchesDir, utils.DefaultDirPermission); err != nil {
		log.Warnf("mkdir dir %s err %v, use default", patchesDir, err)
		return err
	}

	url := op.VersionUrl
	if strings.HasSuffix(url, "/") {
		url = strings.TrimSuffix(url, "/")
	}
	indexURL := fmt.Sprintf("%s/index.yaml", url)
	indexFile := filepath.Join(patchesDir, "index.yaml")

	if err := downloadFn(indexURL, indexFile); err != nil {
		log.Warnf("download file %s err %v, use default", indexURL, err)
		return err
	}
	defer func() {
		if err := fs.Remove(indexFile); err != nil {
			log.Warnf("Remove temp index file %s failed: %v", indexFile, err)
		}
	}()

	data, err := afero.ReadFile(fs, indexFile)
	if err != nil {
		log.Warnf("read index.yaml failed: %v", err)
		return err
	}

	return op.processIndexYAML(downloadFn, setPatchFn, data, url, patchesDir)
}

// 处理index.yaml数据，索引处理
func (op *Options) processIndexYAML(
	downloadFn func(string, string) error,
	setPatchFn func(string, string, string) error,
	data []byte,
	baseURL, patchesDir string,
) error {
	patchRes, err := parseYAMLBytesToSliceMap(data)
	if err != nil {
		log.Warnf("parseYAMLFileToSliceMap err %v, use default", err)
		return err
	}

	for _, value := range patchRes {
		if value["openFuyaoVersion"] == op.OFVersion {
			filePath := value["filePath"]
			downloadURL := fmt.Sprintf("%s/%s", baseURL, filePath)
			downloadFile := filepath.Join(patchesDir, filePath)

			if err = downloadFn(downloadURL, downloadFile); err != nil {
				log.Warnf("download file %s err %v, use default", downloadURL, err)
				return err
			}

			cmKey := fmt.Sprintf("%s%s", utils.PatchValuePrefix, op.OFVersion)
			if err = setPatchFn(op.OFVersion, downloadFile, cmKey); err != nil {
				log.Warnf("generate cm fail %s, use default", err)
				return err
			}
			return nil
		}
	}
	return fmt.Errorf("online patch %s not exist, use default", op.OFVersion)
}

// 生成部署所需的ConfigMap
func (op *Options) generateDeployCM() error {
	// generate patch config map from local image
	if op.ImageFilePath != "" {
		patchesDir := filepath.Join(global.Workspace, utils.LocalPatchDirectory)
		return op.offlineGenerateDeployCM(patchesDir)
	}
	if oc.Repo == "" && oc.Image == "" {
		patchesDir := filepath.Join(global.Workspace, utils.PatchDataDirectory)
		return op.offlineGenerateDeployCM(patchesDir)
	} else {
		return op.onlineGenerateDeployCM()
	}
}

// 获取Cluster API版本
func (op *Options) getClusterAPIVersion(openFuyaoVersion, defaultVersion string) (string, string) {
	var client k8s.KubernetesClient
	if op.K8sClient != nil {
		client = op.K8sClient
	} else if global.K8s != nil {
		client = global.K8s
	} else {
		var err error
		client, err = k8s.NewKubernetesClient("")
		if err != nil {
			log.Warnf("failed to init k8s client: %v", err)
			return defaultVersion, defaultVersion
		}
		global.K8s = client
	}

	patchCmKey := fmt.Sprintf("cm.%s", openFuyaoVersion)
	k8sClient := client.GetClient()
	patchConfigMap, err := k8sClient.CoreV1().ConfigMaps("openfuyao-patch").Get(context.TODO(), patchCmKey, metav1.GetOptions{})
	if err != nil {
		log.Warnf("failed to get patch cm, err: %v", err)
		return defaultVersion, defaultVersion
	}

	data, ok := patchConfigMap.Data[openFuyaoVersion]
	if !ok {
		log.Warnf("cm data not contain %s key", openFuyaoVersion)
		return defaultVersion, defaultVersion
	}

	cfg := &build.BuildConfig{}
	if err = yaml.Unmarshal([]byte(data), cfg); err != nil {
		log.Warnf("Unable to serialize patch config for version %s: %v", openFuyaoVersion, err)
		return defaultVersion, defaultVersion
	}

	return extractVersionsFromConfig(cfg, defaultVersion)
}

// 扁平化镜像列表
func flattenImages(cfg *build.BuildConfig) []build.Image {
	if cfg == nil {
		return nil
	}
	var images []build.Image
	for _, repo := range cfg.Repos {
		for _, sub := range repo.SubImages {
			images = append(images, sub.Images...)
		}
	}
	return images
}

// 查找指定镜像的标签
func findImageTag(cfg *build.BuildConfig, imageName, defaultVersion string) string {
	for _, img := range flattenImages(cfg) {
		if img.Name == imageName && len(img.Tag) > 0 {
			return img.Tag[0]
		}
	}
	return defaultVersion
}

// 从配置中提取版本信息
func extractVersionsFromConfig(cfg *build.BuildConfig, defaultVersion string) (string, string) {
	manifestsVersion := findImageTag(cfg, "bke-manifests", defaultVersion)
	providerVersion := findImageTag(cfg, "cluster-api-provider-bke", defaultVersion)

	return manifestsVersion, providerVersion
}

// Cluster API安装
func (op *Options) ensureClusterAPI() error {
	err := infrastructure.StartLocalKubernetes(k3s.Config{
		OnlineImage:    oc.Image,
		OtherRepo:      oc.Repo,
		OtherRepoIP:    oc.RepoIP,
		HostIP:         op.HostIP,
		ImageRepo:      op.Domain,
		ImageRepoPort:  op.ImageRepoPort,
		KubernetesPort: op.KubernetesPort,
		ConsolePort:    op.ConsolePort,
	}, op.ImageFilePath)
	if err != nil {
		return fmt.Errorf("start kubernetes: %w", err)
	}

	if err = syscompat.SetForwardAccept(); err != nil {
		log.Warnf("Failed to set iptables FORWARD ACCEPT: %v", err)
	}

	// 将需要的openfuyao版本信息写入到configmap供后续安装使用
	if op.OFVersion != "" {
		if err = op.generateDeployCM(); err != nil {
			return fmt.Errorf("version %s not in released version list: %w", op.OFVersion, err)
		}
	}

	if err = containerd.ApplyContainerdCfg(fmt.Sprintf("%s:%s", op.Domain, op.ImageRepoPort)); err != nil {
		return fmt.Errorf("install containerd config: %w", err)
	}

	if err = kubelet.ApplyKubeletCfg(); err != nil {
		return fmt.Errorf("install kubelet config: %w", err)
	}

	if err = bkeagent.InstallBKEAgentCRD(); err != nil {
		return fmt.Errorf("install bkeagent command CRD: %w", err)
	}

	var repo string
	localRepoPath := fmt.Sprintf("%s:%s/%s/", op.Domain, "443", bkecommon.ImageRegistryKubernetes)

	// 优先级：ImageFilePath > oc.Repo > (oc.Image为空时使用本地) > 默认值
	switch {
	case op.ImageFilePath != "":
		// ImageFilePath 不为空，使用本地仓库路径
		repo = localRepoPath
	case oc.Repo != "":
		// oc.Repo 不为空，使用 oc.Repo
		repo = oc.Repo
	case oc.Image == "":
		// oc.Image 为空（离线场景），使用本地仓库路径
		repo = localRepoPath
	default:
	}
	// 如果都不满足，repo 保持为空字符串（使用默认值）
	manifestsVersion, providerVersion := op.getClusterAPIVersion(op.OFVersion, op.ClusterAPI)
	err = clusterapi.DeployClusterAPI(repo, manifestsVersion, providerVersion)
	if err != nil {
		return fmt.Errorf("deploy cluster-api: %w", err)
	}

	log.Info("The cluster-api deployment is complete")
	return nil
}

// BKE Console安装
func (op *Options) ensureConsoleAll() error {
	// 检查是否启用 console 安装
	if !op.InstallConsole {
		log.Info("BKE Console installation is disabled")
		return nil
	}

	log.Info("Starting BKE Console installation...")

	var repo string
	localRepoPath := fmt.Sprintf("%s:%s/%s/", op.Domain, "443", bkecommon.ImageRegistryKubernetes)

	// 优先级：oc.Repo > (oc.Image为空时使用本地) > 默认值
	switch {
	case oc.Repo != "":
		// oc.Repo 不为空，使用 oc.Repo
		repo = oc.Repo
	case oc.Image == "":
		// oc.Image 为空（离线场景），使用本地仓库路径
		repo = localRepoPath
	default:
	}
	var sRestartConfig types.K3sRestartConfig
	sRestartConfig = types.K3sRestartConfig{
		OnlineImage:    oc.Image,
		OtherRepo:      oc.Repo,
		OtherRepoIp:    oc.RepoIP,
		OtherSource:    oc.Source,
		HostIP:         op.HostIP,
		ImageRepo:      op.Domain,
		ImageRepoPort:  op.ImageRepoPort,
		KubernetesPort: op.KubernetesPort,
		ConsolePort:    op.ConsolePort,
	}

	err := bkeconsole.DeployConsoleAll(sRestartConfig, repo, op.OFVersion)
	if err != nil {
		return fmt.Errorf("deploy console: %w", err)
	}
	log.Info("The bke console deployment is complete")
	return nil
}

// 生成集群配置文件
func (op *Options) generateClusterConfig() {
	log.Info("Generate the cluster configuration file")

	data, repo, err := op.prepareClusterConfigData()
	if err != nil {
		log.Errorf("generateClusterConfig func is error %s", err)
		return
	}

	op.createClusterConfigFile(data, repo[0], repo[1], repo[2])
}

func (op *Options) prepareClusterConfigData() (map[string]string, []v1beta1.Repo, error) {
	k8sVersion, err := op.getStartKubernetesVersion()
	if err != nil {
		return nil, []v1beta1.Repo{}, err
	}

	data := map[string]string{
		"chartRepoPort":     fmt.Sprintf("%s", op.ChartRepoPort),
		"clusterapi":        op.ClusterAPI,
		"domain":            op.Domain,
		"host":              op.HostIP,
		"httpDomain":        configinit.DefaultYumRepo,
		"httpIp":            "",
		"httpRepo":          oc.Source,
		"imageRepoPort":     fmt.Sprintf("%s", op.ImageRepoPort),
		"ntpServer":         op.NtpServer,
		"otherRepo":         oc.Repo,
		"otherRepoIp":       oc.RepoIP,
		"runtime":           op.Runtime,
		"yumRepoPort":       fmt.Sprintf("%s", op.YumRepoPort),
		"kubernetesVersion": k8sVersion,
		"agentHealthPort":   op.AgentHealthPort,
	}

	patchesDir := filepath.Join(global.Workspace, utils.PatchDataDirectory)
	if op.ImageFilePath != "" {
		patchesDir = filepath.Join(global.Workspace, utils.LocalPatchDirectory)
	}

	if patchMap := op.ProcessPatchFiles(patchesDir); patchMap != nil {
		for k, v := range patchMap {
			data[k] = v
		}
	}

	imageRepo := op.prepareImageRepoConfig()

	yumRepo := op.prepareHTTPRepoConfig()

	chartRepo := op.prepareChartRepoConfig()

	for k, v := range global.CustomExtra {
		data[k] = v
	}

	return data, []v1beta1.Repo{imageRepo, yumRepo, chartRepo}, nil
}

func (op *Options) prepareImageRepoConfig() v1beta1.Repo {
	imageRepo := v1beta1.Repo{
		Domain: op.Domain,
		Ip:     op.HostIP,
		Port:   op.ImageRepoPort,
		Prefix: bkecommon.ImageRegistryKubernetes,
	}

	if oc.Repo != "" {
		img := strings.Split(oc.Repo, "/")
		img1 := strings.Split(img[0], ":")
		port := "443"
		if len(img1) == utils.HttpUrlFields {
			port = img1[1]
		}
		imageRepo = v1beta1.Repo{
			Domain: img1[0],
			Ip:     oc.RepoIP,
			Port:   port,
			Prefix: strings.TrimRight(strings.Join(img[1:], "/"), "/"),
		}
		return imageRepo
	}
	if oc.Image != "" { // 在线安装，未指定Repo时K8s资源yaml文件就使用默认的镜像，使用prefix做标识
		imageRepo.Prefix = ""
		imageRepo.Domain = "default"
	}

	return imageRepo
}

func (op *Options) prepareChartRepoConfig() v1beta1.Repo {
	ChartRepoIP, err := utils.LoopIP(configinit.DefaultChartRepo)
	if err == nil {
		chartRepo := v1beta1.Repo{
			Domain: configinit.DefaultChartRepo,
			Ip:     ChartRepoIP[0],
			Port:   "443",
			Prefix: "charts",
		}
		return chartRepo
	} else {
		chartRepo := v1beta1.Repo{
			Domain: "",
			Ip:     op.HostIP,
			Port:   op.ChartRepoPort,
			Prefix: "",
		}
		return chartRepo
	}
}

func (op *Options) prepareHTTPRepoConfig() v1beta1.Repo {
	yumRepo := v1beta1.Repo{
		Domain: configinit.DefaultYumRepo,
		Ip:     op.HostIP,
		Port:   op.YumRepoPort,
	}
	if oc.Source == "" {
		return yumRepo
	}

	host, port, prefix := parseHTTPSourceURL(oc.Source)
	yumRepo = v1beta1.Repo{
		Domain: configinit.DefaultYumRepo,
		Port:   port,
		Prefix: prefix,
	}
	if domain, ip := resolveHTTPSourceHost(host); domain != "" {
		yumRepo.Domain = domain
		yumRepo.Ip = ip
	} else {
		yumRepo.Ip = host
	}
	return yumRepo
}

func parseHTTPSourceURL(source string) (host, port, prefix string) {
	// otherSource 仅允许 http://，例如：
	// - http://example.com:40080/prefix
	source = strings.TrimSpace(source)
	if source == "" {
		return "", "", ""
	}

	u, err := url.Parse(source)
	if err != nil {
		httpRepo := strings.TrimPrefix(source, "http://")
		parts := strings.Split(httpRepo, ":")
		port = "80"
		if len(parts) == utils.HttpUrlFields {
			port = parts[1]
		}
		return parts[0], port, ""
	}

	host = u.Hostname()
	port = u.Port()
	if port == "" {
		port = "80"
	}

	// prefix：去掉前后斜杠，保证和 prepareImageRepoConfig 的 Prefix 习惯一致
	// 例如：/openFuyao/bkeadm/ -> openFuyao/bkeadm
	prefix = strings.Trim(u.Path, "/")
	return host, port, prefix
}

func resolveHTTPSourceHost(host string) (domain, ip string) {
	if net.ParseIP(host) != nil {
		return "", ""
	}
	global.CustomExtra["httpIp"] = host
	ips, err := utils.LoopIP(host)
	if err != nil || len(ips) == 0 {
		log.Warnf("Failed to resolve HTTP source host %s: %v", host, err)
		return host, ""
	}
	return host, ips[0]
}

func (op *Options) createClusterConfigFile(data map[string]string, imageRepo, yumRepo, chartRepo v1beta1.Repo) {
	err := bkeconfig.SetKubernetesConfig(data, bkecommon.BKEClusterConfigFileName, "cluster-system")
	if err != nil {
		log.Errorf("Failed to generate the cluster configuration file %v", err)
		return
	}

	if op.File != "" {
		return
	}

	// 创建集群配置
	c := config.Options{
		Directory:       fmt.Sprintf("%s/cluster", global.Workspace),
		Product:         "fuyao-allinone",
		Domain:          op.Domain,
		ImageRepoPort:   op.ImageRepoPort,
		NtpServer:       op.NtpServer,
		AgentHealthPort: op.AgentHealthPort,
	}
	c.Config(global.CustomExtra, imageRepo, yumRepo, chartRepo, op.NtpServer)

	log.Infof("Run `bke cluster create -f %s/cluster/1master-cluster.yaml -n %s/cluster/1master-node.yaml`"+
		"command to deploy the cluster", global.Workspace, global.Workspace)
}

func versionLess(v1, v2 string) bool {
	num1 := strings.TrimPrefix(v1, "v")
	num2 := strings.TrimPrefix(v2, "v")

	parts1 := strings.Split(num1, ".")
	parts2 := strings.Split(num2, ".")

	for i := 0; i < len(parts1) && i < len(parts2); i++ {
		if parts1[i] != parts2[i] {
			return parts1[i] < parts2[i]
		}
	}

	return len(parts1) < len(parts2)
}

func (op *Options) getStartKubernetesVersion() (string, error) {
	sourceRegistry := fmt.Sprintf("%s/mount/source_registry/files", global.Workspace)
	if !utils.Exists(sourceRegistry) {
		return versions.KubernetesVersion(), nil
	}
	entries, err := os.ReadDir(sourceRegistry)
	if err != nil {
		return "", err
	}
	arches := []string{"-arm64", "-amd64", "-x86_64", "-ppc64le", "-s390x"}
	re := regexp.MustCompile(`^kubectl-(v\d+\.\d+\.\d+(?:[-.][a-zA-Z0-9]+)*)$`)

	var versions []string

	for _, entry := range entries {
		base := entry.Name()
		for _, arch := range arches {
			if strings.HasSuffix(base, arch) {
				base = strings.TrimSuffix(base, arch)
				break
			}
		}
		matches := re.FindStringSubmatch(base)
		if len(matches) > 1 {
			versions = append(versions, matches[1])
		}
	}

	if len(versions) == 0 {
		return "", errors.New("no kubernetes version found")
	}

	sort.Slice(versions, func(i, j int) bool {
		return versionLess(versions[i], versions[j])
	})

	return versions[0], nil
}

func (op *Options) ProcessPatchFiles(patchesDir string) map[string]string {
	if _, err := os.Stat(patchesDir); os.IsNotExist(err) {
		return nil
	}

	entries, err := os.ReadDir(patchesDir)
	if err != nil {
		return nil
	}

	patchFiles := make(map[string]string)

	for _, entry := range entries {
		info, err := entry.Info()
		if err != nil {
			log.Warnf("failed to get file info for %s: %v", entry.Name(), err)
			continue
		}

		if info.Size() == 0 {
			log.Warnf("skipping empty file: %s", entry.Name())
			continue
		}
		version := extractVersionFromFilename(entry.Name())
		if version == "" {
			continue
		}
		fullPath := filepath.Join(patchesDir, entry.Name())
		// 根据文件内容构造数据，将patch内容写入到configmap，为了便于处理，统一使用 openfuyao-patch 命名空间
		bkeConfigMapKey := fmt.Sprintf("%s%s", utils.PatchKeyPrefix, version)
		patchConfigMapName := fmt.Sprintf("%s%s", utils.PatchValuePrefix, version)
		if err = bkeconfig.SetPatchConfig(version, fullPath, patchConfigMapName); err != nil {
			continue
		}
		patchFiles[bkeConfigMapKey] = patchConfigMapName
	}

	return patchFiles
}

func (op *Options) modifyPermission() {
	var workDir string
	if utils.IsFile("/opt/BKE_WORKSPACE") {
		f, err := os.ReadFile("/opt/BKE_WORKSPACE")
		if err == nil {
			workDir = string(f)
			workDir = strings.TrimSpace(workDir)
			workDir = strings.TrimRight(workDir, "\n")
			workDir = strings.TrimRight(workDir, "\r")
			workDir = strings.TrimRight(workDir, "\t")
		}
	}

	if os.Getenv("BKE_WORKSPACE") != "" {
		workDir = os.Getenv("BKE_WORKSPACE")
	}
	if workDir == "" {
		workDir = "/bke"
	}

	// modify permission: file 644 dir 755
	err := filepath.Walk(workDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			log.Infof("Walk %s err %v", workDir, err)
			return err
		}

		if info.IsDir() {
			err = os.Chmod(path, utils.DefaultDirPermission)
			if err != nil {
				log.Infof("path %s mod dir permission err %v", path, err)
			}
		} else {
			err = os.Chmod(path, utils.DefaultFilePermission)
			if err != nil {
				log.Infof("path %s mod file permission err %v", path, err)
			}
		}
		return nil
	})

	if err != nil {
		log.Infof("workDir %s mod permission err %v", workDir, err)
	} else {
		log.Infof("workDir %s mod permission success", workDir)
	}
}

func (op *Options) deployCluster() {
	if op.File == "" {
		return
	}
	log.Info("Starting to deploy the cluster...")
	c := cluster.Options{
		File:      op.File,
		NtpServer: op.NtpServer,
	}
	c.Cluster()
}
