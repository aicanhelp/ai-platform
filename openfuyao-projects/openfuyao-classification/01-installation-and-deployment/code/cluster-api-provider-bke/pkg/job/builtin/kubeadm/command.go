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

package kubeadm

import (
	"context"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/etcd"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"sigs.k8s.io/controller-runtime/pkg/client"

	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize/versions"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	bkevalidte "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/validation"
	backupPlugin "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/backup"
	containerdPlugin "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/containerruntime/containerd"
	downloadPlugin "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/downloader"
	certPlugin "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/kubeadm/certs"
	envPlugin "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/kubeadm/env"
	kubeletPlugin "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/kubeadm/kubelet"
	manifestsPlugin "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/kubeadm/manifests"
	nginxPlugin "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/kubeadm/nginx"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/cluster"
	bkeetcd "gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/etcd"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/mfutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/pkiutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/clusterutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	hostArch = runtime.GOARCH
	// IPIndex is the index of the IP address in the service subnet
	IPIndex = 10

	// RwRR is the permission of the file
	RwRR = 0644
	// RwxRxRx is the permission of the directory
	RwxRxRx = 0755
)

// installContainerdCommand used to install and configure containerd in the target cluster node
func (k *KubeadmPlugin) installContainerdCommand() error {
	cfg := bkeinit.BkeConfig(*k.boot.BkeConfig)
	if err := bkevalidte.ValidateCustomExtra(cfg.CustomExtra); err != nil {
		return err
	}
	baseUrl := clusterutil.BuildYumRepoDownloadBaseURL(cfg)
	containerd := cfg.CustomExtra["containerd"]
	containerd = strings.ReplaceAll(containerd, "{.arch}", hostArch)
	url := fmt.Sprintf("%s/%s", baseUrl, containerd)
	repo := cfg.ImageThirdRepo()
	// 后续写在common中
	sandboxImage := fmt.Sprintf("%s/kubernetes/pause:%s", strings.TrimRight(repo, "/"),
		versions.PauseImageTag())
	dataRoot := bkeinit.DefaultCRIContainerdDataRootDir
	if cfg.Cluster.ContainerRuntime.Param != nil {
		if v, ok := cfg.Cluster.ContainerRuntime.Param["data-root"]; ok {
			dataRoot = v
		}
	}

	command := []string{
		containerdPlugin.Name,
		fmt.Sprintf("url=%s", url),
		fmt.Sprintf("sandbox=%s", sandboxImage),
		fmt.Sprintf("repo=%s:%s", cfg.Cluster.ImageRepo.Domain, cfg.Cluster.ImageRepo.Port),
		fmt.Sprintf("runtime=%s", cfg.Cluster.ContainerRuntime.Runtime),
		fmt.Sprintf("dataRoot=%s", dataRoot),
	}

	cp := containerdPlugin.New(k.exec)
	if _, err := cp.Execute(k.execCtx, command); err != nil {
		return errors.Wrap(err, "failed to run containerd plugin")
	}
	return nil
}

// installKubeletCommand used to install kubelet in the target cluster node
// immutable 参数为 true 时跳过二进制下载（仅 worker join 场景使用）
func (k *KubeadmPlugin) installKubeletCommand(immutable bool) error {
	cfg := bkeinit.BkeConfig(*k.boot.BkeConfig)
	k8sVersion := cfg.Cluster.KubernetesVersion
	kubeletUrl := clusterutil.BuildYumRepoDownloadBaseURL(cfg)
	kubelet := fmt.Sprintf("kubelet-%s-%s", k8sVersion, hostArch)
	kubeletUrl = fmt.Sprintf("%s/%s", kubeletUrl, kubelet)
	log.Infof("kubelet download url: %s", kubeletUrl)

	command, err := k.buildKubeletCommand(cfg, kubeletUrl)
	if err != nil {
		return err
	}

	// 不可变 OS：kubelet 镜像内置，通过命令参数通知 plugin 跳过下载
	if immutable {
		command = append(command, "immutableOS=true")
	}

	kp := kubeletPlugin.New(k.k8sClient, k.exec)
	if _, err := kp.Execute(k.execCtx, command); err != nil {
		return errors.Wrap(err, "failed to run kubelet plugin, when install kubelet")
	}

	return nil
}

// buildKubeletCommand 构建kubelet安装命令
func (k *KubeadmPlugin) buildKubeletCommand(cfg bkeinit.BkeConfig, kubeletUrl string) ([]string, error) {
	// 获取集群DNS IP
	_, svcSubNet, err := net.ParseCIDR(cfg.Cluster.Networking.ServiceSubnet)
	if err != nil {
		return nil, err
	}
	clusterDNSIP, err := pkiutil.GetIndexedIP(svcSubNet, IPIndex)
	if err != nil {
		return nil, err
	}

	// 构建基础命令
	command := []string{
		kubeletPlugin.Name,
		fmt.Sprintf("phase=%s", utils.InitControlPlane), // 待删除参数
		fmt.Sprintf("certificatesDir=%s", cfg.Cluster.CertificatesDir),
		fmt.Sprintf("clusterDNSDomain=%s", cfg.Cluster.Networking.DNSDomain),
		fmt.Sprintf("kubernetesVersion=%s", strings.TrimPrefix(cfg.Cluster.KubernetesVersion, "v")),
		fmt.Sprintf("etcdVersion=%s", strings.TrimPrefix(cfg.Cluster.EtcdVersion, "v")),
		fmt.Sprintf("manifestDir=%s", cfg.Cluster.Kubelet.ManifestsDir),
		fmt.Sprintf("hostIP=%s", k.boot.HostIP),
		fmt.Sprintf("imageRepo=%s", cfg.ImageFuyaoRepo()),
		fmt.Sprintf("hostName=%s", k.boot.HostName),
		fmt.Sprintf("extraArgs=%s", getKubeletExtraArgs(k.boot)),
		fmt.Sprintf("extraVolumes=%s", getKubeletExtraVolumes(k.boot)),
		fmt.Sprintf("dataRootDir=%s", getKubeletDataRootDir(k.boot)),
		fmt.Sprintf("providerID=%s", generateProviderID(k.clusterName, k.boot.HostIP)),
		fmt.Sprintf("cgroupDriver=%s", getKubeletCgroupDriver(k.boot)),
		"rename=kubelet",
		"saveto=/usr/bin",
		"chmod=755",
		fmt.Sprintf("url=%s", kubeletUrl),
	}

	// 添加Kubelet配置引用
	if k.boot.KubeletConfigRef != nil && k.boot.KubeletConfigRef.Name != "" {
		command = append(command, "useDeliveredConfig=true")
		command = append(command, "enableVariableSubstitution=true")
		command = append(command, fmt.Sprintf("kubeletConfigName=%s", k.boot.KubeletConfigRef.Name))

		namespace := k.boot.KubeletConfigRef.Namespace
		if namespace == "" {
			namespace = k.GableNameSpace
		}
		command = append(command, fmt.Sprintf("kubeletConfigNamespace=%s", namespace))
	}

	// 检查是否启用nodelocaldns，决定clusterDNSIP的值
	allowNodeLocalDNS := false
	localdns := ""
	for _, Addon := range cfg.Addons {
		if Addon.Name == "nodelocaldns" {
			allowNodeLocalDNS = true
			localdns = Addon.Param["localdns"]
			break
		}
	}

	if allowNodeLocalDNS {
		proxyMode := cfg.CustomExtra["proxyMode"]
		if proxyMode == "ipvs" {
			command = append(command, fmt.Sprintf("clusterDNSIP=%s", localdns))
		}
	} else {
		command = append(command, fmt.Sprintf("clusterDNSIP=%s", clusterDNSIP.String()))
	}

	return command, nil
}

// installKubectlCommand used to install kubectl in the target cluster node
func (k *KubeadmPlugin) installKubectlCommand() error {
	cfg := bkeinit.BkeConfig(*k.boot.BkeConfig)
	k8sVersion := cfg.Cluster.KubernetesVersion

	kubectlUrl := clusterutil.BuildYumRepoDownloadBaseURL(cfg)
	kubectl := fmt.Sprintf("kubectl-%s-%s", k8sVersion, hostArch)
	kubectlUrl = fmt.Sprintf("%s/%s", kubectlUrl, kubectl)
	log.Infof("kubectl download url: %s", kubectlUrl)
	command := []string{
		downloadPlugin.Name,
		fmt.Sprintf("url=%s", kubectlUrl),
		"rename=kubectl",
		"saveto=/usr/bin",
		"chmod=755",
	}
	dp := downloadPlugin.New()
	if _, err := dp.Execute(k.execCtx, command); err != nil {
		// download kubectl failed, only warning, this will not affect the k8s deployment
		log.Warnf("failed to run download plugin, error: %v", err)
	}
	return nil
}

// initControlPlaneCertCommand used to download target cluster all certificates
// and generate kubeconfig for local kubelet
func (k *KubeadmPlugin) initControlPlaneCertCommand() error {
	return k.runControlPlaneCertCommand()
}

// initControlPlaneManifestCommand used to generate k8s components static pod yaml
func (k *KubeadmPlugin) initControlPlaneManifestCommand() error {
	return k.runControlPlaneManifestCommand()
}

// initControlPlaneKubeletCommand used to run kubelet  kubelet安装方式改为二进制部署，这里待删除
func (k *KubeadmPlugin) initControlPlaneKubeletCommand() error {
	cfg := bkeinit.BkeConfig(*k.boot.BkeConfig)

	_, svcSubnet, err := net.ParseCIDR(cfg.Cluster.Networking.ServiceSubnet)
	if err != nil {
		return err
	}
	clusterDNSIP, err := pkiutil.GetIndexedIP(svcSubnet, IPIndex)

	command := []string{
		fmt.Sprintf("hostName=%s", k.boot.HostName),
		kubeletPlugin.Name,
		fmt.Sprintf("phase=%s", utils.InitControlPlane),
		fmt.Sprintf("certificatesDir=%s", cfg.Cluster.CertificatesDir),
		fmt.Sprintf("dataRootDir=%s", getKubeletDataRootDir(k.boot)),
		fmt.Sprintf("clusterDNSDomain=%s", cfg.Cluster.Networking.DNSDomain),
		fmt.Sprintf("clusterDNSIP=%s", clusterDNSIP.String()),
		fmt.Sprintf("kubernetesVersion=%s", cfg.Cluster.KubernetesVersion),
		fmt.Sprintf("manifestDir=%s", cfg.Cluster.Kubelet.ManifestsDir),
		fmt.Sprintf("imageRepo=%s", cfg.ImageFuyaoRepo()),
		"generateKubeletConfig=true",
		fmt.Sprintf("hostIP=%s", k.boot.HostIP),
		fmt.Sprintf("extraArgs=%s", getKubeletExtraArgs(k.boot)),
		fmt.Sprintf("extraVolumes=%s", getKubeletExtraVolumes(k.boot)),
		fmt.Sprintf("providerID=%s", generateProviderID(k.clusterName, k.boot.HostIP)),
		fmt.Sprintf("cgroupDriver=%s", getKubeletCgroupDriver(k.boot)),
	}
	kp := kubeletPlugin.New(k.k8sClient, k.exec)
	if _, err := kp.Execute(k.execCtx, command); err != nil {
		return errors.Wrap(err, "failed to run kubelet plugin when init control plane")
	}

	return nil
}

// joinWorkerCertCommand used to download ca and kubeconfig files from manager cluster
func (k *KubeadmPlugin) joinWorkerCertCommand() error {
	command := []string{
		certPlugin.Name,
		fmt.Sprintf("clusterName=%s", k.clusterName),
		fmt.Sprintf("namespace=%s", k.GableNameSpace),
		fmt.Sprintf("certificatesDir=%s", k.boot.BkeConfig.Cluster.CertificatesDir),

		"generate=false",
		"generateKubeConfig=true",
		"localKubeConfigScope=kubelet,kube-proxy",
		"loadCACert=true",
		"caCertNames=ca,proxy",
		"loadTargetClusterCert=false",
		"loadAdminKubeconfig=true",
		"uploadCerts=false",
		"tlsScope=tls-server",
	}

	cp := certPlugin.New(k.k8sClient, k.exec, k.boot.BkeConfig)
	// loadClusterAPICert is true, uploadCerts is false, caCertSecrets is "ca"
	if _, err := cp.Execute(k.execCtx, command); err != nil {
		return errors.Wrap(err, "failed to run cert plugin")
	}

	return nil
}

// joinWorkerNginxCommand deploys nginx load balancer on worker nodes (loadBalanceType=nginx only)
func (k *KubeadmPlugin) joinWorkerNginxCommand() error {
	if clusterutil.LoadBalanceType(k.boot.BkeConfig) != "nginx" {
		return nil
	}

	nodesData, err := plugin.GetNodesDataFromNs(k.GableNameSpace, k.clusterName)
	if err != nil {
		return errors.Wrap(err, "failed to get nodes data")
	}
	masterNodes := bkenode.Nodes(nodesData).Master()
	if len(masterNodes) == 0 {
		return errors.New("no master nodes found")
	}
	var masterList []string
	for _, node := range masterNodes {
		masterList = append(masterList, node.IP)
	}

	apiServerPort := bkeinit.DefaultAPIBindPort
	if k.boot.BkeConfig.Cluster.APIServer != nil && k.boot.BkeConfig.Cluster.APIServer.Port != 0 {
		apiServerPort = int(k.boot.BkeConfig.Cluster.APIServer.Port)
	}

	command := []string{
		nginxPlugin.Name,
		fmt.Sprintf("masterNodes=%s", strings.Join(masterList, ",")),
		fmt.Sprintf("apiServerPort=%d", apiServerPort),
		fmt.Sprintf("manifestsDir=%s", "/etc/kubernetes/nginx-proxy"),
		fmt.Sprintf("manifestPath=%s", mfutil.GetDefaultManifestsPath()),
	}
	// Pass nginx image tag from BKEConfig when set (e.g., during upgrade);
	// falls back to plugin default when empty (initial install).
	if k.boot.BkeConfig.Cluster.NginxProxyVersion != "" {
		command = append(command, fmt.Sprintf("nginxImageTag=%s", k.boot.BkeConfig.Cluster.NginxProxyVersion))
	}

	np := nginxPlugin.New(k.exec, k.boot)
	if _, err := np.Execute(k.execCtx, command); err != nil {
		return errors.Wrap(err, "failed to run nginx plugin")
	}

	log.Infof("nginx load balancer deployed on worker node")
	return nil
}

// joinWorkerKubeletCommand used to run kubelet  kubelet安装方式改为二进制部署，这里待删除
func (k *KubeadmPlugin) joinWorkerKubeletCommand() error {
	cfg := bkeinit.BkeConfig(*k.boot.BkeConfig)
	_, svcSubnet, err := net.ParseCIDR(cfg.Cluster.Networking.ServiceSubnet)
	if err != nil {
		return err
	}
	clusterDNSIP, err := pkiutil.GetIndexedIP(svcSubnet, IPIndex)
	command := []string{
		fmt.Sprintf("imageRepo=%s", cfg.ImageFuyaoRepo()),
		kubeletPlugin.Name,
		fmt.Sprintf("phase=%s", utils.JoinWorker),
		fmt.Sprintf("certificatesDir=%s", cfg.Cluster.CertificatesDir),
		fmt.Sprintf("hostIP=%s", k.boot.HostIP),
		fmt.Sprintf("clusterDNSDomain=%s", cfg.Cluster.Networking.DNSDomain),
		fmt.Sprintf("clusterDNSIP=%s", clusterDNSIP.String()),
		fmt.Sprintf("extraVolumes=%s", getKubeletExtraVolumes(k.boot)),
		fmt.Sprintf("kubernetesVersion=%s", cfg.Cluster.KubernetesVersion),
		fmt.Sprintf("manifestDir=%s", cfg.Cluster.Kubelet.ManifestsDir),
		fmt.Sprintf("hostName=%s", k.boot.HostName),
		fmt.Sprintf("extraArgs=%s", getKubeletExtraArgs(k.boot)),
		fmt.Sprintf("providerID=%s", generateProviderID(k.clusterName, k.boot.HostIP)),
		fmt.Sprintf("dataRootDir=%s", getKubeletDataRootDir(k.boot)),
		"generateKubeletConfig=true",
		fmt.Sprintf("cgroupDriver=%s", getKubeletCgroupDriver(k.boot)),
	}

	kp := kubeletPlugin.New(k.k8sClient, k.exec)
	if _, err := kp.Execute(k.execCtx, command); err != nil {
		return errors.Wrap(err, "failed to run kubelet plugin in join worker")
	}

	return nil
}

// joinControlPlaneCertCommand used to download target cluster all certificates
// // and generate local kubeconfig for kubelet
func (k *KubeadmPlugin) joinControlPlaneCertCommand() error {
	return k.runControlPlaneCertCommand()
}

// runControlPlaneCertCommand executes the common certificate command for control plane nodes
func (k *KubeadmPlugin) runControlPlaneCertCommand() error {
	command := []string{
		certPlugin.Name,
		fmt.Sprintf("clusterName=%s", k.clusterName),
		fmt.Sprintf("namespace=%s", k.GableNameSpace),
		fmt.Sprintf("certificatesDir=%s", k.boot.BkeConfig.Cluster.CertificatesDir),
		"generate=false",
		"generateKubeConfig=true",
		"loadCACert=false",
		"loadTargetClusterCert=true",
		"loadAdminKubeconfig=false",
		"uploadCerts=false",
		"tlsScope=tls-server",
		fmt.Sprintf("isManagerCluster=%v", k.isManager),
	}
	cp := certPlugin.New(k.k8sClient, k.exec, k.boot.BkeConfig)
	if _, err := cp.Execute(k.execCtx, command); err != nil {
		return errors.Wrap(err, "failed to run cert plugin")
	}
	return nil
}

// joinControlPlaneManifestCommand used to generate k8s components static pod yaml
func (k *KubeadmPlugin) joinControlPlaneManifestCommand() error {
	return k.runControlPlaneManifestCommand()
}

// runControlPlaneManifestCommand executes the common manifest command for control plane nodes
func (k *KubeadmPlugin) runControlPlaneManifestCommand() error {
	// etcd data dir
	etcdDataDir := mfutil.EtcdDataDir
	if k.boot.CurrentNode.Etcd != nil && k.boot.CurrentNode.Etcd.DataDir != "" {
		etcdDataDir = k.boot.CurrentNode.Etcd.DataDir
	} else {
		etcdDataDir = k.boot.BkeConfig.Cluster.Etcd.DataDir
	}
	manifestsDir := mfutil.GetDefaultManifestsPath()
	if k.boot.CurrentNode.Kubelet != nil && k.boot.CurrentNode.Kubelet.ManifestsDir != "" {
		manifestsDir = k.boot.CurrentNode.Kubelet.ManifestsDir
	} else {
		manifestsDir = k.boot.BkeConfig.Cluster.Kubelet.ManifestsDir
	}
	command := []string{
		manifestsPlugin.Name,
		"scope=kube-apiserver,kube-controller-manager,kube-scheduler,etcd",
		fmt.Sprintf("manifestDir=%s", manifestsDir),
		fmt.Sprintf("etcdDataDir=%s", etcdDataDir),
	}
	mf := manifestsPlugin.New(k.boot, k.exec)
	if _, err := mf.Execute(k.execCtx, command); err != nil {
		return errors.Wrap(err, "failed to run manifests plugin")
	}
	return nil
}

// joinControlPlaneKubeletCommand used to run kubelet join control plane   kubelet安装方式改为二进制部署，这里待删除
func (k *KubeadmPlugin) joinControlPlaneKubeletCommand() error {
	cfg := bkeinit.BkeConfig(*k.boot.BkeConfig)
	_, svcSubnet, err := net.ParseCIDR(cfg.Cluster.Networking.ServiceSubnet)
	if err != nil {
		return err
	}
	clusterDNSIP, err := pkiutil.GetIndexedIP(svcSubnet, IPIndex)
	command := []string{
		kubeletPlugin.Name,
		fmt.Sprintf("phase=%s", utils.JoinControlPlane),
		fmt.Sprintf("extraArgs=%s", getKubeletExtraArgs(k.boot)),
		fmt.Sprintf("extraVolumes=%s", getKubeletExtraVolumes(k.boot)),
		fmt.Sprintf("providerID=%s", generateProviderID(k.clusterName, k.boot.HostIP)),
		fmt.Sprintf("certificatesDir=%s", cfg.Cluster.CertificatesDir),
		fmt.Sprintf("clusterDNSDomain=%s", cfg.Cluster.Networking.DNSDomain),
		fmt.Sprintf("clusterDNSIP=%s", clusterDNSIP.String()),
		fmt.Sprintf("kubernetesVersion=%s", cfg.Cluster.KubernetesVersion),
		fmt.Sprintf("manifestDir=%s", cfg.Cluster.Kubelet.ManifestsDir),
		fmt.Sprintf("imageRepo=%s", cfg.ImageFuyaoRepo()),
		fmt.Sprintf("hostName=%s", k.boot.HostName),
		fmt.Sprintf("hostIP=%s", k.boot.HostIP),
		fmt.Sprintf("dataRootDir=%s", getKubeletDataRootDir(k.boot)),
		"generateKubeletConfig=true",
		fmt.Sprintf("cgroupDriver=%s", getKubeletCgroupDriver(k.boot)),
	}
	kp := kubeletPlugin.New(k.k8sClient, k.exec)
	if _, err := kp.Execute(k.execCtx, command); err != nil {
		return errors.Wrap(err, "failed to run kubelet plugin in join control plane")
	}

	return nil
}

func (k *KubeadmPlugin) upgradeControlPlaneManifestCommand(scopes ...string) error {
	if len(scopes) == 0 {
		return nil
	}
	scope := strings.Join(scopes, ",")
	command := []string{
		manifestsPlugin.Name,
		fmt.Sprintf("scope=%s", scope),
		"check=true",
	}
	mf := manifestsPlugin.New(k.boot, k.exec)
	if _, err := mf.Execute(k.execCtx, command); err != nil {
		return errors.Wrap(err, "failed to run manifests plugin")
	}
	return nil
}

// kubelet安装方式改为二进制部署，这里待删除
func (k *KubeadmPlugin) upgradeKubeletCommand(phase string) error {
	cfg := bkeinit.BkeConfig(*k.boot.BkeConfig)
	_, svcSubnet, err := net.ParseCIDR(cfg.Cluster.Networking.ServiceSubnet)
	if err != nil {
		return err
	}
	clusterDNSIP, err := pkiutil.GetIndexedIP(svcSubnet, IPIndex)
	command := []string{
		kubeletPlugin.Name,
		fmt.Sprintf("phase=%s", phase),
		fmt.Sprintf("extraArgs=%s", getKubeletExtraArgs(k.boot)),
		fmt.Sprintf("extraVolumes=%s", getKubeletExtraVolumes(k.boot)),
		fmt.Sprintf("providerID=%s", generateProviderID(k.clusterName, k.boot.HostIP)),
		fmt.Sprintf("certificatesDir=%s", cfg.Cluster.CertificatesDir),
		fmt.Sprintf("kubernetesVersion=%s", cfg.Cluster.KubernetesVersion),
		fmt.Sprintf("manifestDir=%s", cfg.Cluster.Kubelet.ManifestsDir),
		fmt.Sprintf("clusterDNSDomain=%s", cfg.Cluster.Networking.DNSDomain),
		fmt.Sprintf("clusterDNSIP=%s", clusterDNSIP.String()),
		fmt.Sprintf("imageRepo=%s", cfg.ImageFuyaoRepo()),
		fmt.Sprintf("hostName=%s", k.boot.HostName),
		fmt.Sprintf("hostIP=%s", k.boot.HostIP),
		"generateKubeletConfig=true",
		fmt.Sprintf("dataRootDir=%s", getKubeletDataRootDir(k.boot)),
		fmt.Sprintf("cgroupDriver=%s", getKubeletCgroupDriver(k.boot)),
	}
	kp := kubeletPlugin.New(k.k8sClient, k.exec)
	if _, err := kp.Execute(k.execCtx, command); err != nil {
		return errors.Wrap(err, "failed to run kubelet plugin")
	}

	return nil
}

func (k *KubeadmPlugin) upgradePrePullImageCommand() error {
	command := []string{
		"K8sEnvInit",
		"init=true",
		"check=true",
		"scope=image",
	}
	ep := envPlugin.New(k.exec, k.boot.BkeConfig)
	if _, err := ep.Execute(k.execCtx, command); err != nil {
		return errors.Wrap(err, "failed to run env plugin")
	}
	return nil
}

func (k *KubeadmPlugin) backupClusterEtc(clusterType string) error {
	// backup /etc/kubernetes it's a must
	dirs := []string{
		"/etc/kubernetes",
	}
	command := []string{
		backupPlugin.Name,
		fmt.Sprintf("backupDirs=%s", strings.Join(dirs, ",")),
	}
	bp := backupPlugin.New(k.exec)
	if _, err := bp.Execute(k.execCtx, command); err != nil {
		return errors.Wrap(err, "failed to run backup plugin")
	}
	return nil
}

func (k *KubeadmPlugin) backupEtcd() error {
	client, err := kubeadmutil.ClientSetFromFile(pkiutil.GetDefaultKubeConfigPath())
	if err != nil {
		return err
	}
	certDir := k.boot.BkeConfig.Cluster.CertificatesDir
	if certDir == "" {
		certDir = pkiutil.GetDefaultPkiPath()
	}

	etcdClient, err := etcd.NewFromCluster(client, certDir)
	if err != nil {
		return errors.Wrap(err, "failed to create etcd client")
	}

	backupName := fmt.Sprintf("backup-%s.db", time.Now().Format("0601021504"))

	backupDir := filepath.Join(utils.Workspace, "etcd-backup")

	if !utils.Exists(backupDir) {
		if err := os.Mkdir(backupDir, RwRR); err != nil {
			return err
		}
	}

	backupPath := filepath.Join(backupDir, backupName)

	return bkeetcd.Save(etcdClient, backupPath)
}

func getKubeletDataRootDir(boot *mfutil.BootScope) string {
	if boot.CurrentNode.Kubelet != nil && boot.CurrentNode.Kubelet.ExtraVolumes != nil {
		for _, v := range boot.CurrentNode.Kubelet.ExtraVolumes {
			if v.Name == "kubelet-root-dir" {
				return v.HostPath
			}
		}
		return bkeinit.DefaultKubeletRootDir
	}
	if boot.BkeConfig.Cluster.Kubelet != nil && boot.BkeConfig.Cluster.Kubelet.ExtraVolumes != nil {
		for _, v := range boot.BkeConfig.Cluster.Kubelet.ExtraVolumes {
			if v.Name == "kubelet-root-dir" {
				return v.HostPath
			}
		}
	}
	return bkeinit.DefaultKubeletRootDir
}

func getKubeletExtraArgs(boot *mfutil.BootScope) string {
	extraArgs := make(map[string]string)
	nodesData, err := cluster.GetNodesData(boot.ClusterNamespace, boot.ClusterName)
	if err != nil {
		return ""
	}
	nodes := bkenode.Nodes(nodesData)
	currentNode, err := nodes.CurrentNode()
	if err != nil {
		return ""
	}
	if currentNode.Kubelet != nil && currentNode.Kubelet.ExtraArgs != nil {
		extraArgs = currentNode.Kubelet.ExtraArgs
	} else {
		extraArgs = boot.BkeConfig.Cluster.Kubelet.ExtraArgs
	}

	var args []string

	for k, v := range extraArgs {
		args = append(args, fmt.Sprintf("--%s=%s", k, v))
	}

	return strings.Join(args, ";")
}

// processVolume processes a single volume and appends it to extraVolumes if valid
func processVolume(hostPath, mountPath, name string, extraVolumes *[]string) {
	if !utils.Exists(hostPath) {
		if err := os.MkdirAll(hostPath, RwxRxRx); err != nil {
			log.Warnf("failed to create host dir %s: %v", hostPath, err)
		}
	}
	if name == "kubelet-root-dir" {
		return
	}
	volume := fmt.Sprintf("%s:%s", hostPath, mountPath)
	*extraVolumes = append(*extraVolumes, volume)
}

func getKubeletExtraVolumes(boot *mfutil.BootScope) string {
	var extraVolumes []string
	nodesData, err := cluster.GetNodesData(boot.ClusterNamespace, boot.ClusterName)
	if err != nil {
		return ""
	}
	nodes := bkenode.Nodes(nodesData)
	currentNode, err := nodes.CurrentNode()
	if err != nil {
		return ""
	}
	appendConfiguredKubeletVolumes(boot, currentNode, &extraVolumes)
	appendRuntimeDataVolume(boot, &extraVolumes)
	appendEtcdDataVolume(boot, currentNode, &extraVolumes)

	extraVolumes = append(extraVolumes, fmt.Sprintf(
		"%s:%s", boot.BkeConfig.Cluster.CertificatesDir, pkiutil.GetDefaultPkiPath(),
	))
	extraVolumes = append(extraVolumes, fmt.Sprintf(
		"%s:%s", boot.BkeConfig.Cluster.Kubelet.ManifestsDir, mfutil.GetDefaultManifestsPath(),
	))
	return strings.Join(utils.UniqueStringSlice(extraVolumes), ";")
}

func appendConfiguredKubeletVolumes(
	boot *mfutil.BootScope,
	currentNode bkenode.Node,
	extraVolumes *[]string,
) {
	if currentNode.Kubelet != nil && currentNode.Kubelet.ExtraVolumes != nil {
		for _, v := range currentNode.Kubelet.ExtraVolumes {
			processVolume(v.HostPath, v.MountPath, v.Name, extraVolumes)
		}
		return
	}
	if boot.BkeConfig.Cluster.Kubelet == nil {
		return
	}
	for _, v := range boot.BkeConfig.Cluster.Kubelet.ExtraVolumes {
		processVolume(v.HostPath, v.MountPath, v.Name, extraVolumes)
	}
}

func appendRuntimeDataVolume(boot *mfutil.BootScope, extraVolumes *[]string) {
	cri := boot.BkeConfig.Cluster.ContainerRuntime.CRI
	if cri == bkeinit.CRIDocker {
		dataRoot := containerRuntimeDataRoot(boot, bkeinit.DefaultCRIDockerDataRootDir)
		*extraVolumes = append(*extraVolumes, fmt.Sprintf("%s:%s:rw,rslave", dataRoot, dataRoot))
	}
	if cri == bkeinit.CRIContainerd {
		dataRoot := containerRuntimeDataRoot(boot, bkeinit.DefaultCRIContainerdDataRootDir)
		*extraVolumes = append(*extraVolumes, fmt.Sprintf("%s:%s", dataRoot, dataRoot))
	}
}

func containerRuntimeDataRoot(boot *mfutil.BootScope, defaultDataRoot string) string {
	params := boot.BkeConfig.Cluster.ContainerRuntime.Param
	if params != nil {
		if dataRoot, ok := params["data-root"]; ok {
			return dataRoot
		}
	}
	return defaultDataRoot
}

func appendEtcdDataVolume(
	boot *mfutil.BootScope,
	currentNode bkenode.Node,
	extraVolumes *[]string,
) {
	if !currentNode.IsEtcd() {
		return
	}
	if currentNode.Etcd != nil && currentNode.Etcd.DataDir != "" {
		*extraVolumes = append(*extraVolumes, fmt.Sprintf(
			"%s:%s", currentNode.Etcd.DataDir, currentNode.Etcd.DataDir,
		))
		return
	}
	if boot.BkeConfig.Cluster.Etcd != nil && boot.BkeConfig.Cluster.Etcd.DataDir != "" {
		dataDir := boot.BkeConfig.Cluster.Etcd.DataDir
		*extraVolumes = append(*extraVolumes, fmt.Sprintf("%s:%s", dataDir, dataDir))
		return
	}
	log.Debug("etcd data directory is not configured; no extra volume is required")
}

func getKubeletCgroupDriver(boot *mfutil.BootScope) string {
	if boot.BkeConfig.Cluster.ContainerRuntime.Param != nil {
		if v, ok := boot.BkeConfig.Cluster.ContainerRuntime.Param["cgroupDriver"]; ok && v != "" {
			return v
		}
	}
	return bkeinit.DefaultCgroupDriver
}

func generateProviderID(clusterName, hostIP string) string {
	return fmt.Sprintf("bke://%s/%s", clusterName, utils.B64Encode(hostIP))
}

// 加载本地存储的Global CA 保存到Secret中
func (k *KubeadmPlugin) uploadUserCustomConfigAndGlobalCA() error {

	kubeconfigPath := pkiutil.GetDefaultKubeConfigPath()
	config, err := clientcmd.LoadFromFile(kubeconfigPath)
	if err != nil {
		return errors.Wrap(err, "failed to load admin kubeconfig")
	}
	overrides := clientcmd.ConfigOverrides{Timeout: "10s"}
	restConfig, err := clientcmd.NewDefaultClientConfig(*config, &overrides).ClientConfig()
	if err != nil {
		return errors.Wrap(err, "failed to get rest config")
	}

	currentClusterClient, err := client.New(restConfig, client.Options{})
	if err != nil {
		return errors.Wrap(err, "failed to create controller-runtime client for current cluster")
	}

	// 等待集群 API Server 准备就绪
	if err := k.waitForClusterReady(currentClusterClient); err != nil {
		return errors.Wrap(err, "cluster is not ready")
	}
	log.Infof("Cluster API server is ready")
	log.Infof("Begin to upload Global CA to Secret")
	if err := k.uploadGlobalCAAndCertChainToSecret(currentClusterClient); err != nil {
		return errors.Wrap(err, "failed to upload user's Global CA to cluster secret")
	}

	log.Infof("Begin to upload certification config to configmap")
	if err := k.createConfigMapForCertConfig(currentClusterClient); err != nil {
		return errors.Wrap(err, "failed to create user's configmap for cert")
	}

	return nil
}

// waitForClusterReady 等待集群 API Server 准备就绪
func (k *KubeadmPlugin) waitForClusterReady(currentClusterClient client.Client) error {
	const (
		maxRetries    = 5
		retryInterval = 5 * time.Second
		// timeout 设置为最大重试时间 + 缓冲时间，防止某次 API 调用卡住
		timeout = time.Duration(maxRetries+1) * retryInterval
	)

	ctx, cancel := context.WithTimeout(k.contextOrBackground(), timeout)
	defer cancel()

	log.Infof("Waiting for kube-system namespace to be created (max %d attempts, timeout %v)...", maxRetries, timeout)
	for attempt := 1; attempt <= maxRetries; attempt++ {
		select {
		case <-ctx.Done():
			return errors.Errorf("timeout waiting for cluster to be ready after %v", timeout)
		default:
		}

		// 等待 kube-system 命名空间被自动创建
		ns := &corev1.Namespace{}
		if err := currentClusterClient.Get(ctx, client.ObjectKey{Name: pkiutil.CertConfigMapNamespace}, ns); err == nil {
			log.Infof("Namespace %s is available (attempt %d/%d)", pkiutil.CertConfigMapNamespace, attempt, maxRetries)
			return nil
		} else {
			if !apierrors.IsNotFound(err) {
				// 非 NotFound 错误，打印调试日志但继续重试
				log.Debugf("Waiting for namespace %s: transient error: %v", pkiutil.CertConfigMapNamespace, err)
			}
		}

		log.Debugf("Namespace %s not ready yet ", pkiutil.CertConfigMapNamespace)
		log.Debugf("(attempt %d/%d), retrying in %v...", attempt, maxRetries, retryInterval)
		timer := time.NewTimer(retryInterval)
		select {
		case <-ctx.Done():
			timer.Stop()
			return errors.Errorf("timeout waiting for cluster to be ready after %v: %v", timeout, ctx.Err())
		case <-timer.C:
		}
	}

	return errors.Errorf("cluster API Server is not ready after %d attempts", maxRetries)
}

// uploadGlobalCAAndCertChainToSecret save global ca crt and key and certificate chain to secret
func (k *KubeadmPlugin) uploadGlobalCAAndCertChainToSecret(currentClusterClient client.Client) error {
	globalCACert := &pkiutil.BKECert{
		Name:     "global-ca",
		BaseName: pkiutil.GlobalCACertAndKeyBaseName,
		IsCA:     true,
		Config:   pkiutil.CertConfig{},
	}

	// 检查本地全局 CA 证书文件是否存在
	if err := pkiutil.CertExists(globalCACert); err != nil {
		return errors.Wrap(err, "global CA certificate not found in local filesystem")
	}

	// 加载本地全局 CA 到 Secret
	if err := pkiutil.SaveGlobalCAAndCertChainToSecret(currentClusterClient, globalCACert); err != nil {
		return errors.Wrap(err, "failed to upload global CA to cluster secret")
	}

	log.Infof("Successfully uploaded global CA to cluster secret %s/%s", utils.GlobalCANamespace, utils.GlobalCASecretName)
	return nil
}

// createConfigMapForCertConfig load csr config and signing policy config to ConfigMap
func (k *KubeadmPlugin) createConfigMapForCertConfig(currentClusterClient client.Client) error {
	log.Infof("creating cert config ConfigMap %s/%s from local dir %s",
		pkiutil.CertConfigMapNamespace, pkiutil.CertConfigMapName, pkiutil.CertConfigDir)
	if !utils.Exists(pkiutil.CertConfigDir) {
		log.Infof("local dir %s not exists, skip", pkiutil.CertConfigDir)
		return nil
	}
	files, err := os.ReadDir(pkiutil.CertConfigDir)
	if err != nil {
		return errors.Wrapf(err, "read dir %s", pkiutil.CertConfigDir)
	}
	data := make(map[string]string)
	for _, f := range files {
		name := f.Name()
		if !strings.HasSuffix(name, ".json") {
			log.Debugf("skip non-json file %s", name)
			continue
		}
		content, rerr := os.ReadFile(filepath.Join(pkiutil.CertConfigDir, name))
		if rerr != nil {
			log.Warnf("read file %s failed: %v", name, rerr)
			continue
		}
		data[name] = string(content)
	}
	if len(data) == 0 {
		log.Infof("no json files found under %s, skip creating ConfigMap", pkiutil.CertConfigDir)
		return nil
	}

	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: pkiutil.CertConfigMapName, Namespace: pkiutil.CertConfigMapNamespace},
		Data:       data}
	var loggedUpdate bool
	return phaseutil.RetryOnConflict(func() error {
		existingConfigMap := &corev1.ConfigMap{}
		err = currentClusterClient.Get(
			context.Background(),
			client.ObjectKey{Name: cm.Name, Namespace: cm.Namespace},
			existingConfigMap,
		)
		if apierrors.IsNotFound(err) {
			log.Infof("ConfigMap %s/%s not exist, creating", cm.Namespace, cm.Name)
			return currentClusterClient.Create(context.Background(), cm)
		}
		if err != nil {
			return errors.Wrap(err, "get ConfigMap")
		}
		existingConfigMap.Data = data
		if !loggedUpdate {
			log.Infof("updating existing ConfigMap %s/%s with %d files", cm.Namespace, cm.Name, len(data))
			loggedUpdate = true
		}
		return currentClusterClient.Update(context.Background(), existingConfigMap)
	})
}
