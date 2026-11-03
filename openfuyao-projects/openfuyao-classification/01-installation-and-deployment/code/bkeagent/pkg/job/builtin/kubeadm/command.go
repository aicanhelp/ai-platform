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
	bkeinit "gopkg.openfuyao.cn/bkecommon/cluster/initialize"
	bkenode "gopkg.openfuyao.cn/bkecommon/cluster/node"
	bkevalidte "gopkg.openfuyao.cn/bkecommon/cluster/validation"
	bkesource "gopkg.openfuyao.cn/bkecommon/source"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/etcd"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"sigs.k8s.io/controller-runtime/pkg/client"

	backupPlugin "gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/backup"
	containerdPlugin "gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/containerruntime/containerd"
	downloadPlugin "gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/downloader"
	certPlugin "gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/kubeadm/certs"
	envPlugin "gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/kubeadm/env"
	kubeletPlugin "gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/kubeadm/kubelet"
	manifestsPlugin "gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/kubeadm/manifests"
	"gopkg.openfuyao.cn/bkeagent/utils"
	bkeetcd "gopkg.openfuyao.cn/bkeagent/utils/etcd"
	"gopkg.openfuyao.cn/bkeagent/utils/log"
	"gopkg.openfuyao.cn/bkeagent/utils/mfutil"
	"gopkg.openfuyao.cn/bkeagent/utils/pkiutil"
)

const (
	hostArch = runtime.GOARCH
	// IPIndex is the index of the IP address in the service subnet
	IPIndex = 10
)

// installContainerdCommand used to install and configure containerd in the target cluster node
func (k *KubeadmPlugin) installContainerdCommand() error {
	cfg := bkeinit.BkeConfig(*k.boot.BkeConfig)
	if err := bkevalidte.ValidateCustomExtra(cfg.CustomExtra); err != nil {
		return err
	}
	baseUrl := bkesource.GetCustomDownloadPath(cfg.YumRepo())
	containerd := cfg.CustomExtra["containerd"]
	containerd = strings.ReplaceAll(containerd, "{.arch}", hostArch)
	url := fmt.Sprintf("%s/%s", baseUrl, containerd)
	repo := cfg.ImageRepo()
	// 后续写在common中
	sandboxImage := fmt.Sprintf("%s/kubernetes/pause:%s", strings.TrimRight(repo, "/"),
		bkeinit.DefaultPauseImageTag)
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
	if _, err := cp.Execute(command); err != nil {
		return errors.Wrap(err, "failed to run containerd plugin")
	}
	return nil
}

// installKubeletCommand used to install kubelet in the target cluster node
func (k *KubeadmPlugin) installKubeletCommand() error {
	cfg := bkeinit.BkeConfig(*k.boot.BkeConfig)
	k8sVersion := cfg.Cluster.KubernetesVersion
	kubeletUrl := bkesource.GetCustomDownloadPath(cfg.YumRepo())
	kubelet := fmt.Sprintf("kubelet-%s-%s", k8sVersion, hostArch)
	kubeletUrl = fmt.Sprintf("%s/%s", kubeletUrl, kubelet)
	log.Infof("kubelet download url: %s", kubeletUrl)

	_, svcSubNet, err := net.ParseCIDR(cfg.Cluster.Networking.ServiceSubnet)
	if err != nil {
		return err
	}
	clusterDNSIP, err := pkiutil.GetIndexedIP(svcSubNet, IPIndex)

	command := []string{
		kubeletPlugin.Name,
		fmt.Sprintf("phase=%s", utils.InitControlPlane), // 待删除参数
		fmt.Sprintf("clusterDNSIP=%s", clusterDNSIP.String()),
		fmt.Sprintf("certificatesDir=%s", cfg.Cluster.CertificatesDir),
		fmt.Sprintf("clusterDNSDomain=%s", cfg.Cluster.Networking.DNSDomain),
		fmt.Sprintf("kubernetesVersion=%s", strings.TrimPrefix(cfg.Cluster.KubernetesVersion, "v")),
		fmt.Sprintf("manifestDir=%s", cfg.Cluster.Kubelet.ManifestsDir),
		fmt.Sprintf("hostIP=%s", k.boot.HostIP),
		fmt.Sprintf("imageRepo=%s", cfg.ImageRepo()),
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

	// bccluster CR中不填KubeletConfigRef 或者 KubeletConfigRef.Name为空就是默认的值
	if k.boot.KubeletConfigRef != nil && k.boot.KubeletConfigRef.Name != "" {
		command = append(command, "useDeliveredConfig=true")
		command = append(command, "enableVariableSubstitution=true")
		command = append(command, fmt.Sprintf("kubeletConfigName=%s", k.boot.KubeletConfigRef.Name))
		if k.boot.KubeletConfigRef.Namespace != "" {
			command = append(command, fmt.Sprintf("kubeletConfigNamespace=%s", k.boot.KubeletConfigRef.Namespace))
		} else {
			command = append(command, fmt.Sprintf("kubeletConfigNamespace=%s", k.GableNameSpace))
		}
	}

	kp := kubeletPlugin.New(k.k8sClient, k.exec)
	if _, err := kp.Execute(command); err != nil {
		return errors.Wrap(err, "failed to run kubelet plugin, when install kubelet")
	}

	return nil
}

// installKubectlCommand used to install kubectl in the target cluster node
func (k *KubeadmPlugin) installKubectlCommand() error {
	cfg := bkeinit.BkeConfig(*k.boot.BkeConfig)
	k8sVersion := cfg.Cluster.KubernetesVersion

	kubectlUrl := bkesource.GetCustomDownloadPath(cfg.YumRepo())
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
	if _, err := dp.Execute(command); err != nil {
		// download kubectl failed, only warning, this will not affect the k8s deployment
		log.Warnf("failed to run download plugin, error: %v", err)
	}
	return nil
}

// initControlPlaneCertCommand used to download target cluster all certificates
// and generate kubeconfig for local kubelet
func (k *KubeadmPlugin) initControlPlaneCertCommand() error {

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
		fmt.Sprintf("isManagerCluster=%v", k.isManager),
	}

	cp := certPlugin.New(k.k8sClient, k.exec, k.boot.BkeConfig)
	// change loadClusterAPICert = true, uploadCerts=true
	if _, err := cp.Execute(command); err != nil {
		return errors.Wrap(err, "failed to run cert plugin")
	}

	return nil
}

// initControlPlaneManifestCommand used to generate k8s components static pod yaml
func (k *KubeadmPlugin) initControlPlaneManifestCommand() error {
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
	if _, err := mf.Execute(command); err != nil {
		return errors.Wrap(err, "failed to run manifests plugin")
	}
	return nil
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
		fmt.Sprintf("imageRepo=%s", cfg.ImageRepo()),
		"generateKubeletConfig=true",
		fmt.Sprintf("hostIP=%s", k.boot.HostIP),
		fmt.Sprintf("extraArgs=%s", getKubeletExtraArgs(k.boot)),
		fmt.Sprintf("extraVolumes=%s", getKubeletExtraVolumes(k.boot)),
		fmt.Sprintf("providerID=%s", generateProviderID(k.clusterName, k.boot.HostIP)),
		fmt.Sprintf("cgroupDriver=%s", getKubeletCgroupDriver(k.boot)),
	}
	kp := kubeletPlugin.New(k.k8sClient, k.exec)
	if _, err := kp.Execute(command); err != nil {
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
	}

	cp := certPlugin.New(k.k8sClient, k.exec, k.boot.BkeConfig)
	// loadClusterAPICert=true, uploadCerts=false, caCertSecrets="ca"
	if _, err := cp.Execute(command); err != nil {
		return errors.Wrap(err, "failed to run cert plugin")
	}

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
		fmt.Sprintf("imageRepo=%s", cfg.ImageRepo()),
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
	if _, err := kp.Execute(command); err != nil {
		return errors.Wrap(err, "failed to run kubelet plugin")
	}

	return nil
}

// joinControlPlaneCertCommand used to download target cluster all certificates
// // and generate local kubeconfig for kubelet
func (k *KubeadmPlugin) joinControlPlaneCertCommand() error {
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
		fmt.Sprintf("isManagerCluster=%v", k.isManager),
	}
	cp := certPlugin.New(k.k8sClient, k.exec, k.boot.BkeConfig)
	if _, err := cp.Execute(command); err != nil {
		return errors.Wrap(err, "failed to run cert plugin")
	}
	return nil
}

// joinControlPlaneManifestCommand used to generate k8s components static pod yaml
func (k *KubeadmPlugin) joinControlPlaneManifestCommand() error {
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
	if _, err := mf.Execute(command); err != nil {
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
		fmt.Sprintf("imageRepo=%s", cfg.ImageRepo()),
		fmt.Sprintf("hostName=%s", k.boot.HostName),
		fmt.Sprintf("hostIP=%s", k.boot.HostIP),
		fmt.Sprintf("dataRootDir=%s", getKubeletDataRootDir(k.boot)),
		"generateKubeletConfig=true",
		fmt.Sprintf("cgroupDriver=%s", getKubeletCgroupDriver(k.boot)),
	}
	kp := kubeletPlugin.New(k.k8sClient, k.exec)
	if _, err := kp.Execute(command); err != nil {
		return errors.Wrap(err, "failed to run kubelet plugin")
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
	if _, err := mf.Execute(command); err != nil {
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
		fmt.Sprintf("clusterDNSDomain=%s", cfg.Cluster.Networking.DNSDomain),
		fmt.Sprintf("clusterDNSIP=%s", clusterDNSIP.String()),
		fmt.Sprintf("kubernetesVersion=%s", cfg.Cluster.KubernetesVersion),
		fmt.Sprintf("manifestDir=%s", cfg.Cluster.Kubelet.ManifestsDir),
		fmt.Sprintf("imageRepo=%s", cfg.ImageRepo()),
		fmt.Sprintf("hostName=%s", k.boot.HostName),
		fmt.Sprintf("hostIP=%s", k.boot.HostIP),
		"generateKubeletConfig=true",
		fmt.Sprintf("dataRootDir=%s", getKubeletDataRootDir(k.boot)),
		fmt.Sprintf("cgroupDriver=%s", getKubeletCgroupDriver(k.boot)),
	}
	kp := kubeletPlugin.New(k.k8sClient, k.exec)
	if _, err := kp.Execute(command); err != nil {
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
	if _, err := ep.Execute(command); err != nil {
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
	if _, err := bp.Execute(command); err != nil {
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
		if err := os.Mkdir(backupDir, 0644); err != nil {
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
	} else if boot.BkeConfig.Cluster.Kubelet != nil && boot.BkeConfig.Cluster.Kubelet.ExtraVolumes != nil {
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
	nodes := bkenode.Nodes(boot.BkeConfig.Nodes)
	currentNode, err := nodes.CurrentNode()
	if err != nil {
		return ""
	}
	if currentNode.Kubelet != nil && currentNode.Kubelet.ExtraArgs != nil {
		extraArgs = currentNode.Kubelet.ExtraArgs
	} else {
		extraArgs = boot.BkeConfig.Cluster.Kubelet.ExtraArgs
	}

	args := []string{}

	for k, v := range extraArgs {
		args = append(args, fmt.Sprintf("--%s=%s", k, v))
	}

	return strings.Join(args, ";")
}

func getKubeletExtraVolumes(boot *mfutil.BootScope) string {
	extraVolumes := []string{}
	nodes := bkenode.Nodes(boot.BkeConfig.Nodes)
	currentNode, err := nodes.CurrentNode()
	if err != nil {
		return ""
	}
	if currentNode.Kubelet != nil && currentNode.Kubelet.ExtraVolumes != nil {
		for _, v := range currentNode.Kubelet.ExtraVolumes {
			if !utils.Exists(v.HostPath) {
				if err := os.MkdirAll(v.HostPath, 0755); err != nil {
					log.Warnf("failed to create host dir %s: %v", v.HostPath, err)
				}
			}
			if v.Name == "kubelet-root-dir" {
				continue
			}
			volume := fmt.Sprintf("%s:%s", v.HostPath, v.MountPath)
			extraVolumes = append(extraVolumes, volume)
		}
	} else if boot.BkeConfig.Cluster.Kubelet != nil && boot.BkeConfig.Cluster.Kubelet.ExtraVolumes != nil {
		for _, v := range boot.BkeConfig.Cluster.Kubelet.ExtraVolumes {
			if !utils.Exists(v.HostPath) {
				if err := os.MkdirAll(v.HostPath, 0755); err != nil {
					log.Warnf("failed to create host dir %s: %v", v.HostPath, err)
				}
			}
			if v.Name == "kubelet-root-dir" {
				continue
			}
			volume := fmt.Sprintf("%s:%s", v.HostPath, v.MountPath)
			extraVolumes = append(extraVolumes, volume)
		}
	}
	// 挂载 容器运行时目录
	if boot.BkeConfig.Cluster.ContainerRuntime.CRI == bkeinit.CRIDocker {
		dataRoot := bkeinit.DefaultCRIDockerDataRootDir
		if boot.BkeConfig.Cluster.ContainerRuntime.Param != nil {
			if v, ok := boot.BkeConfig.Cluster.ContainerRuntime.Param["data-root"]; ok {
				dataRoot = v
			}
		}
		volume := fmt.Sprintf("%s:%s:rw,rslave", dataRoot, dataRoot)
		extraVolumes = append(extraVolumes, volume)
	}
	if boot.BkeConfig.Cluster.ContainerRuntime.CRI == bkeinit.CRIContainerd {
		dataRoot := bkeinit.DefaultCRIContainerdDataRootDir
		if boot.BkeConfig.Cluster.ContainerRuntime.Param != nil {
			if v, ok := boot.BkeConfig.Cluster.ContainerRuntime.Param["data-root"]; ok {
				dataRoot = v
			}
		}
		volume := fmt.Sprintf("%s:%s", dataRoot, dataRoot)
		extraVolumes = append(extraVolumes, volume)
	}
	// 挂载etcd数据目录
	if currentNode.IsEtcd() {
		if currentNode.Etcd != nil && currentNode.Etcd.DataDir != "" {
			volume := fmt.Sprintf("%s:%s", currentNode.Etcd.DataDir, currentNode.Etcd.DataDir)
			extraVolumes = append(extraVolumes, volume)
		} else if boot.BkeConfig.Cluster.Etcd != nil && boot.BkeConfig.Cluster.Etcd.DataDir != "" {
			volume := fmt.Sprintf("%s:%s", boot.BkeConfig.Cluster.Etcd.DataDir, boot.BkeConfig.Cluster.Etcd.DataDir)
			extraVolumes = append(extraVolumes, volume)
		}
	}

	//所有节点都需要挂载证书目录 /etc/kubernetes/pki
	volume := fmt.Sprintf("%s:%s", boot.BkeConfig.Cluster.CertificatesDir, pkiutil.GetDefaultPkiPath())
	extraVolumes = append(extraVolumes, volume)

	// 所有节点还需要挂载manifests目录
	volume = fmt.Sprintf("%s:%s", boot.BkeConfig.Cluster.Kubelet.ManifestsDir, mfutil.GetDefaultManifestsPath())
	extraVolumes = append(extraVolumes, volume)

	//unique extraVolumes
	extraVolumes = utils.UniqueStringSlice(extraVolumes)
	volumes := strings.Join(extraVolumes, ";")

	return volumes
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

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
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
		time.Sleep(retryInterval)
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
	existing := &corev1.ConfigMap{}
	err = currentClusterClient.Get(
		context.Background(),
		client.ObjectKey{Name: cm.Name, Namespace: cm.Namespace},
		existing,
	)
	if apierrors.IsNotFound(err) {
		log.Infof("ConfigMap %s/%s not exist, creating", cm.Namespace, cm.Name)
		return currentClusterClient.Create(context.Background(), cm)
	} else if err != nil {
		return errors.Wrap(err, "get ConfigMap")
	}
	existing.Data = data
	log.Infof("updating existing ConfigMap %s/%s with %d files", cm.Namespace, cm.Name, len(data))
	return currentClusterClient.Update(context.Background(), existing)
}
