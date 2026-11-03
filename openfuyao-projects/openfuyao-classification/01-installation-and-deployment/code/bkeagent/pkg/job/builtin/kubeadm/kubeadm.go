package kubeadm

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/pkg/errors"
	bkenode "gopkg.openfuyao.cn/bkecommon/cluster/node"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"gopkg.openfuyao.cn/bkeagent/pkg/executor/exec"
	"gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/bkeagent/utils"
	"gopkg.openfuyao.cn/bkeagent/utils/clientutil"
	"gopkg.openfuyao.cn/bkeagent/utils/log"
	"gopkg.openfuyao.cn/bkeagent/utils/mfutil"
	"gopkg.openfuyao.cn/bkeagent/utils/pkiutil"
)

const (
	// Name means module name
	Name = "Kubeadm"
	// PollImmeInternal is used to define internal time for wait.PollImmediate, 500ms
	PollImmeInternal = 500 * time.Millisecond
	// PollImmeTimeout is used to define timeout time for wait.PollImmediate, min
	PollImmeTimeout = 3 * time.Minute
)

type KubeadmPlugin struct {
	k8sClient      client.Client
	localK8sClient *kubernetes.Clientset
	exec           exec.Executor

	boot                 *mfutil.BootScope
	isManager            bool
	clusterName          string
	controlPlaneEndpoint string
	GableNameSpace       string
}

func New(exec exec.Executor, c client.Client) plugin.Plugin {
	return &KubeadmPlugin{
		k8sClient:            c,
		exec:                 exec,
		boot:                 &mfutil.BootScope{},
		controlPlaneEndpoint: "",
	}
}

func (k *KubeadmPlugin) Name() string {
	return Name
}

func (k *KubeadmPlugin) Param() map[string]plugin.PluginParam {
	return map[string]plugin.PluginParam{
		"phase": {
			Key:         "phase",
			Value:       "initControlPlane,joinControlPlane,joinWorker,upgradeControlPlane,upgradeWorker,upgradeEtcd",
			Required:    true,
			Default:     "initControlPlane",
			Description: "phase",
		},
		"bkeConfig": {
			Key:         "bkeConfig",
			Value:       "NameSpace:Name",
			Required:    false,
			Default:     "",
			Description: "bkeconfig ConfigMap  ns:name",
		},
		"backUpEtcd": {
			Key:         "backUpEtcd",
			Value:       "true,false",
			Required:    false,
			Default:     "false",
			Description: "backUpEtcd ,only for upgradeControlPlane",
		},
		"clusterType": {
			Key:         "clusterType",
			Value:       "konk,bocloud",
			Required:    false,
			Default:     "bke",
			Description: "clusterType enum[bke,bocloud]",
		},
		"etcdVersion": {
			Key:         "etcdVersion",
			Value:       "3.6.4",
			Required:    false,
			Default:     "",
			Description: "etcd version number, used only for etcd upgrades",
		},
	}
}

// Execute execute kubeadm plugin
// example:
// ["Kubeadm", "phase=***", "bkeConfig=ns:name"]
func (k *KubeadmPlugin) Execute(commands []string) ([]string, error) {
	parseCommands, err := plugin.ParseCommands(k, commands)
	if err != nil {
		return nil, err
	}
	if v, ok := parseCommands["bkeConfig"]; ok {
		log.Info("get bkeConfig from command")
		if err = k.getBKEConfig(v); err != nil {
			return nil, err
		}
	}

	switch parseCommands["phase"] {
	case utils.InitControlPlane:
		k.boot.Extra["Init"] = true
		return nil, k.initControlPlane()
	case utils.JoinControlPlane:
		return nil, k.joinControlPlane()
	case utils.JoinWorker:
		return nil, k.joinWorker()
	case utils.UpgradeControlPlane:
		backupEtcd := false
		if v, ok := parseCommands["backUpEtcd"]; ok {
			backupEtcd = v == "true"
		}
		// get current cluster client for upgrade components
		k.localK8sClient, err = clientutil.ClientSetFromManagerClusterSecret(k.GableNameSpace, k.clusterName)
		if err != nil {
			return nil, err
		}
		//if parseCommands["clusterType"] != "konk" || parseCommands["clusterType"] != "bocloud" {
		//	return nil, errors.New("clusterType must be konk or bocloud")
		//}

		return nil, k.upgradeControlPlane(backupEtcd, parseCommands["clusterType"])
	case utils.UpgradeWorker:
		return nil, k.upgradeWorker()
	case utils.UpgradeEtcd:
		backupEtcd := false
		if v, ok := parseCommands["backUpEtcd"]; ok {
			backupEtcd = v == "true"
		}

		k.localK8sClient, err = clientutil.ClientSetFromManagerClusterSecret(k.GableNameSpace, k.clusterName)
		if err != nil {
			return nil, err
		}

		return nil, k.upgradeEtcd(backupEtcd, parseCommands["clusterType"])
	default:
		return nil, errors.New("unknown command")
	}
}

// initMaster init cluster master
func (k *KubeadmPlugin) initControlPlane() error {
	log.Info("Deploy k8s in init master node phase")
	// install kubelet in control plane node
	if err := k.installKubectlCommand(); err != nil {
		return err
	}
	// step 1 get CA certificates from cluster-api then generate other certs and upload to cluster-api
	if err := k.initControlPlaneCertCommand(); err != nil {
		return err
	}
	// step 2 generate static pod yaml
	if err := k.initControlPlaneManifestCommand(); err != nil {
		return err
	}
	if err := k.installKubeletCommand(); err != nil {
		return err
	}
	// step 4 upload kubeadm cluster config and kubelet config to cluster-api
	if err := k.uploadTargetClusterKubeletConfig(); err != nil {
		return err
	}
	// step 5 create secret for Global CA
	if k.isManager {
		if err := k.uploadUserCustomConfigAndGlobalCA(); err != nil {
			log.Warnf("No global CA or Config need to upload to current cluster: %v", err)
		}
	}

	return nil
}

func (k *KubeadmPlugin) joinControlPlane() error {
	log.Info("Deploy k8s in join master node phase")
	// install kubelet in control plane node
	if err := k.installKubectlCommand(); err != nil {
		return err
	}
	// step 1 get certificates from manager cluster
	if err := k.joinControlPlaneCertCommand(); err != nil {
		return err
	}
	// step 2 run kubelet
	if err := k.installKubeletCommand(); err != nil {
		return err
	}
	// step 3 generate static pod yaml
	if err := k.joinControlPlaneManifestCommand(); err != nil {
		return err
	}
	// step 4 create secret for Global CA
	if k.isManager {
		if err := k.uploadUserCustomConfigAndGlobalCA(); err != nil {
			log.Warnf("No global CA or Config need to upload to current cluster: %v", err)
		}
	}
	return nil
}

func (k *KubeadmPlugin) joinWorker() error {
	log.Info("Deploy k8s in join worker node phase")

	// step 1 get CA certificates from cluster-api
	if err := k.joinWorkerCertCommand(); err != nil {
		return err
	}
	// step 2 run kubelet
	if err := k.installKubeletCommand(); err != nil {
		return err
	}
	// step 3 install kubelet in worker node
	if err := k.installKubectlCommand(); err != nil {
		return err
	}
	return nil
}

// prepareUpgrade performs pre-upgrade tasks: backup etcd,
// backup cluster config, pre-pull images, and get component pod hash values
func (k *KubeadmPlugin) prepareUpgrade(backUpEtcd bool, clusterType string) (map[string]string, error) {
	// step 1 backup etcd
	log.Infof("backup etcd")
	if backUpEtcd {
		if err := k.backupEtcd(); err != nil {
			return nil, err
		}
	}

	// step 2 backup cluster etc
	log.Infof("backup cluster etc")
	if err := k.backupClusterEtc(clusterType); err != nil {
		return nil, err
	}

	// step 3 pre pull image
	if err := k.upgradePrePullImageCommand(); err != nil {
		log.Errorf("failed to upgrade pre pull image, err: %v", err)
		return nil, err
	}

	// step 4 get component pod hash map
	beforeHash, err := k.getBeforeUpgradeComponentPodHash()
	if err != nil {
		log.Errorf("failed to get before upgrade component pod hash, err: %v", err)
		return nil, err
	}

	return beforeHash, nil
}

func (k *KubeadmPlugin) upgradeControlPlane(backUpEtcd bool, clusterType string) error {
	log.Info("upgrade cluster in upgrade master node phase")

	if clusterType == "bocloud" {
		// 对于bocloud集群，需要先替换证书（重新创建manifests），重启kubelet容器，再升级组件，最后再次重启kubelet容器
		//经过调研，现有bocloud集群的镜像仓库域名和现在一致，所以不需要替换镜像仓库地址
	}

	// step2 load certs

	// 执行升级前的准备工作
	beforeHash, err := k.prepareUpgrade(backUpEtcd, clusterType)
	if err != nil {
		return err
	}

	// step 3.1 add new param to boot
	if k.boot != nil {
		log.Info("add new param to boot when upgrade control plane")
		k.boot.Extra["upgradeWithOpenFuyao"] = k.boot.HasOpenFuyaoAddon()
	}

	// step 4 upgrade components one by one
	log.Infof("upgrade components")
	for _, component := range mfutil.ControlPlaneComponents {
		// 判断是否需要升级该组件
		need, err := k.needUpgradeComponent(component)
		if err != nil {
			log.Errorf("failed to check need upgrade component, err: %v", err)
			return err
		}
		if !need {
			log.Infof("component %s already upgrade to %s, skip upgrade", component, k.boot.BkeConfig.Cluster.KubernetesVersion)
			continue
		}

		// generate new component static pod yaml
		if err := k.upgradeControlPlaneManifestCommand(component); err != nil {
			return err
		}
		podHash := beforeHash[component]
		// wait component ready
		if err := k.waitComponentReady(component, podHash); err != nil {
			return err
		}
		log.Infof("component %s upgrade success", component)
	}

	// step 5 upgrade kubelet
	log.Infof("upgrade kubelet")
	if err := k.installKubeletCommand(); err != nil {
		return err
	}
	log.Infof("upgrade kubectl for control plane node")
	// step 6 install new kubectl in control plane node
	if err := k.installKubectlCommand(); err != nil {
		return err
	}
	return nil
}

func (k *KubeadmPlugin) upgradeWorker() error {
	log.Info("upgrade cluster in upgrade worker node phase")
	// step 1 upgrade kubelet
	if err := k.installKubeletCommand(); err != nil {
		return err
	}
	log.Infof("upgrade kubectl for worker node")
	// step 2 install new kubectl in worker node
	if err := k.installKubectlCommand(); err != nil {
		return err
	}
	return nil
}

func (k *KubeadmPlugin) upgradeEtcd(backUpEtcd bool, clusterType string) error {
	log.Info("upgrade etcd ")

	// 执行升级前的准备工作
	beforeHash, err := k.prepareUpgrade(backUpEtcd, clusterType)
	if err != nil {
		return err
	}

	// step 4 upgrade components one by one
	log.Infof("upgrade components")
	component := mfutil.Etcd
	need, err := k.needUpgradeEtcd()
	if err != nil {
		log.Errorf("failed to check need upgrade component, err: %v", err)
		return err
	}
	if !need {
		log.Infof("component %s already upgrade to %s, skip upgrade", component, k.boot.BkeConfig.Cluster.EtcdVersion)
		return nil
	}

	// generate new component static pod yaml
	if err := k.upgradeControlPlaneManifestCommand(component); err != nil {
		return err
	}
	podHash := beforeHash[component]
	// wait component ready
	if err := k.waitComponentReady(component, podHash); err != nil {
		return err
	}
	log.Infof("component %s upgrade success", component)

	return nil
}

// uploadTargetClusterConfig upload kubelet config to manager cluster
func (k *KubeadmPlugin) uploadTargetClusterKubeletConfig() error {

	kubeletConf, err := os.ReadFile(utils.GetKubeletConfPath())
	if err != nil {
		return errors.Wrapf(err, "failed to read kubelet config")
	}

	kubeletConfigCM := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s%s", utils.KubeletConfigMapNamePrefix, k.boot.BkeConfig.Cluster.KubernetesVersion),
			Namespace: k.GableNameSpace,
		},
		Data: map[string]string{
			"kubelet": string(kubeletConf),
		},
	}
	if err := k.k8sClient.Create(context.Background(), kubeletConfigCM); err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return errors.Wrapf(err, "failed to create %q kubelet config configmap", k.clusterName)
		}
		err := k.k8sClient.Update(context.Background(), kubeletConfigCM)
		if err != nil {
			return errors.Wrapf(err, "failed to update %q kubelet config configmap", k.clusterName)
		}
	}

	return nil
}

func (k *KubeadmPlugin) getBKEConfig(bkeConfigNS string) error {
	bkeCluster, err := plugin.GetBKECluster(bkeConfigNS)
	if err != nil {
		return err
	}
	k.GableNameSpace = bkeCluster.GetNamespace()
	k.clusterName = bkeCluster.GetName()

	if bkeCluster.Spec.ControlPlaneEndpoint.Host != "" {
		k.controlPlaneEndpoint = bkeCluster.Spec.ControlPlaneEndpoint.Host
	}

	config, err := plugin.GetBkeConfigFromBkeCluster(bkeCluster)
	if err != nil {
		return err
	}

	k.boot.BkeConfig = config
	for _, addon := range config.Addons {
		if addon.Name == "cluster-api" {
			k.isManager = true
		}
	}
	currentNode, err := bkenode.Nodes(config.Nodes).CurrentNode()
	if err != nil {
		return errors.Wrapf(err, "failed to get current node")
	}

	k.boot = &mfutil.BootScope{
		BkeConfig:        config,
		KubeletConfigRef: bkeCluster.Spec.KubeletConfigRef,
		ClusterName:      bkeCluster.GetName(),
		HostName:         utils.HostName(),
		HostIP:           currentNode.IP,
		CurrentNode:      currentNode,
		Extra: map[string]interface{}{
			"Init":                 false,
			"gpuEnable":            "false",
			"KubernetesDir":        pkiutil.KubernetesDir,
			"mccs":                 []string{k.GableNameSpace, k.clusterName},
			"upgradeWithOpenFuyao": false,
		},
	}
	return nil
}

func (k *KubeadmPlugin) waitComponentReady(component, previousHash string) error {
	log.Infof("Wait cluster component %q ready", component)

	// wait pod hash change
	err := wait.PollImmediate(PollImmeInternal, PollImmeTimeout, func() (bool, error) {
		currentHash, err := getStaticPodSingleHash(k.localK8sClient, k.boot.HostName, component)
		if err != nil {
			// On error, continue pooling
			return false, nil
		}
		// Continue polling until the hash changes from previousHash
		if currentHash == previousHash {
			return false, nil
		}

		// Stop polling when detect hash change
		return true, nil
	})
	if err != nil {
		return fmt.Errorf("failed waiting for static pod hash to change: %w", err)
	}
	// Wait for the static pod component to come up and register itself as a mirror pod
	lastKnownPodNumber := -1
	kvLabel := "component=" + component
	err = wait.PollImmediate(PollImmeInternal, PollImmeTimeout, func() (bool, error) {
		listOpts := metav1.ListOptions{LabelSelector: kvLabel}
		pods, err := k.localK8sClient.CoreV1().Pods(metav1.NamespaceSystem).List(context.TODO(), listOpts)
		if err != nil {
			return false, nil
		}

		if lastKnownPodNumber != len(pods.Items) {
			lastKnownPodNumber = len(pods.Items)
		}

		if len(pods.Items) == 0 {
			return false, nil
		}

		for _, pod := range pods.Items {
			if pod.Status.Phase != corev1.PodRunning {
				return false, nil
			}
		}
		return true, nil
	})
	return err
}

func (k *KubeadmPlugin) getBeforeUpgradeComponentPodHash() (map[string]string, error) {
	mirrorPodHashes := map[string]string{}
	for _, component := range mfutil.ControlPlaneComponents {
		var (
			componentHash string
			pollErr       error
		)
		pollErr = wait.PollImmediate(PollImmeInternal, PollImmeTimeout, func() (bool, error) {
			componentHash, pollErr = getStaticPodSingleHash(k.localK8sClient, k.boot.HostName, component)
			if pollErr != nil {
				log.Debugf("failed to get pre-upgrade hash for component %s : %v", component, pollErr)
				return false, nil // Continue polling on error
			}
			return true, nil // Success case - stop polling
		})
		if pollErr != nil {
			return nil, fmt.Errorf("timed out waiting for component %s pod hash: %w", component, pollErr)
		}
		mirrorPodHashes[component] = componentHash
	}

	return mirrorPodHashes, nil
}

func (k *KubeadmPlugin) needUpgradeComponent(component string) (bool, error) {
	image, err := getStaticPodImage(k.localK8sClient, k.boot.HostName, component)
	if err != nil {
		return false, err
	}
	if image == "" {
		return false, errors.New("component image is empty")
	}

	// 判断镜像tag是否与bkeconfig中的集群版本共同
	// 如果不同，则需要升级
	if !strings.Contains(image, k.boot.BkeConfig.Cluster.KubernetesVersion) {
		return true, nil
	}
	return false, err
}

func (k *KubeadmPlugin) needUpgradeEtcd() (bool, error) {
	image, err := getStaticPodImage(k.localK8sClient, k.boot.HostName, mfutil.Etcd)
	if err != nil {
		return false, err
	}
	if image == "" {
		return false, errors.New("component image is empty")
	}

	if !strings.Contains(image, k.boot.BkeConfig.Cluster.EtcdVersion) {
		return true, nil
	}
	return false, err
}

func getStaticPodSingleHash(client *kubernetes.Clientset, nodeName, component string) (string, error) {
	podName := fmt.Sprintf("%s-%s", component, nodeName)
	pod, err := client.CoreV1().Pods(metav1.NamespaceSystem).Get(context.Background(), podName, metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	podHash := pod.Annotations["kubernetes.io/config.hash"]
	log.Debugf("Get component %q pod %q hash %q", component, podName, podHash)
	return podHash, nil
}

func getStaticPodImage(client *kubernetes.Clientset, nodeName, component string) (string, error) {
	podName := fmt.Sprintf("%s-%s", component, nodeName)
	pod, err := client.CoreV1().Pods(metav1.NamespaceSystem).Get(context.Background(), podName, metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	podImage := pod.Spec.Containers[0].Image
	log.Debugf("Get component %q pod %q image %q", component, podName, podImage)
	return podImage, nil
}
