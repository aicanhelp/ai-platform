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
	"crypto/tls"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	"sigs.k8s.io/controller-runtime/pkg/client"

	bkecommonv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkecommon "gopkg.openfuyao.cn/cluster-api-provider-bke/common"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/clientutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/mfutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/pkiutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/clusterutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	// Name means module name
	Name = "Kubeadm"
	// PollImmeInternal is used to define internal time for wait.PollImmediate, 500ms
	PollImmeInternal = 500 * time.Millisecond
	// PollImmeTimeout is used to define timeout time for wait.PollImmediate.
	// 1min：异常配置导致 Pod 起不来时更快失败，避免与 Command 默认 5min Wait 叠加过久。
	PollImmeTimeout = 1 * time.Minute
	// RunningStableCount 是 waitComponentPodRunning 判成功的连续采样次数。
	// 无 readinessProbe 的组件（etcd 等）Ready 退化为容器 running 瞬间状态，坏参数
	// "启动即退"的 crash 窗口叠加 kubelet 状态上报延迟会被单次采样撞中；连续 3 次
	// （约 1s）均 Ready 才算稳定运行，代价是正常路径多等约 1s。
	RunningStableCount = 3
	// LocalHealthyWaitTimeout 是 apiserver 全程不可用时启用本地健康判据的最大等待窗口。
	// 必须小于 Command 默认执行截止（DefaultWaitTimeout 5min - ActiveDeadlineWaitSkew 60s = 4min），
	// 为前置步骤与结果回传预留余量，确保 apiserver 全程不可用时内层先返回明确的超时错误，
	// 而不是被外层 active deadline 截杀（导致真实失败原因丢失）。
	LocalHealthyWaitTimeout = 3 * time.Minute
	// etcdMetricsPort 是 etcd static pod 的 metrics 健康端口（见 mfutil etcd.yaml.tmpl 的 --listen-metrics-urls）。
	etcdMetricsPort = "2381"
	// schedulerHealthPort / controllerManagerHealthPort 与对应 static pod 模板中 livenessProbe 端口一致。
	schedulerHealthPort         = int32(10259)
	controllerManagerHealthPort = int32(10257)
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

	// execCtx is set per Execute on a shallow copy so concurrent Tasks do not race the registry singleton.
	execCtx context.Context
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
			Key: "phase",
			Value: "initControlPlane,joinControlPlane,joinWorker," +
				"upgradeControlPlane,upgradeWorker,upgradeEtcd,UpgradeControlPlaneConfig,UpgradeEtcdConfig,reconfigureNginx",
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
		"controlPlaneConfigOldHash": {
			Key:         "controlPlaneConfigOldHash",
			Value:       "",
			Required:    false,
			Default:     "",
			Description: "old control plane component config hash struct JSON, used only for UpgradeControlPlaneConfig",
		},
		"controlPlaneConfigNewHash": {
			Key:         "controlPlaneConfigNewHash",
			Value:       "",
			Required:    false,
			Default:     "",
			Description: "new control plane component config hash struct JSON, used only for UpgradeControlPlaneConfig",
		},
	}
}

// Execute execute kubeadm plugin
// example:
// ["Kubeadm", "phase=***", "bkeConfig=ns:name"]
func (k *KubeadmPlugin) Execute(ctx context.Context, commands []string) ([]string, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	// Shallow copy so execCtx is per-invocation; nested waits/plugin calls use k.execCtx.
	kk := *k
	kk.execCtx = ctx
	return kk.executeWithCtx(commands)
}

func (k *KubeadmPlugin) executeWithCtx(commands []string) ([]string, error) {
	k.exec = exec.Bound(k.exec, k.contextOrBackground())
	parseCommands, err := plugin.ParseCommands(k, commands)
	if err != nil {
		return nil, err
	}
	if err = k.initBKEConfig(parseCommands); err != nil {
		return nil, err
	}

	switch parseCommands["phase"] {
	case utils.InitControlPlane:
		k.boot.Extra["Init"] = true
		return nil, k.initControlPlane()
	case utils.JoinControlPlane:
		return nil, k.joinControlPlane()
	case utils.JoinWorker:
		return nil, k.joinWorker()
	case utils.UpgradeWorker:
		return nil, k.upgradeWorker()
	case utils.UpgradeControlPlane:
		return nil, k.execUpgradeControlPlane(parseCommands)
	case utils.UpgradeEtcd:
		return nil, k.execUpgradeEtcd(parseCommands)
	case utils.UpgradeControlPlaneConfig:
		return nil, k.execUpgradeControlPlaneConfig(parseCommands)
	case utils.UpgradeEtcdConfig:
		return nil, k.execUpgradeEtcdConfig()
	case utils.ReconfigureNginx:
		return nil, k.reconfigureNginx()
	default:
		return nil, errors.Errorf("unknown command: %s", parseCommands["phase"])
	}
}

// --- 以下为提取出的辅助方法，消除分支内部的嵌套复杂度 ---

func (k *KubeadmPlugin) initBKEConfig(parseCommands map[string]string) error {
	if v, ok := parseCommands["bkeConfig"]; ok {
		log.Info("get bkeConfig from command")
		if err := k.getBKEConfig(v); err != nil {
			return err
		}
		k.applyCommandEtcdVersion(parseCommands)
	}
	return nil
}

func (k *KubeadmPlugin) initLocalK8sClient() (err error) {
	if k.localK8sClient != nil {
		return nil
	}
	k.localK8sClient, err = clientutil.ClientSetFromManagerClusterSecret(k.GableNameSpace, k.clusterName)
	return err
}

func (k *KubeadmPlugin) execUpgradeControlPlane(cmd map[string]string) error {
	if err := k.initLocalK8sClient(); err != nil {
		return err
	}
	return k.upgradeControlPlane(cmd["backUpEtcd"] == "true", cmd["clusterType"])
}

func (k *KubeadmPlugin) execUpgradeEtcd(cmd map[string]string) error {
	if err := k.initLocalK8sClient(); err != nil {
		return err
	}
	return k.upgradeEtcd(cmd["backUpEtcd"] == "true", cmd["clusterType"])
}

func (k *KubeadmPlugin) execUpgradeControlPlaneConfig(cmd map[string]string) error {
	if err := k.initLocalK8sClient(); err != nil {
		return err
	}
	oldHash, err := phaseutil.ComponentConfigHashFromJSON(cmd["controlPlaneConfigOldHash"])
	if err != nil {
		return err
	}
	newHash, err := phaseutil.ComponentConfigHashFromJSON(cmd["controlPlaneConfigNewHash"])
	if err != nil {
		return err
	}
	return k.updateControlPlaneConfig(oldHash, newHash)
}

func (k *KubeadmPlugin) execUpgradeEtcdConfig() error {
	if err := k.initLocalK8sClient(); err != nil {
		return err
	}
	return k.updateEtcdConfig()
}

// reconfigureNginx re-runs the nginx plugin to regenerate nginx.conf and the
// static pod manifest on an existing node (used during nginx-proxy version
// upgrade via ensure_nginx_upgrade). kubelet recreates the pod on manifest change.
func (k *KubeadmPlugin) reconfigureNginx() error {
	log.Info("Reconfiguring nginx load balancer on existing node")
	return k.joinWorkerNginxCommand()
}

// initMaster init cluster master
func (k *KubeadmPlugin) initControlPlane() error {
	log.Info("Deploy k8s in init master node phase", "cluster", k.clusterName)
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
	if err := k.installKubeletCommand(false); err != nil {
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

// updateControlPlaneConfig 纯配置更新控制面静态 Pod（不备份 etcd）。
// 仅覆盖 APIServer / ControllerManager / Scheduler；etcd 由 updateEtcdConfig 处理。
//
// 是否更新仅依据 Command 下发的 old/new 配置 hash（与版本升级的 pod hash 无关）：
// 配置 hash 有变化则重刷 manifest 并等待 Pod Running。不读取、不等待
// kubernetes.io/config.hash，避免异常配置把 Pod 打挂后无法取 beforeHash、正确配置也刷不进去。
func (k *KubeadmPlugin) updateControlPlaneConfig(oldHash, newHash bkecommonv1beta1.ComponentConfigHash) error {
	log.Info("update control plane static pod config")

	changedComponents := phaseutil.ComponentConfigHashChangedComponents(
		oldHash,
		newHash,
	)
	var toUpdate []string
	for _, component := range mfutil.GetControlPlaneComponents() {
		if !changedComponents[component] {
			log.Infof("component %s config unchanged, skip", component)
			continue
		}
		toUpdate = append(toUpdate, component)
	}
	if len(toUpdate) == 0 {
		log.Infof("no control plane component config changed, skip all")
		return nil
	}

	for _, component := range toUpdate {
		log.Infof("component %s config changed, regenerate manifest", component)
		// 在重刷 manifest 前记录旧 mirror Pod UID：等待期只认新 Pod 的 Running，
		// 避免 kubelet 重建窗口内旧 Pod 的 Running 状态被误判为更新成功
		baselineUID := k.mirrorPodUID(component)
		if err := k.upgradeControlPlaneManifestCommand(component); err != nil {
			return err
		}
		if err := k.waitComponentPodRunning(component, baselineUID); err != nil {
			return err
		}
		log.Infof("component %s config update success", component)
	}
	log.Infof("control plane config update completed: updated=%d", len(toUpdate))

	return nil
}

// updateEtcdConfig 纯配置更新 etcd 静态 Pod manifest。
// 独立于控制面组件配置更新，支持 etcd 单独挂载的场景。
// phase / Command 已确认 etcd 配置 hash 变化；此处只重刷 manifest 并等待 Pod Running，
// 不依赖 pod hash（版本升级路径仍走 waitComponentReady）。
func (k *KubeadmPlugin) updateEtcdConfig() error {
	log.Info("update etcd static pod config")

	log.Infof("re-generate etcd manifest")
	baselineUID := k.mirrorPodUID(mfutil.Etcd)
	if err := k.upgradeControlPlaneManifestCommand(mfutil.Etcd); err != nil {
		return err
	}

	if err := k.waitComponentPodRunning(mfutil.Etcd, baselineUID); err != nil {
		return err
	}
	log.Infof("etcd config update success")

	return nil
}

func (k *KubeadmPlugin) joinControlPlane() error {
	log.Info("Deploy k8s in join master node phase", "cluster", k.clusterName)
	// install kubelet in control plane node
	if err := k.installKubectlCommand(); err != nil {
		return err
	}
	// step 1 get certificates from manager cluster
	if err := k.joinControlPlaneCertCommand(); err != nil {
		return err
	}
	// step 2 run kubelet
	if err := k.installKubeletCommand(false); err != nil {
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
	log.Info("Deploy k8s in join worker node phase", "cluster", k.clusterName)

	// step 1 get CA certificates from cluster-api
	if err := k.joinWorkerCertCommand(); err != nil {
		return err
	}
	// step 2 deploy nginx load balancer (loadBalanceType=nginx only)
	if err := k.joinWorkerNginxCommand(); err != nil {
		return err
	}
	// step 3 run kubelet
	if err := k.installKubeletCommand(k.boot.IsImmutableOS); err != nil {
		return err
	}
	// step 4 install kubectl (not needed on immutable-OS worker nodes)
	if !k.boot.IsImmutableOS {
		if err := k.installKubectlCommand(); err != nil {
			return err
		}
	}
	return nil
}

// prepareUpgrade performs pre-upgrade tasks: backup etcd,
// backup cluster config, pre-pull images, and get component pod hash values
func (k *KubeadmPlugin) prepareUpgrade(backUpEtcd bool, clusterType string) (map[string]string, error) {
	// step 1 backup etcd
	log.Infof("backup etcd for cluster %s", k.clusterName)
	if backUpEtcd {
		if err := k.backupEtcd(); err != nil {
			return nil, err
		}
	}

	// step 2 backup cluster etc
	log.Infof("backup cluster etc for cluster %s", k.clusterName)
	if err := k.backupClusterEtc(clusterType); err != nil {
		return nil, err
	}

	// step 3 pre pull image
	if err := k.upgradePrePullImageCommand(); err != nil {
		return nil, err
	}

	// step 4 get component pod hash map
	beforeHash, err := k.getBeforeUpgradeComponentPodHash()
	if err != nil {
		return nil, err
	}

	return beforeHash, nil
}

func (k *KubeadmPlugin) upgradeControlPlane(backUpEtcd bool, clusterType string) error {
	log.Info("upgrade cluster in upgrade master node phase", "cluster", k.clusterName)

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
		log.Debugf("add new param to boot when upgrade control plane for cluster %s", k.clusterName)
		k.boot.Extra["upgradeWithOpenFuyao"] = k.boot.HasOpenFuyaoAddon()
	}

	// step 4 upgrade components one by one
	log.Debugf("upgrade components for cluster %s", k.clusterName)
	var upgradedComponents []string
	for _, component := range mfutil.GetControlPlaneComponents() {
		// 判断是否需要升级该组件
		need, err := k.needUpgradeComponent(component)
		if err != nil {
			return fmt.Errorf("failed to check need upgrade component, err: %w", err)
		}
		if !need {
			log.Debugf("component %s already upgrade to %s, skip upgrade", component, k.boot.BkeConfig.Cluster.KubernetesVersion)
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
		log.Debugf("component %s upgrade success", component)
		upgradedComponents = append(upgradedComponents, component)
	}
	log.Infof("control plane components upgrade completed for cluster %s: total=%d, upgraded=%v",
		k.clusterName, len(upgradedComponents), upgradedComponents)

	// step 5 upgrade kubelet
	log.Infof("upgrade kubelet for cluster %s", k.clusterName)
	if err := k.installKubeletCommand(false); err != nil {
		return err
	}
	log.Infof("upgrade kubectl for control plane node, cluster %s", k.clusterName)
	// step 6 install new kubectl in control plane node
	if err := k.installKubectlCommand(); err != nil {
		return err
	}
	return nil
}

func (k *KubeadmPlugin) upgradeWorker() error {
	log.Info("upgrade cluster in upgrade worker node phase", "cluster", k.clusterName)
	// step 1 upgrade kubelet
	if err := k.installKubeletCommand(k.boot.IsImmutableOS); err != nil {
		return err
	}
	log.Infof("upgrade kubectl for worker node, cluster %s", k.clusterName)
	// step 2 install new kubectl in worker node
	if err := k.installKubectlCommand(); err != nil {
		return err
	}
	return nil
}

func (k *KubeadmPlugin) upgradeEtcd(backUpEtcd bool, clusterType string) error {
	log.Info("upgrade etcd", "cluster", k.clusterName)

	// 执行升级前的准备工作
	beforeHash, err := k.prepareUpgrade(backUpEtcd, clusterType)
	if err != nil {
		return err
	}

	// step 4 upgrade components one by one
	log.Infof("upgrade etcd components for cluster %s", k.clusterName)
	component := mfutil.Etcd
	need, err := k.needUpgradeEtcd()
	if err != nil {
		return fmt.Errorf("failed to check need upgrade component, err: %w", err)
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
		err := phaseutil.RetryOnConflict(func() error {
			existingConfigMap := &corev1.ConfigMap{}
			if err := k.k8sClient.Get(context.Background(), client.ObjectKey{Namespace: kubeletConfigCM.Namespace, Name: kubeletConfigCM.Name}, existingConfigMap); err != nil {
				return err
			}
			existingConfigMap.Data = kubeletConfigCM.Data
			return k.k8sClient.Update(context.Background(), existingConfigMap)
		})
		if err != nil {
			return errors.Wrapf(err, "failed to update %q kubelet config configmap", k.clusterName)
		}
	}

	return nil
}

// applyCommandEtcdVersion overrides spec etcd version when the provider passes a
// declarative upgrade target (VersionContext / release bundle).
func (k *KubeadmPlugin) applyCommandEtcdVersion(parseCommands map[string]string) {
	v, ok := parseCommands["etcdVersion"]
	if !ok || v == "" || k.boot == nil || k.boot.BkeConfig == nil {
		return
	}
	k.boot.BkeConfig.Cluster.EtcdVersion = v
	if k.boot.Extra == nil {
		k.boot.Extra = map[string]interface{}{}
	}
	k.boot.Extra["etcdVersion"] = v
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

	clusterData, err := plugin.GetClusterData(bkeConfigNS)
	if err != nil {
		return err
	}

	currentNode, err := bkenode.Nodes(clusterData.Nodes).CurrentNode()
	if err != nil {
		return errors.Wrapf(err, "failed to get current node")
	}

	k.boot = &mfutil.BootScope{
		BkeConfig:        config,
		KubeletConfigRef: bkeCluster.Spec.KubeletConfigRef,
		ClusterName:      bkeCluster.GetName(),
		ClusterNamespace: bkeCluster.GetNamespace(),
		HostName:         utils.HostName(),
		HostIP:           currentNode.IP,
		CurrentNode:      currentNode,
		IsImmutableOS:    k.isImmutableOSMode(bkeCluster, config),
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

// isImmutableOSMode 判断是否为不可变 OS 模式。
// 同时检查 BKECluster annotation 和 BkeConfig.CustomExtra，避免两者不一致导致判断偏差。
func (k *KubeadmPlugin) isImmutableOSMode(bkeCluster *bkecommonv1beta1.BKECluster, config *bkecommonv1beta1.BKEConfig) bool {
	if clusterutil.IsImmutableOSModeConf(bkeCluster) {
		return true
	}
	if config != nil && config.CustomExtra != nil {
		return config.CustomExtra[bkecommon.ImmutableOSCustomExtraKey] == bkecommon.ImmutableOSAnnotationValue
	}
	return false
}

func (k *KubeadmPlugin) waitComponentReady(component, previousHash string) error {
	log.Infof("Wait cluster component %q ready", component)
	ctx := k.contextOrBackground()

	// wait pod hash change（仅版本升级等路径使用；配置更新走 waitComponentPodRunning）
	err := wait.PollImmediateWithContext(ctx, PollImmeInternal, PollImmeTimeout, func(ctx context.Context) (bool, error) {
		if err := ctx.Err(); err != nil {
			return false, err
		}
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
	// hash 已变化即新 mirror Pod 已生成，无需旧 UID 基线，退化为仅判 Running
	return k.waitComponentPodRunning(component, "")
}

func (k *KubeadmPlugin) contextOrBackground() context.Context {
	if k.execCtx != nil {
		return k.execCtx
	}
	return context.Background()
}

// waitComponentPodRunning 等待静态 Pod 以 mirror Pod 形式出现且 Phase==Running。
// 配置更新路径专用（不依赖 kubernetes.io/config.hash）；版本升级在 hash 变化后再调用本方法
// （此时新 Pod 已确认生成，baselineUID 传空退化为仅判 Running）。
//
// baselineUID 为重刷 manifest 前旧 mirror Pod 的 UID（见 mirrorPodUID）：kubelet 重启后
// 删除旧 mirror Pod、创建新 mirror Pod 存在数秒窗口，期间旧 Pod 仍带 Running 状态，
// 若不区分新旧会把"旧 Pod 尚未删除"误判为更新成功（新容器实际可能因无效 flag 起不来）。
// 非空时判据升级为：存在 UID != baselineUID 的新 mirror Pod 且全部 Running；
// 异常配置下新 Pod 永远不会 Running，最终在 normalDeadline/localDeadline 处报超时。
//
// 判据分两层：
//  1. mirror Pod Running（依赖 apiserver）：标准判据，apiserver 可用时维持 1min 快速失败节奏；
//     无 readinessProbe 的组件 Ready 退化为容器 running 瞬间状态，故需连续 RunningStableCount
//     次采样均 Ready，并在满足后追加一次 localComponentHealthy 服务级终审（本地实时探测、
//     无上报延迟），拦住"crash 瞬间撞窗"与"滞留 Ready 快照"两类残留误判；
//  2. 本地健康判据 localComponentHealthy（不依赖 apiserver）：apiserver 持续不可达超过
//     PollImmeTimeout 后启用为主判据，兜底恢复期 "etcd 等 apiserver 创建 mirror Pod、
//     apiserver 等 etcd 恢复" 的互等死锁；apiserver 全程不可用时最多等 LocalHealthyWaitTimeout。
func (k *KubeadmPlugin) waitComponentPodRunning(component, baselineUID string) error {
	log.Infof("Wait cluster component %q pod running (baseline pod uid %q)", component, baselineUID)

	kvLabel := "component=" + component
	// 多 master 共用一个 apiserver，仅按 component label 会查到其它节点的
	// mirror Pod：本节点新 Pod 尚未就绪时，其它节点的健康 Pod 会满足判据，
	// 导致配置更新被秒判成功（坏参数随之串行扩散到全部节点）。
	// 必须限定 spec.nodeName 只看本节点。
	fieldSelector := "spec.nodeName=" + k.boot.HostName
	normalDeadline := time.Now().Add(PollImmeTimeout)
	localDeadline := time.Now().Add(LocalHealthyWaitTimeout)
	apiWasDown := false
	stableCount := 0

	ctx := k.contextOrBackground()
	for {
		if err := ctx.Err(); err != nil {
			return fmt.Errorf("wait for %q pod running cancelled: %w", component, err)
		}
		pods, err := k.localK8sClient.CoreV1().Pods(metav1.NamespaceSystem).List(ctx, metav1.ListOptions{LabelSelector: kvLabel, FieldSelector: fieldSelector})
		if err != nil {
			// apiserver 不可达：不立即判负，超过 PollImmeTimeout 后切换本地健康判据
			apiWasDown = true
			if done, retErr := k.apiDownFallback(component, normalDeadline, localDeadline); done {
				return retErr
			}
		} else {
			// apiserver 恢复后重新给一个完整观察窗口
			if apiWasDown {
				normalDeadline = time.Now().Add(PollImmeTimeout)
				apiWasDown = false
			}
			if done, retErr := k.checkPodsStable(component, baselineUID, pods, &stableCount); done {
				return retErr
			}
			if time.Now().After(normalDeadline) {
				return fmt.Errorf("timed out waiting for %q pod running (apiserver available)", component)
			}
		}
		timer := time.NewTimer(PollImmeInternal)
		select {
		case <-ctx.Done():
			timer.Stop()
			return fmt.Errorf("wait for %q pod running cancelled: %w", component, ctx.Err())
		case <-timer.C:
		}
	}
}

// checkPodsStable 判断 mirror Pod 是否已连续 RunningStableCount 次采样稳定 Ready，
// 并在稳定后追加 localComponentHealthy 服务级终审。
// done=true 时 retErr 为最终结论（成功为 nil）；done=false 时 stableCount 已更新，继续轮询。
func (k *KubeadmPlugin) checkPodsStable(component, baselineUID string, pods *corev1.PodList, stableCount *int) (done bool, retErr error) {
	if !allPodsRunning(pods, baselineUID) {
		*stableCount = 0
		return false, nil
	}
	*stableCount++
	// 连续 RunningStableCount 次采样均 Ready 才判成功，消除
	// "running 瞬间 + 状态上报延迟"被单次采样撞中的误判
	if *stableCount < RunningStableCount {
		return false, nil
	}
	// pod 层稳定后追加服务级终审：无 readinessProbe 的组件 Ready
	// 退化为容器 running 瞬间状态，且 kubelet 状态上报有延迟，
	// 本地实时探测可拦住"crash 瞬间撞窗"与"滞留 Ready 快照"两类残留误判
	if k.localComponentHealthy(component) {
		return true, nil
	}
	*stableCount = 0
	return false, nil
}

// apiDownFallback 处理 apiserver 持续不可达时的本地健康兜底判定：
// 超过 normalDeadline 后启用 localComponentHealthy 判据，兜底恢复期
// "etcd 等 apiserver 创建 mirror Pod、apiserver 等 etcd 恢复" 的互等死锁。
// done=true 时 retErr 为最终结果：本地健康视为 running（nil），本地兜底也超过
// localDeadline 则返回超时错误；否则 done=false 继续轮询。
func (k *KubeadmPlugin) apiDownFallback(component string, normalDeadline, localDeadline time.Time) (done bool, retErr error) {
	if !time.Now().After(normalDeadline) {
		return false, nil
	}
	if k.localComponentHealthy(component) {
		log.Infof("component %q healthy locally while apiserver unavailable, treat as running", component)
		return true, nil
	}
	if time.Now().After(localDeadline) {
		return true, fmt.Errorf("timed out waiting for %q pod running: apiserver unavailable the whole time and local health check failed", component)
	}
	return false, nil
}

// allPodsRunning 判断是否存在"新" mirror Pod 且全部处于 Running 阶段且容器 Ready。
// baselineUID 为空：任意 Pod 全部 Running 且容器 Ready 即成功（空列表视为未就绪）；
// baselineUID 非空：仅统计 UID != baselineUID 的新 Pod，无新 Pod（kubelet 尚未重建）
// 或新 Pod 未全部 Running/Ready 均视为未就绪。
// 注意：容器 crash loop 时 Pod phase 仍为 Running（restartPolicy=Always），
// 因此除 phase 外必须校验容器 Ready（liveness 探测通过），否则坏参数会被误判为更新成功。
func allPodsRunning(pods *corev1.PodList, baselineUID string) bool {
	items := pods.Items
	if baselineUID != "" {
		items = nil
		for _, pod := range pods.Items {
			if string(pod.UID) != baselineUID {
				items = append(items, pod)
			}
		}
		if len(items) == 0 {
			// 新 mirror Pod 尚未生成：旧 Pod 的 Running 状态不可信
			return false
		}
	}
	for _, pod := range items {
		if pod.Status.Phase != corev1.PodRunning {
			return false
		}
		for _, c := range pod.Status.ContainerStatuses {
			if !c.Ready {
				return false
			}
		}
	}
	return true
}

// mirrorPodUID 返回组件当前 mirror Pod 的 UID，作为配置更新路径的旧 Pod 基线。
// 须在重刷 manifest / restart kubelet 之前调用（此时旧配置 Pod 仍在运行）；
// 查询失败或 Pod 不存在时返回空串，waitComponentPodRunning 退化为仅判 Running。
// 限定 spec.nodeName：多 master 共用 apiserver，仅按 component label 的 List
// 可能返回其它节点的 Pod，Items[0] 会抓到错误的 baseline 导致新 Pod 误判。
func (k *KubeadmPlugin) mirrorPodUID(component string) string {
	if k.localK8sClient == nil {
		return ""
	}
	kvLabel := "component=" + component
	pods, err := k.localK8sClient.CoreV1().Pods(metav1.NamespaceSystem).List(context.TODO(), metav1.ListOptions{
		LabelSelector: kvLabel,
		FieldSelector: "spec.nodeName=" + k.boot.HostName,
	})
	if err != nil || len(pods.Items) == 0 {
		log.Warnf("baseline mirror pod of component %q not found, fall back to phase-only check", component)
		return ""
	}
	return string(pods.Items[0].UID)
}

// localComponentHealthy 本地健康判据，不依赖 apiserver，仅用于 apiserver 持续不可用时的兜底：
//   - etcd：metrics /health 端点（HTTP）；
//   - 其他控制面组件：本地 HTTPS 健康端点（与各 static pod 模板的 livenessProbe 同源），
//     服务级判据——进程存活但服务不可用（如启动中、CrashLoop 瞬时进程）不会被误判为健康；
//   - 无可探测端点的组件：回退 pgrep 进程存活探测（低保真兜底）。
func (k *KubeadmPlugin) localComponentHealthy(component string) bool {
	if k.boot == nil {
		return false
	}
	if component == mfutil.Etcd {
		return k.etcdLocalHealthy()
	}
	if path, port, ok := k.componentHealthEndpoint(component); ok {
		return k.componentHealthzHealthy(path, port)
	}
	return k.processAlive(component)
}

// componentHealthEndpoint 返回组件本地健康端点（与 static pod 模板中的 livenessProbe 同源）。
// ok=false 表示该组件没有可探测端点，调用方回退进程存活探测。
func (k *KubeadmPlugin) componentHealthEndpoint(component string) (path string, port int32, ok bool) {
	switch component {
	case mfutil.KubeAPIServer:
		return "/livez", k.apiServerPort(), true
	case mfutil.KubeScheduler:
		return "/healthz", schedulerHealthPort, true
	case mfutil.KubeControllerManager:
		return "/healthz", controllerManagerHealthPort, true
	default:
		return "", 0, false
	}
}

// apiServerPort 与 mfutil 的 apiServerPort funcMap 取值链保持一致。
func (k *KubeadmPlugin) apiServerPort() int32 {
	if k.boot.CurrentNode.APIServer != nil && k.boot.CurrentNode.APIServer.Port != 0 {
		return k.boot.CurrentNode.APIServer.Port
	}
	return k.boot.BkeConfig.Cluster.APIServer.Port
}

// etcdHealthClient 用于 etcd 本地健康探测，端点为 hostNetwork 的 metrics 端口（HTTP 无 TLS）。
var etcdHealthClient = &http.Client{Timeout: 2 * time.Second}

// componentHealthClient 用于控制面组件本地健康探测（HTTPS 自签/集群 CA 证书，跳过验证），
// 与 kubelet livenessProbe 一致不带客户端证书，端点匿名可访问。
var componentHealthClient = &http.Client{
	Timeout: 2 * time.Second,
	Transport: &http.Transport{
		TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
	},
}

// etcdLocalHealthy 通过 metrics 端口健康端点判断，该端点即 etcd.yaml.tmpl 中
// kubelet livenessProbe 所用端点（--listen-metrics-urls），返回健康即 etcd 已可对外提供服务。
// 直接用 HTTP 探测，不依赖节点上是否安装 curl。
func (k *KubeadmPlugin) etcdLocalHealthy() bool {
	resp, err := etcdHealthClient.Get(fmt.Sprintf("http://%s:%s/health", k.boot.HostIP, etcdMetricsPort))
	if err != nil {
		log.Debugf("etcd local health check failed: %v", err)
		return false
	}
	defer resp.Body.Close()
	// etcd /health：健康返回 {"health":"true"}（或新版本布尔 true）+ 200；不健康返回 503
	if resp.StatusCode != http.StatusOK {
		return false
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Debugf("etcd local health check failed to read body: %v", err)
		return false
	}
	return strings.Contains(string(body), "true")
}

// componentHealthzHealthy 请求组件本地 HTTPS 健康端点，200 即认为服务可用。
func (k *KubeadmPlugin) componentHealthzHealthy(path string, port int32) bool {
	resp, err := componentHealthClient.Get(fmt.Sprintf("https://%s:%d%s", k.boot.HostIP, port, path))
	if err != nil {
		log.Debugf("component %q local health check failed: %v", path, err)
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// processAlive 通过进程名判断组件进程存活（低保真兜底，仅用于无可探测健康端点的组件），
// pgrep -f 按 cmdline 匹配，规避进程名 15 字节截断导致精确匹配失败（如 kube-controller-manager）。
func (k *KubeadmPlugin) processAlive(component string) bool {
	cmd := fmt.Sprintf("pgrep -f '^%s( |$)'", component)
	out, err := k.exec.ExecuteCommandWithCombinedOutput("/usr/bin/sh", "-c", cmd)
	if err != nil {
		log.Debugf("check process %q alive failed: %s, err: %v", component, out, err)
		return false
	}
	return strings.TrimSpace(out) != ""
}

func (k *KubeadmPlugin) getBeforeUpgradeComponentPodHash() (map[string]string, error) {
	mirrorPodHashes := map[string]string{}
	for _, component := range mfutil.GetControlPlaneComponents() {
		var (
			componentHash string
			pollErr       error
		)
		pollErr = wait.PollImmediateWithContext(k.contextOrBackground(), PollImmeInternal, PollImmeTimeout, func(ctx context.Context) (bool, error) {
			if err := ctx.Err(); err != nil {
				return false, err
			}
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

func normalizeVersionTag(v string) string {
	return strings.TrimPrefix(strings.TrimSpace(v), "v")
}

func imageMatchesVersion(image, version string) bool {
	tag := normalizeVersionTag(utils.ImageTagFromReference(image))
	want := normalizeVersionTag(version)
	if tag == "" || want == "" {
		return false
	}
	return tag == want
}

func (k *KubeadmPlugin) needUpgradeComponent(component string) (bool, error) {
	image, err := getStaticPodImage(k.localK8sClient, k.boot.HostName, component)
	if err != nil {
		return false, err
	}
	if image == "" {
		return false, errors.New("component image is empty")
	}

	if imageMatchesVersion(image, k.boot.BkeConfig.Cluster.KubernetesVersion) {
		return false, nil
	}
	return true, nil
}

func (k *KubeadmPlugin) needUpgradeEtcd() (bool, error) {
	image, err := getStaticPodImage(k.localK8sClient, k.boot.HostName, mfutil.Etcd)
	if err != nil {
		return false, err
	}
	if image == "" {
		return false, errors.New("component image is empty")
	}

	if imageMatchesVersion(image, k.boot.BkeConfig.Cluster.EtcdVersion) {
		return false, nil
	}
	return true, nil
}

func getStaticPodSingleHash(client *kubernetes.Clientset, nodeName, component string) (string, error) {
	podName := fmt.Sprintf("%s-%s", component, nodeName)
	pod, err := client.CoreV1().Pods(metav1.NamespaceSystem).Get(context.Background(), podName, metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	podHash := pod.Annotations[annotation.KubernetesConfigHashAnnotationKey]
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
