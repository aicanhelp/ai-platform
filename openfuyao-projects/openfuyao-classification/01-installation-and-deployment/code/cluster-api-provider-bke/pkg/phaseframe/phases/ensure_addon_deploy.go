/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package phases

import (
	"context"
	"crypto/sha256"
	"fmt"
	"strings"

	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	kerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"sigs.k8s.io/cluster-api/util/annotations"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common"
	bkeaddon "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/addon"
	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/certs"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/command"
	downloadplugin "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/downloader"
	mfplugin "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/kubeadm/manifests"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/kube"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/pkiutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/clusterutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/condition"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
	labelhelper "gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/label"
)

const (
	// DefaultPasswordLength is the default length for generated passwords
	DefaultPasswordLength = 12
	// DefaultSaltLength is the default length for password salt
	DefaultSaltLength = 16
	// DefaultIterations is the default number of iterations for password hashing
	DefaultIterations = 100000
	// DefaultKeyLength is the default key length for encryption
	DefaultKeyLength = 64
)

const (
	EnsureAddonDeployName confv1beta1.BKEClusterPhase = "EnsureAddonDeploy"

	etcdCertsScretName = "etcd-backup-secrets"
)

type EnsureAddonDeploy struct {
	phaseframe.BasePhase
	addons              []*bkeaddon.AddonTransfer
	targetClusterClient kube.RemoteKubeClient
	remoteClient        *kubernetes.Clientset
	mockClient          kubernetes.Interface
	remoteDynamicClient dynamic.Interface
	addonRecorders      []*kube.AddonRecorder
}

// createCommandSpec 创建命令规范
func createCommandSpec(commands []agentv1beta1.ExecCommand) *agentv1beta1.CommandSpec {
	commandSpec := command.GenerateDefaultCommandSpec()
	commandSpec.Commands = commands
	return commandSpec
}

func NewEnsureAddonDeploy(ctx *phaseframe.PhaseContext) phaseframe.Phase {
	base := phaseframe.NewBasePhase(ctx, EnsureAddonDeployName)
	phase := &EnsureAddonDeploy{BasePhase: base}
	phase.RegisterPostHooks(phase.saveAddonManifestsPostHook)
	return phase
}

func (e *EnsureAddonDeploy) Execute() (ctrl.Result, error) {
	targetClusterClient, err := kube.NewRemoteClientByBKECluster(e.Ctx.Context, e.Ctx.Client, e.Ctx.BKECluster)
	if err != nil {
		return ctrl.Result{}, err
	}
	e.targetClusterClient = targetClusterClient
	e.targetClusterClient.SetLogger(e.Ctx.Log.NormalLogger)
	e.targetClusterClient.SetBKELogger(e.Ctx.Log)
	e.remoteClient, e.remoteDynamicClient = targetClusterClient.KubeClient()

	if err = e.reconcileAddon(); err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{}, nil
}

func (e *EnsureAddonDeploy) NeedExecute(old *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster) bool {
	if !e.BasePhase.DefaultNeedExecute(old, new) {
		return false
	}

	// 使用 NodeFetcher 获取 BKENodes（在 controller 上下文中）
	bkeNodes, err := e.Ctx.GetBKENodes()
	if err != nil {
		e.Ctx.Log.Warn(constant.AddonDeployingReason, "Failed to get BKENodes: %v", err)
	}

	if e.Ctx.Cluster != nil && !phaseutil.AllowDeployAddonWithBKENodes(bkeNodes, e.Ctx.Cluster) {
		return false
	}

	// 检查是否有关联的节点（从 BKENode CRD 获取）
	hasNodes := len(bkeNodes) > 0
	if !hasNodes && new.Spec.ClusterConfig != nil {
		return new.Spec.ClusterConfig.Addons != nil
	}
	if new.Spec.ClusterConfig == nil {
		return false
	}

	_, ok := bkeaddon.CompareBKEConfigAddon(new.Status.AddonStatus, new.Spec.ClusterConfig.Addons)
	if !ok {
		return false
	}
	e.SetStatus(bkev1beta1.PhaseWaiting)
	return true
}

// ValidateAndPrepareParams 包含 validateAndPrepare 函数的参数
type ValidateAndPrepareParams struct {
	Ctx *phaseframe.PhaseContext
}

// ValidateAndPrepareResult 包含 validateAndPrepare 函数的返回结果
type ValidateAndPrepareResult struct {
	AddonsT    []*bkeaddon.AddonTransfer
	BKECluster *bkev1beta1.BKECluster
	Client     client.Client
	Log        *bkev1beta1.BKELogger
	Continue   bool
}

// validateAndPrepare 验证和准备数据
func (e *EnsureAddonDeploy) validateAndPrepare(params ValidateAndPrepareParams) ValidateAndPrepareResult {
	phaseContextFields162 := params.Ctx.Untie()
	_, c, bkeCluster, _, log := phaseContextFields162.Context,
		phaseContextFields162.Client,
		phaseContextFields162.BKECluster,
		phaseContextFields162.Scheme,
		phaseContextFields162.Log
	addonsT, ok := bkeaddon.CompareBKEConfigAddon(bkeCluster.Status.AddonStatus, bkeCluster.Spec.ClusterConfig.Addons)
	if !ok {
		return ValidateAndPrepareResult{Continue: false}
	}

	log.Info(constant.AddonDeployingReason, "start to reconcile addon for target cluster %q", bkeCluster.Name)
	log.Info(constant.AddonDeployingReason, "A total of %d Addons need to be reconcile", len(addonsT))

	condition.ConditionMark(bkeCluster, bkev1beta1.ClusterAddonCondition, confv1beta1.ConditionFalse, "", "addon deloying")

	return ValidateAndPrepareResult{
		AddonsT:    addonsT,
		BKECluster: bkeCluster,
		Client:     c,
		Log:        log,
		Continue:   true,
	}
}

// ProcessAddonParams 包含 processAddon 函数的参数
type ProcessAddonParams struct {
	AddonT              *bkeaddon.AddonTransfer
	BKECluster          *bkev1beta1.BKECluster
	TargetClusterClient kube.RemoteKubeClient
	Client              client.Client
	Ctx                 *phaseframe.PhaseContext
	Log                 *bkev1beta1.BKELogger
}

// ProcessAddonResult 包含 processAddon 函数的返回结果
type ProcessAddonResult struct {
	NewestBKECluster *bkev1beta1.BKECluster
	Error            error
	Continue         bool
}

// processAddon 处理单个addon
func (e *EnsureAddonDeploy) processAddon(params ProcessAddonParams) ProcessAddonResult {
	if params.AddonT.Operate == bkeaddon.CreateAddon {
		if err := e.addonBeforeCreateCustomOperate(params.AddonT.Addon); err != nil {
			return ProcessAddonResult{Error: fmt.Errorf("addon %q before create custom operate failed: %w", params.AddonT.Addon.Name, err), Continue: false}
		}
	}

	params.Log.Info(constant.AddonDeployingReason, "start to %s addon %q", params.AddonT.Operate, params.AddonT.Addon.Name)

	// Get nodes for addon installation
	bkeNodes, err := params.Ctx.GetNodes()
	if err != nil {
		return ProcessAddonResult{Error: fmt.Errorf("failed to get nodes for addon %q: %w", params.AddonT.Addon.Name, err), Continue: false}
	}

	addonRecorder := kube.NewAddonRecorder(params.AddonT)
	operateErr := params.TargetClusterClient.InstallAddon(params.BKECluster, params.AddonT, addonRecorder, params.Client, bkeNodes)

	// 操作单个addon可能会花费许多时间这期间使用者看可能会修改bc，需要使用最新的bc来操作
	newestBkeCluster, err := params.Ctx.GetNewestBKECluster()
	if err != nil {
		return ProcessAddonResult{Error: err, Continue: false}
	}

	if operateErr != nil {
		errInfo := fmt.Sprintf("%s %q failed: %s", params.AddonT.Operate, params.AddonT.Addon.Name, operateErr.Error())
		condition.AddonConditionMark(condition.MarkOptions{
			BKECluster: newestBkeCluster,
			Type:       confv1beta1.ClusterConditionType(params.AddonT.Addon.Name),
			Status:     confv1beta1.ConditionFalse,
			Reason:     "ReconcileAddonError",
			Message:    errInfo,
			AddonName:  params.AddonT.Addon.Name,
		})

		if params.AddonT.Addon.Block {
			return ProcessAddonResult{Error: fmt.Errorf("%s %q failed: %w", params.AddonT.Operate, params.AddonT.Addon.Name, operateErr), Continue: false}
		} else {
			params.Log.Warn(constant.AddonDeployFailedReason, "%s %q failed (ignore): %s", params.AddonT.Operate, params.AddonT.Addon.Name, operateErr.Error())
			return ProcessAddonResult{NewestBKECluster: newestBkeCluster, Continue: true}
		}
	} else {
		successInfo := fmt.Sprintf("%s %s/%s success", params.AddonT.Operate, params.AddonT.Addon.Name, params.AddonT.Addon.Version)
		condition.AddonConditionMark(condition.MarkOptions{
			BKECluster: newestBkeCluster,
			Type:       confv1beta1.ClusterConditionType(params.AddonT.Addon.Name),
			Status:     confv1beta1.ConditionTrue,
			Reason:     "ReconcileAddonSuccess",
			Message:    successInfo,
			AddonName:  params.AddonT.Addon.Name,
		})
		params.Log.Info(constant.AddonDeploySucceededReason, successInfo)

		if params.AddonT.Operate == bkeaddon.CreateAddon {
			if err := e.addonAfterCreateCustomOperate(params.AddonT.Addon, newestBkeCluster); err != nil {
				return ProcessAddonResult{Error: err, Continue: false}
			}
		}
	}

	return ProcessAddonResult{NewestBKECluster: newestBkeCluster, Continue: true}
}

// UpdateAddonStatusParams 包含 updateAddonStatus 函数的参数
type UpdateAddonStatusParams struct {
	AddonT           *bkeaddon.AddonTransfer
	NewestBKECluster *bkev1beta1.BKECluster
	AddonRecorder    *kube.AddonRecorder
	AddonRecorders   *[]*kube.AddonRecorder
	Client           client.Client
	Ctx              *phaseframe.PhaseContext
	Log              *bkev1beta1.BKELogger
}

// updateAddonStatus 更新addon状态
func (e *EnsureAddonDeploy) updateAddonStatus(params UpdateAddonStatusParams) error {
	switch params.AddonT.Operate {
	case bkeaddon.RemoveAddon:
		condition.RemoveCondition(confv1beta1.ClusterConditionType(params.AddonT.Addon.Name), params.NewestBKECluster)
		// remove addon from status
		for i, addon := range params.NewestBKECluster.Status.AddonStatus {
			if addon.Name == params.AddonT.Addon.Name {
				params.NewestBKECluster.Status.AddonStatus = append(params.NewestBKECluster.Status.AddonStatus[:i], params.NewestBKECluster.Status.AddonStatus[i+1:]...)
				break
			}
		}
	case bkeaddon.UpdateAddon:
		fallthrough
	case bkeaddon.UpgradeAddon:
		// update Status
		for i, addon := range params.NewestBKECluster.Status.AddonStatus {
			if addon.Name == params.AddonT.Addon.Name {
				params.NewestBKECluster.Status.AddonStatus[i] = *params.AddonT.Addon
				break
			}
		}
	case bkeaddon.CreateAddon:
		// update Status
		params.NewestBKECluster.Status.AddonStatus = append(params.NewestBKECluster.Status.AddonStatus, *params.AddonT.Addon)
		e.addonAfterCreateNotifyOperate(params.AddonT.Addon, params.NewestBKECluster)
	default:
	}

	// 不删除都记录
	if params.AddonT.Operate != bkeaddon.RemoveAddon {
		*params.AddonRecorders = append(*params.AddonRecorders, params.AddonRecorder)
	}

	if err := mergecluster.SyncStatusUntilComplete(params.Client, params.NewestBKECluster); err != nil {
		return fmt.Errorf("update bkecluster %q Status failed: %w", params.NewestBKECluster.Name, err)
	}
	err := params.Ctx.RefreshCtxBKECluster()
	if err != nil {
		return err
	}

	return nil
}

func (e *EnsureAddonDeploy) reconcileAddon() error {
	// 验證和準備數據
	prepareParams := ValidateAndPrepareParams{
		Ctx: e.Ctx,
	}
	prepareResult := e.validateAndPrepare(prepareParams)
	if !prepareResult.Continue {
		return nil
	}

	var errs []error
	for _, addonT := range prepareResult.AddonsT {
		// 对于暂停的BKECluster，不执行
		if prepareResult.BKECluster.Spec.Pause || annotations.HasPaused(prepareResult.BKECluster) {
			prepareResult.Log.Info(constant.AddonDeployedReason, "BKECluster deploy paused, stop deploy addon")
			break
		}

		// 处理单个addon
		processParams := ProcessAddonParams{
			AddonT:              addonT,
			BKECluster:          prepareResult.BKECluster,
			TargetClusterClient: e.targetClusterClient,
			Client:              prepareResult.Client,
			Ctx:                 e.Ctx,
			Log:                 prepareResult.Log,
		}
		processResult := e.processAddon(processParams)

		if processResult.Error != nil {
			errs = append(errs, errors.Wrap(processResult.Error, addonT.Addon.Name))
			if addonT.Addon.Block {
				return processResult.Error
			} else {
				continue
			}
		}

		if !processResult.Continue {
			continue
		}

		// 更新addon状态
		addonRecorder := kube.NewAddonRecorder(addonT)
		updateParams := UpdateAddonStatusParams{
			AddonT:           addonT,
			NewestBKECluster: processResult.NewestBKECluster,
			AddonRecorder:    addonRecorder,
			AddonRecorders:   &e.addonRecorders,
			Client:           prepareResult.Client,
			Ctx:              e.Ctx,
			Log:              prepareResult.Log,
		}
		if err := e.updateAddonStatus(updateParams); err != nil {
			return err
		}
		prepareResult.BKECluster = processResult.NewestBKECluster
	}

	prepareResult.Log.Info(constant.AddonDeployedReason, "%d Addons were reconciled, %d succeeded, %d failed", len(prepareResult.AddonsT), len(prepareResult.AddonsT)-len(errs), len(errs))
	if len(errs) > 0 {
		return fmt.Errorf("reconcile addons failed: %w", kerrors.NewAggregate(errs))
	}
	condition.ConditionMark(prepareResult.BKECluster, bkev1beta1.ClusterAddonCondition, confv1beta1.ConditionTrue, "", "")

	return nil
}

func (e *EnsureAddonDeploy) addonBeforeCreateCustomOperate(addon *confv1beta1.Product) error {
	switch addon.Name {
	case "etcdbackup":
		return e.handleEtcdBackup(addon)
	case "beyondELB":
		return e.handleBeyondELB()
	case "cluster-api":
		return e.handleClusterAPI()
	case constant.OpenFuyaoSystemController:
		return e.handleOpenFuyaoSystemController()
	case "gpu-manager":
		return e.handleGPUManager()

	default:
		return nil
	}
}

func (e *EnsureAddonDeploy) handleEtcdBackup(addon *confv1beta1.Product) error {
	if err := e.createEtcdBackupDir(addon.Param["backupDir"]); err != nil {
		return fmt.Errorf("create etcdbackup dir failed: %w", err)
	}
	if err := e.createEtcdCertSecret(); err != nil {
		return fmt.Errorf("create etcd cert secret for addon etcdbackup failed: %w", err)
	}
	return nil
}

func (e *EnsureAddonDeploy) handleBeyondELB() error {
	vip, lbNodes := clusterutil.GetIngressConfig(e.Ctx.BKECluster.Spec.ClusterConfig.Addons)
	if vip == "" && len(lbNodes) == 0 {
		e.Ctx.Log.Warn("DeployAddonWarn", "(bke ignore)create beyondELB failed: vip or lbNodes is empty")
		return nil
	}
	if err := e.createBeyondELBVIP(vip, lbNodes); err != nil {
		return fmt.Errorf("create beyondELB VIP failed: %w", err)
	}
	if err := e.labelNodesForELB(lbNodes); err != nil {
		return fmt.Errorf("label nodes for beyondELB failed: %w", err)
	}
	return nil
}

func (e *EnsureAddonDeploy) handleClusterAPI() error {
	if err := e.createClusterAPILocalkubeconfigSecret(); err != nil {
		return fmt.Errorf("create cluster-api local kubeconfig secret failed: %w", err)
	}
	if err := e.createClusterAPILeastPrivilegeKubeConfigSecret(); err != nil {
		return fmt.Errorf("create cluster-api least privilege kubeconfig secret failed: %w", err)
	}
	if err := e.markBKEAgentSwitchPending(); err != nil {
		return fmt.Errorf("mark bkeagent switch pending failed: %w", err)
	}
	if err := e.createClusterAPIBkeconfigCm(); err != nil {
		return fmt.Errorf("create cluster-api bke config configmap failed: %w", err)
	}
	if err := e.createClusterAPIHealthCheckConfigCm(); err != nil {
		return fmt.Errorf("create cluster-api health check config configmap failed: %w", err)
	}
	if err := e.createClusterAPIPatchconfigCm(); err != nil {
		return fmt.Errorf("create cluster-api patch config configmap failed: %w", err)
	}
	if err := e.createChartRefToBKECluster(); err != nil {
		return fmt.Errorf("create chart addon ref to bkecluster failed: %w", err)
	}
	return nil
}

func (e *EnsureAddonDeploy) createChartRefToBKECluster() error {
	for _, addon := range e.Ctx.BKECluster.Spec.ClusterConfig.Addons {
		if addon.Type != bkeaddon.ChartAddon {
			continue
		}
		if err := e.createChartAddonCMRefToBKECluster(addon); err != nil {
			return err
		}
	}

	if err := e.createChartRepoSecretRefToBKECluster(); err != nil {
		return err
	}
	return nil
}

func (e *EnsureAddonDeploy) createChartAddonCMRefToBKECluster(addon confv1beta1.Product) error {
	if addon.ValuesConfigMapRef == nil {
		return nil
	}
	phaseContextFields461 := e.Ctx.Untie()
	ctx, c, bkeCluster, _, _ := phaseContextFields461.Context,
		phaseContextFields461.Client,
		phaseContextFields461.BKECluster,
		phaseContextFields461.Scheme,
		phaseContextFields461.Log

	namespace := addon.ValuesConfigMapRef.Namespace
	if namespace == "" {
		namespace = bkeCluster.Namespace
	}

	// Get local values.yaml configMap
	localCM := &corev1.ConfigMap{}
	if err := c.Get(ctx, client.ObjectKey{Namespace: namespace, Name: addon.ValuesConfigMapRef.Name}, localCM); err != nil {
		return fmt.Errorf("failed to get local values.yaml configmap: %w", err)
	}

	clSet := e.getClient()
	// Ensure namespace exists
	ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: namespace}}
	if _, err := clSet.CoreV1().Namespaces().Create(ctx, ns, metav1.CreateOptions{}); err != nil && !apierrors.IsAlreadyExists(err) {
		return fmt.Errorf("failed to create remote ns %s : %w", namespace, err)
	}

	// Create or update remote configMap
	remoteCM := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      addon.ValuesConfigMapRef.Name,
			Namespace: namespace,
		},
		Data: localCM.Data,
	}

	if _, err := clSet.CoreV1().ConfigMaps(namespace).Create(ctx, remoteCM, metav1.CreateOptions{}); err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("failed to create remote values.yaml cm: %w", err)
		}
		if err := phaseutil.RetryOnConflict(func() error {
			existingConfigMap, err := clSet.CoreV1().ConfigMaps(namespace).Get(ctx, remoteCM.Name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			existingConfigMap.Data = remoteCM.Data
			_, err = clSet.CoreV1().ConfigMaps(namespace).Update(ctx, existingConfigMap, metav1.UpdateOptions{})
			return err
		}); err != nil {
			return fmt.Errorf("failed to update remote values.yaml cm: %w", err)
		}
	}
	return nil
}

func (e *EnsureAddonDeploy) createChartRepoSecretRefToBKECluster() error {
	phaseContextFields510 := e.Ctx.Untie()
	ctx, c, bkeCluster, _, _ := phaseContextFields510.Context,
		phaseContextFields510.Client,
		phaseContextFields510.BKECluster,
		phaseContextFields510.Scheme,
		phaseContextFields510.Log
	authSecretRef := bkeCluster.Spec.ClusterConfig.Cluster.ChartRepo.AuthSecretRef
	tlsSecretRef := bkeCluster.Spec.ClusterConfig.Cluster.ChartRepo.TlsSecretRef

	var secrets [][]string

	if authSecretRef != nil {
		namespace := authSecretRef.Namespace
		if namespace == "" {
			namespace = bkeCluster.Namespace
		}
		secrets = append(secrets, []string{authSecretRef.Name, namespace})
	}

	if tlsSecretRef != nil {
		namespace := tlsSecretRef.Namespace
		if namespace == "" {
			namespace = bkeCluster.Namespace
		}
		secrets = append(secrets, []string{tlsSecretRef.Name, namespace})
	}

	for _, item := range secrets {
		localSecret := &corev1.Secret{}
		if err := c.Get(ctx, client.ObjectKey{Namespace: item[1], Name: item[0]}, localSecret); err != nil {
			return fmt.Errorf("failed to get local repo secret: %w", err)
		}

		clSet := e.getClient()
		// Ensure namespace exists
		ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: item[1]}}
		if _, err := clSet.CoreV1().Namespaces().Create(ctx, ns, metav1.CreateOptions{}); err != nil && !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("failed to create remote ns %s : %w", item[1], err)
		}

		// Create or update remote secret
		remoteSecret := &corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      item[0],
				Namespace: item[1],
			},
			Data: localSecret.Data,
		}

		if _, err := clSet.CoreV1().Secrets(item[1]).Create(ctx, remoteSecret, metav1.CreateOptions{}); err != nil {
			if !apierrors.IsAlreadyExists(err) {
				return fmt.Errorf("failed to create remote repo secret: %w", err)
			}
			if err := phaseutil.RetryOnConflict(func() error {
				existingSecret, err := clSet.CoreV1().Secrets(item[1]).Get(ctx, remoteSecret.Name, metav1.GetOptions{})
				if err != nil {
					return err
				}
				existingSecret.Data = remoteSecret.Data
				_, err = clSet.CoreV1().Secrets(item[1]).Update(ctx, existingSecret, metav1.UpdateOptions{})
				return err
			}); err != nil {
				return fmt.Errorf("failed to update remote repo secretm: %w", err)
			}
		}
	}
	return nil
}

func (e *EnsureAddonDeploy) handleGPUManager() error {
	if err := e.reCreateKubeSchedulerStaticPodYaml(); err != nil {
		return fmt.Errorf("recreate kube-scheduler static pod yaml failed: %w", err)
	}
	return nil
}

func (e *EnsureAddonDeploy) addonAfterCreateCustomOperate(addon *confv1beta1.Product, bkeCluster *bkev1beta1.BKECluster) error {
	switch addon.Name {
	case "cluster-api":
		return e.handleClusterAPIAfterCreate(bkeCluster)
	default:
		return nil
	}
}

// handleClusterAPIAfterCreate cluster-api 安装后同步 ref 引用的 KCT/CCT 到目标集群
func (e *EnsureAddonDeploy) handleClusterAPIAfterCreate(bkeCluster *bkev1beta1.BKECluster) error {
	phaseContextFields592 := e.Ctx.Untie()
	ctx, c, _, _, _ := phaseContextFields592.Context,
		phaseContextFields592.Client,
		phaseContextFields592.BKECluster,
		phaseContextFields592.Scheme,
		phaseContextFields592.Log
	remoteClientSet, remoteDynamic := e.targetClusterClient.KubeClient()
	if err := phaseutil.SyncReferencedKubeletAndContainerdConfig(ctx, c, remoteClientSet, remoteDynamic, bkeCluster); err != nil {
		return fmt.Errorf("sync referenced kubelet/containerd config after cluster-api install failed: %w", err)
	}
	return nil
}

func (e *EnsureAddonDeploy) addonAfterCreateNotifyOperate(addon *confv1beta1.Product, bkeCluster *bkev1beta1.BKECluster) {
	e.Ctx.Log.Info("CustomOperate", fmt.Sprintf("addon %q after create custom operate", addon.Name))
	switch addon.Name {
	case constant.OpenFuyaoSystemController:
		// 创建默认的用户名和密码
		cfg := phaseutil.UserInfoConfig{
			PasswdLength:  DefaultPasswordLength,
			SaltLength:    DefaultSaltLength,
			Iterations:    DefaultIterations,
			KeyLength:     DefaultKeyLength,
			EncryptMethod: sha256.New,
		}

		username, passwd, err := phaseutil.GenerateDefaultUserInfo(e.remoteDynamicClient, cfg)
		if err != nil {
			e.Ctx.Log.Warn("NotCreateDefaultUser", err.Error())
			return
		}
		if len(passwd) == 0 {
			e.Ctx.Log.Info("re-install", "re-install openFuyao-system")
			return
		}

		msg := `

	The website of the openFuyao is as follows:
	
		https://%s:%s

	You can login to the openFuyao using the following username and password:

		username: %s
		password: %s

`
		e.Ctx.Log.Info("openFuyaoSystemReady", fmt.Sprintf(msg, e.Ctx.BKECluster.Spec.ControlPlaneEndpoint.Host,
			constant.OpenFuyaoSystemPort, username, passwd))
	default:
		e.Ctx.Log.Info("CustomOperate", "no custom operate")
	}
}

func (e *EnsureAddonDeploy) addControlPlaneLabels() error {
	phaseContextFields643 := e.Ctx.Untie()
	ctx, _, _, _, log := phaseContextFields643.Context,
		phaseContextFields643.Client,
		phaseContextFields643.BKECluster,
		phaseContextFields643.Scheme,
		phaseContextFields643.Log

	remoteNodes, err := e.targetClusterClient.ListNodes(nil)
	if err != nil {
		return fmt.Errorf("failed to list nodes before deploy openfuyao-system: %w", err)
	}

	allNodes, err := e.Ctx.GetNodes()
	if err != nil {
		return fmt.Errorf("failed to get nodes: %w", err)
	}
	bkeNodes := bkenode.Nodes(allNodes).Master()
	for _, node := range remoteNodes.Items {
		for _, bkeNode := range bkeNodes {
			if node.GetName() != bkeNode.Hostname {
				continue
			}
			log.Info("BeforeCreateCustomOperate", "label node %s as control-plane", node.GetName())
			err = phaseutil.RetryOnConflict(func() error {
				latestNode, err := e.getClient().CoreV1().Nodes().Get(ctx, node.Name, metav1.GetOptions{})
				if err != nil {
					return err
				}
				labelhelper.SetMasterRoleLabel(latestNode)
				_, err = e.getClient().CoreV1().Nodes().Update(ctx, latestNode, metav1.UpdateOptions{})
				return err
			})
			if err != nil {
				return errors.Errorf("failed to set control-plane label to node %s: %s", node.Name, err.Error())
			}
			log.Info("BeforeCreateCustomOperate", "set control-plane label to node %s success", node.Name)
		}
	}
	return nil
}

func (e *EnsureAddonDeploy) getClient() kubernetes.Interface {
	if e.mockClient != nil {
		return e.mockClient
	}
	return e.remoteClient
}

func (e *EnsureAddonDeploy) distributePatchCM() error {
	phaseContextFields687 := e.Ctx.Untie()
	ctx, c, bkeCluster, _, _ := phaseContextFields687.Context,
		phaseContextFields687.Client,
		phaseContextFields687.BKECluster,
		phaseContextFields687.Scheme,
		phaseContextFields687.Log
	openFuyaoVersion := bkeCluster.Spec.ClusterConfig.Cluster.OpenFuyaoVersion
	bkeCMKey := fmt.Sprintf("patch.%s", openFuyaoVersion)
	patchCMKey := fmt.Sprintf("cm.%s", openFuyaoVersion)

	// Get local ConfigMap
	localCM := &corev1.ConfigMap{}
	if err := c.Get(ctx, constant.GetLocalConfigMapObjectKey(), localCM); err != nil {
		return fmt.Errorf("get cm failed: %w", err)
	}
	if _, ok := localCM.Data[bkeCMKey]; !ok {
		return fmt.Errorf("patch info %s not found in local config", bkeCMKey)
	}

	// Get patch ConfigMap
	patchCM := &corev1.ConfigMap{}
	if err := c.Get(ctx, client.ObjectKey{Namespace: "openfuyao-patch", Name: patchCMKey}, patchCM); err != nil {
		return fmt.Errorf("get patch cm failed: %w", err)
	}
	data, ok := patchCM.Data[openFuyaoVersion]
	if !ok {
		return fmt.Errorf("patch info %s not found in patch config", openFuyaoVersion)
	}
	// 确保目标集群存在openfuyao-system-controller命名空间，并创建或更新patch configmap
	clSet := e.getClient()
	// Ensure namespace exists
	nsName := constant.OpenFuyaoSystemController
	ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: nsName}}
	if _, err := clSet.CoreV1().Namespaces().Create(ctx, ns, metav1.CreateOptions{}); err != nil && !apierrors.IsAlreadyExists(err) {
		return fmt.Errorf("create remote ns %q failed: %w", nsName, err)
	}

	// Create or update remote ConfigMap
	remoteCM := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "patch-config",
			Namespace: nsName,
		},
		Data: map[string]string{"patch-data": data},
	}

	if _, err := clSet.CoreV1().ConfigMaps(nsName).Create(ctx, remoteCM, metav1.CreateOptions{}); err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("create remote cm failed: %w", err)
		}
		if err := phaseutil.RetryOnConflict(func() error {
			existingConfigMap, err := clSet.CoreV1().ConfigMaps(nsName).Get(ctx, remoteCM.Name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			existingConfigMap.Data = remoteCM.Data
			_, err = clSet.CoreV1().ConfigMaps(nsName).Update(ctx, existingConfigMap, metav1.UpdateOptions{})
			return err
		}); err != nil {
			return fmt.Errorf("update remote cm failed: %w", err)
		}
	}
	return nil
}

func (e *EnsureAddonDeploy) handleOpenFuyaoSystemController() error {
	err := e.addControlPlaneLabels()
	if err != nil {
		return fmt.Errorf("failed to add controller plane labels before deploy openfuyao-system: %w", err)
	}
	// 下发patch configmap到目标集群
	err = e.distributePatchCM()
	if err != nil {
		return fmt.Errorf("failed to distribute patch cm before deploy openfuyao-system: %w", err)
	}
	return nil
}

func (e *EnsureAddonDeploy) createEtcdBackupDir(dir string) error {
	phaseContextFields761 := e.Ctx.Untie()
	ctx, c, bkeCluster, scheme, log := phaseContextFields761.Context,
		phaseContextFields761.Client,
		phaseContextFields761.BKECluster,
		phaseContextFields761.Scheme,
		phaseContextFields761.Log

	allNodes, err := e.Ctx.GetNodes()
	if err != nil {
		return fmt.Errorf("failed to get nodes: %w", err)
	}
	bkeNodes := bkenode.Nodes(allNodes)

	baseCommandParams := CreateBaseCommandParams{
		Ctx:             ctx,
		NameSpace:       bkeCluster.Namespace,
		Client:          c,
		Scheme:          scheme,
		OwnerObj:        bkeCluster,
		ClusterName:     bkeCluster.Name,
		Unique:          true,
		RemoveAfterWait: true,
	}
	createDirCommand := command.Custom{
		BaseCommand:  createBaseCommand(baseCommandParams),
		Nodes:        bkeNodes.Etcd(),
		CommandName:  "create-etcd-backup-dir",
		CommandLabel: command.BKEClusterLabel,
	}

	commandSpec := command.GenerateDefaultCommandSpec()
	commandSpec.Commands = []agentv1beta1.ExecCommand{
		{
			ID: "mkdir",
			Command: []string{
				fmt.Sprintf("mkdir -p %s", dir),
			},
			Type:          agentv1beta1.CommandShell,
			BackoffIgnore: false,
		},
	}
	createDirCommand.CommandSpec = commandSpec

	if err := createDirCommand.New(); err != nil {
		return errors.Errorf("create create-etcd-backup-dir command failed: %s", err.Error())
	}

	err, successNodes, failedNodes := createDirCommand.Wait()
	if err != nil {
		return errors.Errorf("create create-etcd-backup-dir command failed: %s", err.Error())
	}
	if len(failedNodes) > 0 {
		errInfo := "create-etcd-backup-dir command run failed"
		commandErrs, err := phaseutil.LogCommandFailed(*createDirCommand.Command, failedNodes, log, "create beyondELB VIP failed")
		phaseutil.MarkNodeStatusByCommandErrs(ctx, c, bkeCluster, commandErrs)
		return fmt.Errorf("%s, createDirCommand run failed in flow nodes: %v, err: %w", errInfo, strings.Join(failedNodes, ","), err)
	}
	log.Info("BeforeCreateCustomOperate", "create etcd backup dir to fellow nodes success: %v", successNodes)
	return nil
}

func (e *EnsureAddonDeploy) createEtcdCertSecret() error {
	phaseContextFields818 := e.Ctx.Untie()
	ctx, c, bkeCluster, _, log := phaseContextFields818.Context,
		phaseContextFields818.Client,
		phaseContextFields818.BKECluster,
		phaseContextFields818.Scheme,
		phaseContextFields818.Log
	// 获取secret
	_, err := e.getClient().CoreV1().Secrets("kube-system").Get(ctx, etcdCertsScretName, metav1.GetOptions{})
	if err != nil {
		if !apierrors.IsNotFound(err) {
			return err
		}

		log.Debug("create etcd cert secret for addon etcdbackup")

		certGetter := certs.NewBKEKubernetesCertGetter(ctx, c, bkeCluster)

		etcdCaCertContent, err := certGetter.GetCertContent(pkiutil.BKECertEtcdCA())
		if err != nil {
			return err
		}
		etcdClientCertContent, err := certGetter.GetCertContent(pkiutil.BKECertEtcdAPIClient())
		if err != nil {
			return err
		}

		etcdBackupSecret := &corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      etcdCertsScretName,
				Namespace: "kube-system",
			},
			StringData: map[string]string{
				"etcd-ca":   etcdCaCertContent.Cert,
				"etcd-cert": etcdClientCertContent.Cert,
				"etcd-key":  etcdClientCertContent.Key,
			},
		}
		_, err = e.getClient().CoreV1().Secrets(etcdBackupSecret.Namespace).Create(ctx, etcdBackupSecret, metav1.CreateOptions{})
		if err != nil {
			return err
		}
	}

	log.Info("BeforeCreateCustomOperate", "create etcd cert secret for addon etcdbackup success")
	return nil
}

func (e *EnsureAddonDeploy) createBeyondELBVIP(vip string, lbNodes []string) error {
	if vip == "" {
		return nil
	}

	phaseContextFields865 := e.Ctx.Untie()
	ctx, c, bkeCluster, scheme, log := phaseContextFields865.Context,
		phaseContextFields865.Client,
		phaseContextFields865.BKECluster,
		phaseContextFields865.Scheme,
		phaseContextFields865.Log

	cfg := bkeinit.BkeConfig(*bkeCluster.Spec.ClusterConfig)

	allNodes, err := e.Ctx.GetNodes()
	if err != nil {
		return fmt.Errorf("failed to get nodes: %w", err)
	}
	ingressNodes := phaseutil.ConvertELBNodesToBKENodes(lbNodes, allNodes)
	if len(ingressNodes) == 0 {
		return fmt.Errorf("no ingress node set for beyondELB")
	}

	log.Info("BeforeCreateCustomOperate", "create beyondELB VIP: %s to follow node(s): %v, please wait", vip, lbNodes)

	baseCommandParams1 := CreateBaseCommandParams{
		Ctx:             ctx,
		NameSpace:       bkeCluster.Namespace,
		Client:          c,
		Scheme:          scheme,
		OwnerObj:        bkeCluster,
		ClusterName:     bkeCluster.Name,
		Unique:          true,
		RemoveAfterWait: false, // HA command may not need RemoveAfterWait
	}
	VIPCommand := command.HA{
		BaseCommand:     createBaseCommand(baseCommandParams1),
		IngressNodes:    ingressNodes,
		IngressVIP:      vip,
		ThirdImageRepo:  cfg.ImageThirdRepo(),
		FuyaoImageRepo:  cfg.ImageFuyaoRepo(),
		ManifestsDir:    cfg.Cluster.Kubelet.ManifestsDir,
		VirtualRouterId: cfg.CustomExtra["ingressVirtualRouterId"],
		WaitVIP:         true,
	}

	if err := VIPCommand.New(); err != nil {
		errInfo := "failed to create beyondELB VIP command"
		return errors.Wrap(err, errInfo)
	}

	err, _, failedNodes := VIPCommand.Wait()
	if err != nil {
		errInfo := "failed to create beyondELB VIP"
		return errors.Wrap(err, errInfo)
	}
	if len(failedNodes) > 0 {
		errInfo := "failed to create beyondELB VIP"
		commandErrs, err := phaseutil.LogCommandFailed(*VIPCommand.Command, failedNodes, log, "create beyondELB VIP failed")
		phaseutil.MarkNodeStatusByCommandErrs(ctx, c, bkeCluster, commandErrs)
		return fmt.Errorf("%s, VIPCommand run failed in flow nodes: %v, err: %w", errInfo, strings.Join(failedNodes, ","), err)
	}

	log.Info("BeforeCreateCustomOperate", "create beyondELB VIP success")
	return nil
}

// FindMatchingNodesParams 包含 findMatchingNodes 函数的参数
type FindMatchingNodesParams struct {
	LbNodes             []string
	BKECluster          *bkev1beta1.BKECluster
	TargetClusterClient kube.RemoteKubeClient
}

// findMatchingNodes 查找匹配的节点
func (e *EnsureAddonDeploy) findMatchingNodes(params FindMatchingNodesParams) ([]corev1.Node, error) {
	// 从 BKENode CRD 获取节点列表
	nodes, err := e.Ctx.GetNodes()
	if err != nil {
		return nil, errors.Wrapf(err, "failed to fetch nodes for cluster")
	}
	ingressNodes := phaseutil.ConvertELBNodesToBKENodes(params.LbNodes, nodes)

	// set loadblance label to remote nodes
	remoteNodes, err := e.targetClusterClient.ListNodes(nil)
	if err != nil {
		return nil, errors.Errorf("failed to list remote nodes: %s", err.Error())
	}

	var labelNodes []corev1.Node
	for _, elbNode := range ingressNodes {
		for _, node := range remoteNodes.Items {
			if nodeMatchesELBNode(node, elbNode.IP, elbNode.Hostname) {
				labelNodes = append(labelNodes, node)
				break
			}
		}
	}

	return labelNodes, nil
}

func nodeMatchesELBNode(node corev1.Node, ip, hostname string) bool {
	for _, addr := range node.Status.Addresses {
		switch addr.Type {
		case corev1.NodeInternalIP:
			if addr.Address == ip {
				return true
			}
		case corev1.NodeHostName:
			if addr.Address == hostname {
				return true
			}
		default:
			continue
		}
	}
	return false
}

// LabelAndSaveNodesParams 包含 labelAndSaveNodes 函数的参数
type LabelAndSaveNodesParams struct {
	LabelNodes []corev1.Node
	Ctx        context.Context
	Client     kubernetes.Interface
	Log        *bkev1beta1.BKELogger
}

// labelAndSaveNodes 为节点设置标签并保存
func (e *EnsureAddonDeploy) labelAndSaveNodes(params LabelAndSaveNodesParams) error {
	for _, node := range params.LabelNodes {
		err := phaseutil.RetryOnConflict(func() error {
			latestNode, err := params.Client.CoreV1().Nodes().Get(params.Ctx, node.Name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			labelhelper.SetLabel(latestNode, labelhelper.BeyondELBLabelKey, labelhelper.BeyondELBLabelValue)
			_, err = params.Client.CoreV1().Nodes().Update(params.Ctx, latestNode, metav1.UpdateOptions{})
			return err
		})
		if err != nil {
			return errors.Errorf("failed to set beyondELB label to node %s: %s", node.Name, err.Error())
		}
		params.Log.Info("BeforeCreateCustomOperate", "set beyondELB label to node %s success", node.Name)
	}
	return nil
}

func (e *EnsureAddonDeploy) labelNodesForELB(lbNodes []string) error {
	phaseContextFields1002 := e.Ctx.Untie()
	ctx, _, bkeCluster, _, log := phaseContextFields1002.Context,
		phaseContextFields1002.Client,
		phaseContextFields1002.BKECluster,
		phaseContextFields1002.Scheme,
		phaseContextFields1002.Log

	// 查找匹配的节点
	findParams := FindMatchingNodesParams{
		LbNodes:             lbNodes,
		BKECluster:          bkeCluster,
		TargetClusterClient: e.targetClusterClient,
	}
	labelNodes, err := e.findMatchingNodes(findParams)
	if err != nil {
		return err
	}

	// 为节点设置标签并保存
	saveParams := LabelAndSaveNodesParams{
		LabelNodes: labelNodes,
		Ctx:        ctx,
		Client:     e.remoteClient,
		Log:        log,
	}
	return e.labelAndSaveNodes(saveParams)
}

func (e *EnsureAddonDeploy) createClusterAPILocalkubeconfigSecret() error {
	phaseContextFields1026 := e.Ctx.Untie()
	ctx, c, bkeCluster, _, log := phaseContextFields1026.Context,
		phaseContextFields1026.Client,
		phaseContextFields1026.BKECluster,
		phaseContextFields1026.Scheme,
		phaseContextFields1026.Log
	// 获取secret
	_, err := e.getClient().CoreV1().Secrets("kube-system").Get(ctx, constant.LocalKubeConfigName, metav1.GetOptions{})
	if err != nil {
		if !apierrors.IsNotFound(err) {
			return err
		}
		log.Debug("create localkubeconfig secret for cluster-api")
		certGetter := certs.NewBKEKubernetesCertGetter(ctx, c, bkeCluster)
		kubeconfigContent, err := certGetter.GetTargetClusterKubeconfig()
		if err != nil {
			return err
		}

		localKubeconfigSecret := &corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      constant.LocalKubeConfigName,
				Namespace: "kube-system",
			},
			StringData: map[string]string{
				"config": kubeconfigContent,
			},
		}
		_, err = e.getClient().CoreV1().Secrets(localKubeconfigSecret.Namespace).Create(ctx, localKubeconfigSecret, metav1.CreateOptions{})
		if err != nil {
			return err
		}
	}

	log.Info("BeforeCreateCustomOperate", "create localkubeconfig secret for cluster-api success")
	return nil
}

func (e *EnsureAddonDeploy) createClusterAPILeastPrivilegeKubeConfigSecret() error {
	phaseContextFields1060 := e.Ctx.Untie()
	ctx, c, bkeCluster, _, log := phaseContextFields1060.Context,
		phaseContextFields1060.Client,
		phaseContextFields1060.BKECluster,
		phaseContextFields1060.Scheme,
		phaseContextFields1060.Log

	log.Debug("create least privilege kubeconfig secret for cluster-api")

	remoteLocalKubeConfig, err := phaseutil.GetRemoteLocalKubeConfig(ctx, e.remoteClient)
	if err != nil {
		return errors.Wrap(err, "failed to get localkubeconfig from remote cluster")
	}

	leastPrivilegeKubeConfig, err := phaseutil.GenerateLowPrivilegeKubeConfig(ctx, c, bkeCluster, remoteLocalKubeConfig)
	if err != nil {
		return errors.Wrap(err, "failed to generate least privilege kubeconfig")
	}

	_, err = e.getClient().CoreV1().Secrets(metav1.NamespaceSystem).Get(ctx, constant.LeastPrivilegeKubeConfigName, metav1.GetOptions{})
	if err == nil {
		log.Info("BeforeCreateCustomOperate", "least privilege kubeconfig secret already exists in remote cluster")
		return nil
	}
	if !apierrors.IsNotFound(err) {
		return errors.Wrap(err, "failed to check least privilege kubeconfig secret in remote cluster")
	}

	leastPrivilegeKubeConfigSecret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      constant.LeastPrivilegeKubeConfigName,
			Namespace: metav1.NamespaceSystem,
		},
		StringData: map[string]string{
			"config": string(leastPrivilegeKubeConfig),
		},
	}
	_, err = e.getClient().CoreV1().Secrets(metav1.NamespaceSystem).Create(ctx, leastPrivilegeKubeConfigSecret, metav1.CreateOptions{})
	if err != nil {
		return errors.Wrap(err, "failed to create least privilege kubeconfig secret in remote cluster")
	}

	log.Info("BeforeCreateCustomOperate", "create least privilege kubeconfig secret for cluster-api success")
	return nil
}

func (e *EnsureAddonDeploy) createClusterAPIBkeconfigCm() error {
	phaseContextFields1102 := e.Ctx.Untie()
	ctx, c, bkeCluster, _, log := phaseContextFields1102.Context,
		phaseContextFields1102.Client,
		phaseContextFields1102.BKECluster,
		phaseContextFields1102.Scheme,
		phaseContextFields1102.Log
	config, err := phaseutil.GetRemoteBKEConfigCM(ctx, e.remoteClient)
	if err != nil {
		return fmt.Errorf("failed to get BKECluster %q remote cluster bke-config cm: %w",
			utils.ClientObjNS(bkeCluster), err)
	}
	if config == nil {
		if err = phaseutil.MigrateBKEConfigCM(ctx, c, e.remoteClient); err != nil {
			return fmt.Errorf("failed to migrate BKECluster %q bke-config cm to remote cluster: %w",
				utils.ClientObjNS(bkeCluster), err)
		}
	}
	log.Info("BeforeCreateCustomOperate", "create cluster-api bke-config cm for cluster-api success")
	return nil
}

func (e *EnsureAddonDeploy) createClusterAPIHealthCheckConfigCm() error {
	phaseContextFields1119 := e.Ctx.Untie()
	ctx, c, bkeCluster, _, log := phaseContextFields1119.Context,
		phaseContextFields1119.Client,
		phaseContextFields1119.BKECluster,
		phaseContextFields1119.Scheme,
		phaseContextFields1119.Log
	config, err := phaseutil.GetRemoteHealthCheckConfigCM(ctx, e.remoteClient)
	if err != nil {
		return fmt.Errorf("failed to get BKECluster %q remote cluster health-check-config cm: %w",
			utils.ClientObjNS(bkeCluster), err)
	}
	if config == nil {
		if err = phaseutil.MigrateHealthCheckConfigCM(ctx, c, e.remoteClient); err != nil {
			return fmt.Errorf("failed to migrate BKECluster %q health-check-config cm to remote cluster: %w",
				utils.ClientObjNS(bkeCluster), err)
		}
	}
	log.Info("BeforeCreateCustomOperate", "create cluster-api health-check-config cm for cluster-api success")
	return nil
}
func (e *EnsureAddonDeploy) createClusterAPIPatchconfigCm() error {
	phaseContextFields1135 := e.Ctx.Untie()
	ctx, c, bkeCluster, _, log := phaseContextFields1135.Context,
		phaseContextFields1135.Client,
		phaseContextFields1135.BKECluster,
		phaseContextFields1135.Scheme,
		phaseContextFields1135.Log

	if err := phaseutil.MigratePatchConfigCM(ctx, c, e.remoteClient); err != nil {
		return fmt.Errorf("failed to migrate BKECluster %q patch-config cm to remote cluster: %w",
			utils.ClientObjNS(bkeCluster), err)
	}

	log.Info("BeforeCreateCustomOperate", "create cluster-api bke-config cm for cluster-api success")
	return nil
}

func (e *EnsureAddonDeploy) markBKEAgentSwitchPending() error {
	phaseContextFields1147 := e.Ctx.Untie()
	_, c, bkeCluster, _, log := phaseContextFields1147.Context,
		phaseContextFields1147.Client,
		phaseContextFields1147.BKECluster,
		phaseContextFields1147.Scheme,
		phaseContextFields1147.Log
	annotation.SetAnnotation(bkeCluster, common.BKEAgentListenerAnnotationKey, common.BKEAgentListenerBkecluster)
	log.Info("BeforeCreateCustomOperate", "marked BKEAgent switch pending for %q, will execute after postprocess", utils.ClientObjNS(bkeCluster))
	return mergecluster.SyncStatusUntilComplete(c, bkeCluster)
}

// createBaseCommand 创建基础命令结构
/**
 * Creates a new BaseCommand with the provided parameters
 *
 * @param params - The parameters needed to create the BaseCommand
 * @return command.BaseCommand - The newly created BaseCommand instance
 */
func createBaseCommand(params CreateBaseCommandParams) command.BaseCommand {
	// Create and return a new BaseCommand with all the provided parameters
	return command.BaseCommand{
		Ctx:             params.Ctx,             // Context for the command
		NameSpace:       params.NameSpace,       // Namespace where the command will operate
		Client:          params.Client,          // Client to be used for the command
		Scheme:          params.Scheme,          // Scheme for the command execution
		OwnerObj:        params.OwnerObj,        // Owner object of the command
		ClusterName:     params.ClusterName,     // Name of the cluster
		Unique:          params.Unique,          // Unique identifier for the command
		RemoveAfterWait: params.RemoveAfterWait, // Flag to indicate removal after wait
	}
}

func (e *EnsureAddonDeploy) reCreateKubeSchedulerStaticPodYaml() error {
	phaseContextFields1175 := e.Ctx.Untie()
	ctx, c, bkeCluster, scheme, log := phaseContextFields1175.Context,
		phaseContextFields1175.Client,
		phaseContextFields1175.BKECluster,
		phaseContextFields1175.Scheme,
		phaseContextFields1175.Log

	allNodes, err := e.Ctx.GetNodes()
	if err != nil {
		return fmt.Errorf("failed to get nodes: %w", err)
	}
	bkeNodes := bkenode.Nodes(allNodes)

	baseCommandParams2 := CreateBaseCommandParams{
		Ctx:             ctx,
		NameSpace:       bkeCluster.Namespace,
		Client:          c,
		Scheme:          scheme,
		OwnerObj:        bkeCluster,
		ClusterName:     bkeCluster.Name,
		Unique:          true,
		RemoveAfterWait: true,
	}
	reCreateCommand := command.Custom{
		BaseCommand:  createBaseCommand(baseCommandParams2),
		Nodes:        bkeNodes.Master(),
		CommandName:  "recreate-kube-scheduler-static-pod-yaml",
		CommandLabel: command.BKEClusterLabel,
	}

	commands := []agentv1beta1.ExecCommand{
		{
			ID: "recreate-kube-scheduler-static-pod-yaml",
			Command: []string{
				mfplugin.Name,
				fmt.Sprintf("bkeConfig=%s:%s", bkeCluster.Namespace, bkeCluster.Name),
				"gpuEnable=true",
				"scope=kube-scheduler",
				fmt.Sprintf("manifestDir=%s", bkeCluster.Spec.ClusterConfig.Cluster.Kubelet.ManifestsDir),
			},
			Type:          agentv1beta1.CommandBuiltIn,
			BackoffIgnore: false,
		},
	}
	commandSpec := createCommandSpec(commands)
	reCreateCommand.CommandSpec = commandSpec

	if err := reCreateCommand.New(); err != nil {
		return errors.Errorf("create recreate kube-scheduler static pod yaml command failed: %s", err.Error())
	}

	err, successNodes, failedNodes := reCreateCommand.Wait()
	if err != nil {
		return errors.Errorf("wait recreate kube-scheduler static pod yaml command failed: %s", err.Error())
	}
	if len(failedNodes) > 0 {
		errInfo := "recreate-kube-scheduler-static-pod-yaml command run failed"
		commandErrs, err := phaseutil.LogCommandFailed(*reCreateCommand.Command, failedNodes, log, "recreate kube-scheduler static pod yaml failed")
		phaseutil.MarkNodeStatusByCommandErrs(ctx, c, bkeCluster, commandErrs)
		return fmt.Errorf("%s, recreate kube-scheduler static pod yaml command run failed in flow nodes: %v, err: %w", errInfo, strings.Join(failedNodes, ","), err)
	}
	log.Info("BeforeCreateCustomOperate", "recreate kube-scheduler static pod yaml command run success in flow nodes: %v", strings.Join(successNodes, ","))
	return nil
}

// PrepareDownloadCalicoCtlParamsResult 包含 prepareDownloadCalicoCtlParams 函数的返回结果
type PrepareDownloadCalicoCtlParamsResult struct {
	Ctx          context.Context
	Client       client.Client
	BKECluster   *bkev1beta1.BKECluster
	Scheme       *runtime.Scheme
	Log          *bkev1beta1.BKELogger
	CalicoCtlUrl string
	CfgString    string
	BkeNodes     bkenode.Nodes
}

// prepareDownloadCalicoCtlParams 准备下载calicoctl的参数
func (e *EnsureAddonDeploy) prepareDownloadCalicoCtlParams(version string) PrepareDownloadCalicoCtlParamsResult {
	phaseContextFields1249 := e.Ctx.Untie()
	ctx, c, bkeCluster, scheme, log := phaseContextFields1249.Context,
		phaseContextFields1249.Client,
		phaseContextFields1249.BKECluster,
		phaseContextFields1249.Scheme,
		phaseContextFields1249.Log
	cfg := bkeinit.BkeConfig(*bkeCluster.Spec.ClusterConfig)
	baseUrl := clusterutil.BuildYumRepoDownloadBaseURL(cfg)
	calicoCtlUrl := fmt.Sprintf("url=%s/calicoctl-%s-linux-{.arch}", baseUrl, version)
	cfgString := "apiVersion: projectcalico.org/v3\nkind: CalicoAPIConfig\nmetadata:\nspec:\n  datastoreType: 'kubernetes'\n  kubeconfig: '/etc/kubernetes/admin.conf'"

	// 从 BKENode CRD 获取节点列表
	bkeNodes, err := e.Ctx.GetNodes()
	if err != nil {
		log.Warn(constant.AddonDeployingReason, "Failed to fetch nodes for cluster, using empty nodes: %v", err)
		bkeNodes = bkenode.Nodes{}
	}

	return PrepareDownloadCalicoCtlParamsResult{
		Ctx:          ctx,
		Client:       c,
		BKECluster:   bkeCluster,
		Scheme:       scheme,
		Log:          log,
		CalicoCtlUrl: calicoCtlUrl,
		CfgString:    cfgString,
		BkeNodes:     bkeNodes,
	}
}

// CreateDownloadCommandParams 包含 createDownloadCommand 函数的参数
type CreateDownloadCommandParams struct {
	Ctx          context.Context
	Client       client.Client
	BKECluster   *bkev1beta1.BKECluster
	Scheme       *runtime.Scheme
	CalicoCtlUrl string
	CfgString    string
	BkeNodes     bkenode.Nodes
}

// CreateDownloadCommandResult 包含 createDownloadCommand 函数的返回结果
type CreateDownloadCommandResult struct {
	DownloadCommand command.Custom
	Error           error
}

// createDownloadCommand 创建下载命令
func (e *EnsureAddonDeploy) createDownloadCommand(params CreateDownloadCommandParams) CreateDownloadCommandResult {
	baseCommandParams3 := CreateBaseCommandParams{
		Ctx:             params.Ctx,
		NameSpace:       params.BKECluster.Namespace,
		Client:          params.Client,
		Scheme:          params.Scheme,
		OwnerObj:        params.BKECluster,
		ClusterName:     params.BKECluster.Name,
		Unique:          true,
		RemoveAfterWait: true,
	}
	downloadCommand := command.Custom{
		BaseCommand:  createBaseCommand(baseCommandParams3),
		Nodes:        params.BkeNodes.Master(),
		CommandName:  "config-download-calicoctl",
		CommandLabel: command.BKEClusterLabel,
	}

	commands := []agentv1beta1.ExecCommand{
		{
			ID: "download-calicoctl",
			Command: []string{
				downloadplugin.Name,
				params.CalicoCtlUrl,
				"rename=calicoctl",
				"saveto=/usr/bin",
				"chmod=755",
			},
			Type:          agentv1beta1.CommandBuiltIn,
			BackoffIgnore: false,
		},
		{
			ID: "calicoctl-config-dir",
			Command: []string{
				"sudo mkdir -p /etc/calico",
			},
			Type:          agentv1beta1.CommandShell,
			BackoffIgnore: false,
		},
		{
			ID: "calicoctl-config",
			Command: []string{
				fmt.Sprintf("sudo echo -e %q > /etc/calico/calicoctl.cfg", params.CfgString),
			},
			Type:          agentv1beta1.CommandShell,
			BackoffIgnore: false,
		},
	}
	commandSpec := createCommandSpec(commands)
	downloadCommand.CommandSpec = commandSpec

	return CreateDownloadCommandResult{
		DownloadCommand: downloadCommand,
	}
}

// ExecuteDownloadCommandParams 包含 executeDownloadCommand 函数的参数
type ExecuteDownloadCommandParams struct {
	DownloadCommand *command.Custom
	Log             *bkev1beta1.BKELogger
	BKECluster      *bkev1beta1.BKECluster
}

// ExecuteDownloadCommandResult 包含 executeDownloadCommand 函数的返回结果
type ExecuteDownloadCommandResult struct {
	Error error
}

// executeDownloadCommand 执行下载命令
func (e *EnsureAddonDeploy) executeDownloadCommand(params ExecuteDownloadCommandParams) ExecuteDownloadCommandResult {
	if err := params.DownloadCommand.New(); err != nil {
		return ExecuteDownloadCommandResult{
			Error: errors.Errorf("create download calicoctl command failed: %s", err.Error()),
		}
	}

	err, successNodes, failedNodes := params.DownloadCommand.Wait()
	if err != nil {
		return ExecuteDownloadCommandResult{
			Error: errors.Errorf("wait download calicoctl command failed: %s", err.Error()),
		}
	}
	if len(failedNodes) > 0 {
		errInfo := "download-calicoctl command run failed"
		commandErrs, err := phaseutil.LogCommandFailed(*params.DownloadCommand.Command, failedNodes, params.Log, "download calicoctl failed")
		phaseutil.MarkNodeStatusByCommandErrs(e.Ctx.Context, e.Ctx.Client, params.BKECluster, commandErrs)
		return ExecuteDownloadCommandResult{
			Error: errors.Errorf("%s, download calicoctl command run failed in flow nodes: %v, err: %v", errInfo, strings.Join(failedNodes, ","), err),
		}
	}
	params.Log.Info("BeforeCreateCustomOperate", "download calicoctl command run success in flow nodes: %v", strings.Join(successNodes, ","))
	return ExecuteDownloadCommandResult{
		Error: nil,
	}
}

func (e *EnsureAddonDeploy) downloadCalicoCtl(version string) error {
	// 准備參數
	params := e.prepareDownloadCalicoCtlParams(version)

	// 創建下載命令
	createCmdParams := CreateDownloadCommandParams{
		Ctx:          params.Ctx,
		Client:       params.Client,
		BKECluster:   params.BKECluster,
		Scheme:       params.Scheme,
		CalicoCtlUrl: params.CalicoCtlUrl,
		CfgString:    params.CfgString,
		BkeNodes:     params.BkeNodes,
	}
	createResult := e.createDownloadCommand(createCmdParams)

	// 執行下載命令
	executeParams := ExecuteDownloadCommandParams{
		DownloadCommand: &createResult.DownloadCommand,
		Log:             params.Log,
		BKECluster:      params.BKECluster,
	}
	executeResult := e.executeDownloadCommand(executeParams)

	return executeResult.Error
}

func (e *EnsureAddonDeploy) saveAddonManifestsPostHook(_ phaseframe.Phase, _ error) error {
	phaseContextFields1416 := e.Ctx.Untie()
	ctx, c, bkeCluster, scheme, log := phaseContextFields1416.Context,
		phaseContextFields1416.Client,
		phaseContextFields1416.BKECluster,
		phaseContextFields1416.Scheme,
		phaseContextFields1416.Log

	if e.addonRecorders != nil && len(e.addonRecorders) != 0 {
		allNodes, err := e.Ctx.GetNodes()
		if err != nil {
			return fmt.Errorf("failed to get nodes: %w", err)
		}
		bkeNodes := bkenode.Nodes(allNodes)
		for _, recorder := range e.addonRecorders {
			// convert recorder to agent command ,one recorder one command
			if len(recorder.AddonObjects) == 0 || recorder.AddonObjects == nil {
				continue
			}

			commandName := fmt.Sprintf("save-addon-manifests-%s-%s", strings.ToLower(recorder.AddonName), recorder.AddonVersion)
			baseCommandParams4 := CreateBaseCommandParams{
				Ctx:             ctx,
				NameSpace:       bkeCluster.Namespace,
				Client:          c,
				Scheme:          scheme,
				OwnerObj:        bkeCluster,
				ClusterName:     bkeCluster.Name,
				Unique:          true,
				RemoveAfterWait: true,
			}
			saveAddonManifestsCommand := command.Custom{
				BaseCommand:  createBaseCommand(baseCommandParams4),
				Nodes:        bkeNodes.Master(),
				CommandName:  commandName,
				CommandLabel: command.BKEClusterLabel,
			}

			// 生成addon对象的命令
			generateParams := GenerateCommandsForAddonObjectsParams{
				Recorder: recorder,
			}
			generateResult := e.generateCommandsForAddonObjects(generateParams)

			initialCommands := []agentv1beta1.ExecCommand{
				{
					ID: "mkdir",
					Command: []string{
						fmt.Sprintf("mkdir -p %s", generateResult.AddonManifestsDir),
					},
					Type:          agentv1beta1.CommandShell,
					BackoffIgnore: false,
					BackoffDelay:  3,
				},
			}
			// 合并初始命令和循环生成的命令
			allCommands := append(initialCommands, generateResult.Commands...)
			commandSpec := createCommandSpec(allCommands)
			saveAddonManifestsCommand.CommandSpec = commandSpec

			if err := saveAddonManifestsCommand.New(); err != nil {
				log.Warn(constant.AddonDeployedReason, "failed to create save addon manifests command: %v", err)
				continue
			}
			// Don't wait ttl 10m will delete this command
		}
	}

	return nil
}

// GenerateCommandsForAddonObjectsParams 包含 generateCommandsForAddonObjects 函数的参数
type GenerateCommandsForAddonObjectsParams struct {
	Recorder *kube.AddonRecorder
}

// GenerateCommandsForAddonObjectsResult 包含 generateCommandsForAddonObjects 函数的返回结果
type GenerateCommandsForAddonObjectsResult struct {
	Commands          []agentv1beta1.ExecCommand
	AddonManifestsDir string
	Error             error
}

// generateCommandsForAddonObjects 为addon对象生成命令
func (e *EnsureAddonDeploy) generateCommandsForAddonObjects(params GenerateCommandsForAddonObjectsParams) GenerateCommandsForAddonObjectsResult {
	addonManifestsDir := fmt.Sprintf("%s/%s-%s", constant.AddonManifestsDir, params.Recorder.AddonName, params.Recorder.AddonVersion)
	var commands []agentv1beta1.ExecCommand

	for _, obj := range params.Recorder.AddonObjects {
		cmd := ""
		objFileName := ""
		if obj.NameSpace != "" {
			objFileName = fmt.Sprintf("%s/%s_%s_%s.yaml", addonManifestsDir, obj.Kind, obj.NameSpace, obj.Name)
			cmd = fmt.Sprintf("KUBECONFIG=/etc/kubernetes/admin.conf kubectl get %s -n %s %s -oyaml > %s", obj.Kind, obj.NameSpace, obj.Name, objFileName)
		} else {
			objFileName = fmt.Sprintf("%s/%s_%s.yaml", addonManifestsDir, obj.Kind, obj.Name)
			cmd = fmt.Sprintf("KUBECONFIG=/etc/kubernetes/admin.conf kubectl get %s %s -oyaml > %s", obj.Kind, obj.Name, objFileName)
		}
		command := agentv1beta1.ExecCommand{
			ID:            objFileName,
			Command:       []string{cmd},
			Type:          agentv1beta1.CommandShell,
			BackoffIgnore: false,
		}

		commands = append(commands, command)
	}

	return GenerateCommandsForAddonObjectsResult{
		Commands:          commands,
		AddonManifestsDir: addonManifestsDir,
	}
}
