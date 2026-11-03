/******************************************************************
 * Copyright (c) 2026 ICBC Co., Ltd.
 * installer is licensed under Mulan PSL v2.
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

	"github.com/pkg/errors"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/statusmanage"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
)

const (
	// EnsureMasterConfigUpdateName 静态Pod配置更新阶段名称
	EnsureMasterConfigUpdateName confv1beta1.BKEClusterPhase = "EnsureMasterConfigUpdate"
)

// EnsureMasterConfigUpdate 用于在 Kubernetes 版本未变化、但控制面静态 Pod 配置
// （APIServer / ControllerManager / Scheduler 的 ExtraArgs、ExtraVolumes、
// CertSANs 等）发生变化时，仅重新生成静态 Pod manifest，跳过完整的版本升级流程。
//
// 触发判定基于 Status.ControlPlaneConfigHash 与 Spec 期望 hash 是否不同。
// 调谐失败次数上限取 statusmanage.ReconcileAllowedFailedCount（环境变量
// ALLOWED_FAILED_COUNT，默认 10）。未达上限时不改 Status hash（同 Spec 可自动
// 再试）；达到上限（彻底失败）后回写 desired hash。成功始终回写 desired。
// 人工 retry 只清失败计数，不改 Status hash；是否再执行只看 Spec 与 Status hash 是否不一致。
type EnsureMasterConfigUpdate struct {
	phaseframe.BasePhase
}

func NewEnsureMasterConfigUpdate(ctx *phaseframe.PhaseContext) phaseframe.Phase {
	base := phaseframe.NewBasePhase(ctx, EnsureMasterConfigUpdateName)
	return &EnsureMasterConfigUpdate{BasePhase: base}
}

// Execute 执行纯配置更新流程：为每个 Master 节点下发升级命令，
// 重新生成静态 Pod manifest 并等待组件就绪。
func (e *EnsureMasterConfigUpdate) Execute() (ctrl.Result, error) {
	return e.reconcileMasterConfigUpdate()
}

// NeedExecute 基于 Status.ControlPlaneConfigHash 与 Spec 期望 hash 判定
// 是否需要执行纯配置更新。
//
// 判定规则：
//  1. 满足通用执行条件（DefaultNeedExecute）；
//  2. 控制面已初始化（能获取到 Nodes）；
//  3. Kubernetes 版本已对齐（Spec==Status，升级由 EnsureMasterUpgrade 处理）；
//  4. Etcd 版本已对齐（Spec==Status，升级由 EnsureEtcdUpgrade 处理）；
//  5. 存在 Master 节点；
//  6. 期望哈希（Spec）与 Status.ControlPlaneConfigHash（上次尝试/成功指纹）不同。
//
// 注意：管控面三组件空值校验已由 Webhook（validateStaticPodConfigUpdate）在
// apply 时拦截，etcd 不可变字段变更也由 Webhook 拒绝，此处不再重复判断。
func (e *EnsureMasterConfigUpdate) NeedExecute(old *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster) bool {
	if !e.BasePhase.DefaultNeedExecute(old, new) {
		return false
	}
	bkeNodes, ok := canExecuteConfigUpdate(e.Ctx, new)
	if !ok {
		return false
	}
	// 必须存在 Master 节点
	masterNodes := phaseutil.GetBKEAllNodesFromNodesStatusWithBKENodes(bkeNodes).Master()
	if masterNodes.Length() == 0 {
		return false
	}

	// 基于 Status.ControlPlaneConfigHash（上次成功部署基线）计算期望 hash 结构体。
	desiredHash, err := phaseutil.CalculateComponentConfigHash(&new.Spec.ClusterConfig.Cluster)
	if err != nil {
		e.Ctx.Log.Error("failed to calculate desired component config hash", "err", err)
		return false
	}
	// 期望结构体为空说明 Spec 未配置任何控制面组件可变字段，无需触发。
	if desiredHash == (confv1beta1.ComponentConfigHash{}) {
		return false
	}
	// 成功基线为空说明集群尚未完成首次健康检查（创建/纳管阶段），
	// 或为存量集群首次升级到新代码。此时无法区分"从未部署过"与"配置发生变化"，
	// 不触发，避免创建阶段误刷新静态 Pod；基线由 EnsureCluster.initializeConfigHashes 建立。
	if new.Status.ControlPlaneConfigHash == (confv1beta1.ComponentConfigHash{}) {
		return false
	}
	if !phaseutil.ComponentConfigHashChanged(new.Status.ControlPlaneConfigHash, desiredHash) {
		return false
	}

	e.SetStatus(bkev1beta1.PhaseWaiting)
	return true
}

func (e *EnsureMasterConfigUpdate) reconcileMasterConfigUpdate() (ctrl.Result, error) {
	phaseContextFields121 := e.Ctx.Untie()
	ctx, c, bkeCluster, _, log := phaseContextFields121.Context,
		phaseContextFields121.Client,
		phaseContextFields121.BKECluster,
		phaseContextFields121.Scheme,
		phaseContextFields121.Log

	// 获取所有 Master 节点
	bkeNodes, err := e.Ctx.NodeFetcher().GetBKENodesWrapperForCluster(e.Ctx, bkeCluster)
	if err != nil {
		log.Error(constant.MasterConfigUpdateFailedReason, "failed to get BKENodes: %v", err)
		return ctrl.Result{Requeue: true}, errors.Wrap(err, "failed to get BKENodes")
	}
	masterNodes := phaseutil.GetBKEAllNodesFromNodesStatusWithBKENodes(bkeNodes).Master()
	if masterNodes.Length() == 0 {
		log.Info(constant.MasterConfigUpdatedReason, "no master node need config update")
		return ctrl.Result{}, nil
	}
	// 计算期望的 hash 结构体（用于下发命令与成功/失败后回写）
	desiredHash, err := phaseutil.CalculateComponentConfigHash(&bkeCluster.Spec.ClusterConfig.Cluster)
	if err != nil {
		log.Error(constant.MasterConfigUpdateFailedReason, "failed to calculate component config hash: %v", err)
		return ctrl.Result{}, err
	}

	// 升级节点，携带旧/新 hash 结构体供 bkeagent 逐组件判断
	upgradeParams := UpgradeMasterConfigNodesParams{
		Ctx:                       ctx,
		Client:                    c,
		BKECluster:                bkeCluster,
		NeedUpgradeNodes:          masterNodes,
		Log:                       log,
		ControlPlaneConfigOldHash: bkeCluster.Status.ControlPlaneConfigHash,
		ControlPlaneConfigNewHash: desiredHash,
	}
	if err = e.updateMasterConfig(upgradeParams); err != nil {
		log.Error(constant.MasterConfigUpdateFailedReason, "failed to update master config: %v", err)
		thorough := false
		syncErr := mergecluster.SyncStatusUntilComplete(c, bkeCluster, func(cluster *bkev1beta1.BKECluster) {
			thorough = phaseutil.RecordControlPlaneConfigUpdateFailure(
				cluster, desiredHash, statusmanage.ReconcileAllowedFailedCount)
		})
		if syncErr != nil {
			log.Error(constant.MasterConfigUpdateFailedReason, "failed to record ControlPlaneConfigUpdate failure: %v", syncErr)
		} else {
			if thorough {
				log.Info(constant.MasterConfigUpdateFailedReason, "control plane config update thoroughly failed, persisted attempted hash")
			}
		}
		return ctrl.Result{}, err
	}

	// 全部 Master 节点成功应用新 manifest 后，回写成功基线 hash 并清零失败计数。
	if err := mergecluster.SyncStatusUntilComplete(c, bkeCluster, func(cluster *bkev1beta1.BKECluster) {
		phaseutil.RecordControlPlaneConfigUpdateSuccess(cluster, desiredHash)
	}); err != nil {
		log.Error(constant.MasterConfigUpdateFailedReason, "failed to persist ControlPlaneConfigHash: %v", err)
		return ctrl.Result{}, err
	}

	log.Info(constant.MasterConfigUpdateSucceedReason, "update all master static pod config success")
	return ctrl.Result{}, nil
}

type UpgradeMasterConfigNodesParams struct {
	Ctx              context.Context
	Client           client.Client
	BKECluster       *bkev1beta1.BKECluster
	NeedUpgradeNodes bkenode.Nodes
	Log              *bkev1beta1.BKELogger
	// ControlPlaneConfigOldHash 旧的控制面 hash 结构体（来自 Status.ControlPlaneConfigHash）
	ControlPlaneConfigOldHash confv1beta1.ComponentConfigHash
	// ControlPlaneConfigNewHash 期望的控制面 hash 结构体（来自 Spec 计算）
	ControlPlaneConfigNewHash confv1beta1.ComponentConfigHash
}

// updateMasterConfig 对每个 Master 节点下发 Phase: bkev1beta1.UpgradeControlPlaneConfig 的信息，
// 重新生成静态 Pod manifest。
func (e *EnsureMasterConfigUpdate) updateMasterConfig(params UpgradeMasterConfigNodesParams) error {
	params.Log.Info(constant.MasterConfigUpdatingReason, "Start update master node static pod config")
	for _, node := range params.NeedUpgradeNodes {

		// 创建纯配置更新的 Command
		masterParams := CreateUpgradeCommandParams{
			Ctx:                       params.Ctx,
			Namespace:                 params.BKECluster.Namespace,
			Client:                    params.Client,
			Scheme:                    e.Ctx.Scheme,
			OwnerObj:                  params.BKECluster,
			ClusterName:               params.BKECluster.Name,
			Node:                      &node,
			BKEConfig:                 params.BKECluster.Name,
			Phase:                     bkev1beta1.UpgradeControlPlaneConfig,
			ControlPlaneConfigOldHash: params.ControlPlaneConfigOldHash,
			ControlPlaneConfigNewHash: params.ControlPlaneConfigNewHash,
		}

		upgrade := createUpgradeCommand(masterParams)
		params.Log.Info(constant.MasterConfigUpdatingReason, "start update MasterConfig node %s", phaseutil.NodeInfo(node))
		if err := upgrade.New(); err != nil {
			params.Log.Error(constant.MasterConfigUpdateFailedReason, "create config update command failed, node %q failed: %v", phaseutil.NodeInfo(node), err)
			return errors.Errorf("create config update command failed，node: %q, failed: %v", phaseutil.NodeInfo(node), err)
		}
		params.Log.Info(constant.MasterConfigUpdatingReason, "wait config update node %s finish", phaseutil.NodeInfo(node))
		err, _, failedNodes := upgrade.Wait()
		if err != nil {
			return errors.Errorf("wait config update command failed， node: %q, failed: %v", phaseutil.NodeInfo(node), err)
		}
		if len(failedNodes) != 0 {
			params.Log.Error(constant.MasterConfigUpdateFailedReason, "config update failed，node: %q, failedNodes: %v", phaseutil.NodeInfo(node), failedNodes)
			commandErrs, err := phaseutil.LogCommandFailed(*upgrade.Command, failedNodes, params.Log, constant.MasterConfigUpdateFailedReason)
			phaseutil.MarkNodeStatusByCommandErrs(params.Ctx, params.Client, params.BKECluster, commandErrs)
			return errors.Errorf("config update node %q failed: %v", phaseutil.NodeInfo(node), err)
		}
	}

	return nil
}

// kubernetesVersionNotSynced 判断 Spec 期望版本与 Status 实际版本是否尚未对齐。
// 二者不一致说明版本升级仍在进行中或异常卡住，此时不应执行纯配置更新，
// 避免与版本升级阶段并发。相比比较 old/new 的 Spec，Spec/Status 对齐状态才能
// 真实反映「升级是否已完成」。
func kubernetesVersionNotSynced(c *bkev1beta1.BKECluster) bool {
	if c == nil || c.Spec.ClusterConfig == nil {
		return true
	}
	return c.Spec.ClusterConfig.Cluster.KubernetesVersion != c.Status.KubernetesVersion
}

// etcdVersionNotSynced 判断 Etcd 的 Spec 期望版本与 Status 实际版本是否尚未对齐。
// Etcd 版本升级由 EnsureEtcdUpgrade 独立负责，本阶段在 hash 计算前显式排除，
// 避免纯版本变化时仍执行一次无意义的配置哈希计算。
func etcdVersionNotSynced(c *bkev1beta1.BKECluster) bool {
	if c == nil || c.Spec.ClusterConfig == nil {
		return true
	}
	return c.Spec.ClusterConfig.Cluster.EtcdVersion != c.Status.EtcdVersion
}

// canExecuteConfigUpdate 校验控制面已初始化、版本已对齐（Spec==Status），
// 并返回指定角色的节点列表。版本未对齐时返回 false，阻止纯配置更新与版本升级并发。
func canExecuteConfigUpdate(
	ctx *phaseframe.PhaseContext,
	new *bkev1beta1.BKECluster,
) (bkev1beta1.BKENodes, bool) {
	bkeNodes, ok := fetchBKENodesIfCPInitialized(ctx, new)
	if !ok {
		return nil, false
	}
	if etcdVersionNotSynced(new) || kubernetesVersionNotSynced(new) {
		return nil, false
	}
	return bkeNodes, true
}
