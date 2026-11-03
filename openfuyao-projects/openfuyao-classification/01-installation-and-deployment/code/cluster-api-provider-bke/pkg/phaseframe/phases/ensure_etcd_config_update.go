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
	// EnsureEtcdConfigUpdateName etcd静态Pod配置更新阶段名称
	EnsureEtcdConfigUpdateName confv1beta1.BKEClusterPhase = "EnsureEtcdConfigUpdate"
)

// EnsureEtcdConfigUpdate 用于在 Etcd 版本未变化、但 etcd 静态 Pod 配置
// （ExtraArgs、ExtraVolumes 等可变字段）发生变化时，仅重新生成 etcd 静态 Pod
// manifest，跳过完整的版本升级流程。
//
// 触发判定基于 Status.EtcdConfigHash 与 Spec 期望 hash 是否不同。
// 调谐失败次数上限取 statusmanage.ReconcileAllowedFailedCount（环境变量
// ALLOWED_FAILED_COUNT，默认 10）。未达上限时不改 Status hash；达到上限后回写
// desired。成功始终回写。人工 retry 只清失败计数，不改 Status hash；
// 是否再执行只看 Spec 与 Status hash 是否不一致。
type EnsureEtcdConfigUpdate struct {
	phaseframe.BasePhase
}

func NewEnsureEtcdConfigUpdate(ctx *phaseframe.PhaseContext) phaseframe.Phase {
	base := phaseframe.NewBasePhase(ctx, EnsureEtcdConfigUpdateName)
	return &EnsureEtcdConfigUpdate{BasePhase: base}
}

// Execute 执行 etcd 纯配置更新流程：为每个 etcd 节点下发升级命令，
// 重新生成 etcd 静态 Pod manifest 并等待组件就绪。
func (e *EnsureEtcdConfigUpdate) Execute() (ctrl.Result, error) {
	return e.reconcileEtcdConfigUpdate()
}

// NeedExecute 基于 Status.EtcdConfigHash 与 Spec 期望 hash 判定是否需要执行。
// 详情：
//  1. DefaultNeedExecute；
//  2. 控制面已初始化；
//  3. Etcd/K8s 版本已对齐（Spec==Status，由对应升级 phase 处理）；
//  4. 存在 etcd 节点；
//  5. 期望哈希（Spec）与 Status.EtcdConfigHash（上次尝试/成功指纹）不同。
//
// 注意：DataDir、ServerCertSANs、PeerCertSANs 不可变字段变更已由 Webhook
// validateStaticPodConfigUpdate 拒绝，此处不再重复判断。
func (e *EnsureEtcdConfigUpdate) NeedExecute(old *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster) bool {
	if !e.BasePhase.DefaultNeedExecute(old, new) {
		return false
	}

	bkeNodes, ok := canExecuteConfigUpdate(e.Ctx, new)
	if !ok {
		return false
	}

	etcdNodes := phaseutil.GetBKEAllNodesFromNodesStatusWithBKENodes(bkeNodes).Etcd()
	if etcdNodes.Length() == 0 {
		return false
	}

	// 基于 Status.EtcdConfigHash（上次成功部署基线）计算期望哈希。
	desiredHash, err := phaseutil.CalculateEtcdConfigHash(&new.Spec.ClusterConfig.Cluster)
	if err != nil {
		e.Ctx.Log.Error("failed to calculate desired etcd config hash", "err", err)
		return false
	}
	// 期望哈希为空说明 Spec 无 etcd 可变配置，无需触发。
	if desiredHash == "" {
		return false
	}
	// 成功基线为空说明集群尚未完成首次健康检查（创建/纳管阶段），
	// 或为存量集群首次升级到新代码。此时无法区分"从未部署过"与"配置发生变化"，
	// 不触发，避免创建阶段误刷新静态 Pod；基线由 EnsureCluster.initializeConfigHashes 建立。
	if new.Status.EtcdConfigHash == "" {
		return false
	}
	if desiredHash == new.Status.EtcdConfigHash {
		return false
	}

	e.SetStatus(bkev1beta1.PhaseWaiting)
	return true
}

func (e *EnsureEtcdConfigUpdate) reconcileEtcdConfigUpdate() (ctrl.Result, error) {
	phaseContextFields118 := e.Ctx.Untie()
	ctx, c, bkeCluster, _, log := phaseContextFields118.Context,
		phaseContextFields118.Client,
		phaseContextFields118.BKECluster,
		phaseContextFields118.Scheme,
		phaseContextFields118.Log

	bkeNodes, err := e.Ctx.NodeFetcher().GetBKENodesWrapperForCluster(e.Ctx, bkeCluster)
	if err != nil {
		log.Error(constant.EtcdConfigUpdateFailedReason, "failed to get BKENodes: %v", err)
		return ctrl.Result{Requeue: true}, errors.Wrap(err, "failed to get BKENodes")
	}
	etcdNodes := phaseutil.GetBKEAllNodesFromNodesStatusWithBKENodes(bkeNodes).Etcd()
	if etcdNodes.Length() == 0 {
		log.Info(constant.EtcdConfigUpdatedReason, "no etcd node need config update")
		return ctrl.Result{}, nil
	}

	upgradeParams := UpgradeEtcdConfigNodesParams{
		Ctx:              ctx,
		Client:           c,
		BKECluster:       bkeCluster,
		NeedUpgradeNodes: etcdNodes,
		Log:              log,
	}
	desiredHash, err := phaseutil.CalculateEtcdConfigHash(&bkeCluster.Spec.ClusterConfig.Cluster)
	if err != nil {
		log.Error(constant.EtcdConfigUpdateFailedReason, "failed to calculate etcd hash: %v", err)
		return ctrl.Result{}, err
	}
	if err := e.updateEtcdConfig(upgradeParams); err != nil {
		log.Error(constant.EtcdConfigUpdatingReason, "failed to update etcd config: %v", err)
		thorough := false
		syncErr := mergecluster.SyncStatusUntilComplete(c, bkeCluster, func(cluster *bkev1beta1.BKECluster) {
			thorough = phaseutil.RecordEtcdConfigUpdateFailure(
				cluster, desiredHash, statusmanage.ReconcileAllowedFailedCount)
		})
		if syncErr != nil {
			log.Error(constant.EtcdConfigUpdateFailedReason, "failed to record EtcdConfigUpdate failure: %v", syncErr)
		} else {
			if thorough {
				log.Info(constant.EtcdConfigUpdateFailedReason, "etcd config update thoroughly failed, persisted attempted hash")
			}
		}
		return ctrl.Result{}, err
	}

	// 全部 etcd 节点成功应用新 manifest 后，回写成功基线 hash 并清零失败计数。
	if err := mergecluster.SyncStatusUntilComplete(c, bkeCluster, func(cluster *bkev1beta1.BKECluster) {
		phaseutil.RecordEtcdConfigUpdateSuccess(cluster, desiredHash)
	}); err != nil {
		log.Error(constant.EtcdConfigUpdateFailedReason, "failed to persist EtcdConfigHash: %v", err)
		return ctrl.Result{}, err
	}

	log.Info(constant.EtcdConfigUpdateSucceedReason, "update all etcd static pod config success")
	return ctrl.Result{}, nil
}

type UpgradeEtcdConfigNodesParams struct {
	Ctx              context.Context
	Client           client.Client
	BKECluster       *bkev1beta1.BKECluster
	NeedUpgradeNodes bkenode.Nodes
	Log              *bkev1beta1.BKELogger
}

// updateEtcdConfig 对每个 etcd 节点下发 Phase: bkev1beta1.UpgradeEtcdConfig 的升级命令。
func (e *EnsureEtcdConfigUpdate) updateEtcdConfig(params UpgradeEtcdConfigNodesParams) error {
	params.Log.Info(constant.EtcdConfigUpdatingReason, "Start update etcd node static pod config")
	for _, node := range params.NeedUpgradeNodes {

		etcdParams := CreateUpgradeCommandParams{
			Ctx:         params.Ctx,
			Namespace:   params.BKECluster.Namespace,
			Client:      params.Client,
			Scheme:      e.Ctx.Scheme,
			OwnerObj:    params.BKECluster,
			ClusterName: params.BKECluster.Name,
			Node:        &node,
			BKEConfig:   params.BKECluster.Name,
			Phase:       bkev1beta1.UpgradeEtcdConfig,
		}

		upgrade := createUpgradeCommand(etcdParams)
		params.Log.Info(constant.EtcdConfigUpdatingReason, "start update EtcdConfig node %s", phaseutil.NodeInfo(node))
		if err := upgrade.New(); err != nil {
			params.Log.Error(constant.EtcdConfigUpdateFailedReason, "create config update command failed, node %q failed: %v", phaseutil.NodeInfo(node), err)
			return errors.Errorf("create config update command failed，node: %q, failed: %v", phaseutil.NodeInfo(node), err)
		}
		params.Log.Info(constant.EtcdConfigUpdatingReason, "wait config update node %s finish", phaseutil.NodeInfo(node))
		err, _, failedNodes := upgrade.Wait()
		if err != nil {
			return errors.Errorf("wait config update command failed， node: %q, failed: %v", phaseutil.NodeInfo(node), err)
		}
		if len(failedNodes) != 0 {
			params.Log.Error(constant.EtcdConfigUpdateFailedReason, "config update failed，node: %q, failedNodes: %v", phaseutil.NodeInfo(node), failedNodes)
			commandErrs, err := phaseutil.LogCommandFailed(*upgrade.Command, failedNodes, params.Log, constant.EtcdConfigUpdateFailedReason)
			phaseutil.MarkNodeStatusByCommandErrs(params.Ctx, params.Client, params.BKECluster, commandErrs)
			return errors.Errorf("config update node %q failed: %v", phaseutil.NodeInfo(node), err)
		}
	}

	return nil
}
