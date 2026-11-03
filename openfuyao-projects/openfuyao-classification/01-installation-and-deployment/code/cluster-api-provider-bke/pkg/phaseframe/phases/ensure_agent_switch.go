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

	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/command"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/condition"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
)

const (
	EnsureAgentSwitchName confv1beta1.BKEClusterPhase = "EnsureAgentSwitch"
)

type EnsureAgentSwitch struct {
	phaseframe.BasePhase
}

func NewEnsureAgentSwitch(ctx *phaseframe.PhaseContext) phaseframe.Phase {
	base := phaseframe.NewBasePhase(ctx, EnsureAgentSwitchName)
	return &EnsureAgentSwitch{
		BasePhase: base,
	}
}

func (e *EnsureAgentSwitch) Execute() (ctrl.Result, error) {
	if err := e.reconcileAgentSwitch(); err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{}, nil
}

func (e *EnsureAgentSwitch) NeedExecute(old *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster) bool {
	if !e.BasePhase.DefaultNeedExecute(old, new) {
		return false
	}

	if listener, ok := annotation.HasAnnotation(e.Ctx.BKECluster, common.BKEAgentListenerAnnotationKey); ok && listener == common.BKEAgentListenerCurrent {
		return false
	}
	if condition.HasConditionStatus(bkev1beta1.SwitchBKEAgentCondition, e.Ctx.BKECluster, confv1beta1.ConditionTrue) {
		return false
	}
	e.SetStatus(bkev1beta1.PhaseWaiting)
	return true
}

func (e *EnsureAgentSwitch) reconcileAgentSwitch() error {
	phaseContextFields73 := e.Ctx.Untie()
	ctx, c, bkeCluster, scheme, log := phaseContextFields73.Context,
		phaseContextFields73.Client,
		phaseContextFields73.BKECluster,
		phaseContextFields73.Scheme,
		phaseContextFields73.Log

	listener, ok := annotation.HasAnnotation(bkeCluster, common.BKEAgentListenerAnnotationKey)
	switch {
	case !ok:
		annotation.SetAnnotation(bkeCluster, common.BKEAgentListenerAnnotationKey, common.BKEAgentListenerCurrent)
		log.Info(constant.SwitchClusterSuccessReason, "skip switch BKEAgent listener, BKEAgent is listening current cluster")
		return mergecluster.SyncStatusUntilComplete(c, bkeCluster)
	case listener == common.BKEAgentListenerCurrent || !ok:
		log.Info(constant.SwitchClusterSuccessReason, "skip switch BKEAgent listener, BKEAgent is listening current cluster")
		return nil
	case listener == common.BKEAgentListenerBkecluster:
		log.Info(constant.SwitchClusterSuccessReason, "switch BKEAgent to listen BKECluster")
		nodes, err := e.Ctx.GetNodes()
		if err != nil {
			log.Finish(constant.SwitchClusterFailedReason, "BKEAgent switch BKECluster %q failed: %s", utils.ClientObjNS(bkeCluster), err.Error())
			return err
		}
		switchCommand := createSwitchCommand(ctx, c, bkeCluster, scheme, bkenode.Nodes(nodes))
		if err := switchCommand.New(); err != nil {
			log.Finish(constant.SwitchClusterFailedReason, "BKEAgent switch BKECluster %q failed: %s", utils.ClientObjNS(bkeCluster), err.Error())
			return nil
		}
		log.Info(constant.SwitchClusterSuccessReason, "BKEAgent switch BKECluster %q success", utils.ClientObjNS(bkeCluster))
		condition.ConditionMark(bkeCluster, bkev1beta1.SwitchBKEAgentCondition, confv1beta1.ConditionTrue, constant.SwitchClusterSuccessReason, "switch BKECluster %q success")

		return nil
	default:
		return nil
	}
}

// createSwitchCommand creates a new switch command with the given parameters
func createSwitchCommand(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, scheme *runtime.Scheme, nodes bkenode.Nodes) command.Switch {
	return command.Switch{
		BaseCommand: command.BaseCommand{
			Ctx:         ctx,
			Client:      c,
			NameSpace:   bkeCluster.Namespace,
			Scheme:      scheme,
			OwnerObj:    bkeCluster,
			ClusterName: bkeCluster.Name,
			Unique:      true,
		},
		Nodes:       nodes,
		ClusterName: bkeCluster.Name,
	}
}
