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

package phases

import (
	"fmt"

	"github.com/pkg/errors"
	"sigs.k8s.io/cluster-api/util/version"
	ctrl "sigs.k8s.io/controller-runtime"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/command"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/clusterutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
)

const (
	EnsureContainerdUpgradeName confv1beta1.BKEClusterPhase = "EnsureContainerdUpgrade"
)

type EnsureContainerdUpgrade struct {
	phaseframe.BasePhase
}

func NewEnsureContainerdUpgrade(ctx *phaseframe.PhaseContext) phaseframe.Phase {
	base := phaseframe.NewBasePhase(ctx, EnsureContainerdUpgradeName)
	return &EnsureContainerdUpgrade{BasePhase: base}
}

// Execute 执行具体的升级操作
func (e *EnsureContainerdUpgrade) Execute() (ctrl.Result, error) {
	return e.rolloutContainerd()
}

func (e *EnsureContainerdUpgrade) getCommand() *command.ENV {
	phaseContextFields51 := e.Ctx.Untie()
	ctx, c, bkeCluster, scheme, log := phaseContextFields51.Context,
		phaseContextFields51.Client,
		phaseContextFields51.BKECluster,
		phaseContextFields51.Scheme,
		phaseContextFields51.Log
	targetVersion := e.resolveContainerdUpgradeVersion()
	// Use NodeFetcher to get BKENodes from API server
	bkeNodes, err := e.Ctx.NodeFetcher().GetBKENodesWrapperForCluster(e.Ctx, bkeCluster)
	if err != nil {
		log.Warn(constant.ContainerdUpgradingReason, "failed to get BKENodes: %v", err)
		return nil
	}
	exceptEnvNodes := phaseutil.GetNeedUpgradeContainerdNodesWithBKENodes(bkeCluster, bkeNodes)
	log.Info(constant.ContainerdUpgradingReason,
		"containerd upgrade node selection: current=%s target=%s fetchedBKENodes=%d selectedNodes=%d",
		bkeCluster.Status.ContainerdVersion, targetVersion, len(bkeNodes), exceptEnvNodes.Length())
	if exceptEnvNodes.Length() == 0 {
		log.Warn(constant.ContainerdUpgradingReason,
			"containerd upgrade selected zero nodes: current=%s target=%s fetchedBKENodes=%d",
			bkeCluster.Status.ContainerdVersion, targetVersion, len(bkeNodes))
	}

	nodes, err := e.Ctx.GetNodes()
	if err != nil {
		log.Warn(constant.ContainerdUpgradingReason, "failed to get nodes: %v", err)
		return nil
	}

	var extra []string
	var extraHosts []string
	if clusterutil.AvailableLoadBalancerEndPoint(bkeCluster.Spec.ControlPlaneEndpoint, nodes) {
		extra = append(extra, bkeCluster.Spec.ControlPlaneEndpoint.Host)
		extraHosts = append(extraHosts, fmt.Sprintf("%s:%s", constant.MasterHADomain, bkeCluster.Spec.ControlPlaneEndpoint.Host))
	}
	ingressVip, _ := clusterutil.GetIngressConfig(bkeCluster.Spec.ClusterConfig.Addons)
	if ingressVip != "" && ingressVip != bkeCluster.Spec.ControlPlaneEndpoint.Host {
		extra = append(extra, ingressVip)
	}

	params := BuildCommonEnvCommandParams{
		Ctx:               ctx,
		Client:            c,
		BKECluster:        bkeCluster,
		Scheme:            scheme,
		ExceptEnvNodes:    exceptEnvNodes,
		ContainerdVersion: targetVersion,
		Extra:             extra,
		ExtraHosts:        extraHosts,
		DryRun:            bkeCluster.Spec.DryRun,
		Log:               log,
	}

	envCmd, err := BuildCommonEnvCommand(params)
	if err != nil {
		log.Warn(constant.ContainerdUpgradingReason, "failed to build common env command: %v", err)
		return nil
	}

	return envCmd
}

func (e *EnsureContainerdUpgrade) resetContainerd() error {
	phaseContextFields109 := e.Ctx.Untie()
	_, _, _, _, log := phaseContextFields109.Context,
		phaseContextFields109.Client,
		phaseContextFields109.BKECluster,
		phaseContextFields109.Scheme,
		phaseContextFields109.Log
	envCommand := e.getCommand()
	if envCommand == nil {
		return fmt.Errorf("new containerd command fail")
	}
	if err := envCommand.NewConatinerdReset(); err != nil {
		return fmt.Errorf("failed to create k8s containerd reset: %w", err)
	}

	log.Info(constant.ContainerdUpgradingReason, "Waiting for the k8s containerd reset to complete")
	err, successNodes, failedNodes := envCommand.Wait()
	if err != nil || len(failedNodes) > 0 {
		if err != nil {
			return fmt.Errorf("failed to reset k8s containerd: %d/%d: %w", len(successNodes), len(failedNodes)+len(successNodes), err)
		}
		return fmt.Errorf("failed to reset k8s containerd: %d/%d", len(successNodes), len(failedNodes)+len(successNodes))
	}
	return nil
}

func (e *EnsureContainerdUpgrade) redeployContainerd() error {
	phaseContextFields130 := e.Ctx.Untie()
	_, _, _, _, log := phaseContextFields130.Context,
		phaseContextFields130.Client,
		phaseContextFields130.BKECluster,
		phaseContextFields130.Scheme,
		phaseContextFields130.Log
	envCommand := e.getCommand()
	if envCommand == nil {
		return fmt.Errorf("new containerd command fail")
	}
	if err := envCommand.NewConatinerdRedeploy(); err != nil {
		return fmt.Errorf("failed to create k8s containerd redeploy: %w", err)
	}

	log.Info(constant.ContainerdUpgradingReason, "Waiting for the k8s containerd redeploy to complete")
	err, successNodes, failedNodes := envCommand.Wait()
	if err != nil || len(failedNodes) > 0 {
		if err != nil {
			return fmt.Errorf("failed to redeploy k8s containerd: %d/%d: %w", len(successNodes), len(failedNodes)+len(successNodes), err)
		}
		return fmt.Errorf("failed to redeploy k8s containerd: %d/%d", len(successNodes), len(failedNodes)+len(successNodes))
	}
	return nil
}

func (e *EnsureContainerdUpgrade) rolloutContainerd() (ctrl.Result, error) {
	phaseContextFields151 := e.Ctx.Untie()
	_, _, bkeCluster, _, log := phaseContextFields151.Context,
		phaseContextFields151.Client,
		phaseContextFields151.BKECluster,
		phaseContextFields151.Scheme,
		phaseContextFields151.Log
	if err := e.Ctx.SyncUpgradeTargetsToClusterSpec(); err != nil {
		return ctrl.Result{}, errors.Wrap(err, "sync containerd upgrade target to cluster spec")
	}
	targetVersion := e.resolveContainerdUpgradeVersion()

	err := e.resetContainerd()
	if err != nil {
		return ctrl.Result{}, err
	}
	err = e.redeployContainerd()
	if err != nil {
		return ctrl.Result{}, err
	}

	log.Info(constant.ContainerdUpgradeSuccess, "upgrade containerd success")
	bkeCluster.Status.ContainerdVersion = targetVersion

	return ctrl.Result{}, nil
}

func (e *EnsureContainerdUpgrade) resolveContainerdUpgradeVersion() string {
	if e.Ctx == nil || e.Ctx.BKECluster == nil {
		return ""
	}
	if vc := e.Ctx.VersionContext; vc != nil {
		if v := vc.GetTarget(upgrade.ComponentContainerd); v != "" {
			return v
		}
	}
	if e.Ctx.BKECluster.Spec.ClusterConfig == nil {
		return ""
	}
	return e.Ctx.BKECluster.Spec.ClusterConfig.Cluster.ContainerdVersion
}

func (e *EnsureContainerdUpgrade) isContainerdNeedUpgrade(old *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster) bool {
	newConfig := new.Spec.ClusterConfig
	if new.Status.ContainerdVersion == "" {
		return false
	}

	oldv, err := version.ParseMajorMinorPatch(new.Status.ContainerdVersion)
	if err != nil {
		return false
	}
	newv, err := version.ParseMajorMinorPatch(newConfig.Cluster.ContainerdVersion)
	if err != nil {
		return false
	}
	// step 2 compare cluster version upgrade
	switch version.Compare(newv, oldv) {
	case -1:
	case 0:
	case 1:
		bkeNodes, err := e.Ctx.NodeFetcher().GetBKENodesForBKECluster(e.Ctx.Context, e.Ctx.BKECluster)
		if err != nil {
			return false
		}
		statusNodes := bkev1beta1.NewBKENodes(bkeNodes)
		for _, node := range statusNodes {
			if node.Status.NeedSkip {
				continue
			}
			if statusNodes.GetNodeStateFlag(node.Spec.IP, bkev1beta1.NodeFailedFlag) {
				continue
			}
			return true
		}
	default:
		return false
	}
	return false
}

// Version returns the current running containerd version.
func (e *EnsureContainerdUpgrade) Version() string {
	if e.Ctx == nil || e.Ctx.BKECluster == nil {
		return ""
	}
	return e.Ctx.BKECluster.Status.ContainerdVersion
}

// NeedExecute 这个阶段，只有在初始新建补丁版本时才需要执行，如何判断是初始新建补丁版本？old为空，new中openFuyao version带小版本
func (e *EnsureContainerdUpgrade) NeedExecute(old *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster) bool {
	if !e.BasePhase.DefaultNeedExecute(old, new) {
		return false
	}
	if !e.NeedExecuteWithVersionContext(upgrade.ComponentContainerd, old, new, e.isContainerdNeedUpgrade) {
		return false
	}
	e.SetStatus(bkev1beta1.PhaseWaiting)
	return true
}
