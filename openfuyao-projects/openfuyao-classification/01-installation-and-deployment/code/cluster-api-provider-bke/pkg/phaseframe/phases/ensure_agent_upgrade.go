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
	ctrl "sigs.k8s.io/controller-runtime"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil/agentssh"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
)

const (
	EnsureAgentUpgradeName confv1beta1.BKEClusterPhase = "EnsureAgentUpgrade"
	bkeagentAddonName                                  = "bkeagent"
	// legacyReleaseBKEAgentComponent matches older release.yaml upgrade.components[].name.
	legacyReleaseBKEAgentComponent = "bkeagent-upgrade"
)

// EnsureAgentUpgrade upgrades bkeagent on cluster nodes via Provider SSH push from HTTP binary repo.
type EnsureAgentUpgrade struct {
	phaseframe.BasePhase
}

func NewEnsureAgentUpgrade(ctx *phaseframe.PhaseContext) phaseframe.Phase {
	return &EnsureAgentUpgrade{BasePhase: phaseframe.NewBasePhase(ctx, EnsureAgentUpgradeName)}
}

func (e *EnsureAgentUpgrade) Version() string {
	if e.Ctx == nil || e.Ctx.BKECluster == nil {
		return ""
	}
	for _, addon := range e.Ctx.BKECluster.Status.AddonStatus {
		if addon.Name == bkeagentAddonName {
			return addon.Version
		}
	}
	return ""
}

func (e *EnsureAgentUpgrade) NeedExecute(old, new *bkev1beta1.BKECluster) bool {
	if !e.BasePhase.DefaultNeedExecute(old, new) {
		return false
	}
	if !e.NeedExecuteWithVersionContext(upgrade.ComponentBKEAgent, old, new, nil) {
		return false
	}
	e.SetStatus(bkev1beta1.PhaseWaiting)
	return true
}

func (e *EnsureAgentUpgrade) Execute() (ctrl.Result, error) {
	phaseContextFields71 := e.Ctx.Untie()
	_, _, _, _, log := phaseContextFields71.Context,
		phaseContextFields71.Client,
		phaseContextFields71.BKECluster,
		phaseContextFields71.Scheme,
		phaseContextFields71.Log
	if err := e.upgradeBKEAgentViaSSH(); err != nil {
		return ctrl.Result{}, fmt.Errorf("failed to upgrade bkeagent via ssh: %w", err)
	}
	log.Info("BKEAgentUpgradeSuccess", "bkeagent upgrade completed via ssh push")
	return ctrl.Result{}, nil
}

func (e *EnsureAgentUpgrade) agentTargetVersion() string {
	vc := e.GetVersionContext()
	if vc == nil {
		return ""
	}
	if v := vc.GetTarget(upgrade.ComponentBKEAgent); v != "" {
		return v
	}
	return vc.GetTarget(legacyReleaseBKEAgentComponent)
}

func (e *EnsureAgentUpgrade) upgradeBKEAgentViaSSH() error {
	phaseContextFields91 := e.Ctx.Untie()
	ctx, c, bkeCluster, scheme, log := phaseContextFields91.Context,
		phaseContextFields91.Client,
		phaseContextFields91.BKECluster,
		phaseContextFields91.Scheme,
		phaseContextFields91.Log

	bkeNodes, err := e.Ctx.NodeFetcher().GetBKENodesWrapperForCluster(e.Ctx, bkeCluster)
	if err != nil {
		log.Warn("BKEAgentUpgrade", "failed to get BKENodes: %v", err)
		return nil
	}
	nodes := bkeNodes.ToNodes()
	if nodes.Length() == 0 {
		return errors.New("no cluster nodes found for bkeagent upgrade")
	}
	log.Info("BKEAgentUpgrade", "upgrading bkeagent on %d cluster node(s)", nodes.Length())

	targetVersion := e.agentTargetVersion()
	params := agentssh.ParamsFromCluster(bkeCluster, targetVersion)
	log.Info("BKEAgentUpgrade", "ssh upgrade targetVersion=%q binaryArtifact=%s baseURL=%s",
		targetVersion, params.BinaryArtifact, params.BaseURL)

	archSet, discoverErrs, err := agentssh.DiscoverArchs(ctx, nodes, log.NormalLogger)
	if err != nil {
		return err
	}
	if len(discoverErrs) > 0 {
		if failErr := e.upgradeFailure(nodes, discoverErrs, len(nodes)-len(discoverErrs), nodes.Length()); failErr != nil {
			return failErr
		}
	}

	staging, err := agentssh.PrepareStaging(bkeCluster, params, agentssh.ArchsFromMap(archSet))
	if err != nil {
		return errors.Wrap(err, "prepare upgrade artifacts")
	}
	defer staging.Cleanup()

	pushErrs, err := agentssh.SSHUpgrade(ctx, nodes, staging, log.NormalLogger)
	if err != nil {
		return err
	}

	successCount := nodes.Length() - len(pushErrs)
	if len(pushErrs) > 0 {
		return e.upgradeFailure(nodes, pushErrs, successCount, nodes.Length())
	}

	log.Info("BKEAgentUpgrade", "ssh push succeeded on %d nodes, verifying agent health via ping", successCount)
	if err, _, failedNodes := phaseutil.PingBKEAgentOnNodes(ctx, c, scheme, bkeCluster, nodes); err != nil || len(failedNodes) > 0 {
		msg := fmt.Sprintf("bkeagent ping after upgrade failed: %v, failed nodes: %v", err, failedNodes)
		return errors.New(msg)
	}
	return nil
}

func (e *EnsureAgentUpgrade) upgradeFailure(
	nodes bkenode.Nodes,
	failures map[string]error,
	successCount, total int,
) error {
	failedIPs := make([]string, 0, len(failures))
	for ip := range failures {
		failedIPs = append(failedIPs, ip)
	}

	for _, master := range nodes.Master() {
		if utils.ContainsString(failedIPs, master.IP) {
			return fmt.Errorf("bkeagent ssh upgrade failed on master node(s) %v: %d/%d succeeded",
				failedIPs, successCount, total)
		}
	}

	return fmt.Errorf("bkeagent ssh upgrade failed: %d/%d nodes succeeded, failed: %v",
		successCount, total, failedIPs)
}
