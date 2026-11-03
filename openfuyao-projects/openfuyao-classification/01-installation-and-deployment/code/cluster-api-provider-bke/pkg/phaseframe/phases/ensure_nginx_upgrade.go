/******************************************************************
 * Copyright (c) 2026 China Mobile (Suzhou) Software Technology Co., Ltd
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
	"fmt"
	"strings"
	"time"

	"github.com/pkg/errors"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/command"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/clusterutil"
)

const (
	// EnsureNginxUpgradeName is the phase name for nginx-proxy version upgrades.
	EnsureNginxUpgradeName confv1beta1.BKEClusterPhase = "EnsureNginxUpgrade"
)

// nginxUpgradeWaitTimeout bounds how long Execute waits for the dispatched
// reconfigure commands to finish on all worker agents.
const nginxUpgradeWaitTimeout = 10 * time.Minute

// EnsureNginxUpgrade upgrades the nginx-proxy image on every worker node by
// dispatching Kubeadm phase=ReconfigureNginx commands through the declarative
// upgrade framework.
type EnsureNginxUpgrade struct {
	phaseframe.BasePhase
}

// NewEnsureNginxUpgrade constructs the EnsureNginxUpgrade phase.
func NewEnsureNginxUpgrade(ctx *phaseframe.PhaseContext) phaseframe.Phase {
	base := phaseframe.NewBasePhase(ctx, EnsureNginxUpgradeName)
	return &EnsureNginxUpgrade{BasePhase: base}
}

// NeedExecute checks whether nginx-proxy needs a version upgrade via VersionContext.
// It is restricted to nginx load-balance clusters: HA clusters run no nginx-proxy
// workload, so they must skip this phase even when the release catalog carries
// an nginx-proxy upgrade target.
func (e *EnsureNginxUpgrade) NeedExecute(old *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster) bool {
	if e.Ctx == nil || e.Ctx.BKECluster == nil ||
		clusterutil.LoadBalanceType(e.Ctx.BKECluster.Spec.ClusterConfig) != "nginx" {
		return false
	}
	return e.BasePhase.NeedExecuteWithVersionContext(upgrade.ComponentNginxProxy, old, new, nil)
}

// Execute dispatches a reconfigure command to all Ready worker nodes.
// The nginx plugin regenerates the static pod manifest with the new image tag
// (synced from VersionContext into BKEConfig.Cluster.NginxProxyVersion).
// Kubelet detects the manifest change and recreates the nginx-proxy pod
// with the new image. Status.NginxProxyVersion is only advanced after every
// eligible worker has been upgraded; until then the phase returns an error
// so it is retried and skipped workers are not stranded on the old image.
func (e *EnsureNginxUpgrade) Execute() (ctrl.Result, error) {
	phaseContextFields := e.Ctx.Untie()
	ctx := phaseContextFields.Context
	c := phaseContextFields.Client
	bkeCluster := phaseContextFields.BKECluster
	scheme := phaseContextFields.Scheme
	log := phaseContextFields.Log

	// Sync VersionContext targets into BKECluster spec so the agent
	// reads the new NginxProxyVersion from BKEConfig.
	if err := e.Ctx.SyncUpgradeTargetsToClusterSpec(); err != nil {
		return ctrl.Result{}, errors.Wrap(err, "sync nginx upgrade target to cluster spec")
	}

	targetVersion := e.resolveNginxUpgradeVersion()
	if targetVersion == "" {
		return ctrl.Result{}, errors.New("nginx proxy upgrade version not specified")
	}

	log.Info("Upgrading nginx-proxy",
		"current", bkeCluster.Status.NginxProxyVersion, "target", targetVersion)

	allNodes, err := e.Ctx.NodeFetcher().GetNodesForBKECluster(e.Ctx, bkeCluster)
	if err != nil {
		return ctrl.Result{}, errors.Wrap(err, "failed to get nodes for cluster")
	}
	eligibleNodes, targetNodes, notReadyNodes := e.selectUpgradeTargets(allNodes, bkeCluster)

	if len(eligibleNodes) == 0 {
		// nginx-proxy only runs on worker nodes; a cluster without eligible
		// workers has nothing to upgrade.
		log.Debug("no worker nodes for nginx upgrade, marking complete")
		return ctrl.Result{}, e.markNginxUpgraded(c, bkeCluster, targetVersion)
	}

	if len(targetNodes) == 0 {
		return ctrl.Result{}, fmt.Errorf("nginx upgrade blocked: all %d worker agent(s) not ready: %s",
			len(eligibleNodes), strings.Join(notReadyNodes, ","))
	}

	if err := e.dispatchUpgradeCommand(ctx, c, scheme, bkeCluster, targetNodes); err != nil {
		return ctrl.Result{}, err
	}

	if len(notReadyNodes) > 0 {
		// The ready workers were upgraded, but skipped ones keep the upgrade
		// open: return an error so this phase retries and picks them up once
		// their agent recovers (the dispatched command is idempotent, so
		// re-running it on already-upgraded workers is a no-op).
		log.Warn("nginx upgrade incomplete, some worker agents not ready",
			"notReady", notReadyNodes)
		return ctrl.Result{}, fmt.Errorf("nginx upgrade incomplete: %d/%d worker(s) not ready: %s",
			len(notReadyNodes), len(eligibleNodes), strings.Join(notReadyNodes, ","))
	}

	log.Info("nginx-proxy upgraded successfully",
		"version", targetVersion, "nodes", len(targetNodes))
	return ctrl.Result{}, e.markNginxUpgraded(c, bkeCluster, targetVersion)
}

// selectUpgradeTargets partitions worker nodes into eligible (not being
// deleted), agent-ready targets, and not-ready node names.
// Status.NginxProxyVersion may only be recorded once every eligible worker
// has been upgraded: once spec equals status, NeedExecute never fires again
// and any skipped worker would stay on the old image forever.
func (e *EnsureNginxUpgrade) selectUpgradeTargets(
	allNodes bkenode.Nodes, bkeCluster *bkev1beta1.BKECluster,
) (eligible, targets bkenode.Nodes, notReady []string) {
	nodeFetcher := e.Ctx.NodeFetcher()
	for _, node := range allNodes.Worker() {
		deleting, _ := nodeFetcher.GetNodeStateFlagForCluster(e.Ctx, bkeCluster, node.IP, bkev1beta1.NodeDeletingFlag)
		if deleting {
			continue
		}
		eligible = append(eligible, node)
		agentReady, _ := nodeFetcher.GetNodeStateFlagForCluster(e.Ctx, bkeCluster, node.IP, bkev1beta1.NodeAgentReadyFlag)
		if agentReady {
			targets = append(targets, node)
		} else {
			notReady = append(notReady, phaseutil.NodeInfo(node))
		}
	}
	return eligible, targets, notReady
}

// dispatchUpgradeCommand creates and waits for the reconfigure nginx command.
// The agent runs Kubeadm phase=ReconfigureNginx which calls the nginx plugin
// to regenerate nginx.conf + the static pod manifest with the new image tag.
func (e *EnsureNginxUpgrade) dispatchUpgradeCommand(
	ctx context.Context, c client.Client, scheme *runtime.Scheme,
	bkeCluster *bkev1beta1.BKECluster, targetNodes bkenode.Nodes,
) error {
	spec := command.GenerateDefaultCommandSpec()
	spec.Commands = []agentv1beta1.ExecCommand{{
		ID: "upgrade-nginx",
		Command: []string{
			"Kubeadm",
			fmt.Sprintf("phase=%s", utils.ReconfigureNginx),
			fmt.Sprintf("bkeConfig=%s:%s", bkeCluster.Namespace, bkeCluster.Name),
		},
		Type: agentv1beta1.CommandBuiltIn,
	}}

	customCmd := &command.Custom{
		BaseCommand: command.BaseCommand{
			Ctx:             ctx,
			Client:          c,
			Scheme:          scheme,
			OwnerObj:        bkeCluster,
			ClusterName:     bkeCluster.Name,
			NameSpace:       bkeCluster.Namespace,
			Unique:          true,
			RemoveAfterWait: true,
			WaitTimeout:     nginxUpgradeWaitTimeout,
		},
		Nodes:        targetNodes,
		CommandLabel: command.BKEClusterLabel,
		CommandName:  fmt.Sprintf("upgrade-nginx-%d", time.Now().Unix()),
		CommandSpec:  spec,
	}

	if err := customCmd.New(); err != nil {
		return errors.Wrap(err, "failed to create nginx upgrade command")
	}

	if err, _, failedNodes := customCmd.Wait(); err != nil {
		return fmt.Errorf("failed to upgrade nginx: %w", err)
	} else if len(failedNodes) > 0 {
		return fmt.Errorf("nginx upgrade failed on nodes: %s",
			strings.Join(failedNodes, ","))
	}
	return nil
}

// markNginxUpgraded records the deployed nginx-proxy version in the cluster
// status and syncs it until the phase completes.
func (e *EnsureNginxUpgrade) markNginxUpgraded(
	c client.Client, bkeCluster *bkev1beta1.BKECluster, version string,
) error {
	bkeCluster.Status.NginxProxyVersion = version
	return mergecluster.SyncStatusUntilComplete(c, bkeCluster)
}

// resolveNginxUpgradeVersion returns the target nginx-proxy version from VersionContext,
// falling back to BKECluster spec.
func (e *EnsureNginxUpgrade) resolveNginxUpgradeVersion() string {
	if e.Ctx == nil || e.Ctx.BKECluster == nil {
		return ""
	}
	if vc := e.Ctx.VersionContext; vc != nil {
		if v := vc.GetTarget(upgrade.ComponentNginxProxy); v != "" {
			return v
		}
	}
	if e.Ctx.BKECluster.Spec.ClusterConfig == nil {
		return ""
	}
	return e.Ctx.BKECluster.Spec.ClusterConfig.Cluster.NginxProxyVersion
}
