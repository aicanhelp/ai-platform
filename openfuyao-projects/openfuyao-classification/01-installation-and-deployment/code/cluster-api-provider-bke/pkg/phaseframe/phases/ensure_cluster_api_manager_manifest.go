/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"github.com/pkg/errors"
	ctrl "sigs.k8s.io/controller-runtime"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common"
	bkeaddon "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/addon"
	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/kube"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/condition"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
)

const (
	// EnsureClusterAPIManagerManifestName applies cluster-api manage manifests
	// (003-manage.yaml + 004-manage-bkecluster.yaml) after postprocess.
	// 003 (Namespace + BKENode) is applied first, then 004 (Secret + BKECluster),
	// so that BKENodes are created before BKECluster to avoid the master node
	// creation webhook from blocking the initial deployment.
	EnsureClusterAPIManagerManifestName confv1beta1.BKEClusterPhase = "EnsureClusterAPIManagerManifest"
)

type EnsureClusterAPIManagerManifest struct {
	phaseframe.BasePhase
}

func NewEnsureClusterAPIManagerManifest(ctx *phaseframe.PhaseContext) phaseframe.Phase {
	base := phaseframe.NewBasePhase(ctx, EnsureClusterAPIManagerManifestName)
	return &EnsureClusterAPIManagerManifest{BasePhase: base}
}

func (e *EnsureClusterAPIManagerManifest) NeedExecute(old *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster) bool {
	if !e.BasePhase.DefaultNeedExecute(old, new) {
		return false
	}

	// Only run when cluster-api addon exists.
	version, ok := findAddonVersion(new, "cluster-api")
	if !ok || strings.TrimSpace(version) == "" {
		return false
	}

	// Skip if already applied.
	if v, ok := annotation.HasAnnotation(new, common.ClusterAPIManagerAppliedAnnotationKey); ok && v == "true" {
		return false
	}

	// Defer until postprocess is completed.
	if !condition.HasConditionStatus(bkev1beta1.NodesPostProcessCondition, new, confv1beta1.ConditionTrue) {
		e.SetStatus(bkev1beta1.PhaseWaiting)
		return true
	}

	e.SetStatus(bkev1beta1.PhaseWaiting)
	return true
}

func (e *EnsureClusterAPIManagerManifest) Execute() (ctrl.Result, error) {
	phaseContextFields84 := e.Ctx.Untie()
	_, c, bkeCluster, _, log := phaseContextFields84.Context,
		phaseContextFields84.Client,
		phaseContextFields84.BKECluster,
		phaseContextFields84.Scheme,
		phaseContextFields84.Log

	// Ensure postprocess completed (double check).
	bkeNodes, err := e.Ctx.NodeFetcher().GetBKENodesWrapperForCluster(e.Ctx, bkeCluster)
	if err != nil {
		return ctrl.Result{}, err
	}
	if phaseutil.GetNeedPostProcessNodesWithBKENodes(bkeCluster, bkeNodes).Length() > 0 {
		condition.ConditionMark(bkeCluster, bkev1beta1.NodesPostProcessCondition, confv1beta1.ConditionFalse, constant.NodesPostProcessNotReadyReason, "")
		return ctrl.Result{RequeueAfter: 10 * time.Second}, errors.Errorf("postprocess not finished")
	}

	version, ok := findAddonVersion(bkeCluster, "cluster-api")
	if !ok {
		return ctrl.Result{}, nil
	}
	addon, ok := findAddon(bkeCluster, "cluster-api")
	if !ok {
		return ctrl.Result{}, nil
	}

	targetClusterClient, err := kube.NewRemoteClientByBKECluster(e.Ctx.Context, e.Ctx.Client, bkeCluster)
	if err != nil {
		return ctrl.Result{}, err
	}
	targetClusterClient.SetLogger(log.NormalLogger)
	targetClusterClient.SetBKELogger(log)

	impl, ok := targetClusterClient.(*kube.Client)
	if !ok {
		return ctrl.Result{}, errors.Errorf("unsupported remote client type %T", targetClusterClient)
	}

	if err := e.applyManageManifests(impl, bkeCluster, addon, version); err != nil {
		return ctrl.Result{}, err
	}

	annotation.SetAnnotation(bkeCluster, common.ClusterAPIManagerAppliedAnnotationKey, "true")
	log.Info(constant.AddonDeployedReason, "deferred cluster-api manage manifest applied for %q", utils.ClientObjNS(bkeCluster))
	if err := mergecluster.SyncStatusUntilComplete(c, bkeCluster); err != nil {
		return ctrl.Result{}, err
	}

	return ctrl.Result{}, nil
}

// applyManageManifests applies 003-manage.yaml and 004-manage-bkecluster.yaml in order.
// 003 (Namespace + BKENode) is applied first, then 004 (Secret + BKECluster),
// so that BKENodes are created before BKECluster to avoid the master node
// creation webhook from blocking the initial deployment.
func (e *EnsureClusterAPIManagerManifest) applyManageManifests(
	targetClusterClient *kube.Client,
	bkeCluster *bkev1beta1.BKECluster,
	addon *confv1beta1.Product,
	version string,
) error {
	phaseContextFields140 := e.Ctx.Untie()
	_, _, _, _, log := phaseContextFields140.Context,
		phaseContextFields140.Client,
		phaseContextFields140.BKECluster,
		phaseContextFields140.Scheme,
		phaseContextFields140.Log

	managerYamlPath := filepath.Join(constant.K8sManifestsDir, "cluster-api", version, "003-manage.yaml")
	managerClusterYamlPath := filepath.Join(constant.K8sManifestsDir, "cluster-api", version, "004-manage-bkecluster.yaml")
	config := bkeinit.BkeConfig(*bkeCluster.Spec.ClusterConfig)
	repo := config.ImageRepo()

	nodes, err := e.Ctx.GetNodes()
	if err != nil {
		return err
	}

	param, err := targetClusterClient.PrepareRenderParamForAddonFile(bkeCluster, addon, managerYamlPath, repo, nodes)
	if err != nil {
		return err
	}

	// Inherit bootstrap management cluster / BN names for self-manage CRs.
	param["clusterName"] = bkeCluster.Name
	param["namespace"] = bkeCluster.Namespace
	if fullNodes, err := e.Ctx.GetBKENodes(); err == nil {
		param["nodes"] = kube.ConvertBKENodesToManageTemplateData(fullNodes, bkeCluster.Name)
	}

	// Apply 003-manage.yaml (Namespace + BKENode) first.
	log.Info(constant.AddonDeployingReason, "apply deferred cluster-api manage manifest, file=%s", managerYamlPath)
	task003 := kube.NewTask("cluster-api-manage-nodes", managerYamlPath, param).
		AddRepo(repo).
		SetOperate(bkeaddon.CreateAddon).
		SetWaiter(true, bkeinit.DefaultAddonTimeout, bkeinit.DefaultAddonInterval)
	if err := targetClusterClient.ApplyYaml(task003); err != nil {
		return fmt.Errorf("apply deferred cluster-api manage nodes manifest failed: %w", err)
	}

	// Apply 004-manage-bkecluster.yaml (Secret + BKECluster) after BKENodes are created.
	log.Info(constant.AddonDeployingReason, "apply deferred cluster-api manage bkecluster manifest, file=%s", managerClusterYamlPath)
	task004 := kube.NewTask("cluster-api-manage-cluster", managerClusterYamlPath, param).
		AddRepo(repo).
		SetOperate(bkeaddon.CreateAddon).
		SetWaiter(true, bkeinit.DefaultAddonTimeout, bkeinit.DefaultAddonInterval)
	if err := targetClusterClient.ApplyYaml(task004); err != nil {
		return fmt.Errorf("apply deferred cluster-api manage bkecluster manifest failed: %w", err)
	}

	return nil
}

func findAddonVersion(bkeCluster *bkev1beta1.BKECluster, addonName string) (string, bool) {
	if bkeCluster == nil || bkeCluster.Spec.ClusterConfig == nil {
		return "", false
	}
	for _, a := range bkeCluster.Spec.ClusterConfig.Addons {
		if a.Name == addonName {
			return a.Version, true
		}
	}
	return "", false
}

func findAddon(bkeCluster *bkev1beta1.BKECluster, addonName string) (*confv1beta1.Product, bool) {
	if bkeCluster == nil || bkeCluster.Spec.ClusterConfig == nil {
		return nil, false
	}
	for i := range bkeCluster.Spec.ClusterConfig.Addons {
		a := &bkeCluster.Spec.ClusterConfig.Addons[i]
		if a.Name == addonName {
			return a, true
		}
	}
	return nil, false
}
