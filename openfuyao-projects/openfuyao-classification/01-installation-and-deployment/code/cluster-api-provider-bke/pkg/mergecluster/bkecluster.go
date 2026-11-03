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

package mergecluster

import (
	"context"
	"time"

	jsonpatch "github.com/evanphx/json-patch"
	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/rand"
	"sigs.k8s.io/cluster-api/util/patch"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/statusmanage"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/condition"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/config"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/patchutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	// SyncStatusTimeout represents the timeout duration for sync status operations
	SyncStatusTimeout = 2 * time.Minute
	// MaxSleepSeconds represents the maximum number of seconds for random sleep duration
	MaxSleepSeconds = 2
)

type bkeNodes struct {
	spec bkenode.Nodes
}

// SyncStatusUntilComplete sync bkecluster Status until complete
func SyncStatusUntilComplete(c client.Client, bkeCluster *v1beta1.BKECluster, patchs ...PatchFunc) (err error) {
	log := log.With("name", "syncer")
	ctx, cancel := context.WithTimeout(context.Background(), SyncStatusTimeout)
	defer cancel()
	for {
		select {
		case <-ctx.Done():
			return errors.New("The update failed to complete after 2 minutes. ")
		default:
		}
		// Execute concurrent tasks at different peaks.
		// When the number of concurrent tasks is greater than 100, a random value of 1-15 is preferred
		err = UpdateCombinedBKECluster(ctx, c, bkeCluster, []string{}, patchs...)
		if err != nil {
			if apierrors.IsNotFound(err) {
				log.Warnf("bkeCluster %q not found, skip update, error(ignore): %v", utils.ClientObjNS(bkeCluster), err)
				break
			}
			if apierrors.IsConflict(err) {
				log.Warnf("Update bkeCluster %q failed error: %v", utils.ClientObjNS(bkeCluster), err)
				continue
			}
			if apierrors.IsForbidden(err) || apierrors.IsBadRequest(err) || apierrors.IsInvalid(err) {
				return err
			}

			log.Warnf("Update bkeCluster %q failed error: %v", utils.ClientObjNS(bkeCluster), err)
			continue
		}
		time.Sleep(time.Duration(rand.IntnRange(0, MaxSleepSeconds)) * time.Second)
		break
	}
	return nil
}

func (b *bkeNodes) toCMData() (map[string]string, error) {
	data := map[string]string{}
	if b.spec != nil {
		specByte, err := json.Marshal(b.spec)
		if err != nil {
			return nil, err
		}
		data["nodes"] = string(specByte)
	}
	// Note: Node status is now managed via BKENode CRDs, no longer stored in ConfigMap
	return data, nil
}

// 获取组合后的BKECluster
func GetCombinedBKECluster(ctx context.Context, c client.Client, namespace, name string) (*v1beta1.BKECluster, error) {
	bkeCluster := &v1beta1.BKECluster{}
	if err := c.Get(ctx, types.NamespacedName{Namespace: namespace, Name: name}, bkeCluster); err != nil {
		return nil, err
	}
	cm, err := GetCombinedBKEClusterCM(ctx, c, bkeCluster)
	if err != nil {
		return nil, err
	}

	return CombinedBKECluster(bkeCluster, cm)
}

type PatchFunc func(currentCombinedBkeCluster *v1beta1.BKECluster)

// PrepareClusterDataParams 包含 prepareClusterData 函数的参数
type PrepareClusterDataParams struct {
	Ctx             context.Context
	Client          client.Client
	CombinedCluster *v1beta1.BKECluster
	Patchs          []PatchFunc
}

// prepareClusterData prepares the current combined BKE cluster and applies patches
func prepareClusterData(params PrepareClusterDataParams) (*v1beta1.BKECluster, error) {
	currentCombinedBkeCluster, err := GetCombinedBKECluster(params.Ctx, params.Client, params.CombinedCluster.Namespace, params.CombinedCluster.Name)
	if err != nil {
		return nil, err
	}

	currentCombinedBkeCluster.Status.PhaseStatus =
		fixPhaseStatus(currentCombinedBkeCluster.Status.PhaseStatus)

	for _, p := range params.Patchs {
		p(currentCombinedBkeCluster)
		p(params.CombinedCluster)
	}

	// The in-memory combined cluster can lag behind the API during declarative DAG execution.
	// Always align declarative upgrade progress from the fresh GET before status is written back.
	PreserveDeclarativeUpgradeFromFresh(currentCombinedBkeCluster, params.CombinedCluster)

	return currentCombinedBkeCluster, nil
}

// PreserveDeclarativeUpgradeFromFresh copies DeclarativeUpgrade status from fresh into target.
func PreserveDeclarativeUpgradeFromFresh(fresh, target *v1beta1.BKECluster) {
	if fresh == nil || target == nil || fresh.Status.DeclarativeUpgrade == nil {
		return
	}
	target.Status.DeclarativeUpgrade = fresh.Status.DeclarativeUpgrade.DeepCopy()
}

// handleExternalUpdates handles external updates to the cluster
func handleExternalUpdates(combinedCluster *v1beta1.BKECluster, currentCombinedBkeCluster *v1beta1.BKECluster) error {
	// Check if there is manual update to currentCombinedBkeCluster
	//
	// Get the content of external updates
	patches, err := GetCurrentBkeClusterPatches(combinedCluster.DeepCopy(), currentCombinedBkeCluster.DeepCopy())
	if err != nil {
		return err
	}

	// If there are external updates, merge the content of external updates into combinedCluster
	if patches != nil {
		combinedByte, err := json.Marshal(combinedCluster)
		if err != nil {
			return err
		}
		apply, err := patches.Apply(combinedByte)
		if err != nil {
			return err
		}
		err = json.Unmarshal(apply, combinedCluster)
		if err != nil {
			return err
		}
	}

	return nil
}

// initializePatchHelper initializes the patch helper for the current BKE cluster
func initializePatchHelper(ctx context.Context, c client.Client, combinedCluster *v1beta1.BKECluster) (*v1beta1.BKECluster, *patch.Helper, error) {
	currentBkeCluster := &v1beta1.BKECluster{}
	if err := c.Get(ctx, types.NamespacedName{Namespace: combinedCluster.Namespace, Name: combinedCluster.Name}, currentBkeCluster); err != nil {
		if apierrors.IsNotFound(err) {
			return nil, nil, nil
		}
		return nil, nil, err
	}

	// Initialize the patch helper
	patchHelper, err := patch.NewHelper(currentBkeCluster, c)
	if err != nil {
		return nil, nil, err
	}

	return currentBkeCluster, patchHelper, nil
}

// HandleInternalUpdateConditionParams 包含 handleInternalUpdateCondition 函数的参数
type HandleInternalUpdateConditionParams struct {
	Ctx               context.Context
	PatchHelper       *patch.Helper
	CurrentBkeCluster *v1beta1.BKECluster
	Patchs            []PatchFunc
}

// handleInternalUpdateCondition handles internal update conditions
func handleInternalUpdateCondition(ctx context.Context, patchHelper *patch.Helper, currentBkeCluster *v1beta1.BKECluster, patchs []PatchFunc) error {
	params := HandleInternalUpdateConditionParams{
		Ctx:               ctx,
		PatchHelper:       patchHelper,
		CurrentBkeCluster: currentBkeCluster,
		Patchs:            patchs,
	}
	return handleInternalUpdateConditionWithParams(params)
}

// handleInternalUpdateConditionWithParams 使用参数结构体处理内部更新条件
func handleInternalUpdateConditionWithParams(params HandleInternalUpdateConditionParams) error {
	if !config.EnableInternalUpdate {
		return nil
	}

	if len(params.Patchs) != 0 {
		return handlePatchesNotEmpty(params.Ctx, params.PatchHelper, params.CurrentBkeCluster)
	} else {
		return handlePatchesEmpty(params.Ctx, params.PatchHelper, params.CurrentBkeCluster)
	}
}

// handlePatchesNotEmpty 处理补丁不为空的情况
func handlePatchesNotEmpty(ctx context.Context, patchHelper *patch.Helper, currentBkeCluster *v1beta1.BKECluster) error {
	// Add a special annotation to prevent internal updates from triggering enqueue
	condition.ConditionMark(currentBkeCluster, v1beta1.InternalSpecChangeCondition, "", "", "")
	return patchHelper.Patch(ctx, currentBkeCluster)
}

// handlePatchesEmpty 处理补丁为空的情况
func handlePatchesEmpty(ctx context.Context, patchHelper *patch.Helper, currentBkeCluster *v1beta1.BKECluster) error {
	if _, ok := condition.HasCondition(v1beta1.InternalSpecChangeCondition, currentBkeCluster); ok {
		condition.RemoveCondition(v1beta1.InternalSpecChangeCondition, currentBkeCluster)
		return patchHelper.Patch(ctx, currentBkeCluster)
	}
	return nil
}

// ProcessNodeDataParams 包含 processNodeData 函数的参数
type ProcessNodeDataParams struct {
	Ctx               context.Context
	Client            client.Client
	CombinedCluster   *v1beta1.BKECluster
	CurrentBkeCluster *v1beta1.BKECluster
	DeleteNodes       []string
}

// ProcessNodeDataResult 包含 processNodeData 函数的执行结果
type ProcessNodeDataResult struct {
	CM                *corev1.ConfigMap
	NodesCM           *bkeNodes
	FinalClusterNodes *bkeNodes
	FinalCMNodes      *bkeNodes
}

// processNodeData processes node data from cluster and configmap
func processNodeData(params ProcessNodeDataParams) (ProcessNodeDataResult, error) {
	// step 2 Extract nodes and status from cm
	cm, nodesCM, err := getBkeClusterAssociateNodesCM(params.Ctx, params.Client, params.CombinedCluster)
	if err != nil {
		return ProcessNodeDataResult{}, err
	}

	// step 3 Extract spec.nodes and status.nodes from combinedCluster
	// nodesCombined is the final spec.nodes and status.nodes after reconciliation
	// When creating, it consists of nodes from combinedCluster's spec and status
	nodesCombined := newTmpBkeNodesCluster(params.CombinedCluster)

	// step 4 Extract spec.nodes and status.nodes from currentBkeCluster, not the final spec.nodes and status.nodes
	// nodesCluster is a temporary variable that will be updated to newBKECuster's spec.nodes and status.nodes,
	// When creating, it consists of nodes from currentBkeCluster's spec and status
	finalClusterNodes := newTmpBkeNodesCluster(&v1beta1.BKECluster{})
	finalCMNodes := newTmpBkeNodesCluster(&v1beta1.BKECluster{})

	// step 5 Remove spec.nodes from nodesCM in nodesCombined.spec.nodes,
	//        and add remaining nodes to nodesCluster.spec.nodes
	for _, node := range nodesCombined.spec {
		if utils.ContainsString(params.DeleteNodes, node.IP) {
			continue
		}
		if nodesCM.spec.Filter(bkenode.FilterOptions{"IP": node.IP}).Length() == 1 {
			finalCMNodes.spec = append(finalCMNodes.spec, node)
			continue
		}

		finalClusterNodes.spec = append(finalClusterNodes.spec, node)
	}
	// step 6 Node status processing is now deprecated
	// Node status is now managed via BKENode CRDs, no longer in BKECluster or ConfigMap

	return ProcessNodeDataResult{
		CM:                cm,
		NodesCM:           nodesCM,
		FinalClusterNodes: finalClusterNodes,
		FinalCMNodes:      finalCMNodes,
	}, nil
}

// UpdateCombinedBKEClusterParams 包含 UpdateCombinedBKECluster 函数的参数
type UpdateCombinedBKEClusterParams struct {
	Ctx             context.Context
	Client          client.Client
	CombinedCluster *v1beta1.BKECluster
	DeleteNodes     []string
	Patchs          []PatchFunc
}

// UpdateCombinedBKECluster updates bkecluster and configmap
// ps: This function is important, run unit tests after each modification to check for issues
func UpdateCombinedBKECluster(ctx context.Context, c client.Client, combinedCluster *v1beta1.BKECluster, deleteNodes []string, patchs ...PatchFunc) error {
	params := UpdateCombinedBKEClusterParams{
		Ctx:             ctx,
		Client:          c,
		CombinedCluster: combinedCluster,
		DeleteNodes:     deleteNodes,
		Patchs:          patchs,
	}
	return updateCombinedBKEClusterWithParams(params)
}

// updateCombinedBKEClusterWithParams 使用参数结构体更新BKECluster
func updateCombinedBKEClusterWithParams(params UpdateCombinedBKEClusterParams) error {
	// Prepare cluster data
	prepareParams := PrepareClusterDataParams{
		Ctx:             params.Ctx,
		Client:          params.Client,
		CombinedCluster: params.CombinedCluster,
		Patchs:          params.Patchs,
	}
	currentCombinedBkeCluster, err := prepareClusterData(prepareParams)
	if err != nil {
		return err
	}

	// Handle external updates
	if err := handleExternalUpdates(params.CombinedCluster, currentCombinedBkeCluster); err != nil {
		return err
	}

	// Initialize patch helper
	currentBkeCluster, patchHelper, err := initializePatchHelper(params.Ctx, params.Client, params.CombinedCluster)
	if err != nil {
		return err
	}
	// If currentBkeCluster is nil, it means it was not found, so we return
	if currentBkeCluster == nil {
		return nil
	}

	// Handle internal update condition
	if err := handleInternalUpdateCondition(params.Ctx, patchHelper, currentBkeCluster, params.Patchs); err != nil {
		return err
	}

	// Process node data
	processParams := ProcessNodeDataParams{
		Ctx:               params.Ctx,
		Client:            params.Client,
		CombinedCluster:   params.CombinedCluster,
		CurrentBkeCluster: currentBkeCluster,
		DeleteNodes:       params.DeleteNodes,
	}
	result, err := processNodeData(processParams)
	if err != nil {
		return err
	}
	cm := result.CM
	finalClusterNodes := result.FinalClusterNodes
	finalCMNodes := result.FinalCMNodes

	// Update cluster and configmap
	updateParams := UpdateClusterAndConfigMapParams{
		Ctx:               params.Ctx,
		Client:            params.Client,
		PatchHelper:       patchHelper,
		CombinedCluster:   params.CombinedCluster,
		CurrentBkeCluster: currentBkeCluster,
		FinalClusterNodes: finalClusterNodes,
		FinalCMNodes:      finalCMNodes,
		CM:                cm,
		Patchs:            params.Patchs,
	}
	return updateClusterAndConfigMapWithParams(updateParams)
}

// UpdateClusterAndConfigMapParams 包含 updateClusterAndConfigMap 函数的参数
type UpdateClusterAndConfigMapParams struct {
	Ctx               context.Context
	Client            client.Client
	PatchHelper       *patch.Helper
	CombinedCluster   *v1beta1.BKECluster
	CurrentBkeCluster *v1beta1.BKECluster
	FinalClusterNodes *bkeNodes
	FinalCMNodes      *bkeNodes
	CM                *corev1.ConfigMap
	Patchs            []PatchFunc
}

// updateClusterAndConfigMapWithParams 使用参数结构体更新集群和配置映射
func updateClusterAndConfigMapWithParams(params UpdateClusterAndConfigMapParams) error {
	// step 7 Add spec.nodes and status.nodes from nodesCluster to newBKECuster
	// Note: With BKENode CRD split, nodes are now stored in BKENode CRDs, not in BKECluster
	newBKECuster := newTmpBkeCluster(params.CombinedCluster, params.CurrentBkeCluster)

	// fix phaseStatus too large > 1.5M
	newBKECuster.Status.PhaseStatus = fixPhaseStatus(newBKECuster.Status.PhaseStatus)

	// Set this update to newBKECuster's annotation
	configByte, err := json.Marshal(cleanBkeCluster(newBKECuster))
	if err != nil {
		return err
	}
	annotation.SetAnnotation(newBKECuster, annotation.LastUpdateConfigurationAnnotationKey, string(configByte))

	bkeNodes, err := getBKENodesForCluster(params.Ctx, params.Client, newBKECuster)
	if err != nil {
		bkeNodes = v1beta1.BKENodes{}
	}
	statusmanage.DefaultStatusManager().SetStatus(newBKECuster, bkeNodes)

	if err := updateModifiedBKENodes(params.Ctx, params.Client, bkeNodes); err != nil {
		// Log warning but don't fail - node status update is not critical for cluster operation
		log.With("name", "syncer").Warnf("Failed to update modified BKENodes: %v", err)
	}

	params.CombinedCluster.Status.ClusterStatus = newBKECuster.Status.ClusterStatus
	phaseutil.SyncStatusFields(params.CombinedCluster)
	params.CombinedCluster.Status.Conditions = newBKECuster.Status.Conditions

	if config.EnableInternalUpdate {
		defer func() {
			if len(params.Patchs) != 0 {
				condition.RemoveCondition(v1beta1.InternalSpecChangeCondition, newBKECuster)
				_ = params.PatchHelper.Patch(params.Ctx, newBKECuster)
			}
		}()
	}

	// step 8 Update newBKECuster
	if err = params.PatchHelper.Patch(params.Ctx, newBKECuster); err != nil {
		return err
	}

	// step 9 Update cm
	data, err := params.FinalCMNodes.toCMData()
	if err != nil {
		return err
	}
	params.CM.Data = data
	// Set this update to cm's annotation
	cmByte, err := json.Marshal(params.CM)
	if err != nil {
		return err
	}
	annotation.SetAnnotation(params.CM, annotation.LastUpdateConfigurationAnnotationKey, string(cmByte))

	if err = params.Client.Update(params.Ctx, params.CM, &client.UpdateOptions{}); err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	}

	return nil
}

func DeleteCombinedBKECluster(ctx context.Context, c client.Client, namespace, name string) error {
	err := c.Delete(ctx, &v1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
	}, &client.DeleteOptions{})
	if err != nil {
		return err
	}

	err = c.Delete(ctx, &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		}}, &client.DeleteOptions{})
	if err != nil {
		return err
	}
	return nil
}

func ListCombinedBKECluster(ctx context.Context, c client.Client) (*v1beta1.BKEClusterList, error) {
	clusterList := &v1beta1.BKEClusterList{}
	if err := c.List(ctx, clusterList); err != nil {
		return nil, err
	}

	newClusterList := &v1beta1.BKEClusterList{}
	for _, cl := range clusterList.Items {
		cm, err := GetCombinedBKEClusterCM(ctx, c, &cl)
		if err != nil {
			return nil, err
		}
		cl, err := CombinedBKECluster(&cl, cm)
		if err != nil {
			return nil, err
		}
		newClusterList.Items = append(newClusterList.Items, *cl)
	}
	return newClusterList, nil
}

// handleLastUpdateConfiguration handles the last update configuration annotation
func handleLastUpdateConfiguration(bkeCluster *v1beta1.BKECluster, cm *corev1.ConfigMap) error {
	lastBkeClusterConfig, lastBkeFlag := annotation.HasAnnotation(bkeCluster, annotation.LastUpdateConfigurationAnnotationKey)
	lastCMConfig, lastCMFlag := annotation.HasAnnotation(cm, annotation.LastUpdateConfigurationAnnotationKey)

	if lastBkeFlag && lastCMFlag {
		annotation.RemoveAnnotation(bkeCluster, annotation.LastUpdateConfigurationAnnotationKey)
		lastBkeCluster := &v1beta1.BKECluster{}
		lastCM := &corev1.ConfigMap{}

		if err := json.Unmarshal([]byte(lastBkeClusterConfig), lastBkeCluster); err != nil {
			return err
		}
		if err := json.Unmarshal([]byte(lastCMConfig), lastCM); err != nil {
			return err
		}
		lastAnnotation, err := CombinedBKECluster(lastBkeCluster, lastCM)
		if err != nil {
			return err
		}

		lastConfigByte, err := json.Marshal(lastAnnotation)
		if err != nil {
			return err
		}
		annotation.SetAnnotation(bkeCluster, annotation.LastUpdateConfigurationAnnotationKey, string(lastConfigByte))
	}
	return nil
}

// setDefaultCustomExtra sets default custom extra configuration
func setDefaultCustomExtra(bkeCluster *v1beta1.BKECluster) {
	if bkeCluster.Spec.ClusterConfig.CustomExtra == nil {
		bkeCluster.Spec.ClusterConfig.CustomExtra = map[string]string{}
	}
}

// CombinedBKECluster combines BKE cluster configuration from cluster and configmap
func CombinedBKECluster(bkeCluster *v1beta1.BKECluster, cm *corev1.ConfigMap) (*v1beta1.BKECluster, error) {
	// Handle last update configuration
	if err := handleLastUpdateConfiguration(bkeCluster, cm); err != nil {
		return nil, err
	}

	// Set default custom extra configuration
	setDefaultCustomExtra(bkeCluster)

	return bkeCluster, nil
}

// createDefaultConfigMap creates a new ConfigMap with default data
func createDefaultConfigMap(namespace, name string, bkeCluster *v1beta1.BKECluster, scheme *runtime.Scheme) (*corev1.ConfigMap, error) {
	newCm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
		Data: map[string]string{
			"nodes":  "[]",
			"status": "[]",
		},
	}
	// set owner reference
	if err := controllerutil.SetControllerReference(bkeCluster, newCm, scheme); err != nil {
		return nil, err
	}
	return newCm, nil
}

// HandleConfigMapNotFoundParams 包含 handleConfigMapNotFound 函数的参数
type HandleConfigMapNotFoundParams struct {
	Ctx        context.Context
	Client     client.Client
	Namespace  string
	Name       string
	BKECluster *v1beta1.BKECluster
}

// handleConfigMapNotFound handles the case when ConfigMap is not found
func handleConfigMapNotFound(ctx context.Context, c client.Client, namespace, name string, bkeCluster *v1beta1.BKECluster) (*corev1.ConfigMap, error) {
	params := HandleConfigMapNotFoundParams{
		Ctx:        ctx,
		Client:     c,
		Namespace:  namespace,
		Name:       name,
		BKECluster: bkeCluster,
	}
	return handleConfigMapNotFoundWithParams(params)
}

// handleConfigMapNotFoundWithParams 使用参数结构体处理ConfigMap未找到的情况
func handleConfigMapNotFoundWithParams(params HandleConfigMapNotFoundParams) (*corev1.ConfigMap, error) {
	scheme := params.Client.Scheme()
	newCm, err := createDefaultConfigMap(params.Namespace, params.Name, params.BKECluster, scheme)
	if err != nil {
		return nil, err
	}

	if err = params.Client.Create(params.Ctx, newCm, &client.CreateOptions{}); err != nil {
		if apierrors.HasStatusCause(err, corev1.NamespaceTerminatingCause) {
			return newBlankBKEConfigMap(params.BKECluster), nil
		}
		return nil, err
	}
	return newCm, nil
}

// ensureConfigMapData ensures the ConfigMap has default data if Data is nil
func ensureConfigMapData(cm *corev1.ConfigMap) {
	if cm.Data == nil {
		cm.Data = map[string]string{
			"nodes":  "[]",
			"status": "[]",
		}
	}
}

// GetCombinedBKEClusterCM gets or creates a combined BKE cluster ConfigMap
func GetCombinedBKEClusterCM(ctx context.Context, c client.Client, bkeCluster *v1beta1.BKECluster) (*corev1.ConfigMap, error) {
	namespace := bkeCluster.GetNamespace()
	name := bkeCluster.GetName()

	cm := &corev1.ConfigMap{}
	if err := c.Get(ctx, client.ObjectKey{Namespace: namespace, Name: name}, cm); err != nil {
		if apierrors.IsNotFound(err) {
			return handleConfigMapNotFound(ctx, c, namespace, name, bkeCluster)
		}
		return nil, err
	}

	ensureConfigMapData(cm)
	return cm, nil
}

func getBkeClusterAssociateNodesCM(ctx context.Context, c client.Client, bkeCluster *v1beta1.BKECluster) (*corev1.ConfigMap, *bkeNodes, error) {
	cm, err := GetCombinedBKEClusterCM(ctx, c, bkeCluster)
	if err != nil {
		return nil, nil, err
	}
	nodesCM := &bkeNodes{}
	if v, ok := cm.Data["nodes"]; ok {
		err := json.Unmarshal([]byte(v), &nodesCM.spec)
		if err != nil {
			return nil, nil, err
		}
	}
	// Note: Node status is now managed via BKENode CRDs, no longer stored in ConfigMap
	// The "status" key in ConfigMap is deprecated
	annotation.RemoveAnnotation(cm, annotation.LastUpdateConfigurationAnnotationKey)
	return cm, nodesCM, nil
}

func newTmpBkeCluster(combinedCluster *v1beta1.BKECluster, currentBkeCluster *v1beta1.BKECluster) *v1beta1.BKECluster {
	// 关于这段赋值的说明
	// newBKECuster 是最终要更新到集群内的bkecluster
	// cluster 是组合的bkecluster
	// bkeCluster 是集群中还未更新的bkecluster
	// 在这个方法中，不止要将node信息分别更新到cm和bkecluster中，还要将其他的字段更新到bkecluster中
	// 首先直接拷贝cluster，让除node(spec、status)字段之外的字段都是最新的
	newBKECuster := combinedCluster.DeepCopy()
	// 保留现有集群中bkeCluster的metadata里的UID、ResourceVersion、Generation、SelfLink、CreationTimestamp、
	// DeletionTimestamp、DeletionGracePeriodSeconds等字段。
	// 如果不使用集群中的bkeCluster的metadata，会导致更新失败
	// 错误信息: Operation cannot be fulfilled on bkeclusters.bke.bocloud.io "bkecluster": the object has been modified; please apply your changes to the latest version and try again
	newBKECuster.ObjectMeta = *currentBkeCluster.ObjectMeta.DeepCopy()
	// 使用最新的Labels、Annotations、OwnerReferences、Finalizers
	newBKECuster.Annotations = combinedCluster.Annotations
	annotation.RemoveAnnotation(newBKECuster, annotation.LastUpdateConfigurationAnnotationKey)
	newBKECuster.Labels = combinedCluster.Labels
	newBKECuster.OwnerReferences = combinedCluster.OwnerReferences
	newBKECuster.Finalizers = combinedCluster.Finalizers
	return newBKECuster
}

// newTmpBkeNodesCluster
// This function is kept for backwards compatibility but returns empty nodes
func newTmpBkeNodesCluster(bkeCluster *v1beta1.BKECluster) *bkeNodes {
	nodesCluster := &bkeNodes{}
	nodesCluster.spec = []confv1beta1.Node{}
	return nodesCluster
}

// GetCurrentBkeClusterPatches ignores status
func GetCurrentBkeClusterPatches(old, new *v1beta1.BKECluster) (jsonpatch.Patch, error) {
	// Remove last-update annotation
	annotation.RemoveAnnotation(old, annotation.LastUpdateConfigurationAnnotationKey)
	annotation.RemoveAnnotation(new, annotation.LastUpdateConfigurationAnnotationKey)

	// Only consider spec, labels, annotations, finalizers
	// Clear other fields
	old.ObjectMeta = metav1.ObjectMeta{
		Labels:          old.ObjectMeta.Labels,
		Annotations:     old.ObjectMeta.Annotations,
		ResourceVersion: old.ObjectMeta.ResourceVersion,
	}
	new.ObjectMeta = metav1.ObjectMeta{
		Labels:          new.ObjectMeta.Labels,
		Annotations:     new.ObjectMeta.Annotations,
		ResourceVersion: new.ObjectMeta.ResourceVersion,
	}
	// Ignore status
	old.Status = confv1beta1.BKEClusterStatus{}
	new.Status = confv1beta1.BKEClusterStatus{}

	return patchutil.Diff(old, new)
}

func cleanBkeCluster(bkeCluster *v1beta1.BKECluster) *v1beta1.BKECluster {
	cluster := &v1beta1.BKECluster{}
	cluster.Name = bkeCluster.Name
	cluster.Namespace = bkeCluster.Namespace
	cluster.Spec = *bkeCluster.Spec.DeepCopy()
	return cluster
}

func GetLastUpdatedBKECluster(bkeCluster *v1beta1.BKECluster) (*v1beta1.BKECluster, error) {
	oldBkeCluster := &v1beta1.BKECluster{}
	v, ok := annotation.HasAnnotation(bkeCluster, annotation.LastUpdateConfigurationAnnotationKey)
	if !ok || v == "" {
		return nil, nil
	} else {
		err := json.Unmarshal([]byte(v), oldBkeCluster)
		if err != nil {
			return nil, err
		}
	}
	return oldBkeCluster, nil
}

func newBlankBKEConfigMap(bkeCluster *v1beta1.BKECluster) *corev1.ConfigMap {
	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      bkeCluster.Name,
			Namespace: bkeCluster.Namespace,
		},
		Data: map[string]string{
			"nodes":  "[]",
			"status": "[]",
		},
	}
}

func fixPhaseStatus(ph confv1beta1.PhaseStatus) confv1beta1.PhaseStatus {
	if len(ph) == 0 {
		return ph
	}

	const maxEnsureClusterFailures = 3

	deduped := deduplicatePhaseStatus(ph)

	lastEnsureIndices := getLastFailedEnsureClusterIndices(deduped, maxEnsureClusterFailures)
	if len(lastEnsureIndices) == 0 {
		return deduped
	}

	return removeOldFailedEnsureCluster(deduped, lastEnsureIndices)
}

func deduplicatePhaseStatus(ph confv1beta1.PhaseStatus) confv1beta1.PhaseStatus {
	type key struct {
		Name    confv1beta1.BKEClusterPhase
		Status  confv1beta1.BKEClusterPhaseStatus
		Message string
	}

	lastIndex := make(map[key]int)
	for index, value := range ph {
		mapKey := key{
			Name:    value.Name,
			Status:  value.Status,
			Message: value.Message,
		}
		lastIndex[mapKey] = index
	}

	var deduped confv1beta1.PhaseStatus
	for index, value := range ph {
		mapKey := key{
			Name:    value.Name,
			Status:  value.Status,
			Message: value.Message,
		}
		if lastIndex[mapKey] == index {
			deduped = append(deduped, value)
		}
	}
	return deduped
}

func getLastFailedEnsureClusterIndices(ph confv1beta1.PhaseStatus, maxFailures int) []int {
	var indices []int
	for i := len(ph) - 1; i >= 0; i-- {
		if ph[i].Name == "EnsureCluster" && ph[i].Status == v1beta1.PhaseFailed {
			indices = append(indices, i)
			if len(indices) == maxFailures {
				break
			}
		}
	}
	return indices
}

func removeOldFailedEnsureCluster(ph confv1beta1.PhaseStatus, indices []int) confv1beta1.PhaseStatus {
	lastStart := indices[len(indices)-1]
	tail := ph[lastStart:]

	var cleaned confv1beta1.PhaseStatus
	for i := 0; i < lastStart && i < len(ph); i++ {
		if ph[i].Name == "EnsureCluster" && ph[i].Status == v1beta1.PhaseFailed {
			continue
		}
		cleaned = append(cleaned, ph[i])
	}

	return append(cleaned, tail...)
}

// getBKENodesForCluster fetches BKENodes associated with a BKECluster using controller-runtime client.
// This function is used in the controller context where we have access to the client with informer caching.
func getBKENodesForCluster(ctx context.Context, c client.Client, bkeCluster *v1beta1.BKECluster) (v1beta1.BKENodes, error) {
	bkeNodeList := &confv1beta1.BKENodeList{}

	if err := c.List(ctx, bkeNodeList,
		client.InNamespace(bkeCluster.Namespace),
		client.MatchingLabels{nodeutil.ClusterNameLabel: bkeCluster.Name},
	); err != nil {
		return nil, errors.Wrapf(err, "failed to list BKENodes for cluster %s/%s", bkeCluster.Namespace, bkeCluster.Name)
	}

	return v1beta1.NewBKENodesFromList(bkeNodeList), nil
}

// UpdateModifiedBKENodes updates BKENodes that have been modified by status management.
// It checks for nodes with the NodeStateNeedRecord flag set and updates their status in the API server.
func UpdateModifiedBKENodes(ctx context.Context, c client.Client, bkeNodes v1beta1.BKENodes) error {
	modifiedNodes := bkeNodes.GetModifiedNodes()
	if len(modifiedNodes) == 0 {
		return nil
	}

	for i := range modifiedNodes {
		node := &modifiedNodes[i]
		err := phaseutil.RetryOnConflict(func() error {
			latestBKENode := &confv1beta1.BKENode{}
			if err := c.Get(ctx, client.ObjectKey{Name: node.Name, Namespace: node.Namespace}, latestBKENode); err != nil {
				return err
			}
			// Copy in-memory modifications to the API-fetched object
			latestBKENode.Status.State = node.Status.State
			latestBKENode.Status.Message = node.Status.Message
			latestBKENode.Status.StateCode = node.Status.StateCode
			latestBKENode.Status.StateCode &= ^v1beta1.NodeStateNeedRecord
			return c.Status().Update(ctx, latestBKENode)
		})
		if err != nil {
			if apierrors.IsNotFound(err) {
				continue
			}
			return errors.Wrapf(err, "failed to update BKENode %s/%s status", node.Namespace, node.Name)
		}
	}

	bkeNodes.ClearRecordFlags()
	return nil
}

// updateModifiedBKENodes is an alias for UpdateModifiedBKENodes for backward compatibility
func updateModifiedBKENodes(ctx context.Context, c client.Client, bkeNodes v1beta1.BKENodes) error {
	return UpdateModifiedBKENodes(ctx, c, bkeNodes)
}
