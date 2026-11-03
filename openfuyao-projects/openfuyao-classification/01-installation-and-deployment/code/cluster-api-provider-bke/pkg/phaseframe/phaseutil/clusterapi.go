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

package phaseutil

import (
	"context"
	"fmt"

	"github.com/pkg/errors"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	clusterv1beta1 "sigs.k8s.io/cluster-api/api/v1beta1"
	bootstrapv1 "sigs.k8s.io/cluster-api/bootstrap/kubeadm/api/v1beta1"
	controlv1beta1 "sigs.k8s.io/cluster-api/controlplane/kubeadm/api/v1beta1"
	"sigs.k8s.io/cluster-api/util"
	"sigs.k8s.io/cluster-api/util/conditions"
	"sigs.k8s.io/controller-runtime/pkg/client"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/clusterutil"
)

type ClusterAPIObjs struct {
	Cluster             *clusterv1beta1.Cluster
	KubeadmControlPlane *controlv1beta1.KubeadmControlPlane
	MachineDeployment   *clusterv1beta1.MachineDeployment
	Machines            []*clusterv1beta1.Machine
}

type MachineAndNode struct {
	Machine *clusterv1beta1.Machine
	Node    confv1beta1.Node
}

// GetClusterAPIClusterByBKECluster get Cluster obj by BKECluster
func GetClusterAPIClusterByBKECluster(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) (*clusterv1beta1.Cluster, error) {
	// get Cluster obj
	cluster, err := util.GetOwnerCluster(ctx, c, bkeCluster.ObjectMeta)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to get owner cluster")
	}
	if cluster == nil {
		return nil, errors.New("Waiting for owner cluster to be initialized")
	}
	return cluster, nil
}

// GetClusterAPIKubeadmControlPlane get KubeadmControlPlane obj
func GetClusterAPIKubeadmControlPlane(ctx context.Context, c client.Client, cluster *clusterv1beta1.Cluster) (*controlv1beta1.KubeadmControlPlane, error) {
	if cluster == nil {
		return nil, errors.New("cluster is nil")
	}

	cp := &controlv1beta1.KubeadmControlPlane{}
	key := client.ObjectKey{
		Namespace: cluster.Spec.ControlPlaneRef.Namespace,
		Name:      cluster.Spec.ControlPlaneRef.Name,
	}
	if err := c.Get(ctx, key, cp); err != nil {
		return nil, err
	}
	return cp, nil
}

// GetClusterAPIMachineDeployment get MachineDeployment obj
func GetClusterAPIMachineDeployment(ctx context.Context, c client.Client, cluster *clusterv1beta1.Cluster) (*clusterv1beta1.MachineDeployment, error) {
	if cluster == nil {
		return nil, errors.New("client is nil")
	}
	md := &clusterv1beta1.MachineDeployment{}
	mdList := &clusterv1beta1.MachineDeploymentList{}
	err := c.List(ctx, mdList, client.InNamespace(cluster.Namespace), client.MatchingLabels{
		clusterv1beta1.ClusterNameLabel: cluster.Name,
	})
	if err != nil {
		return nil, err
	}
	if len(mdList.Items) == 0 {
		return nil, nil
	}
	for _, mdItem := range mdList.Items {
		for _, ref := range mdItem.GetOwnerReferences() {
			if ref.Kind == cluster.Kind && ref.Name == cluster.Name && ref.UID == cluster.UID {
				md = &mdItem
				break
			}
		}
	}

	if md == nil {
		return nil, nil
	}
	return md, nil
}

// GetClusterAPIAssociateObjs get cluster-api associate objs
func GetClusterAPIAssociateObjs(ctx context.Context, c client.Client, cluster *clusterv1beta1.Cluster) (*ClusterAPIObjs, error) {
	if cluster == nil {
		return nil, errors.New("cluster is nil")
	}
	kcp, err := GetClusterAPIKubeadmControlPlane(ctx, c, cluster)
	if err != nil || kcp == nil {
		return nil, fmt.Errorf("get kubeadm control plane failed. err: %w", err)
	}

	md, err := GetClusterAPIMachineDeployment(ctx, c, cluster)
	if err != nil || md == nil {
		return nil, fmt.Errorf("get machine deployment failed. err: %w", err)
	}

	return &ClusterAPIObjs{
		MachineDeployment:   md,
		KubeadmControlPlane: kcp,
		Cluster:             cluster,
	}, nil
}

// PauseClusterAPIObj add pause Annotations to cluster api obj
func PauseClusterAPIObj(ctx context.Context, c client.Client, obj client.Object, extraAnnotations ...string) error {
	err := RetryOnConflict(func() error {
		if err := c.Get(ctx, client.ObjectKeyFromObject(obj), obj); err != nil {
			return err
		}
		annotations := obj.GetAnnotations()
		if annotations == nil {
			annotations = map[string]string{}
		}
		annotations[clusterv1beta1.PausedAnnotation] = ""
		for _, a := range extraAnnotations {
			annotations[a] = ""
		}
		obj.SetAnnotations(annotations)
		return c.Update(ctx, obj)
	})
	if err != nil {
		return errors.Wrapf(err, "retry pause cluster api obj failed, obj: %s/%s/%s", obj.GetObjectKind().GroupVersionKind().Kind, obj.GetNamespace(), obj.GetName())
	}
	return nil
}

// ResumeClusterAPIObj remove pause Annotations from cluster api obj
func ResumeClusterAPIObj(ctx context.Context, c client.Client, obj client.Object, extraAnnotations ...string) error {
	err := RetryOnConflict(func() error {
		if err := c.Get(ctx, client.ObjectKeyFromObject(obj), obj); err != nil {
			return err
		}
		annotations := obj.GetAnnotations()
		if annotations == nil {
			annotations = map[string]string{}
		}
		delete(annotations, clusterv1beta1.PausedAnnotation)
		for _, a := range extraAnnotations {
			delete(annotations, a)
		}
		obj.SetAnnotations(annotations)
		return c.Update(ctx, obj)
	})
	if err != nil {
		return errors.Wrapf(err, "retry resume cluster api obj failed, obj: %s/%s/%s", obj.GetObjectKind().GroupVersionKind().Kind, obj.GetNamespace(), obj.GetName())
	}
	return nil
}

// UpdateMachineDeploymentReplicas updates the replicas of a MachineDeployment with conflict retry.
func UpdateMachineDeploymentReplicas(ctx context.Context, c client.Client, md *clusterv1beta1.MachineDeployment, replicas int32) error {
	err := RetryOnConflict(func() error {
		if err := c.Get(ctx, client.ObjectKeyFromObject(md), md); err != nil {
			return err
		}
		md.Spec.Replicas = &replicas
		return c.Update(ctx, md)
	})
	if err != nil {
		return errors.Wrapf(err, "retry update MachineDeployment replicas failed, obj: %s/%s", md.GetNamespace(), md.GetName())
	}
	return nil
}

// UpdateKubeadmControlPlaneReplicas updates the replicas of a KubeadmControlPlane with conflict retry.
func UpdateKubeadmControlPlaneReplicas(ctx context.Context, c client.Client, kcp *controlv1beta1.KubeadmControlPlane, replicas int32) error {
	err := RetryOnConflict(func() error {
		if err := c.Get(ctx, client.ObjectKeyFromObject(kcp), kcp); err != nil {
			return err
		}
		kcp.Spec.Replicas = &replicas
		return c.Update(ctx, kcp)
	})
	if err != nil {
		return errors.Wrapf(err, "retry update KubeadmControlPlane replicas failed, obj: %s/%s", kcp.GetNamespace(), kcp.GetName())
	}
	return nil
}

// ResumeAndUpdateMachineDeploymentReplicas resumes (removes paused annotation) and updates
// the replicas of a MachineDeployment in a single retry loop, avoiding a separate API round-trip.
func ResumeAndUpdateMachineDeploymentReplicas(ctx context.Context, c client.Client, md *clusterv1beta1.MachineDeployment, replicas int32) error {
	err := RetryOnConflict(func() error {
		if err := c.Get(ctx, client.ObjectKeyFromObject(md), md); err != nil {
			return err
		}
		annotations := md.GetAnnotations()
		if annotations == nil {
			annotations = map[string]string{}
		}
		delete(annotations, clusterv1beta1.PausedAnnotation)
		md.SetAnnotations(annotations)
		md.Spec.Replicas = &replicas
		return c.Update(ctx, md)
	})
	if err != nil {
		return errors.Wrapf(err, "retry resume and update MachineDeployment replicas failed, obj: %s/%s", md.GetNamespace(), md.GetName())
	}
	return nil
}

// ResumeAndUpdateKubeadmControlPlaneReplicas resumes (removes paused annotation) and updates
// the replicas of a KubeadmControlPlane in a single retry loop, avoiding a separate API round-trip.
func ResumeAndUpdateKubeadmControlPlaneReplicas(ctx context.Context, c client.Client, kcp *controlv1beta1.KubeadmControlPlane, replicas int32) error {
	err := RetryOnConflict(func() error {
		if err := c.Get(ctx, client.ObjectKeyFromObject(kcp), kcp); err != nil {
			return err
		}
		annotations := kcp.GetAnnotations()
		if annotations == nil {
			annotations = map[string]string{}
		}
		delete(annotations, clusterv1beta1.PausedAnnotation)
		kcp.SetAnnotations(annotations)
		kcp.Spec.Replicas = &replicas
		return c.Update(ctx, kcp)
	})
	if err != nil {
		return errors.Wrapf(err, "retry resume and update KubeadmControlPlane replicas failed, obj: %s/%s", kcp.GetNamespace(), kcp.GetName())
	}
	return nil
}

// GetMachineAssociateBKEMachine get bke machine by machine
func GetMachineAssociateBKEMachine(ctx context.Context, c client.Client, machine *clusterv1beta1.Machine) (*bkev1beta1.BKEMachine, error) {
	bkeMachineRef := machine.Spec.InfrastructureRef

	bkeMachine := &bkev1beta1.BKEMachine{}
	key := client.ObjectKey{
		Namespace: bkeMachineRef.Namespace,
		Name:      bkeMachineRef.Name,
	}
	if err := c.Get(ctx, key, bkeMachine); err != nil {
		return nil, err
	}
	return bkeMachine, nil
}

// NodeToMachine get machine by node
func NodeToMachine(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, node confv1beta1.Node) (*clusterv1beta1.Machine, error) {
	machineLi := &clusterv1beta1.MachineList{}
	providerID := GenerateProviderID(bkeCluster, node)

	filters := GetListFiltersByBKECluster(bkeCluster)

	if err := c.List(ctx, machineLi, filters...); err != nil {
		if apierrors.IsNotFound(err) {
			return nil, err
		}
		return nil, errors.Wrapf(err, "list machine failed, providerID: %s", providerID)
	}
	if len(machineLi.Items) == 0 {
		return nil, errors.Errorf("machine not found, providerID: %s", providerID)
	}
	for _, m := range machineLi.Items {
		machineProviderID := m.Spec.ProviderID
		if clusterutil.IsBocloudCluster(bkeCluster) {
			if v, ok := annotation.HasAnnotation(&m, annotation.BKEMachineProviderIDAnnotationKey); ok && v != "" {
				machineProviderID = &v
			}
		}
		if machineProviderID != nil && *machineProviderID == providerID {
			return &m, nil
		}
	}
	return nil, errors.Errorf("machine not found, providerID: %s", providerID)
}

// MarkMachineForDeletion set delete Annotations for machine
func MarkMachineForDeletion(ctx context.Context, c client.Client, machine *clusterv1beta1.Machine) error {
	return RetryOnConflict(func() error {
		latestMachine := machine.DeepCopy()
		if err := c.Get(ctx, client.ObjectKeyFromObject(machine), latestMachine); err != nil {
			return err
		}
		annotations := latestMachine.GetAnnotations()
		if annotations == nil {
			annotations = map[string]string{}
		}
		annotations[clusterv1beta1.DeleteMachineAnnotation] = ""
		latestMachine.SetAnnotations(annotations)
		return c.Update(ctx, latestMachine)
	})
}

// GetMachineAssociateKubeadmConfig fetches the kubeadm config for given machine
func GetMachineAssociateKubeadmConfig(ctx context.Context, cl client.Client, m *clusterv1beta1.Machine) (*bootstrapv1.KubeadmConfig, error) {
	bootstrapRef := m.Spec.Bootstrap.ConfigRef
	if bootstrapRef == nil {
		return nil, errors.Errorf("failed to retrieve bootstrap config for machine %q: missing configRef", m.Name)
	}
	machineConfig := &bootstrapv1.KubeadmConfig{}
	if err := cl.Get(ctx, client.ObjectKey{Name: bootstrapRef.Name, Namespace: bootstrapRef.Namespace}, machineConfig); err != nil {
		if apierrors.IsNotFound(errors.Cause(err)) {
			return nil, nil
		}
		return nil, errors.Wrapf(err, "failed to retrieve bootstrap config for machine %q", m.Name)
	}
	return machineConfig, nil
}

// ClusterEndDeployed check cluster is end deployed
// Deprecated: In controller context, use ClusterEndDeployedWithContext instead.
func ClusterEndDeployed(ctx context.Context, c client.Client, cluster *clusterv1beta1.Cluster, bkeCluster *bkev1beta1.BKECluster) bool {
	if cluster == nil {
		return false
	}
	bkeMachines, err := GetBKEClusterAssociateBKEMachines(ctx, c, bkeCluster)
	if err != nil {
		return false
	}

	bkenodes := GetBKENodesFromCluster(bkeCluster)
	return ClusterEndDeployedWithBKENodes(bkeMachines, bkenodes)
}

// ClusterEndDeployedWithContext check cluster is end deployed using NodeFetcher to get BKENodes.
// Use this in controller context where local kubeconfig is not available.
func ClusterEndDeployedWithContext(ctx context.Context, c client.Client, cluster *clusterv1beta1.Cluster, bkeCluster *bkev1beta1.BKECluster, bkeNodes bkev1beta1.BKENodes) bool {
	if cluster == nil {
		return false
	}
	bkeMachines, err := GetBKEClusterAssociateBKEMachines(ctx, c, bkeCluster)
	if err != nil {
		return false
	}

	return ClusterEndDeployedWithBKENodes(bkeMachines, bkeNodes)
}

// ClusterEndDeployedWithBKENodes check cluster is end deployed using pre-fetched BKENodes.
// Use this in controller context where local kubeconfig is not available.
func ClusterEndDeployedWithBKENodes(bkeMachines []bkev1beta1.BKEMachine, bkenodes bkev1beta1.BKENodes) bool {
	// 计算 needSkip 节点数
	skipNodeCount := 0
	for _, bkenode := range bkenodes {
		if bkenode.Status.NeedSkip {
			skipNodeCount++
		}
	}

	// 期望引导的节点数 = 总数 - needSkip 数
	totalNodes := len(bkenodes)
	expectBootNodeNum := totalNodes - skipNodeCount
	failedBootNodeNum, successBootNodeNum := CalculateBKEMachineBootNum(bkeMachines)

	// 如果成功+失败的数量达到期望数量，说明所有非 needSkip 节点都完成了引导
	if successBootNodeNum+failedBootNodeNum >= expectBootNodeNum {
		return true
	}
	return false
}

// IsNodeBootFlagSet checks if any worker or master-worker node has the boot flag set
// Deprecated: Use IsNodeBootFlagSetWithBKENodes in controller context
func IsNodeBootFlagSet(bkeCluster *bkev1beta1.BKECluster) bool {
	// 集群中有一主一从点了就允许部署addon了
	bkenodes := GetBKENodesFromNodesStatus(bkeCluster)
	return IsNodeBootFlagSetWithBKENodes(bkenodes)
}

// IsNodeBootFlagSetWithBKENodes checks if any worker or master-worker node has the boot flag set.
// Use this function in controller context where BKENodes are fetched via NodeFetcher.
func IsNodeBootFlagSetWithBKENodes(bkenodes bkev1beta1.BKENodes) bool {
	statusWorkerNodes := bkenodes.ToNodes().Worker()
	for _, statusWorkerNode := range statusWorkerNodes {
		if bkenodes.GetNodeStateFlag(statusWorkerNode.IP, bkev1beta1.NodeBootFlag) {
			return true
		}
	}

	statusMasterWorkerNodes := bkenodes.ToNodes().MasterWorker()
	for _, statusMasterWorkerNode := range statusMasterWorkerNodes {
		if bkenodes.GetNodeStateFlag(statusMasterWorkerNode.IP, bkev1beta1.NodeBootFlag) {
			return true
		}
	}

	return false
}

func ClusterAllowTracker(bkeCluster *bkev1beta1.BKECluster, cluster *clusterv1beta1.Cluster) bool {
	// 如果master节点未被初始化，不允许部署addon
	if !conditions.IsTrue(cluster, clusterv1beta1.ControlPlaneInitializedCondition) {
		return false
	}

	return IsNodeBootFlagSet(bkeCluster)
}

// ClusterAllowTrackerWithBKENodes checks if cluster tracking is allowed using pre-fetched BKENodes.
// Use this function in controller context where BKENodes are fetched via NodeFetcher.
func ClusterAllowTrackerWithBKENodes(bkenodes bkev1beta1.BKENodes, cluster *clusterv1beta1.Cluster) bool {
	// 如果master节点未被初始化，不允许部署addon
	if !conditions.IsTrue(cluster, clusterv1beta1.ControlPlaneInitializedCondition) {
		return false
	}

	return IsNodeBootFlagSetWithBKENodes(bkenodes)
}
