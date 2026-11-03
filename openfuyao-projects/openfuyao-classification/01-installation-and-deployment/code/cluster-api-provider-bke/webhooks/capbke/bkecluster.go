/******************************************************************
 * Copyright (c) 2024 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package capbke

import (
	"context"
	"fmt"
	"net"
	"strconv"
	"strings"

	"github.com/blang/semver"
	jsonpatch "github.com/evanphx/json-patch"
	"github.com/pkg/errors"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	kerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"sigs.k8s.io/cluster-api/util/version"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/webhook"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	bkevalidate "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/validation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/clusterutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

var (
	logger = log.With("name", "BKECluster Webhook")

	notAllowedPaths = [][]string{
		{"metadata", "annotations", "bke.bocloud.com/cluster-from"},
		{"spec", "controlPlaneEndpoint", "*"},
		{"spec", "clusterConfig", "cluster", "networking", "*"},
		{"spec", "clusterConfig", "cluster", "loadBalanceType"},
	}
	nonStandNotAllowedPaths = [][]string{
		{"metadata", "annotations", "bke.bocloud.com/cluster-from"},
		{"spec", "controlPlaneEndpoint", "*"},
	}
)

// ComponentVersionUpdateParams encapsulates parameters for component version update validation
type ComponentVersionUpdateParams struct {
	NewBKECluster *bkev1beta1.BKECluster
	OldBKECluster *bkev1beta1.BKECluster
	OldVersion    string
	NewVersion    string
	FieldName     string
	ComponentName string
	MinVersion    *semver.Version
	MaxVersion    *semver.Version
}

type BKECluster struct {
	Client      client.Client
	NodeFetcher *nodeutil.NodeFetcher
	// APIReader 用于绕过 informer 缓存直接从 API Server 读取资源。
	// 在 webhook Default() 阶段，BKENode 可能刚创建完 informer 缓存尚未同步，
	// 此时需要 APIReader 来确保能读取到最新的 BKENode 数据。
	APIReader client.Reader
}

func (webhook *BKECluster) SetupWebhookWithManager(mgr ctrl.Manager) error {
	return ctrl.NewWebhookManagedBy(mgr).
		For(&bkev1beta1.BKECluster{}).
		WithValidator(webhook).
		WithDefaulter(webhook).
		Complete()
}

//+kubebuilder:webhook:verbs=create;update,path=/mutate-bke-bocloud-com-v1beta1-bkecluster,mutating=true,failurePolicy=fail,sideEffects=None,groups=bke.bocloud.com,resources=bkeclusters,versions=v1beta1,name=mbkecluster.kb.io,admissionReviewVersions=v1
//+kubebuilder:webhook:verbs=create;update,path=/validate-bke-bocloud-com-v1beta1-bkecluster,mutating=false,failurePolicy=fail,sideEffects=None,groups=bke.bocloud.com,resources=bkeclusters,versions=v1beta1,name=vbkecluster.kb.io,admissionReviewVersions=v1

var _ webhook.CustomDefaulter = &BKECluster{}
var _ webhook.CustomValidator = &BKECluster{}

// Default implements webhook.Defaulter so a webhook will be registered for the type
func (webhook *BKECluster) Default(ctx context.Context, obj runtime.Object) error {
	in, ok := obj.(*bkev1beta1.BKECluster)
	if !ok {
		return apierrors.NewBadRequest(fmt.Sprintf("expected a ClusterClass but got a %T", obj))
	}
	if in.Spec.ClusterConfig == nil {
		in.Spec.ClusterConfig = &confv1beta1.BKEConfig{}
	}
	cfg := bkeinit.BkeConfig(*in.Spec.ClusterConfig)
	bkeinit.SetDefaultBKEConfig(&cfg)
	in.Spec.ClusterConfig = (*confv1beta1.BKEConfig)(&cfg)
	setBKEClusterDefaultConfig(ctx, webhook.Client, in, webhook.NodeFetcher)

	// 设置默认注释功能门
	annotation.SetBKEClusterDefaultAnnotation(in)

	// 从 BKENode CRD 获取节点信息设置默认集群入口
	nodes, err := webhook.getNodesForBKEClusterDirect(ctx, in)
	if err != nil {
		logger.Warnf("Failed to get nodes for BKECluster %s: %v, skipping endpoint default", in.Name, err)
		nodes = bkenode.Nodes{}
	}

	if nodes.Master().Length() != 0 && in.Spec.ControlPlaneEndpoint.IsZero() {
		endpointNode := nodes.Master()[0]
		if in.Spec.ControlPlaneEndpoint.Host == "" {
			in.Spec.ControlPlaneEndpoint.Host = endpointNode.IP
		}
		if in.Spec.ControlPlaneEndpoint.Port == 0 {
			in.Spec.ControlPlaneEndpoint.Port = bkeinit.DefaultAPIBindPort
		}
	}
	if in.Spec.ControlPlaneEndpoint.Host != "" && in.Spec.ControlPlaneEndpoint.Port == 0 {
		in.Spec.ControlPlaneEndpoint.Port = int32(bkeinit.DefaultLoadBalancerBindPort)
	}
	// 判断是否为 ha 集群 port是36443 并且 endpoint不在master节点上
	precheckPhase, ok := annotation.HasAnnotation(in, annotation.AtPrecheckPhaseAnnotationKey)
	// 如果有这个注释，且值为true，且暂停标志为true，则不进行webhook校验
	if ok && precheckPhase == "true" && in.Spec.Pause {
		if in.Spec.ControlPlaneEndpoint.IsValid() && nodes.Master().Length() != 0 {
			portflag := in.Spec.ControlPlaneEndpoint.Port == bkeinit.DefaultLoadBalancerBindPort
			// 不是36443端口且host不在master节点上 那说明需要更改host
			if !portflag && nodes.Filter(bkenode.FilterOptions{"IP": in.Spec.ControlPlaneEndpoint.Host}).Length() == 0 {
				in.Spec.ControlPlaneEndpoint.Host = nodes.Master()[0].IP
				in.Spec.ControlPlaneEndpoint.Port = bkeinit.DefaultAPIBindPort
			}
		}
	}

	//对于需要纳管的集群，不要设置集群版本默认值
	if !clusterutil.IsBKECluster(in) && !clusterutil.ClusterInfoHasCollected(in) {
		in.Spec.ClusterConfig.Cluster.KubernetesVersion = ""
	}

	return nil
}

// ValidateUpdate implements webhook.Validator so a webhook will be registered for the type
func (webhook *BKECluster) ValidateUpdate(ctx context.Context, oldObj, newObj runtime.Object) (admission.Warnings, error) {
	oldBKECluster, ok := oldObj.(*bkev1beta1.BKECluster)
	if !ok {
		return nil, apierrors.NewBadRequest(fmt.Sprintf("expected a BKECluster but got a %T", oldObj))
	}
	newBKECluster, ok := newObj.(*bkev1beta1.BKECluster)
	if !ok {
		return nil, apierrors.NewBadRequest(fmt.Sprintf("expected a BKECluster but got a %T", newObj))
	}

	preCheckPhase, ok := annotation.HasAnnotation(newBKECluster, annotation.AtPrecheckPhaseAnnotationKey)
	// 如果有这个注释，且值为true，且暂停标志为true，则不进行webhook校验
	if ok && preCheckPhase == "true" && newBKECluster.Spec.Pause {
		return nil, nil
	}

	// 集群在部署中时的一些校验
	if newBKECluster.Status.ClusterStatus == bkev1beta1.ClusterInitializing ||
		newBKECluster.Status.ClusterStatus == bkev1beta1.ClusterDeployingAddon {
		// 在部署addon时，来暂停bc 不允许
		if newBKECluster.Status.ClusterStatus == bkev1beta1.ClusterDeployingAddon && newBKECluster.Spec.Pause {
			return nil, &apierrors.StatusError{
				ErrStatus: metav1.Status{
					Status:  metav1.StatusFailure,
					Code:    123,
					Message: "The cluster is deploying addon and cannot set pause",
				},
			}
		}
	}

	if clusterutil.FullyControlled(newBKECluster) {
		return nil, webhook.validateStandBKECluster(ctx, newBKECluster, oldBKECluster)
	}
	return nil, webhook.validateNonStandBKECluster(newBKECluster, oldBKECluster)
}

// ValidateCreate implements webhook.Validator so a webhook will be registered for the type
func (webhook *BKECluster) ValidateCreate(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	bkeCluster, ok := obj.(*bkev1beta1.BKECluster)
	if !ok {
		return nil, apierrors.NewBadRequest(fmt.Sprintf("expected a ClusterClass but got a %T", obj))
	}
	// validate create params
	if err := webhook.validateCreateParams(ctx, bkeCluster); err != nil {
		return nil, err
	}

	// check all nodes all vip config is unique
	if err := webhook.ValidateBKEClustersNodesUnique(ctx, bkeCluster); err != nil {
		return nil, err
	}

	// 校验关联的 BKENode 是否存在且包含 master 节点
	if err := webhook.validateAssociatedNodes(ctx, bkeCluster); err != nil {
		return nil, err
	}

	// skip validate bkecluster which is not bkecluster
	if !clusterutil.IsBKECluster(bkeCluster) {
		if bkeCluster.Spec.DryRun {
			return nil, apierrors.NewForbidden(
				bkev1beta1.GroupVersion.WithResource("bkeclusters").GroupResource(),
				bkeCluster.Name,
				errors.Errorf("BKECluster %s is not 'bke' type cluster, not support dryRun", bkeCluster.Name),
			)
		}
		return nil, nil
	}

	if err := bkevalidate.ValidateBKEConfig(*bkeCluster.Spec.ClusterConfig); err != nil {
		return nil, apierrors.NewBadRequest(fmt.Sprintf("invalide Spec.ClusterConfig, %v", err))
	}
	if err := webhook.ValidateControlPlaneEndpoint(bkeCluster); err != nil {
		return nil, err
	}
	// 不可变 OS 模式下校验节点角色约束
	if err := webhook.validateImmutableOSConstraints(ctx, bkeCluster); err != nil {
		return nil, err
	}
	if err := webhook.ValidateLoadBalanceType(ctx, bkeCluster); err != nil {
		return nil, err
	}
	return nil, nil
}

// ValidateDelete implements webhook.Validator so a webhook will be registered for the type
func (webhook *BKECluster) ValidateDelete(_ context.Context, _ runtime.Object) (admission.Warnings, error) {
	return nil, nil
}

func (webhook *BKECluster) validateCreateParams(ctx context.Context, bkeCluster *bkev1beta1.BKECluster) error {
	nodes, err := webhook.NodeFetcher.GetNodesForBKECluster(ctx, bkeCluster)
	if err != nil {
		return errors.Wrapf(err, "failed to get nodes for BKECluster %s", bkeCluster.Name)
	}
	nodeLabelCount := countVictoriaMetricsNodeLabels(nodes)
	allErrs := webhook.validateVictoriaMetricsAddons(bkeCluster, nodeLabelCount)
	if len(allErrs) == 0 {
		return nil
	}
	return apierrors.NewInvalid(bkev1beta1.GroupVersion.WithKind("BKECluster").GroupKind(), bkeCluster.Name, allErrs)
}

func countVictoriaMetricsNodeLabels(nodes bkenode.Nodes) map[string]int {
	counts := map[string]int{
		"vmstorage":          0,
		"vminsert":           0,
		"vmselect":           0,
		"vmagent":            0,
		"vmalert":            0,
		"vmalertmanager":     0,
		"kube-state-metrics": 0,
	}
	for _, node := range nodes {
		for _, label := range node.Labels {
			if _, ok := counts[label.Value]; ok {
				counts[label.Value]++
			}
		}
	}
	return counts
}

func (webhook *BKECluster) validateVictoriaMetricsAddons(bkeCluster *bkev1beta1.BKECluster,
	nodeLabelCount map[string]int) field.ErrorList {
	var allErrs field.ErrorList
	for i, addon := range bkeCluster.Spec.ClusterConfig.Addons {
		if addon.Name != "victoriametrics-controller" {
			continue
		}
		allErrs = append(allErrs, webhook.validateVictoriaMetricsReplicas(bkeCluster, i, nodeLabelCount)...)
	}
	return allErrs
}

func (webhook *BKECluster) validateVictoriaMetricsReplicas(bkeCluster *bkev1beta1.BKECluster, addonIndex int, nodeLabelCount map[string]int) field.ErrorList {
	var allErrs field.ErrorList
	addon := bkeCluster.Spec.ClusterConfig.Addons[addonIndex]
	paramPath := field.NewPath("spec", "clusterConfig", "addons").Index(addonIndex).Child("param")

	// 如果使用 VMSingle 模式，不需要校验组件副本数
	if useVMSingle, ok := addon.Param["useVMSingle"]; ok && useVMSingle == "true" {
		return allErrs
	}

	allErrs = append(allErrs, webhook.validateBasicVMComponents(addon, paramPath, nodeLabelCount)...)
	allErrs = append(allErrs, webhook.validateVMAgentSpecialRule(addon, paramPath, nodeLabelCount)...)

	return allErrs
}

// 校验基本的 VM 组件副本数
func (webhook *BKECluster) validateBasicVMComponents(addon confv1beta1.Product, paramPath *field.Path, nodeLabelCount map[string]int) field.ErrorList {
	var allErrs field.ErrorList

	// 需要校验副本数的组件列表
	components := []struct {
		name         string
		replicaField string
	}{
		{"vmstorage", "vmStorageReplicaCount"},
		{"vminsert", "vmInsertReplicaCount"},
		{"vmselect", "vmSelectReplicaCount"},
		{"vmagent", "vmAgentReplicaCount"},
		{"vmagent", "vmAgentShareCount"},
		{"vmalert", "vmAlertReplicaCount"},
		{"vmalertmanager", "vmAlertManagerReplicaCount"},
		{"kube-state-metrics", "kubeStateMetricsReplicaCount"},
	}

	for _, comp := range components {
		replicaStr, ok := addon.Param[comp.replicaField]
		if !ok {
			continue
		}

		// 解析副本数
		replicas, err := strconv.Atoi(replicaStr)
		if err != nil {
			allErrs = append(allErrs, field.Invalid(paramPath.Child(comp.replicaField), replicaStr,
				fmt.Sprintf("%s replica count must be a valid integer", comp.name)))
			continue
		}

		if replicas < 1 {
			allErrs = append(allErrs, field.Invalid(paramPath.Child(comp.replicaField), replicaStr,
				fmt.Sprintf("%s replica count must be at least 1", comp.name)))
			continue
		}

		// 校验副本数是否超过可用节点数
		availableNodes := nodeLabelCount[comp.name]
		if availableNodes < replicas {
			allErrs = append(allErrs, field.Invalid(paramPath.Child(comp.replicaField), replicaStr,
				fmt.Sprintf("%s requires %d %s but only %d nodes can be used",
					comp.name, replicas, comp.replicaField, availableNodes)))
		}
	}

	return allErrs
}

// 校验 vmagent 的特殊乘积规则
func (webhook *BKECluster) validateVMAgentSpecialRule(addon confv1beta1.Product, paramPath *field.Path, nodeLabelCount map[string]int) field.ErrorList {
	var allErrs field.ErrorList

	replicaStr, ok1 := addon.Param["vmAgentReplicaCount"]
	shareStr, ok2 := addon.Param["vmAgentShareCount"]
	if !ok1 || !ok2 {
		return allErrs
	}

	replicas, err1 := strconv.Atoi(replicaStr)
	share, err2 := strconv.Atoi(shareStr)

	if err1 != nil {
		return append(allErrs, field.Invalid(paramPath.Child("vmAgentReplicaCount"), replicaStr,
			"vmAgentReplicaCount must be a valid integer"))
	}
	if err2 != nil {
		return append(allErrs, field.Invalid(paramPath.Child("vmAgentShareCount"), shareStr,
			"vmAgentShareCount must be a valid integer"))
	}

	totalRequired := replicas * share
	availableNodes := nodeLabelCount["vmagent"]

	if availableNodes < totalRequired {
		allErrs = append(allErrs, field.Invalid(paramPath.Child("vmAgentShareCount"), shareStr,
			fmt.Sprintf("vmagent requires %d total instances (replicas:%d × share:%d) but only %d nodes available",
				totalRequired, replicas, share, availableNodes)))
	}

	return allErrs
}

func (webhook *BKECluster) ValidateBKEClustersNodesUnique(ctx context.Context, bkeCluster *bkev1beta1.BKECluster) error {
	// 不是bke集群直接返回
	if !clusterutil.IsBKECluster(bkeCluster) {
		return nil
	}

	// get all bkeCluster
	bkeClusterLi := &bkev1beta1.BKEClusterList{}
	if err := webhook.Client.List(ctx, bkeClusterLi, &client.ListOptions{}); err != nil {
		return err
	}

	// 从 BKENode CRD 获取当前集群的节点
	inNodes, err := webhook.NodeFetcher.GetNodesForBKECluster(ctx, bkeCluster)
	if err != nil {
		return errors.Wrapf(err, "failed to get nodes for BKECluster %s", bkeCluster.Name)
	}
	endPoint := bkeCluster.Spec.ControlPlaneEndpoint

	var errs []metav1.StatusCause

	for _, bc := range bkeClusterLi.Items {
		if bc.Name == bkeCluster.Name {
			continue
		}
		if !clusterutil.IsBKECluster(&bc) {
			continue
		}

		if bc.Spec.ControlPlaneEndpoint.Host == endPoint.Host {
			errs = append(errs, metav1.StatusCause{
				Message: fmt.Sprintf("cluster %q controlPlaneEndpoint host %q is equal with cluster %q controlPlaneEndpoint host", bc.Name, endPoint.Host, bkeCluster.Name),
				Field:   field.NewPath("spec", "controlPlaneEndpoint", "host").String(),
			})
		}
		// 从 BKENode CRD 获取其他集群的节点
		bcNodes, err := webhook.NodeFetcher.GetNodesForBKECluster(ctx, &bc)
		if err != nil {
			logger.Warnf("failed to get nodes for BKECluster %s: %v", bc.Name, err)
			continue
		}
		for i, node := range inNodes {
			if bcNodes.Filter(bkenode.FilterOptions{"IP": node.IP}).Length() != 0 {
				errs = append(errs, metav1.StatusCause{
					Message: fmt.Sprintf("Node %s is configured in both BKECluster %s and BKECluster %s", node.IP, utils.ClientObjNS(bkeCluster), utils.ClientObjNS(&bc)),
					Field:   field.NewPath("bkenodes").Index(i).String(),
				})
			}
		}
	}

	if len(errs) > 0 {
		return apierrors.NewApplyConflict(errs, bkeCluster.Name)
	}

	return nil
}

// ValidateControlPlaneEndpoint validates the control plane endpoint
func (webhook *BKECluster) ValidateControlPlaneEndpoint(bkeCluster *bkev1beta1.BKECluster) error {
	ControlPlaneEndpointPath := field.NewPath("Spec", "ControlPlaneEndpoint")
	ControlPlaneEndpointHostPath := ControlPlaneEndpointPath.Child("Host")
	ControlPlaneEndpointPortPath := ControlPlaneEndpointPath.Child("Port")

	switch {
	case bkeCluster.Spec.ControlPlaneEndpoint.Host != "" && net.ParseIP(bkeCluster.Spec.ControlPlaneEndpoint.Host) == nil:
		return field.Invalid(ControlPlaneEndpointHostPath, bkeCluster.Spec.ControlPlaneEndpoint.Host, "Host must be a valid IP string")
	case bkeCluster.Spec.ControlPlaneEndpoint.IsZero():
		return field.Required(ControlPlaneEndpointPath, "Host and Port must be set")
	case bkeCluster.Spec.ControlPlaneEndpoint.Host != "" && bkeCluster.Spec.ControlPlaneEndpoint.Port == 0:
		return field.Required(ControlPlaneEndpointPortPath, "Port must be set")
	case bkeCluster.Spec.ControlPlaneEndpoint.Host == "" && bkeCluster.Spec.ControlPlaneEndpoint.Port != 0:
		return field.Required(ControlPlaneEndpointHostPath, "Host must be set")
	case bkeCluster.Spec.ControlPlaneEndpoint.Port < 0 || bkeCluster.Spec.ControlPlaneEndpoint.Port > 65535:
		return field.Invalid(ControlPlaneEndpointPortPath, bkeCluster.Spec.ControlPlaneEndpoint.Port, "port can only be an integer between 0 and 65535")
	case bkeCluster.Spec.ControlPlaneEndpoint.IsValid():
		return nil
	default:
		return nil
	}
}

// validateImmutableOSConstraints 不可变 OS 模式下校验节点角色约束。
// master 跑普通 openEuler、worker 跑 KubeOS VM，同一节点不能同时作为纯 master 和纯 worker。
func (webhook *BKECluster) validateImmutableOSConstraints(ctx context.Context, bkeCluster *bkev1beta1.BKECluster) error {
	if !clusterutil.IsImmutableOSMode(bkeCluster) {
		return nil
	}

	nodes, err := webhook.NodeFetcher.GetNodesForBKECluster(ctx, bkeCluster)
	if err != nil {
		return errors.Wrapf(err, "failed to get BKENode list")
	}

	// 按 IP 分组，检查同一 IP 是否被分配为冲突的纯角色
	ipRoles := make(map[string][]string)
	for _, node := range nodes {
		n := bkenode.Node(node)
		if n.IsMasterWorker() {
			continue
		}
		if n.IsMaster() {
			ipRoles[node.IP] = append(ipRoles[node.IP], "master("+node.Hostname+")")
		}
		if n.IsWorker() {
			ipRoles[node.IP] = append(ipRoles[node.IP], "worker("+node.Hostname+")")
		}
	}

	for ip, roles := range ipRoles {
		if len(roles) > 1 {
			return apierrors.NewForbidden(
				bkev1beta1.GroupVersion.WithResource("bkeclusters").GroupResource(),
				bkeCluster.Name,
				fmt.Errorf("immutable OS mode does not allow the same node %s to be assigned as both master and worker "+
					"(master runs plain openEuler, worker runs KubeOS VM): %v", ip, roles),
			)
		}
	}
	return nil
}

// ValidateLoadBalanceType validates the loadBalanceType field and its combination constraints:
// only ha/nginx are allowed (empty defaults to ha via the CRD default);
// nginx mode is mutually exclusive with an external LB (customExtra["extraLoadBalanceIP"]);
// when nginx mode is set, an explicit ControlPlaneEndpoint.Host must be a master node IP
// (VIP and 127.0.0.1 are rejected).
func (webhook *BKECluster) ValidateLoadBalanceType(ctx context.Context, bkeCluster *bkev1beta1.BKECluster) error {
	cfg := bkeCluster.Spec.ClusterConfig
	if cfg == nil {
		return nil
	}
	lbTypePath := field.NewPath("spec", "clusterConfig", "cluster", "loadBalanceType")
	switch cfg.Cluster.LoadBalanceType {
	case "", "ha":
		return nil
	case "nginx":
	default:
		return field.Invalid(lbTypePath, cfg.Cluster.LoadBalanceType, `must be "ha" or "nginx"`)
	}

	if v, ok := cfg.CustomExtra["extraLoadBalanceIP"]; ok && v != "" {
		return field.Invalid(lbTypePath, cfg.Cluster.LoadBalanceType,
			`"nginx" is mutually exclusive with customExtra["extraLoadBalanceIP"]`)
	}

	host := bkeCluster.Spec.ControlPlaneEndpoint.Host
	if host == "" {
		return nil
	}
	nodes, err := webhook.NodeFetcher.GetNodesForBKECluster(ctx, bkeCluster)
	if err != nil {
		return errors.Wrapf(err, "failed to get nodes for BKECluster %s", bkeCluster.Name)
	}
	if nodes.Master().Filter(bkenode.FilterOptions{"IP": host}).Length() == 0 {
		return field.Invalid(field.NewPath("spec", "controlPlaneEndpoint", "host"), host,
			`host must be a master node IP when loadBalanceType is "nginx" (VIP or 127.0.0.1 is not allowed)`)
	}
	return nil
}

// validateVersionUpdate validate the cluster version update
func (webhook *BKECluster) validateVersionUpdate(ctx context.Context, newBKECluster, oldBKECluster *bkev1beta1.BKECluster) error {
	if oldBKECluster.Spec.ClusterConfig.Cluster.KubernetesVersion == "" {
		return nil
	}

	versionPath := field.NewPath("spec", "clusterConfig", "cluster", "kubernetesVersion")
	fromVersion, err := version.ParseMajorMinorPatch(oldBKECluster.Spec.ClusterConfig.Cluster.KubernetesVersion)
	if err != nil {
		return field.InternalError(versionPath, fmt.Errorf("parse kubernetes version %q failed, err: %w", oldBKECluster.Spec.ClusterConfig.Cluster.KubernetesVersion, err))
	}
	toVersion, err := version.ParseMajorMinorPatch(newBKECluster.Spec.ClusterConfig.Cluster.KubernetesVersion)
	if err != nil {
		return field.InternalError(versionPath, fmt.Errorf("parse kubernetes version %q failed, err: %w", newBKECluster.Spec.ClusterConfig.Cluster.KubernetesVersion, err))
	}

	// return early if kubernetes version not change
	if version.Compare(toVersion, fromVersion) == 0 {
		return nil
	}

	if err := webhook.validateKubernetesVersionUpgrade(newBKECluster,
		oldBKECluster, fromVersion, toVersion, versionPath); err != nil {
		return err
	}

	if err := webhook.validateKubernetesClusterUpgradeability(ctx, newBKECluster, oldBKECluster); err != nil {
		return err
	}

	return nil
}

// validateKubernetesVersionUpgrade validates kubernetes version upgrade rules
func (webhook *BKECluster) validateKubernetesVersionUpgrade(newBKECluster,
	oldBKECluster *bkev1beta1.BKECluster, fromVersion, toVersion semver.Version,
	versionPath *field.Path) error {
	// compare version is upgrade
	if version.Compare(toVersion, fromVersion) < 0 {
		return field.Invalid(
			versionPath,
			newBKECluster.Spec.ClusterConfig.Cluster.KubernetesVersion,
			fmt.Sprintf(
				"new kubernetes version %q is lower than old kubernetes version %q, that is not allowed",
				newBKECluster.Spec.ClusterConfig.Cluster.KubernetesVersion,
				oldBKECluster.Spec.ClusterConfig.Cluster.KubernetesVersion,
			),
		)
	}

	// return early if BKECluster is 'other' type
	if clusterutil.IsOtherCluster(newBKECluster) || clusterutil.IsOtherCluster(oldBKECluster) {
		return field.InternalError(
			versionPath,
			errors.Errorf(
				"BKECluster %s is 'other' type cluster, not support upgrade kubernetes version",
				newBKECluster.Name,
			),
		)
	}

	// compare toVersion is lower than ExpectMinK8sVersion
	if toVersion.LT(bkev1beta1.ExpectMinK8sVersion) {
		return field.Invalid(
			versionPath,
			newBKECluster.Spec.ClusterConfig.Cluster.KubernetesVersion,
			fmt.Sprintf(
				"new kubernetes version %q is lower than expect min kubernetes version %q, that is not allowed",
				newBKECluster.Spec.ClusterConfig.Cluster.KubernetesVersion,
				bkev1beta1.ExpectMinK8sVersion,
			),
		)
	}

	return nil
}

// validateKubernetesClusterUpgradeability checks if kubernetes cluster can be upgraded
func (webhook *BKECluster) validateKubernetesClusterUpgradeability(ctx context.Context, newBKECluster,
	oldBKECluster *bkev1beta1.BKECluster) error {
	return webhook.validateCommonUpgradeability(ctx, newBKECluster, oldBKECluster)
}

// validateEtcdVersionUpdate validate the etcd version update
func (webhook *BKECluster) validateEtcdVersionUpdate(ctx context.Context, newBKECluster, oldBKECluster *bkev1beta1.BKECluster) error {
	if oldBKECluster.Spec.ClusterConfig.Cluster.EtcdVersion == "" {
		return nil
	}

	versionPath := field.NewPath("spec", "clusterConfig", "cluster", "etcdVersion")
	fromVersion, err := version.ParseMajorMinorPatch(oldBKECluster.Spec.ClusterConfig.Cluster.EtcdVersion)
	if err != nil {
		return field.InternalError(versionPath, fmt.Errorf("parse etcd version %q failed, err: %w", oldBKECluster.Spec.ClusterConfig.Cluster.EtcdVersion, err))
	}
	toVersion, err := version.ParseMajorMinorPatch(newBKECluster.Spec.ClusterConfig.Cluster.EtcdVersion)
	if err != nil {
		return field.InternalError(versionPath, fmt.Errorf("parse etcd version %q failed, err: %w", newBKECluster.Spec.ClusterConfig.Cluster.EtcdVersion, err))
	}

	// return early if etcd version not change
	if version.Compare(toVersion, fromVersion) == 0 {
		return nil
	}

	if err := webhook.validateEtcdVersionUpgrade(newBKECluster, oldBKECluster,
		fromVersion, toVersion, versionPath); err != nil {
		return err
	}

	if err := webhook.validateEtcdClusterUpgradeability(ctx, newBKECluster, oldBKECluster); err != nil {
		return err
	}

	return nil
}

// validateEtcdVersionUpgrade validates etcd version upgrade rules
func (webhook *BKECluster) validateEtcdVersionUpgrade(newBKECluster,
	oldBKECluster *bkev1beta1.BKECluster, fromVersion,
	toVersion semver.Version, versionPath *field.Path) error {
	// compare version is upgrade
	if version.Compare(toVersion, fromVersion) < 0 {
		return field.Invalid(
			versionPath,
			newBKECluster.Spec.ClusterConfig.Cluster.EtcdVersion,
			fmt.Sprintf(
				"new etcd version %q is lower than old etcd version %q, that is not allowed",
				newBKECluster.Spec.ClusterConfig.Cluster.EtcdVersion,
				oldBKECluster.Spec.ClusterConfig.Cluster.EtcdVersion,
			),
		)
	}

	// it's upgrade now

	// return early if BKECluster is 'other' type
	if clusterutil.IsOtherCluster(newBKECluster) || clusterutil.IsOtherCluster(oldBKECluster) {
		return field.InternalError(
			versionPath,
			errors.Errorf(
				"BKECluster %s is 'other' type cluster, not support upgrade etcd version",
				newBKECluster.Name,
			),
		)
	}

	return nil
}

// validateNodeAgentStatus checks if all nodes have alive agents
func (webhook *BKECluster) validateNodeAgentStatus(ctx context.Context, bkeCluster *bkev1beta1.BKECluster) error {
	// 从 BKENode CRD 获取节点状态
	bkeNodes, err := webhook.NodeFetcher.GetBKENodesWrapperForCluster(ctx, bkeCluster)
	if err != nil {
		return apierrors.NewInternalError(errors.Wrapf(err, "failed to get nodes for BKECluster %s", bkeCluster.Name))
	}

	var errs []error
	// 排除所有具有失败状态的节点判断agent是否alive
	for _, bkeNode := range bkeNodes {
		nodeIP := bkeNode.Spec.IP
		if bkeNodes.GetNodeStateFlag(nodeIP, bkev1beta1.NodeFailedFlag) {
			continue
		}
		if !bkeNodes.GetNodeStateFlag(nodeIP, bkev1beta1.NodeAgentReadyFlag) {
			errs = append(errs, fmt.Errorf("node %s BKEAgent is not alive, cannot be upgraded", nodeIP))
		}
	}
	if len(errs) > 0 {
		return apierrors.NewForbidden(
			bkev1beta1.GroupVersion.WithResource("bkeclusters").GroupResource(),
			bkeCluster.Name,
			kerrors.NewAggregate(errs),
		)
	}
	return nil
}

// validateCommonUpgradeability provides common logic for kubernetes and etcd cluster upgradeability
func (webhook *BKECluster) validateCommonUpgradeability(ctx context.Context, newBKECluster,
	_ *bkev1beta1.BKECluster) error {
	// not allow upgrade if newBKECluster not healthy
	if newBKECluster.Status.ClusterStatus != bkev1beta1.ClusterStatusReady {
		return apierrors.NewForbidden(
			bkev1beta1.GroupVersion.WithResource("bkeclusters").GroupResource(),
			newBKECluster.Name,
			errors.Errorf("BKECluster %s is not in a ready state and cannot be upgraded", newBKECluster.Name),
		)
	}

	return webhook.validateNodeAgentStatus(ctx, newBKECluster)
}

// validateEtcdClusterUpgradeability checks if etcd cluster can be upgraded
func (webhook *BKECluster) validateEtcdClusterUpgradeability(ctx context.Context, newBKECluster,
	oldBKECluster *bkev1beta1.BKECluster) error {
	return webhook.validateCommonUpgradeability(ctx, newBKECluster, oldBKECluster)
}

// validateDryRun check status is not allowed to update
func (webhook *BKECluster) validateDryRun(newBKECluster, oldBKECluster *bkev1beta1.BKECluster) error {
	dryRunFlag := newBKECluster.Spec.DryRun || oldBKECluster.Spec.DryRun
	if dryRunFlag && !clusterutil.IsBKECluster(newBKECluster) {
		return apierrors.NewForbidden(
			bkev1beta1.GroupVersion.WithResource("bkeclusters").GroupResource(),
			newBKECluster.Name,
			errors.Errorf(
				"BKECluster %s is %q type cluster, not support dryRun",
				newBKECluster.Name,
				clusterutil.GetClusterType(newBKECluster),
			),
		)
	}
	return nil
}

func (webhook *BKECluster) validateStandBKECluster(ctx context.Context, newBKECluster, oldBKECluster *bkev1beta1.BKECluster) error {
	if err := webhook.validateFieldUpdate(newBKECluster, oldBKECluster, notAllowedPaths); err != nil {
		return err
	}

	if err := webhook.validateVersionUpdate(ctx, newBKECluster, oldBKECluster); err != nil {
		return err
	}

	if err := webhook.validateEtcdVersionUpdate(ctx, newBKECluster, oldBKECluster); err != nil {
		return err
	}

	if err := webhook.validateDryRun(newBKECluster, oldBKECluster); err != nil {
		return err
	}

	// 校验静态 Pod 配置：管控面组件非空 + etcd 不可变字段保护
	if err := webhook.validateStaticPodConfigUpdate(newBKECluster, oldBKECluster); err != nil {
		return err
	}

	if !clusterutil.IsBKECluster(newBKECluster) {
		if err := bkevalidate.ValidateNonStandardBKEConfig(*newBKECluster.Spec.ClusterConfig); err != nil {
			return apierrors.NewBadRequest(fmt.Sprintf("invalide Spec.ClusterConfig, %v", err))
		}
		return nil
	}

	if err := bkevalidate.ValidateBKEConfig(*newBKECluster.Spec.ClusterConfig); err != nil {
		return apierrors.NewBadRequest(fmt.Sprintf("invalide Spec.ClusterConfig, %v", err))
	}
	return nil
}

func (webhook *BKECluster) validateNonStandBKECluster(newBKECluster, oldBKECluster *bkev1beta1.BKECluster) error {
	if err := webhook.validateFieldUpdate(newBKECluster, oldBKECluster, nonStandNotAllowedPaths); err != nil {
		return err
	}
	if err := bkevalidate.ValidateNonStandardBKEConfig(*newBKECluster.Spec.ClusterConfig); err != nil {
		return apierrors.NewBadRequest(fmt.Sprintf("invalide Spec.ClusterConfig, %v", err))
	}
	return nil
}
func (webhook *BKECluster) validateStaticPodConfigUpdate(new, old *bkev1beta1.BKECluster) error {
	if new == nil || new.Spec.ClusterConfig == nil {
		return nil
	}
	var allErrs field.ErrorList
	cluster := new.Spec.ClusterConfig.Cluster

	// 1) 管控面三组件必须同时存在；逐字段校验并指向具体子路径，便于定位缺失项
	clusterPath := field.NewPath("spec", "clusterConfig", "cluster")
	if cluster.APIServer == nil {
		allErrs = append(allErrs, field.Required(
			clusterPath.Child("apiServer"),
			"APIServer must be present",
		))
	}
	if cluster.ControllerManager == nil {
		allErrs = append(allErrs, field.Required(
			clusterPath.Child("controllerManager"),
			"ControllerManager must be present",
		))
	}
	if cluster.Scheduler == nil {
		allErrs = append(allErrs, field.Required(
			clusterPath.Child("scheduler"),
			"Scheduler must be present",
		))
	}

	// 2) etcd 不可变字段保护（仅更新场景）
	// etcd 配置段缺失时其不可变字段按零值参与比较，因此新增 etcd 段时引入 DataDir /
	// ServerCertSANs / PeerCertSANs，或删除 etcd 段导致这些字段被移除，同样会被拦截；
	// 仅新增/修改 ExtraArgs、ExtraVolumes 等可变字段不受影响。
	if old != nil && old.Spec.ClusterConfig != nil {
		changed, fieldName := phaseutil.EtcdImmutableFieldsChanged(
			&old.Spec.ClusterConfig.Cluster, &cluster,
		)
		if changed {
			allErrs = append(allErrs, field.Forbidden(
				field.NewPath("spec", "clusterConfig", "cluster", "etcd", fieldName),
				"bootstrap-time immutable field; control plane re-deployment required",
			))
		}
	}

	if len(allErrs) > 0 {
		return apierrors.NewInvalid(
			bkev1beta1.GroupVersion.WithKind("BKECluster").GroupKind(),
			new.Name,
			allErrs,
		)
	}
	return nil
}

// validateFieldUpdate validates that certain fields in BKECluster are not modified
func (webhook *BKECluster) validateFieldUpdate(newBKECluster, oldBKECluster *bkev1beta1.BKECluster, notAllowedPaths [][]string) error {
	// Serialize old and new objects to JSON for comparison
	originalJSON, err := json.Marshal(oldBKECluster)
	if err != nil {
		return apierrors.NewInternalError(err)
	}
	modifiedJSON, err := json.Marshal(newBKECluster)
	if err != nil {
		return apierrors.NewInternalError(err)
	}

	// Create a merge patch to identify differences
	diff, err := jsonpatch.CreateMergePatch(originalJSON, modifiedJSON)
	if err != nil {
		return apierrors.NewInternalError(err)
	}

	// Parse the patch into a map structure
	jsonPatch := map[string]interface{}{}
	if err := json.Unmarshal(diff, &jsonPatch); err != nil {
		return apierrors.NewInternalError(err)
	}

	// Extract all modified paths from the patch
	modifiedPaths := extractPaths([]string{}, jsonPatch)

	// Validate that no disallowed paths are being modified
	allErrs := field.ErrorList{}
	for _, path := range modifiedPaths {
		// Skip empty paths
		if len(path) == 0 {
			continue
		}

		// Check if this path is in the not-allowed list
		if isPathNotAllowed(notAllowedPaths, path) {
			// Build the field path for the error message
			if len(path) == 1 {
				allErrs = append(allErrs, field.Forbidden(field.NewPath(path[0]), "cannot be modified"))
			} else {
				allErrs = append(allErrs, field.Forbidden(field.NewPath(path[0], path[1:]...), "cannot be modified"))
			}
		}
	}

	// Return validation errors if any were found
	if len(allErrs) > 0 {
		return apierrors.NewInvalid(bkev1beta1.GroupVersion.WithKind("BKECluster").GroupKind(), newBKECluster.Name, allErrs)
	}
	return nil
}

// isPathNotAllowed checks if a given path matches any of the not-allowed paths
func isPathNotAllowed(notAllowedPaths [][]string, path []string) bool {
	for _, notAllowedPath := range notAllowedPaths {
		if pathsMatch(notAllowedPath, path) {
			return true
		}
	}
	return false
}

// extractPaths recursively builds a list of all paths that are being modified in the diff
func extractPaths(currentPath []string, diff map[string]interface{}) [][]string {
	var allPaths [][]string

	for key, value := range diff {
		// Try to cast value to a nested map
		nestedMap, isNested := value.(map[string]interface{})

		if !isNested {
			// Leaf node - create a copy of the current path and append the key
			pathCopy := make([]string, len(currentPath))
			copy(pathCopy, currentPath)
			allPaths = append(allPaths, append(pathCopy, key))
		} else {
			// Nested object - recurse into it
			allPaths = append(allPaths, extractPaths(append(currentPath, key), nestedMap)...)
		}
	}

	return allPaths
}

// pathsMatch determines if a given path matches a pattern path (which may contain wildcards)
func pathsMatch(patternPath, actualPath []string) bool {
	// Empty paths cannot match
	if len(patternPath) == 0 || len(actualPath) == 0 {
		return false
	}

	// Iterate through the actual path
	for i := range actualPath {
		// If we've exceeded the pattern length, no match
		if i > len(patternPath)-1 {
			return false
		}

		// Wildcard matches everything from this point
		if patternPath[i] == "*" {
			return true
		}

		// Exact match required
		if actualPath[i] != patternPath[i] {
			return false
		}
	}

	// The actual path must match the entire pattern path
	// (or end at a wildcard, which is handled above)
	return len(actualPath)-1 >= len(patternPath)-1
}

// defaultConfigContext holds the context for setting default configurations
type defaultConfigContext struct {
	ctx        context.Context
	bkeCluster *bkev1beta1.BKECluster
	data       map[string]string
	bkeNodes   bkenode.Nodes
	patchFuncs []func(cluster *bkev1beta1.BKECluster)
}

func setBKEClusterDefaultConfig(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, nodeFetcher *nodeutil.NodeFetcher) {
	if bkeCluster == nil {
		return
	}

	log := logger.With("bkeCluster", utils.ClientObjNS(bkeCluster))

	data, err := clusterutil.GetBKEConfigCMData(ctx, c)
	if err != nil {
		log.Errorf("get bke-config configmap data failed, err:%v", err)
		return
	}
	if data == nil {
		log.Errorf("bke-config configmap data is nil")
		return
	}

	// 从 BKENode CRD 获取节点信息
	var bkeNodes bkenode.Nodes
	if nodeFetcher != nil {
		bkeNodes, err = nodeFetcher.GetNodesForBKECluster(ctx, bkeCluster)
		if err != nil {
			log.Warnf("get nodes for BKECluster failed, err:%v", err)
			bkeNodes = bkenode.Nodes{}
		}
	}

	// 初始化 CustomExtra
	if bkeCluster.Spec.ClusterConfig.CustomExtra == nil {
		bkeCluster.Spec.ClusterConfig.CustomExtra = map[string]string{}
	}

	// 构建配置上下文
	configCtx := &defaultConfigContext{
		ctx:        ctx,
		bkeCluster: bkeCluster,
		data:       data,
		bkeNodes:   bkeNodes,
		patchFuncs: []func(cluster *bkev1beta1.BKECluster){},
	}

	// 分步设置各项配置
	configCtx.setRepoDefaults()
	configCtx.setNTPServerDefault()
	configCtx.setContainerdDefault()
	configCtx.setHostRelatedDefaults()
	configCtx.setAddonDefaults()
	configCtx.setVirtualRouterIdDefaults()

	// 应用所有补丁函数
	for _, patchFunc := range configCtx.patchFuncs {
		patchFunc(bkeCluster)
	}
}

// setRepoDefaults 设置仓库相关默认配置
func (c *defaultConfigContext) setRepoDefaults() {
	// set yum repo port
	if yumPort := c.data["yumRepoPort"]; yumPort != "" && c.bkeCluster.Spec.ClusterConfig.Cluster.HTTPRepo.Port == "" {
		c.patchFuncs = append(c.patchFuncs, func(cluster *bkev1beta1.BKECluster) {
			cluster.Spec.ClusterConfig.Cluster.HTTPRepo.Port = yumPort
		})
	}

	// set image repo port
	if imagePort := c.data["imageRepoPort"]; imagePort != "" && c.bkeCluster.Spec.ClusterConfig.Cluster.ImageRepo.Port == "" {
		c.patchFuncs = append(c.patchFuncs, func(cluster *bkev1beta1.BKECluster) {
			cluster.Spec.ClusterConfig.Cluster.ImageRepo.Port = imagePort
		})
	}

	// set image repo prefix
	if imageRepoPrefix := c.data["imageRepoPrefix"]; imageRepoPrefix != "" && c.bkeCluster.Spec.ClusterConfig.Cluster.ImageRepo.Prefix == "" {
		c.patchFuncs = append(c.patchFuncs, func(cluster *bkev1beta1.BKECluster) {
			cluster.Spec.ClusterConfig.Cluster.ImageRepo.Prefix = imageRepoPrefix
		})
	}
}

// setNTPServerDefault 设置 NTP 服务器默认配置
func (c *defaultConfigContext) setNTPServerDefault() {
	// 纳管集群不强制设置ntp默认值
	if clusterutil.IsBocloudCluster(c.bkeCluster) || clusterutil.IsOtherCluster(c.bkeCluster) {
		return
	}

	if ntpServer := c.data["ntpServer"]; ntpServer != "" && c.bkeCluster.Spec.ClusterConfig.Cluster.NTPServer == "" {
		c.patchFuncs = append(c.patchFuncs, func(cluster *bkev1beta1.BKECluster) {
			cluster.Spec.ClusterConfig.Cluster.NTPServer = ntpServer
		})
	}
}

// setContainerdDefault 设置 containerd 默认配置
func (c *defaultConfigContext) setContainerdDefault() {
	containerd := c.data["containerd"]
	if containerd == "" {
		return
	}

	containerd = strings.ReplaceAll(containerd, "amd64", "{.arch}")
	containerd = strings.ReplaceAll(containerd, "arm64", "{.arch}")

	if _, ok := c.bkeCluster.Spec.ClusterConfig.CustomExtra["containerd"]; !ok {
		c.patchFuncs = append(c.patchFuncs, func(cluster *bkev1beta1.BKECluster) {
			cluster.Spec.ClusterConfig.CustomExtra["containerd"] = containerd
		})
	}
}

// setHostRelatedDefaults 设置与 host 相关的默认配置
func (c *defaultConfigContext) setHostRelatedDefaults() {
	host := c.data["host"]
	if host == "" {
		return
	}

	c.setRepoIPDefaults(host)
	c.setCustomExtraHostDefaults(host)
	c.setNFSDefaults(host)
	c.setBocoperatorDeployServerIP(host)
}

// setRepoIPDefaults 设置仓库 IP 默认值
func (c *defaultConfigContext) setRepoIPDefaults(host string) {
	if c.bkeCluster.Spec.ClusterConfig.Cluster.HTTPRepo.Ip == "" {
		c.patchFuncs = append(c.patchFuncs, func(cluster *bkev1beta1.BKECluster) {
			cluster.Spec.ClusterConfig.Cluster.HTTPRepo.Ip = host
		})
	}

	if c.bkeCluster.Spec.ClusterConfig.Cluster.ImageRepo.Ip == "" {
		c.patchFuncs = append(c.patchFuncs, func(cluster *bkev1beta1.BKECluster) {
			cluster.Spec.ClusterConfig.Cluster.ImageRepo.Ip = host
		})
	}
}

// setCustomExtraHostDefaults 设置 CustomExtra 中 host 相关默认值
func (c *defaultConfigContext) setCustomExtraHostDefaults(host string) {
	if _, ok := c.bkeCluster.Spec.ClusterConfig.CustomExtra["host"]; !ok {
		c.patchFuncs = append(c.patchFuncs, func(cluster *bkev1beta1.BKECluster) {
			cluster.Spec.ClusterConfig.CustomExtra["host"] = host
		})
	}
}

// setNFSDefaults 设置 NFS 相关默认配置
func (c *defaultConfigContext) setNFSDefaults(host string) {
	if _, ok := c.bkeCluster.Spec.ClusterConfig.CustomExtra["nfsServer"]; !ok {
		c.patchFuncs = append(c.patchFuncs, func(cluster *bkev1beta1.BKECluster) {
			cluster.Spec.ClusterConfig.CustomExtra["nfsServer"] = host
		})
	}

	if _, ok := c.bkeCluster.Spec.ClusterConfig.CustomExtra["nfsRootDir"]; !ok {
		c.patchFuncs = append(c.patchFuncs, func(cluster *bkev1beta1.BKECluster) {
			cluster.Spec.ClusterConfig.CustomExtra["nfsRootDir"] = "/"
		})
	}

	if _, ok := c.bkeCluster.Spec.ClusterConfig.CustomExtra["nfsVersion"]; !ok {
		c.patchFuncs = append(c.patchFuncs, func(cluster *bkev1beta1.BKECluster) {
			cluster.Spec.ClusterConfig.CustomExtra["nfsVersion"] = "4.1"
		})
	}
}

// setBocoperatorDeployServerIP 设置 bocoperator 的 deployServerIp
func (c *defaultConfigContext) setBocoperatorDeployServerIP(host string) {
	for i, addon := range c.bkeCluster.Spec.ClusterConfig.Addons {
		if addon.Name == "bocoperator" {
			idx := i // capture loop variable
			c.patchFuncs = append(c.patchFuncs, func(cluster *bkev1beta1.BKECluster) {
				cluster.Spec.ClusterConfig.Addons[idx].Param["deployServerIp"] = host
			})
		}
	}
}

// setAddonDefaults 设置插件相关默认配置
func (c *defaultConfigContext) setAddonDefaults() {
	for i, addon := range c.bkeCluster.Spec.ClusterConfig.Addons {
		switch addon.Name {
		case "bocoperator":
			c.setBocoperatorDefaults()
		case "kubeproxy":
			c.setKubeproxyDefaults(addon)
		case "victoriametrics-controller":
			c.setVictoriaMetricsControllerDefaultConfig(addon, i)
		default:
			// do nothing
		}
	}
}

// setBocoperatorDefaults 设置 bocoperator 相关默认配置
func (c *defaultConfigContext) setBocoperatorDefaults() {
	masterNodes := c.bkeNodes.Master().Decrypt()

	// 设置 pipelineServer
	if len(masterNodes) > 0 {
		if _, ok := c.bkeCluster.Spec.ClusterConfig.CustomExtra["pipelineServer"]; !ok {
			pipelineServerIP := masterNodes[0].IP
			c.patchFuncs = append(c.patchFuncs, func(cluster *bkev1beta1.BKECluster) {
				cluster.Spec.ClusterConfig.CustomExtra["pipelineServer"] = pipelineServerIP
			})
		}
	}

	// 设置 pipelineServerEnableCleanImages
	if _, ok := c.bkeCluster.Spec.ClusterConfig.CustomExtra["pipelineServerEnableCleanImages"]; !ok {
		c.patchFuncs = append(c.patchFuncs, func(cluster *bkev1beta1.BKECluster) {
			cluster.Spec.ClusterConfig.CustomExtra["pipelineServerEnableCleanImages"] = "false"
		})
	}
}

// setKubeproxyDefaults 设置 kubeproxy 相关默认配置
func (c *defaultConfigContext) setKubeproxyDefaults(addon confv1beta1.Product) {
	proxyMode := "iptables"
	if addon.Param["proxyMode"] != "" {
		proxyMode = addon.Param["proxyMode"]
	}

	c.patchFuncs = append(c.patchFuncs, func(cluster *bkev1beta1.BKECluster) {
		cluster.Spec.ClusterConfig.CustomExtra["proxyMode"] = proxyMode
	})
}

// setVictoriaMetricsControllerDefaultConfig 设置 victoriametrics-controller 插件的默认配置
func (c *defaultConfigContext) setVictoriaMetricsControllerDefaultConfig(addon confv1beta1.Product, addonIndex int) {

	// 定义默认配置映射
	defaultConfigs := map[string]string{
		"useVMSingle":                  "false",
		"vmSingleStorageSize":          "50Gi",
		"vmAgentAllowStatefulSet":      "true",
		"vmAgentCpuCount":              "4",
		"vmAgentMemorySize":            "8Gi",
		"vmAgentStorageSize":           "60Gi",
		"vmAgentReplicaCount":          "2",
		"vmAgentShareCount":            "2",
		"vmAgentScrapeInterval":        "20s",
		"vmSelectCpuCount":             "12",
		"vmSelectMemorySize":           "24Gi",
		"vmSelectStorageSize":          "35Gi",
		"vmSelectReplicaCount":         "2",
		"vmStorageCPUCount":            "5",
		"vmStorageMemorySize":          "32Gi",
		"vmStorageStorageSize":         "720Gi",
		"vmStorageReplicaCount":        "10",
		"vmInsertReplicaCount":         "6",
		"vmInsertCpuCount":             "4",
		"vmInsertMemorySize":           "8Gi",
		"vmAlertReplicaCount":          "2",
		"vmAlertCpuCount":              "4",
		"vmAlertMemorySize":            "16Gi",
		"vmAlertManagerReplicaCount":   "2",
		"vmAlertManagerCpuCount":       "4",
		"vmAlertManagerMemorySize":     "16Gi",
		"vmClusterRetentionPeriod":     "15d",
		"vmClusterReplicationFactor":   "2",
		"grafanaNodePort":              "30010",
		"kubeStateMetricsAutoSharding": "true",
		"kubeStateMetricsReplicaCount": "3",
		"kubeStateMetricsCpuCount":     "4",
		"kubeStateMetricsMemorySize":   "12Gi",
	}

	// 初始化 Param 如果为空
	if addon.Param == nil {
		c.patchFuncs = append(c.patchFuncs, func(cluster *bkev1beta1.BKECluster) {
			cluster.Spec.ClusterConfig.Addons[addonIndex].Param = make(map[string]string)
		})
	}

	// 为每个缺失的配置设置默认值
	for key, defaultValue := range defaultConfigs {
		if _, ok := addon.Param[key]; !ok {
			// 捕获局部变量
			key := key
			defaultValue := defaultValue
			c.patchFuncs = append(c.patchFuncs, func(cluster *bkev1beta1.BKECluster) {
				if cluster.Spec.ClusterConfig.Addons[addonIndex].Param == nil {
					cluster.Spec.ClusterConfig.Addons[addonIndex].Param = make(map[string]string)
				}
				cluster.Spec.ClusterConfig.Addons[addonIndex].Param[key] = defaultValue
			})
		}
	}
}

// setVirtualRouterIdDefaults 设置虚拟路由器 ID 默认配置
func (c *defaultConfigContext) setVirtualRouterIdDefaults() {
	masterVirtualRouterId, masterSet := c.setMasterVirtualRouterId()
	ingressVirtualRouterId, ingressSet := c.setIngressVirtualRouterId()

	// 确保 masterVirtualRouterId != ingressVirtualRouterId
	if masterSet && ingressSet && masterVirtualRouterId == ingressVirtualRouterId {
		for masterVirtualRouterId == ingressVirtualRouterId {
			ingressVirtualRouterId = utils.Random(1, 255)
		}
		c.patchFuncs = append(c.patchFuncs, func(cluster *bkev1beta1.BKECluster) {
			cluster.Spec.ClusterConfig.CustomExtra["ingressVirtualRouterId"] = strconv.Itoa(ingressVirtualRouterId)
		})
	}
}

// setMasterVirtualRouterId 设置 master 虚拟路由器 ID
func (c *defaultConfigContext) setMasterVirtualRouterId() (int, bool) {
	if !clusterutil.AvailableLoadBalancerEndPoint(c.bkeCluster.Spec.ControlPlaneEndpoint, c.bkeNodes) {
		return 0, false
	}

	if _, ok := c.bkeCluster.Spec.ClusterConfig.CustomExtra["masterVirtualRouterId"]; ok {
		return 0, false
	}

	masterVirtualRouterId := utils.Random(1, 255)
	c.patchFuncs = append(c.patchFuncs, func(cluster *bkev1beta1.BKECluster) {
		cluster.Spec.ClusterConfig.CustomExtra["masterVirtualRouterId"] = strconv.Itoa(masterVirtualRouterId)
	})
	return masterVirtualRouterId, true
}

// setIngressVirtualRouterId 设置 ingress 虚拟路由器 ID
func (c *defaultConfigContext) setIngressVirtualRouterId() (int, bool) {
	vip, _ := clusterutil.GetIngressConfig(c.bkeCluster.Spec.ClusterConfig.Addons)
	if vip == "" {
		return 0, false
	}

	if _, ok := c.bkeCluster.Spec.ClusterConfig.CustomExtra["ingressVirtualRouterId"]; ok {
		return 0, false
	}

	ingressVirtualRouterId := utils.Random(1, 255)
	c.patchFuncs = append(c.patchFuncs, func(cluster *bkev1beta1.BKECluster) {
		cluster.Spec.ClusterConfig.CustomExtra["ingressVirtualRouterId"] = strconv.Itoa(ingressVirtualRouterId)
	})
	return ingressVirtualRouterId, true
}

// getNodesForBKEClusterDirect 绕过 informer 缓存，直接从 API Server 读取 BKENode 列表。
func (webhook *BKECluster) getNodesForBKEClusterDirect(ctx context.Context, bkeCluster *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
	reader := webhook.getReader()

	bkeNodeList := &confv1beta1.BKENodeList{}
	if err := reader.List(ctx, bkeNodeList,
		client.InNamespace(bkeCluster.Namespace),
		client.MatchingLabels{nodeutil.ClusterNameLabel: bkeCluster.Name},
	); err != nil {
		return nil, err
	}

	nodes := bkenode.ConvertBKENodeListToNodes(bkeNodeList)
	nodes = bkenode.SetDefaultsForNodes(nodes)
	return nodes, nil
}

// validateAssociatedNodes 校验 BKECluster 创建时关联的 BKENode 是否满足要求
// 复用 ValidateNodesFields 校验：至少一个 master 节点、master 数量为奇数、至少一个 etcd 节点
// 单节点字段校验已在 BKENode 创建时通过 webhook ValidateSingleNode 完成，此处不再重复
func (webhook *BKECluster) validateAssociatedNodes(ctx context.Context, bkeCluster *bkev1beta1.BKECluster) error {
	nodes, err := webhook.getNodesForBKEClusterDirect(ctx, bkeCluster)
	if err != nil {
		return apierrors.NewInternalError(
			errors.Wrapf(err, "failed to get nodes for BKECluster %s", bkeCluster.Name))
	}
	if len(nodes) == 0 {
		return apierrors.NewForbidden(
			bkev1beta1.GroupVersion.WithResource("bkeclusters").GroupResource(),
			bkeCluster.Name,
			errors.New("BKECluster creation requires at least one associated BKENode"),
		)
	}
	if err := bkevalidate.ValidateNodesFields(nodes); err != nil {
		return apierrors.NewForbidden(
			bkev1beta1.GroupVersion.WithResource("bkeclusters").GroupResource(),
			bkeCluster.Name,
			err,
		)
	}
	return nil
}

// getReader 返回一个用于读取的 client.Reader，优先使用 APIReader 以绕过 informer 缓存。
func (webhook *BKECluster) getReader() client.Reader {
	if webhook.APIReader != nil {
		return webhook.APIReader
	}
	return webhook.Client
}
