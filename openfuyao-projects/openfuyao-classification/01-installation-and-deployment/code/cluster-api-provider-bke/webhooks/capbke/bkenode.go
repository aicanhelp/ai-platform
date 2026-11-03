/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package capbke

import (
	"context"
	"fmt"
	"reflect"
	"strings"

	"github.com/pkg/errors"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/json"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/webhook"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	bkevalidate "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/validation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/security"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

var bkeNodeLog = log.With("name", "BKENode")

// bkeNodeGR is the GroupResource for BKENode, derived from BKENodeGVK
var bkeNodeGR = schema.GroupResource{Group: confv1beta1.BKENodeGVK.Group, Resource: "bkenodes"}

// BKENode implements webhook for BKENode CRD
type BKENode struct {
	Client client.Client
}

func (webhook *BKENode) SetupWebhookWithManager(mgr ctrl.Manager) error {
	return ctrl.NewWebhookManagedBy(mgr).
		For(&confv1beta1.BKENode{}).
		WithValidator(webhook).
		WithDefaulter(webhook).
		Complete()
}

//+kubebuilder:webhook:verbs=create;update,path=/mutate-bke-bocloud-com-v1beta1-bkenode,mutating=true,failurePolicy=fail,sideEffects=None,groups=bke.bocloud.com,resources=bkenodes,versions=v1beta1,name=mbkenode.kb.io,admissionReviewVersions=v1
//+kubebuilder:webhook:verbs=create;update;delete,path=/validate-bke-bocloud-com-v1beta1-bkenode,mutating=false,failurePolicy=fail,sideEffects=None,groups=bke.bocloud.com,resources=bkenodes,versions=v1beta1,name=vbkenode.kb.io,admissionReviewVersions=v1

var _ webhook.CustomDefaulter = &BKENode{}
var _ webhook.CustomValidator = &BKENode{}

// Default implements webhook.Defaulter so a webhook will be registered for the type
func (webhook *BKENode) Default(ctx context.Context, obj runtime.Object) error {
	bkeNode, ok := obj.(*confv1beta1.BKENode)
	if !ok {
		return apierrors.NewBadRequest(fmt.Sprintf("expected a BKENode but got a %T", obj))
	}

	bkeNodeLog.Debugf("Defaulting BKENode %s/%s", bkeNode.Namespace, bkeNode.Name)

	// 设置默认端口
	if bkeNode.Spec.Port == "" {
		bkeNode.Spec.Port = "22"
	}

	// 加密密码
	if bkeNode.Spec.Password != "" {
		encryptedPassword, err := encryptPasswordIfNeeded(bkeNode.Spec.Password)
		if err != nil {
			// fail closed：密钥经启动预加载保证可用，运行期失败必为真实故障；
			// 放行会导致明文密码写入 etcd，危害大于准入失败。
			bkeNodeLog.Errorf("Failed to encrypt password for node %s: %v", bkeNode.Spec.IP, err)
			return apierrors.NewInternalError(
				fmt.Errorf("failed to encrypt password for node %s: %w", bkeNode.Spec.IP, err))
		}
		bkeNode.Spec.Password = encryptedPassword
		sanitizeLastAppliedPassword(bkeNode, encryptedPassword)
	}

	return nil
}

// encryptPasswordIfNeeded 如果密码未加密则进行加密
func encryptPasswordIfNeeded(password string) (string, error) {
	// 尝试解密，如果成功说明已经加密
	_, err := security.AesDecrypt(password)
	if err == nil {
		// 已加密，返回原始密码
		return password, nil
	}

	// 未加密，进行加密
	encrypted, err := security.AesEncrypt(password)
	if err != nil {
		return password, err
	}
	return encrypted, nil
}

// sanitizeLastAppliedPassword replaces the plaintext password in the
// kubectl.kubernetes.io/last-applied-configuration annotation with the
// encrypted value, preventing plaintext from being stored in etcd.
func sanitizeLastAppliedPassword(bkeNode *confv1beta1.BKENode, encryptedPassword string) {
	const annotationKey = "kubectl.kubernetes.io/last-applied-configuration"
	if bkeNode.Annotations == nil {
		return
	}
	lastApplied, ok := bkeNode.Annotations[annotationKey]
	if !ok {
		return
	}
	var applied map[string]interface{}
	if err := json.Unmarshal([]byte(lastApplied), &applied); err != nil {
		bkeNodeLog.Warnf("Failed to parse last-applied-configuration annotation: %v", err)
		return
	}
	specVal, exists := applied["spec"]
	if !exists {
		return
	}
	spec, ok := specVal.(map[string]interface{})
	if !ok {
		return
	}
	if _, hasPassword := spec["password"]; !hasPassword {
		return
	}
	spec["password"] = encryptedPassword
	updated, err := json.Marshal(applied)
	if err != nil {
		bkeNodeLog.Warnf("Failed to marshal updated last-applied-configuration: %v", err)
		return
	}
	bkeNode.Annotations[annotationKey] = string(updated)
}

// ValidateCreate implements webhook.Validator so a webhook will be registered for the type
func (webhook *BKENode) ValidateCreate(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	bkeNode, ok := obj.(*confv1beta1.BKENode)
	if !ok {
		return nil, apierrors.NewBadRequest(fmt.Sprintf("expected a BKENode but got a %T", obj))
	}

	bkeNodeLog.Debugf("Validating BKENode create %s/%s", bkeNode.Namespace, bkeNode.Name)

	// 验证节点配置
	if err := webhook.validateBKENodeSpec(bkeNode); err != nil {
		return nil, err
	}

	// 验证节点 IP 唯一性
	if err := webhook.validateNodeIPUnique(ctx, bkeNode); err != nil {
		return nil, err
	}

	// 拦截扩容 master 节点
	if err := webhook.validateMasterNodeCreation(ctx, bkeNode); err != nil {
		return nil, err
	}

	return nil, nil
}

// ValidateUpdate implements webhook.Validator so a webhook will be registered for the type
func (webhook *BKENode) ValidateUpdate(ctx context.Context, oldObj, newObj runtime.Object) (admission.Warnings, error) {
	oldBKENode, ok := oldObj.(*confv1beta1.BKENode)
	if !ok {
		return nil, apierrors.NewBadRequest(fmt.Sprintf("expected a BKENode but got a %T", oldObj))
	}
	newBKENode, ok := newObj.(*confv1beta1.BKENode)
	if !ok {
		return nil, apierrors.NewBadRequest(fmt.Sprintf("expected a BKENode but got a %T", newObj))
	}

	bkeNodeLog.Debugf("Validating BKENode update %s/%s", newBKENode.Namespace, newBKENode.Name)

	// 验证节点配置
	if err := webhook.validateBKENodeSpec(newBKENode); err != nil {
		return nil, err
	}

	// IP 不允许变更
	if oldBKENode.Spec.IP != newBKENode.Spec.IP {
		return nil, apierrors.NewForbidden(
			bkeNodeGR,
			newBKENode.Name,
			errors.New("BKENode IP cannot be changed after creation"),
		)
	}

	// 如果节点正在操作中，限制某些字段的变更
	if err := webhook.validateNodeStateTransition(oldBKENode, newBKENode); err != nil {
		return nil, err
	}

	// 拦截 worker 角色变更为 master
	if err := webhook.validateRoleChangeToMaster(oldBKENode, newBKENode); err != nil {
		return nil, err
	}

	return nil, nil
}

// ValidateDelete implements webhook.Validator so a webhook will be registered for the type
func (webhook *BKENode) ValidateDelete(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	bkeNode, ok := obj.(*confv1beta1.BKENode)
	if !ok {
		return nil, apierrors.NewBadRequest(fmt.Sprintf("expected a BKENode but got a %T", obj))
	}

	bkeNodeLog.Debugf("Validating BKENode delete %s/%s", bkeNode.Namespace, bkeNode.Name)

	// 检查节点是否处于可删除状态
	if err := webhook.validateNodeDeletable(ctx, bkeNode); err != nil {
		return nil, err
	}

	return nil, nil
}

// validateBKENodeSpec 验证 BKENode 规格
func (webhook *BKENode) validateBKENodeSpec(bkeNode *confv1beta1.BKENode) error {
	// 转换为 Node 类型进行验证
	n := bkenode.Node(bkeNode.ToNode())
	if err := bkevalidate.ValidateSingleNode(n); err != nil {
		return apierrors.NewBadRequest(fmt.Sprintf("invalid BKENode spec: %v", err))
	}
	return nil
}

// validateNodeIPUnique 验证节点 IP 在集群中唯一
func (webhook *BKENode) validateNodeIPUnique(ctx context.Context, bkeNode *confv1beta1.BKENode) error {
	clusterName := bkeNode.Labels[nodeutil.ClusterNameLabel]
	if clusterName == "" {
		return apierrors.NewBadRequest("BKENode must have cluster.x-k8s.io/cluster-name label")
	}

	// 获取同一集群下的所有 BKENode
	bkeNodeList := &confv1beta1.BKENodeList{}
	if err := webhook.Client.List(ctx, bkeNodeList,
		client.InNamespace(bkeNode.Namespace),
		client.MatchingLabels{nodeutil.ClusterNameLabel: clusterName},
	); err != nil {
		return apierrors.NewInternalError(err)
	}

	// 检查 IP 是否重复
	for _, existingNode := range bkeNodeList.Items {
		if existingNode.Name == bkeNode.Name {
			continue
		}
		if existingNode.Spec.IP == bkeNode.Spec.IP {
			return apierrors.NewConflict(
				bkeNodeGR,
				bkeNode.Name,
				fmt.Errorf("node with IP %s already exists in cluster %s", bkeNode.Spec.IP, clusterName),
			)
		}
	}

	return nil
}

// validateNodeStateTransition 验证节点状态转换
func (webhook *BKENode) validateNodeStateTransition(oldNode, newNode *confv1beta1.BKENode) error {
	// 如果节点正在操作中（Pending 或 Upgrading），不允许修改角色
	if oldNode.Status.State == confv1beta1.NodePending || oldNode.Status.State == confv1beta1.NodeUpgrading {
		if !reflect.DeepEqual(oldNode.Spec.Role, newNode.Spec.Role) {
			return apierrors.NewForbidden(
				bkeNodeGR,
				newNode.Name,
				errors.New("cannot change node role while node is in pending or upgrading state"),
			)
		}
	}

	return nil
}

// validateNodeDeletable 验证节点是否可删除
func (webhook *BKENode) validateNodeDeletable(ctx context.Context, bkeNode *confv1beta1.BKENode) error {
	// 关联的 BKECluster 正在删除时放行（集群级删除场景，所有节点都将被清理）
	if webhook.isBKEClusterDeleting(ctx, bkeNode) {
		return nil
	}

	// Pending 状态的节点正在部署中，不能删除
	if bkeNode.Status.State == confv1beta1.NodePending {
		return apierrors.NewForbidden(
			bkeNodeGR,
			bkeNode.Name,
			errors.New("cannot delete node while in pending state"),
		)
	}

	// Upgrading 状态的节点正在升级中，不能删除
	if bkeNode.Status.State == confv1beta1.NodeUpgrading {
		return apierrors.NewForbidden(
			bkeNodeGR,
			bkeNode.Name,
			errors.New("cannot delete node while upgrading"),
		)
	}

	// 检查是否为需要拦截的 master 节点删除
	if err := webhook.validateMasterNodeDeletion(ctx, bkeNode); err != nil {
		return err
	}

	return nil
}

// validateMasterNodeDeletion 拦截非控制器编排的 master 节点删除
func (webhook *BKENode) validateMasterNodeDeletion(ctx context.Context, bkeNode *confv1beta1.BKENode) error {
	// 控制器内部清理时已将节点状态设为 Deleting，放行
	if bkeNode.Status.State == confv1beta1.NodeDeleting {
		return nil
	}
	// Failed 及其衍生失败态（InitFailed、BootStrapFailed、DeleteFailed 等）的节点故障不可达，允许直接删除
	if strings.HasSuffix(string(bkeNode.Status.State), "Failed") {
		return nil
	}
	// 非 master 角色或 BKECluster 已不存在时不拦截
	if !webhook.shouldBlockMasterNode(ctx, bkeNode) {
		return nil
	}
	// 其余情况的 master 节点删除一律拦截
	return apierrors.NewForbidden(
		bkeNodeGR,
		bkeNode.Name,
		errors.New("master node deletion is not allowed"),
	)
}

// isBKEClusterMissing 检查 BKENode 关联的 BKECluster 是否已不存在
func (webhook *BKENode) isBKEClusterMissing(ctx context.Context, bkeNode *confv1beta1.BKENode) bool {
	clusterName := bkeNode.Labels[nodeutil.ClusterNameLabel]
	// 没有 cluster label 视为无关联 BKECluster，不拦截
	if clusterName == "" {
		return true
	}
	bkeCluster := &bkev1beta1.BKECluster{}
	err := webhook.Client.Get(ctx, client.ObjectKey{Namespace: bkeNode.Namespace, Name: clusterName}, bkeCluster)
	// BKECluster 已被删除（NotFound）时返回 true
	return apierrors.IsNotFound(err)
}

// isBKEClusterDeleting 检查关联的 BKECluster 是否正在删除。
// DeletionTimestamp 覆盖 kubectl delete bc 路径；ClusterDeleting 状态覆盖
// spec.reset=true 驱动的卸载路径（前台删除与文档主推的后台删除均不删除
// bc 对象，全程仅有 ClusterStatus 被控制器置为 Deleting）。
// 两种信号均只在删除/重置调谐中出现，不会在集群正常运行时误触发。
func (webhook *BKENode) isBKEClusterDeleting(ctx context.Context, bkeNode *confv1beta1.BKENode) bool {
	if webhook.Client == nil {
		return false
	}
	clusterName := bkeNode.Labels[nodeutil.ClusterNameLabel]
	if clusterName == "" {
		return false
	}
	bkeCluster := &bkev1beta1.BKECluster{}
	if err := webhook.Client.Get(ctx, client.ObjectKey{Namespace: bkeNode.Namespace, Name: clusterName}, bkeCluster); err != nil {
		// NotFound / 出错均不放行，走原有校验（NotFound 场景由 isBKEClusterMissing 处理）
		return false
	}
	return !bkeCluster.DeletionTimestamp.IsZero() ||
		bkeCluster.Status.ClusterStatus == bkev1beta1.ClusterDeleting
}

// shouldBlockMasterNode 判断是否需要拦截 master 节点操作
// 非 master 角色或关联 BKECluster 已不存在时返回 false
func (webhook *BKENode) shouldBlockMasterNode(ctx context.Context, bkeNode *confv1beta1.BKENode) bool {
	node := bkenode.Node(bkeNode.ToNode())
	if !node.IsMaster() && !node.IsMasterWorker() {
		return false
	}
	return !webhook.isBKEClusterMissing(ctx, bkeNode)
}

// validateMasterNodeCreation 拦截扩容场景下新建 master 角色的 BKENode
func (webhook *BKENode) validateMasterNodeCreation(ctx context.Context, bkeNode *confv1beta1.BKENode) error {
	// 非 master 角色或 BKECluster 已不存在（初始部署）时不拦截
	if !webhook.shouldBlockMasterNode(ctx, bkeNode) {
		return nil
	}
	// BKECluster 已存在 → 扩容场景，拦截
	return apierrors.NewForbidden(
		bkeNodeGR,
		bkeNode.Name,
		errors.New("master node scale-up is not allowed"),
	)
}

// validateRoleChangeToMaster 拦截任何情况下 worker 角色变更为 master
func (webhook *BKENode) validateRoleChangeToMaster(oldNode, newNode *confv1beta1.BKENode) error {
	oldTyped := bkenode.Node(oldNode.ToNode())
	newTyped := bkenode.Node(newNode.ToNode())
	// 原本不是 master，变更为 master → 拦截
	if !oldTyped.IsMaster() && !oldTyped.IsMasterWorker() {
		if newTyped.IsMaster() || newTyped.IsMasterWorker() {
			return apierrors.NewForbidden(
				bkeNodeGR,
				newNode.Name,
				errors.New("changing node role to master is not allowed"),
			)
		}
	}
	return nil
}
