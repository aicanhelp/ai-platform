/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Package controller 实现 KAE Operator 的控制器
package controller

import (
	"context"
	"fmt"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/util/workqueue"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
	"sigs.k8s.io/controller-runtime/pkg/source"

	kaev1 "gopkg.openfuyao.cn/kae-operator/api/v1"
	"gopkg.openfuyao.cn/kae-operator/internal/utils"
	"gopkg.openfuyao.cn/kae-operator/internal/zlog"
)

// String constants
const (
	kaePolicyControllerIndexKey = "metadata.kae.kaepolicy.controller"
	reconciledReason            = "Reconciled"
	reconcileFailedReason       = "ReconcileFailed"
	operandNotReadyReason       = "OperandNotReady"
	noKAENodeReason             = "NoKAENode"
	noNFDLabelsReason           = "NoNFDLabels"
)

// Time constants & other
const (
	maxConcurrentReconciles = 1
	delayMin                = 100 * time.Millisecond
	delayMax                = 3 * time.Second
	requeueAfterTime        = 5 * time.Second
)

// KAEPolicyReconciler 协调 KAEPolicy 对象。
type KAEPolicyReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=device.openfuyao.com,resources=kaepolicies,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=device.openfuyao.com,resources=kaepolicies/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=device.openfuyao.com,resources=kaepolicies/finalizers,verbs=update
// +kubebuilder:rbac:groups=rbac.authorization.k8s.io,resources=clusterroles;clusterrolebindings;roles;rolebindings,verbs=*
// +kubebuilder:rbac:groups="",resources=namespaces;serviceaccounts;pods;pods/eviction;services;services/finalizers;endpoints,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=persistentvolumeclaims;events;configmaps;secrets;nodes,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=apps,resources=deployments;daemonsets;replicasets;statefulsets,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=apps,resources=controllerrevisions,verbs=get;list;watch
// +kubebuilder:rbac:groups=monitoring.coreos.com,resources=servicemonitors;prometheusrules,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=scheduling.k8s.io,resources=priorityclasses,verbs=get;list;watch;create
// +kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;list;watch;create;update;patch
// +kubebuilder:rbac:groups=node.k8s.io,resources=runtimeclasses,verbs=get;list;create;update;watch;delete
// +kubebuilder:rbac:groups=apiextensions.k8s.io,resources=customresourcedefinitions,verbs=get;list;watch

// Reconcile 是 kubernetes 主要的调谐循环的一部分，其目标是使集群的当前状态逐步接近期望状态。
func (r *KAEPolicyReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	_ = log.FromContext(ctx)

	// 获取实例
	instance := &kaev1.KAEPolicy{}
	if err := r.Client.Get(ctx, req.NamespacedName, instance); err != nil {
		return r.handlerGetInstanceErr(ctx, err, instance)
	}

	// 已经存在实例了，则不处理新的实例
	if kaePolicyManager.kaeInstance != nil && kaePolicyManager.kaeInstance.ObjectMeta.Name != instance.ObjectMeta.Name {
		return r.handleInstanceExists(ctx, instance)
	}

	// 初始化KAEPolicyManager
	if err := kaePolicyManager.init(ctx, r, instance); err != nil {
		zlog.Errorf("failed to initialize KAEPolicy manager: %v", err)
		if condErr := setConditionsError(r.Client, ctx, instance, reconcileFailedReason, err.Error()); condErr != nil {
			zlog.Errorf("set Conditions Error: %v", condErr)
		}
		return ctrl.Result{}, err
	}
	if !kaePolicyManager.hasNFDLabels {
		zlog.Warn("WARNING: No NFD labels in the cluster, KAE nodes cannot be discovered.")
	}
	if !kaePolicyManager.hasKAENodes {
		zlog.Warn("WARNING: No KAE node found, watching new nodes joining the cluster.")
	}

	// 没有KAE节点也需要继续，因为这不是第一次，可能是节点标签变化导致的
	if overallStatus, statesNotReady, err := r.deployAllComponents(ctx, req, instance); err != nil {
		return ctrl.Result{RequeueAfter: requeueAfterTime}, err
	} else {
		return r.updateStateAndConditions(ctx, req, overallStatus, statesNotReady, instance)
	}
}

func (r *KAEPolicyReconciler) updateStateAndConditions(ctx context.Context, req ctrl.Request, overallStatus kaev1.State,
	statesNotReady []string, instance *kaev1.KAEPolicy) (ctrl.Result, error) {
	// 若状态未就绪，等待五秒
	if overallStatus != kaev1.Ready {
		zlog.Errorf("KAEPolicy state is not ready: %v", statesNotReady)
		updateKAEPolicyState(ctx, r, req.NamespacedName, kaev1.NotReady)
		if err := setConditionsError(
			r.Client, ctx, instance, operandNotReadyReason,
			fmt.Sprintf("KAEPolicy state is not ready: %v", statesNotReady),
		); err != nil {
			zlog.Infof("setConditionsError: %v", err)
		}
		return ctrl.Result{RequeueAfter: requeueAfterTime}, nil
	}

	// 待全部状态为就绪后更新 CR
	updateKAEPolicyState(ctx, r, req.NamespacedName, kaev1.Ready)
	if !kaePolicyManager.hasKAENodes {
		if !kaePolicyManager.hasNFDLabels {
			if err := setConditionsReady(
				r.Client, ctx, instance, noNFDLabelsReason,
				"NFD labels missing in the cluster, KAE nodes cannot be discovered.",
			); err != nil {
				return ctrl.Result{}, err
			}
		} else {
			if err := setConditionsReady(
				r.Client, ctx, instance, noKAENodeReason,
				"There is no KAE node in the cluster, waiting for new nodes to join.",
			); err != nil {
				return ctrl.Result{}, err
			}
		}
	} else {
		if err := setConditionsReady(
			r.Client, ctx, instance, reconciledReason,
			"KAEPolicy ready: all resources reconciled",
		); err != nil {
			return ctrl.Result{}, err
		}
	}

	return ctrl.Result{}, nil
}

func (r *KAEPolicyReconciler) handlerGetInstanceErr(ctx context.Context, err error,
	instance *kaev1.KAEPolicy) (ctrl.Result, error) {
	zlog.Warn(err.Error())
	if errors.IsNotFound(err) {
		zlog.Infof("KaePolicy not found,  the instance is deleted and returned without processing")
		return reconcile.Result{}, nil
	}
	// 读取对象错误，需要重新放入队列中处理
	setCondError := setConditionsError(r.Client, ctx, instance, reconcileFailedReason, err.Error())
	if setCondError != nil {
		zlog.Debugf("set Conditions Error: %v", setCondError)
	}
	return reconcile.Result{}, err
}

func (r *KAEPolicyReconciler) handleInstanceExists(ctx context.Context, instance *kaev1.KAEPolicy) (ctrl.Result,
	error) {
	utils.SetStatus(instance, kaev1.Ignored, kaePolicyManager.namespace)
	zlog.Infof("There is already a master CR instance: %s. "+
		"Please modify the instance instead of creating instance: %s",
		kaePolicyManager.kaeInstance.ObjectMeta.Name, instance.ObjectMeta.Name)
	if err := r.Client.Status().Update(ctx, instance); err != nil {
		zlog.Error("failed to update KAEPolicy status")
	}
	return ctrl.Result{}, nil
}

func (r *KAEPolicyReconciler) deployAllComponents(ctx context.Context, req ctrl.Request,
	instance *kaev1.KAEPolicy) (kaev1.State, []string, error) {
	overallStatus := kaev1.Ready
	var componentsNotReady []string
	for {
		status, statusError := kaePolicyManager.deployComponent()
		if statusError != nil {
			zlog.Infof("KAEPolicy deploy component. err: %v", statusError)
			updateKAEPolicyState(ctx, r, req.NamespacedName, kaev1.NotReady)
			errMsg := fmt.Sprintf("Failed to reconcile %s: %v", getComponentName(kaePolicyManager.index),
				statusError)
			if err := setConditionsError(r.Client, ctx, instance, reconcileFailedReason, errMsg); err != nil {
				zlog.Debugf("set Conditions Error: %v", err)
			}
			return kaev1.NotReady, componentsNotReady, statusError
		}

		if status == kaev1.NotReady {
			overallStatus = kaev1.NotReady
			componentsNotReady = append(componentsNotReady, getComponentName(kaePolicyManager.index-1))
		}
		zlog.Infof("KAEPolicy deploy component: %s completed. state: %s",
			getComponentName(kaePolicyManager.index-1), status)

		if kaePolicyManager.deployComplete() {
			break
		}
	}
	return overallStatus, componentsNotReady, nil
}

func getComponentName(index int) string {
	if index >= len(kaePolicyManager.componentNames) || index < 0 {
		return "outbounds"
	}
	return kaePolicyManager.componentNames[index]
}

// SetupWithManager 使用管理器设置控制器。
func (r *KAEPolicyReconciler) SetupWithManager(manager ctrl.Manager) error {
	// 创建一个控制器
	rateLimiter := workqueue.NewItemExponentialFailureRateLimiter(delayMin, delayMax)
	opts := controller.Options{
		Reconciler:              r,
		MaxConcurrentReconciles: maxConcurrentReconciles,
		RateLimiter:             rateLimiter,
	}
	newController, err := controller.New("kaepolicy-controller", manager, opts)
	if err != nil {
		return err
	}
	if err = r.setupWatch(newController, manager); err != nil {
		return err
	}
	// 添加一个索引键，它允许我们的调解程序快速查找它拥有的 DaemonSet。
	if err = r.setupDaemonSetIndexer(manager); err != nil {
		return err
	}
	return nil
}

func (r *KAEPolicyReconciler) setupWatch(controller controller.Controller, manager ctrl.Manager) error {
	// 监听 KAEPoliy 资源变化
	if err := r.watchKAEPolicy(controller, manager); err != nil {
		return err
	}
	// 监听所有节点上有关 KAE 标签的变化
	if err := r.watchKAENodeLabels(controller, manager); err != nil {
		return err
	}
	// 监视资源守护进程的更改
	if err := r.watchKAEDaemonSet(controller, manager); err != nil {
		return err
	}
	return nil
}

func (r *KAEPolicyReconciler) watchKAEPolicy(c controller.Controller, manager ctrl.Manager) error {
	watchSource := source.Kind(manager.GetCache(), &kaev1.KAEPolicy{})
	return c.Watch(
		watchSource,
		&handler.EnqueueRequestForObject{},
		predicate.GenerationChangedPredicate{},
	)
}

func (r *KAEPolicyReconciler) watchKAENodeLabels(c controller.Controller, manager ctrl.Manager) error {
	watchSource := source.Kind(manager.GetCache(), &corev1.Node{})
	eventHandler := handler.EnqueueRequestsFromMapFunc(r.mapToKAEPolicyRequests)
	return c.Watch(
		watchSource,
		eventHandler,
		predicate.Funcs{
			CreateFunc: predicateCreate,
			UpdateFunc: predicateUpdate,
			DeleteFunc: predicateDelete,
		},
	)
}

func (r *KAEPolicyReconciler) mapToKAEPolicyRequests(ctx context.Context, a client.Object) []reconcile.Request {
	var requests []reconcile.Request
	kaePolicyList := &kaev1.KAEPolicyList{}
	if err := r.Client.List(ctx, kaePolicyList); err != nil {
		zlog.Errorf("Unable to list KAEPolicies: %v", err)
		return requests
	}

	for _, kaePolicy := range kaePolicyList.Items {
		namespacedName := types.NamespacedName{
			Name:      kaePolicy.ObjectMeta.GetName(),
			Namespace: kaePolicy.ObjectMeta.GetNamespace(),
		}
		requests = append(requests, reconcile.Request{NamespacedName: namespacedName})
	}
	return requests
}

func (r *KAEPolicyReconciler) watchKAEDaemonSet(c controller.Controller, manager ctrl.Manager) error {
	watchDaemonSetSource := source.Kind(manager.GetCache(), &appsv1.DaemonSet{})
	eventhandler := handler.EnqueueRequestForOwner(
		manager.GetScheme(),
		manager.GetRESTMapper(),
		&kaev1.KAEPolicy{},
		handler.OnlyControllerOwner(),
	)
	return c.Watch(
		watchDaemonSetSource,
		eventhandler,
	)
}

func predicateCreate(e event.CreateEvent) bool {
	return hasKAELabels(e.Object.GetLabels(), true, true, true)
}

func predicateUpdate(e event.UpdateEvent) bool {
	if hasLabelChanged(e.ObjectOld.GetLabels(), e.ObjectNew.GetLabels()) {
		zlog.Infof("Node: %s watch labels has change", e.ObjectNew.GetName())
		return true
	}
	return false
}

func predicateDelete(e event.DeleteEvent) bool {
	return hasKAELabels(e.Object.GetLabels(), true, true, true)
}

func (r *KAEPolicyReconciler) setupDaemonSetIndexer(manager ctrl.Manager) error {
	extractValueFunc := func(rawObj client.Object) []string {
		ds, ok := rawObj.(*appsv1.DaemonSet)
		if !ok {
			return nil
		}
		owner := metav1.GetControllerOf(ds)
		if owner == nil {
			return nil
		}
		if owner.APIVersion != kaev1.GroupVersion.String() || owner.Kind != "KAEPolicy" {
			return nil
		}
		return []string{owner.Name}
	}
	err := manager.GetFieldIndexer().IndexField(
		context.Background(),
		&appsv1.DaemonSet{},
		kaePolicyControllerIndexKey,
		extractValueFunc,
	)
	if err != nil {
		return fmt.Errorf("failed to add index key: %w", err)
	}
	return nil
}
