// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

// Package controller contains code for controllers
package controller

import (
	"context"
	"fmt"
	"os"
	"reflect"
	"time"

	"golang.org/x/time/rate"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/workqueue"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	v1 "openfuyao.com/npu-operator/api/v1"
)

const (
	perItemMinDelay           = time.Millisecond * 50
	perItemMaxDelay           = time.Second * 3
	overallQps                = rate.Limit(100)
	overallBurst              = 1000
	notReadyRequeueInterval   = time.Second * 5
	noNFDLabelRequeueInterval = time.Second * 30
	nsEnv                     = "OPERATOR_NAMESPACE"
	controllerIndexKey        = "metadata.openfuyao.npu.controller"
)

// NPUClusterPolicyReconciler reconciles a NPUClusterPolicy object
type NPUClusterPolicyReconciler struct {
	client.Client
	Scheme               *runtime.Scheme
	client               *kubernetes.Clientset
	namespace            string
	instance             *v1.NPUClusterPolicy
	components           []component
	runtimes             map[string]struct{}
	nodeRuntimeEndpoints map[string]string
	hasNFDLabels         bool
	hasNPUNodes          bool
}

// +kubebuilder:rbac:groups=npu.openfuyao.com,resources=npuclusterpolicies,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=npu.openfuyao.com,resources=npuclusterpolicies/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=npu.openfuyao.com,resources=npuclusterpolicies/finalizers,verbs=update
// +kubebuilder:rbac:groups=rbac.authorization.k8s.io,resources=clusterroles;clusterrolebindings;roles;rolebindings,verbs=*
// +kubebuilder:rbac:groups="",resources=namespaces;serviceaccounts;pods;pods/eviction,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=nodes;nodes/proxy,verbs=get;list;watch
// +kubebuilder:rbac:groups=apps,resources=daemonsets;deployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=apps,resources=controllerrevisions,verbs=get;list;watch
// +kubebuilder:rbac:groups=scheduling.k8s.io,resources=priorityclasses,verbs=get;list;watch;create
// +kubebuilder:rbac:groups=node.k8s.io,resources=runtimeclasses,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=configmaps,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=apiextensions.k8s.io,resources=customresourcedefinitions,verbs=get;list;watch;create;update;patch;delete

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// Modify the Reconcile function to compare the state specified by
// the NPUClusterPolicy object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.19.0/pkg/reconcile
func (r *NPUClusterPolicyReconciler) Reconcile(ctx context.Context, req ctrl.Request) (result ctrl.Result, err error) {
	logger := log.FromContext(ctx, "reconciling", req.NamespacedName).
		WithName("controller").
		WithName("NPUClusterPolicy")
	ctx = log.IntoContext(ctx, logger)

	if err = r.getInstance(ctx, req.Name); err != nil {
		if errors.IsNotFound(err) {
			err = nil
		} else {
			ignoreError(r.setConditionsError(ctx, reconcileFailedReason, err.Error()))
		}
		return
	}

	if err = r.loadComponents(ctx); err != nil {
		err = fmt.Errorf("load components: %w", err)
		ignoreError(r.setConditionsError(ctx, reconcileFailedReason, err.Error()))
		return
	}

	if err = r.labelNPUNodes(ctx); err != nil {
		err = fmt.Errorf("label NPU nodes: %w", err)
		ignoreError(r.setConditionsError(ctx, reconcileFailedReason, err.Error()))
		return
	}
	if !r.hasNFDLabels {
		logger.V(-1).Info("NFD labels missing in the cluster, NPU nodes cannot be discovered")
	}
	if !r.hasNPUNodes {
		logger.V(-1).Info("No NPU nodes with supported CRI runtime can be found in the cluster")
	}

	conflicts, err := r.detectComponentConflicts(ctx)
	if err != nil {
		err = fmt.Errorf("detect component conflicts: %w", err)
		ignoreError(r.setConditionsError(ctx, reconcileFailedReason, err.Error()))
		return
	}
	if len(conflicts) > 0 {
		msg := formatComponentConflicts(conflicts)
		logger.V(-1).Info("Component node selector conflict detected", "message", msg)
		ignoreError(r.setConditionsError(ctx, componentConflictReason, msg))
		return ctrl.Result{}, nil
	}

	return r.reconcileComponents(ctx)
}

// SetupWithManager sets up the controller with the Manager.
func (r *NPUClusterPolicyReconciler) SetupWithManager(mgr ctrl.Manager) error {
	r.namespace = os.Getenv(nsEnv)
	if r.namespace == "" {
		return fmt.Errorf("failed to proceed: %s environment variable not set", nsEnv)
	}

	var err error
	if r.client, err = kubernetes.NewForConfig(mgr.GetConfig()); err != nil {
		return fmt.Errorf("create clientset: %w", err)
	}

	return ctrl.NewControllerManagedBy(mgr).
		WithOptions(controller.Options{
			RateLimiter: workqueue.NewMaxOfRateLimiter(
				&workqueue.BucketRateLimiter{Limiter: rate.NewLimiter(overallQps, overallBurst)},
				workqueue.NewItemExponentialFailureRateLimiter(perItemMinDelay, perItemMaxDelay),
			),
		}).
		For(&v1.NPUClusterPolicy{}, builder.WithPredicates(predicate.GenerationChangedPredicate{})).
		Owns(&appsv1.DaemonSet{}).
		Owns(&appsv1.Deployment{}).
		Owns(&corev1.ConfigMap{}).
		Watches(
			&corev1.Node{},
			handler.EnqueueRequestsFromMapFunc(r.getReferrerToNode),
			builder.WithPredicates(nodeEventFilter),
		).
		Complete(r)
}

func (r *NPUClusterPolicyReconciler) getInstance(
	ctx context.Context,
	name string,
) error {
	instance := &v1.NPUClusterPolicy{}
	if err := r.Get(ctx, types.NamespacedName{Name: name}, instance); err != nil {
		err = fmt.Errorf("Failed to get ClusterPolicy object: %w", err)
		log.FromContext(ctx).Error(nil, err.Error())
		return err
	}
	r.instance = instance
	return nil
}

func (r *NPUClusterPolicyReconciler) getReferrerToNode(ctx context.Context, obj client.Object) (reqs []ctrl.Request) {
	reqs = []ctrl.Request{}

	list := &v1.NPUClusterPolicyList{}
	if err := r.List(ctx, list); err != nil {
		log.FromContext(ctx).Error(err, "Failed to list NPUClusterPolicy")
		return
	}

	for _, instance := range list.Items {
		reqs = append(reqs, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(&instance)})
	}

	return
}

var nodeEventFilter = predicate.Funcs{
	CreateFunc: func(e event.CreateEvent) bool {
		return true
	},
	UpdateFunc: func(e event.UpdateEvent) bool {
		return !reflect.DeepEqual(objectLabels(e.ObjectOld), objectLabels(e.ObjectNew))
	},
	DeleteFunc: func(e event.DeleteEvent) bool {
		return true
	},
}

func objectLabels(obj client.Object) map[string]string {
	if obj == nil {
		return nil
	}
	return obj.GetLabels()
}
