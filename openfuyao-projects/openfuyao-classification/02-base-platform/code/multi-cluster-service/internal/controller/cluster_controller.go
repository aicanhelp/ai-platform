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

// Package controller contains the reconcile func for cluster object.
package controller

import (
	"context"
	"time"

	"github.com/karmada-io/karmada/pkg/generated/clientset/versioned"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/util/retry"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"openfuyao.com/multi-cluster-service/api/v1beta1"
	"openfuyao.com/multi-cluster-service/pkg/cluster"
	"openfuyao.com/multi-cluster-service/pkg/zlog"
)

// ClusterReconciler reconciles a Cluster object
type ClusterReconciler struct {
	client.Client
	Scheme        *runtime.Scheme
	KarmadaClient *versioned.Clientset
}

const (
	defaultOpenFuyaoFinalizers = "finalizers.multicluster.openfuyao.com"
	defaultUpdateFrequency     = 30 * time.Second
)

// +kubebuilder:rbac:groups=multicluster.openfuyao.com,resources=clusters,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=multicluster.openfuyao.com,resources=clusters/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=multicluster.openfuyao.com,resources=clusters/finalizers,verbs=update
// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// Modify the Reconcile function to compare the state specified by
// the Cluster object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
func (r *ClusterReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	_ = log.FromContext(ctx)

	clusterObj := &v1beta1.Cluster{}
	if err := r.Get(ctx, req.NamespacedName, clusterObj); err != nil {
		if errors.IsNotFound(err) {
			zlog.Warnf("NotFound cluster object to reconcile.")
			return ctrl.Result{}, client.IgnoreNotFound(err)
		}
		zlog.LogErrorf("Error Retrieving cluster object : %v", err)
		return ctrl.Result{}, err
	}

	return r.syncClusterStatus(ctx, clusterObj)
}

// SetupWithManager sets up the controller with the Manager.
func (r *ClusterReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&v1beta1.Cluster{}).
		Complete(r)
}

func (r *ClusterReconciler) syncClusterStatus(ctx context.Context, obj *v1beta1.Cluster) (ctrl.Result, error) {
	currentClusterStatus := *obj.Status.DeepCopy()

	// sync from karmada cluster object
	cpuAndMemoryUsage, currentCondition, err := cluster.RetrieveClusterStatus(obj.Name, r.KarmadaClient)
	if err != nil {
		if err.Error() == "cluster object is not ready" {
			return ctrl.Result{}, err
		}
		zlog.LogErrorf("Error Retrieving cluster object resources: %v", err)
		return ctrl.Result{}, err
	}

	zlog.LogInfof("cluster %s is ready, synchronization of object status can proceed.\n", obj.Name)
	currentClusterStatus.ResourceList.AllocatableCPU = cpuAndMemoryUsage.AllocatableCPU
	currentClusterStatus.ResourceList.AllocatableMemory = cpuAndMemoryUsage.AllocatableMemory
	currentClusterStatus.ResourceList.AllocatedCPU = cpuAndMemoryUsage.AllocatedCPU
	currentClusterStatus.ResourceList.AllocatedMemory = cpuAndMemoryUsage.AllocatedMemory

	currentClusterStatus.Condition = nil
	currentClusterStatus.Condition = currentCondition

	return r.updateStatus(ctx, obj, currentClusterStatus)
}

func (r *ClusterReconciler) updateStatus(ctx context.Context, obj *v1beta1.Cluster,
	newStatus v1beta1.ClusterStatus) (ctrl.Result, error) {
	// retry avoid conflict update
	err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		clusterObj := v1beta1.Cluster{}
		if err := r.Get(ctx, client.ObjectKeyFromObject(obj), &clusterObj); err != nil {
			zlog.LogErrorf("Error Retrieving cluster object : %v", err)
			return err
		}

		clusterObj.Status.Condition = nil
		clusterObj.Status.Condition = newStatus.Condition

		clusterObj.Status.ResourceList.AllocatableCPU = newStatus.ResourceList.AllocatableCPU
		clusterObj.Status.ResourceList.AllocatableMemory = newStatus.ResourceList.AllocatableMemory
		clusterObj.Status.ResourceList.AllocatedCPU = newStatus.ResourceList.AllocatedCPU
		clusterObj.Status.ResourceList.AllocatedMemory = newStatus.ResourceList.AllocatedMemory

		return r.Status().Update(ctx, &clusterObj)
	})

	if err != nil {
		zlog.LogErrorf("Error Updating cluster status : %v", err)
		return ctrl.Result{}, nil
	}

	return ctrl.Result{RequeueAfter: defaultUpdateFrequency}, nil
}
