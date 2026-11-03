/*
 *
 *  * Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
 *  * openFuyao is licensed under Mulan PSL v2.
 *  * You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  * You may obtain a copy of Mulan PSL v2 at:
 *  *          http://license.coscl.org.cn/MulanPSL2
 *  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *  * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *  * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *  * See the Mulan PSL v2 for more details.
 *
 */

// Package controller contains the controller implementation for the ContainerCheckpoint resource.
package controller

import (
	"context"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	"sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"openfuyao.com/colocation-management/pkg/apis/v1"
	"openfuyao.com/colocation-management/pkg/colocation-manager/aggregate"
)

// ContainerCheckpointReconciler reconciles a ContainerCheckpoint object
type ContainerCheckpointReconciler struct {
	ctx context.Context
	mgr controllerruntime.Manager
	client.Client
	Scheme       *runtime.Scheme
	Recorder     record.EventRecorder
	clusterState *aggregate.ClusterState
}

// NewContainerCheckpointReconciler returns a new ContainerCheckpointReconciler instance.
func NewContainerCheckpointReconciler(ctx context.Context, mgr controllerruntime.Manager,
	clusterState *aggregate.ClusterState) *ContainerCheckpointReconciler {
	return &ContainerCheckpointReconciler{
		ctx:          ctx,
		mgr:          mgr,
		Client:       mgr.GetClient(),
		Scheme:       mgr.GetScheme(),
		Recorder:     mgr.GetEventRecorderFor("ContainerCheckpoint"),
		clusterState: clusterState,
	}
}

// +kubebuilder:rbac:groups=colocation.openfuyao.com,resources=containercheckpoints,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=colocation.openfuyao.com,resources=containercheckpoints/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=colocation.openfuyao.com,resources=containercheckpoints/finalizers,verbs=update

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
func (r *ContainerCheckpointReconciler) Reconcile(ctx context.Context,
	req controllerruntime.Request) (controllerruntime.Result, error) {
	ckpt := &v1.ContainerCheckpoint{}
	if err := r.Client.Get(ctx, req.NamespacedName, ckpt); err != nil {
		if errors.IsNotFound(err) {
			r.clusterState.DeleteCheckpoint(req.NamespacedName)
			return controllerruntime.Result{}, nil
		}
		return controllerruntime.Result{Requeue: true}, err
	}

	r.clusterState.AddOrUpdateCheckpoint(req.NamespacedName, ckpt)

	return controllerruntime.Result{}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *ContainerCheckpointReconciler) SetupWithManager() error {
	return controllerruntime.NewControllerManagedBy(r.mgr).
		For(&v1.ContainerCheckpoint{}).
		Complete(r)
}
