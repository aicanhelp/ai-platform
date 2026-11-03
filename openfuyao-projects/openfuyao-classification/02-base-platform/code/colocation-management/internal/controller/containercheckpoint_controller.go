/*
Copyright 2024.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package controller contains functions for controllers to do actions for the ContainerCheckpoint
package controller

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"openfuyao.com/colocation-management/pkg/apis/v1"
)

// ContainerCheckpointReconciler reconciles a ContainerCheckpoint object
type ContainerCheckpointReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=colocation.openfuyao.com,resources=containercheckpoints,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=colocation.openfuyao.com,resources=containercheckpoints/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=colocation.openfuyao.com,resources=containercheckpoints/finalizers,verbs=update

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
func (r *ContainerCheckpointReconciler) Reconcile(ctx context.Context,
	req controllerruntime.Request) (controllerruntime.Result, error) {
	_ = log.FromContext(ctx)

	return controllerruntime.Result{}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *ContainerCheckpointReconciler) SetupWithManager(mgr controllerruntime.Manager) error {
	return controllerruntime.NewControllerManagedBy(mgr).
		For(&v1.ContainerCheckpoint{}).
		Complete(r)
}
