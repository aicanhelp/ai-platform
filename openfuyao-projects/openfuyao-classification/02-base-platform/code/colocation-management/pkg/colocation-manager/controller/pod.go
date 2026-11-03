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

package controller

import (
	"context"
	"fmt"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	"sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"openfuyao.com/colocation-management/pkg/colocation-manager/aggregate"
	"openfuyao.com/colocation-management/pkg/common"
	"openfuyao.com/colocation-management/pkg/utils"
)

// PodReconciler reconciles a Pod object
type PodReconciler struct {
	ctx context.Context
	client.Client
	mgr          controllerruntime.Manager
	Scheme       *runtime.Scheme
	Recorder     record.EventRecorder
	clusterState *aggregate.ClusterState

	refreshPodsInterval      time.Duration
	aggregateStateGCInterval time.Duration
}

// NewPodReconciler returns a new PodReconciler.
func NewPodReconciler(
	ctx context.Context,
	mgr controllerruntime.Manager,
	clusterState *aggregate.ClusterState,
	refreshPodsInterval time.Duration,
	aggregateStateGCInterval time.Duration,
) *PodReconciler {
	return &PodReconciler{
		ctx:                      ctx,
		mgr:                      mgr,
		Client:                   mgr.GetClient(),
		Scheme:                   mgr.GetScheme(),
		Recorder:                 mgr.GetEventRecorderFor("Pod"),
		clusterState:             clusterState,
		refreshPodsInterval:      refreshPodsInterval,
		aggregateStateGCInterval: aggregateStateGCInterval,
	}
}

// Reconcile is part of the main kubernetes reconciliation loop which aims to
func (r *PodReconciler) Reconcile(ctx context.Context,
	req controllerruntime.Request) (controllerruntime.Result, error) {
	klog.V(common.TraceDebugLog).Infof("PodReconciler: received request: %v", req.NamespacedName)

	pod := &v1.Pod{}
	if err := r.Get(ctx, req.NamespacedName, pod); err != nil {
		if errors.IsNotFound(err) {
			r.clusterState.DeletePod(aggregate.PodID{
				Namespace: req.Namespace,
				PodName:   req.Name,
			})
			klog.V(common.AdvanceDebugLog).Infof("PodReconciler: deleted a pod:%v from clusterState done", req.NamespacedName)
			return controllerruntime.Result{}, client.IgnoreNotFound(err)
		}
		klog.Errorf("PodReconciler: get pod:%v failed. %v", req.NamespacedName, err)

		return controllerruntime.Result{Requeue: true}, err
	}

	if utils.OfflinePod(pod) || !r.clusterState.IsNodeExisted(pod.Spec.NodeName) {
		return controllerruntime.Result{}, nil
	}

	if err := r.clusterState.AddPodAndContainer(pod); err != nil {
		klog.Errorf("PodReconciler: add or updated pod:%v and it's containers to clusterState failed. %v",
			req.NamespacedName, err)
	} else {
		klog.V(common.AdvanceDebugLog).Infof("PodReconciler: add or updated pod:%v and it's"+
			" containers to clusterState done", req.NamespacedName)
	}
	return controllerruntime.Result{}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *PodReconciler) SetupWithManager(mgr controllerruntime.Manager) error {
	return controllerruntime.NewControllerManagedBy(mgr).
		For(&v1.Pod{}).
		Complete(r)
}

// Run runs the controller.
func (r *PodReconciler) Run() error {
	if err := r.SetupWithManager(r.mgr); err != nil {
		klog.Fatalf("PodReconciler: SetupWithManager failed. %v", err)
		return err
	}

	go func() {
		utils.WaitCacheSync()

		startTime := time.Now()
		if err := r.doRefreshPods(); err == nil {
			klog.V(common.GeneralDebugLog).Infof("PodReconciler: refresh all pods done. elapsed: %v", time.Since(startTime))
		} else {
			klog.Errorf("PodReconciler: refresh all pods fail. %v", err)
		}

		ticker := time.NewTicker(r.refreshPodsInterval)
		defer ticker.Stop()

		for {
			select {
			case <-r.ctx.Done():
				return
			case startTime := <-ticker.C:
				if err := r.doRefreshPods(); err == nil {
					klog.V(common.GeneralDebugLog).Infof("PodReconciler: refresh all pods done. elapsed: %v", time.Since(startTime))
				} else {
					klog.Errorf("PodReconciler: refresh all pods fail. %v", err)
				}
			}
		}
	}()

	go func() {
		ticker := time.NewTicker(r.aggregateStateGCInterval)
		defer ticker.Stop()

		r.GCAggregateStatesCollection(time.Now())
		for {
			select {
			case <-r.ctx.Done():
				return
			case startTime := <-ticker.C:
				r.GCAggregateStatesCollection(startTime)
			}
		}
	}()

	return nil
}

// GCAggregateStatesCollection garbage collects aggregate states collection.
func (r *PodReconciler) GCAggregateStatesCollection(startTime time.Time) {
	nodeStates := r.clusterState.GetAllNodeStates()
	for _, nodeState := range nodeStates {
		nodeState.RateLimitedGarbageCollectAggregateCollectionStates(time.Now())
	}
	klog.V(common.VerboseDebugLog).Infof("Garbage collect aggregate state collection done. "+
		"elapsed: %v", time.Since(startTime))
}

func (r *PodReconciler) doRefreshPods() error {
	pods := &v1.PodList{}
	if err := r.List(r.ctx, pods); err != nil {
		return fmt.Errorf("list pods err:%v", err)
	}

	curPods := make(map[aggregate.PodID]*v1.Pod)
	for _, pod := range pods.Items {
		podCopy := pod.DeepCopy()
		if utils.OfflinePod(podCopy) || !r.clusterState.IsNodeExisted(podCopy.Spec.NodeName) {
			continue
		}

		podID := aggregate.PodID{
			Namespace: podCopy.Namespace,
			PodName:   podCopy.Name,
		}
		curPods[podID] = podCopy

		if err := r.clusterState.AddPodAndContainer(podCopy); err != nil {
			klog.Errorf("PodReconciler: add or updated pod:%v "+
				"and it's containers to clusterState failed. %v", podID, err)
		} else {
			klog.V(common.AdvanceDebugLog).Infof("PodReconciler: add or updated pod:%v"+
				" and it's containers to clusterState done", podID)
		}
	}
	r.clusterState.DeleteNotExistedPods(curPods)
	return nil
}
