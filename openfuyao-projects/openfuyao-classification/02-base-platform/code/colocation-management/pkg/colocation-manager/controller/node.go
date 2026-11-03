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

// NodeReconciler reconciles a Node object
type NodeReconciler struct {
	ctx context.Context
	mgr controllerruntime.Manager
	client.Client
	Scheme       *runtime.Scheme
	Recorder     record.EventRecorder
	clusterState *aggregate.ClusterState

	refreshNodesInterval time.Duration
}

// NewNodeReconciler returns a new NodeReconciler
func NewNodeReconciler(
	ctx context.Context,
	mgr controllerruntime.Manager,
	clusterState *aggregate.ClusterState,
	refreshNodesInterval time.Duration,
) *NodeReconciler {
	return &NodeReconciler{
		ctx:                  ctx,
		mgr:                  mgr,
		Client:               mgr.GetClient(),
		Scheme:               mgr.GetScheme(),
		Recorder:             mgr.GetEventRecorderFor("Node"),
		clusterState:         clusterState,
		refreshNodesInterval: refreshNodesInterval,
	}
}

// Reconcile is part of the main kubernetes reconciliation loop
func (r *NodeReconciler) Reconcile(ctx context.Context,
	req controllerruntime.Request) (controllerruntime.Result, error) {
	klog.V(common.TraceDebugLog).Infof("NodeReconciler: received request: %v", req.NamespacedName)

	node := &v1.Node{}
	if err := r.Get(ctx, req.NamespacedName, node); err != nil {
		if errors.IsNotFound(err) {
			r.clusterState.DeleteNode(req.Name)
			klog.V(common.AdvanceDebugLog).Infof("NodeReconciler: deleted a node:%v from clusteState done", req.NamespacedName)
			return controllerruntime.Result{}, nil
		}
		klog.Errorf("NodeReconciler: get node:%v failed. %v", req.NamespacedName, err)
		return controllerruntime.Result{Requeue: true}, err
	}

	if !utils.IsColocationNode(node) {
		r.clusterState.DeleteNode(node.Name)
		return controllerruntime.Result{}, nil
	}
	r.clusterState.AddOrUpdateNode(node)
	klog.V(common.AdvanceDebugLog).Infof("NodeReconciler: add a node:%v to clusterState done", req.NamespacedName)

	return controllerruntime.Result{}, nil
}

// Run starts the controller
func (r *NodeReconciler) Run() error {
	if err := r.SetupWithManager(r.mgr); err != nil {
		klog.Fatalf("NodeReconciler: SetupWithManager failed. %v", err)
		return err
	}

	// 额外开启轮询更新node状态
	go func() {
		utils.WaitCacheSync()

		startTime := time.Now()
		if err := r.doRefreshNode(); err == nil {
			klog.V(common.GeneralDebugLog).Infof("NodeReconciler: refresh all nodes done. elapsed: %v", time.Since(startTime))
		} else {
			klog.Errorf("NodeReconciler: refresh all nodes fail. %v", err)
		}

		ticker := time.NewTicker(r.refreshNodesInterval)
		defer ticker.Stop()
		for {
			select {
			case <-r.ctx.Done():
				return
			case startTime = <-ticker.C:
				if err := r.doRefreshNode(); err == nil {
					klog.V(common.GeneralDebugLog).Infof("NodeReconciler: refresh all nodes done. elapsed: %v", time.Since(startTime))
				} else {
					klog.Errorf("NodeReconciler: refresh all nodes fail. %v", err)
				}
			}
		}
	}()

	return nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *NodeReconciler) SetupWithManager(mgr controllerruntime.Manager) error {
	return controllerruntime.NewControllerManagedBy(mgr).
		For(&v1.Node{}).
		Complete(r)
}

func (r *NodeReconciler) doRefreshNode() error {
	nodes := &v1.NodeList{}
	if err := r.List(r.ctx, nodes); err != nil {
		klog.Errorf("NodeReconciler: list nodes failed. %v", err)
		return err
	}

	nodeMap := make(map[string]*v1.Node)
	for _, node := range nodes.Items {
		nodeCopy := node.DeepCopy()
		if !utils.IsColocationNode(nodeCopy) {
			r.clusterState.DeleteNode(nodeCopy.Name)
			continue
		}
		nodeMap[node.Name] = nodeCopy
		r.clusterState.AddOrUpdateNode(nodeCopy)
		klog.V(common.AdvanceDebugLog).Infof("add a node:%v listed to clusterState done", nodeCopy.Name)
	}
	r.clusterState.DeleteNotExistedNodes(nodeMap)
	return nil
}
