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

// Package checkpoint
package checkpoint

import (
	"context"
	"fmt"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2"
	"sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"openfuyao.com/colocation-management/cmd/colocation-manager/apps"
	colocationv1 "openfuyao.com/colocation-management/pkg/apis/v1"
	"openfuyao.com/colocation-management/pkg/colocation-manager/aggregate"
	"openfuyao.com/colocation-management/pkg/common"
	"openfuyao.com/colocation-management/pkg/utils"
)

const (
	cleanupInitialAggregateStateMapTimeout = 24 * time.Hour
)

// LoaderWriter is a loader and writer for aggregate container state.
type LoaderWriter struct {
	ctx context.Context
	client.Client
	mgr                   controllerruntime.Manager
	clusterState          *aggregate.ClusterState
	useCheckPoint         bool
	saveCheckpointPeriod  time.Duration
	checkpointGCInterval  time.Duration
	parallelSaveNodeCount int
	namespace             string
}

// NewLoaderWriter creates a new LoaderWriter instance.
func NewLoaderWriter(ctx context.Context, mgr controllerruntime.Manager,
	config *apps.Configuration, clusterState *aggregate.ClusterState) *LoaderWriter {
	return &LoaderWriter{
		ctx:                   ctx,
		Client:                mgr.GetClient(),
		mgr:                   mgr,
		clusterState:          clusterState,
		useCheckPoint:         config.UseCheckpoint,
		saveCheckpointPeriod:  config.SaveCheckpointPeriod,
		checkpointGCInterval:  config.CheckpointGCInterval,
		parallelSaveNodeCount: config.ParallelSaveNodeCount,
		namespace:             config.Namespace,
	}
}

// InitFromCheckpoints loads aggregate container state from checkpoints.
func (lw *LoaderWriter) InitFromCheckpoints() {
	klog.V(common.VerboseDebugLog).Info("Initializing LoadAndWriter from checkpoints")
	startTime := time.Now()

	ckptList := &colocationv1.ContainerCheckpointList{}
	if err := lw.Client.List(lw.ctx, ckptList); err != nil {
		klog.ErrorS(err, "InitFromCheckpoints: list checkpoints failed.")
		return
	}

	for _, ckpt := range ckptList.Items {
		podID := aggregate.PodID{
			Namespace: ckpt.Spec.Namespace,
			PodName:   ckpt.Spec.PodName,
		}
		containerName := ckpt.Spec.ContainerName
		cs := aggregate.NewAggregateContainerState()
		if err := cs.LoadFromCheckpoint(&ckpt.Status); err != nil {
			klog.ErrorS(err, "Can not load this checkpoint data", "PodID", podID, "container", containerName)
			continue
		}

		lw.clusterState.AddOrUpdateInitialAggregateState(
			aggregate.MakeAggregateStateKey(podID, containerName),
			cs)
		lw.clusterState.AddOrUpdateCheckpoint(types.NamespacedName{
			Namespace: ckpt.Namespace,
			Name:      ckpt.Name,
		}, &ckpt)
	}
	klog.V(common.GeneralDebugLog).InfoS("Initialized aggregate state list from checkpoint done",
		"checkpoint count", lw.clusterState.InitialAggregateStateSize(), "elapsed", time.Since(startTime))
}

// Run starts the LoaderWriter.
func (lw *LoaderWriter) Run() {
	utils.WaitCacheSync()

	lw.InitFromCheckpoints()

	go func() {
		timerChan := time.After(cleanupInitialAggregateStateMapTimeout)

		select {
		case <-lw.ctx.Done():
		case <-timerChan:
			count := lw.clusterState.CleanupInitialAggregateState()
			klog.InfoS("Clean initialized aggregate states", "cleanup count", count)
		}
	}()

	if lw.useCheckPoint {
		go func() {
			ticker := time.NewTicker(lw.saveCheckpointPeriod)
			defer ticker.Stop()

			for {
				select {
				case <-lw.ctx.Done():
					return
				case startTime := <-ticker.C:
					lw.MaintainCheckpoints(startTime)
				}
			}
		}()

		go func() {
			ticker := time.NewTicker(lw.checkpointGCInterval)
			defer ticker.Stop()

			for {
				select {
				case <-lw.ctx.Done():
					return
				case startTime := <-ticker.C:
					lw.GarbageCollectCheckpoints(startTime)
				}
			}
		}()
	}
}

// MaintainCheckpoints saves aggregate container state to checkpoints.
func (lw *LoaderWriter) MaintainCheckpoints(startTime time.Time) {
	aggregateKeys, ckpts := lw.clusterState.SaveCheckpoints(lw.parallelSaveNodeCount)
	for i, ckpt := range ckpts {
		aggregateKey := aggregateKeys[i]
		cachedCkpt := lw.clusterState.GetCheckpoint(aggregateKey)
		if cachedCkpt != nil {
			cachedCkpt.Status = *ckpt
			if err := lw.Client.Status().Update(lw.ctx, cachedCkpt); err != nil {
				klog.ErrorS(err, "Update ContainerCheckpoint‘s Status failed.", "checkpointName", cachedCkpt.Name)
				continue
			}

		} else {
			checkpointName := fmt.Sprintf("%s--%s--%s",
				aggregateKey.Namespace(), aggregateKey.PodName(), aggregateKey.ContainerName())

			newCkpt := &colocationv1.ContainerCheckpoint{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: lw.namespace,
					Name:      checkpointName,
				},
				Spec: colocationv1.ContainerCheckpointSpec{
					Namespace:     aggregateKey.Namespace(),
					PodName:       aggregateKey.PodName(),
					ContainerName: aggregateKey.ContainerName(),
				},
				Status: *ckpt,
			}
			if err := lw.Client.Create(lw.ctx, newCkpt); err != nil {
				klog.ErrorS(err, "Create ContainerCheckpoint failed.", "aggregateKey", aggregateKey)
				continue
			}
		}
		klog.V(common.AdvanceDebugLog).Infof("Saved aggregate state keys are %+v", aggregateKey)
	}
	klog.V(common.GeneralDebugLog).InfoS("Save all checkpoints of aggregate container state done",
		"count", len(ckpts), "elapsed", time.Since(startTime))
	cs := lw.clusterState.GetAllNodeStates()
	klog.Infof("all node states: %v", cs)
}

// GarbageCollectCheckpoints deletes checkpoints that corresponding pod no longer exists.
func (lw *LoaderWriter) GarbageCollectCheckpoints(startTime time.Time) {
	var deletedCount int
	allCkpts := lw.clusterState.GetAllCheckpoint()
	for _, ckpt := range allCkpts {
		deletedCount += lw.processCheckpoint(ckpt)
	}
	klog.V(common.GeneralDebugLog).InfoS("Garbage Collect checkpoints of aggregate container state done",
		"deleted count", deletedCount, "elapsed", time.Since(startTime))
}

func (lw *LoaderWriter) processCheckpoint(ckpt *colocationv1.ContainerCheckpoint) int {
	pod := &corev1.Pod{}
	if err := lw.Get(lw.ctx, types.NamespacedName{
		Namespace: ckpt.Spec.Namespace,
		Name:      ckpt.Spec.PodName}, pod); err != nil {
		if errors.IsNotFound(err) {
			if err = lw.Delete(lw.ctx, ckpt); err != nil {
				klog.ErrorS(err, "Delete checkpoint that corresponding pod no long exists is failed.",
					"checkpoint", ckpt.Name)
				return 0
			} else {
				klog.V(common.GeneralDebugLog).InfoS("Deleted checkpoint that corresponding pod no long exists",
					"checkpoint", ckpt.Name)
				return 1
			}
		}

		klog.ErrorS(err, "Get pod from manager cache failed.", "pod namespace", ckpt.Spec.Namespace,
			"pod name", ckpt.Spec.PodName, "checkpoint", ckpt.Name)
		return 0
	}

	if !utils.ContributivePod(pod.Status.Phase) {
		if err := lw.Delete(lw.ctx, ckpt); err != nil {
			klog.ErrorS(err, "Delete checkpoint that corresponding pod is not contributive failed.",
				"checkpoint", ckpt.Name)
			return 0
		} else {

			klog.V(common.GeneralDebugLog).InfoS("Deleted checkpoint that corresponding pod is not contributive",
				"checkpoint", ckpt.Name)
			return 1
		}
	}
	return 0
}
