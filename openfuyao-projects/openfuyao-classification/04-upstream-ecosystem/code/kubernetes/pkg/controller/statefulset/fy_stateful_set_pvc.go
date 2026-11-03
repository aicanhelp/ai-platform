/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Package statefulset
package statefulset

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	coreinformers "k8s.io/client-go/informers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
)

const (
	// StatefulSetInPVCAnnotations the key in the annotations of the PVC, can get statefulset
	StatefulSetInPVCAnnotations = "fy.stateful-set.name"

	// PVCTemplateInPVCAnnotations the key in the annotations of the PVC, can get pvc template
	PVCTemplateInPVCAnnotations = "fy.pvc-template.name"
)

// registerPVCInformerEvent
func (ssc *StatefulSetController) registerPVCInformerEvent(pvcInformer coreinformers.PersistentVolumeClaimInformer) {
	ssc.pvcQueue = workqueue.NewRateLimitingQueueWithConfig(workqueue.DefaultControllerRateLimiter(), workqueue.RateLimitingQueueConfig{Name: "statefulset_pvc"})
	ssc.pvcLister = pvcInformer.Lister()
	ssc.pvcsSynced = pvcInformer.Informer().HasSynced

	_, err := pvcInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		// lookup current and old statefulset if pvcs changed
		UpdateFunc: func(old, cur interface{}) {
			ssc.enqueuePVC(cur)
		},
	})
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("register PVC informer event failed: %v", err))
	}
}

// pvcWorker runs a pvcWorker goroutine that invokes processNextPVCWorkItem until the controller's pvcQueue is closed
func (ssc *StatefulSetController) pvcWorker(ctx context.Context) {
	for ssc.processNextPVCWorkItem(ctx) {
	}
}

// processNextPVCWorkItem
func (ssc *StatefulSetController) processNextPVCWorkItem(ctx context.Context) bool {
	key, shutdown := ssc.pvcQueue.Get()
	if shutdown {
		return false
	}
	defer ssc.pvcQueue.Done(key)

	err := ssc.syncPVC(ctx, key.(string))
	if err == nil {
		ssc.pvcQueue.Forget(key)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("%v failed with : %v", key, err))
	ssc.pvcQueue.AddRateLimited(key)

	return true
}

// enqueuePVC
func (ssc *StatefulSetController) enqueuePVC(obj interface{}) {
	pvc, ok := obj.(*v1.PersistentVolumeClaim)
	if !ok {
		return
	}

	if pvc.Status.Phase == v1.ClaimBound {
		key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(pvc)
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("couldn't get key for object %#v: %v", pvc, err))
			return
		}
		ssc.pvcQueue.Add(key)
	}
}

// sync syncs the given statefulset.
func (ssc *StatefulSetController) syncPVC(ctx context.Context, key string) error {
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}
	pvc, err := ssc.pvcLister.PersistentVolumeClaims(namespace).Get(name)
	if errors.IsNotFound(err) {
		return nil
	}
	logger := klog.FromContext(ctx)
	if err != nil {
		logger.V(2).Info("Error getting PVC from informer", "pvcKey", key, "err", err)
		return err
	}

	return ssc.updatePVCTemplateStatus(pvc)
}

// updatePVCTemplateStatus
func (ssc *StatefulSetController) updatePVCTemplateStatus(pvc *v1.PersistentVolumeClaim) error {
	stsName, ok := pvc.Annotations[StatefulSetInPVCAnnotations]
	if !ok || len(stsName) == 0 {
		return nil
	}
	pvcTemplateName, ok := pvc.Annotations[PVCTemplateInPVCAnnotations]
	if !ok || len(pvcTemplateName) == 0 {
		return nil
	}

	sts, err := ssc.setLister.StatefulSets(pvc.Namespace).Get(stsName)
	if err != nil || sts == nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get statefulset from the key in the annotations of the PVC: %s", stsName))
		return nil
	}

	// changed whether the pvc status template is updated, avoid unnecessary requests.
	changed := false
	for i, pvcTemplateStatus := range sts.Status.VolumeClaimTemplates {
		if pvcTemplateName == pvcTemplateStatus.TemplateName && pvcTemplateStatus.FinishedReconciliationGeneration != sts.Generation {
			if pvc.Spec.Resources.Requests.Storage().Cmp(pvc.Status.Capacity[v1.ResourceStorage]) == 0 {
				pvcTemplateStatus.ReadyReplicas++
				changed = true
			}
			if pvcTemplateStatus.ReadyReplicas == *sts.Spec.Replicas {
				pvcTemplateStatus.FinishedReconciliationGeneration = sts.Generation
				changed = true
			}
			sts.Status.VolumeClaimTemplates[i] = pvcTemplateStatus
		}
	}
	if !changed {
		return nil
	}

	_, updateErr := ssc.kubeClient.AppsV1().StatefulSets(sts.Namespace).UpdateStatus(context.TODO(), sts, metav1.UpdateOptions{})

	return updateErr
}
