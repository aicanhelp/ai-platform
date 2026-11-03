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

package statefulset

import (
	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
)

func fyInconsistentVolumeStatus(set *apps.StatefulSet, status *apps.StatefulSetStatus) bool {
	if utilfeature.DefaultFeatureGate.Enabled(features.FYStatefulSetVolumeExpansion) && set.Status.VolumeClaimTemplates == nil {
		return true
	}
	for _, template := range status.VolumeClaimTemplates {
		for _, statefulSetPVCTemplate := range set.Status.VolumeClaimTemplates {
			if template.TemplateName == statefulSetPVCTemplate.TemplateName && (template.FinishedReconciliationGeneration != statefulSetPVCTemplate.FinishedReconciliationGeneration ||
				template.ReadyReplicas != statefulSetPVCTemplate.ReadyReplicas) {
				return true
			}
		}
	}
	return false
}

func fyPVCStorageRequestsMatch(ssc *defaultStatefulSetControl, currentSet *apps.StatefulSet, pod *v1.Pod, logger klog.Logger) bool {
	for _, currentClaim := range currentSet.Spec.VolumeClaimTemplates {
		currentStorageRequest, updatedStorageRequest, result := getPersistentVolumeClaimStorageSize(ssc, currentSet, pod, logger, currentClaim)
		if result && currentStorageRequest.Cmp(updatedStorageRequest) != 0 {
			return false
		}
	}
	return true
}

func getPersistentVolumeClaimStorageSize(ssc *defaultStatefulSetControl, currentSet *apps.StatefulSet, pod *v1.Pod, logger klog.Logger, currentClaim v1.PersistentVolumeClaim) (resource.Quantity, resource.Quantity, bool) {
	ordinal := getOrdinal(pod)
	claimName := getPersistentVolumeClaimName(currentSet, &currentClaim, ordinal)
	pvc, err := ssc.podControl.objectMgr.GetClaim(currentSet.Namespace, claimName)
	if err != nil {
		logger.Error(err, "Get persistentVolumeClaim %s for pod %s failed", claimName, pod.Name)
		return resource.Quantity{}, resource.Quantity{}, false
	}
	currentStorageRequest := currentClaim.Spec.Resources.Requests[v1.ResourceStorage]
	updatedStorageRequest := pvc.Spec.Resources.Requests[v1.ResourceStorage]
	return currentStorageRequest, updatedStorageRequest, true
}

func fyAddAnnotationsWithExpansionFeatureEnable(claim *v1.PersistentVolumeClaim, statefulSetName string, templateName string) {
	if utilfeature.DefaultFeatureGate.Enabled(features.FYStatefulSetVolumeExpansion) {
		if claim.Annotations == nil {
			claim.Annotations = make(map[string]string)
		}
		claim.Annotations["fy.stateful-set.name"] = statefulSetName
		claim.Annotations["fy.pvc-template.name"] = templateName
	}
}
