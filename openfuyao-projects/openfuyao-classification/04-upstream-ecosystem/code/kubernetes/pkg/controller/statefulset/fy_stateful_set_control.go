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
	"fmt"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
)

func (ssc *defaultStatefulSetControl) fyUpdateVolumeClaimTemplatesStatus(pods []*v1.Pod, set *apps.StatefulSet, status *apps.StatefulSetStatus) {
	status.VolumeClaimTemplates = []apps.VolumeClaimTemplate{}
	for _, setPersistentVolumeClaimTemplate := range set.Spec.VolumeClaimTemplates {
		var readyReplicas int32 = 0
		finishedReconciliationGeneration := set.Generation - 1
		for _, pod := range pods {
			ordinal := getOrdinal(pod)
			claimName := getPersistentVolumeClaimName(set, &setPersistentVolumeClaimTemplate, ordinal)
			persistentVolumeClaim, err := ssc.podControl.objectMgr.GetClaim(set.Namespace, claimName)
			if err != nil {
				klog.Errorf("get persistentVolumeClaim %s from pod %s failed %v", claimName, pod.Name, err)
				continue
			}
			if setPersistentVolumeClaimTemplate.Spec.Resources.Requests.Storage().Cmp(*persistentVolumeClaim.Status.Capacity.Storage()) == 0 {
				readyReplicas++
			}
		}
		if readyReplicas == *set.Spec.Replicas {
			finishedReconciliationGeneration = set.Generation
		}
		template := &apps.VolumeClaimTemplate{
			TemplateName:                     setPersistentVolumeClaimTemplate.Name,
			ReadyReplicas:                    readyReplicas,
			FinishedReconciliationGeneration: finishedReconciliationGeneration,
		}
		status.VolumeClaimTemplates = append(status.VolumeClaimTemplates, *template)
	}
}

func (ssc *defaultStatefulSetControl) fyUpdatePersistentVolumeClaims(enableFYStatefulSetVolumeExpansion bool, set *apps.StatefulSet, pod *v1.Pod) (bool, error) {
	if !enableFYStatefulSetVolumeExpansion {
		return false, nil
	}
	for _, persistentVolumeClaimTemplate := range set.Spec.VolumeClaimTemplates {
		ordinal := getOrdinal(pod)
		claimName := getPersistentVolumeClaimName(set, &persistentVolumeClaimTemplate, ordinal)
		persistentVolumeClaim, err := ssc.podControl.objectMgr.GetClaim(set.Namespace, claimName)
		if err != nil {
			ssc.recorder.Eventf(set, v1.EventTypeWarning, "FailedToResizePVC",
				"StatefulSet %s failed to resize PVC %s of pod %s : %v", set.Name, claimName, pod.Name, err)
			return true, fmt.Errorf("StatefulSet %s failed to resize PVC %s of pod %s: %v",
				set.Name, claimName, pod.Name, err)
		}

		if persistentVolumeClaim.Status.Phase != v1.ClaimBound {
			ssc.recorder.Eventf(set, v1.EventTypeWarning, "FailedToResizePVC",
				"StatefulSet %s failed to resize PVC %s of pod %s : %v", set.Name, claimName, pod.Name, err)
			return true, fmt.Errorf("StatefulSet %s failed to resize PVC %s of pod %s : Because PVC status is %s",
				set.Name, claimName, pod.Name, persistentVolumeClaim.Status.Phase)
		}

		currentStorageRequestSize := persistentVolumeClaim.Spec.Resources.Requests.Storage()
		updatedStorageRequestSize := persistentVolumeClaimTemplate.Spec.Resources.Requests.Storage()
		capacitySize := persistentVolumeClaim.Status.Capacity.Storage()

		if updatedStorageRequestSize.Cmp(*currentStorageRequestSize) == 0 {
			continue
		} else if updatedStorageRequestSize.Cmp(*capacitySize) <= 0 {
			ssc.recorder.Eventf(set, v1.EventTypeWarning, "FailedToPatchPVC",
				"StatefulSet %s failed to resize PVC %s of pod %s : PersistentVolumeClaim %s is invalid: spec.resources.requests.storage: Forbidden: field can not be less than previous value",
				set.Name, claimName, pod.Name, claimName)
			return true, fmt.Errorf("StatefulSet %s failed to patch PVC %s of pod %s : PersistentVolumeClaim %s is invalid: spec.resources.requests.storage: Forbidden: field can not be less than previous value",
				set.Name, claimName, pod.Name, claimName)
		}

		patch := fmt.Sprintf(`{"metadata":{"annotations":{"fy.stateful-set.name":"%s", "fy.pvc-template.name":"%s"}},"spec":{"resources":{"requests":{"storage":"%s"}}}}`,
			set.Name, persistentVolumeClaim.Name, updatedStorageRequestSize)
		if err = ssc.podControl.objectMgr.PatchClaim(set.Namespace, claimName, []byte(patch)); err == nil {
			ssc.recorder.Eventf(set, v1.EventTypeNormal, "SuccessToResizePVC",
				"StatefulSet %s success to resize PVC %s of pod %s", set.Name, claimName, pod.Name)
			continue
		}
		ssc.recorder.Eventf(set, v1.EventTypeWarning, "FailedToResizePVC",
			"StatefulSet %s failed to resize PVC %s of pod %s : %v", set.Name, claimName, pod.Name, err)

		return true, fmt.Errorf("StatefulSet %s failed to resize PVC %s of pod %s: %v",
			set.Name, claimName, pod.Name, err)
	}
	return false, nil
}
