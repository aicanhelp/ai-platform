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
	"testing"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/component-base/featuregate"
	"k8s.io/kubernetes/pkg/features"
)

func TestFYInconsistentVolumeStatus(t *testing.T) {
	features := map[featuregate.Feature]bool{
		features.FYStatefulSetVolumeExpansion: true,
	}
	revert := enableFeaturesForTest(t, features)
	defer revert()
	set := newStatefulSet(3)
	set.Status = *newStatefulSetStatus()
	newStatus := newStatefulSetStatus()
	result := fyInconsistentVolumeStatus(set, newStatus)
	if result {
		t.Fatalf("fyInconsistentVolumeStauts is failed")
	}
	newStatus.VolumeClaimTemplates[0].FinishedReconciliationGeneration = 1
	newStatus.VolumeClaimTemplates[0].ReadyReplicas = 0
	result = fyInconsistentVolumeStatus(set, newStatus)
	if !result {
		t.Fatalf("fyInconsistentVolumeStatus is failed")
	}
}

func TestFYInconsistentVolumeStatusFeatureDisable(t *testing.T) {
	set := newStatefulSet(3)
	set.Status = *newStatefulSetStatus()
	newStatus := newStatefulSetStatus()
	result := fyInconsistentVolumeStatus(set, newStatus)
	if result {
		t.Fatalf("fyInconsistentVolumeStauts is failed")
	}
	newStatus.VolumeClaimTemplates[0].FinishedReconciliationGeneration = 1
	newStatus.VolumeClaimTemplates[0].ReadyReplicas = 0
	result = fyInconsistentVolumeStatus(set, newStatus)
	if !result {
		t.Fatalf("fyInconsistentVolumeStatus is failed")
	}
	features := map[featuregate.Feature]bool{
		features.FYStatefulSetVolumeExpansion: true,
	}
	revert := enableFeaturesForTest(t, features)
	defer revert()
	set.Status.VolumeClaimTemplates = nil
	result = fyInconsistentVolumeStatus(set, newStatus)
	if !result {
		t.Fatalf("fyInconsistentVolumeStatus is failed")
	}
}

func TestFYAddAnnotationWithExpansionFeatureEnable(t *testing.T) {
	features := map[featuregate.Feature]bool{
		features.FYStatefulSetVolumeExpansion: true,
	}
	revert := enableFeaturesForTest(t, features)
	defer revert()
	claim := newPersistentVolumeClaim()
	fyAddAnnotationsWithExpansionFeatureEnable(&claim, "teststs", "testTemplateName")
	if claim.Annotations["fy.stateful-set.name"] != "teststs" {
		t.Fatalf("Annotations statefulsetName is error")
	}
	if claim.Annotations["fy.pvc-template.name"] != "testTemplateName" {
		t.Fatalf("Annotations pvc template name is error")
	}
}

func TestFYAddAnnotationWithExpansionFeatureDisable(t *testing.T) {
	claim := newPersistentVolumeClaim()
	fyAddAnnotationsWithExpansionFeatureEnable(&claim, "teststs", "testTemplateName")
	if claim.Annotations != nil {
		t.Fatalf("Annotations is error")
	}

}

func newStatefulSetStatus() *apps.StatefulSetStatus {
	status := apps.VolumeClaimTemplate{}
	status.TemplateName = "test"
	status.FinishedReconciliationGeneration = 2
	status.ReadyReplicas = 3
	return &apps.StatefulSetStatus{
		VolumeClaimTemplates: []apps.VolumeClaimTemplate{
			status,
		},
	}
}

func newPersistentVolumeClaim() v1.PersistentVolumeClaim {
	return v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "datadir"},
		Spec: v1.PersistentVolumeClaimSpec{
			Resources: v1.VolumeResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceStorage: *resource.NewQuantity(1, resource.BinarySI),
				},
			},
		},
	}
}
