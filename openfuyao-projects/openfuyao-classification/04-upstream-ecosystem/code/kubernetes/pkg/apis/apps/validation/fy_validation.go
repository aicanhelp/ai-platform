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

// Package validation
package validation

import (
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/apis/apps"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

// validateStatefulSetPVCUpdate
// validate the volumeClaimTemplates updating of statefulset.
func validateStatefulSetPVCUpdate(set *apps.StatefulSet, newVolumeClaimTemplates, oldVolumeClaimTemplates []api.PersistentVolumeClaim) field.ErrorList {
	allErrs := field.ErrorList{}

	findTemplate := func(templates []api.PersistentVolumeClaim, name string) (*api.PersistentVolumeClaim, int) {
		for index, template := range templates {
			if template.Name == name {
				return &template, index
			}
		}

		return nil, -1
	}

	recoverVolumeExpansionFailureEnabled := utilfeature.DefaultFeatureGate.Enabled(features.RecoverVolumeExpansionFailure)

	for _, oldTemplate := range oldVolumeClaimTemplates {
		newTemplate, _ := findTemplate(newVolumeClaimTemplates, oldTemplate.Name)
		if newTemplate == nil {
			return allErrs
		}

		oldSize := oldTemplate.Spec.Resources.Requests["storage"]
		newSize := newTemplate.Spec.Resources.Requests["storage"]

		if newSize.Cmp(oldSize) < 0 {
			allErrs = append(allErrs, validateStatefulSetStatus(recoverVolumeExpansionFailureEnabled, set, newTemplate.Name)...)
		}

		// statefulset's volumeClaimTemplates updates aren't super common and general updates are likely to be touching spec, so we'll do this
		// deep copy right away. This avoids mutating our inputs
		newTemplateClone := newTemplate.DeepCopy()
		newTemplateClone.Spec.Resources.Requests["storage"] = oldTemplate.Spec.Resources.Requests["storage"] // +k8s:verify-mutation:reason=clone

		if !apiequality.Semantic.DeepEqual(newTemplateClone.Spec, oldTemplate.Spec) {
			allErrs = append(allErrs, field.Forbidden(field.NewPath("spec"), "update to statefulset spec for field other than 'replicas', 'ordinals', 'template', 'updateStrategy', 'persistentVolumeClaimRetentionPolicy', 'minReadySecond' and 'volumeClaimTemplate.Spec.Resources.Requests[storage]' are forbidden"))
		}
	}

	return allErrs
}

func validateStatefulSetStatus(recoverVolumeExpansionFailureEnabled bool, set *apps.StatefulSet, templateName string) field.ErrorList {
	allErrs := field.ErrorList{}
	if !recoverVolumeExpansionFailureEnabled || !checkSetTemplateStatusNeedScale(set, templateName) {
		allErrs = append(allErrs, field.Forbidden(field.NewPath("spec", "volumeClaimTemplates", "spec", "resources", "requests", "storage"), "storage request can not be less than previous value"))
	}
	return allErrs
}

func checkSetTemplateStatusNeedScale(set *apps.StatefulSet, templateName string) bool {
	templates := set.Status.VolumeClaimTemplates
	if templates != nil && len(templates) != 0 {
		for _, template := range templates {
			if template.TemplateName == templateName && template.ReadyReplicas != 0 {
				return false
			}
		}
	}
	return true
}
