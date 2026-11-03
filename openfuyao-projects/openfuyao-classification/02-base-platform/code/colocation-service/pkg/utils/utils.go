/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package utils

import (
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"

	colocationv1 "openfuyao.com/colocation-service/pkg/apis/colocation/v1"
)

func getColocationToleration() corev1.Toleration {
	return corev1.Toleration{
		Key:      colocationv1.ColocationNodeLabel,
		Operator: corev1.TolerationOpExists,
		Effect:   corev1.TaintEffectNoExecute,
	}
}

func getColocationNodeSelectorRequirement() corev1.NodeSelectorRequirement {
	return corev1.NodeSelectorRequirement{
		Key:      colocationv1.ColocationTaintLabel,
		Operator: corev1.NodeSelectorOpExists,
	}
}

func getColocationNodeSelector() *corev1.NodeSelector {
	return &corev1.NodeSelector{
		NodeSelectorTerms: []corev1.NodeSelectorTerm{
			{MatchExpressions: []corev1.NodeSelectorRequirement{getColocationNodeSelectorRequirement()}},
		},
	}
}

func hasColocationToleration(tolerations []corev1.Toleration) bool {
	for _, t := range tolerations {
		if t.Key == colocationv1.ColocationNodeLabel &&
			t.Operator == corev1.TolerationOpExists &&
			t.Effect == corev1.TaintEffectNoExecute {
			return true
		}
	}
	return false
}

func addColocationToleration(podSpec *corev1.PodSpec) bool {
	if hasColocationToleration(podSpec.Tolerations) {
		return false
	}
	podSpec.Tolerations = append(podSpec.Tolerations, getColocationToleration())
	klog.Info("add toleration")
	return true
}

func addColocationNodeAffinity(podSpec *corev1.PodSpec) bool {
	nodeSelector := getColocationNodeSelector()
	if podSpec.Affinity == nil {
		podSpec.Affinity = &corev1.Affinity{NodeAffinity: &corev1.NodeAffinity{RequiredDuringSchedulingIgnoredDuringExecution: nodeSelector}}
		klog.Info("add node affinity")
		return true
	}
	if podSpec.Affinity.NodeAffinity == nil {
		podSpec.Affinity.NodeAffinity = &corev1.NodeAffinity{RequiredDuringSchedulingIgnoredDuringExecution: nodeSelector}
		klog.Info("add node affinity")
		return true
	}
	if podSpec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution == nil {
		podSpec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution = nodeSelector
		klog.Info("add node affinity")
		return true
	}
	return addColocationNodeSelectorTerm(podSpec)
}

func addColocationNodeSelectorTerm(podSpec *corev1.PodSpec) bool {
	nsr := getColocationNodeSelectorRequirement()
	terms := podSpec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms
	var newTerms []corev1.NodeSelectorTerm
	found, hasEmpty := false, false

	for _, nst := range terms {
		if len(nst.MatchFields) == 0 && len(nst.MatchExpressions) == 0 {
			hasEmpty = true
			continue
		}
		if containsNodeSelectorRequirement(nst, nsr) {
			found = true
		}
		newTerms = append(newTerms, nst)
	}

	if !found {
		newTerms = append(newTerms, corev1.NodeSelectorTerm{MatchExpressions: []corev1.NodeSelectorRequirement{nsr}})
	}
	if !found || hasEmpty {
		podSpec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms = newTerms
		klog.Info("add node affinity")
		return true
	}
	return false
}

func containsNodeSelectorRequirement(nst corev1.NodeSelectorTerm, target corev1.NodeSelectorRequirement) bool {
	for _, nsr := range nst.MatchExpressions {
		if nsr.Key == target.Key && nsr.Operator == target.Operator {
			return true
		}
	}
	for _, nf := range nst.MatchFields {
		if nf.Key == target.Key && nf.Operator == target.Operator {
			return true
		}
	}
	return false
}

func ToDesiredColocation(podSpec *corev1.PodSpec) bool {
	changed := addColocationToleration(podSpec)
	if addColocationNodeAffinity(podSpec) {
		changed = true
	}
	return changed
}

func ToDesiredColocationJob(job *batchv1.Job) bool {
	return ToDesiredColocation(&job.Spec.Template.Spec)
}

func ToDesiredColocationCronJob(cronJob *batchv1.CronJob) bool {
	return ToDesiredColocation(&cronJob.Spec.JobTemplate.Spec.Template.Spec)
}

func removeColocationToleration(podSpec *corev1.PodSpec) bool {
	var tolerations []corev1.Toleration
	found := false
	for _, t := range podSpec.Tolerations {
		if t.Key == colocationv1.ColocationNodeLabel {
			found = true
		} else {
			tolerations = append(tolerations, t)
		}
	}
	if found {
		klog.Info("delete toleration")
		podSpec.Tolerations = tolerations
	}
	return found
}

func removeColocationNodeAffinity(podSpec *corev1.PodSpec) bool {
	if podSpec.Affinity == nil || podSpec.Affinity.NodeAffinity == nil ||
		podSpec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution == nil {
		return false
	}
	terms := podSpec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms
	newTerms, changed := filterNodeSelectorTerms(terms)
	if changed {
		updateNodeAffinity(podSpec, newTerms)
	}
	return changed
}

func filterNodeSelectorTerms(terms []corev1.NodeSelectorTerm) ([]corev1.NodeSelectorTerm, bool) {
	var result []corev1.NodeSelectorTerm
	changed := false
	for _, nst := range terms {
		newExprs, found := filterMatchExpressions(nst.MatchExpressions)
		if found {
			changed = true
			klog.Info("delete node affinity")
		}
		if len(newExprs) > 0 || len(nst.MatchFields) > 0 {
			result = append(result, corev1.NodeSelectorTerm{MatchExpressions: newExprs, MatchFields: nst.MatchFields})
		} else {
			// The term was fully filtered out; mark changed so the caller
			// knows the node affinity needs updating.
			if found {
				changed = true
			}
		}
	}
	return result, changed
}

func filterMatchExpressions(exprs []corev1.NodeSelectorRequirement) ([]corev1.NodeSelectorRequirement, bool) {
	var result []corev1.NodeSelectorRequirement
	found := false
	for _, expr := range exprs {
		if expr.Key == colocationv1.ColocationTaintLabel && expr.Operator == corev1.NodeSelectorOpExists {
			found = true
		} else {
			result = append(result, expr)
		}
	}
	return result, found
}

func updateNodeAffinity(podSpec *corev1.PodSpec, terms []corev1.NodeSelectorTerm) {
	if len(terms) == 0 {
		podSpec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution = nil
		if len(podSpec.Affinity.NodeAffinity.PreferredDuringSchedulingIgnoredDuringExecution) == 0 {
			podSpec.Affinity.NodeAffinity = nil
		}
		klog.Info("okNodeSelectorTerm")
	} else {
		podSpec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms = terms
	}
}

func ToDesiredNonColocation(podSpec *corev1.PodSpec) bool {
	changed := removeColocationToleration(podSpec)
	if removeColocationNodeAffinity(podSpec) {
		changed = true
	}
	return changed
}

type StringCheckMode string

const (
	// 常规模式
	NormalMode StringCheckMode = "normal"
	// 自定义模式: 标签Key的格式化校验
	LabelKeyMode StringCheckMode = "label_key_mode"
)
