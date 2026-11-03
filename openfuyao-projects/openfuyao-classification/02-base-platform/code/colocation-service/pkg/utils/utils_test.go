/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 */

package utils

import (
	"testing"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"

	colocationv1 "openfuyao.com/colocation-service/pkg/apis/colocation/v1"
)

// Test cases for ToDesiredColocation
func TestToDesiredColocationEmptyPodSpec(t *testing.T) {
	podSpec := &corev1.PodSpec{Tolerations: []corev1.Toleration{}}
	verifyColocationChanges(t, podSpec, true, true, true)
}

func TestToDesiredColocationExistingTolerationNoAffinity(t *testing.T) {
	podSpec := &corev1.PodSpec{
		Tolerations: []corev1.Toleration{buildColocationToleration()},
	}
	verifyColocationChanges(t, podSpec, false, true, true)
}

func TestToDesiredColocationExistingAffinityNoToleration(t *testing.T) {
	podSpec := &corev1.PodSpec{Affinity: buildColocationAffinity()}
	verifyColocationChanges(t, podSpec, true, false, true)
}

func TestToDesiredColocationEmptyNodeSelectorTerms(t *testing.T) {
	podSpec := &corev1.PodSpec{
		Affinity: &corev1.Affinity{
			NodeAffinity: &corev1.NodeAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: &corev1.NodeSelector{
					NodeSelectorTerms: []corev1.NodeSelectorTerm{{}},
				},
			},
		},
	}
	verifyColocationChanges(t, podSpec, true, true, true)
}

func TestToDesiredColocationAlreadyConfigured(t *testing.T) {
	podSpec := &corev1.PodSpec{
		Tolerations: []corev1.Toleration{buildColocationToleration()},
		Affinity:    buildColocationAffinity(),
	}
	verifyColocationChanges(t, podSpec, false, false, false)
}

// Test cases for Job and CronJob functions
func TestToDesiredColocationJob(t *testing.T) {
	job := createTestJob()
	changed := ToDesiredColocationJob(job)
	verifyJobOrCronJobChanges(t, changed, job.Spec.Template.Spec, true)
}

func TestToDesiredColocationCronJob(t *testing.T) {
	cronJob := createTestCronJob()
	changed := ToDesiredColocationCronJob(cronJob)
	verifyJobOrCronJobChanges(t, changed, cronJob.Spec.JobTemplate.Spec.Template.Spec, true)
}

// Test cases for ToDesiredNonColocation
func TestToDesiredNonColocationWithBothSettings(t *testing.T) {
	podSpec := createPodSpecWithColocation(true, true)
	verifyNonColocationChanges(t, podSpec, true)
}

func TestToDesiredNonColocationWithTolerationOnly(t *testing.T) {
	podSpec := createPodSpecWithColocation(true, false)
	verifyNonColocationChanges(t, podSpec, true)
}

func TestToDesiredNonColocationWithAffinityOnly(t *testing.T) {
	podSpec := createPodSpecWithColocation(false, true)
	verifyNonColocationChanges(t, podSpec, true)
}

func TestToDesiredNonColocationWithoutSettings(t *testing.T) {
	podSpec := createPodSpecWithColocation(false, false)
	verifyNonColocationChanges(t, podSpec, false)
}

// Helper verification functions
func verifyColocationChanges(t *testing.T, podSpec *corev1.PodSpec,
	expectTolerationChange, expectAffinityChange, expectChanged bool) {
	changed := ToDesiredColocation(podSpec)

	if changed != expectChanged {
		t.Errorf("Expected changed=%v, got %v", expectChanged, changed)
	}

	if expectTolerationChange && !hasColocationTolerationInSpec(podSpec.Tolerations) {
		t.Errorf("Expected colocation toleration to be added")
	}

	if expectAffinityChange && !hasColocationAffinityInSpec(podSpec.Affinity) {
		t.Errorf("Expected colocation affinity to be added")
	}
}

func verifyJobOrCronJobChanges(t *testing.T, changed bool, podSpec corev1.PodSpec, expectChanged bool) {
	if changed != expectChanged {
		t.Errorf("Expected changed=%v, got %v", expectChanged, changed)
	}

	if len(podSpec.Tolerations) != 1 {
		t.Error("Expected toleration to be added")
	}

	if podSpec.Affinity == nil {
		t.Error("Expected affinity to be added")
	}
}

func verifyNonColocationChanges(t *testing.T, podSpec *corev1.PodSpec, expectChanged bool) {
	changed := ToDesiredNonColocation(podSpec)
	if changed != expectChanged {
		t.Errorf("Expected changed=%v, got %v", expectChanged, changed)
	}

	if expectChanged {
		if hasColocationTolerationInSpec(podSpec.Tolerations) {
			t.Error("Colocation toleration should be removed")
		}
		if hasColocationAffinityInSpec(podSpec.Affinity) {
			t.Error("Colocation affinity should be removed")
		}
	}
}

// Helper construction functions
func createTestJob() *batchv1.Job {
	return &batchv1.Job{
		Spec: batchv1.JobSpec{
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{Tolerations: []corev1.Toleration{}},
			},
		},
	}
}

func createTestCronJob() *batchv1.CronJob {
	return &batchv1.CronJob{
		Spec: batchv1.CronJobSpec{
			JobTemplate: batchv1.JobTemplateSpec{
				Spec: batchv1.JobSpec{
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{Tolerations: []corev1.Toleration{}},
					},
				},
			},
		},
	}
}

func createPodSpecWithColocation(hasToleration, hasAffinity bool) *corev1.PodSpec {
	spec := &corev1.PodSpec{}
	if hasToleration {
		spec.Tolerations = []corev1.Toleration{buildColocationToleration()}
	}
	if hasAffinity {
		spec.Affinity = buildColocationAffinity()
	}
	return spec
}

func hasColocationTolerationInSpec(tolerations []corev1.Toleration) bool {
	for _, t := range tolerations {
		if t.Key == colocationv1.ColocationNodeLabel {
			return true
		}
	}
	return false
}

func hasColocationAffinityInSpec(affinity *corev1.Affinity) bool {
	if affinity == nil || affinity.NodeAffinity == nil {
		return false
	}

	terms := affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution
	if terms == nil {
		return false
	}

	for _, term := range terms.NodeSelectorTerms {
		for _, expr := range term.MatchExpressions {
			if expr.Key == colocationv1.ColocationTaintLabel {
				return true
			}
		}
	}
	return false
}

// Shared builder functions
func buildColocationToleration() corev1.Toleration {
	return corev1.Toleration{
		Key:      colocationv1.ColocationNodeLabel,
		Operator: corev1.TolerationOpExists,
		Effect:   corev1.TaintEffectNoExecute,
	}
}

func buildColocationAffinity() *corev1.Affinity {
	return &corev1.Affinity{
		NodeAffinity: &corev1.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &corev1.NodeSelector{
				NodeSelectorTerms: []corev1.NodeSelectorTerm{
					{
						MatchExpressions: []corev1.NodeSelectorRequirement{
							{
								Key:      colocationv1.ColocationTaintLabel,
								Operator: corev1.NodeSelectorOpExists,
							},
						},
					},
				},
			},
		},
	}
}
