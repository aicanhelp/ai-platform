/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */


package controller

import infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"

// EnginePrefillEffective reports whether prefill defines a workload template (PD mode).
func EnginePrefillEffective(e *infernexv1alpha1.InferenceEngineSpec) bool {
	if e == nil || e.Prefill == nil {
		return false
	}
	return e.Prefill.Template != nil && len(e.Prefill.Template.Spec.Containers) > 0
}

// EngineIsPDMode is true when merge-complete engine has an effective prefill template.
func EngineIsPDMode(e *infernexv1alpha1.InferenceEngineSpec) bool {
	return EnginePrefillEffective(e)
}

// EngineRootWorkload returns a copy of the root inline workload (aggregate or decode).
func EngineRootWorkload(e *infernexv1alpha1.InferenceEngineSpec) *infernexv1alpha1.InferenceEngineWorkloadSpec {
	if e == nil {
		return nil
	}
	w := e.InferenceEngineWorkloadSpec
	return &w
}

// EngineAggregateWorkload returns the aggregate workload when not in PD mode.
func EngineAggregateWorkload(e *infernexv1alpha1.InferenceEngineSpec) *infernexv1alpha1.InferenceEngineWorkloadSpec {
	if EngineIsPDMode(e) {
		return nil
	}
	return EngineRootWorkload(e)
}

// engineWorkloadEffective reports whether a workload has a non-empty pod template.
func engineWorkloadEffective(w *infernexv1alpha1.InferenceEngineWorkloadSpec) bool {
	return w != nil && w.Template != nil && len(w.Template.Spec.Containers) > 0
}

// EngineDecodeWorkload returns the decode workload when in PD mode (root inline fields).
func EngineDecodeWorkload(e *infernexv1alpha1.InferenceEngineSpec) *infernexv1alpha1.InferenceEngineWorkloadSpec {
	if !EngineIsPDMode(e) {
		return nil
	}
	w := EngineRootWorkload(e)
	if !engineWorkloadEffective(w) {
		return nil
	}
	return w
}

// EnginePrefillWorkload returns the prefill workload when in PD mode.
func EnginePrefillWorkload(e *infernexv1alpha1.InferenceEngineSpec) *infernexv1alpha1.InferenceEngineWorkloadSpec {
	if !EngineIsPDMode(e) {
		return nil
	}
	if !engineWorkloadEffective(e.Prefill) {
		return nil
	}
	return e.Prefill
}
