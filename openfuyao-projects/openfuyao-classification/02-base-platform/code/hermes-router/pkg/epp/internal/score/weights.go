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

package score

// Sum returns the total of all prefill scoring weights.
func (w PrefillWeights) Sum() float64 {
	return w.KVWeight + w.PrefixWeight + w.QueueWeight + w.InflightWeight
}

// HasNegative reports whether any prefill scoring weight is negative.
func (w PrefillWeights) HasNegative() bool {
	return HasNegativeWeight(w.KVWeight, w.PrefixWeight, w.QueueWeight, w.InflightWeight)
}

// FillDefaults writes def onto every field when the group sum is not positive.
func (w *PrefillWeights) FillDefaults(def float64) {
	if w == nil || w.Sum() > 0 {
		return
	}
	w.KVWeight = def
	w.PrefixWeight = def
	w.QueueWeight = def
	w.InflightWeight = def
}

// Sum returns the total of all decode scoring weights.
func (w DecodeWeights) Sum() float64 {
	return w.PrefixWeight + w.RunningWeight + w.KVWeight + w.QueueWeight + w.InflightWeight
}

// HasNegative reports whether any decode scoring weight is negative.
func (w DecodeWeights) HasNegative() bool {
	return HasNegativeWeight(
		w.PrefixWeight,
		w.RunningWeight,
		w.KVWeight,
		w.QueueWeight,
		w.InflightWeight,
	)
}

// FillDefaults writes def onto every field when the group sum is not positive.
func (w *DecodeWeights) FillDefaults(def float64) {
	if w == nil || w.Sum() > 0 {
		return
	}
	w.PrefixWeight = def
	w.RunningWeight = def
	w.KVWeight = def
	w.QueueWeight = def
	w.InflightWeight = def
}

// Sum returns the total of all group scoring weights.
func (w GroupWeights) Sum() float64 {
	return w.PrefillScoreWeight + w.DecodeScoreWeight
}

// HasNegative reports whether any group scoring weight is negative.
func (w GroupWeights) HasNegative() bool {
	return HasNegativeWeight(w.PrefillScoreWeight, w.DecodeScoreWeight)
}

// FillDefaults writes def onto every field when the group sum is not positive.
func (w *GroupWeights) FillDefaults(def float64) {
	if w == nil || w.Sum() > 0 {
		return
	}
	w.PrefillScoreWeight = def
	w.DecodeScoreWeight = def
}

// Sum returns the total of all aggregate scoring weights.
func (w AggregateWeights) Sum() float64 {
	return w.KVCacheNotHitRateWeight + w.XPUCacheUsageWeight + w.WaitingRequestWeight + w.InflightWeight
}

// HasNegative reports whether any aggregate scoring weight is negative.
func (w AggregateWeights) HasNegative() bool {
	return HasNegativeWeight(
		w.KVCacheNotHitRateWeight,
		w.XPUCacheUsageWeight,
		w.WaitingRequestWeight,
		w.InflightWeight,
	)
}

// ResolvePDWeightDefaults fills empty PD scoring weight groups with the
// provided defaults. A group is empty when its field sum is not positive.
func ResolvePDWeightDefaults(
	prefill *PrefillWeights,
	decode *DecodeWeights,
	group *GroupWeights,
	defaultWeight, defaultGroupWeight float64,
) {
	prefill.FillDefaults(defaultWeight)
	decode.FillDefaults(defaultWeight)
	group.FillDefaults(defaultGroupWeight)
}
