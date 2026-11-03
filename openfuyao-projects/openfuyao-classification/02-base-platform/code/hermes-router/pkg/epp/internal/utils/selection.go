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
	"math/rand"

	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

// SelectExtremeScoreResult picks a single target endpoint whose score is the
// extreme among the supplied candidates, breaking ties uniformly at random.
// Callers are responsible for any candidate preparation (e.g. PD-leader
// filling) before invoking this generic selector.
//
// better reports whether a candidate score should replace the current best
// (greater-than for max-score selection, less-than for min-score selection). An
// empty ProfileRunResult is returned when there are no usable candidates.
func SelectExtremeScoreResult(
	candidates []*fwkscheduling.ScoredEndpoint,
	rng *rand.Rand,
	better func(candidate, current float64) bool,
) *fwkscheduling.ProfileRunResult {
	if len(candidates) == 0 {
		return &fwkscheduling.ProfileRunResult{}
	}

	winners := extremeScoreWinners(candidates, better)
	selected := winners[rng.Intn(len(winners))]
	if selected == nil || selected.Endpoint == nil {
		return &fwkscheduling.ProfileRunResult{}
	}
	return &fwkscheduling.ProfileRunResult{TargetEndpoints: []fwkscheduling.Endpoint{selected.Endpoint}}
}

func extremeScoreWinners(
	scoredPods []*fwkscheduling.ScoredEndpoint,
	better func(candidate, current float64) bool,
) []*fwkscheduling.ScoredEndpoint {
	best := scoredPods[0].Score
	winners := []*fwkscheduling.ScoredEndpoint{scoredPods[0]}

	for _, scoredPod := range scoredPods[1:] {
		switch {
		case better(scoredPod.Score, best):
			best = scoredPod.Score
			winners = []*fwkscheduling.ScoredEndpoint{scoredPod}
		case scoredPod.Score == best:
			winners = append(winners, scoredPod)
		default:
			continue
		}
	}

	return winners
}

// PickTiedIndex returns one index from a non-empty tied-candidate set, choosing
// uniformly at random via nextRandIntn when more than one index is tied.
func PickTiedIndex(indices []int, nextRandIntn func(int) int) int {
	if len(indices) <= 1 || nextRandIntn == nil {
		return indices[0]
	}
	return indices[nextRandIntn(len(indices))]
}
