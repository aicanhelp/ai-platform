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

// Package score provides shared scoring helpers for Hermes Router plugins.
package score

import (
	"math"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	prefixattr "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"

	internalinflight "hermes-router/pkg/epp/internal/inflight"
	"hermes-router/pkg/epp/internal/pdgroup"
	"hermes-router/pkg/epp/internal/utils"
)

// MaxMissingBadness is an exported constant used by Hermes Router.
const MaxMissingBadness = 1.0

// baselineLatencyMillis anchors the saturating latency badness mapping. A
// latency equal to the baseline yields badness 0.5, while much larger
// values approach 1. The 1-second choice gives useful resolution across the
// LLM serving range, where TTFT and TPOT typically span milliseconds to
// seconds.
const baselineLatencyMillis = 1000.0

// AggregateWeights describes AggregateWeights data used by Hermes Router.
type AggregateWeights struct {
	// KVCacheNotHitRateWeight stores the KVCacheNotHitRateWeight value.
	KVCacheNotHitRateWeight float64
	// XPUCacheUsageWeight stores the XPUCacheUsageWeight value.
	XPUCacheUsageWeight float64
	// WaitingRequestWeight stores the WaitingRequestWeight value.
	WaitingRequestWeight float64
	// InflightWeight stores the InflightWeight value.
	InflightWeight float64
}

// PrefillWeights describes PrefillWeights data used by Hermes Router.
type PrefillWeights struct {
	// KVWeight stores the KVWeight value.
	KVWeight float64
	// PrefixWeight stores the PrefixWeight value.
	PrefixWeight float64
	// QueueWeight stores the QueueWeight value.
	QueueWeight float64
	// InflightWeight stores the InflightWeight value.
	InflightWeight float64
}

// DecodeWeights describes DecodeWeights data used by Hermes Router.
type DecodeWeights struct {
	// PrefixWeight stores the PrefixWeight value.
	PrefixWeight float64
	// RunningWeight stores the RunningWeight value.
	RunningWeight float64
	// KVWeight stores the KVWeight value.
	KVWeight float64
	// QueueWeight stores the QueueWeight value.
	QueueWeight float64
	// InflightWeight stores the InflightWeight value.
	InflightWeight float64
}

// DecodeLimits carries the per-cycle maxima used to normalize decode scoring.
type DecodeLimits struct {
	// MaxRunning is the maximum observed running request count among decode pods.
	MaxRunning int
	// MaxQueue is the maximum observed waiting queue size among decode pods.
	MaxQueue int
	// MaxInflight is the maximum observed tracked inflight token count among decode pods.
	MaxInflight float64
}

// PrefillLimits carries the per-cycle maxima used to normalize prefill scoring.
type PrefillLimits struct {
	// MaxQueue is the maximum observed waiting queue size among prefill pods.
	MaxQueue int
	// MaxInflight is the maximum observed tracked inflight token count among prefill pods.
	MaxInflight float64
}

// PDLimits carries the pool-wide maxima used to normalize PD group scoring.
// Prefill and decode pools are normalized independently. The maxima are gathered
// across every group's members (see CollectPDLimits), not per group, so
// queue/inflight badness stays comparable across groups. A per-group denominator
// collapses to a binary busy/idle flag in 1P1D topologies, where each pool's lone
// member is always its own max.
type PDLimits struct {
	// Prefill carries the pool-wide prefill normalization maxima.
	Prefill PrefillLimits
	// Decode carries the pool-wide decode normalization maxima.
	Decode DecodeLimits
}

// GroupWeights describes GroupWeights data used by Hermes Router.
type GroupWeights struct {
	// PrefillScoreWeight stores the PrefillScoreWeight value.
	PrefillScoreWeight float64
	// DecodeScoreWeight stores the DecodeScoreWeight value.
	DecodeScoreWeight float64
}

// Aggregate provides the exported Aggregate operation.
func Aggregate(endpoint fwkscheduling.Endpoint, maxQueue int, maxInflight float64, weights AggregateWeights) float64 {
	return FromBadness(AggregateBadness(endpoint, maxQueue, maxInflight, weights))
}

// AggregateBadness provides the exported AggregateBadness operation.
func AggregateBadness(
	endpoint fwkscheduling.Endpoint,
	maxQueue int,
	maxInflight float64,
	weights AggregateWeights,
) float64 {
	metrics := ZeroMetrics(endpoint)
	badness, _ := WeightedBadness([]Signal{
		{
			Name:    "kv-cache-not-hit",
			Weight:  weights.KVCacheNotHitRateWeight,
			Badness: 1 - PrefixCacheScoreFromEndpoint(endpoint),
		},
		{
			Name:    "xpu-cache-usage",
			Weight:  weights.XPUCacheUsageWeight,
			Badness: Clamp01(metrics.KVCacheUsagePercent),
		},
		{
			Name:    "waiting-queue",
			Weight:  weights.WaitingRequestWeight,
			Badness: RelativeValue(metrics.WaitingQueueSize, maxQueue),
		},
		{Name: "inflight", Weight: weights.InflightWeight, Badness: RelativeFloat(InflightTokens(endpoint), maxInflight)},
	})
	return badness
}

// PDScorer bundles the per-request inputs used to score one PD group.
// Limits must be pool-wide maxima from CollectPDLimits so queue/inflight
// badness stays comparable across groups, including 1P1D topologies.
type PDScorer struct {
	Prefill      PrefillWeights
	Decode       DecodeWeights
	Group        GroupWeights
	Limits       PDLimits
	NextRandIntn func(int) int
}

// DecodeScorer bundles the per-request inputs used to score one PD group on the
// decode-first path (decode members only). Limits must be pool-wide decode
// maxima from CollectPDLimits so queue/inflight badness stays comparable.
type DecodeScorer struct {
	Decode       DecodeWeights
	Limits       DecodeLimits
	NextRandIntn func(int) int
}

// DecodeGroup scores a PD group leader by decode pods only and records the best
// SelectedDecodePod. SelectedPrefillPod is cleared for the decode-first d path.
func DecodeGroup(info *pdgroup.Info, scorer DecodeScorer) (float64, *pdgroup.Info, bool) {
	updated := pdgroup.Clone(info)
	if err := pdgroup.Validate(updated); err != nil {
		return 0, updated, false
	}

	bestDecode, bestDecodeBadness := SelectBestDecode(
		updated.DecodePods,
		scorer.Limits,
		scorer.Decode,
		scorer.NextRandIntn,
	)
	updated.SelectedPrefillPod = nil
	updated.SelectedDecodePod = bestDecode
	return FromBadness(bestDecodeBadness), updated, true
}

// PDGroup scores a single PD group and records its best prefill/decode members.
func PDGroup(info *pdgroup.Info, scorer PDScorer) (float64, *pdgroup.Info, bool) {
	updated := pdgroup.Clone(info)
	if err := pdgroup.Validate(updated); err != nil {
		return 0, updated, false
	}

	bestPrefill, bestPrefillBadness := SelectBestPrefill(
		updated.PrefillPods,
		scorer.Limits.Prefill.MaxQueue,
		scorer.Limits.Prefill.MaxInflight,
		scorer.Prefill,
		scorer.NextRandIntn,
	)
	updated.SelectedPrefillPod = bestPrefill

	bestDecode, bestDecodeBadness := SelectBestDecode(
		updated.DecodePods,
		scorer.Limits.Decode,
		scorer.Decode,
		scorer.NextRandIntn,
	)
	updated.SelectedDecodePod = bestDecode

	groupBadness, _ := WeightedBadness([]Signal{
		{Name: "prefill", Weight: scorer.Group.PrefillScoreWeight, Badness: bestPrefillBadness},
		{Name: "decode", Weight: scorer.Group.DecodeScoreWeight, Badness: bestDecodeBadness},
	})
	return FromBadness(groupBadness), updated, true
}

// CollectPDLimits gathers pool-wide prefill/decode normalization maxima across
// every PD group represented in pods. Endpoints without group metadata are
// skipped. Feeding the result to PDGroup keeps queue/inflight badness comparable
// across groups; see PDLimits.
func CollectPDLimits(pods []fwkscheduling.Endpoint) PDLimits {
	limits := PDLimits{}
	for _, endpoint := range pods {
		info, ok := pdgroup.GetDirect(endpoint)
		if !ok {
			continue
		}
		accumulatePrefillLimits(&limits.Prefill, info.PrefillPods)
		accumulateDecodeLimits(&limits.Decode, info.DecodePods)
	}
	return limits
}

func accumulatePrefillLimits(limits *PrefillLimits, pods []fwkscheduling.ScoredEndpoint) {
	if queue := MaxWaitingQueueFromScored(pods); queue > limits.MaxQueue {
		limits.MaxQueue = queue
	}
	if inflight := MaxInflightTokensFromScored(pods); inflight > limits.MaxInflight {
		limits.MaxInflight = inflight
	}
}

func accumulateDecodeLimits(limits *DecodeLimits, pods []fwkscheduling.ScoredEndpoint) {
	if running := MaxRunningRequests(pods); running > limits.MaxRunning {
		limits.MaxRunning = running
	}
	if queue := MaxWaitingQueueFromScored(pods); queue > limits.MaxQueue {
		limits.MaxQueue = queue
	}
	if inflight := MaxInflightTokensFromScored(pods); inflight > limits.MaxInflight {
		limits.MaxInflight = inflight
	}
}

// SelectBestPrefill provides the exported SelectBestPrefill operation.
func SelectBestPrefill(
	pods []fwkscheduling.ScoredEndpoint,
	maxQueue int,
	maxInflight float64,
	weights PrefillWeights,
	nextRandIntn func(int) int,
) (*fwkscheduling.ScoredEndpoint, float64) {
	return SelectBestScoredEndpoint(
		pods,
		nextRandIntn,
		MaxMissingBadness,
		func(endpoint fwkscheduling.Endpoint) (float64, bool) {
			return PrefillBadness(endpoint, maxQueue, maxInflight, weights), true
		},
	)
}

// SelectBestDecode provides the exported SelectBestDecode operation.
func SelectBestDecode(
	pods []fwkscheduling.ScoredEndpoint,
	limits DecodeLimits,
	weights DecodeWeights,
	nextRandIntn func(int) int,
) (*fwkscheduling.ScoredEndpoint, float64) {
	return SelectBestScoredEndpoint(
		pods,
		nextRandIntn,
		MaxMissingBadness,
		func(endpoint fwkscheduling.Endpoint) (float64, bool) {
			return DecodeBadness(endpoint, limits.MaxRunning, limits.MaxQueue, limits.MaxInflight, weights), true
		},
	)
}

// SelectBestScoredEndpoint scores candidate pods by badness and returns the pod
// tied for the lowest badness, chosen uniformly at random via nextRandIntn.
//
// For each pod, badnessOf reports its badness and whether it qualifies. A
// qualifying pod has its Score overwritten with FromBadness(badness); pods that
// do not qualify keep their existing Score and are excluded from selection.
// emptyBadness is returned when no pod qualifies.
func SelectBestScoredEndpoint(
	pods []fwkscheduling.ScoredEndpoint,
	nextRandIntn func(int) int,
	emptyBadness float64,
	badnessOf func(fwkscheduling.Endpoint) (float64, bool),
) (*fwkscheduling.ScoredEndpoint, float64) {
	bestIndices := make([]int, 0, 1)
	bestBadness := 0.0
	for i := range pods {
		badness, ok := badnessOf(pods[i].Endpoint)
		if !ok {
			continue
		}
		pods[i].Score = FromBadness(badness)
		switch {
		case len(bestIndices) == 0 || badness < bestBadness:
			bestIndices = []int{i}
			bestBadness = badness
		case badness == bestBadness:
			bestIndices = append(bestIndices, i)
		default:
			continue
		}
	}
	if len(bestIndices) == 0 {
		return nil, emptyBadness
	}
	selected := pods[utils.PickTiedIndex(bestIndices, nextRandIntn)]
	return &selected, bestBadness
}

// PrefillBadness provides the exported PrefillBadness operation.
func PrefillBadness(
	endpoint fwkscheduling.Endpoint,
	maxQueue int,
	maxInflight float64,
	weights PrefillWeights,
) float64 {
	metrics, ok := EndpointMetrics(endpoint)
	if !ok {
		return MaxMissingBadness
	}
	badness, _ := WeightedBadness([]Signal{
		{Name: "kv-cache-usage", Weight: weights.KVWeight, Badness: Clamp01(metrics.KVCacheUsagePercent)},
		{Name: "prefix-cache-miss", Weight: weights.PrefixWeight, Badness: 1 - PrefixCacheScoreFromEndpoint(endpoint)},
		{Name: "waiting-queue", Weight: weights.QueueWeight, Badness: RelativeValue(metrics.WaitingQueueSize, maxQueue)},
		{Name: "inflight", Weight: weights.InflightWeight, Badness: RelativeFloat(InflightTokens(endpoint), maxInflight)},
	})
	return badness
}

// DecodeBadness provides the exported DecodeBadness operation.
func DecodeBadness(
	endpoint fwkscheduling.Endpoint,
	maxRunning, maxQueue int,
	maxInflight float64,
	weights DecodeWeights,
) float64 {
	metrics, ok := EndpointMetrics(endpoint)
	if !ok {
		return MaxMissingBadness
	}
	badness, _ := WeightedBadness([]Signal{
		{
			Name:    "prefix-cache-miss",
			Weight:  weights.PrefixWeight,
			Badness: 1 - PrefixCacheScoreFromEndpoint(endpoint),
		},
		{
			Name:    "running-requests",
			Weight:  weights.RunningWeight,
			Badness: RelativeValue(metrics.RunningRequestsSize, maxRunning),
		},
		{Name: "kv-cache-usage", Weight: weights.KVWeight, Badness: Clamp01(metrics.KVCacheUsagePercent)},
		{Name: "waiting-queue", Weight: weights.QueueWeight, Badness: RelativeValue(metrics.WaitingQueueSize, maxQueue)},
		{Name: "inflight", Weight: weights.InflightWeight, Badness: RelativeFloat(InflightTokens(endpoint), maxInflight)},
	})
	return badness
}

// MaxWaitingQueue provides the exported MaxWaitingQueue operation.
func MaxWaitingQueue(pods []fwkscheduling.Endpoint) int {
	maxQueue := 0
	for _, endpoint := range pods {
		metrics := ZeroMetrics(endpoint)
		if metrics.WaitingQueueSize > maxQueue {
			maxQueue = metrics.WaitingQueueSize
		}
	}
	return maxQueue
}

// MaxWaitingQueueFromScored provides the exported MaxWaitingQueueFromScored operation.
func MaxWaitingQueueFromScored(pods []fwkscheduling.ScoredEndpoint) int {
	maxQueue := 0
	for _, pod := range pods {
		metrics, ok := EndpointMetrics(pod.Endpoint)
		if ok && metrics.WaitingQueueSize > maxQueue {
			maxQueue = metrics.WaitingQueueSize
		}
	}
	return maxQueue
}

// MaxRunningRequests provides the exported MaxRunningRequests operation.
func MaxRunningRequests(pods []fwkscheduling.ScoredEndpoint) int {
	maxRunning := 0
	for _, pod := range pods {
		metrics, ok := EndpointMetrics(pod.Endpoint)
		if ok && metrics.RunningRequestsSize > maxRunning {
			maxRunning = metrics.RunningRequestsSize
		}
	}
	return maxRunning
}

// RelativeValue provides the exported RelativeValue operation.
func RelativeValue(value, max int) float64 {
	if max <= 0 {
		return 0
	}
	return float64(value) / float64(max)
}

// RelativeFloat normalizes a float-valued signal against the max observed in
// the candidate pool. Returns 0 when the pool max is non-positive.
func RelativeFloat(value, max float64) float64 {
	if max <= 0 {
		return 0
	}
	return value / max
}

// LatencyBadnessFromMillis maps a latency in milliseconds to badness in [0, 1).
// The 1-second baseline gives a useful resolution curve across typical LLM serving
// latencies. Non-positive, NaN, or zero inputs return 0 (most favorable),
// matching the project-wide "missing data is best" convention; +Inf returns 1.
func LatencyBadnessFromMillis(latencyMillis float64) float64 {
	if math.IsNaN(latencyMillis) || latencyMillis <= 0 {
		return 0
	}
	if math.IsInf(latencyMillis, 1) {
		return 1
	}
	return latencyMillis / (latencyMillis + baselineLatencyMillis)
}

// InflightTokens returns the EPP-tracked inflight token count for an endpoint.
// Inflight load is the router's own bookkeeping of requests dispatched but not
// yet reflected in scraped backend metrics — it closes the staleness window
// between StartRequest and the next metrics poll.
func InflightTokens(endpoint fwkscheduling.Endpoint) float64 {
	if endpoint == nil {
		return 0
	}
	value, ok := endpoint.Get(internalinflight.DefaultAttributeKey)
	if !ok {
		return 0
	}
	load, ok := value.(*internalinflight.Load)
	if !ok || load == nil || load.Tokens <= 0 {
		return 0
	}
	return float64(load.Tokens)
}

// MaxInflightTokens returns the largest inflight token count across the
// candidate pool. Used as the normalization denominator for inflight badness.
func MaxInflightTokens(pods []fwkscheduling.Endpoint) float64 {
	maxLoad := 0.0
	for _, endpoint := range pods {
		if tokens := InflightTokens(endpoint); tokens > maxLoad {
			maxLoad = tokens
		}
	}
	return maxLoad
}

// MaxInflightTokensFromScored is the ScoredEndpoint variant used by PD scorers.
func MaxInflightTokensFromScored(pods []fwkscheduling.ScoredEndpoint) float64 {
	maxLoad := 0.0
	for _, pod := range pods {
		if tokens := InflightTokens(pod.Endpoint); tokens > maxLoad {
			maxLoad = tokens
		}
	}
	return maxLoad
}

// FromBadness converts a badness value in [0, 1] (where higher means
// worse) into a score in [0, 1] (where higher means better) via a linear flip.
// The mapping is `1 - badness`, which preserves the linear influence of each
// weighted signal — unlike a sigmoid such as `1/(1+x)` that would compress
// score differences at high badness and absorb weight scaling.
func FromBadness(badness float64) float64 {
	return 1.0 - Clamp01(badness)
}

// Clamp01 forces a value into the [0, 1] interval, treating NaN and negative
// infinity as 0 and positive infinity as 1. Values outside [0, 1] are clamped
// to the nearest bound.
func Clamp01(value float64) float64 {
	if math.IsNaN(value) {
		return 0
	}
	if value <= 0 {
		return 0
	}
	if value >= 1 {
		return 1
	}
	return value
}

// Signal is one weighted contribution to a weighted-average badness. Callers
// supply a slice of Signals already populated with the per-endpoint badness
// values; the helper handles weight zero/negative skipping and divides by the
// total active weight so the result stays in [0, 1].
type Signal struct {
	// Name stores the Name value.
	Name string
	// Weight stores the Weight value.
	Weight float64
	// Badness stores the Badness value.
	Badness float64
}

// WeightedBadness returns Σ(wᵢ · clamp01(bᵢ)) / Σwᵢ across signals whose weight
// is positive. Signals with non-positive weight are skipped entirely (they do
// not contribute to either numerator or denominator). Returns (0, false) when
// no signal carries a positive weight — callers should treat that as "no
// scoring signal applicable" rather than "best possible".
func WeightedBadness(signals []Signal) (float64, bool) {
	totalWeight := 0.0
	totalBadness := 0.0
	for _, signal := range signals {
		if signal.Weight <= 0 {
			continue
		}
		totalBadness += signal.Weight * Clamp01(signal.Badness)
		totalWeight += signal.Weight
	}
	if totalWeight <= 0 {
		return 0, false
	}
	return Clamp01(totalBadness / totalWeight), true
}

// HasNegativeWeight reports whether any of the supplied weights is negative.
func HasNegativeWeight(weights ...float64) bool {
	for _, weight := range weights {
		if weight < 0 {
			return true
		}
	}
	return false
}

// PrefixCacheScoreFromEndpoint provides the exported PrefixCacheScoreFromEndpoint operation.
func PrefixCacheScoreFromEndpoint(endpoint fwkscheduling.Endpoint) float64 {
	if endpoint == nil {
		return 0
	}
	value, ok := endpoint.Get(prefixattr.PrefixCacheMatchInfoKey)
	if !ok {
		return 0
	}
	matchInfo, ok := value.(*prefixattr.PrefixCacheMatchInfo)
	if !ok || matchInfo == nil {
		return 0
	}
	return ComputePrefixCacheScore(matchInfo.MatchBlocks(), matchInfo.TotalBlocks())
}

// EndpointMetrics returns the endpoint's metrics along with whether they are
// present. A nil endpoint or absent metrics yields (nil, false).
func EndpointMetrics(endpoint fwkscheduling.Endpoint) (*fwkdl.Metrics, bool) {
	if endpoint == nil {
		return nil, false
	}
	metrics := endpoint.GetMetrics()
	if metrics == nil {
		return nil, false
	}
	return metrics, true
}

// ZeroMetrics returns the endpoint's metrics when present, or a freshly
// allocated zero-valued Metrics so callers can read fields without nil checks.
func ZeroMetrics(endpoint fwkscheduling.Endpoint) *fwkdl.Metrics {
	if metrics, ok := EndpointMetrics(endpoint); ok {
		return metrics
	}
	return fwkdl.NewMetrics()
}
