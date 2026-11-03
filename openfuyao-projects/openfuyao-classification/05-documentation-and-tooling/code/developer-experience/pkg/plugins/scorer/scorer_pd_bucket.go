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

// Package scorer provides scoring plugins for various routing strategies.
package scorer

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sync"

	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"

	"hermes-router/pkg/plugins/common"
	"hermes-router/pkg/plugins/prerequest"
)

const (
	// PDBucketType defines the plugin type name.
	PDBucketType = "scorer-pd-bucket"

	// requestLenDivisor is used to normalize request length for prefill score.
	requestLenDivisor = 4.0

	// prefillCoeff is the linear coefficient for prefill token estimation.
	prefillCoeff = 0.0345

	// prefillOffset is the constant offset for prefill token estimation.
	prefillOffset = 120.0745

	// epsilon is a small value to prevent division by zero.
	epsilon = 1e-6

	// balanceThreshold defines the spread threshold to switch scoring strategies.
	balanceThreshold = 0.5

	// defaultDecodeCount is the fallback divisor when no decode pods exist.
	defaultDecodeCount = 1

	// defaultTPSum is the fallback TP sum when TP size is not found.
	defaultTPSum = 1.0

	// minParamValue is the lower bound for valid parameters (exclusive).
	minParamValue = 0.0

	// initialZeroTokens represents zero tokens.
	initialZeroTokens = 0.0
)

type scorerPDBucketParameters struct {
	Alpha                float64 `json:"alpha"`
	Beta                 float64 `json:"beta"`
	DecayFactor          float64 `json:"decayFactor"`
	BucketSeperateLength float64 `json:"bucketSeperateLength"`
}

// compile-time type assertion
var _ framework.Scorer = &PDBucket{}
var _ framework.Picker = &PDBucket{}
var _ requestcontrol.PreRequest = &PDBucket{}
var _ requestcontrol.ResponseComplete = &PDBucket{}

// ScorerPDBucketFactory defines the factory function for the ScorerPDBucket.
// It parses parameters from raw JSON and initializes a new PDBucket plugin instance.
func PDBucketFactory(name string, rawParameters json.RawMessage, handle plugins.Handle) (plugins.Plugin, error) {
	parameters := scorerPDBucketParameters{}
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' scorer - %w", PDBucketType, err)
		}
	}

	return NewPDBucket(handle.Context(), &parameters).WithName(name), nil
}

// TypedName returns the typed name of the plugin.
func (pdb *PDBucket) TypedName() plugins.TypedName {
	return pdb.typedName
}

// WithName sets the name of the plugin instance and returns the modified instance.
func (pdb *PDBucket) WithName(name string) *PDBucket {
	pdb.typedName.Name = name
	return pdb
}

// NewPDBucket creates a new PDBucket scorer instance.
// It validates the parameters and initializes internal maps.
func NewPDBucket(ctx context.Context, params *scorerPDBucketParameters) *PDBucket {
	logger := log.FromContext(ctx).WithName("PDBucket.NewPDBucket")

	if err := validateParams(params); err != nil {
		logger.V(logging.DEFAULT).Error(err, "failed to create PDBucket")
		return nil
	}

	return &PDBucket{
		typedName:            plugins.TypedName{Type: PDBucketType},
		alpha:                params.Alpha,
		beta:                 params.Beta,
		decayFactor:          params.DecayFactor,
		bucketSeperateLength: params.BucketSeperateLength,
		activeTokens:         make(map[string]float64),
		prefillTPSize:        make(map[string]int),
		inflight:             make(map[string]*requestRoute),
		mutex:                sync.Mutex{},
	}
}

func validateParams(params *scorerPDBucketParameters) error {
	if params == nil {
		return fmt.Errorf("parameters are nil")
	}
	if params.Alpha <= minParamValue {
		return fmt.Errorf("alpha must be between 0 and 1")
	}
	if params.Beta <= minParamValue {
		return fmt.Errorf("beta must be between 0 and 1")
	}
	if params.DecayFactor <= minParamValue {
		return fmt.Errorf("decayFactor must be between 0 and 1")
	}
	if params.BucketSeperateLength <= minParamValue {
		return fmt.Errorf("bucketSeperateLength must be positive")
	}
	return nil
}

// PDBucket scorer that is based on request length, active tokens and prefill tensor parallel size.
// It implements Scorer, Picker, PreRequest, and ResponseComplete interfaces.
type PDBucket struct {
	typedName            plugins.TypedName
	alpha                float64
	beta                 float64
	decayFactor          float64
	bucketSeperateLength float64

	// store the active tokens for each Prefill Pod and Decode Pod
	activeTokens map[string]float64

	// store the Tensor Parallel size for each Prefill Pod and Decode Pod
	prefillTPSize map[string]int

	// in-flight routing records: requestID -> routed pods and token deltas
	inflight map[string]*requestRoute

	mutex sync.Mutex
}

// requestRoute records the triplet routing and token increments for a request
type requestRoute struct {
	leaderPodName      string
	prefillPodName     string
	decodePodName      string
	prefillTokensAdded float64
	decodeTokensAdded  float64
}

// PreRequest updates internal state before a request is processed.
// It extracts routing info, calculates token costs, and updates active load.
func (pdb *PDBucket) PreRequest(
	ctx context.Context,
	request *types.LLMRequest,
	schedulingResult *types.SchedulingResult,
) {
	logger := log.FromContext(ctx).WithName("PDBucket.PreRequest")

	route := prerequest.ExtractPrimaryPDRoute(ctx, schedulingResult)
	if route == nil {
		logger.V(logging.DEFAULT).Info("no primary pd route found")
		return
	}

	requestLength, err := common.RequestLengthOf(request)
	if err != nil {
		logger.V(logging.DEFAULT).Error(err, "failed to get request length")
		return
	}

	pdb.updateInflightState(request, route, requestLength)

	prerequest.InjectPDHeadersFromRoute(ctx, request, route)
	logger.V(logging.DEFAULT).Info("injected PD prefill/decode headers",
		"prefill", route.Prefill, "decode", route.Decode)
}

func (pdb *PDBucket) updateInflightState(
	request *types.LLMRequest,
	route *prerequest.PDRoute,
	requestLength int,
) {
	pdb.mutex.Lock()
	defer pdb.mutex.Unlock()

	lengthScore := float64(requestLength) / requestLenDivisor
	prefillScore := lengthScore*prefillCoeff + prefillOffset

	pdb.activeTokens[route.Prefill.PodName] += prefillScore
	pdb.activeTokens[route.Decode.PodName] += float64(requestLength)

	leaderName := ""
	if route.Leader != nil {
		leaderName = route.Leader.PodName
	}

	pdb.inflight[request.RequestId] = &requestRoute{
		leaderPodName:      leaderName,
		prefillPodName:     route.Prefill.PodName,
		decodePodName:      route.Decode.PodName,
		prefillTokensAdded: prefillScore,
		decodeTokensAdded:  float64(requestLength),
	}
}

// ResponseComplete cleans up internal state after a request completes.
// It releases the token load reserved for the request.
func (pdb *PDBucket) ResponseComplete(
	ctx context.Context,
	request *types.LLMRequest,
	response *requestcontrol.Response,
	targetPod *backend.Pod,
) {
	logger := log.FromContext(ctx).WithName("PDBucket.ResponseComplete")
	if request == nil {
		logger.V(logging.DEFAULT).Info("request is nil on ResponseComplete")
		return
	}
	if request.RequestId == "" {
		logger.V(logging.DEFAULT).Info("request id is empty on ResponseComplete")
		return
	}

	pdb.releaseInflightTokens(ctx, request.RequestId)
}

func (pdb *PDBucket) releaseInflightTokens(ctx context.Context, reqID string) {
	pdb.mutex.Lock()
	defer pdb.mutex.Unlock()

	logger := log.FromContext(ctx)
	route, ok := pdb.inflight[reqID]
	if !ok {
		logger.V(logging.DEFAULT).Info("inflight route not found on completion", "requestID", reqID)
		return
	}

	if route.prefillPodName != "" {
		pdb.activeTokens[route.prefillPodName] -= route.prefillTokensAdded
		if pdb.activeTokens[route.prefillPodName] < initialZeroTokens {
			pdb.activeTokens[route.prefillPodName] = initialZeroTokens
		}
	}
	if route.decodePodName != "" {
		pdb.activeTokens[route.decodePodName] -= route.decodeTokensAdded
		if pdb.activeTokens[route.decodePodName] < initialZeroTokens {
			pdb.activeTokens[route.decodePodName] = initialZeroTokens
		}
	}
	delete(pdb.inflight, reqID)
}

func (pdb *PDBucket) scorePDGroupList(pdGroupList common.PDGroupList, requestLength int) *common.PDGroupList {
	groupLoads, groupPrefillTP, bucketSeperateLength := pdb.calculateGroupLoadsAndPrefillTP(pdGroupList)

	activeFactors, minActiveFactor, maxActiveFactor := pdb.calculateActiveFactors(groupLoads)

	var finalScores []float64
	if pdb.isNotLoadBalance(minActiveFactor, maxActiveFactor) {
		finalScores = activeFactors
	} else {
		totalPrefillTP := 0.0
		for _, sum := range groupPrefillTP {
			totalPrefillTP += sum
		}
		tpFactors := pdb.calculateTPFactors(groupPrefillTP, totalPrefillTP, bucketSeperateLength, requestLength)
		finalScores = pdb.calculateFinalScores(activeFactors, tpFactors)
	}

	for i := range pdGroupList.Groups {
		if i < len(finalScores) {
			pdGroupList.Groups[i].LeaderPod.Score = finalScores[i]
		}
	}

	return &pdGroupList
}

// calculateGroupLoadsAndPrefillTP calculates metrics for each PD group.
func (pdb *PDBucket) calculateGroupLoadsAndPrefillTP(
	pdGroupList common.PDGroupList,
) ([]float64, []float64, float64) {
	groupLoads := make([]float64, len(pdGroupList.Groups))
	groupPrefillTP := make([]float64, len(pdGroupList.Groups))

	pdb.mutex.Lock()
	defer pdb.mutex.Unlock()
	bucketSeperateLength := pdb.bucketSeperateLength

	for i := range pdGroupList.Groups {
		group := pdGroupList.Groups[i]
		load, tpSum := pdb.calculateSingleGroupMetrics(group)
		groupLoads[i] = load
		groupPrefillTP[i] = tpSum
	}

	return groupLoads, groupPrefillTP, bucketSeperateLength
}

func (pdb *PDBucket) calculateSingleGroupMetrics(group common.PDGroup) (float64, float64) {
	load := 0.0
	tpSum := 0.0

	for _, pod := range group.PrefillPods {
		if pod.Pod == nil || pod.GetPod() == nil {
			continue
		}
		podName := pod.GetPod().PodName
		load += pdb.activeTokens[podName]
		if tpSize, ok := pdb.prefillTPSize[podName]; ok && tpSize > 0 {
			tpSum += float64(tpSize)
		} else {
			tpSum += defaultTPSum
		}
	}

	for _, pod := range group.DecodePods {
		if pod.Pod == nil || pod.GetPod() == nil {
			continue
		}
		load += pdb.activeTokens[pod.GetPod().PodName]
	}

	decodeCount := len(group.DecodePods)
	if decodeCount == 0 {
		decodeCount = defaultDecodeCount
	}

	return load / float64(decodeCount), tpSum
}

func (pdb *PDBucket) calculateActiveFactors(groupLoads []float64) ([]float64, float64, float64) {
	activeFactors := make([]float64, len(groupLoads))
	if len(groupLoads) == 0 {
		return activeFactors, 0, 0
	}

	totalLoad := 0.0
	for _, load := range groupLoads {
		totalLoad += load
	}

	minActiveFactor := math.MaxFloat64
	maxActiveFactor := -math.MaxFloat64
	denominator := totalLoad + epsilon

	for i, load := range groupLoads {
		activeFactor := load / denominator
		activeFactors[i] = activeFactor

		if activeFactor < minActiveFactor {
			minActiveFactor = activeFactor
		}
		if activeFactor > maxActiveFactor {
			maxActiveFactor = activeFactor
		}
	}

	return activeFactors, minActiveFactor, maxActiveFactor
}

func (pdb *PDBucket) isNotLoadBalance(minActiveFactor, maxActiveFactor float64) bool {
	return maxActiveFactor-minActiveFactor > balanceThreshold
}

func (pdb *PDBucket) calculateTPFactors(
	groupPrefillTP []float64,
	totalPrefillTP float64,
	bucketSeperateLength float64,
	requestLength int,
) []float64 {
	tpFactors := make([]float64, len(groupPrefillTP))
	if len(groupPrefillTP) == 0 {
		return tpFactors
	}

	avgTPShare := 1.0 / float64(len(groupPrefillTP))
	requestLengthFactor := 1.0
	if bucketSeperateLength > 0 {
		requestLengthFactor -= float64(requestLength) / bucketSeperateLength
	}

	for i, tpSum := range groupPrefillTP {
		tpShare := 0.0
		if totalPrefillTP > 0 {
			tpShare = tpSum / totalPrefillTP
		}
		tpAtom := tpShare - avgTPShare
		tpFactors[i] = tpAtom * requestLengthFactor
	}

	return tpFactors
}

func (pdb *PDBucket) calculateFinalScores(activeFactors, tpFactors []float64) []float64 {
	finalScores := make([]float64, len(activeFactors))
	for i := range activeFactors {
		tpFactor := 0.0
		if i < len(tpFactors) {
			tpFactor = tpFactors[i]
		}
		finalScores[i] = pdb.alpha*activeFactors[i] + pdb.beta*tpFactor
	}
	return finalScores
}

func (pdb *PDBucket) updateBucketSeperateLength(requestLength int) {
	pdb.mutex.Lock()
	defer pdb.mutex.Unlock()
	diff := float64(requestLength) - pdb.bucketSeperateLength
	pdb.bucketSeperateLength = pdb.bucketSeperateLength + (1-pdb.decayFactor)*diff
}

// Score calculates scores for PDGroups and updates the cycle state.
// It returns an empty map as per interface requirement.
func (pdb *PDBucket) Score(
	ctx context.Context,
	cycleState *types.CycleState,
	request *types.LLMRequest,
	pods []types.Pod,
) map[types.Pod]float64 {
	logger := log.FromContext(ctx).WithName("PDBucket.Score")
	podScoreMap := make(map[types.Pod]float64, len(pods))

	pdGroupList, err := types.ReadCycleStateKey[*common.PDGroupList](cycleState, common.PDGroupsCycleStateKey)
	if err != nil {
		logger.V(logging.DEFAULT).Error(err, "failed to read PDGroupList from cycleState")
		return podScoreMap
	}

	requestLength, err := common.RequestLengthOf(request)
	if err != nil {
		logger.V(logging.DEFAULT).Error(err, "failed to get request length")
		return podScoreMap
	}

	pdb.updateBucketSeperateLength(requestLength)
	pdGroupList = pdb.scorePDGroupList(*pdGroupList, requestLength)

	for i := range pdGroupList.Groups {
		logger.V(logging.DEBUG).Info("pdGroup",
			"groupID", pdGroupList.Groups[i].ID,
			"groupScore", pdGroupList.Groups[i].LeaderPod.Score)
	}

	cycleState.Write(common.PDGroupsCycleStateKey, pdGroupList)
	return podScoreMap
}

// Pick selects the best PDGroup and pods for the request.
func (pdb *PDBucket) Pick(
	ctx context.Context,
	cycleState *types.CycleState,
	scoredPods []*types.ScoredPod,
) *types.ProfileRunResult {
	logger := log.FromContext(ctx).WithName("PDBucket.Pick")

	pdGroupList, err := types.ReadCycleStateKey[*common.PDGroupList](cycleState, common.PDGroupsCycleStateKey)
	if err != nil {
		logger.V(logging.DEFAULT).Error(err, "failed to read PDGroupList from cycleState")
		return &types.ProfileRunResult{}
	}

	if pdGroupList == nil || len(pdGroupList.Groups) == 0 {
		logger.V(logging.DEFAULT).Info("PDGroupList is empty")
		return &types.ProfileRunResult{}
	}

	selectedGroup := pdb.findBestPDGroup(pdGroupList)
	if selectedGroup == nil {
		logger.V(logging.DEFAULT).Info("no valid PDGroup found")
		return &types.ProfileRunResult{}
	}

	return pdb.constructProfileResult(ctx, selectedGroup)
}

func (pdb *PDBucket) findBestPDGroup(list *common.PDGroupList) *common.PDGroup {
	minGroupScore := math.MaxFloat64
	var selectedGroup *common.PDGroup

	for i := range list.Groups {
		group := &list.Groups[i]
		if group.LeaderPod.Score < minGroupScore {
			minGroupScore = group.LeaderPod.Score
			selectedGroup = group
		}
	}
	return selectedGroup
}

func (pdb *PDBucket) constructProfileResult(
	ctx context.Context,
	group *common.PDGroup,
) *types.ProfileRunResult {
	logger := log.FromContext(ctx)

	if group.LeaderPod.Pod == nil || group.LeaderPod.GetPod() == nil {
		logger.V(logging.DEFAULT).Info("selected PDGroup has no valid LeaderPod")
		return &types.ProfileRunResult{}
	}

	minPrefillPod, minPrefillTokens := pdb.pickMinActiveTokenPod(group.PrefillPods)
	if minPrefillPod == nil || minPrefillPod.GetPod() == nil {
		logger.V(logging.DEFAULT).Info("selected PDGroup has no valid Prefill Pod")
		return &types.ProfileRunResult{}
	}

	minDecodePod, minDecodeTokens := pdb.pickMinActiveTokenPod(group.DecodePods)
	if minDecodePod == nil || minDecodePod.GetPod() == nil {
		logger.V(logging.DEFAULT).Info("selected PDGroup has no valid Decode Pod")
		return &types.ProfileRunResult{}
	}

	logger.V(logging.DEFAULT).Info("selected PDGroup and pods",
		"groupID", group.ID,
		"groupScore", group.LeaderPod.Score,
		"leaderPod", group.LeaderPod.GetPod().PodName,
		"prefillPod", minPrefillPod.GetPod().PodName,
		"prefillTokens", minPrefillTokens,
		"decodePod", minDecodePod.GetPod().PodName,
		"decodeTokens", minDecodeTokens)

	return &types.ProfileRunResult{
		TargetPods: []types.Pod{
			group.LeaderPod.Pod,
			minPrefillPod.Pod,
			minDecodePod.Pod,
		},
	}
}

func (pdb *PDBucket) pickMinActiveTokenPod(pods []types.ScoredPod) (*types.ScoredPod, float64) {
	pdb.mutex.Lock()
	defer pdb.mutex.Unlock()
	var selected *types.ScoredPod
	minTokens := math.MaxFloat64

	for i := range pods {
		pod := &pods[i]
		if pod.Pod == nil || pod.GetPod() == nil {
			continue
		}
		podName := pod.GetPod().PodName
		tokens := pdb.activeTokens[podName]
		if tokens < minTokens {
			minTokens = tokens
			selected = pod
		}
	}

	return selected, minTokens
}
