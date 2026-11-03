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

// Package bucket provides the PD bucket scorer plugin.
// It only scores validated PD leader candidates and fails open when mixed or
// incomplete endpoint metadata reaches the scorer.
package bucket

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	"hermes-router/pkg/epp/internal/endpointmeta"
	"hermes-router/pkg/epp/internal/pdgroup"
	"hermes-router/pkg/epp/internal/score"
	"hermes-router/pkg/epp/internal/utils"
)

const (
	// PDBucketType is an exported constant used by Hermes Router.
	PDBucketType = "pd-bucket-scorer"

	epsilon            = 1e-6
	balanceThreshold   = 0.5
	defaultDecodeCount = 1
	defaultTPSum       = 1.0
	minParamValue      = 0.0
)

type scorerPDBucketParameters struct {
	Alpha                float64 `json:"alpha"`
	Beta                 float64 `json:"beta"`
	DecayFactor          float64 `json:"decayFactor"`
	BucketSeparateLength float64 `json:"bucketSeparateLength,omitempty"`
	TPSizeLabelKey       string  `json:"tpSizeLabelKey,omitempty"`
}

func (p *scorerPDBucketParameters) separateLength() float64 {
	if p == nil {
		return 0
	}
	return p.BucketSeparateLength
}

var _ fwkscheduling.Scorer = &PDBucket{}

// PDBucketFactory provides the exported PDBucketFactory operation.
func PDBucketFactory(name string, rawParameters json.RawMessage, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	parameters := scorerPDBucketParameters{}
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' scorer - %w", PDBucketType, err)
		}
	}
	if err := validateParams(&parameters); err != nil {
		return nil, fmt.Errorf("invalid parameters for the '%s' scorer - %w", PDBucketType, err)
	}

	return NewPDBucket(handle.Context(), &parameters).WithName(name), nil
}

// TypedName returns the framework plugin type and instance name.
func (pdb *PDBucket) TypedName() fwkplugin.TypedName {
	return pdb.typedName
}

// Category returns the scheduling scorer category.
func (pdb *PDBucket) Category() fwkscheduling.ScorerCategory {
	return fwkscheduling.Balance
}

// WithName provides the exported WithName operation.
func (pdb *PDBucket) WithName(name string) *PDBucket {
	pdb.typedName.Name = name
	return pdb
}

// NewPDBucket constructs a PDBucket instance.
func NewPDBucket(ctx context.Context, params *scorerPDBucketParameters) *PDBucket {
	logger := log.FromContext(ctx).WithName("PDBucket.NewPDBucket")

	if err := validateParams(params); err != nil {
		logger.V(logutil.DEFAULT).Error(err, "failed to create PDBucket")
		return nil
	}

	bucket := &PDBucket{
		typedName:      fwkplugin.TypedName{Type: PDBucketType},
		alpha:          params.Alpha,
		beta:           params.Beta,
		decayFactor:    params.DecayFactor,
		tpSizeLabelKey: strings.TrimSpace(params.TPSizeLabelKey),
		rng:            rand.New(rand.NewSource(time.Now().UnixNano())),
	}
	bucket.bucketSeparateLengthBits.Store(math.Float64bits(params.separateLength()))
	return bucket
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
	if params.separateLength() <= minParamValue {
		return fmt.Errorf("bucketSeparateLength must be positive")
	}
	return nil
}

// PDBucket describes PDBucket data used by Hermes Router.
type PDBucket struct {
	typedName                fwkplugin.TypedName
	alpha                    float64
	beta                     float64
	decayFactor              float64
	tpSizeLabelKey           string
	rng                      *rand.Rand
	randMu                   sync.Mutex
	bucketSeparateLengthBits atomic.Uint64
}

type leaderGroup struct {
	endpoint fwkscheduling.Endpoint
	info     *pdgroup.Info
}

func (pdb *PDBucket) scoreLeaderGroups(
	ctx context.Context,
	pods []fwkscheduling.Endpoint,
	requestLength int,
) map[fwkscheduling.Endpoint]float64 {
	logger := log.FromContext(ctx).WithName("PDBucket.scoreLeaderGroups")
	podScoreMap := make(map[fwkscheduling.Endpoint]float64, len(pods))
	groups := pdb.prepareLeaderGroups(logger, pods)
	if len(groups) == 0 {
		return podScoreMap
	}

	groupLoads, groupPrefillTP, bucketSeparateLength := pdb.calculateGroupLoadsAndPrefillTP(groups)
	activeFactors, minActiveFactor, maxActiveFactor := pdb.calculateActiveFactors(groupLoads)

	finalScores := activeFactors
	if !pdb.isNotLoadBalance(minActiveFactor, maxActiveFactor) {
		totalPrefillTP := 0.0
		for _, sum := range groupPrefillTP {
			totalPrefillTP += sum
		}
		tpFactors := pdb.calculateTPFactors(groupPrefillTP, totalPrefillTP, bucketSeparateLength, requestLength)
		finalScores = pdb.calculateFinalScores(activeFactors, tpFactors)
	}

	for i := range groups {
		if i >= len(finalScores) || groups[i].endpoint == nil {
			continue
		}
		podScoreMap[groups[i].endpoint] = finalScores[i]
		logger.V(logutil.DEBUG).Info("pd leader group scored",
			"groupID", groups[i].info.GroupID,
			"leaderPod", endpointmeta.EndpointPodName(groups[i].endpoint),
			"groupScore", finalScores[i],
			"prefillPod", endpointmeta.EndpointPodName(groups[i].info.SelectedPrefillPod.Endpoint),
			"decodePod", endpointmeta.EndpointPodName(groups[i].info.SelectedDecodePod.Endpoint))
	}

	return podScoreMap
}

func (pdb *PDBucket) prepareLeaderGroups(logger logr.Logger, pods []fwkscheduling.Endpoint) []leaderGroup {
	groups := make([]leaderGroup, 0, len(pods))
	for _, pod := range pods {
		if pod == nil {
			continue
		}

		info, ok := pdgroup.GetDirect(pod)
		if !ok {
			logger.V(logutil.DEBUG).Info("pd group info missing on leader endpoint",
				"leaderPod", endpointmeta.EndpointPodName(pod))
			continue
		}

		updated := pdgroup.Clone(info)
		updated.SelectedPrefillPod = nil
		updated.SelectedDecodePod = nil
		if err := pdgroup.Validate(updated); err != nil {
			logger.V(logutil.DEFAULT).Info("invalid pd group info on leader endpoint",
				"leaderPod", endpointmeta.EndpointPodName(pod),
				"error", err.Error())
			pdgroup.SetOwned(pod, updated)
			continue
		}

		selectedPrefill, _ := pdb.pickMinInflightPod(updated.PrefillPods)
		selectedDecode, _ := pdb.pickMinInflightPod(updated.DecodePods)
		if selectedPrefill == nil || selectedPrefill.Endpoint == nil ||
			selectedDecode == nil || selectedDecode.Endpoint == nil {
			logger.V(logutil.DEFAULT).Info("pd group info has no selectable execution pods",
				"groupID", updated.GroupID,
				"leaderPod", endpointmeta.EndpointPodName(pod))
			pdgroup.SetOwned(pod, updated)
			continue
		}

		selectedPrefillCopy := endpointmeta.CloneScoredEndpoint(*selectedPrefill)
		selectedDecodeCopy := endpointmeta.CloneScoredEndpoint(*selectedDecode)
		updated.SelectedPrefillPod = &selectedPrefillCopy
		updated.SelectedDecodePod = &selectedDecodeCopy
		pdgroup.SetOwned(pod, updated)

		groups = append(groups, leaderGroup{endpoint: pod, info: updated})
	}
	return groups
}

func (pdb *PDBucket) calculateGroupLoadsAndPrefillTP(
	groups []leaderGroup,
) ([]float64, []float64, float64) {
	groupLoads := make([]float64, len(groups))
	groupPrefillTP := make([]float64, len(groups))
	bucketSeparateLength := pdb.separateLength()

	for i := range groups {
		if groups[i].info == nil {
			continue
		}
		load, tpSum := pdb.calculateSingleGroupMetrics(groups[i].info)
		groupLoads[i] = load
		groupPrefillTP[i] = tpSum
	}

	return groupLoads, groupPrefillTP, bucketSeparateLength
}

func (pdb *PDBucket) calculateSingleGroupMetrics(group *pdgroup.Info) (float64, float64) {
	load := 0.0
	tpSum := 0.0
	if group == nil {
		return load, tpSum
	}

	for _, pod := range group.PrefillPods {
		if pod.Endpoint == nil {
			continue
		}
		load += score.InflightTokens(pod.Endpoint)
		tpSum += pdb.prefillTPSum(pod.Endpoint)
	}

	for _, pod := range group.DecodePods {
		if pod.Endpoint == nil {
			continue
		}
		load += score.InflightTokens(pod.Endpoint)
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
	bucketSeparateLength float64,
	requestLength int,
) []float64 {
	tpFactors := make([]float64, len(groupPrefillTP))
	if len(groupPrefillTP) == 0 {
		return tpFactors
	}

	avgTPShare := 1.0 / float64(len(groupPrefillTP))
	requestLengthFactor := 1.0
	if bucketSeparateLength > 0 {
		requestLengthFactor -= float64(requestLength) / bucketSeparateLength
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

func (pdb *PDBucket) updateBucketSeparateLength(requestLength int) {
	for {
		currentBits := pdb.bucketSeparateLengthBits.Load()
		current := math.Float64frombits(currentBits)
		diff := float64(requestLength) - current
		next := current + (1-pdb.decayFactor)*diff
		if pdb.bucketSeparateLengthBits.CompareAndSwap(currentBits, math.Float64bits(next)) {
			return
		}
	}
}

func (pdb *PDBucket) separateLength() float64 {
	return math.Float64frombits(pdb.bucketSeparateLengthBits.Load())
}

// Score assigns scheduling scores to candidate endpoints.
func (pdb *PDBucket) Score(
	ctx context.Context,
	cycleState *fwkscheduling.CycleState,
	request *fwkscheduling.LLMRequest,
	pods []fwkscheduling.Endpoint,
) map[fwkscheduling.Endpoint]float64 {
	_ = cycleState
	logger := log.FromContext(ctx).WithName("PDBucket.Score")
	if err := validateLeaderOnlyInputs(pods); err != nil {
		logger.V(logutil.DEFAULT).Info("pd bucket scorer skipped unsupported candidates", "reason", err.Error())
		return map[fwkscheduling.Endpoint]float64{}
	}

	requestLength, err := utils.RequestLengthOf(request)
	if err != nil {
		logger.V(logutil.DEFAULT).Error(err, "failed to get request length")
		return map[fwkscheduling.Endpoint]float64{}
	}

	pdb.updateBucketSeparateLength(requestLength)
	return pdb.scoreLeaderGroups(ctx, pods, requestLength)
}

func validateLeaderOnlyInputs(pods []fwkscheduling.Endpoint) error {
	for _, pod := range pods {
		if pod == nil {
			continue
		}
		info, ok := pdgroup.GetDirect(pod)
		if !ok {
			return fmt.Errorf("endpoint %q is missing PD group info", endpointmeta.EndpointPodName(pod))
		}
		if err := pdgroup.Validate(info); err != nil {
			return fmt.Errorf("endpoint %q has invalid PD group info: %w", endpointmeta.EndpointPodName(pod), err)
		}
	}
	return nil
}

func (pdb *PDBucket) pickMinInflightPod(pods []fwkscheduling.ScoredEndpoint) (*fwkscheduling.ScoredEndpoint, float64) {
	tiedIndices := make([]int, 0, 1)
	minTokens := math.MaxFloat64

	for i := range pods {
		pod := &pods[i]
		if pod.Endpoint == nil {
			continue
		}
		tokens := score.InflightTokens(pod.Endpoint)
		switch {
		case len(tiedIndices) == 0 || tokens < minTokens:
			minTokens = tokens
			tiedIndices = []int{i}
		case tokens == minTokens:
			tiedIndices = append(tiedIndices, i)
		default:
			continue
		}
	}

	if len(tiedIndices) == 0 {
		return nil, minTokens
	}
	return &pods[utils.PickTiedIndex(tiedIndices, pdb.nextRandIntn)], minTokens
}

func (pdb *PDBucket) nextRandIntn(count int) int {
	if count <= 1 {
		return 0
	}
	pdb.randMu.Lock()
	defer pdb.randMu.Unlock()
	if pdb.rng == nil {
		pdb.rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}
	return pdb.rng.Intn(count)
}

func (pdb *PDBucket) prefillTPSum(endpoint fwkscheduling.Endpoint) float64 {
	if endpoint == nil || pdb.tpSizeLabelKey == "" {
		return defaultTPSum
	}
	value, ok := endpointmeta.EndpointLabelValue(endpoint, pdb.tpSizeLabelKey)
	if !ok {
		return defaultTPSum
	}
	typedValue, err := strconv.Atoi(strings.TrimSpace(value))
	if err != nil || typedValue <= 0 {
		return defaultTPSum
	}
	return float64(typedValue)
}
