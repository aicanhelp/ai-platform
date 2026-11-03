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

package prediction

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"strings"
	"sync"
	"time"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	latencyattr "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/latency"

	"hermes-router/pkg/epp/internal/pdgroup"
	"hermes-router/pkg/epp/internal/score"
	"hermes-router/pkg/epp/internal/utils"
)

const (
	// PluginType is the registered plugin type identifier.
	PluginType            = "prediction-scorer"
	defaultTTFTWeight     = 0.6
	defaultTPOTWeight     = 0.4
	defaultKVWeight       = 1.0
	defaultQueueWeight    = 1.0
	defaultPrefixWeight   = 1.0
	defaultInflightWeight = 1.0
	maxMissingBadness     = 1.0

	// ModeActive dispatches predictions and routes on predicted latency.
	// ModeShadow skips prediction entirely and routes via the snapshot
	// fallback, while the prediction-feature-extractor and lifecycle tracker
	// still collect PredictionInput training data. Shadow is the bootstrap /
	// off-policy collection mode: it needs no predictor sidecar and never lets
	// an unproven model influence routing.
	ModeActive  = "active"
	ModeShadow  = "shadow"
	defaultMode = ModeActive
)

type pluginParameters struct {
	PredictorAddress string          `json:"predictorAddress,omitempty"`
	PredictionMode   string          `json:"predictionMode,omitempty"`
	Timeout          *utils.Duration `json:"timeout,omitempty"`
	MaxBatchSize     int             `json:"maxBatchSize,omitempty"`
	TTFTWeight       *float64        `json:"ttftWeight,omitempty"`
	TPOTWeight       *float64        `json:"tpotWeight,omitempty"`
	KVWeight         *float64        `json:"kvWeight,omitempty"`
	QueueWeight      *float64        `json:"queueWeight,omitempty"`
	PrefixWeight     *float64        `json:"prefixWeight,omitempty"`
	InflightWeight   *float64        `json:"inflightWeight,omitempty"`
}

// Config configures prediction-mode and fallback-mode weights.
// InflightWeight only contributes on the fallback path; the predicted path
// already encodes load via the latency model.
type Config struct {
	// PredictorAddress stores the PredictorAddress value.
	PredictorAddress string
	// Mode is "active" (route on predictions) or "shadow" (collect only,
	// route via fallback). Empty resolves to active.
	Mode string
	// Timeout stores the Timeout value.
	Timeout time.Duration
	// MaxBatchSize stores the MaxBatchSize value.
	MaxBatchSize int
	// PredictorClient stores the PredictorClient value.
	PredictorClient PredictorClient
	// TTFTWeight stores the TTFTWeight value.
	TTFTWeight float64
	// TPOTWeight stores the TPOTWeight value.
	TPOTWeight float64
	// KVWeight stores the KVWeight value.
	KVWeight float64
	// QueueWeight stores the QueueWeight value.
	QueueWeight float64
	// PrefixWeight stores the PrefixWeight value.
	PrefixWeight float64
	// InflightWeight stores the InflightWeight value.
	InflightWeight float64
}

// Scorer ranks endpoints by predicted latency when available, otherwise by local snapshots.
type Scorer struct {
	typedName      fwkplugin.TypedName
	client         PredictorClient
	mode           string
	timeout        time.Duration
	maxBatchSize   int
	ttftWeight     float64
	tpotWeight     float64
	kvWeight       float64
	queueWeight    float64
	prefixWeight   float64
	inflightWeight float64
	rng            *rand.Rand
	randMu         sync.Mutex
}

var _ fwkscheduling.Scorer = (*Scorer)(nil)

// Factory constructs a scorer from user parameters.
func Factory(name string, raw json.RawMessage, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	config, err := parseAndValidateParameters(raw)
	if err != nil {
		return nil, fmt.Errorf("prediction scorer %q: %w", name, err)
	}
	scorer, err := newScorer(name, config)
	if err != nil {
		return nil, fmt.Errorf("prediction scorer %q: %w", name, err)
	}
	registerShutdownClose(handle, scorer.client)
	return scorer, nil
}

// New creates a scorer from already-resolved configuration.
func New(name string, config Config) *Scorer {
	scorer, err := newScorer(name, config)
	if err != nil {
		panic(fmt.Sprintf("prediction scorer %q: %v", name, err))
	}
	return scorer
}

func newScorer(name string, config Config) (*Scorer, error) {
	resolved := resolveConfig(config)
	// Shadow mode never dispatches, so it needs no predictor connection — this
	// lets data collection run before any sidecar/model exists.
	if resolved.PredictorClient == nil && resolved.PredictorAddress != "" && resolved.Mode != ModeShadow {
		client, err := NewGRPCPredictorClient(GRPCPredictorConfig{
			Address:        resolved.PredictorAddress,
			Timeout:        resolved.Timeout,
			MaxMessageSize: defaultMaxMessageSize,
		})
		if err != nil {
			return nil, err
		}
		resolved.PredictorClient = client
	}
	return &Scorer{
		typedName:      fwkplugin.TypedName{Type: PluginType, Name: name},
		client:         resolved.PredictorClient,
		mode:           resolved.Mode,
		timeout:        resolved.Timeout,
		maxBatchSize:   resolved.MaxBatchSize,
		ttftWeight:     resolved.TTFTWeight,
		tpotWeight:     resolved.TPOTWeight,
		kvWeight:       resolved.KVWeight,
		queueWeight:    resolved.QueueWeight,
		prefixWeight:   resolved.PrefixWeight,
		inflightWeight: resolved.InflightWeight,
		rng:            rand.New(rand.NewSource(time.Now().UnixNano())),
	}, nil
}

// TypedName returns the framework plugin type and instance name.
func (s *Scorer) TypedName() fwkplugin.TypedName {
	return s.typedName
}

// Category returns the scheduling scorer category.
func (s *Scorer) Category() fwkscheduling.ScorerCategory {
	return fwkscheduling.Balance
}

// predictionsActive reports whether the scorer should dispatch and route on
// predictions. Shadow mode collects features but always routes via fallback.
func (s *Scorer) predictionsActive() bool {
	return s.mode != ModeShadow
}

// Score assigns scheduling scores to candidate endpoints.
func (s *Scorer) Score(
	ctx context.Context,
	state *fwkscheduling.CycleState,
	request *fwkscheduling.LLMRequest,
	pods []fwkscheduling.Endpoint,
) map[fwkscheduling.Endpoint]float64 {
	_ = state
	logger := log.FromContext(nonNilContext(ctx)).WithName("Scorer.Score")
	// Shadow mode collects PredictionInput (via the feature extractor +
	// lifecycle tracker, which run regardless of this scorer) but never routes
	// on predictions: skip dispatch so every endpoint falls back to snapshot
	// scoring.
	if s.predictionsActive() {
		s.dispatchPredictions(ctx, request, pods)
	}
	scores := make(map[fwkscheduling.Endpoint]float64, len(pods))
	aggregatePods := aggregateEndpoints(pods)
	aggregateMaxQueue := score.MaxWaitingQueue(aggregatePods)
	aggregateMaxInflight := score.MaxInflightTokens(aggregatePods)
	pdLimits := score.CollectPDLimits(pods)
	missingAggregatePredictions := countMissingAggregatePredictions(aggregatePods)
	aggregateUsePredictions := s.predictionsActive() && missingAggregatePredictions == 0
	if s.predictionsActive() && missingAggregatePredictions > 0 {
		logger.V(logutil.DEFAULT).Info(
			"Prediction scorer fell back to snapshot scoring because some aggregate endpoints are missing valid predictions",
			"aggregateEndpoints", len(aggregatePods),
			"missingPredictions", missingAggregatePredictions,
		)
	}

	for _, endpoint := range pods {
		if info, ok := pdgroup.GetDirect(endpoint); ok {
			groupScore, updated := s.scoreGroup(logger, info, pdLimits)
			pdgroup.SetOwned(endpoint, updated)
			if endpoint != nil {
				scores[endpoint] = groupScore
			}
			continue
		}
		if !aggregateUsePredictions && !hasMetrics(endpoint) {
			logger.V(logutil.DEFAULT).Info(
				"Prediction scorer used zero-value metrics because endpoint metrics are missing",
				"endpoint", endpointLogValue(endpoint),
			)
		}
		scores[endpoint] = s.scoreAggregate(endpoint, aggregateMaxQueue, aggregateMaxInflight, aggregateUsePredictions)
	}
	return scores
}

func (s *Scorer) scoreGroup(logger logr.Logger, info *pdgroup.Info, limits score.PDLimits) (float64, *pdgroup.Info) {
	updated := pdgroup.Clone(info)
	if err := pdgroup.Validate(updated); err != nil {
		logger.V(logutil.DEFAULT).Info(
			"Prediction scorer failed open for PD group because group metadata is invalid",
			"groupID", groupIDLogValue(updated),
			"error", err.Error(),
		)
		return 0, updated
	}
	groupID := groupIDLogValue(updated)
	inspection := pdgroup.InspectMembers(
		updated,
		func(endpoint fwkscheduling.Endpoint) bool {
			_, ok := ttftPrediction(endpoint)
			return ok
		},
		func(endpoint fwkscheduling.Endpoint) bool {
			_, ok := tpotPrediction(endpoint)
			return ok
		},
		hasMetrics,
	)
	if s.predictionsActive() && inspection.PredictionsAvailable() {
		return s.predictedGroupScore(updated, limits)
	}
	if s.predictionsActive() && inspection.HasMissingPredictions() {
		logger.V(logutil.DEFAULT).Info(
			"Prediction scorer fell back to snapshot scoring because PD group members are missing valid predictions",
			"groupID", groupID,
			"missingPrefillPredictions", inspection.MissingPrefillPredictions,
			"missingDecodePredictions", inspection.MissingDecodePredictions,
		)
	}
	if inspection.HasMissingMetrics() {
		logger.V(logutil.DEFAULT).Info(
			"Prediction scorer used zero-value metrics for PD group members because metrics are missing",
			"groupID", groupID,
			"missingPrefillMetrics", inspection.MissingPrefillMetrics,
			"missingDecodeMetrics", inspection.MissingDecodeMetrics,
		)
	}
	return s.scoreFallbackGroup(updated, limits)
}

func (s *Scorer) predictedGroupScore(info *pdgroup.Info, limits score.PDLimits) (float64, *pdgroup.Info) {
	bestPrefill, bestPrefillBadness := s.bestByPredictedTTFT(info.PrefillPods)
	bestDecode, bestDecodeBadness := s.bestByPredictedTPOT(info.DecodePods)
	if bestPrefill == nil || bestDecode == nil {
		return s.scoreFallbackGroup(info, limits)
	}
	info.SelectedPrefillPod = bestPrefill
	info.SelectedDecodePod = bestDecode

	groupBadness, _ := score.WeightedBadness([]score.Signal{
		{Name: "ttft", Weight: s.ttftWeight, Badness: bestPrefillBadness},
		{Name: "tpot", Weight: s.tpotWeight, Badness: bestDecodeBadness},
	})
	return score.FromBadness(groupBadness), info
}

func (s *Scorer) scoreFallbackGroup(info *pdgroup.Info, limits score.PDLimits) (float64, *pdgroup.Info) {
	bestPrefill, bestPrefillBadness := s.bestPrefillFallback(
		info.PrefillPods, limits.Prefill.MaxQueue, limits.Prefill.MaxInflight)
	info.SelectedPrefillPod = bestPrefill

	bestDecode, bestDecodeBadness := s.bestDecodeFallback(
		info.DecodePods, limits.Decode.MaxQueue, limits.Decode.MaxInflight)
	info.SelectedDecodePod = bestDecode

	groupBadness, _ := score.WeightedBadness([]score.Signal{
		{Name: "prefill", Weight: s.ttftWeight, Badness: bestPrefillBadness},
		{Name: "decode", Weight: s.tpotWeight, Badness: bestDecodeBadness},
	})
	return score.FromBadness(groupBadness), info
}

func (s *Scorer) bestPrefillFallback(
	pods []fwkscheduling.ScoredEndpoint,
	maxQueue int,
	maxInflight float64,
) (*fwkscheduling.ScoredEndpoint, float64) {
	return score.SelectBestScoredEndpoint(pods, s.nextRandIntn, maxMissingBadness,
		func(endpoint fwkscheduling.Endpoint) (float64, bool) {
			return s.prefillBadness(endpoint, maxQueue, maxInflight), true
		})
}

func (s *Scorer) bestDecodeFallback(
	pods []fwkscheduling.ScoredEndpoint,
	maxQueue int,
	maxInflight float64,
) (*fwkscheduling.ScoredEndpoint, float64) {
	return score.SelectBestScoredEndpoint(pods, s.nextRandIntn, maxMissingBadness,
		func(endpoint fwkscheduling.Endpoint) (float64, bool) {
			return s.decodeBadness(endpoint, maxQueue, maxInflight), true
		})
}

func (s *Scorer) nextRandIntn(count int) int {
	if count <= 1 {
		return 0
	}
	s.randMu.Lock()
	defer s.randMu.Unlock()
	if s.rng == nil {
		s.rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}
	return s.rng.Intn(count)
}

func (s *Scorer) prefillBadness(endpoint fwkscheduling.Endpoint, maxQueue int, maxInflight float64) float64 {
	metrics := score.ZeroMetrics(endpoint)
	badness, _ := score.WeightedBadness([]score.Signal{
		{Name: "kv-cache-usage", Weight: s.kvWeight, Badness: score.Clamp01(metrics.KVCacheUsagePercent)},
		{Name: "waiting-queue", Weight: s.queueWeight, Badness: relativeValue(metrics.WaitingQueueSize, maxQueue)},
		{Name: "prefix-cache-miss", Weight: s.prefixWeight, Badness: 1 - score.PrefixCacheScoreFromEndpoint(endpoint)},
		{
			Name:    "inflight",
			Weight:  s.inflightWeight,
			Badness: score.RelativeFloat(score.InflightTokens(endpoint), maxInflight),
		},
	})
	return badness
}

func (s *Scorer) decodeBadness(endpoint fwkscheduling.Endpoint, maxQueue int, maxInflight float64) float64 {
	metrics := score.ZeroMetrics(endpoint)
	badness, _ := score.WeightedBadness([]score.Signal{
		{Name: "kv-cache-usage", Weight: s.kvWeight, Badness: score.Clamp01(metrics.KVCacheUsagePercent)},
		{Name: "waiting-queue", Weight: s.queueWeight, Badness: relativeValue(metrics.WaitingQueueSize, maxQueue)},
		{
			Name:    "inflight",
			Weight:  s.inflightWeight,
			Badness: score.RelativeFloat(score.InflightTokens(endpoint), maxInflight),
		},
	})
	return badness
}

func (s *Scorer) scoreAggregate(
	endpoint fwkscheduling.Endpoint,
	maxQueue int,
	maxInflight float64,
	usePredictions bool,
) float64 {
	if usePredictions {
		prediction, _ := aggregatePrediction(endpoint)
		badness, _ := score.WeightedBadness([]score.Signal{
			{Name: "ttft", Weight: s.ttftWeight, Badness: score.LatencyBadnessFromMillis(prediction.TTFT())},
			{Name: "tpot", Weight: s.tpotWeight, Badness: score.LatencyBadnessFromMillis(prediction.TPOT())},
		})
		return score.FromBadness(badness)
	}
	metrics := score.ZeroMetrics(endpoint)
	badness, _ := score.WeightedBadness([]score.Signal{
		{Name: "kv-cache-usage", Weight: s.kvWeight, Badness: score.Clamp01(metrics.KVCacheUsagePercent)},
		{Name: "waiting-queue", Weight: s.queueWeight, Badness: relativeValue(metrics.WaitingQueueSize, maxQueue)},
		{Name: "prefix-cache-miss", Weight: s.prefixWeight, Badness: 1 - score.PrefixCacheScoreFromEndpoint(endpoint)},
		{
			Name:    "inflight",
			Weight:  s.inflightWeight,
			Badness: score.RelativeFloat(score.InflightTokens(endpoint), maxInflight),
		},
	})
	return score.FromBadness(badness)
}

func aggregateEndpoints(pods []fwkscheduling.Endpoint) []fwkscheduling.Endpoint {
	aggregate := make([]fwkscheduling.Endpoint, 0, len(pods))
	for _, endpoint := range pods {
		if _, ok := pdgroup.GetDirect(endpoint); ok {
			continue
		}
		aggregate = append(aggregate, endpoint)
	}
	return aggregate
}

func countMissingAggregatePredictions(pods []fwkscheduling.Endpoint) int {
	missing := 0
	for _, endpoint := range pods {
		if _, ok := aggregatePrediction(endpoint); !ok {
			missing++
		}
	}
	return missing
}

// bestByPredictedTTFT picks the prefill pod whose predicted TTFT yields the
// lowest saturating badness. Skips pods missing a TTFT prediction. Per-pod
// Score is set to the linear-flipped badness so downstream readers see "higher
// is better" semantics. The absolute saturating mapping (latency/(latency+1s))
// keeps single-pod groups comparable to multi-pod groups without depending on
// pool composition.
func (s *Scorer) bestByPredictedTTFT(pods []fwkscheduling.ScoredEndpoint) (*fwkscheduling.ScoredEndpoint, float64) {
	return score.SelectBestScoredEndpoint(pods, s.nextRandIntn, 0,
		func(endpoint fwkscheduling.Endpoint) (float64, bool) {
			ttft, ok := ttftPrediction(endpoint)
			if !ok {
				return 0, false
			}
			return score.LatencyBadnessFromMillis(ttft), true
		})
}

// bestByPredictedTPOT is the decode-pool counterpart to bestByPredictedTTFT.
func (s *Scorer) bestByPredictedTPOT(pods []fwkscheduling.ScoredEndpoint) (*fwkscheduling.ScoredEndpoint, float64) {
	return score.SelectBestScoredEndpoint(pods, s.nextRandIntn, 0,
		func(endpoint fwkscheduling.Endpoint) (float64, bool) {
			tpot, ok := tpotPrediction(endpoint)
			if !ok {
				return 0, false
			}
			return score.LatencyBadnessFromMillis(tpot), true
		})
}

func aggregatePrediction(endpoint fwkscheduling.Endpoint) (*latencyattr.LatencyPredictionInfo, bool) {
	if endpoint == nil {
		return nil, false
	}
	value, ok := endpoint.Get(latencyattr.LatencyPredictionInfoKey)
	if !ok {
		return nil, false
	}
	prediction, ok := value.(*latencyattr.LatencyPredictionInfo)
	if !ok || prediction == nil || !prediction.TTFTValid() || !prediction.TPOTValid() {
		return nil, false
	}
	return prediction, true
}

func ttftPrediction(endpoint fwkscheduling.Endpoint) (float64, bool) {
	prediction, ok := endpointPrediction(endpoint)
	if !ok || !prediction.TTFTValid() {
		return 0, false
	}
	return prediction.TTFT(), true
}

func tpotPrediction(endpoint fwkscheduling.Endpoint) (float64, bool) {
	prediction, ok := endpointPrediction(endpoint)
	if !ok || !prediction.TPOTValid() {
		return 0, false
	}
	return prediction.TPOT(), true
}

func endpointPrediction(endpoint fwkscheduling.Endpoint) (*latencyattr.LatencyPredictionInfo, bool) {
	if endpoint == nil {
		return nil, false
	}
	value, ok := endpoint.Get(latencyattr.LatencyPredictionInfoKey)
	if !ok {
		return nil, false
	}
	prediction, ok := value.(*latencyattr.LatencyPredictionInfo)
	if !ok || prediction == nil {
		return nil, false
	}
	return prediction, true
}

func hasMetrics(endpoint fwkscheduling.Endpoint) bool {
	_, ok := score.EndpointMetrics(endpoint)
	return ok
}

func groupIDLogValue(info *pdgroup.Info) string {
	if info == nil || strings.TrimSpace(info.GroupID) == "" {
		return "<unknown>"
	}
	return info.GroupID
}

func endpointLogValue(endpoint fwkscheduling.Endpoint) string {
	if endpoint == nil {
		return "<nil>"
	}
	metadata := endpoint.GetMetadata()
	if metadata == nil {
		return endpoint.String()
	}
	if metadata.PodName != "" {
		return metadata.PodName
	}
	if metadata.NamespacedName.Name != "" {
		return metadata.NamespacedName.String()
	}
	if metadata.Address != "" {
		return metadata.Address
	}
	return endpoint.String()
}

func relativeValue(value, max int) float64 {
	if max <= 0 {
		return 0
	}
	return float64(value) / float64(max)
}

func parseAndValidateParameters(raw json.RawMessage) (Config, error) {
	params := pluginParameters{}
	if err := utils.DecodePluginParameters(raw, &params); err != nil {
		return Config{}, err
	}
	config := resolveConfig(Config{})
	config.PredictorAddress = strings.TrimSpace(params.PredictorAddress)
	if mode := strings.TrimSpace(params.PredictionMode); mode != "" {
		if mode != ModeActive && mode != ModeShadow {
			return Config{}, fmt.Errorf("predictionMode must be %q or %q", ModeActive, ModeShadow)
		}
		config.Mode = mode
	}
	if params.Timeout != nil {
		if params.Timeout.Std() <= 0 {
			return Config{}, fmt.Errorf("timeout must be greater than 0")
		}
		config.Timeout = params.Timeout.Std()
	}
	if params.MaxBatchSize < 0 {
		return Config{}, fmt.Errorf("maxBatchSize must be greater than or equal to 0")
	}
	if params.MaxBatchSize > 0 {
		config.MaxBatchSize = params.MaxBatchSize
	}
	if params.TTFTWeight != nil {
		config.TTFTWeight = *params.TTFTWeight
	}
	if params.TPOTWeight != nil {
		config.TPOTWeight = *params.TPOTWeight
	}
	if params.KVWeight != nil {
		config.KVWeight = *params.KVWeight
	}
	if params.QueueWeight != nil {
		config.QueueWeight = *params.QueueWeight
	}
	if params.PrefixWeight != nil {
		config.PrefixWeight = *params.PrefixWeight
	}
	if params.InflightWeight != nil {
		config.InflightWeight = *params.InflightWeight
	}
	if config.PredictorAddress != "" && !strings.HasPrefix(config.PredictorAddress, "unix://") {
		return Config{}, fmt.Errorf("predictorAddress must use unix://")
	}
	if score.HasNegativeWeight(scorerWeights(config)...) {
		return Config{}, fmt.Errorf("weights must be greater than or equal to 0")
	}
	if config.TTFTWeight+config.TPOTWeight <= 0 {
		return Config{}, fmt.Errorf("ttftWeight and tpotWeight must sum to greater than 0")
	}
	if fallbackBadnessSum(config) <= 0 {
		return Config{}, fmt.Errorf("kvWeight, queueWeight, prefixWeight, and inflightWeight must sum to greater than 0")
	}
	return config, nil
}

func resolveConfig(config Config) Config {
	resolved := config
	if resolved.Mode == "" {
		resolved.Mode = defaultMode
	}
	if resolved.Timeout <= 0 {
		resolved.Timeout = defaultTimeout
	}
	if resolved.MaxBatchSize <= 0 {
		resolved.MaxBatchSize = defaultMaxBatchSize
	}
	if resolved.TTFTWeight+resolved.TPOTWeight <= 0 {
		resolved.TTFTWeight = defaultTTFTWeight
		resolved.TPOTWeight = defaultTPOTWeight
	}
	if fallbackBadnessSum(resolved) <= 0 {
		resolved.KVWeight = defaultKVWeight
		resolved.QueueWeight = defaultQueueWeight
		resolved.PrefixWeight = defaultPrefixWeight
		resolved.InflightWeight = defaultInflightWeight
	}
	return resolved
}

func fallbackBadnessSum(config Config) float64 {
	return config.KVWeight + config.QueueWeight + config.PrefixWeight + config.InflightWeight
}

func scorerWeights(config Config) []float64 {
	return []float64{
		config.TTFTWeight,
		config.TPOTWeight,
		config.KVWeight,
		config.QueueWeight,
		config.PrefixWeight,
		config.InflightWeight,
	}
}

type closeablePredictorClient interface {
	Close() error
}

func registerShutdownClose(handle fwkplugin.Handle, client PredictorClient) {
	closer, ok := client.(closeablePredictorClient)
	if !ok || closer == nil || handle == nil {
		return
	}
	ctx := handle.Context()
	if ctx == nil {
		return
	}
	go func() {
		<-ctx.Done()
		_ = closer.Close()
	}()
}
