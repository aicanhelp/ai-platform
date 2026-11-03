/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Package prediction provides latency prediction scheduling support.
package prediction

import (
	"context"
	"fmt"
	"strings"

	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	latencyattr "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/latency"

	predictioninputattr "hermes-router/pkg/epp/framework/plugins/datalayer/attribute/predictioninput"
	"hermes-router/pkg/epp/internal/pdgroup"
	"hermes-router/pkg/epp/internal/predictiondata"
)

func (s *Scorer) dispatchPredictions(
	ctx context.Context,
	request *fwkscheduling.LLMRequest,
	pods []fwkscheduling.Endpoint,
) {
	logger := log.FromContext(nonNilContext(ctx)).WithName("Scorer.dispatchPredictions")
	if s == nil || s.client == nil {
		return
	}
	if request == nil || strings.TrimSpace(request.TargetModel) == "" {
		logger.V(logutil.DEBUG).Info("Skipping prediction call because target model is empty")
		return
	}
	features := collectFeatureInputs(pods)
	if features.stats.skippedMissingSnapshots > 0 {
		logger.V(logutil.DEFAULT).Info(
			"Prediction scorer skipped endpoints with missing or invalid feature snapshots; "+
				"affected endpoints will fail open to fallback scoring",
			"skippedEndpoints", features.stats.skippedMissingSnapshots,
			"collectedInputs", len(features.inputs),
		)
	}
	if len(features.inputs) == 0 {
		logger.V(logutil.DEBUG).Info("Skipping prediction because no endpoint exposed a feature snapshot")
		return
	}

	predictCtx, cancel := context.WithTimeout(nonNilContext(ctx), s.timeout)
	defer cancel()
	outputs, err := predictBatches(predictCtx, s.client, request.TargetModel, features.inputs, s.maxBatchSize)
	if err != nil {
		logger.V(logutil.DEBUG).Info("Skipping prediction because predictor client failed", "error", err.Error())
		return
	}
	publishStats := publishOutputs(outputs, features.endpointByInputID)
	if publishStats.invalidOutputs > 0 || publishStats.unknownInputIDs > 0 {
		logger.V(logutil.DEFAULT).Info(
			"Prediction scorer ignored predictor outputs that did not match the expected schema; "+
				"affected endpoints will fail open to fallback scoring",
			"invalidOutputs", publishStats.invalidOutputs,
			"unknownInputIDs", publishStats.unknownInputIDs,
		)
	}
	// Info comes from Endpoint.Get, which clones attributes. Write the
	// updated group clone back so the scoring pass reads member predictions.
	publishGroups(features.groups)
}

// collectFeatureInputs reads PredictionInputKey from aggregate candidates and
// PD group members, then renumbers each snapshot's InputID so predictor
// responses can be correlated back to endpoints by batch-local position.
type featureCollectionStats struct {
	skippedMissingSnapshots int
}

type featureInputCollection struct {
	inputs            []predictiondata.Input
	endpointByInputID map[int]fwkscheduling.Endpoint
	groups            []predictionGroup
	stats             featureCollectionStats
}

func collectFeatureInputs(endpoints []fwkscheduling.Endpoint) featureInputCollection {
	inputs := make([]predictiondata.Input, 0, len(endpoints))
	endpointByInputID := make(map[int]fwkscheduling.Endpoint, len(endpoints))
	groups := make([]predictionGroup, 0)
	seen := make(map[string]struct{}, len(endpoints))
	stats := featureCollectionStats{}

	appendEndpoint := func(endpoint fwkscheduling.Endpoint) {
		if endpoint == nil {
			return
		}
		key := endpointKey(endpoint)
		if _, ok := seen[key]; ok {
			return
		}
		seen[key] = struct{}{}
		snapshot, ok := readInputSnapshot(endpoint)
		if !ok {
			stats.skippedMissingSnapshots++
			return
		}
		input := snapshot.Input
		input.InputID = len(inputs)
		inputs = append(inputs, input)
		endpointByInputID[input.InputID] = endpoint
	}

	for _, endpoint := range endpoints {
		if info, ok := pdgroup.GetDirect(endpoint); ok {
			collectGroupFeatureInputs(info, appendEndpoint)
			groups = append(groups, predictionGroup{leader: endpoint, info: info})
			continue
		}
		appendEndpoint(endpoint)
	}
	return featureInputCollection{
		inputs:            inputs,
		endpointByInputID: endpointByInputID,
		groups:            groups,
		stats:             stats,
	}
}

type predictionGroup struct {
	leader fwkscheduling.Endpoint
	info   *pdgroup.Info
}

func endpointKey(endpoint fwkscheduling.Endpoint) string {
	if endpoint == nil {
		return ""
	}
	metadata := endpoint.GetMetadata()
	if metadata == nil {
		return endpoint.String()
	}
	if metadata.PodName != "" {
		return fmt.Sprintf("pod://%s/%s", metadata.NamespacedName.Namespace, metadata.PodName)
	}
	if metadata.Address != "" || metadata.Port != "" {
		return fmt.Sprintf("addr://%s:%s", metadata.Address, metadata.Port)
	}
	return endpoint.String()
}

func collectGroupFeatureInputs(info *pdgroup.Info, appendEndpoint func(fwkscheduling.Endpoint)) {
	if info == nil || appendEndpoint == nil {
		return
	}
	for _, prefill := range info.PrefillPods {
		appendEndpoint(prefill.Endpoint)
	}
	for _, decode := range info.DecodePods {
		appendEndpoint(decode.Endpoint)
	}
}

func readInputSnapshot(endpoint fwkscheduling.Endpoint) (*predictioninputattr.InputSnapshot, bool) {
	if endpoint == nil {
		return nil, false
	}
	value, ok := endpoint.Get(predictioninputattr.PredictionInputKey)
	if !ok {
		return nil, false
	}
	snapshot, ok := value.(*predictioninputattr.InputSnapshot)
	if !ok || snapshot == nil {
		return nil, false
	}
	return snapshot, true
}

type publishOutputStats struct {
	invalidOutputs  int
	unknownInputIDs int
}

func publishOutputs(
	outputs []predictiondata.Output,
	endpointByInputID map[int]fwkscheduling.Endpoint,
) publishOutputStats {
	stats := publishOutputStats{}
	for _, output := range outputs {
		endpoint, ok := endpointByInputID[output.InputID]
		if !ok {
			stats.unknownInputIDs++
			continue
		}
		if !validPredictionOutput(output) {
			stats.invalidOutputs++
			continue
		}
		endpoint.Put(
			latencyattr.LatencyPredictionInfoKey,
			latencyattr.NewLatencyPredictionInfo(true, true, 0, 0, output.PredictedTTFTMillis, output.PredictedTPOTMillis),
		)
	}
	return stats
}

func publishGroups(groups []predictionGroup) {
	for _, group := range groups {
		if group.leader == nil || group.info == nil {
			continue
		}
		pdgroup.SetOwned(group.leader, group.info)
	}
}

func predictBatches(
	ctx context.Context,
	client PredictorClient,
	targetModel string,
	inputs []predictiondata.Input,
	maxBatchSize int,
) ([]predictiondata.Output, error) {
	if client == nil {
		return nil, nil
	}
	if maxBatchSize <= 0 {
		maxBatchSize = defaultMaxBatchSize
	}
	outputs := make([]predictiondata.Output, 0, len(inputs))
	for start := 0; start < len(inputs); start += maxBatchSize {
		end := min(start+maxBatchSize, len(inputs))
		batchOutputs, err := client.Predict(ctx, targetModel, inputs[start:end])
		if err != nil {
			return nil, err
		}
		outputs = append(outputs, batchOutputs...)
	}
	return outputs, nil
}

func validPredictionOutput(output predictiondata.Output) bool {
	return predictiondata.ValidateOutput(output) == nil
}

func nonNilContext(ctx context.Context) context.Context {
	if ctx == nil {
		return context.Background()
	}
	return ctx
}
