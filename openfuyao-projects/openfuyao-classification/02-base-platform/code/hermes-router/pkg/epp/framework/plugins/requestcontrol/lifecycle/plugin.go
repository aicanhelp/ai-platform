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

// Package lifecycle tracks request lifecycle metadata around scheduling.
package lifecycle

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"reflect"
	"time"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	requestcontrol "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	latencyattr "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/latency"

	npuattr "hermes-router/pkg/epp/framework/plugins/datalayer/attribute/npu"
	prerequest "hermes-router/pkg/epp/framework/plugins/requestcontrol/prerequest"
	internalinflight "hermes-router/pkg/epp/internal/inflight"
	"hermes-router/pkg/epp/internal/pdgroup"
	"hermes-router/pkg/epp/internal/score"
	"hermes-router/pkg/epp/internal/utils"
)

// PluginType is the registered plugin type identifier.
const PluginType = "request-lifecycle-tracker"

type clock func() time.Time

type pluginParameters struct {
	StoreName   string                `json:"storeName"`
	Persistence persistenceParameters `json:"persistence"`
}

type persistenceParameters struct {
	Enabled        bool `json:"enabled"`
	FlushThreshold int  `json:"flushThreshold"`
	FlushInterval  time.Duration
	OutputPath     string `json:"outputPath"`
}

type rawPluginParameters struct {
	StoreName   string                   `json:"storeName"`
	Persistence rawPersistenceParameters `json:"persistence"`
}

type rawPersistenceParameters struct {
	Enabled        bool   `json:"enabled"`
	FlushThreshold int    `json:"flushThreshold"`
	FlushInterval  string `json:"flushInterval"`
	OutputPath     string `json:"outputPath"`
}

// RequestLifecycleTracker keeps per-endpoint inflight counters in sync across
// the request and response hooks.
type RequestLifecycleTracker struct {
	typedName fwkplugin.TypedName
	store     *internalinflight.Store
	now       clock
}

var _ requestcontrol.PreRequest = (*RequestLifecycleTracker)(nil)
var _ requestcontrol.ResponseBody = (*RequestLifecycleTracker)(nil)
var _ fwkdl.EndpointExtractor = (*RequestLifecycleTracker)(nil)

// Factory constructs a lifecycle tracker instance.
func Factory(name string, raw json.RawMessage, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params, err := parseParameters(raw)
	if err != nil {
		return nil, fmt.Errorf("request lifecycle tracker %q: %w", name, err)
	}
	store := internalinflight.StoreForNameWithPersistence(params.StoreName, internalinflight.PersistenceConfig{
		Enabled:        params.Persistence.Enabled,
		FlushThreshold: params.Persistence.FlushThreshold,
		FlushInterval:  params.Persistence.FlushInterval,
		OutputPath:     params.Persistence.OutputPath,
	})
	store.ConfigureLogger(storeLogger(handle, name, params.StoreName))
	registerShutdownClose(handle, store)
	return NewRequestLifecycleTracker(name, store), nil
}

// NewRequestLifecycleTracker creates a tracker bound to the provided shared store.
func NewRequestLifecycleTracker(name string, store *internalinflight.Store) *RequestLifecycleTracker {
	return &RequestLifecycleTracker{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: name},
		store:     store,
		now:       time.Now,
	}
}

// TypedName returns the plugin identity.
func (t *RequestLifecycleTracker) TypedName() fwkplugin.TypedName {
	return t.typedName
}

// ExpectedInputType satisfies the base datalayer.Extractor interface.
func (t *RequestLifecycleTracker) ExpectedInputType() reflect.Type {
	return fwkdl.EndpointEventReflectType
}

// Extract is unused for endpoint lifecycle notifications.
func (t *RequestLifecycleTracker) Extract(_ context.Context, _ any, _ fwkdl.Endpoint) error {
	return nil
}

// PreRequest starts tracking the selected primary-profile endpoint.
func (t *RequestLifecycleTracker) PreRequest(
	ctx context.Context,
	request *fwkscheduling.LLMRequest,
	schedulingResult *fwkscheduling.SchedulingResult,
) {
	logger := log.FromContext(ctx).WithName("RequestLifecycleTracker.PreRequest")
	if t == nil || t.store == nil || request == nil || request.RequestId == "" {
		logger.V(logutil.DEBUG).Info("Skipping request lifecycle tracking due to missing request or request ID")
		return
	}
	route := prerequest.ExtractPrimaryRoute(ctx, schedulingResult)
	selected := selectedTarget(schedulingResult)
	if selected == nil || selected.GetMetadata() == nil {
		logger.V(logutil.DEBUG).Info("Skipping request lifecycle tracking because no selected endpoint is available")
		return
	}

	trackedEndpoints := trackedRouteEndpoints(route)
	predictionInputs, missingFeatureReasons := selectedPredictionInputs(selected, route)
	inputTokenLength := requestInputTokens(request)

	cacheScore, prediction, predictionMode := selectedRoutingSnapshot(selected, route)

	t.store.StartRequest(internalinflight.RequestStart{
		RequestID:             request.RequestId,
		Model:                 request.TargetModel,
		InputTokenLength:      inputTokenLength,
		SelectedEndpoint:      selected.GetMetadata(),
		TrackedEndpoints:      trackedEndpoints,
		DispatchInflight:      t.store.Snapshot(selected.GetMetadata()),
		PrefixCacheScore:      cacheScore,
		Prediction:            prediction,
		PredictionMode:        predictionMode,
		SelectedRoute:         selectedRouteSnapshot(t.store, selected, route),
		PredictionInputs:      predictionInputs,
		MissingFeatureReasons: missingFeatureReasons,
		StartedAt:             t.now(),
	})
}

// ResponseBody records the first body callback as TTFT and completes on EOS.
func (t *RequestLifecycleTracker) ResponseBody(
	ctx context.Context,
	request *fwkscheduling.LLMRequest,
	response *requestcontrol.Response,
	targetEndpoint *fwkdl.EndpointMetadata,
) {
	logger := log.FromContext(ctx).WithName("RequestLifecycleTracker.ResponseBody")
	if !canTrackResponse(t, request, response) {
		logger.V(logutil.DEBUG).Info("Skipping response lifecycle tracking due to missing request, request ID, or response")
		return
	}

	t.store.MarkFirstChunk(request.RequestId, t.now())
	if !response.EndOfStream {
		return
	}

	t.store.CompleteRequest(internalinflight.RequestCompletion{
		RequestID:        request.RequestId,
		CompletedAt:      t.now(),
		OutputLength:     outputLength(response),
		SelectedEndpoint: targetEndpoint,
	})
}

// ExtractEndpoint keeps endpoint lifecycle state aligned with the shared store.
func (t *RequestLifecycleTracker) ExtractEndpoint(_ context.Context, event fwkdl.EndpointEvent) error {
	if t == nil || t.store == nil {
		return nil
	}
	t.store.ApplyEndpointEvent(event, t.now())
	return nil
}

func parseParameters(raw json.RawMessage) (pluginParameters, error) {
	decoded := rawPluginParameters{StoreName: internalinflight.DefaultStoreName}
	if len(raw) == 0 {
		return pluginParameters{StoreName: internalinflight.DefaultStoreName}, nil
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&decoded); err != nil {
		return pluginParameters{}, fmt.Errorf("invalid parameters: %w", err)
	}
	params := pluginParameters{
		StoreName: decoded.StoreName,
		Persistence: persistenceParameters{
			Enabled:        decoded.Persistence.Enabled,
			FlushThreshold: decoded.Persistence.FlushThreshold,
			OutputPath:     decoded.Persistence.OutputPath,
		},
	}
	if params.StoreName == "" {
		params.StoreName = internalinflight.DefaultStoreName
	}
	if params.Persistence.FlushThreshold < 0 {
		return pluginParameters{}, fmt.Errorf("invalid parameters: persistence.flushThreshold must be >= 0")
	}
	if decoded.Persistence.FlushInterval != "" {
		flushInterval, err := time.ParseDuration(decoded.Persistence.FlushInterval)
		if err != nil {
			return pluginParameters{}, fmt.Errorf(
				"invalid parameters: persistence.flushInterval must be a valid duration: %w",
				err,
			)
		}
		if flushInterval < 0 {
			return pluginParameters{}, fmt.Errorf("invalid parameters: persistence.flushInterval must be >= 0")
		}
		params.Persistence.FlushInterval = flushInterval
	}
	if params.Persistence.Enabled && params.Persistence.OutputPath == "" {
		return pluginParameters{}, fmt.Errorf(
			"invalid parameters: persistence.outputPath is required when persistence is enabled",
		)
	}
	return params, nil
}

func requestInputTokens(request *fwkscheduling.LLMRequest) int {
	if request == nil {
		return 0
	}
	if request.TokenizedPrompt != nil && len(request.TokenizedPrompt.TokenIDs) > 0 {
		return len(request.TokenizedPrompt.TokenIDs)
	}
	length, err := utils.RequestLengthOf(request)
	if err != nil {
		return 0
	}
	return length
}

func selectedTarget(result *fwkscheduling.SchedulingResult) fwkscheduling.Endpoint {
	targets := primaryProfileTargets(result)
	if len(targets) == 0 {
		return nil
	}
	leader := targets[0]
	if leader == nil || leader.GetMetadata() == nil {
		return nil
	}
	return leader
}

func primaryProfileTargets(result *fwkscheduling.SchedulingResult) []fwkscheduling.Endpoint {
	if result == nil || result.PrimaryProfileName == "" || result.ProfileResults == nil {
		return nil
	}
	runResult := result.ProfileResults[result.PrimaryProfileName]
	if runResult == nil || len(runResult.TargetEndpoints) == 0 {
		return nil
	}
	return runResult.TargetEndpoints
}

// readLatencyAttribute fetches a LatencyPredictionInfo payload from the given
// attribute key. A missing key returns the zero value, matching the upstream
// attribute convention.
func readLatencyAttribute(endpoint fwkscheduling.Endpoint, key string) internalinflight.LatencyPrediction {
	if endpoint == nil {
		return internalinflight.LatencyPrediction{}
	}
	value, ok := endpoint.Get(key)
	if !ok {
		return internalinflight.LatencyPrediction{}
	}
	prediction, ok := value.(*latencyattr.LatencyPredictionInfo)
	if !ok || prediction == nil {
		return internalinflight.LatencyPrediction{}
	}
	return internalinflight.LatencyPrediction{
		TTFTMillis: prediction.TTFT(),
		TPOTMillis: prediction.TPOT(),
	}
}

func latencyPrediction(endpoint fwkscheduling.Endpoint) internalinflight.LatencyPrediction {
	return readLatencyAttribute(endpoint, latencyattr.LatencyPredictionInfoKey)
}

// pdPrediction combines a prefill TTFT and a decode TPOT into a single
// LatencyPrediction for disaggregated routes. Missing sides contribute zero,
// so the result may be partial (only TTFT or only TPOT populated) when one
// side has data and the other does not.
func pdPrediction(prefill, decode fwkscheduling.Endpoint, key string) internalinflight.LatencyPrediction {
	return internalinflight.LatencyPrediction{
		TTFTMillis: readLatencyAttribute(prefill, key).TTFTMillis,
		TPOTMillis: readLatencyAttribute(decode, key).TPOTMillis,
	}
}

// selectedRoutingSnapshot extracts cache score and active prediction in a
// single D/PD-or-aggregate branch so PreRequest visits the disaggregation
// decision exactly once per request.
func selectedRoutingSnapshot(selected fwkscheduling.Endpoint, route *prerequest.Route) (
	float64,
	internalinflight.LatencyPrediction,
	string,
) {
	if prefillEP, decodeEP, ok := selectedMemberEndpoints(selected, route); ok {
		if isDecodeOnlyRoute(route) {
			active := latencyPrediction(decodeEP)
			return score.PrefixCacheScoreFromEndpoint(decodeEP), active, derivePredictionMode(active)
		}
		active := pdPrediction(prefillEP, decodeEP, latencyattr.LatencyPredictionInfoKey)
		return score.PrefixCacheScoreFromEndpoint(prefillEP), active, derivePredictionMode(active)
	}

	active := latencyPrediction(selected)
	return score.PrefixCacheScoreFromEndpoint(selected), active, derivePredictionMode(active)
}

// derivePredictionMode records whether the scorer had an active prediction.
// A real predictor's output is effectively never exactly zero, so a zero-value
// check is a sufficient presence signal in practice.
func derivePredictionMode(active internalinflight.LatencyPrediction) string {
	zero := internalinflight.LatencyPrediction{}
	if active != zero {
		return internalinflight.PredictionModeActive
	}
	return ""
}

func outputLength(response *requestcontrol.Response) int {
	if response == nil || response.Usage.CompletionTokens < 0 {
		return 0
	}
	return response.Usage.CompletionTokens
}

func trackedRouteEndpoints(route *prerequest.Route) []*fwkdl.EndpointMetadata {
	if route == nil || route.Leader == nil {
		return nil
	}
	if isDecodeOnlyRoute(route) {
		return []*fwkdl.EndpointMetadata{route.Decode}
	}
	if isPDRoute(route) {
		return []*fwkdl.EndpointMetadata{route.Prefill, route.Decode}
	}

	return nil
}

func selectedRouteSnapshot(
	store *internalinflight.Store,
	selected fwkscheduling.Endpoint,
	route *prerequest.Route,
) *internalinflight.SelectedRoute {
	if selected == nil {
		return nil
	}
	if prefillEP, decodeEP, ok := selectedMemberEndpoints(selected, route); ok {
		if isDecodeOnlyRoute(route) {
			return &internalinflight.SelectedRoute{
				Leader: endpointIdentitySnapshot(selected),
				Decode: endpointSnapshot(store, decodeEP),
			}
		}
		return &internalinflight.SelectedRoute{
			Leader:  endpointIdentitySnapshot(selected),
			Prefill: endpointSnapshot(store, prefillEP),
			Decode:  endpointSnapshot(store, decodeEP),
		}
	}
	return &internalinflight.SelectedRoute{Leader: endpointSnapshot(store, selected)}
}

func canTrackResponse(
	tracker *RequestLifecycleTracker,
	request *fwkscheduling.LLMRequest,
	response *requestcontrol.Response,
) bool {
	if tracker == nil || tracker.store == nil || request == nil {
		return false
	}
	return request.RequestId != "" && response != nil
}

func selectedMemberEndpoints(
	selected fwkscheduling.Endpoint,
	route *prerequest.Route,
) (fwkscheduling.Endpoint, fwkscheduling.Endpoint, bool) {
	if !isDecodeOnlyRoute(route) && !isPDRoute(route) {
		return nil, nil, false
	}
	info, ok := pdgroup.GetDirect(selected)
	if !ok || info == nil {
		return nil, nil, false
	}
	if isDecodeOnlyRoute(route) {
		if info.SelectedPrefillPod != nil || info.SelectedDecodePod == nil {
			return nil, nil, false
		}
		decodeEP := info.SelectedDecodePod.Endpoint
		return nil, decodeEP, decodeEP != nil
	}
	if info.SelectedPrefillPod == nil || info.SelectedDecodePod == nil {
		return nil, nil, false
	}
	prefillEP := info.SelectedPrefillPod.Endpoint
	decodeEP := info.SelectedDecodePod.Endpoint
	return prefillEP, decodeEP, prefillEP != nil && decodeEP != nil
}

func isDecodeOnlyRoute(route *prerequest.Route) bool {
	return route != nil && route.Leader != nil && route.Prefill == nil && route.Decode != nil
}

func isPDRoute(route *prerequest.Route) bool {
	return route != nil && route.Leader != nil && route.Prefill != nil && route.Decode != nil
}

func endpointIdentitySnapshot(endpoint fwkscheduling.Endpoint) *internalinflight.RouteEndpointSnapshot {
	if endpoint == nil {
		return nil
	}
	return &internalinflight.RouteEndpointSnapshot{ID: endpointID(endpoint.GetMetadata())}
}

func endpointSnapshot(
	store *internalinflight.Store,
	endpoint fwkscheduling.Endpoint,
) *internalinflight.RouteEndpointSnapshot {
	if endpoint == nil {
		return nil
	}
	metadata := endpoint.GetMetadata()
	load := store.Snapshot(metadata)
	return &internalinflight.RouteEndpointSnapshot{
		ID:       endpointID(metadata),
		Metrics:  endpointMetricsSnapshot(endpoint.GetMetrics()),
		Inflight: &internalinflight.Load{Tokens: load.Tokens, Requests: load.Requests},
		NPU:      endpointNPUSnapshot(endpoint),
	}
}

func endpointID(metadata *fwkdl.EndpointMetadata) string {
	if metadata == nil {
		return ""
	}
	if metadata.PodName != "" {
		return fmt.Sprintf("pod://%s/%s", metadata.NamespacedName.Namespace, metadata.PodName)
	}
	if metadata.Address == "" || metadata.Port == "" {
		return ""
	}
	return fmt.Sprintf("addr://%s:%s", metadata.Address, metadata.Port)
}

func endpointMetricsSnapshot(metrics *fwkdl.Metrics) *internalinflight.EndpointMetricsSnapshot {
	if metrics == nil {
		return nil
	}
	return &internalinflight.EndpointMetricsSnapshot{
		ActiveModels:            maps.Clone(metrics.ActiveModels),
		WaitingModels:           maps.Clone(metrics.WaitingModels),
		MaxActiveModels:         metrics.MaxActiveModels,
		RunningRequestsSize:     metrics.RunningRequestsSize,
		WaitingQueueSize:        metrics.WaitingQueueSize,
		KVCacheUsagePercent:     metrics.KVCacheUsagePercent,
		KVCacheMaxTokenCapacity: metrics.KvCacheMaxTokenCapacity,
		CacheBlockSize:          metrics.CacheBlockSize,
		CacheNumBlocks:          metrics.CacheNumBlocks,
		UpdateTime:              metrics.UpdateTime,
	}
}

func endpointNPUSnapshot(endpoint fwkscheduling.Endpoint) *internalinflight.NPUStateSnapshot {
	if endpoint == nil {
		return nil
	}
	value, ok := endpoint.Get(npuattr.DefaultStateAttributeKey)
	if !ok {
		return nil
	}
	state, ok := value.(*npuattr.State)
	if !ok || state == nil {
		return nil
	}
	devices := make([]internalinflight.NPUDeviceSnapshot, 0, len(state.Devices))
	for _, device := range state.Devices {
		devices = append(devices, internalinflight.NPUDeviceSnapshot{
			Target:                  device.Target,
			DeviceID:                device.DeviceID,
			AICoreUtilization:       device.AICoreUtilization,
			HBMUtilization:          device.HBMUtilization,
			HBMBandwidthUtilization: device.HBMBandwidthUtilization,
		})
	}
	return &internalinflight.NPUStateSnapshot{
		ObservationTime:            state.ObservationTime,
		DataStatus:                 string(state.DataStatus),
		DeviceCount:                state.DeviceCount,
		AICoreUtilizationAvg:       state.AICoreUtilizationAvg,
		AICoreUtilizationMax:       state.AICoreUtilizationMax,
		HBMUtilizationAvg:          state.HBMUtilizationAvg,
		HBMUtilizationMax:          state.HBMUtilizationMax,
		HBMBandwidthUtilizationAvg: state.HBMBandwidthUtilizationAvg,
		HBMBandwidthUtilizationMax: state.HBMBandwidthUtilizationMax,
		Devices:                    devices,
	}
}

func registerShutdownClose(handle fwkplugin.Handle, store *internalinflight.Store) {
	if handle == nil || store == nil {
		return
	}
	ctx := handle.Context()
	if ctx == nil {
		return
	}
	go func() {
		<-ctx.Done()
		_ = store.Close()
	}()
}

func storeLogger(handle fwkplugin.Handle, pluginName string, storeName string) logr.Logger {
	if handle == nil {
		return logr.Discard()
	}
	ctx := handle.Context()
	if ctx == nil {
		return logr.Discard()
	}
	return log.FromContext(ctx).WithName("RequestLifecycleTracker.Store").WithValues(
		"plugin", pluginName,
		"store", storeName,
	)
}
