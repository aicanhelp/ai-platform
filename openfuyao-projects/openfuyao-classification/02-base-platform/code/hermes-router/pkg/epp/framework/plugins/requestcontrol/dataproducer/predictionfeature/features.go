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

package predictionfeature

import (
	"context"
	"strings"

	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	npuattr "hermes-router/pkg/epp/framework/plugins/datalayer/attribute/npu"
	"hermes-router/pkg/epp/internal/endpointmeta"
	internalinflight "hermes-router/pkg/epp/internal/inflight"
	"hermes-router/pkg/epp/internal/predictiondata"
	"hermes-router/pkg/epp/internal/score"
)

// Missing-feature reason strings emitted when an endpoint is skipped. Keeping
// them as named constants gives downstream metrics and log analyzers a stable
// vocabulary instead of free-form sentences. reasonStaleNPUState is distinct
// from reasonMissingNPUState because the two failure modes carry different
// signals offline: "missing" usually means a fresh pod the extractor has not
// scraped yet, while "stale" indicates exporter or device-side trouble that
// downstream dataset auditing should treat separately.
const (
	reasonMissingAddress     = "missing endpoint address"
	reasonMissingMetrics     = "missing endpoint metrics"
	reasonMissingInflight    = "missing inflight load"
	reasonMissingNPUState    = "missing npu state"
	reasonStaleNPUState      = "stale npu state"
	reasonMissingInputTokens = "missing input token length"

	percentScale = 100.0
)

// buildInput assembles a normalized prediction feature row for a single
// endpoint. The bool return is false when any required signal is absent; the
// second string carries the precise reason so callers can publish a
// MissingFeatureReason attribute for downstream tracing.
func (e *FeatureExtractor) buildInput(
	ctx context.Context,
	inputTokenLength int,
	endpoint fwkscheduling.Endpoint,
) (predictiondata.Input, bool, string) {
	address := endpointmeta.EndpointAddress(endpoint)
	if endpoint == nil || address == "" {
		return predictiondata.Input{}, false, reasonMissingAddress
	}
	metrics := endpoint.GetMetrics()
	if metrics == nil {
		return predictiondata.Input{}, false, reasonMissingMetrics
	}
	load, ok := readInflightLoad(endpoint)
	if !ok {
		return predictiondata.Input{}, false, reasonMissingInflight
	}
	npuState, ok := readNPUState(endpoint)
	if !ok {
		return predictiondata.Input{}, false, reasonMissingNPUState
	}
	// Stale snapshots carry the most recent utilization sample but its
	// observation time has aged past the extractor's staleAfter window.
	// Feeding such values into the predictor risks routing traffic on signals
	// that no longer reflect reality (e.g. a hung device that last reported
	// low utilization). Treat Stale as feature-unavailable so the producer
	// skips this endpoint and the scorer's all-or-nothing fallback engages
	// instead of trusting expired numbers.
	if npuState.DataStatus == npuattr.DataStale {
		return predictiondata.Input{}, false, reasonStaleNPUState
	}
	return predictiondata.Input{
		PodType:              e.endpointPodType(ctx, endpoint),
		InputTokenLength:     inputTokenLength,
		PrefixCacheScore:     score.PrefixCacheScoreFromEndpoint(endpoint),
		NumRequestWaiting:    metrics.WaitingQueueSize,
		NumRequestRunning:    metrics.RunningRequestsSize,
		KVCacheUsage:         score.Clamp01(metrics.KVCacheUsagePercent),
		InflightTokens:       load.Tokens,
		HBMUtilization:       npuState.HBMUtilizationMax / percentScale,
		AICoreUtilization:    npuState.AICoreUtilizationMax / percentScale,
		BandwidthUtilization: npuState.HBMBandwidthUtilizationMax / percentScale,
		ServerIP:             address,
	}, true, ""
}

// endpointPodType resolves the canonical pod role used as a categorical
// feature. Selector-based labels take precedence so configurations migrating
// off the legacy pdLabelName scheme keep their classification stable.
func (e *FeatureExtractor) endpointPodType(ctx context.Context, endpoint fwkscheduling.Endpoint) string {
	if strings.TrimSpace(e.targetPodRoleLabel) != "" {
		labelValue, ok := endpointmeta.EndpointLabelValueAllowed(endpoint, e.targetPodRoleLabel, e.targetPodRoles)
		if ok {
			return e.canonicalRoleValue(ctx, labelValue, e.targetPodRoleLabel)
		}
	}
	if strings.TrimSpace(e.pdLabel) == "" {
		return predictiondata.PodTypeAggregate
	}
	labelValue, ok := endpointmeta.EndpointLabelValue(endpoint, e.pdLabel)
	if !ok {
		return predictiondata.PodTypeAggregate
	}
	return e.canonicalRoleValue(ctx, labelValue, e.pdLabel)
}

func (e *FeatureExtractor) canonicalRoleValue(ctx context.Context, labelValue, labelName string) string {
	switch labelValue {
	case e.prefill:
		return predictiondata.PodTypePrefill
	case e.decode:
		return predictiondata.PodTypeDecode
	default:
		// Defaulting silently would hide misconfigured label values (for
		// example a typoed prefillValue) behind a successful aggregate
		// classification. Debug logging keeps the signal available during
		// troubleshooting without spamming healthy paths.
		log.FromContext(ctx).V(logutil.DEBUG).Info(
			"Endpoint pod label value did not match configured prefill or decode; defaulting to aggregate",
			"label", labelName,
			"labelValue", labelValue,
			"prefillValue", e.prefill,
			"decodeValue", e.decode,
		)
		return predictiondata.PodTypeAggregate
	}
}

func requestInputTokenLength(request *fwkscheduling.LLMRequest) (int, bool) {
	if request == nil || request.TokenizedPrompt == nil || len(request.TokenizedPrompt.TokenIDs) == 0 {
		return 0, false
	}
	return len(request.TokenizedPrompt.TokenIDs), true
}

func readInflightLoad(endpoint fwkscheduling.Endpoint) (*internalinflight.Load, bool) {
	value, ok := endpoint.Get(internalinflight.DefaultAttributeKey)
	if !ok {
		return nil, false
	}
	load, ok := value.(*internalinflight.Load)
	if !ok || load == nil {
		return nil, false
	}
	// Zero load is still valid data; only a missing attribute counts as
	// missing inflight feature signal.
	return load, true
}

func readNPUState(endpoint fwkscheduling.Endpoint) (*npuattr.State, bool) {
	value, ok := endpoint.Get(npuattr.DefaultStateAttributeKey)
	if !ok {
		return nil, false
	}
	state, ok := value.(*npuattr.State)
	if !ok || state == nil {
		return nil, false
	}
	return state, true
}
