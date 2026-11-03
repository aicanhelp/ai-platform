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

// Package predictionfeature assembles per-endpoint prediction feature vectors
// and publishes them as endpoint attributes. The extractor is always
// deterministic and free of network calls so it can run regardless of whether
// a downstream predictor is configured — feature collection for offline
// training works even when no model is live.
package predictionfeature

import (
	"context"
	"encoding/json"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	requestcontrol "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	prefixattr "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"

	npuattr "hermes-router/pkg/epp/framework/plugins/datalayer/attribute/npu"
	predictioninputattr "hermes-router/pkg/epp/framework/plugins/datalayer/attribute/predictioninput"
	tokenizedpromptattr "hermes-router/pkg/epp/framework/plugins/datalayer/attribute/tokenizedprompt"
	"hermes-router/pkg/epp/internal/endpointmeta"
	internalinflight "hermes-router/pkg/epp/internal/inflight"
)

// PluginType is the registered plugin type identifier.
const PluginType = "prediction-feature-extractor"

// FeatureExtractor publishes a PredictionInputKey snapshot for every
// candidate endpoint that exposes a complete feature vector and a
// MissingFeatureReasonKey otherwise. It declares Consumes against the
// upstream attributes so the framework's DAG ordering guarantees this
// extractor observes a fully-populated endpoint before serializing features.
type FeatureExtractor struct {
	typedName          fwkplugin.TypedName
	pdLabel            string
	prefill            string
	decode             string
	selector           endpointmeta.LabelSelector
	hasSelector        bool
	targetPodRoleLabel string
	targetPodRoles     map[string]struct{}
}

var _ requestcontrol.PrepareDataPlugin = (*FeatureExtractor)(nil)

// Factory parses raw plugin parameters and constructs an extractor.
func Factory(name string, raw json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	config, err := parseAndValidateParameters(raw)
	if err != nil {
		return nil, fmt.Errorf("prediction feature extractor %q: %w", name, err)
	}
	return New(name, config)
}

// New builds an extractor from a resolved Config. It is exposed so other
// packages and tests can wire an extractor without round-tripping through
// JSON.
func New(name string, config Config) (*FeatureExtractor, error) {
	resolved := resolveConfig(config)
	selector, hasSelector, err := endpointmeta.CompileProducerTargetSelector(
		resolved.TargetPodSelector, resolved.TargetPodRoleLabel, resolved.TargetPodRoles,
	)
	if err != nil {
		return nil, err
	}
	return &FeatureExtractor{
		typedName:          fwkplugin.TypedName{Type: PluginType, Name: name},
		pdLabel:            resolved.PDLabelName,
		prefill:            resolved.PrefillValue,
		decode:             resolved.DecodeValue,
		selector:           selector,
		hasSelector:        hasSelector,
		targetPodRoleLabel: resolved.TargetPodRoleLabel,
		targetPodRoles:     endpointmeta.LabelValueSet(resolved.TargetPodRoles),
	}, nil
}

// TypedName returns the framework plugin type and instance name.
func (e *FeatureExtractor) TypedName() fwkplugin.TypedName { return e.typedName }

// Produces declares the request-control data this plugin writes.
func (e *FeatureExtractor) Produces() map[string]any {
	return map[string]any{
		predictioninputattr.PredictionInputKey:      predictioninputattr.InputSnapshot{},
		predictioninputattr.MissingFeatureReasonKey: predictioninputattr.MissingFeatureReason{},
	}
}

// Consumes declares the request-control data this plugin reads.
func (e *FeatureExtractor) Consumes() map[string]any {
	return map[string]any{
		tokenizedpromptattr.TokenizedPromptKey: tokenizedpromptattr.Info{},
		prefixattr.PrefixCacheMatchInfoKey:     prefixattr.PrefixCacheMatchInfo{},
		internalinflight.DefaultAttributeKey:   internalinflight.Load{},
		npuattr.DefaultStateAttributeKey:       npuattr.State{},
	}
}

// PrepareRequestData walks every candidate endpoint and publishes either a
// PredictionInputKey snapshot or a MissingFeatureReasonKey explanation.
// Returning a nil error is intentional even when no endpoint can be
// featurized: the downstream tracker and predictor must remain free to run
// without prediction inputs.
func (e *FeatureExtractor) PrepareRequestData(
	ctx context.Context,
	request *fwkscheduling.LLMRequest,
	endpoints []fwkscheduling.Endpoint,
) error {
	if e == nil {
		return nil
	}
	logger := log.FromContext(ctx).WithName("PredictionFeatureExtractor.PrepareRequestData")
	inputTokenLength, ok := requestInputTokenLength(request)
	if !ok {
		logger.V(logutil.DEBUG).Info(
			"Skipping prediction feature extraction because request has no tokenized prompt",
		)
		return nil
	}
	for _, endpoint := range endpoints {
		if endpoint == nil {
			continue
		}
		if e.hasSelector && !e.selector.Matches(endpoint) {
			continue
		}
		input, ok, reason := e.buildInput(ctx, inputTokenLength, endpoint)
		if !ok {
			endpoint.Put(predictioninputattr.MissingFeatureReasonKey, &predictioninputattr.MissingFeatureReason{Reason: reason})
			metadata := endpoint.GetMetadata()
			logger.V(logutil.DEBUG).Info(
				"Skipping endpoint because prediction feature is missing",
				"reason", reason,
				"namespace", endpointNamespace(metadata),
				"name", endpointName(metadata),
				"pod", endpointPodName(metadata),
				"serverIP", endpointmeta.EndpointAddress(endpoint),
			)
			continue
		}
		endpoint.Put(predictioninputattr.PredictionInputKey, &predictioninputattr.InputSnapshot{Input: input})
	}
	return nil
}

func endpointNamespace(metadata *fwkdl.EndpointMetadata) string {
	if metadata == nil {
		return ""
	}
	return metadata.NamespacedName.Namespace
}

func endpointName(metadata *fwkdl.EndpointMetadata) string {
	if metadata == nil {
		return ""
	}
	return metadata.NamespacedName.Name
}

func endpointPodName(metadata *fwkdl.EndpointMetadata) string {
	if metadata == nil {
		return ""
	}
	return metadata.PodName
}
