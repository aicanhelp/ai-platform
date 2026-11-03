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

// Package inflight provides in-flight request tracking plugins and state.
package inflight

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"

	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	requestcontrol "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	"hermes-router/pkg/epp/internal/endpointmeta"
	internalinflight "hermes-router/pkg/epp/internal/inflight"
)

// PluginType is the registered plugin type identifier.
const PluginType = "inflight-data-producer"

type pluginParameters struct {
	StoreName          string                            `json:"storeName"`
	AttributeKey       string                            `json:"attributeKey"`
	TargetPodSelector  *endpointmeta.LabelSelectorConfig `json:"targetPodSelector,omitempty"`
	TargetPodRoleLabel string                            `json:"targetPodRoleLabel,omitempty"`
	TargetPodRoles     []string                          `json:"targetPodRoles,omitempty"`
}

// LoadProducer publishes the shared inflight snapshot onto each
// candidate endpoint attribute map before scheduling runs.
type LoadProducer struct {
	typedName    fwkplugin.TypedName
	store        *internalinflight.Store
	attributeKey string
	selector     endpointmeta.LabelSelector
	hasSelector  bool
}

var _ requestcontrol.PrepareDataPlugin = (*LoadProducer)(nil)

// Factory constructs an inflight producer bound to a named shared store.
func Factory(name string, raw json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params, err := parseParameters(raw)
	if err != nil {
		return nil, fmt.Errorf("inflight producer %q: %w", name, err)
	}
	selector, hasSelector, err := endpointmeta.CompileProducerTargetSelector(
		params.TargetPodSelector,
		params.TargetPodRoleLabel,
		params.TargetPodRoles,
	)
	if err != nil {
		return nil, fmt.Errorf("inflight producer %q: %w", name, err)
	}
	return newLoadProducer(
		name,
		internalinflight.StoreForName(params.StoreName),
		params.AttributeKey,
		selector,
		hasSelector,
	), nil
}

// NewLoadProducer creates a producer bound to the provided store.
func NewLoadProducer(name string, store *internalinflight.Store, attributeKey string) *LoadProducer {
	return newLoadProducer(name, store, attributeKey, endpointmeta.LabelSelector{}, false)
}

func newLoadProducer(
	name string,
	store *internalinflight.Store,
	attributeKey string,
	selector endpointmeta.LabelSelector,
	hasSelector bool,
) *LoadProducer {
	resolvedAttributeKey := attributeKey
	if resolvedAttributeKey == "" {
		resolvedAttributeKey = internalinflight.DefaultAttributeKey
	}
	return &LoadProducer{
		typedName:    fwkplugin.TypedName{Type: PluginType, Name: name},
		store:        store,
		attributeKey: resolvedAttributeKey,
		selector:     selector,
		hasSelector:  hasSelector,
	}
}

// TypedName returns the plugin identity.
func (p *LoadProducer) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// Produces declares the endpoint attribute key published by the producer.
func (p *LoadProducer) Produces() map[string]any {
	return map[string]any{
		p.attributeKey: internalinflight.Load{},
	}
}

// Consumes declares no upstream request-control keys.
func (p *LoadProducer) Consumes() map[string]any {
	return map[string]any{}
}

// PrepareRequestData snapshots current inflight counters onto each candidate endpoint.
func (p *LoadProducer) PrepareRequestData(
	_ context.Context,
	_ *fwkscheduling.LLMRequest,
	pods []fwkscheduling.Endpoint,
) error {
	if p == nil || p.store == nil {
		return nil
	}
	for _, pod := range pods {
		if pod == nil {
			continue
		}
		if p.hasSelector && !p.selector.Matches(pod) {
			continue
		}
		load := p.store.Snapshot(pod.GetMetadata())
		pod.Put(p.attributeKey, (&load).Clone())
	}
	return nil
}

func parseParameters(raw json.RawMessage) (pluginParameters, error) {
	params := pluginParameters{
		StoreName:    internalinflight.DefaultStoreName,
		AttributeKey: internalinflight.DefaultAttributeKey,
	}
	if len(raw) == 0 {
		return params, nil
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&params); err != nil {
		return pluginParameters{}, fmt.Errorf("invalid parameters: %w", err)
	}
	if params.StoreName == "" {
		params.StoreName = internalinflight.DefaultStoreName
	}
	if params.AttributeKey == "" {
		params.AttributeKey = internalinflight.DefaultAttributeKey
	}
	params.TargetPodRoleLabel = strings.TrimSpace(params.TargetPodRoleLabel)
	params.TargetPodRoles = endpointmeta.NormalizeLabelValues(params.TargetPodRoles)
	if _, _, err := endpointmeta.CompileProducerTargetSelector(
		params.TargetPodSelector,
		params.TargetPodRoleLabel,
		params.TargetPodRoles,
	); err != nil {
		return pluginParameters{}, err
	}
	return params, nil
}
