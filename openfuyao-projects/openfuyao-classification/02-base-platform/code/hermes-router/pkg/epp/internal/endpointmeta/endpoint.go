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

// Package endpointmeta provides Hermes Router support.
package endpointmeta

import (
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

// MetadataOf returns the endpoint metadata when present.
func MetadataOf(endpoint fwkscheduling.Endpoint) *fwkdl.EndpointMetadata {
	if endpoint == nil {
		return nil
	}
	return endpoint.GetMetadata()
}

// EndpointPodName returns the endpoint pod name when present.
func EndpointPodName(endpoint fwkscheduling.Endpoint) string {
	if metadata := MetadataOf(endpoint); metadata != nil {
		return metadata.PodName
	}
	return ""
}

// EndpointAddress returns the endpoint address when present.
func EndpointAddress(endpoint fwkscheduling.Endpoint) string {
	if metadata := MetadataOf(endpoint); metadata != nil {
		return metadata.Address
	}
	return ""
}

// EndpointPort returns the endpoint port when present.
func EndpointPort(endpoint fwkscheduling.Endpoint) string {
	if metadata := MetadataOf(endpoint); metadata != nil {
		return metadata.Port
	}
	return ""
}

// EndpointLabelValue looks up a metadata label on an endpoint.
func EndpointLabelValue(endpoint fwkscheduling.Endpoint, key string) (string, bool) {
	metadata := MetadataOf(endpoint)
	if metadata == nil || metadata.Labels == nil {
		return "", false
	}
	value, ok := metadata.Labels[key]
	return value, ok
}

// CloneScoredEndpoint creates a deep copy of a scored endpoint.
//
// Contract:
//   - if the scored endpoint has a nil Endpoint, the clone keeps a nil Endpoint
//   - if the Endpoint exists, the clone always returns a writable Endpoint copy
//   - nil metadata or metrics on the source Endpoint are normalized into empty
//     values on the clone so downstream code can safely mutate the copy without
//     mutating the original
func CloneScoredEndpoint(scored fwkscheduling.ScoredEndpoint) fwkscheduling.ScoredEndpoint {
	if scored.Endpoint == nil {
		return fwkscheduling.ScoredEndpoint{Score: scored.Score}
	}

	metadata := scored.Endpoint.GetMetadata()
	metrics := scored.Endpoint.GetMetrics()
	attributes := scored.Endpoint.Clone()
	if metadata == nil {
		metadata = &fwkdl.EndpointMetadata{}
	} else {
		metadata = metadata.Clone()
	}
	if metrics == nil {
		metrics = fwkdl.NewMetrics()
	} else {
		metrics = metrics.Clone()
	}

	return fwkscheduling.ScoredEndpoint{
		Endpoint: fwkscheduling.NewEndpoint(metadata, metrics, attributes),
		Score:    scored.Score,
	}
}
