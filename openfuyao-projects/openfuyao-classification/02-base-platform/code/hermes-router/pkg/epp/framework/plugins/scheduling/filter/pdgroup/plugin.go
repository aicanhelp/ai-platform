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

// Package pdgroup provides PD group request data and scheduling filters.
package pdgroup

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"

	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	pdg "hermes-router/pkg/epp/internal/pdgroup"
)

// PluginType is the registered plugin type identifier.
const PluginType = "pd-group-filter"

// Filter describes Filter data used by Hermes Router.
type Filter struct {
	typedName fwkplugin.TypedName
}

var _ fwkscheduling.Filter = (*Filter)(nil)

// Factory constructs a plugin instance from raw parameters.
func Factory(name string, raw json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	if err := parseParameters(raw); err != nil {
		return nil, fmt.Errorf("pd group filter %q: %w", name, err)
	}
	return New(name), nil
}

// New constructs a plugin instance with validated configuration.
func New(name string) *Filter {
	return &Filter{typedName: fwkplugin.TypedName{Type: PluginType, Name: name}}
}

// TypedName returns the framework plugin type and instance name.
func (f *Filter) TypedName() fwkplugin.TypedName {
	return f.typedName
}

// Filter removes endpoints that do not satisfy the plugin policy.
func (f *Filter) Filter(
	ctx context.Context,
	state *fwkscheduling.CycleState,
	request *fwkscheduling.LLMRequest,
	pods []fwkscheduling.Endpoint,
) []fwkscheduling.Endpoint {
	_ = ctx
	_ = state
	_ = request
	validLeaders := make([]fwkscheduling.Endpoint, 0)
	groupInfos := make([]*pdg.Info, 0)
	for _, endpoint := range pods {
		info, ok := pdg.GetDirect(endpoint)
		if ok {
			groupInfos = append(groupInfos, info)
			if pdg.Validate(info) == nil {
				validLeaders = append(validLeaders, endpoint)
			}
		}
	}
	if len(validLeaders) > 0 {
		return validLeaders
	}

	filtered := make([]fwkscheduling.Endpoint, 0, len(pods))
	for _, endpoint := range pods {
		if _, ok := pdg.GetDirect(endpoint); ok {
			continue
		}
		if isPDMember(endpoint, groupInfos) {
			continue
		}
		filtered = append(filtered, endpoint)
	}
	return filtered
}

func isPDMember(endpoint fwkscheduling.Endpoint, groups []*pdg.Info) bool {
	for _, info := range groups {
		if pdg.ContainsMember(info, endpoint) {
			return true
		}
	}
	return false
}

func parseParameters(raw json.RawMessage) error {
	if len(raw) == 0 {
		return nil
	}
	var params struct{}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&params); err != nil {
		return fmt.Errorf("invalid parameters: %w", err)
	}
	return nil
}
