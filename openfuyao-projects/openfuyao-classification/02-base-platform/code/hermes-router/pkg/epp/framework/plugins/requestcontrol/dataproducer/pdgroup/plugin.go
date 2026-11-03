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
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"strings"

	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	requestcontrol "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	"hermes-router/pkg/epp/internal/endpointmeta"
	pdg "hermes-router/pkg/epp/internal/pdgroup"
	"hermes-router/pkg/epp/internal/utils"
)

// PluginType is the registered plugin type identifier.
const PluginType = "pd-group-producer"

type pluginParameters struct {
	RoleLabelName  string `json:"pdLabelName"`
	GroupLabelName string `json:"pdGroupLabelName"`
	PrefillValue   string `json:"prefillValue"`
	DecodeValue    string `json:"decodeValue"`
	LeaderValue    string `json:"leaderValue"`
}

// Config describes Config data used by Hermes Router.
type Config struct {
	// RoleLabelName stores the RoleLabelName value.
	RoleLabelName string
	// GroupLabelName stores the GroupLabelName value.
	GroupLabelName string
	// PrefillValue stores the PrefillValue value.
	PrefillValue string
	// DecodeValue stores the DecodeValue value.
	DecodeValue string
	// LeaderValue stores the LeaderValue value.
	LeaderValue string
}

// Producer describes Producer data used by Hermes Router.
type Producer struct {
	typedName      fwkplugin.TypedName
	roleLabelName  string
	groupLabelName string
	prefillValue   string
	decodeValue    string
	leaderValue    string
}

type groupMembers struct {
	leaders  []fwkscheduling.Endpoint
	prefills []fwkscheduling.ScoredEndpoint
	decodes  []fwkscheduling.ScoredEndpoint
}

var _ requestcontrol.PrepareDataPlugin = (*Producer)(nil)

// Factory constructs a plugin instance from raw parameters.
func Factory(name string, raw json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	config, err := parseAndValidateParameters(raw)
	if err != nil {
		return nil, fmt.Errorf("pd group producer %q: %w", name, err)
	}
	return New(name, config), nil
}

// New constructs a plugin instance with validated configuration.
func New(name string, config Config) *Producer {
	return &Producer{
		typedName:      fwkplugin.TypedName{Type: PluginType, Name: name},
		roleLabelName:  config.RoleLabelName,
		groupLabelName: config.GroupLabelName,
		prefillValue:   config.PrefillValue,
		decodeValue:    config.DecodeValue,
		leaderValue:    config.LeaderValue,
	}
}

// TypedName returns the framework plugin type and instance name.
func (p *Producer) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// Produces declares the request-control data this plugin writes.
func (p *Producer) Produces() map[string]any {
	return map[string]any{
		pdg.AttributeKey: &pdg.Info{},
	}
}

// Consumes declares the request-control data this plugin reads.
func (p *Producer) Consumes() map[string]any {
	return map[string]any{}
}

// PrepareRequestData derives request data before scheduling.
func (p *Producer) PrepareRequestData(
	ctx context.Context,
	request *fwkscheduling.LLMRequest,
	endpoints []fwkscheduling.Endpoint,
) error {
	_ = request
	logger := log.FromContext(ctx).WithName("Producer.PrepareRequestData")
	groups := make(map[string]*groupMembers)

	for _, endpoint := range endpoints {
		roleValue, roleOK := endpointmeta.EndpointLabelValue(endpoint, p.roleLabelName)
		groupID, groupOK := endpointmeta.EndpointLabelValue(endpoint, p.groupLabelName)
		if !roleOK || !groupOK || strings.TrimSpace(groupID) == "" {
			continue
		}

		group := groups[groupID]
		if group == nil {
			group = &groupMembers{}
			groups[groupID] = group
		}

		switch roleValue {
		case p.prefillValue:
			group.prefills = append(group.prefills, fwkscheduling.ScoredEndpoint{Endpoint: endpoint, Score: 0})
		case p.decodeValue:
			group.decodes = append(group.decodes, fwkscheduling.ScoredEndpoint{Endpoint: endpoint, Score: 0})
		case p.leaderValue:
			group.leaders = append(group.leaders, endpoint)
		default:
			logger.V(logutil.DEBUG).Info(
				"endpoint role did not match PD role values, skipping",
				"groupID", groupID,
				"roleValue", roleValue,
			)
		}
	}

	for groupID, group := range groups {
		if len(group.leaders) == 0 || len(group.prefills) == 0 || len(group.decodes) == 0 {
			logger.V(logutil.DEFAULT).Info(
				"skipping invalid PD group",
				"groupID", groupID,
				"leaderCount", len(group.leaders),
				"prefillCount", len(group.prefills),
				"decodeCount", len(group.decodes),
			)
			continue
		}

		info := &pdg.Info{
			GroupID:     groupID,
			PrefillPods: group.prefills,
			DecodePods:  group.decodes,
		}
		// Store the freshly assembled group without cloning so later PrepareData
		// plugins that add endpoint attributes remain visible when the group is
		// read and cloned by scheduling plugins.
		pdg.SetOwned(group.leaders[rand.Intn(len(group.leaders))], info)
	}

	return nil
}

func parseAndValidateParameters(raw json.RawMessage) (Config, error) {
	params := pluginParameters{}
	if err := utils.DecodePluginParameters(raw, &params); err != nil {
		return Config{}, err
	}
	config := Config{
		RoleLabelName:  strings.TrimSpace(params.RoleLabelName),
		GroupLabelName: strings.TrimSpace(params.GroupLabelName),
		PrefillValue:   strings.TrimSpace(params.PrefillValue),
		DecodeValue:    strings.TrimSpace(params.DecodeValue),
		LeaderValue:    strings.TrimSpace(params.LeaderValue),
	}
	if missingRequiredConfigValue(config) {
		return Config{}, fmt.Errorf("pdLabelName, pdGroupLabelName, prefillValue, decodeValue, and leaderValue are required")
	}
	return config, nil
}

func missingRequiredConfigValue(config Config) bool {
	requiredValues := []string{
		config.RoleLabelName,
		config.GroupLabelName,
		config.PrefillValue,
		config.DecodeValue,
		config.LeaderValue,
	}
	for _, value := range requiredValues {
		if value == "" {
			return true
		}
	}
	return false
}
