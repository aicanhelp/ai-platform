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

// Package alwaysdisagg provides a Decider that always schedules Prefill (pd path).
package alwaysdisagg

import (
	"context"
	"encoding/json"

	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	"hermes-router/pkg/epp/framework/plugins/scheduling/profilehandler/decider"
	"hermes-router/pkg/epp/internal/utils"
)

const (
	// PluginType always schedules Prefill (pd path).
	PluginType = "always-disagg-prefill-decider"
)

// Decider always returns shouldRunPrefill=true.
type Decider struct {
	typedName fwkplugin.TypedName
}

var _ decider.Strategy = (*Decider)(nil)

// Factory constructs an always-disagg Decider.
func Factory(name string, raw json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	if err := utils.ValidateEmptyPluginParameters(raw); err != nil {
		return nil, err
	}
	return New(name), nil
}

// New constructs an always-disagg Decider.
func New(name string) *Decider {
	return &Decider{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: name},
	}
}

// TypedName returns the plugin type and instance name.
func (d *Decider) TypedName() fwkplugin.TypedName {
	return d.typedName
}

// Decide always requests Prefill scheduling.
func (d *Decider) Decide(
	_ context.Context,
	_ *fwkscheduling.LLMRequest,
	_ fwkscheduling.Endpoint,
) (bool, error) {
	return true, nil
}
