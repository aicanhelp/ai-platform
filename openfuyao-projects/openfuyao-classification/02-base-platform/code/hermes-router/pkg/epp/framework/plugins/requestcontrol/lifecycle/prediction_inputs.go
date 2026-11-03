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

package lifecycle

import (
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	predictioninputattr "hermes-router/pkg/epp/framework/plugins/datalayer/attribute/predictioninput"
	prerequest "hermes-router/pkg/epp/framework/plugins/requestcontrol/prerequest"
	"hermes-router/pkg/epp/internal/predictiondata"
)

func selectedPredictionInputs(
	selected fwkscheduling.Endpoint,
	route *prerequest.Route,
) (map[string]predictiondata.Input, map[string]string) {
	if selected == nil {
		return nil, nil
	}
	if prefillEP, decodeEP, ok := selectedMemberEndpoints(selected, route); ok {
		inputs := make(map[string]predictiondata.Input, 2)
		reasons := make(map[string]string, 2)
		if isPDRoute(route) {
			collectPredictionInput(predictiondata.RolePrefill, prefillEP, inputs, reasons)
		}
		collectPredictionInput(predictiondata.RoleDecode, decodeEP, inputs, reasons)
		return emptyToNil(inputs), emptyStringMapToNil(reasons)
	}

	inputs := make(map[string]predictiondata.Input, 1)
	reasons := make(map[string]string, 1)
	collectPredictionInput(predictiondata.RoleLeader, selected, inputs, reasons)
	return emptyToNil(inputs), emptyStringMapToNil(reasons)
}

func collectPredictionInput(
	role string,
	endpoint fwkscheduling.Endpoint,
	inputs map[string]predictiondata.Input,
	reasons map[string]string,
) {
	if endpoint == nil {
		return
	}
	if input, ok := predictionInput(endpoint); ok {
		if inputs != nil {
			inputs[role] = input
		}
		return
	}
	if reason, ok := missingFeatureReason(endpoint); ok {
		if reasons != nil {
			reasons[role] = reason
		}
	}
}

func predictionInput(endpoint fwkscheduling.Endpoint) (predictiondata.Input, bool) {
	value, ok := endpoint.Get(predictioninputattr.PredictionInputKey)
	if !ok {
		return predictiondata.Input{}, false
	}
	snapshot, ok := value.(*predictioninputattr.InputSnapshot)
	if !ok || snapshot == nil {
		return predictiondata.Input{}, false
	}
	return snapshot.Input, true
}

func missingFeatureReason(endpoint fwkscheduling.Endpoint) (string, bool) {
	value, ok := endpoint.Get(predictioninputattr.MissingFeatureReasonKey)
	if !ok {
		return "", false
	}
	reason, ok := value.(*predictioninputattr.MissingFeatureReason)
	if !ok || reason == nil || reason.Reason == "" {
		return "", false
	}
	return reason.Reason, true
}

func emptyToNil(inputs map[string]predictiondata.Input) map[string]predictiondata.Input {
	if len(inputs) == 0 {
		return nil
	}
	return inputs
}

func emptyStringMapToNil(values map[string]string) map[string]string {
	if len(values) == 0 {
		return nil
	}
	return values
}
