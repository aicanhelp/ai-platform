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

// Package predictioninput provides data-layer attributes for prediction inputs.
package predictioninput

import (
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"

	"hermes-router/pkg/epp/internal/predictiondata"
)

const (
	// PredictionInputKey is an exported constant used by Hermes Router.
	PredictionInputKey = "prediction_input"
	// MissingFeatureReasonKey is an exported constant used by Hermes Router.
	MissingFeatureReasonKey = "prediction_missing_feature_reason"
)

// InputSnapshot describes InputSnapshot data used by Hermes Router.
type InputSnapshot struct {
	// Input stores the Input value.
	Input predictiondata.Input
}

// Clone returns an independent copy of the receiver.
func (s *InputSnapshot) Clone() fwkdl.Cloneable {
	if s == nil {
		return (*InputSnapshot)(nil)
	}
	cloned := *s
	return &cloned
}

// MissingFeatureReason describes MissingFeatureReason data used by Hermes Router.
type MissingFeatureReason struct {
	// Reason stores the Reason value.
	Reason string
}

// Clone returns an independent copy of the receiver.
func (r *MissingFeatureReason) Clone() fwkdl.Cloneable {
	if r == nil {
		return (*MissingFeatureReason)(nil)
	}
	cloned := *r
	return &cloned
}
