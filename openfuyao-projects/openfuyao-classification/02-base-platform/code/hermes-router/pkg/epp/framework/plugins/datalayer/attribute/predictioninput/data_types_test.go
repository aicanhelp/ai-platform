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

package predictioninput

import (
	"testing"

	"hermes-router/pkg/epp/internal/predictiondata"
)

func TestInputSnapshotCloneIsIndependent(t *testing.T) {
	original := &InputSnapshot{Input: predictiondata.Input{InputID: 1, ServerIP: "10.0.0.1"}}
	cloned, ok := original.Clone().(*InputSnapshot)
	if !ok || cloned == nil {
		t.Fatalf("expected *InputSnapshot clone, got %T", cloned)
	}
	cloned.Input.ServerIP = "10.0.0.2"
	if original.Input.ServerIP != "10.0.0.1" {
		t.Fatalf("expected original to be unchanged, got %q", original.Input.ServerIP)
	}
}

func TestMissingFeatureReasonCloneIsIndependent(t *testing.T) {
	original := &MissingFeatureReason{Reason: "missing npu state"}
	cloned, ok := original.Clone().(*MissingFeatureReason)
	if !ok || cloned == nil {
		t.Fatalf("expected *MissingFeatureReason clone, got %T", cloned)
	}
	cloned.Reason = "changed"
	if original.Reason != "missing npu state" {
		t.Fatalf("expected original to be unchanged, got %q", original.Reason)
	}
}

func TestAttributeKeysAreStable(t *testing.T) {
	if PredictionInputKey != "prediction_input" {
		t.Fatalf("unexpected prediction input key %q", PredictionInputKey)
	}
	if MissingFeatureReasonKey != "prediction_missing_feature_reason" {
		t.Fatalf("unexpected missing feature reason key %q", MissingFeatureReasonKey)
	}
}
