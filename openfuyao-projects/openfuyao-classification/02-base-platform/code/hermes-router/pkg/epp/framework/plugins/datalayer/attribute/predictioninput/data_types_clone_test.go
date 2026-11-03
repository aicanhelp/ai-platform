/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *         http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package predictioninput

import (
	"testing"

	"hermes-router/pkg/epp/internal/predictiondata"
)

func TestInputSnapshotCloneReturnsDeepCopy(t *testing.T) {
	var nilSnapshot *InputSnapshot
	nilClone, ok := nilSnapshot.Clone().(*InputSnapshot)
	if !ok || nilClone != nil {
		t.Fatalf("expected typed-nil clone for nil snapshot, got %+v", nilSnapshot.Clone())
	}

	snapshot := &InputSnapshot{Input: predictiondata.Input{InputID: 7, ServerIP: "10.0.94.1"}}
	cloned := snapshot.Clone().(*InputSnapshot)
	if cloned.Input.InputID != 7 || cloned.Input.ServerIP != "10.0.94.1" {
		t.Fatalf("unexpected cloned snapshot: %+v", cloned)
	}
	cloned.Input.InputID = 99
	if snapshot.Input.InputID != 7 {
		t.Fatal("expected Clone to return an independent copy")
	}
}

func TestMissingFeatureReasonCloneReturnsDeepCopy(t *testing.T) {
	var nilReason *MissingFeatureReason
	nilClone, ok := nilReason.Clone().(*MissingFeatureReason)
	if !ok || nilClone != nil {
		t.Fatalf("expected typed-nil clone for nil reason, got %+v", nilReason.Clone())
	}

	reason := &MissingFeatureReason{Reason: "no-prediction"}
	cloned := reason.Clone().(*MissingFeatureReason)
	if cloned.Reason != "no-prediction" {
		t.Fatalf("unexpected cloned reason: %+v", cloned)
	}
	cloned.Reason = "mutated"
	if reason.Reason != "no-prediction" {
		t.Fatal("expected Clone to return an independent copy")
	}
}
