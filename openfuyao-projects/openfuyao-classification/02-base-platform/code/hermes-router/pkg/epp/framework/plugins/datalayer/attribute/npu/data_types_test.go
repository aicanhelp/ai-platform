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

package npu

import (
	"testing"
	"time"
)

func TestCloneNilReceiverReturnsTypedNil(t *testing.T) {
	var src *State
	cloned := src.Clone()
	if cloned == nil {
		t.Fatalf("Clone() on nil should return a typed nil Cloneable, got nil interface")
	}
	state, ok := cloned.(*State)
	if !ok {
		t.Fatalf("Clone() returned %T, want *State", cloned)
	}
	if state != nil {
		t.Fatalf("Clone() on nil should yield (*State)(nil), got %p", state)
	}
}

func populatedState() *State {
	return &State{
		ObservationTime:            time.Unix(1_000, 0),
		DataStatus:                 DataPresent,
		DeviceCount:                2,
		AICoreUtilizationAvg:       49,
		AICoreUtilizationMax:       51,
		HBMUtilizationAvg:          40,
		HBMUtilizationMax:          50,
		HBMBandwidthUtilizationAvg: 40,
		HBMBandwidthUtilizationMax: 60,
		Devices: []DeviceState{
			{DeviceID: "0", AICoreUtilization: 47, HBMUtilization: 30, HBMBandwidthUtilization: 20},
			{DeviceID: "1", AICoreUtilization: 51, HBMUtilization: 50, HBMBandwidthUtilization: 60},
		},
	}
}

func TestClonePopulatedStateCopiesAllFields(t *testing.T) {
	src := populatedState()
	cloned, ok := src.Clone().(*State)
	if !ok || cloned == nil {
		t.Fatalf("Clone() returned %T", src.Clone())
	}
	if cloned == src {
		t.Fatalf("Clone() should return a new pointer, got the same %p", src)
	}
	if cloned.ObservationTime != src.ObservationTime ||
		cloned.DataStatus != src.DataStatus ||
		cloned.DeviceCount != src.DeviceCount ||
		cloned.AICoreUtilizationAvg != src.AICoreUtilizationAvg ||
		cloned.AICoreUtilizationMax != src.AICoreUtilizationMax ||
		cloned.HBMUtilizationAvg != src.HBMUtilizationAvg ||
		cloned.HBMUtilizationMax != src.HBMUtilizationMax ||
		cloned.HBMBandwidthUtilizationAvg != src.HBMBandwidthUtilizationAvg ||
		cloned.HBMBandwidthUtilizationMax != src.HBMBandwidthUtilizationMax {
		t.Fatalf("Clone() field mismatch: src=%+v cloned=%+v", src, cloned)
	}
	if len(cloned.Devices) != len(src.Devices) {
		t.Fatalf("Clone Devices len = %d, want %d", len(cloned.Devices), len(src.Devices))
	}
}

func TestCloneDeepCopiesDevices(t *testing.T) {
	src := populatedState()
	cloned := src.Clone().(*State)
	const mutated = 99.0
	cloned.Devices[0].AICoreUtilization = mutated

	if src.Devices[0].AICoreUtilization == mutated {
		t.Fatalf("mutating clone Devices changed source: src=%v", src.Devices[0].AICoreUtilization)
	}
}

func TestCloneNilDevicesStaysNil(t *testing.T) {
	src := &State{DeviceCount: 0, Devices: nil}
	cloned := src.Clone().(*State)
	if cloned.Devices != nil {
		t.Fatalf("Clone() should preserve nil Devices, got %v", cloned.Devices)
	}
}

func TestNPUDataStatusConstants(t *testing.T) {
	if DataPresent != "Present" {
		t.Fatalf("DataPresent = %q, want %q", DataPresent, "Present")
	}
	if DataStale != "Stale" {
		t.Fatalf("DataStale = %q, want %q", DataStale, "Stale")
	}
}

func TestDefaultStateAttributeKey(t *testing.T) {
	if DefaultStateAttributeKey != "NPUStateKey" {
		t.Fatalf("DefaultStateAttributeKey = %q, want %q", DefaultStateAttributeKey, "NPUStateKey")
	}
}
