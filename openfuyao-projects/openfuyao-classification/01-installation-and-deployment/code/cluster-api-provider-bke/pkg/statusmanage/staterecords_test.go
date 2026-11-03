/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package statusmanage

import (
	"testing"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
)

func TestStatusRecord_Inc(t *testing.T) {
	sr := &StatusRecord{StatusCount: 5}
	sr.Inc()
	if sr.StatusCount != 6 {
		t.Errorf("expected 6, got %d", sr.StatusCount)
	}
}

func TestStatusRecord_Dec(t *testing.T) {
	sr := &StatusRecord{StatusCount: 5}
	sr.Dec()
	if sr.StatusCount != 4 {
		t.Errorf("expected 4, got %d", sr.StatusCount)
	}
}

func TestStatusRecord_Reset(t *testing.T) {
	sr := &StatusRecord{StatusCount: 5, LatestFailedState: "failed"}
	sr.Reset()
	if sr.StatusCount != 0 || sr.LatestFailedState != "" {
		t.Errorf("expected count=0 and empty state, got count=%d, state=%s", sr.StatusCount, sr.LatestFailedState)
	}
}

func TestStatusRecord_Equal(t *testing.T) {
	sr := &StatusRecord{LatestFailedState: "testState"}
	if !sr.Equal("testState") {
		t.Error("expected true")
	}
	if sr.Equal("otherState") {
		t.Error("expected false")
	}
}

func TestStatusRecord_AllowFailed(t *testing.T) {
	ReconcileAllowedFailedCount = 10
	sr := &StatusRecord{StatusCount: 5}
	if !sr.AllowFailed() {
		t.Error("expected true when count < limit")
	}
	sr.StatusCount = 10
	if sr.AllowFailed() {
		t.Error("expected false when count >= limit")
	}
}

func TestStatusRecord_SetLatestFailedState(t *testing.T) {
	sr := &StatusRecord{}
	sr.SetLatestFailedState("failed")
	if sr.LatestFailedState != "failed" {
		t.Errorf("expected 'failed', got %s", sr.LatestFailedState)
	}
}

func TestStatusRecord_SetLatestNormalState(t *testing.T) {
	sr := &StatusRecord{}
	sr.SetLatestNormalState("normal")
	if sr.LatestNormalState != "normal" {
		t.Errorf("expected 'normal', got %s", sr.LatestNormalState)
	}
}

func TestStatusRecord_SetCurrentClusterState(t *testing.T) {
	sr := &StatusRecord{}
	state := confv1beta1.ClusterStatus("Ready")
	sr.SetCurrentClusterState(state)
	if sr.CurrentClusterState != state {
		t.Errorf("expected Ready, got %v", sr.CurrentClusterState)
	}
}
