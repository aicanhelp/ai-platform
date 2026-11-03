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


package l3

import (
	"testing"

	"github.com/go-logr/logr"

	"gitcode.com/openFuyao/cache-indexer/pkg/apis"
)

func mc(lora string, blockSize int64) apis.ModelContext {
	return apis.ModelContext{LoraName: lora, BlockSize: blockSize}
}

func bh(s ...string) []apis.BlockHash {
	out := make([]apis.BlockHash, len(s))
	for i, k := range s {
		out[i] = apis.BlockHash(k)
	}
	return out
}

func recs(ep apis.TransportEndpoint, replicaType string, keys ...string) []apis.L3ReplicaRecord {
	out := make([]apis.L3ReplicaRecord, 0, len(keys))
	for _, k := range keys {
		out = append(out, apis.L3ReplicaRecord{
			BlockKey:          apis.BlockHash(k),
			TransportEndpoint: ep,
			ReplicaType:       replicaType,
		})
	}
	return out
}

func TestApplyAndMatchPrefix_Basic(t *testing.T) {
	idx := New(logr.Discard())
	ctx := mc("m1", 16)

	if err := idx.ApplyReplicaSnapshot(ctx, "ep-A", recs("ep-A", "MEMORY", "k1", "k2", "k3")); err != nil {
		t.Fatal(err)
	}
	res, _ := idx.MatchPrefix(ctx, "ep-A", bh("k1", "k2", "k3", "k4"))
	if res.MatchedBlocks != 3 {
		t.Errorf("matched=%d want 3", res.MatchedBlocks)
	}
	if res.MatchedTokens != 3*16 {
		t.Errorf("tokens=%d want 48", res.MatchedTokens)
	}
	// Unknown endpoint → 0
	res, _ = idx.MatchPrefix(ctx, "ep-X", bh("k1"))
	if res.MatchedBlocks != 0 {
		t.Errorf("unknown endpoint matched=%d want 0", res.MatchedBlocks)
	}
}

func TestMatchPrefix_StopsAtFirstMiss(t *testing.T) {
	idx := New(logr.Discard())
	ctx := mc("m1", 8)
	_ = idx.ApplyReplicaSnapshot(ctx, "ep", recs("ep", "MEM", "a", "b", "d"))
	res, _ := idx.MatchPrefix(ctx, "ep", bh("a", "b", "c", "d"))
	if res.MatchedBlocks != 2 {
		t.Errorf("matched=%d want 2 (truncated at first miss)", res.MatchedBlocks)
	}
}

func TestMatchPrefix_NoMediumGate(t *testing.T) {
	// L3 does not gate on medium/replica type; even an empty replicaType
	// row is a hit.
	idx := New(logr.Discard())
	ctx := mc("m1", 16)
	_ = idx.ApplyReplicaSnapshot(ctx, "ep", recs("ep", "" /*no type*/, "k1"))
	res, _ := idx.MatchPrefix(ctx, "ep", bh("k1"))
	if res.MatchedBlocks != 1 {
		t.Errorf("expected hit even with empty ReplicaType, got %d", res.MatchedBlocks)
	}
}

func TestRemove_ReapsForwardEntry(t *testing.T) {
	idx := New(logr.Discard())
	ctx := mc("m1", 16)
	_ = idx.ApplyReplicaSnapshot(ctx, "epA", recs("epA", "M", "k1", "k2"))
	_ = idx.ApplyReplicaSnapshot(ctx, "epB", recs("epB", "M", "k1"))

	_ = idx.RemoveReplicaKeys(ctx, "epA", bh("k1"))
	res, _ := idx.MatchPrefix(ctx, "epA", bh("k1"))
	if res.MatchedBlocks != 0 {
		t.Errorf("epA after remove k1: matched=%d want 0", res.MatchedBlocks)
	}
	res, _ = idx.MatchPrefix(ctx, "epB", bh("k1"))
	if res.MatchedBlocks != 1 {
		t.Errorf("epB still has k1: matched=%d want 1", res.MatchedBlocks)
	}

	_ = idx.RemoveReplicaKeys(ctx, "epB", bh("k1"))
	cd := idx.(*indexer).getContext(ctx)
	if cd == nil {
		t.Fatal("context disappeared")
	}
	cd.mu.RLock()
	if _, present := cd.keyIndex["k1"]; present {
		t.Errorf("keyIndex[k1] should be reaped after last endpoint removed")
	}
	if _, present := cd.serverIndex["epB"]; present {
		t.Errorf("serverIndex[epB] should be removed when empty")
	}
	cd.mu.RUnlock()
}

func TestSharedPointer_BothIndexes(t *testing.T) {
	idx := New(logr.Discard())
	ctx := mc("m1", 16)
	_ = idx.ApplyReplicaSnapshot(ctx, "epA", recs("epA", "MEM", "k1"))
	cd := idx.(*indexer).getContext(ctx)
	cd.mu.RLock()
	defer cd.mu.RUnlock()
	stateRev := cd.serverIndex["epA"]["k1"]
	stateFwd := cd.keyIndex["k1"].ServerStates["epA"]
	if stateRev != stateFwd {
		t.Error("serverIndex and keyIndex must share *L3ServerState pointer")
	}
}

func TestApply_ReplicaTypeUnioned(t *testing.T) {
	idx := New(logr.Discard())
	ctx := mc("m1", 16)
	_ = idx.ApplyReplicaSnapshot(ctx, "ep", recs("ep", "MEM", "k1"))
	_ = idx.ApplyReplicaSnapshot(ctx, "ep", recs("ep", "DISK", "k1"))
	cd := idx.(*indexer).getContext(ctx)
	cd.mu.RLock()
	defer cd.mu.RUnlock()
	state := cd.serverIndex["ep"]["k1"]
	if _, ok := state.ReplicaTypeSet["MEM"]; !ok {
		t.Error("MEM not in set")
	}
	if _, ok := state.ReplicaTypeSet["DISK"]; !ok {
		t.Error("DISK not in set")
	}
}

func TestApply_RecordEndpointOverridesArg(t *testing.T) {
	// rec.TransportEndpoint takes precedence over the call-site endpoint.
	idx := New(logr.Discard())
	ctx := mc("m1", 16)
	rs := []apis.L3ReplicaRecord{{BlockKey: "k1", TransportEndpoint: "ep-real", ReplicaType: "M"}}
	_ = idx.ApplyReplicaSnapshot(ctx, "ep-default", rs)
	res, _ := idx.MatchPrefix(ctx, "ep-real", bh("k1"))
	if res.MatchedBlocks != 1 {
		t.Errorf("expected hit on ep-real, got %d", res.MatchedBlocks)
	}
	res, _ = idx.MatchPrefix(ctx, "ep-default", bh("k1"))
	if res.MatchedBlocks != 0 {
		t.Errorf("expected no hit on ep-default, got %d", res.MatchedBlocks)
	}
}

func TestApply_SkipsEmptyEndpointAndKey(t *testing.T) {
	idx := New(logr.Discard())
	ctx := mc("m1", 16)
	rs := []apis.L3ReplicaRecord{
		{BlockKey: "", TransportEndpoint: "ep", ReplicaType: "M"},     // skip
		{BlockKey: "k1", TransportEndpoint: "", ReplicaType: "M"},     // skip (no fallback)
		{BlockKey: "k2", TransportEndpoint: "ep", ReplicaType: "M"},   // keep
	}
	_ = idx.ApplyReplicaSnapshot(ctx, "" /*no fallback*/, rs)
	res, _ := idx.MatchPrefix(ctx, "ep", bh("k2"))
	if res.MatchedBlocks != 1 {
		t.Errorf("expected k2 stored, got %d", res.MatchedBlocks)
	}
	res, _ = idx.MatchPrefix(ctx, "ep", bh("k1"))
	if res.MatchedBlocks != 0 {
		t.Errorf("k1 with no endpoint must be skipped, got %d", res.MatchedBlocks)
	}
}
