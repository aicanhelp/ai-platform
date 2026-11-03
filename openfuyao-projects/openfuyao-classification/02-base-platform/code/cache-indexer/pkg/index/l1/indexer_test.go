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


package l1

import (
	"testing"

	"github.com/go-logr/logr"

	"gitcode.com/openFuyao/cache-indexer/pkg/apis"
)

const (
	testServerA     = "10.0.0.1"
	testServerB     = "10.0.0.2"
	testModelName   = "m1"
	testBlockSize16 = int64(16)
	testGPU         = "GPU"
	testCPU         = "CPU"
	testKeyOne      = "k1"
	testKeyTwo      = "k2"
)

func mc(loraName string, blockSize int64) apis.ModelContext {
	return apis.ModelContext{LoraName: loraName, BlockSize: blockSize}
}

func keys(s ...string) []apis.BlockHash {
	out := make([]apis.BlockHash, len(s))
	for i, k := range s {
		out[i] = apis.BlockHash(k)
	}
	return out
}

func TestUpsertAndMatchPrefix_GPUOnly(t *testing.T) {
	idx := New(logr.Discard())
	ctx := mc("m1", 16)

	// Server A holds k1,k2 with GPU; server B holds k1 only with non-GPU.
	if err := idx.UpsertStoredEvent(&apis.L1StoredEvent{
		ModelContext: ctx,
		ServerIP:     "10.0.0.1",
		BlockHashes:  keys("k1", "k2", "k3"),
		BlockSize:    16,
		Medium:       "GPU",
	}); err != nil {
		t.Fatal(err)
	}
	if err := idx.UpsertStoredEvent(&apis.L1StoredEvent{
		ModelContext: ctx,
		ServerIP:     "10.0.0.2",
		BlockHashes:  keys("k1"),
		BlockSize:    16,
		Medium:       "CPU",
	}); err != nil {
		t.Fatal(err)
	}

	// A: full prefix hit on k1,k2,k3 (k3 missing → matched=2 since req is k1,k2,k3,k4)
	res, _ := idx.MatchPrefix(ctx, "10.0.0.1", keys("k1", "k2", "k3", "k4"))
	if res.MatchedBlocks != 3 {
		t.Errorf("A matched = %d, want 3", res.MatchedBlocks)
	}
	if res.MatchedTokens != 3*16 {
		t.Errorf("A tokens = %d, want %d", res.MatchedTokens, 3*16)
	}
	// B: k1 exists but only CPU medium → no GPU hit → matched=0
	res, _ = idx.MatchPrefix(ctx, "10.0.0.2", keys("k1", "k2"))
	if res.MatchedBlocks != 0 {
		t.Errorf("B (CPU only) matched = %d, want 0", res.MatchedBlocks)
	}
	// Unknown server: matched=0
	res, _ = idx.MatchPrefix(ctx, "10.0.0.99", keys("k1"))
	if res.MatchedBlocks != 0 {
		t.Errorf("unknown server matched = %d, want 0", res.MatchedBlocks)
	}
	// Empty keys: matched=0
	res, _ = idx.MatchPrefix(ctx, "10.0.0.1", nil)
	if res.MatchedBlocks != 0 {
		t.Errorf("empty keys matched = %d, want 0", res.MatchedBlocks)
	}
}

func TestMatchPrefix_StopsAtFirstMiss(t *testing.T) {
	idx := New(logr.Discard())
	ctx := mc("m1", 8)
	_ = idx.UpsertStoredEvent(&apis.L1StoredEvent{
		ModelContext: ctx,
		ServerIP:     "ip",
		BlockHashes:  keys("a", "b", "d"), // missing c
		Medium:       "GPU",
	})
	// Request a,b,c,d — must stop at c.
	res, _ := idx.MatchPrefix(ctx, "ip", keys("a", "b", "c", "d"))
	if res.MatchedBlocks != 2 {
		t.Errorf("matched = %d, want 2 (truncated at first miss)", res.MatchedBlocks)
	}
}

func TestRemoveBlocks_ReapsForwardEntry(t *testing.T) {
	idx := New(logr.Discard())
	ctx := mc("m1", 16)
	_ = idx.UpsertStoredEvent(&apis.L1StoredEvent{
		ModelContext: ctx,
		ServerIP:     "ipA",
		BlockHashes:  keys("k1", "k2"),
		Medium:       "GPU",
	})
	_ = idx.UpsertStoredEvent(&apis.L1StoredEvent{
		ModelContext: ctx,
		ServerIP:     "ipB",
		BlockHashes:  keys("k1"),
		Medium:       "GPU",
	})

	// Remove k1 from A — k1 still on B.
	_ = idx.RemoveBlocks(&apis.L1RemovedEvent{
		ModelContext: ctx,
		ServerIP:     "ipA",
		BlockHashes:  keys("k1"),
	})
	res, _ := idx.MatchPrefix(ctx, "ipA", keys("k1"))
	if res.MatchedBlocks != 0 {
		t.Errorf("A after remove k1: matched=%d, want 0", res.MatchedBlocks)
	}
	res, _ = idx.MatchPrefix(ctx, "ipB", keys("k1"))
	if res.MatchedBlocks != 1 {
		t.Errorf("B still has k1: matched=%d, want 1", res.MatchedBlocks)
	}

	// Now remove k1 from B too — forward entry should be reaped.
	_ = idx.RemoveBlocks(&apis.L1RemovedEvent{
		ModelContext: ctx,
		ServerIP:     "ipB",
		BlockHashes:  keys("k1"),
	})
	cd := idx.(*indexer).getContext(ctx)
	if cd == nil {
		t.Fatal("context disappeared unexpectedly")
	}
	cd.mu.RLock()
	_, present := cd.keyIndex["k1"]
	cd.mu.RUnlock()
	if present {
		t.Errorf("keyIndex[k1] should be reaped after last server removed")
	}
}

func TestClearServer_ReapsAllForwardEntries(t *testing.T) {
	idx := New(logr.Discard())
	ctx := mc("m1", 16)
	_ = idx.UpsertStoredEvent(&apis.L1StoredEvent{
		ModelContext: ctx,
		ServerIP:     "ipA",
		BlockHashes:  keys("k1", "k2", "k3"),
		Medium:       "GPU",
	})
	if err := idx.ClearServer(ctx, "ipA"); err != nil {
		t.Fatal(err)
	}
	res, _ := idx.MatchPrefix(ctx, "ipA", keys("k1", "k2", "k3"))
	if res.MatchedBlocks != 0 {
		t.Errorf("after ClearServer matched = %d, want 0", res.MatchedBlocks)
	}
	cd := idx.(*indexer).getContext(ctx)
	cd.mu.RLock()
	for _, k := range []apis.BlockHash{"k1", "k2", "k3"} {
		if _, ok := cd.keyIndex[k]; ok {
			t.Errorf("keyIndex[%s] not reaped", k)
		}
	}
	if _, ok := cd.serverIndex["ipA"]; ok {
		t.Errorf("serverIndex[ipA] not removed")
	}
	cd.mu.RUnlock()
}

func TestSharedPointer_OneStateInBothIndexes(t *testing.T) {
	idx := New(logr.Discard())
	ctx := mc("m1", 16)
	_ = idx.UpsertStoredEvent(&apis.L1StoredEvent{
		ModelContext: ctx,
		ServerIP:     "ipA",
		BlockHashes:  keys("k1"),
		Medium:       "GPU",
		DPRank:       3,
	})
	cd := idx.(*indexer).getContext(ctx)
	cd.mu.RLock()
	defer cd.mu.RUnlock()
	stateRev := cd.serverIndex["ipA"]["k1"]
	stateFwd := cd.keyIndex["k1"].ServerStates["ipA"]
	if stateRev != stateFwd {
		t.Errorf("serverIndex and keyIndex must share *L1ServerState pointer")
	}
	if _, ok := stateRev.MediumSet["GPU"]; !ok {
		t.Errorf("MediumSet missing GPU")
	}
	if _, ok := stateRev.DPRankSet[3]; !ok {
		t.Errorf("DPRankSet missing 3")
	}
}

func TestUpsert_MultipleMediumsUnioned(t *testing.T) {
	idx := New(logr.Discard())
	ctx := mc("m1", 16)
	_ = idx.UpsertStoredEvent(&apis.L1StoredEvent{
		ModelContext: ctx, ServerIP: "ipA", BlockHashes: keys("k1"), Medium: "GPU",
	})
	_ = idx.UpsertStoredEvent(&apis.L1StoredEvent{
		ModelContext: ctx, ServerIP: "ipA", BlockHashes: keys("k1"), Medium: "CPU",
	})
	cd := idx.(*indexer).getContext(ctx)
	cd.mu.RLock()
	state := cd.serverIndex["ipA"]["k1"]
	cd.mu.RUnlock()
	if _, ok := state.MediumSet["GPU"]; !ok {
		t.Error("missing GPU after union")
	}
	if _, ok := state.MediumSet["CPU"]; !ok {
		t.Error("missing CPU after union")
	}
}

func TestUpsertStoredEvent_EmptyInputNoOp(t *testing.T) {
	idx := New(logr.Discard())
	cases := []*apis.L1StoredEvent{
		nil,
		{ModelContext: mc(testModelName, testBlockSize16)},
		{ModelContext: mc(testModelName, testBlockSize16), ServerIP: testServerA},
	}
	for _, item := range cases {
		if err := idx.UpsertStoredEvent(item); err != nil {
			t.Fatalf("UpsertStoredEvent(%+v): %v", item, err)
		}
	}
	res, _ := idx.MatchPrefix(mc(testModelName, testBlockSize16), testServerA, keys(testKeyOne))
	if res.MatchedBlocks != 0 {
		t.Fatalf("matched=%d want 0", res.MatchedBlocks)
	}
}

func TestUpsertStoredEvent_UnionDPRankAndGroupIndex(t *testing.T) {
	idx := New(logr.Discard())
	ctx := mc(testModelName, testBlockSize16)
	groupOne := int64(1)
	groupTwo := int64(2)
	_ = idx.UpsertStoredEvent(&apis.L1StoredEvent{
		ModelContext: ctx,
		ServerIP:     testServerA,
		BlockHashes:  keys(testKeyOne),
		Medium:       testGPU,
		DPRank:       3,
		GroupIndex:   &groupOne,
	})
	_ = idx.UpsertStoredEvent(&apis.L1StoredEvent{
		ModelContext: ctx,
		ServerIP:     testServerA,
		BlockHashes:  keys(testKeyOne),
		Medium:       testCPU,
		DPRank:       5,
		GroupIndex:   &groupTwo,
	})

	cd := idx.(*indexer).getContext(ctx)
	cd.mu.RLock()
	state := cd.serverIndex[testServerA][testKeyOne]
	cd.mu.RUnlock()
	if len(state.DPRankSet) != 2 || len(state.GroupIndexSet) != 2 || len(state.MediumSet) != 2 {
		t.Fatalf("state=%+v", state)
	}
}

func TestRemoveBlocks_UnknownInputNoOp(t *testing.T) {
	idx := New(logr.Discard())
	ctx := mc(testModelName, testBlockSize16)
	cases := []*apis.L1RemovedEvent{
		nil,
		{ModelContext: ctx},
		{ModelContext: ctx, ServerIP: testServerA},
		{ModelContext: ctx, ServerIP: testServerA, BlockHashes: keys(testKeyOne)},
	}
	for _, item := range cases {
		if err := idx.RemoveBlocks(item); err != nil {
			t.Fatalf("RemoveBlocks(%+v): %v", item, err)
		}
	}
}

func TestClearServer_EmptyServerNoOp(t *testing.T) {
	idx := New(logr.Discard())
	if err := idx.ClearServer(mc(testModelName, testBlockSize16), ""); err != nil {
		t.Fatalf("ClearServer empty server: %v", err)
	}
}

func TestMatchPrefix_BlockSizeZeroLeavesMatchedTokensZero(t *testing.T) {
	idx := New(logr.Discard())
	ctx := mc(testModelName, 0)
	_ = idx.UpsertStoredEvent(&apis.L1StoredEvent{
		ModelContext: ctx,
		ServerIP:     testServerA,
		BlockHashes:  keys(testKeyOne, testKeyTwo),
		Medium:       testGPU,
	})
	res, _ := idx.MatchPrefix(ctx, testServerA, keys(testKeyOne, testKeyTwo))
	if res.MatchedBlocks != 2 || res.MatchedTokens != 0 {
		t.Fatalf("result=%+v", res)
	}
}
