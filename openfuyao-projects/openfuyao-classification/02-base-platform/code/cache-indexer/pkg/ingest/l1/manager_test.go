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
	"context"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"net"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/go-zeromq/zmq4"
	"github.com/vmihailenco/msgpack/v5"

	"gitcode.com/openFuyao/cache-indexer/pkg/apis"
	"gitcode.com/openFuyao/cache-indexer/pkg/config"
	indexl1 "gitcode.com/openFuyao/cache-indexer/pkg/index/l1"
)

const (
	testSubPodIP         = "10.0.0.1"
	testSubPort          = int32(5557)
	testSubLora          = "lora-a"
	testSubGPU           = "GPU"
	testStringHashInput  = "AB"
	testStringHashHex    = "4142"
	testTSSeconds        = 2
	testTSMillis         = 2000
	testBackoffDuration  = 5 * time.Millisecond
	testRecvDelay        = 20 * time.Millisecond
	testZMQLocalEndpoint = "tcp://127.0.0.1:0"
	testLocalhostIP      = "127.0.0.1"
)

// encodeKVEventBatch builds a vLLM-like msgpack payload.
// It uses [ts, events, data_parallel_rank] and tagged event arrays.
func encodeKVEventBatch(t *testing.T, ts float64, events []interface{}, dpRank int64) []byte {
	t.Helper()
	batch := []interface{}{ts, events, dpRank}
	b, err := msgpack.Marshal(batch)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return b
}

func intHex(v uint64) apis.BlockHash {
	var buf [8]byte
	binary.BigEndian.PutUint64(buf[:], v)
	return apis.BlockHash(hex.EncodeToString(buf[:]))
}

func TestNormalizeBlockHash_IntBigEndianHex(t *testing.T) {
	cases := []struct {
		name string
		in   interface{}
		want string
	}{
		{"uint64", uint64(0x0123456789abcdef), "0123456789abcdef"},
		{"int64", int64(0x0123456789abcdef), "0123456789abcdef"},
		{"int", int(1), "0000000000000001"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			h, ok := normalizeBlockHash(c.in)
			if !ok {
				t.Fatalf("normalize returned ok=false for %v", c.in)
			}
			if string(h) != c.want {
				t.Errorf("got %q, want %q", h, c.want)
			}
		})
	}
}

func TestNormalizeBlockHash_BytesPath(t *testing.T) {
	raw := []byte{0xde, 0xad, 0xbe, 0xef}
	h, ok := normalizeBlockHash(raw)
	if !ok || string(h) != "deadbeef" {
		t.Errorf("bytes path: %q ok=%v", h, ok)
	}
}

func TestNormalizeBlockHash_NilUnsupported(t *testing.T) {
	if _, ok := normalizeBlockHash(nil); ok {
		t.Error("nil should return ok=false")
	}
	if _, ok := normalizeBlockHash(3.14); ok {
		t.Error("float should return ok=false")
	}
}

func TestNormalizeBlockHashList_FiltersInvalid(t *testing.T) {
	in := []interface{}{uint64(1), nil, []byte{0xab, 0xcd}, 3.14}
	out := normalizeBlockHashList(in)
	if len(out) != 2 {
		t.Fatalf("len = %d, want 2 (nil and float dropped)", len(out))
	}
	wantFirst := apis.BlockHash("0000000000000001")
	if out[0] != wantFirst {
		t.Errorf("[0] = %q, want %q", out[0], wantFirst)
	}
	if out[1] != "abcd" {
		t.Errorf("[1] = %q, want abcd", out[1])
	}
}

// Sanity: the byte representation chosen for ints is big-endian uint64,
// matching vLLM's `i.to_bytes(8, "big")`.
func TestIntEncodingIsBigEndianUint64(t *testing.T) {
	const v uint64 = 0x102030405060708
	var buf [8]byte
	binary.BigEndian.PutUint64(buf[:], v)
	if hex.EncodeToString(buf[:]) != "0102030405060708" {
		t.Fatalf("unexpected encoding: %s", hex.EncodeToString(buf[:]))
	}
	h, _ := normalizeBlockHash(v)
	if string(h) != "0102030405060708" {
		t.Errorf("normalize uint64 = %q, want 0102030405060708", h)
	}
}

func TestEventModelContext(t *testing.T) {
	mc := eventModelContext("lora-x", 16)
	if mc.LoraName != "lora-x" || mc.BlockSize != 16 {
		t.Errorf("got %+v", mc)
	}
}

func TestHandlePayload_BlockStored_ThenMatchPrefix(t *testing.T) {
	idx := indexl1.New(logr.Discard())
	sub := &subscription{
		target:  VLLMSubscribeTarget{PodIP: "10.0.0.1", ZMQPort: 5557},
		log:     logr.Discard(),
		indexer: idx,
		seenCtx: map[apis.ModelContext]struct{}{},
	}

	stored := []interface{}{
		"BlockStored",
		[]interface{}{uint64(0xaa), uint64(0xbb), uint64(0xcc)}, // block_hashes
		nil,                               // parent_block_hash
		[]interface{}{int64(1), int64(2)}, // token_ids
		int64(16),                         // block_size
		nil,                               // lora_id
		"GPU",                             // medium
		"adapter-a",                       // lora_name
		nil,                               // extra_keys
		int64(7),                          // group_idx
	}
	payload := encodeKVEventBatch(t, 1234.5, []interface{}{stored}, 0)
	if err := sub.handlePayload(payload); err != nil {
		t.Fatalf("handlePayload: %v", err)
	}

	mc := apis.ModelContext{LoraName: "adapter-a", BlockSize: 16}
	res, _ := idx.MatchPrefix(mc, "10.0.0.1",
		[]apis.BlockHash{intHex(0xaa), intHex(0xbb), intHex(0xcc), intHex(0xdd)})
	if res.MatchedBlocks != 3 {
		t.Errorf("matched = %d, want 3 (last is miss)", res.MatchedBlocks)
	}
}

func TestHandlePayload_BlockRemoved_AfterStored(t *testing.T) {
	idx := indexl1.New(logr.Discard())
	sub := &subscription{
		target:  VLLMSubscribeTarget{PodIP: "10.0.0.1"},
		log:     logr.Discard(),
		indexer: idx,
		seenCtx: map[apis.ModelContext]struct{}{},
	}
	// First store.
	stored := []interface{}{"BlockStored", []interface{}{uint64(1), uint64(2)},
		nil, nil, int64(16), nil, "GPU", "lora-a", nil, int64(0)}
	if err := sub.handlePayload(encodeKVEventBatch(t, 1, []interface{}{stored}, 0)); err != nil {
		t.Fatal(err)
	}

	// Then remove block 1.
	removed := []interface{}{"BlockRemoved", []interface{}{uint64(1)}, "GPU", int64(0)}
	if err := sub.handlePayload(encodeKVEventBatch(t, 2, []interface{}{removed}, 0)); err != nil {
		t.Fatal(err)
	}

	mc := apis.ModelContext{LoraName: "lora-a", BlockSize: 16}
	res, _ := idx.MatchPrefix(mc, "10.0.0.1", []apis.BlockHash{intHex(1)})
	if res.MatchedBlocks != 0 {
		t.Errorf("after BlockRemoved, matched = %d, want 0", res.MatchedBlocks)
	}
	res, _ = idx.MatchPrefix(mc, "10.0.0.1", []apis.BlockHash{intHex(2)})
	if res.MatchedBlocks != 1 {
		t.Errorf("block 2 still present, matched = %d, want 1", res.MatchedBlocks)
	}
}

func TestHandlePayload_AllBlocksCleared(t *testing.T) {
	idx := indexl1.New(logr.Discard())
	sub := &subscription{
		target:  VLLMSubscribeTarget{PodIP: "10.0.0.1"},
		log:     logr.Discard(),
		indexer: idx,
		seenCtx: map[apis.ModelContext]struct{}{},
	}
	stored := []interface{}{"BlockStored", []interface{}{uint64(1), uint64(2)},
		nil, nil, int64(16), nil, "GPU", "lora-a", nil, int64(0)}
	if err := sub.handlePayload(encodeKVEventBatch(t, 1, []interface{}{stored}, 0)); err != nil {
		t.Fatal(err)
	}
	cleared := []interface{}{"AllBlocksCleared"}
	if err := sub.handlePayload(encodeKVEventBatch(t, 2, []interface{}{cleared}, 0)); err != nil {
		t.Fatal(err)
	}

	mc := apis.ModelContext{LoraName: "lora-a", BlockSize: 16}
	res, _ := idx.MatchPrefix(mc, "10.0.0.1", []apis.BlockHash{intHex(1), intHex(2)})
	if res.MatchedBlocks != 0 {
		t.Errorf("after AllBlocksCleared matched = %d, want 0", res.MatchedBlocks)
	}
}

func TestHandlePayload_UnknownTagIgnored(t *testing.T) {
	idx := indexl1.New(logr.Discard())
	sub := &subscription{
		target:  VLLMSubscribeTarget{PodIP: "ip"},
		log:     logr.Discard(),
		indexer: idx,
		seenCtx: map[apis.ModelContext]struct{}{},
	}
	bogus := []interface{}{"FutureEventV2", []interface{}{uint64(1)}}
	if err := sub.handlePayload(encodeKVEventBatch(t, 1, []interface{}{bogus}, 0)); err != nil {
		t.Errorf("unknown tag should not error, got %v", err)
	}
}

func TestHandlePayload_MalformedBatchReturnsError(t *testing.T) {
	sub := &subscription{
		target:  VLLMSubscribeTarget{PodIP: "ip"},
		log:     logr.Discard(),
		indexer: indexl1.New(logr.Discard()),
		seenCtx: map[apis.ModelContext]struct{}{},
	}
	if err := sub.handlePayload([]byte{0xff, 0xff, 0xff}); err == nil {
		t.Error("expected error on garbage payload")
	}
}

//mapBlockStored handles map-encoded BlockStored events(vLLM >= 0.24.0)
func mapBlockStored(hashes []interface{}, blockSize int64, loraName string) map[string]interface{} {
	return map[string]interface{}{
		"type":               "BlockStored",
		"block_hashes":       hashes,
		"parent_block_hash":  nil,
		"token_ids":          []interface{}{int64(1), int64(2), int64(3)},
		"block_size":         blockSize,
		"lora_id":            nil,
		"medium":             testSubGPU,
		"lora_name":          loraName,
		"extra_keys":         nil,
		"group_idx":          int64(0),
		"kv_cache_spec_kind": "full_attention",
	}
}

func TestMapEventToArray(t *testing.T) {
	arr := mapEventToArray(mapBlockStored(
		[]interface{}{uint64(0xaa)}, 128, "lora-a"))
	if len(arr) != blockStoredGroupIndexIndex+1 {
		t.Fatalf("len=%d want %d", len(arr), blockStoredGroupIndexIndex+1)
	}
	if arr[taggedEventTagIndex] != tagBlockStored {
		t.Errorf("tag=%v want %q", arr[taggedEventTagIndex], tagBlockStored)
	}
	if got, ok := arr[blockStoredHashesIndex].([]interface{}); !ok || len(got) != 1 {
		t.Errorf("block_hashes=%v", arr[blockStoredHashesIndex])
	}
	if arr[blockStoredMediumIndex] != testSubGPU {
		t.Errorf("medium=%v want %q", arr[blockStoredMediumIndex], testSubGPU)
	}

	cleared := mapEventToArray(map[string]interface{}{"type": "AllBlocksCleared"})
	if len(cleared) != 1 || cleared[taggedEventTagIndex] != tagAllBlocksCleared {
		t.Errorf("cleared=%v", cleared)
	}

	unknown := mapEventToArray(map[string]interface{}{"type": "FutureEventV2"})
	if len(unknown) != 1 || unknown[taggedEventTagIndex] != "FutureEventV2" {
		t.Errorf("unknown=%v", unknown)
	}
}

func TestHandlePayload_MapEncodedBlockStored_ThenMatchPrefix(t *testing.T) {
	idx := indexl1.New(logr.Discard())
	sub := &subscription{
		target:  VLLMSubscribeTarget{PodIP: "10.0.0.1", ZMQPort: 5557},
		log:     logr.Discard(),
		indexer: idx,
		seenCtx: map[apis.ModelContext]struct{}{},
	}

	stored := mapBlockStored(
		[]interface{}{uint64(0xaa), uint64(0xbb), uint64(0xcc)}, 128, "lora-a")
	payload := encodeKVEventBatch(t, 1234.5, []interface{}{stored}, 0)
	if err := sub.handlePayload(payload); err != nil {
		t.Fatalf("handlePayload: %v", err)
	}

	mc := apis.ModelContext{LoraName: "lora-a", BlockSize: 128}
	res, _ := idx.MatchPrefix(mc, "10.0.0.1",
		[]apis.BlockHash{intHex(0xaa), intHex(0xbb), intHex(0xcc), intHex(0xdd)})
	if res.MatchedBlocks != 3 {
		t.Errorf("matched = %d, want 3 (last is miss)", res.MatchedBlocks)
	}
}

func TestHandlePayload_MapEncodedBlockStoredFields(t *testing.T) {
	idx := &recordingIndexer{}
	sub := &subscription{
		target:  VLLMSubscribeTarget{PodIP: testSubPodIP, ZMQPort: testSubPort},
		log:     logr.Discard(),
		indexer: idx,
		seenCtx: map[apis.ModelContext]struct{}{},
	}

	stored := mapBlockStored(
		[]interface{}{uint64(1), uint64(2)}, 128, "lora-a")
	if err := sub.handlePayload(encodeKVEventBatch(t, 1, []interface{}{stored}, 3)); err != nil {
		t.Fatal(err)
	}

	if len(idx.upserts) != 1 {
		t.Fatalf("upserts=%d want 1", len(idx.upserts))
	}
	ev := idx.upserts[0]
	if ev.ServerIP != testSubPodIP || ev.Medium != testSubGPU {
		t.Errorf("serverIP=%q medium=%q", ev.ServerIP, ev.Medium)
	}
	if ev.BlockSize != 128 || ev.DPRank != 3 {
		t.Errorf("blockSize=%d dpRank=%d", ev.BlockSize, ev.DPRank)
	}
	if len(ev.BlockHashes) != 2 || len(ev.TokenIDs) != 3 {
		t.Errorf("hashes=%d tokenIDs=%d", len(ev.BlockHashes), len(ev.TokenIDs))
	}
	if ev.ParentBlockHash != nil {
		t.Errorf("parentBlockHash=%v want nil", ev.ParentBlockHash)
	}
	if ev.GroupIndex == nil || *ev.GroupIndex != 0 {
		t.Errorf("groupIndex=%v want 0", ev.GroupIndex)
	}
	if ev.ModelContext.LoraName != "lora-a" || ev.ModelContext.BlockSize != 128 {
		t.Errorf("modelContext=%+v", ev.ModelContext)
	}
}

func TestHandlePayload_MapEncodedBlockRemoved_AfterStored(t *testing.T) {
	idx := indexl1.New(logr.Discard())
	sub := &subscription{
		target:  VLLMSubscribeTarget{PodIP: "10.0.0.1"},
		log:     logr.Discard(),
		indexer: idx,
		seenCtx: map[apis.ModelContext]struct{}{},
	}
	stored := mapBlockStored([]interface{}{uint64(1), uint64(2)}, 16, "lora-a")
	if err := sub.handlePayload(encodeKVEventBatch(t, 1, []interface{}{stored}, 0)); err != nil {
		t.Fatal(err)
	}

	removed := map[string]interface{}{
		"type":          "BlockRemoved",
		"block_hashes":  []interface{}{uint64(1)},
		"medium":        testSubGPU,
		"group_idx":     int64(0),
	}
	if err := sub.handlePayload(encodeKVEventBatch(t, 2, []interface{}{removed}, 0)); err != nil {
		t.Fatal(err)
	}

	mc := apis.ModelContext{LoraName: "lora-a", BlockSize: 16}
	res, _ := idx.MatchPrefix(mc, "10.0.0.1", []apis.BlockHash{intHex(1)})
	if res.MatchedBlocks != 0 {
		t.Errorf("after BlockRemoved, matched = %d, want 0", res.MatchedBlocks)
	}
	res, _ = idx.MatchPrefix(mc, "10.0.0.1", []apis.BlockHash{intHex(2)})
	if res.MatchedBlocks != 1 {
		t.Errorf("block 2 still present, matched = %d, want 1", res.MatchedBlocks)
	}
}

func TestHandlePayload_MapEncodedAllBlocksCleared(t *testing.T) {
	idx := indexl1.New(logr.Discard())
	sub := &subscription{
		target:  VLLMSubscribeTarget{PodIP: "10.0.0.1"},
		log:     logr.Discard(),
		indexer: idx,
		seenCtx: map[apis.ModelContext]struct{}{},
	}
	stored := mapBlockStored([]interface{}{uint64(1), uint64(2)}, 16, "lora-a")
	if err := sub.handlePayload(encodeKVEventBatch(t, 1, []interface{}{stored}, 0)); err != nil {
		t.Fatal(err)
	}
	cleared := map[string]interface{}{"type": "AllBlocksCleared"}
	if err := sub.handlePayload(encodeKVEventBatch(t, 2, []interface{}{cleared}, 0)); err != nil {
		t.Fatal(err)
	}

	mc := apis.ModelContext{LoraName: "lora-a", BlockSize: 16}
	res, _ := idx.MatchPrefix(mc, "10.0.0.1", []apis.BlockHash{intHex(1), intHex(2)})
	if res.MatchedBlocks != 0 {
		t.Errorf("after AllBlocksCleared matched = %d, want 0", res.MatchedBlocks)
	}
}

func TestHandlePayload_MixedArrayAndMapEvents(t *testing.T) {
	idx := &recordingIndexer{}
	sub := &subscription{
		target:  VLLMSubscribeTarget{PodIP: testSubPodIP},
		log:     logr.Discard(),
		indexer: idx,
		seenCtx: map[apis.ModelContext]struct{}{},
	}
	arrEvent := []interface{}{tagBlockStored, []interface{}{uint64(1)},
		nil, nil, int64(16), nil, testSubGPU, "lora-a", nil, int64(0)}
	mapEvent := mapBlockStored([]interface{}{uint64(2)}, 16, "lora-b")
	// Malformed entries must not block the legal events after them.
	events := []interface{}{
		"not-a-kv-event",
		map[string]interface{}{"type": "FutureEventV2"},
		arrEvent,
		mapEvent,
	}
	if err := sub.handlePayload(encodeKVEventBatch(t, 1, events, 0)); err != nil {
		t.Fatal(err)
	}
	if len(idx.upserts) != 2 {
		t.Fatalf("upserts=%d want 2 (array + map)", len(idx.upserts))
	}
	loras := map[string]bool{}
	for _, ev := range idx.upserts {
		loras[ev.ModelContext.LoraName] = true
	}
	if !loras["lora-a"] || !loras["lora-b"] {
		t.Errorf("missing loras: %+v", loras)
	}
}

func TestHandlePayload_MapEventMissingOptionalKeys(t *testing.T) {
	idx := &recordingIndexer{}
	sub := &subscription{
		target:  VLLMSubscribeTarget{PodIP: testSubPodIP},
		log:     logr.Discard(),
		indexer: idx,
		seenCtx: map[apis.ModelContext]struct{}{},
	}
	stored := map[string]interface{}{
		"type":         tagBlockStored,
		"block_hashes": []interface{}{uint64(1)},
		"token_ids":    []interface{}{int64(7)},
		"block_size":   int64(32),
		"medium":       testSubGPU,
	}
	if err := sub.handlePayload(encodeKVEventBatch(t, 1, []interface{}{stored}, 0)); err != nil {
		t.Fatal(err)
	}
	if len(idx.upserts) != 1 {
		t.Fatalf("upserts=%d want 1", len(idx.upserts))
	}
	ev := idx.upserts[0]
	if ev.BlockSize != 32 || ev.Medium != testSubGPU {
		t.Errorf("blockSize=%d medium=%q", ev.BlockSize, ev.Medium)
	}
	if ev.ParentBlockHash != nil || ev.GroupIndex != nil {
		t.Errorf("parent=%v group=%v want nil", ev.ParentBlockHash, ev.GroupIndex)
	}
	if ev.ModelContext.LoraName != "" || ev.ModelContext.BlockSize != 32 {
		t.Errorf("modelContext=%+v", ev.ModelContext)
	}
}

type recordingIndexer struct {
	upserts []*apis.L1StoredEvent
	removes []*apis.L1RemovedEvent
	clears  []string
}

func (r *recordingIndexer) UpsertStoredEvent(ev *apis.L1StoredEvent) error {
	r.upserts = append(r.upserts, ev)
	return nil
}

func (r *recordingIndexer) RemoveBlocks(ev *apis.L1RemovedEvent) error {
	r.removes = append(r.removes, ev)
	return nil
}

func (r *recordingIndexer) ClearServer(_ apis.ModelContext, serverIP string) error {
	r.clears = append(r.clears, serverIP)
	return nil
}

func (r *recordingIndexer) MatchPrefix(apis.ModelContext, string, []apis.BlockHash) (*apis.MatchResult, error) {
	return nil, nil
}

func TestNormalizeBlockHash_StringPath(t *testing.T) {
	hash, ok := normalizeBlockHash(testStringHashInput)
	if !ok || string(hash) != testStringHashHex {
		t.Fatalf("hash=%q ok=%v want=%q", hash, ok, testStringHashHex)
	}
}

func TestManager_SubscribeInvalidTarget(t *testing.T) {
	mgr := NewWithConfig(logr.Discard(), &recordingIndexer{}, config.IngestL1Config{}).(*manager)
	err := mgr.Subscribe(VLLMSubscribeTarget{})
	if err == nil {
		t.Fatal("expected invalid target error")
	}
}

func TestManager_SubscribeDeduplicatesAndUnsubscribeClearsIndex(t *testing.T) {
	idx := &recordingIndexer{}
	mgr := NewWithConfig(logr.Discard(), idx, config.IngestL1Config{
		BackoffInitial: testBackoffDuration,
		BackoffMax:     testBackoffDuration,
	}).(*manager)

	target := VLLMSubscribeTarget{PodIP: testSubPodIP, ZMQPort: testSubPort}
	if err := mgr.Subscribe(target); err != nil {
		t.Fatalf("Subscribe: %v", err)
	}
	if err := mgr.Subscribe(target); err != nil {
		t.Fatalf("Subscribe dedupe: %v", err)
	}
	if got := len(mgr.subs); got != 1 {
		t.Fatalf("subs=%d want 1", got)
	}

	sub := mgr.subs[testSubPodIP]
	sub.recordContext(apis.ModelContext{LoraName: testSubLora, BlockSize: 16})
	if err := mgr.Unsubscribe(testSubPodIP); err != nil {
		t.Fatalf("Unsubscribe: %v", err)
	}
	if len(idx.clears) != 1 || idx.clears[0] != testSubPodIP {
		t.Fatalf("clears=%v", idx.clears)
	}
	if err := mgr.Unsubscribe(testSubPodIP); err != nil {
		t.Fatalf("Unsubscribe missing: %v", err)
	}
}

func TestSubscription_HandleBlockRemovedAcrossSeenContexts(t *testing.T) {
	idx := &recordingIndexer{}
	sub := &subscription{
		target:  VLLMSubscribeTarget{PodIP: testSubPodIP},
		log:     logr.Discard(),
		indexer: idx,
		seenCtx: map[apis.ModelContext]struct{}{
			{LoraName: "a", BlockSize: 16}: {},
			{LoraName: "b", BlockSize: 32}: {},
		},
	}
	sub.handleBlockRemoved([]interface{}{tagBlockRemoved, []interface{}{uint64(1)}, testSubGPU, int64(3)}, testTSMillis)
	if len(idx.removes) != 2 {
		t.Fatalf("removes=%d want 2", len(idx.removes))
	}
}

func TestNormalizeConfigAndHelpers(t *testing.T) {
	cfg := normalizeConfig(config.IngestL1Config{})
	if cfg.BackoffInitial <= 0 || cfg.BackoffMax < cfg.BackoffInitial {
		t.Fatalf("cfg=%+v", cfg)
	}
	if safeIdx([]interface{}{1}, 2) != nil {
		t.Fatal("safeIdx out of range should return nil")
	}
	if got := tsFloatToMillis(float64(testTSSeconds)); got != testTSMillis {
		t.Fatalf("tsFloatToMillis=%d want=%d", got, testTSMillis)
	}
	if got, ok := toInt64(float32(testTSSeconds)); !ok || got != testTSSeconds {
		t.Fatalf("toInt64=%d ok=%v", got, ok)
	}
	if _, ok := integerToUint64(errors.New("x")); ok {
		t.Fatal("integerToUint64 on non-integer should fail")
	}
}

func TestWaitBackoffAndStopTimer(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	timer := time.NewTimer(testBackoffDuration)
	cancel()
	if waitBackoff(ctx, timer, testBackoffDuration) {
		t.Fatal("waitBackoff should stop on cancelled ctx")
	}
	stopTimer(timer)
}

func TestManager_StopClearsSubscriptions(t *testing.T) {
	mgr := NewWithConfig(logr.Discard(), &recordingIndexer{}, config.IngestL1Config{
		BackoffInitial: testBackoffDuration,
		BackoffMax:     testBackoffDuration,
	}).(*manager)
	mgr.subs[testSubPodIP] = &subscription{
		cancel: func() {},
		done:   closedDoneChan(),
	}
	mgr.Stop()
	if len(mgr.subs) != 0 {
		t.Fatalf("subs=%d want 0", len(mgr.subs))
	}
}

func closedDoneChan() chan struct{} {
	ch := make(chan struct{})
	close(ch)
	return ch
}

func TestSubscription_RunOnceShortFrames(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	pub := zmq4.NewPub(ctx)
	defer pub.Close()
	if err := pub.Listen(testZMQLocalEndpoint); err != nil {
		t.Fatalf("Listen: %v", err)
	}
	addr := pub.Addr()
	if addr == nil {
		t.Fatal("pub.Addr is nil")
	}

	sub := &subscription{
		target: VLLMSubscribeTarget{
			PodIP:   testLocalhostIP,
			ZMQPort: int32(addr.(*net.TCPAddr).Port),
		},
		cfg:     normalizeConfig(config.IngestL1Config{BackoffInitial: testBackoffDuration, BackoffMax: testBackoffDuration}),
		log:     logr.Discard(),
		indexer: &recordingIndexer{},
		seenCtx: map[apis.ModelContext]struct{}{},
	}

	done := make(chan error, 1)
	go func() {
		done <- sub.runOnce(ctx)
	}()

	time.Sleep(testRecvDelay)
	if err := pub.Send(zmq4.NewMsgString("topic")); err != nil {
		t.Fatalf("Send: %v", err)
	}
	time.Sleep(testRecvDelay)
	cancel()

	err := <-done
	if err == nil {
		t.Fatal("expected ctx cancellation error")
	}
}
