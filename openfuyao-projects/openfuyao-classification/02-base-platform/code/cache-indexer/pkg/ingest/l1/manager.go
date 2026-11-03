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

// Package l1 owns per-Pod ZMQ SUB connections for vLLM KVEventBatch.
// Decoded events are dispatched to l1.Indexer.
package l1

import (
	"context"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"reflect"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-logr/logr"
	"github.com/go-zeromq/zmq4"
	"github.com/vmihailenco/msgpack/v5"

	"gitcode.com/openFuyao/cache-indexer/pkg/apis"
	"gitcode.com/openFuyao/cache-indexer/pkg/config"
	indexl1 "gitcode.com/openFuyao/cache-indexer/pkg/index/l1"
)

// VLLMSubscribeTarget is the minimal set of fields the L1 subscriber
// needs from a discovered vLLM Pod.
type VLLMSubscribeTarget struct {
	PodIP   string
	ZMQPort int32
	PodName string
}

// SubscriptionManager owns ZMQ SUB lifecycles. Discovery drives it via
// Subscribe and Unsubscribe.
type SubscriptionManager interface {
	// Subscribe starts consuming KV events for one discovered vLLM Pod.
	Subscribe(target VLLMSubscribeTarget) error
	// Unsubscribe stops consuming KV events for one vLLM Pod IP.
	Unsubscribe(podIP string) error
	// Stop shuts down all active subscriptions.
	Stop()
}

const (
	// vLLM sends multipart ZMQ frames as [topic, seq_be8, msgpack(KVEventBatch)].
	_ = iota
	_
	payloadFrameIndex
	expectedZMQFrameCount
)

const (
	// KVEventBatch msgspec array fields: [ts, events, data_parallel_rank?].
	eventBatchTimestampIndex = iota
	eventBatchEventsIndex
	eventBatchDPRankIndex
)

const (
	eventBatchMinLen    = eventBatchEventsIndex + 1
	eventBatchDPRankLen = eventBatchDPRankIndex + 1
)

const (
	// msgspec tag=True stores the event tag in slot 0 for every KV event.
	// BlockStored fields: [tag, block_hashes, parent_block_hash, token_ids,
	// block_size, lora_id, medium, lora_name, extra_keys, group_idx].
	taggedEventTagIndex = iota
	blockStoredHashesIndex
	blockStoredParentIndex
	blockStoredTokenIDsIndex
	blockStoredBlockSizeIndex
	_
	blockStoredMediumIndex
	blockStoredLoraNameIndex
	_
	blockStoredGroupIndexIndex
)

const (
	// BlockRemoved fields: [tag, block_hashes, medium, group_idx].
	_ = iota
	blockRemovedHashesIndex
	blockRemovedMediumIndex
	blockRemovedGroupIndexIndex
)

const (
	uint64Bytes           = 8
	millisecondsPerSecond = int64(time.Second / time.Millisecond)
	// ZMQ event tag strings produced by msgspec-encoded tagged union.
	tagBlockStored      = "BlockStored"
	tagBlockRemoved     = "BlockRemoved"
	tagAllBlocksCleared = "AllBlocksCleared"
	// Map-encoded events carry the event tag under the "type" key.
	mapEventTagKey = "type"
)

// mapEventKeys lists the map keys of each event type in tagged-array slot order.
var mapEventKeys = map[string][]string{
	tagBlockStored: {
		"block_hashes",
		"parent_block_hash",
		"token_ids",
		"block_size",
		"lora_id",
		"medium",
		"lora_name",
		"extra_keys",
		"group_idx",
	},
	tagBlockRemoved: {
		"block_hashes",
		"medium",
		"group_idx",
	},
	tagAllBlocksCleared: nil,
}

// mapEventToArray reshapes a map-encoded KV event into the tagged-array form.
func mapEventToArray(m map[string]interface{}) []interface{} {
	tag, _ := m[mapEventTagKey].(string)
	keys := mapEventKeys[tag]
	arr := make([]interface{}, len(keys)+1)
	arr[taggedEventTagIndex] = tag
	for i, key := range keys {
		arr[i+1] = m[key]
	}
	return arr
}

// normalizeKVEvent unifies array(vllm-ascend<0.24.0) and map(>=0.24.0) KV events into tagged-array form.
func normalizeKVEvent(evRaw interface{}) ([]interface{}, bool) {
	switch ev := evRaw.(type) {
	case []interface{}:
		return ev, len(ev) > 0
	case map[string]interface{}:
		arr := mapEventToArray(ev)
		return arr, len(arr) > 0
	default:
		return nil, false
	}
}

type subscription struct {
	target  VLLMSubscribeTarget
	cfg     config.IngestL1Config
	cancel  context.CancelFunc
	done    chan struct{}
	log     logr.Logger
	indexer indexl1.Indexer

	mu      sync.Mutex
	seenCtx map[apis.ModelContext]struct{}

	// Stats / one-shot log gates (atomic so handlePayload can update
	// them without holding mu).
	firstFrameLogged atomic.Bool
	batchesRecv      atomic.Int64
	storedBlocks     atomic.Int64
	removedBlocks    atomic.Int64
}

type manager struct {
	log     logr.Logger
	cfg     config.IngestL1Config
	indexer indexl1.Indexer

	mu   sync.Mutex
	subs map[string]*subscription
}

// New builds a real L1 ZMQ subscription manager.
func New(log logr.Logger, indexer indexl1.Indexer) SubscriptionManager {
	return NewWithConfig(log, indexer, config.IngestL1Config{})
}

// NewWithConfig builds a real L1 ZMQ subscription manager with explicit settings.
func NewWithConfig(log logr.Logger, indexer indexl1.Indexer, cfg config.IngestL1Config) SubscriptionManager {
	cfg = normalizeConfig(cfg)
	return &manager{
		log:     log,
		cfg:     cfg,
		indexer: indexer,
		subs:    map[string]*subscription{},
	}
}

// Subscribe starts a per-Pod SUB goroutine. Idempotent: a re-subscribe
// for the same PodIP is a no-op if already active.
func (m *manager) Subscribe(t VLLMSubscribeTarget) error {
	if t.PodIP == "" || t.ZMQPort <= 0 {
		return fmt.Errorf("l1 subscribe: invalid target pod_ip=%q zmq_port=%d", t.PodIP, t.ZMQPort)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.subs[t.PodIP]; ok {
		return nil
	}
	ctx, cancel := context.WithCancel(context.Background())
	sub := &subscription{
		target:  t,
		cfg:     m.cfg,
		cancel:  cancel,
		done:    make(chan struct{}),
		log:     m.log.WithValues("podIP", t.PodIP, "zmqPort", t.ZMQPort),
		indexer: m.indexer,
		seenCtx: map[apis.ModelContext]struct{}{},
	}
	m.subs[t.PodIP] = sub
	go sub.run(ctx)
	m.log.Info("L1 subscribe started", "podIP", t.PodIP, "zmqPort", t.ZMQPort)
	return nil
}

// Unsubscribe stops the Pod SUB loop and clears its observed ModelContexts.
// This prevents stale L1 hits for offline servers.
func (m *manager) Unsubscribe(podIP string) error {
	m.mu.Lock()
	sub, ok := m.subs[podIP]
	if ok {
		delete(m.subs, podIP)
	}
	m.mu.Unlock()
	if !ok {
		return nil
	}
	sub.cancel()
	<-sub.done
	sub.clearIndexForServer()
	m.log.Info("L1 unsubscribe completed", "podIP", podIP)
	return nil
}

// Stop tears down all active subscriptions; used during graceful shutdown.
func (m *manager) Stop() {
	m.mu.Lock()
	subs := make([]*subscription, 0, len(m.subs))
	for _, s := range m.subs {
		subs = append(subs, s)
	}
	m.subs = map[string]*subscription{}
	m.mu.Unlock()
	for _, s := range subs {
		s.cancel()
		<-s.done
	}
}

// run is the per-Pod connection / reconnect loop with exponential backoff.
func (s *subscription) run(ctx context.Context) {
	defer close(s.done)
	cfg := s.cfg
	backoff := cfg.BackoffInitial
	timer := time.NewTimer(cfg.BackoffInitial)
	stopTimer(timer)
	defer timer.Stop()
	for {
		if ctx.Err() != nil {
			return
		}
		err := s.runOnce(ctx)
		if ctx.Err() != nil {
			return
		}
		if err != nil {
			s.log.Error(err, "L1 SUB session terminated; retrying after backoff",
				"backoff", backoff.String())
		}
		if !waitBackoff(ctx, timer, backoff) {
			return
		}
		if backoff < cfg.BackoffMax {
			backoff *= 2
			if backoff > cfg.BackoffMax {
				backoff = cfg.BackoffMax
			}
		}
	}
}

func normalizeConfig(cfg config.IngestL1Config) config.IngestL1Config {
	if cfg.BackoffInitial <= 0 {
		cfg.BackoffInitial = config.DefaultIngestL1BackoffInitial
	}
	if cfg.BackoffMax <= 0 {
		cfg.BackoffMax = config.DefaultIngestL1BackoffMax
	}
	if cfg.BackoffMax < cfg.BackoffInitial {
		cfg.BackoffMax = cfg.BackoffInitial
	}
	return cfg
}

func waitBackoff(ctx context.Context, timer *time.Timer, backoff time.Duration) bool {
	stopTimer(timer)
	timer.Reset(backoff)
	select {
	case <-ctx.Done():
		stopTimer(timer)
		return false
	case <-timer.C:
		return true
	}
}

func stopTimer(timer *time.Timer) {
	if timer.Stop() {
		return
	}
	select {
	case <-timer.C:
	default:
	}
}

// runOnce dials, subscribes, and pumps frames until ctx ends or the
// socket fails.
func (s *subscription) runOnce(ctx context.Context) error {
	sock := zmq4.NewSub(ctx)
	defer sock.Close()

	addr := fmt.Sprintf("tcp://%s:%d", s.target.PodIP, s.target.ZMQPort)
	if err := sock.Dial(addr); err != nil {
		return fmt.Errorf("dial %s: %w", addr, err)
	}
	// Empty topic subscribes to everything; vLLM default topic is "".
	if err := sock.SetOption(zmq4.OptionSubscribe, ""); err != nil {
		return fmt.Errorf("subscribe: %w", err)
	}
	s.log.Info("L1 ZMQ SUB connected", "addr", addr)

	for {
		msg, err := sock.Recv()
		if err != nil {
			// If parent ctx was cancelled, surface that cause.
			// The run loop uses ctx.Err() to exit cleanly.
			if ctx.Err() != nil {
				return fmt.Errorf("ctx cancelled: %w", ctx.Err())
			}
			return fmt.Errorf("recv: %w", err)
		}
		if s.firstFrameLogged.CompareAndSwap(false, true) {
			s.log.V(1).Info("L1 first ZMQ frame received",
				"addr", addr, "frames", len(msg.Frames))
		}
		// vLLM publisher sends 3 frames: topic, seq(big-endian uint64),
		// msgpack(KVEventBatch). Anything shorter is malformed; skip.
		if len(msg.Frames) < expectedZMQFrameCount {
			s.log.V(1).Info("L1 skipped short multipart", "frames", len(msg.Frames))
			continue
		}
		if err := s.handlePayload(msg.Frames[payloadFrameIndex]); err != nil {
			s.log.Error(err, "L1 decode/dispatch error; skipping batch",
				"payloadBytes", len(msg.Frames[payloadFrameIndex]))
		}
	}
}

// handlePayload decodes the msgpack KVEventBatch and dispatches each normalized event by tag.
func (s *subscription) handlePayload(payload []byte) error {
	var batch []interface{}
	if err := msgpack.Unmarshal(payload, &batch); err != nil {
		return fmt.Errorf("msgpack decode batch: %w", err)
	}
	if len(batch) < eventBatchMinLen {
		return fmt.Errorf("event batch too short: len=%d", len(batch))
	}
	eventsRaw, ok := batch[eventBatchEventsIndex].([]interface{})
	if !ok {
		return fmt.Errorf("batch.events is not an array (got %T)", batch[eventBatchEventsIndex])
	}
	var dpRank int64
	if len(batch) >= eventBatchDPRankLen {
		dpRank, _ = toInt64(batch[eventBatchDPRankIndex])
	}
	tsMs := tsFloatToMillis(batch[eventBatchTimestampIndex])

	batchSeq := s.batchesRecv.Add(1)
	s.log.V(1).Info("L1 batch decoded",
		"events", len(eventsRaw), "dpRank", dpRank, "tsMs", tsMs, "batchSeq", batchSeq)

	for _, evRaw := range eventsRaw {
		evArr, ok := normalizeKVEvent(evRaw)
		if !ok {
			s.log.V(1).Info("L1 skipped unsupported event encoding (ignored)",
				"goType", fmt.Sprintf("%T", evRaw))
			continue
		}
		tag, _ := evArr[taggedEventTagIndex].(string)
		switch tag {
		case tagBlockStored:
			s.handleBlockStored(evArr, dpRank, tsMs)
		case tagBlockRemoved:
			s.handleBlockRemoved(evArr, tsMs)
		case tagAllBlocksCleared:
			s.handleAllBlocksCleared()
		default:
			s.log.V(1).Info("L1 unknown event tag (ignored)", "tag", tag)
		}
	}
	return nil
}

// handleBlockStored normalizes a BlockStored event for the indexer.
// Guarded access handles msgspec.omit_defaults truncation.
func (s *subscription) handleBlockStored(arr []interface{}, dpRank, tsMs int64) {
	blockHashes := normalizeBlockHashList(safeIdx(arr, blockStoredHashesIndex))
	if len(blockHashes) == 0 {
		return
	}
	var parent *apis.BlockHash
	if h, ok := normalizeBlockHash(safeIdx(arr, blockStoredParentIndex)); ok {
		parent = &h
	}
	tokenIDs := toInt32Slice(safeIdx(arr, blockStoredTokenIDsIndex))
	blockSize, _ := toInt64(safeIdx(arr, blockStoredBlockSizeIndex))
	medium, _ := safeIdx(arr, blockStoredMediumIndex).(string)
	loraName, _ := safeIdx(arr, blockStoredLoraNameIndex).(string)
	var groupIdx *int64
	if g, ok := toInt64(safeIdx(arr, blockStoredGroupIndexIndex)); ok {
		groupIdx = &g
	}

	mc := eventModelContext(loraName, blockSize)
	s.recordContext(mc)

	ev := &apis.L1StoredEvent{
		ModelContext:     mc,
		ServerIP:         s.target.PodIP,
		BlockHashes:      blockHashes,
		ParentBlockHash:  parent,
		TokenIDs:         tokenIDs,
		BlockSize:        blockSize,
		Medium:           medium,
		DPRank:           dpRank,
		GroupIndex:       groupIdx,
		EventTimestampMs: tsMs,
	}
	if err := s.indexer.UpsertStoredEvent(ev); err != nil {
		s.log.Error(err, "L1 UpsertStoredEvent failed")
		return
	}
	total := s.storedBlocks.Add(int64(len(blockHashes)))
	s.log.V(1).Info("L1 BlockStored ingested",
		"blocks", len(blockHashes), "blockSize", blockSize, "medium", medium,
		"loraName", loraName, "dpRank", dpRank, "cumBlocks", total)
}

func (s *subscription) handleBlockRemoved(arr []interface{}, tsMs int64) {
	blockHashes := normalizeBlockHashList(safeIdx(arr, blockRemovedHashesIndex))
	if len(blockHashes) == 0 {
		return
	}
	medium, _ := safeIdx(arr, blockRemovedMediumIndex).(string)
	var groupIdx *int64
	if g, ok := toInt64(safeIdx(arr, blockRemovedGroupIndexIndex)); ok {
		groupIdx = &g
	}

	// BlockRemoved does not carry block_size / lora; we use a wildcard
	// removal across every ModelContext we have observed on this Pod.
	for mc := range s.snapshotContexts() {
		ev := &apis.L1RemovedEvent{
			ModelContext:     mc,
			ServerIP:         s.target.PodIP,
			BlockHashes:      blockHashes,
			Medium:           medium,
			GroupIndex:       groupIdx,
			EventTimestampMs: tsMs,
		}
		if err := s.indexer.RemoveBlocks(ev); err != nil {
			s.log.Error(err, "L1 RemoveBlocks failed")
		}
	}
	total := s.removedBlocks.Add(int64(len(blockHashes)))
	s.log.V(1).Info("L1 BlockRemoved ingested",
		"blocks", len(blockHashes), "medium", medium, "cumBlocks", total)
}

func (s *subscription) handleAllBlocksCleared() {
	for mc := range s.snapshotContexts() {
		if err := s.indexer.ClearServer(mc, s.target.PodIP); err != nil {
			s.log.Error(err, "L1 ClearServer failed")
		}
	}
	s.log.V(1).Info("L1 AllBlocksCleared applied")
}

// clearIndexForServer removes every L1 entry written by this Pod.
// It is called during Unsubscribe.
func (s *subscription) clearIndexForServer() {
	for mc := range s.snapshotContexts() {
		if err := s.indexer.ClearServer(mc, s.target.PodIP); err != nil {
			s.log.Error(err, "L1 ClearServer (unsubscribe) failed")
		}
	}
}

func (s *subscription) recordContext(mc apis.ModelContext) {
	s.mu.Lock()
	s.seenCtx[mc] = struct{}{}
	s.mu.Unlock()
}

func (s *subscription) snapshotContexts() map[apis.ModelContext]struct{} {
	s.mu.Lock()
	out := make(map[apis.ModelContext]struct{}, len(s.seenCtx))
	for k := range s.seenCtx {
		out[k] = struct{}{}
	}
	s.mu.Unlock()
	return out
}

// eventModelContext builds the ModelContext used for L1 partitioning.
// BlockStored currently carries only lora_name and block_size.
func eventModelContext(loraName string, blockSize int64) apis.ModelContext {
	return apis.ModelContext{
		LoraName:  loraName,
		BlockSize: blockSize,
	}
}

// safeIdx returns nil instead of panicking when i is out of range.
func safeIdx(arr []interface{}, i int) interface{} {
	if i < 0 || i >= len(arr) {
		return nil
	}
	return arr[i]
}

// normalizeBlockHashList normalizes the wire-form list of ExternalBlockHash
// entries (which can be int / uint / bytes / string) into hex BlockHash.
func normalizeBlockHashList(v interface{}) []apis.BlockHash {
	arr, ok := v.([]interface{})
	if !ok {
		return nil
	}
	out := make([]apis.BlockHash, 0, len(arr))
	for _, x := range arr {
		if h, ok := normalizeBlockHash(x); ok {
			out = append(out, h)
		}
	}
	return out
}

// normalizeBlockHash converts bytes/strings to byte hex and ints to uint64 hex.
// nil/unsupported types return ok=false.
func normalizeBlockHash(v interface{}) (apis.BlockHash, bool) {
	switch x := v.(type) {
	case nil:
		return "", false
	case []byte:
		return apis.BlockHash(hex.EncodeToString(x)), true
	case string:
		// vLLM uses bytes when VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=False.
		// msgpack may decode short bin/str into Go string; keep raw bytes.
		return apis.BlockHash(hex.EncodeToString([]byte(x))), true
	case int8, int16, int32, int64, int, uint8, uint16, uint32, uint64, uint:
		u, _ := toUint64(x)
		var buf [uint64Bytes]byte
		binary.BigEndian.PutUint64(buf[:], u)
		return apis.BlockHash(hex.EncodeToString(buf[:])), true
	default:
		return "", false
	}
}

func toInt64(v interface{}) (int64, bool) {
	switch x := v.(type) {
	case float32:
		return int64(x), true
	case float64:
		return int64(x), true
	default:
		u, ok := integerToUint64(v)
		return int64(u), ok
	}
}

func toUint64(v interface{}) (uint64, bool) {
	return integerToUint64(v)
}

func integerToUint64(v interface{}) (uint64, bool) {
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return uint64(rv.Int()), true
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return rv.Uint(), true
	default:
		return 0, false
	}
}

func toInt32Slice(v interface{}) []int32 {
	arr, ok := v.([]interface{})
	if !ok {
		return nil
	}
	out := make([]int32, 0, len(arr))
	for _, x := range arr {
		if n, ok := toInt64(x); ok {
			out = append(out, int32(n))
		}
	}
	return out
}

func tsFloatToMillis(v interface{}) int64 {
	switch x := v.(type) {
	case float64:
		return int64(x * float64(millisecondsPerSecond))
	case float32:
		return int64(float64(x) * float64(millisecondsPerSecond))
	case int64:
		return x * millisecondsPerSecond
	default:
		return 0
	}
}
