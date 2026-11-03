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

// Package l3 polls Mooncake Master and converges the L3 index.
// UpdateTarget swaps the target; nil halts polling, Stop halts shutdown.
package l3

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/go-logr/logr"

	"gitcode.com/openFuyao/cache-indexer/pkg/apis"
	"gitcode.com/openFuyao/cache-indexer/pkg/config"
	indexl3 "gitcode.com/openFuyao/cache-indexer/pkg/index/l3"
)

// MooncakePollTarget is the minimal target set the L3 poller needs.
type MooncakePollTarget struct {
	PodIP    string
	HTTPPort int32
	RPCPort  int32 // unused by HTTP poller; kept for symmetry with discovery
	PodName  string
}

// MooncakePollTargetUpdater swaps the polling target atomically when the
// Mooncake Master Pod changes.
type MooncakePollTargetUpdater interface {
	UpdateTarget(target *MooncakePollTarget) error
	Stop()
}

// PollerConfig governs the HTTP poller behaviour. Zero values fall back
// to sane defaults via withDefaults().
type PollerConfig struct {
	Scheme       string        // http | https; default "http"
	PollInterval time.Duration // default 10s
	HTTPTimeout  time.Duration // default 5s
	// EmitInt64 mirrors VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES.
	// True keeps the last 16 hex chars so L3 matches vLLM ZMQ hashes.
	EmitInt64 bool
}

const (
	int64HashHexChars = 16
	hexNibbleShift    = 4
	hexLowNibbleMask  = 0x0f
)

func (c PollerConfig) withDefaults() PollerConfig {
	if c.Scheme == "" {
		c.Scheme = config.DefaultIngestL3Scheme
	}
	if c.PollInterval <= 0 {
		c.PollInterval = config.DefaultIngestL3PollInterval
	}
	if c.HTTPTimeout <= 0 {
		c.HTTPTimeout = config.DefaultIngestL3HTTPTimeout
	}
	return c
}

// httpDoer is the subset of *http.Client used by the poller; abstracted
// so unit tests can inject a stub without spinning a httptest.Server.
type httpDoer interface {
	Do(req *http.Request) (*http.Response, error)
}

type updater struct {
	log     logr.Logger
	cfg     PollerConfig
	indexer indexl3.Indexer
	client  httpDoer

	mu      sync.Mutex
	cancel  context.CancelFunc // current loop's cancel; nil if not running
	target  *MooncakePollTarget
	stopped bool

	// keyOwnership maps raw Mooncake keys to holder endpoints.
	// It supports diff-driven removals without re-querying Mooncake.
	keyOwnership map[string][]apis.TransportEndpoint
}

// New returns a real Mooncake HTTP poller. UpdateTarget must be called
// at least once with a non-nil target before any polling happens.
func New(log logr.Logger, indexer indexl3.Indexer) MooncakePollTargetUpdater {
	return NewWithConfig(log, indexer, PollerConfig{})
}

// NewWithConfig is the same as New but lets callers override defaults.
func NewWithConfig(log logr.Logger, indexer indexl3.Indexer, cfg PollerConfig) MooncakePollTargetUpdater {
	cfg = cfg.withDefaults()
	return &updater{
		log:          log,
		cfg:          cfg,
		indexer:      indexer,
		client:       &http.Client{Timeout: cfg.HTTPTimeout},
		keyOwnership: map[string][]apis.TransportEndpoint{},
	}
}

// UpdateTarget atomically halts any in-flight loop and starts a new one
// against `target`. A nil target halts polling without restarting.
func (u *updater) UpdateTarget(target *MooncakePollTarget) error {
	u.mu.Lock()
	defer u.mu.Unlock()
	if u.stopped {
		return errors.New("l3 poller already stopped")
	}
	// Halt previous loop if any.
	if u.cancel != nil {
		u.cancel()
		u.cancel = nil
	}
	u.target = target
	if target == nil {
		u.log.Info("L3 mooncake target cleared")
		return nil
	}
	if target.PodIP == "" || target.HTTPPort <= 0 {
		u.log.Info("L3 mooncake target invalid (PodIP/HTTPPort missing); not polling",
			"podIP", target.PodIP, "httpPort", target.HTTPPort)
		return nil
	}
	ctx, cancel := context.WithCancel(context.Background())
	u.cancel = cancel
	base := fmt.Sprintf("%s://%s:%d", u.cfg.Scheme, target.PodIP, target.HTTPPort)
	u.log.Info("L3 mooncake target updated", "base", base, "interval", u.cfg.PollInterval)
	go u.runLoop(ctx, base)
	return nil
}

// Stop halts any in-flight loop and prevents future restarts.
func (u *updater) Stop() {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.stopped = true
	if u.cancel != nil {
		u.cancel()
		u.cancel = nil
	}
}

func (u *updater) runLoop(ctx context.Context, base string) {
	// Previous raw all-keys set for diff.
	// UpdateTarget resets it by starting a fresh poll loop.
	prev := map[string]struct{}{}
	tick := time.NewTicker(u.cfg.PollInterval)
	defer tick.Stop()

	// Fire once immediately so the first reconcile doesn't wait one
	// interval after Pod start.
	prev = u.pollOnce(ctx, base, prev)
	for {
		select {
		case <-ctx.Done():
			u.log.V(1).Info("L3 poll loop exited", "base", base)
			return
		case <-tick.C:
			prev = u.pollOnce(ctx, base, prev)
		}
	}
}

// pollOnce executes one refresh cycle. `prev` is mutated in place to
// reflect the new all-keys set (carry-over for next diff).
func (u *updater) pollOnce(ctx context.Context, base string, prev map[string]struct{}) map[string]struct{} {
	allKeys, err := u.fetchAllKeys(ctx, base)
	if err != nil {
		// Don't poison `prev`; next round will retry from the same baseline.
		u.log.Error(err, "L3 /get_all_keys failed", "base", base)
		return prev
	}

	curSet := make(map[string]struct{}, len(allKeys))
	for _, k := range allKeys {
		curSet[k] = struct{}{}
	}
	if prev == nil {
		prev = make(map[string]struct{}, len(curSet))
	}
	var refreshSet, removedSet []string
	for k := range curSet {
		if _, ok := prev[k]; !ok {
			refreshSet = append(refreshSet, k)
		}
	}
	for k := range prev {
		if _, ok := curSet[k]; !ok {
			removedSet = append(removedSet, k)
		}
	}

	// Apply removals first so re-added keys (refresh path) win on conflict.
	if len(removedSet) > 0 {
		u.applyRemovals(removedSet)
	}
	if len(refreshSet) > 0 {
		u.applyRefresh(ctx, base, refreshSet)
	}

	// Carry over.
	for k := range prev {
		delete(prev, k)
	}
	for k := range curSet {
		prev[k] = struct{}{}
	}

	u.log.V(1).Info("L3 poller refresh",
		"base", base,
		"keys", len(curSet),
		"refresh", len(refreshSet),
		"removed", len(removedSet))
	return prev
}

// applyRemovals issues per-endpoint RemoveReplicaKeys based on the
// keyOwnership mirror built during the previous applyRefresh.
func (u *updater) applyRemovals(removed []string) {
	u.mu.Lock()
	byEP := map[apis.TransportEndpoint][]apis.BlockHash{}
	for _, raw := range removed {
		bh := u.mooncakeKeyToBlockHash(raw)
		if bh == "" {
			continue
		}
		for _, ep := range u.keyOwnership[raw] {
			byEP[ep] = append(byEP[ep], bh)
		}
		delete(u.keyOwnership, raw)
	}
	u.mu.Unlock()
	for ep, keys := range byEP {
		if err := u.indexer.RemoveReplicaKeys(apis.ModelContext{}, ep, keys); err != nil {
			u.log.Error(err, "L3 RemoveReplicaKeys failed", "endpoint", ep, "keys", len(keys))
		}
	}
}

// applyRefresh fetches /batch_query_keys, normalizes rows, writes them
// via ApplyReplicaSnapshot, and updates keyOwnership for the next diff.
func (u *updater) applyRefresh(ctx context.Context, base string, refresh []string) {
	rows, err := u.fetchBatchQueryKeys(ctx, base, refresh)
	if err != nil {
		u.log.Error(err, "L3 /batch_query_keys failed", "base", base, "keys", len(refresh))
		return
	}
	if len(rows) == 0 {
		return
	}
	byEP := map[apis.TransportEndpoint][]apis.L3ReplicaRecord{}
	u.mu.Lock()
	for _, r := range rows {
		byEP[r.TransportEndpoint] = append(byEP[r.TransportEndpoint], r)
		u.keyOwnership[r.RawMooncakeKey] = appendUniqueEP(u.keyOwnership[r.RawMooncakeKey], r.TransportEndpoint)
	}
	u.mu.Unlock()
	for ep, recs := range byEP {
		if err := u.indexer.ApplyReplicaSnapshot(apis.ModelContext{}, ep, recs); err != nil {
			u.log.Error(err, "L3 ApplyReplicaSnapshot failed", "endpoint", ep, "rows", len(recs))
			continue
		}
		// V(1): surface what L3 just stored so we can diff against
		// score's computed block keys when troubleshooting hash mismatches.
		first := ""
		if len(recs) > 0 {
			first = string(recs[0].BlockKey)
		}
		u.log.V(1).Info("L3 ApplyReplicaSnapshot",
			"endpoint", ep, "rows", len(recs), "firstBlockKey", first)
	}
}

func appendUniqueEP(s []apis.TransportEndpoint, ep apis.TransportEndpoint) []apis.TransportEndpoint {
	for _, e := range s {
		if e == ep {
			return s
		}
	}
	return append(s, ep)
}

// queryKeyDescriptor mirrors one Mooncake /query_key JSON row.
// The poller uses per-key /query_key for older masters.
type queryKeyDescriptor struct {
	TransportEndpoint string `json:"transport_endpoint_"`
	Protocol          string `json:"protocol_"`
}

func (u *updater) fetchAllKeys(ctx context.Context, base string) ([]string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, base+"/get_all_keys", nil)
	if err != nil {
		return nil, err
	}
	resp, err := u.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, readErr := io.ReadAll(io.LimitReader(resp.Body, 1024))
		if readErr != nil {
			return nil, fmt.Errorf("/get_all_keys: status=%d body-read: %w", resp.StatusCode, readErr)
		}
		return nil, fmt.Errorf("/get_all_keys: status=%d body=%s", resp.StatusCode, string(body))
	}
	// text/plain, one key per line.
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("/get_all_keys read: %w", err)
	}
	lines := strings.Split(string(body), "\n")
	out := make([]string, 0, len(lines))
	for _, l := range lines {
		l = strings.TrimSpace(l)
		if l == "" {
			continue
		}
		out = append(out, l)
	}
	return out, nil
}

func (u *updater) fetchBatchQueryKeys(ctx context.Context, base string, keys []string) ([]apis.L3ReplicaRecord, error) {
	if len(keys) == 0 {
		return nil, nil
	}
	rows := make([]apis.L3ReplicaRecord, 0, len(keys))
	for _, raw := range keys {
		// Bail early if the loop is being torn down (Stop / target swap).
		if err := ctx.Err(); err != nil {
			return rows, err
		}
		bh := u.mooncakeKeyToBlockHash(raw)
		if bh == "" {
			continue
		}
		descs, err := u.fetchQueryKey(ctx, base, raw)
		if err != nil {
			// One bad key shouldn't blow the whole refresh — log
			// and move on. The diff loop will retry next tick.
			u.log.V(1).Info("L3 /query_key failed", "key", raw, "error", err.Error())
			continue
		}
		for _, d := range descs {
			if d.TransportEndpoint == "" {
				// Without transport_endpoint_ we cannot index this row.
				continue
			}
			rows = append(rows, apis.L3ReplicaRecord{
				BlockKey:          bh,
				RawMooncakeKey:    raw,
				TransportEndpoint: apis.TransportEndpoint(d.TransportEndpoint),
			})
		}
	}
	return rows, nil
}

// fetchQueryKey reads /query_key for one raw key.
// Missing keys return (nil, nil) because evictions are expected during polling.
func (u *updater) fetchQueryKey(ctx context.Context, base, raw string) ([]queryKeyDescriptor, error) {
	// Cinatra does not percent-decode queries, so keep '@' and ':' literal.
	// Escape only bytes that would corrupt the query string.
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, base+"/query_key?key="+escapeMooncakeKeyForQuery(raw), nil)
	if err != nil {
		return nil, err
	}
	resp, err := u.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	// Some masters return 200 + OBJECT_NOT_FOUND for evicted keys.
	// Treat it like 404 and let the diff loop retry next tick.
	if resp.StatusCode == http.StatusNotFound {
		return nil, nil
	}
	if resp.StatusCode != http.StatusOK {
		b, readErr := io.ReadAll(io.LimitReader(resp.Body, 256))
		if readErr != nil {
			return nil, fmt.Errorf("status=%d body-read: %w", resp.StatusCode, readErr)
		}
		return nil, fmt.Errorf("status=%d body=%s", resp.StatusCode, string(b))
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read: %w", err)
	}
	// Cinatra-based master returns the literal text "OBJECT_NOT_FOUND"
	// (HTTP 200) instead of 404 for evicted/missing keys — treat as miss.
	if bytes.Equal(bytes.TrimSpace(body), []byte("OBJECT_NOT_FOUND")) {
		return nil, nil
	}
	out := make([]queryKeyDescriptor, 0, 2)
	for _, line := range strings.Split(string(body), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var d queryKeyDescriptor
		if err := json.Unmarshal([]byte(line), &d); err != nil {
			return nil, fmt.Errorf("decode line %q: %w", line, err)
		}
		out = append(out, d)
	}
	return out, nil
}

// mooncakeKeyToBlockHash returns the lower-case suffix after the last '@'.
// EmitInt64 keeps only the last 16 hex chars.
func (u *updater) mooncakeKeyToBlockHash(raw string) apis.BlockHash {
	return normalizeMooncakeKey(raw, u.cfg.EmitInt64)
}

func normalizeMooncakeKey(raw string, emitInt64 bool) apis.BlockHash {
	if raw == "" {
		return ""
	}
	var suffix string
	if idx := strings.LastIndexByte(raw, '@'); idx >= 0 {
		suffix = raw[idx+1:]
	} else {
		suffix = raw
	}
	suffix = strings.ToLower(suffix)
	if emitInt64 && len(suffix) > int64HashHexChars {
		suffix = suffix[len(suffix)-int64HashHexChars:]
	}
	return apis.BlockHash(suffix)
}

// mooncakeKeyToBlockHash keeps legacy tests on full-suffix behavior.
// New code should use updater.mooncakeKeyToBlockHash.
func mooncakeKeyToBlockHash(raw string) apis.BlockHash {
	return normalizeMooncakeKey(raw, false)
}

// escapeMooncakeKeyForQuery encodes only query-breaking bytes.
// Cinatra does not decode queries, so '@', ':' and safe delimiters stay literal.
func escapeMooncakeKeyForQuery(raw string) string {
	const hex = "0123456789abcdef"
	var b strings.Builder
	b.Grow(len(raw))
	for i := 0; i < len(raw); i++ {
		c := raw[i]
		switch c {
		case '%', '#', '&', '+', ' ', '\t', '\r', '\n', '?':
			b.WriteByte('%')
			b.WriteByte(hex[c>>hexNibbleShift])
			b.WriteByte(hex[c&hexLowNibbleMask])
		default:
			b.WriteByte(c)
		}
	}
	return b.String()
}
