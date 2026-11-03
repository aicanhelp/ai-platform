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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-logr/logr"

	"gitcode.com/openFuyao/cache-indexer/pkg/apis"
	indexl3 "gitcode.com/openFuyao/cache-indexer/pkg/index/l3"
)

const (
	testMooncakeKeyA       = "m@pcp0@AAAA"
	testMooncakeKeyB       = "m@pcp0@BBBB"
	testMooncakeKeyC       = "m@pcp0@CCCC"
	testMooncakeEP1        = "ep-1:8000"
	testMooncakeEP2        = "ep-2:8000"
	testObjectNotFoundBody = "OBJECT_NOT_FOUND"
	testQueryEscapeRaw     = "host@a:b?c d+#%\t\r\n&"
	testQueryEscapeWant    = "host@a:b%3fc%20d%2b%23%25%09%0d%0a%26"
)

// fakeDoer answers /get_all_keys with a sequence of key-sets and
// /batch_query_keys with a static map[key] -> values.
type fakeDoer struct {
	mu          sync.Mutex
	allKeysSeq  [][]string // pop one per /get_all_keys
	batchValues map[string][]struct {
		TransportEndpoint string `json:"transport_endpoint_"`
		Protocol          string `json:"protocol_"`
	}
	allKeysCalls   atomic.Int32
	batchKeysCalls atomic.Int32
	batchKeysSeen  [][]string
}

func (f *fakeDoer) Do(req *http.Request) (*http.Response, error) {
	switch req.URL.Path {
	case "/get_all_keys":
		f.allKeysCalls.Add(1)
		f.mu.Lock()
		var keys []string
		if len(f.allKeysSeq) > 0 {
			keys = f.allKeysSeq[0]
			f.allKeysSeq = f.allKeysSeq[1:]
		}
		f.mu.Unlock()
		// text/plain, one key per line (matches Mooncake admin server).
		body := []byte(strings.Join(keys, "\n") + "\n")
		return &http.Response{StatusCode: 200, Body: io.NopCloser(bytes.NewReader(body))}, nil
	case "/query_key":
		f.batchKeysCalls.Add(1)
		// GET ?key=<rawkey> — one HTTP call per key.
		asked := req.URL.Query().Get("key")
		f.mu.Lock()
		f.batchKeysSeen = append(f.batchKeysSeen, []string{asked})
		vs, ok := f.batchValues[asked]
		f.mu.Unlock()
		if !ok {
			return &http.Response{StatusCode: 404, Body: io.NopCloser(bytes.NewReader(nil))}, nil
		}
		// text/plain — one descriptor JSON per line.
		var b bytes.Buffer
		for _, v := range vs {
			line, _ := json.Marshal(struct {
				TransportEndpoint string `json:"transport_endpoint_"`
				Protocol          string `json:"protocol_"`
			}{TransportEndpoint: v.TransportEndpoint, Protocol: v.Protocol})
			b.Write(line)
			b.WriteByte('\n')
		}
		return &http.Response{StatusCode: 200, Body: io.NopCloser(&b)}, nil
	}
	return &http.Response{StatusCode: 404, Body: io.NopCloser(bytes.NewReader(nil))}, nil
}

type responseSpec struct {
	status int
	body   string
	err    error
}

type scriptedDoer struct {
	mu      sync.Mutex
	byPath  map[string][]responseSpec
	lastURL string
}

func (s *scriptedDoer) Do(req *http.Request) (*http.Response, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastURL = req.URL.String()
	specs := s.byPath[req.URL.Path]
	if len(specs) == 0 {
		return nil, errors.New("unexpected request: " + req.URL.Path)
	}
	spec := specs[0]
	s.byPath[req.URL.Path] = specs[1:]
	if spec.err != nil {
		return nil, spec.err
	}
	return &http.Response{
		StatusCode: spec.status,
		Body:       io.NopCloser(strings.NewReader(spec.body)),
	}, nil
}

func TestMooncakeKeyToBlockHash(t *testing.T) {
	cases := []struct{ in, want string }{
		{"Qwen3-8B@pcp0@dcp0@head_or_tp_rank:0@pp_rank:0@DEADBEEFDEADBEEF", "deadbeefdeadbeef"},
		{"deadbeef", "deadbeef"},
		{"", ""},
		{"only@", ""}, // trailing @ → fall through to bare hash branch returns "" because idx+1 == len
	}
	for _, c := range cases {
		got := string(mooncakeKeyToBlockHash(c.in))
		if got != c.want {
			t.Errorf("in=%q got=%q want=%q", c.in, got, c.want)
		}
	}
}

// When EmitInt64 is true, a 64-char sha256 hex suffix must be truncated
// to its last 16 chars so it matches blockkey.Builder(EmitInt64=true).
func TestMooncakeKeyToBlockHash_EmitInt64Truncation(t *testing.T) {
	const fullHex = "198bcba7945a909d278ff8b13c418bea79aecf30da51c0663f061826a245be2e"
	const wantInt = "3f061826a245be2e"
	u := &updater{cfg: PollerConfig{EmitInt64: true}.withDefaults()}
	got := string(u.mooncakeKeyToBlockHash("Qwen3-8B@pcp0@dcp0@head_or_tp_rank:0@pp_rank:0@" + fullHex))
	if got != wantInt {
		t.Errorf("EmitInt64=true got=%q want=%q", got, wantInt)
	}
	// EmitInt64=false keeps the full suffix.
	u2 := &updater{cfg: PollerConfig{EmitInt64: false}.withDefaults()}
	got2 := string(u2.mooncakeKeyToBlockHash("Qwen3-8B@pcp0@" + fullHex))
	if got2 != fullHex {
		t.Errorf("EmitInt64=false got=%q want=%q", got2, fullHex)
	}
	// Suffix shorter than 16 chars is a no-op even with truncation on.
	u3 := &updater{cfg: PollerConfig{EmitInt64: true}.withDefaults()}
	if got3 := string(u3.mooncakeKeyToBlockHash("m@pcp0@DEAD")); got3 != "dead" {
		t.Errorf("short suffix got=%q want=%q", got3, "dead")
	}
}

func TestPollOnce_RefreshAndRemoveDiff(t *testing.T) {
	idx := indexl3.New(logr.Discard())
	doer := &fakeDoer{
		allKeysSeq: [][]string{
			{testMooncakeKeyA, testMooncakeKeyB}, // round 1: both new → refresh
			{testMooncakeKeyB, testMooncakeKeyC}, // round 2: AAAA removed, CCCC new
		},
		batchValues: map[string][]struct {
			TransportEndpoint string `json:"transport_endpoint_"`
			Protocol          string `json:"protocol_"`
		}{
			testMooncakeKeyA: {{TransportEndpoint: testMooncakeEP1}},
			testMooncakeKeyB: {{TransportEndpoint: testMooncakeEP1}},
			testMooncakeKeyC: {{TransportEndpoint: testMooncakeEP2}},
		},
	}
	u := &updater{
		log: logr.Discard(), cfg: PollerConfig{}.withDefaults(),
		indexer: idx, client: doer,
		keyOwnership: map[string][]apis.TransportEndpoint{},
	}
	prev := map[string]struct{}{}
	ctx := context.Background()

	// Round 1
	prev = u.pollOnce(ctx, "http://x", prev)
	res, _ := idx.MatchPrefix(apis.ModelContext{}, "ep-1:8000",
		[]apis.BlockHash{"aaaa", "bbbb"})
	if res.MatchedBlocks != 2 {
		t.Fatalf("round1 ep-1 matched=%d want 2", res.MatchedBlocks)
	}

	// Round 2
	prev = u.pollOnce(ctx, "http://x", prev)
	// AAAA must be removed from ep-1 → only BBBB remains.
	res, _ = idx.MatchPrefix(apis.ModelContext{}, "ep-1:8000",
		[]apis.BlockHash{"aaaa"})
	if res.MatchedBlocks != 0 {
		t.Errorf("AAAA should be removed, got %d", res.MatchedBlocks)
	}
	res, _ = idx.MatchPrefix(apis.ModelContext{}, "ep-1:8000",
		[]apis.BlockHash{"bbbb"})
	if res.MatchedBlocks != 1 {
		t.Errorf("BBBB should remain, got %d", res.MatchedBlocks)
	}
	res, _ = idx.MatchPrefix(apis.ModelContext{}, "ep-2:8000",
		[]apis.BlockHash{"cccc"})
	if res.MatchedBlocks != 1 {
		t.Errorf("CCCC should be on ep-2, got %d", res.MatchedBlocks)
	}

	// /query_key is called once per refresh key:
	// round1 = {AAAA, BBBB} → 2 calls; round2 = {CCCC} → 1 call.
	if got := doer.batchKeysCalls.Load(); got != 3 {
		t.Errorf("query_key calls = %d want 3", got)
	}
	if n := len(doer.batchKeysSeen); n == 0 || len(doer.batchKeysSeen[n-1]) != 1 ||
		doer.batchKeysSeen[n-1][0] != testMooncakeKeyC {
		t.Errorf("last refresh entry = %v want [%s]", doer.batchKeysSeen, testMooncakeKeyC)
	}
}

func TestPollOnce_SkipsRowsWithoutTransportEndpoint(t *testing.T) {
	idx := indexl3.New(logr.Discard())
	doer := &fakeDoer{
		allKeysSeq: [][]string{{"m@AAAA"}},
		batchValues: map[string][]struct {
			TransportEndpoint string `json:"transport_endpoint_"`
			Protocol          string `json:"protocol_"`
		}{
			"m@AAAA": {
				{TransportEndpoint: ""}, // skipped
				{TransportEndpoint: "ep:1"},
			},
		},
	}
	u := &updater{log: logr.Discard(), cfg: PollerConfig{}.withDefaults(),
		indexer: idx, client: doer, keyOwnership: map[string][]apis.TransportEndpoint{}}
	u.pollOnce(context.Background(), "http://x", map[string]struct{}{})
	res, _ := idx.MatchPrefix(apis.ModelContext{}, "ep:1", []apis.BlockHash{"aaaa"})
	if res.MatchedBlocks != 1 {
		t.Errorf("ep:1 should hold AAAA, got %d", res.MatchedBlocks)
	}
}

func TestPollOnce_NilPrevInitializesCarryOver(t *testing.T) {
	idx := indexl3.New(logr.Discard())
	doer := &fakeDoer{
		allKeysSeq: [][]string{{"m@AAAA"}},
		batchValues: map[string][]struct {
			TransportEndpoint string `json:"transport_endpoint_"`
			Protocol          string `json:"protocol_"`
		}{
			"m@AAAA": {{TransportEndpoint: "ep:1"}},
		},
	}
	u := &updater{log: logr.Discard(), cfg: PollerConfig{}.withDefaults(),
		indexer: idx, client: doer, keyOwnership: map[string][]apis.TransportEndpoint{}}

	prev := u.pollOnce(context.Background(), "http://x", nil)
	if prev == nil {
		t.Fatal("expected pollOnce to initialize prev")
	}
	if _, ok := prev["m@AAAA"]; !ok {
		t.Fatalf("expected carry-over to contain refreshed key, got %v", prev)
	}
	res, _ := idx.MatchPrefix(apis.ModelContext{}, "ep:1", []apis.BlockHash{"aaaa"})
	if res.MatchedBlocks != 1 {
		t.Fatalf("expected refreshed key to be ingested, got %d", res.MatchedBlocks)
	}
}

func TestEscapeMooncakeKeyForQuery(t *testing.T) {
	got := escapeMooncakeKeyForQuery(testQueryEscapeRaw)
	if got != testQueryEscapeWant {
		t.Fatalf("got=%q want=%q", got, testQueryEscapeWant)
	}
}

func TestFetchQueryKey_ObjectNotFound200(t *testing.T) {
	doer := &scriptedDoer{
		byPath: map[string][]responseSpec{
			"/query_key": {{status: http.StatusOK, body: testObjectNotFoundBody}},
		},
	}
	u := &updater{log: logr.Discard(), cfg: PollerConfig{}.withDefaults(), client: doer}
	rows, err := u.fetchQueryKey(context.Background(), "http://x", testMooncakeKeyA)
	if err != nil {
		t.Fatalf("fetchQueryKey: %v", err)
	}
	if rows != nil {
		t.Fatalf("rows=%v want nil", rows)
	}
	if !strings.Contains(doer.lastURL, escapeMooncakeKeyForQuery(testMooncakeKeyA)) {
		t.Fatalf("lastURL=%q missing escaped key", doer.lastURL)
	}
}

func TestPollOnce_FetchAllKeysErrorPreservesPrev(t *testing.T) {
	prev := map[string]struct{}{testMooncakeKeyA: {}}
	doer := &scriptedDoer{
		byPath: map[string][]responseSpec{
			"/get_all_keys": {{err: errors.New("boom")}},
		},
	}
	u := &updater{
		log:          logr.Discard(),
		cfg:          PollerConfig{}.withDefaults(),
		indexer:      indexl3.New(logr.Discard()),
		client:       doer,
		keyOwnership: map[string][]apis.TransportEndpoint{testMooncakeKeyA: {testMooncakeEP1}},
	}
	got := u.pollOnce(context.Background(), "http://x", prev)
	if len(got) != 1 {
		t.Fatalf("prev mutated: %v", got)
	}
	if _, ok := got[testMooncakeKeyA]; !ok {
		t.Fatalf("expected prev to keep %q", testMooncakeKeyA)
	}
	if len(u.keyOwnership[testMooncakeKeyA]) != 1 {
		t.Fatalf("keyOwnership mutated: %+v", u.keyOwnership)
	}
}

func TestUpdateTarget_StopsPreviousLoop(t *testing.T) {
	idx := indexl3.New(logr.Discard())
	doer := &fakeDoer{allKeysSeq: nil, batchValues: nil}
	u := NewWithConfig(logr.Discard(), idx, PollerConfig{PollInterval: 50 * time.Millisecond}).(*updater)
	u.client = doer

	if err := u.UpdateTarget(&MooncakePollTarget{PodIP: "1.2.3.4", HTTPPort: 8000}); err != nil {
		t.Fatal(err)
	}
	time.Sleep(120 * time.Millisecond)
	first := doer.allKeysCalls.Load()
	if first == 0 {
		t.Fatal("expected at least one /get_all_keys call before swap")
	}
	if err := u.UpdateTarget(nil); err != nil {
		t.Fatal(err)
	}
	time.Sleep(120 * time.Millisecond)
	after := doer.allKeysCalls.Load()
	// A short race window is possible (one tick may fire after we read
	// `first` but before cancel takes effect). Allow at most 1 extra call.
	if after-first > 1 {
		t.Errorf("loop kept polling after nil target: first=%d after=%d", first, after)
	}

	u.Stop()
	if err := u.UpdateTarget(&MooncakePollTarget{PodIP: "1.2.3.4", HTTPPort: 8000}); err == nil {
		t.Error("UpdateTarget after Stop should error")
	}
}

func TestUpdateTarget_InvalidTargetIsNoOp(t *testing.T) {
	idx := indexl3.New(logr.Discard())
	doer := &fakeDoer{}
	u := NewWithConfig(logr.Discard(), idx, PollerConfig{PollInterval: 50 * time.Millisecond}).(*updater)
	u.client = doer

	if err := u.UpdateTarget(&MooncakePollTarget{PodIP: "", HTTPPort: 0}); err != nil {
		t.Fatal(err)
	}
	time.Sleep(80 * time.Millisecond)
	if doer.allKeysCalls.Load() != 0 {
		t.Errorf("invalid target must not trigger HTTP calls")
	}
	u.Stop()
}
