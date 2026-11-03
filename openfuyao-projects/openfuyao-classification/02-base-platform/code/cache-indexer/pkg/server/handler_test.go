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

package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/go-logr/logr/testr"

	"gitcode.com/openFuyao/cache-indexer/pkg/apis"
	"gitcode.com/openFuyao/cache-indexer/pkg/config"
)

const (
	testReadyPath      = "/readyz"
	testLargeBodyLimit = int64(8)
)

type stubScore struct {
	got struct {
		mc       apis.ModelContext
		ips      []string
		tokenIDs []int32
	}
	out []apis.ServerScore
	err error
}

func (s *stubScore) Compute(_ context.Context, mc apis.ModelContext, ips []string, t []int32) ([]apis.ServerScore, error) {
	s.got.mc = mc
	s.got.ips = ips
	s.got.tokenIDs = t
	return s.out, s.err
}

func newTestMux(t *testing.T, st *stubScore) http.Handler {
	t.Helper()
	return newMux(testr.New(t), st, normalizeHTTPConfig(config.HTTPConfig{}))
}

func TestHealthz_OK(t *testing.T) {
	mux := newTestMux(t, &stubScore{})
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/healthz", nil))
	if rr.Code != http.StatusOK {
		t.Fatalf("want 200 got %d", rr.Code)
	}
}

func TestReadyz_OK(t *testing.T) {
	mux := newTestMux(t, &stubScore{})
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, testReadyPath, nil))
	if rr.Code != http.StatusOK {
		t.Fatalf("want 200 got %d", rr.Code)
	}
}

func TestHitRate_OK(t *testing.T) {
	st := &stubScore{out: []apis.ServerScore{
		{ServerIP: "1.1.1.1", L1HitRatio: 0.75, L3HitRatio: 0},
		{ServerIP: "2.2.2.2", L1HitRatio: 0.25, L3HitRatio: 0},
	}}
	mux := newTestMux(t, st)
	body := `{"server_ip":["1.1.1.1","2.2.2.2"],"body":{"token_ids":[1,2,3,4],"cache_salt":"s","block_size":4}}`
	req := httptest.NewRequest(http.MethodPost, "/kv-cache/hit-rate", strings.NewReader(body))
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("want 200 got %d body=%s", rr.Code, rr.Body.String())
	}
	if got, want := rr.Header().Get("Content-Type"), "application/json"; got != want {
		t.Fatalf("ct=%s", got)
	}
	var resp hitRateResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if resp.Status != 0 || len(resp.ServerScoreList) != 2 {
		t.Fatalf("resp=%+v", resp)
	}
	if resp.ServerScoreList[0].ServerIP != "1.1.1.1" || resp.ServerScoreList[0].L1HitRatio != 0.75 {
		t.Fatalf("first=%+v", resp.ServerScoreList[0])
	}
	if st.got.mc.BlockSize != 4 || st.got.mc.CacheSalt != "s" {
		t.Fatalf("modelCtx not propagated: %+v", st.got.mc)
	}
	if len(st.got.tokenIDs) != 4 || st.got.tokenIDs[0] != 1 {
		t.Fatalf("tokens not propagated: %+v", st.got.tokenIDs)
	}
}

func TestHitRate_BadJSON(t *testing.T) {
	mux := newTestMux(t, &stubScore{})
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, httptest.NewRequest(http.MethodPost, "/kv-cache/hit-rate", strings.NewReader("not json")))
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("want 400 got %d", rr.Code)
	}
}

func TestHitRate_EmptyBody(t *testing.T) {
	mux := newTestMux(t, &stubScore{})
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, httptest.NewRequest(http.MethodPost, "/kv-cache/hit-rate", strings.NewReader("")))
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("want 400 got %d", rr.Code)
	}
}

func TestHitRate_EmptyServerIP(t *testing.T) {
	mux := newTestMux(t, &stubScore{})
	rr := httptest.NewRecorder()
	body := `{"server_ip":[],"body":{"token_ids":[1],"block_size":1}}`
	mux.ServeHTTP(rr, httptest.NewRequest(http.MethodPost, "/kv-cache/hit-rate", strings.NewReader(body)))
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("want 400 got %d", rr.Code)
	}
}

func TestHitRate_BadBlockSize(t *testing.T) {
	mux := newTestMux(t, &stubScore{})
	rr := httptest.NewRecorder()
	body := `{"server_ip":["x"],"body":{"token_ids":[1],"block_size":0}}`
	mux.ServeHTTP(rr, httptest.NewRequest(http.MethodPost, "/kv-cache/hit-rate", strings.NewReader(body)))
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("want 400 got %d", rr.Code)
	}
}

func TestHitRate_NegativeBlockSize(t *testing.T) {
	mux := newTestMux(t, &stubScore{})
	rr := httptest.NewRecorder()
	body := `{"server_ip":["x"],"body":{"token_ids":[1],"block_size":-1}}`
	mux.ServeHTTP(rr, httptest.NewRequest(http.MethodPost, "/kv-cache/hit-rate", strings.NewReader(body)))
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("want 400 got %d", rr.Code)
	}
}

func TestHitRate_MethodNotAllowed(t *testing.T) {
	mux := newTestMux(t, &stubScore{})
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/kv-cache/hit-rate", nil))
	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("want 405 got %d", rr.Code)
	}
}

func TestHitRate_ScoreError_500(t *testing.T) {
	st := &stubScore{err: errors.New("boom")}
	mux := newTestMux(t, st)
	rr := httptest.NewRecorder()
	body := `{"server_ip":["x"],"body":{"token_ids":[1,2,3,4],"block_size":4}}`
	mux.ServeHTTP(rr, httptest.NewRequest(http.MethodPost, "/kv-cache/hit-rate", strings.NewReader(body)))
	if rr.Code != http.StatusInternalServerError {
		t.Fatalf("want 500 got %d", rr.Code)
	}
}

func TestHitRate_UnknownFieldsRejected(t *testing.T) {
	mux := newTestMux(t, &stubScore{})
	rr := httptest.NewRecorder()
	body := `{"server_ip":["x"],"surprise":1,"body":{"token_ids":[1],"block_size":1}}`
	mux.ServeHTTP(rr, httptest.NewRequest(http.MethodPost, "/kv-cache/hit-rate", bytes.NewBufferString(body)))
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("want 400 got %d", rr.Code)
	}
}

func TestHitRate_BodyTooLarge(t *testing.T) {
	cfg := normalizeHTTPConfig(config.HTTPConfig{MaxHitRateBodyBytes: testLargeBodyLimit})
	mux := newMux(testr.New(t), &stubScore{}, cfg)
	rr := httptest.NewRecorder()
	body := `{"server_ip":["x"],"body":{"token_ids":[1],"block_size":1}}`
	req := httptest.NewRequest(http.MethodPost, "/kv-cache/hit-rate", strings.NewReader(body))
	mux.ServeHTTP(rr, req)
	if rr.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("want 413 got %d", rr.Code)
	}
}
