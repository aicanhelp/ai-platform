/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You can obtain a copy of PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// fake-e2b is a programmable stand-in for the E2B API-Server used by the warm
// e2e suite. It implements the warm-snapshot dispatch endpoint plus an admin
// API for fault injection and request-log inspection, and doubles as a
// MoonCakeStore reporter that periodically writes SuperPod storage records
// into the (fake) MoonCake etcd.
//
// Endpoints:
//
//	POST /v1/warm-snapshot  warm dispatch (1 success / 2 faulted)
//	POST /_admin/fault      fault injection: always/once/off per target
//	GET  /_admin/requests   dispatch log for dispatch-count assertions
//	POST /_admin/report     MoonCakeStore report control (pause/backdate/bytes)
//
// Environment:
//
//	FAKE_E2B_PORT         listen port (default 8080)
//	FAKE_ETCD_ENDPOINTS   etcd endpoints; empty disables the reporter
//	FAKE_ETCD_PREFIX      etcd key prefix (default /mooncake_cluster/masters/primary)
//	FAKE_SUPERPODS        initial reporters "id:available_bytes,..." (default
//	                      superpod1:4000,superpod2:3000,superpod3:2000)
//	FAKE_REPORT_INTERVAL  reporter period (default 3s)
package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
)

// dispatch request/response mirror internal/snapshotwarmup.E2BWarmSnapshot*.

type warmRequest struct {
	SnapshotID       string `json:"snapshot_id"`
	WarmedSnapshotID string `json:"warmed_snapshot_id"`
	TargetSuperPodID string `json:"target_superpod_id"`
}

type warmResponse struct {
	SnapshotID string `json:"snapshot_id"`
	Result     int    `json:"result"`
}

type requestRecord struct {
	SnapshotID       string `json:"snapshot_id"`
	WarmedSnapshotID string `json:"warmed_snapshot_id"`
	TargetSuperPodID string `json:"target_superpod_id"`
	At               string `json:"at"`
}

// fault modes
const (
	faultAlways = "always" // fail every dispatch until switched off
	faultOnce   = "once"   // fail the first dispatch, then auto-recover
	faultOff    = "off"    // clear the rule
)

type faultRule struct {
	Mode string `json:"mode"`
}

type reportRequest struct {
	SuperPod       string `json:"superpod"`
	Pause          bool   `json:"pause"`
	AvailableBytes int64  `json:"available_bytes"`
	TotalBytes     int64  `json:"total_bytes"`
	UsedBytes      int64  `json:"used_bytes"`
	BackdateMS     int64  `json:"backdate_ms"`
}

type superpodReport struct {
	id       string
	total    int64
	used     int64
	avail    int64
	paused   bool
	backdate time.Duration
}

type server struct {
	mu        sync.Mutex
	requests  []requestRecord
	faults    map[string]string // warmed_snapshot_id -> mode
	reporters map[string]*superpodReport
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("write response failed: %v", err)
	}
}

func (s *serverWithEtcd) handleWarmSnapshot(w http.ResponseWriter, r *http.Request) {
	var req warmRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "bad request body"})
		return
	}

	s.mu.Lock()
	s.requests = append(s.requests, requestRecord{
		SnapshotID:       req.SnapshotID,
		WarmedSnapshotID: req.WarmedSnapshotID,
		TargetSuperPodID: req.TargetSuperPodID,
		At:               time.Now().UTC().Format(time.RFC3339Nano),
	})
	mode, hasFault := s.faults[req.WarmedSnapshotID]
	if hasFault && mode == faultOnce {
		delete(s.faults, req.WarmedSnapshotID) // one-shot: auto-recover
	}
	s.mu.Unlock()

	if d := s.currentDelay(); d > 0 {
		select {
		case <-time.After(d):
		case <-r.Context().Done():
			return
		}
	}

	result := 1
	if hasFault && (mode == faultAlways || mode == faultOnce) {
		result = 2
	}
	writeJSON(w, http.StatusOK, warmResponse{SnapshotID: req.SnapshotID, Result: result})
}

func (s *server) handleFault(w http.ResponseWriter, r *http.Request) {
	var rule faultRule
	if err := json.NewDecoder(r.Body).Decode(&rule); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "bad request body"})
		return
	}
	if rule.Mode != faultAlways && rule.Mode != faultOnce && rule.Mode != faultOff {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "mode must be always|once|off"})
		return
	}

	// The rule key (warmed_snapshot_id) is passed as ?id=...
	id := r.URL.Query().Get("id")
	if id == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "missing ?id=warmed_snapshot_id"})
		return
	}

	s.mu.Lock()
	if rule.Mode == faultOff {
		delete(s.faults, id)
	} else {
		s.faults[id] = rule.Mode
	}
	s.mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]string{"ok": "true"})
}

func (s *server) handleRequests(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	switch r.Method {
	case http.MethodGet:
		writeJSON(w, http.StatusOK, map[string]any{"requests": s.requests})
	case http.MethodDelete:
		s.requests = nil
		writeJSON(w, http.StatusOK, map[string]string{"ok": "true"})
	default:
		writeJSON(w, http.StatusMethodNotAllowed, nil)
	}
}

// currentDelay snapshots the dispatch delay under the lock.
func (s *serverWithEtcd) currentDelay() time.Duration {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.dispatchDelay
}

// handleSetDelay adjusts the artificial per-dispatch delay at runtime (ms).
// Scale specs shrink it so a 10k-snapshot warm finishes in minutes instead
// of hours; other specs restore the 2s default for a visible WARMING window.
func (s *serverWithEtcd) handleSetDelay(w http.ResponseWriter, r *http.Request) {
	var body struct {
		DelayMS int64 `json:"delay_ms"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "bad request body"})
		return
	}
	s.mu.Lock()
	s.dispatchDelay = time.Duration(body.DelayMS) * time.Millisecond
	s.mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]string{"ok": "true"})
}

func (s *serverWithEtcd) handleReport(w http.ResponseWriter, r *http.Request) {
	var req reportRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "bad request body"})
		return
	}

	s.mu.Lock()
	rep, ok := s.reporters[req.SuperPod]
	s.mu.Unlock()
	if !ok {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "unknown superpod"})
		return
	}

	s.mu.Lock()
	if req.Pause {
		rep.paused = true
	} else {
		rep.paused = false
	}
	if req.AvailableBytes > 0 {
		rep.avail = req.AvailableBytes
	}
	if req.TotalBytes > 0 {
		rep.total = req.TotalBytes
	}
	if req.UsedBytes > 0 {
		rep.used = req.UsedBytes
	}
	rep.backdate = time.Duration(req.BackdateMS) * time.Millisecond
	s.mu.Unlock()

	// Report immediately so tests do not have to wait for the next tick.
	s.reportOnce(rep)
	writeJSON(w, http.StatusOK, map[string]string{"ok": "true"})
}

type etcdRecord struct {
	FileTotalBytes     int64  `json:"file_total_bytes"`
	FileUsedBytes      int64  `json:"file_used_bytes"`
	FileAvailableBytes int64  `json:"file_available_bytes"`
	UpdatedAt          string `json:"updated_at"`
}

func (s *serverWithEtcd) reportOnce(rep *superpodReport) {
	if s.etcd == nil || rep.paused {
		return
	}
	s.mu.Lock()
	rec := etcdRecord{
		FileTotalBytes:     rep.total,
		FileUsedBytes:      rep.used,
		FileAvailableBytes: rep.avail,
		UpdatedAt:          time.Now().UTC().Add(-rep.backdate).Format(time.RFC3339Nano),
	}
	s.mu.Unlock()
	val, err := json.Marshal(rec)
	if err != nil {
		log.Printf("marshal report %s failed: %v", rep.id, err)
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	_, err = s.etcd.Put(ctx, s.prefix+"/"+rep.id, string(val))
	if err != nil {
		log.Printf("report %s failed: %v", rep.id, err)
	}
}

func (s *serverWithEtcd) runReporter(ctx context.Context, interval time.Duration) {
	if s.etcd == nil {
		return
	}
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			s.mu.Lock()
			reps := make([]*superpodReport, 0, len(s.reporters))
			for _, rep := range s.reporters {
				reps = append(reps, rep)
			}
			s.mu.Unlock()
			for _, rep := range reps {
				s.reportOnce(rep)
			}
		}
	}
}

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func parseSuperpods(spec string) map[string]*superpodReport {
	out := map[string]*superpodReport{}
	for _, item := range strings.Split(spec, ",") {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		id, avail := item, int64(1000)
		if i := strings.IndexByte(item, ':'); i > 0 {
			id = item[:i]
			if n, err := strconv.ParseInt(item[i+1:], 10, 64); err == nil {
				avail = n
			}
		}
		out[id] = &superpodReport{
			id:    id,
			total: avail * 2,
			used:  avail, // used = total - available keeps the fields consistent
			avail: avail,
		}
	}
	return out
}

type serverWithEtcd struct {
	server
	etcd          *clientv3.Client
	prefix        string
	dispatchDelay time.Duration
}

func main() {
	port := envOrDefault("FAKE_E2B_PORT", "8080")
	etcdEndpoints := os.Getenv("FAKE_ETCD_ENDPOINTS")
	prefix := envOrDefault("FAKE_ETCD_PREFIX", "/mooncake_cluster/masters/primary")
	interval, err := time.ParseDuration(envOrDefault("FAKE_REPORT_INTERVAL", "3s"))
	if err != nil {
		log.Fatalf("invalid FAKE_REPORT_INTERVAL: %v", err)
	}

	s := &serverWithEtcd{
		server: server{
			faults:    map[string]string{},
			reporters: parseSuperpods(envOrDefault("FAKE_SUPERPODS", "superpod1:4000,superpod2:3000,superpod3:2000")),
		},
		prefix: prefix,
	}

	if d := os.Getenv("FAKE_DISPATCH_DELAY"); d != "" {
		if delay, err := time.ParseDuration(d); err != nil {
			log.Fatalf("invalid FAKE_DISPATCH_DELAY: %v", err)
		} else {
			s.dispatchDelay = delay
			log.Printf("dispatch delay: %s", delay)
		}
	}

	if etcdEndpoints != "" {
		s.etcd, err = clientv3.New(clientv3.Config{
			Endpoints:   strings.Split(etcdEndpoints, ","),
			DialTimeout: 5 * time.Second,
		})
		if err != nil {
			log.Fatalf("connect etcd: %v", err)
		}
		defer s.etcd.Close()
		log.Printf("mooncake reporter enabled: etcd=%s prefix=%s interval=%s", etcdEndpoints, prefix, interval)
	} else {
		log.Print("mooncake reporter disabled (FAKE_ETCD_ENDPOINTS empty)")
	}

	ctx, stop := context.WithCancel(context.Background())
	defer stop()
	go s.runReporter(ctx, interval)

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/warm-snapshot", s.handleWarmSnapshot)
	// The per-node third-party warm service contract uses /warm-snapshot
	// (no /v1 prefix); serve both from the same handler.
	mux.HandleFunc("POST /warm-snapshot", s.handleWarmSnapshot)
	mux.HandleFunc("POST /_admin/fault", s.handleFault)
	mux.HandleFunc("GET /_admin/requests", s.handleRequests)
	mux.HandleFunc("DELETE /_admin/requests", s.handleRequests)
	mux.HandleFunc("POST /_admin/report", s.handleReport)
	mux.HandleFunc("POST /_admin/delay", s.handleSetDelay)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
	})

	addr := ":" + port
	log.Printf("fake-e2b listening on %s", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatalf("listen: %v", err)
	}
}
