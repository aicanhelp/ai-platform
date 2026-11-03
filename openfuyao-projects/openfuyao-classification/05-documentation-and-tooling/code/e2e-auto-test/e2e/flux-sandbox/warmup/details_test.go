//go:build e2e

/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You can obtain a copy of PSL v2 at:
 *          http://license.coscl.org.cn/Mulan PSL v2.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package warmup

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"time"

	ginkgo "github.com/onsi/ginkgo/v2"
	gomega "github.com/onsi/gomega"
	clientv3 "go.etcd.io/etcd/client/v3"
)

// cacheDetailsResponse decodes GET /api/v1/snapshots/cache-details:
// map[superpodID][]cachedSnapshot.
type cacheDetailsResponse map[string][]struct {
	SnapshotID string    `json:"snapshot_id"`
	Category   string    `json:"category"`
	SizeBytes  int64     `json:"size_bytes"`
	CachedAt   time.Time `json:"cached_at"`
}

func (e *warmEnv) getCacheDetails(tb ginkgo.GinkgoTInterface, query string) (cacheDetailsResponse, int, []byte) {
	tb.Helper()
	path := "/api/v1/snapshots/cache-details"
	if query != "" {
		path += "?" + query
	}
	resp, raw := e.httpDo(tb, http.MethodGet, path, nil)
	var out cacheDetailsResponse
	if resp.StatusCode == http.StatusOK {
		if err := json.Unmarshal(raw, &out); err != nil {
			tb.Fatalf("decode cache-details %s: %v", raw, err)
		}
	}
	return out, resp.StatusCode, raw
}

// storageDetailsResponse decodes GET /api/v1/superpods/storage-details.
type storageDetailsResponse map[string]struct {
	Health         string    `json:"health"`
	TotalBytes     int64     `json:"total_bytes"`
	UsedBytes      int64     `json:"used_bytes"`
	AvailableBytes int64     `json:"available_bytes"`
	ObservedAt     time.Time `json:"observed_at"`
}

func (e *warmEnv) getStorageDetails(tb ginkgo.GinkgoTInterface) (storageDetailsResponse, int, []byte) {
	tb.Helper()
	resp, raw := e.httpDo(tb, http.MethodGet, "/api/v1/superpods/storage-details", nil)
	var out storageDetailsResponse
	if resp.StatusCode == http.StatusOK {
		if err := json.Unmarshal(raw, &out); err != nil {
			tb.Fatalf("decode storage-details %s: %v", raw, err)
		}
	}
	return out, resp.StatusCode, raw
}

// ── Feature group: warm cache and storage details queries ──

var _ = ginkgo.Describe("Warm Details Query", ginkgo.Label("flux-sandbox", "details"), func() {

	ginkgo.It("returns cache details aggregated by SuperPod (WARM-E2E-005)", func() {
		tb := ginkgo.GinkgoT()
		ids := []string{"snapshot-e2e-001", "snapshot-e2e-002"}

		// Reuse the snapshots warmed by WARM-E2E-003; if this spec runs
		// standalone, warm them now.
		wEnv.db(tb).deleteDetails(tb, ids...)
		created, code, raw := wEnv.createWarmTask(tb, ids)
		gomega.Expect(code).To(gomega.Equal(http.StatusCreated), "body %s", raw)
		gomega.Expect(wEnv.waitForTaskTerminal(tb, created.TaskID, 2*time.Minute)).To(gomega.Equal(2))

		// Step 1: query cache details.
		details, code, raw := wEnv.getCacheDetails(tb, "")
		gomega.Expect(code).To(gomega.Equal(http.StatusOK), "body %s", raw)
		gomega.Expect(details).NotTo(gomega.BeEmpty())

		// Step 2: the API view must equal the DB state=1 rows expanded by
		// warmed_superpod_id. Build the expected view from the DB.
		rows, err := wEnv.db(tb).conn.Query(context.Background(),
			`SELECT snapshot_id, category, warmed_superpod_id FROM superpod_warm_detail WHERE state = 1`)
		if err != nil {
			tb.Fatalf("query details: %v", err)
		}
		defer rows.Close()
		expected := map[string]map[string]bool{}
		for rows.Next() {
			var snap string
			var cat int16
			var warmed json.RawMessage
			if err := rows.Scan(&snap, &cat, &warmed); err != nil {
				tb.Fatalf("scan detail: %v", err)
			}
			for _, sp := range jsonStrings(tb, warmed) {
				if expected[sp] == nil {
					expected[sp] = map[string]bool{}
				}
				expected[sp][snap] = true
			}
		}

		// Each SuperPod key in the API response matches the expected set.
		apiView := map[string]map[string]bool{}
		for sp, snaps := range details {
			apiView[sp] = map[string]bool{}
			for _, s := range snaps {
				apiView[sp][s.SnapshotID] = true
				gomega.Expect(s.Category).To(gomega.Equal("REGULAR"), "%s category", s.SnapshotID)
				gomega.Expect(s.SizeBytes).To(gomega.Equal(int64(0)), "%s size_bytes (fixed 0)", s.SnapshotID)
				gomega.Expect(s.CachedAt).NotTo(gomega.BeZero(), "%s cached_at", s.SnapshotID)
			}
		}
		// Restrict the comparison to the snapshots this suite warmed (other
		// rows may exist from prior suite runs in the same namespace).
		for sp, snaps := range expected {
			for snap := range snaps {
				gomega.Expect(apiView[sp]).To(gomega.HaveKey(snap),
					"snapshot %s missing under superpod %s", snap, sp)
			}
		}
		// Each of the two snapshots appears under 2 SuperPods.
		for _, snap := range ids {
			owners := 0
			for _, snaps := range apiView {
				if snaps[snap] {
					owners++
				}
			}
			gomega.Expect(owners).To(gomega.Equal(2), "snapshot %s cached on %d superpods", snap, owners)
		}

		// Step 3: an out-of-range page returns 200 with an empty object.
		empty, code, raw := wEnv.getCacheDetails(tb, "page_size=10000&page_num=100")
		gomega.Expect(code).To(gomega.Equal(http.StatusOK), "body %s", raw)
		gomega.Expect(empty).To(gomega.BeEmpty())
	})

	ginkgo.It("returns SuperPod storage details from etcd reports (WARM-E2E-006)", func() {
		tb := ginkgo.GinkgoT()

		// Step 1: query storage details - all N SuperPods READY.
		details, code, raw := wEnv.getStorageDetails(tb)
		gomega.Expect(code).To(gomega.Equal(http.StatusOK), "body %s", raw)
		gomega.Expect(details).To(gomega.HaveLen(len(wSuperPodIDs)))
		for _, sp := range wSuperPodIDs {
			d, ok := details[sp]
			gomega.Expect(ok).To(gomega.BeTrue(), "superpod %s missing", sp)
			gomega.Expect(d.Health).To(gomega.Equal("READY"), "superpod %s health", sp)
			gomega.Expect(d.TotalBytes).NotTo(gomega.BeZero())
			gomega.Expect(d.ObservedAt).NotTo(gomega.BeZero())
		}

		// Step 2: capacity must equal the etcd report fields
		// (file_total/used/available_bytes).
		resp, err := wEnv.etcdClient.Get(context.Background(), "/mooncake_cluster/masters/primary", clientv3.WithPrefix())
		if err != nil {
			tb.Fatalf("etcd get prefix: %v", err)
		}
		gomega.Expect(resp.Kvs).NotTo(gomega.BeEmpty())
		for _, kv := range resp.Kvs {
			key := string(kv.Key)
			sp := key[strings.LastIndex(key, "/")+1:]
			var rec struct {
				FileTotalBytes     int64     `json:"file_total_bytes"`
				FileUsedBytes      int64     `json:"file_used_bytes"`
				FileAvailableBytes int64     `json:"file_available_bytes"`
				UpdatedAt          time.Time `json:"updated_at"`
			}
			if err := json.Unmarshal(kv.Value, &rec); err != nil {
				tb.Fatalf("decode etcd record %s: %v", key, err)
			}
			d, ok := details[sp]
			gomega.Expect(ok).To(gomega.BeTrue(), "superpod %s in etcd but not API", sp)
			gomega.Expect(d.TotalBytes).To(gomega.Equal(rec.FileTotalBytes), "%s total_bytes", sp)
			gomega.Expect(d.UsedBytes).To(gomega.Equal(rec.FileUsedBytes), "%s used_bytes", sp)
			gomega.Expect(d.AvailableBytes).To(gomega.Equal(rec.FileAvailableBytes), "%s available_bytes", sp)
			gomega.Expect(d.ObservedAt).To(gomega.Equal(rec.UpdatedAt), "%s observed_at", sp)
		}

		// Step 3: observed_at refreshes after a report cycle (3s interval).
		before := details["superpod1"].ObservedAt
		waitFor(tb, "observed_at refresh", 15*time.Second, func() bool {
			after, _, _ := wEnv.getStorageDetails(tb)
			return after["superpod1"].ObservedAt.After(before)
		})
	})
})
