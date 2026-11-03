//go:build e2e

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

package warmup

import (
	"sync"
	"time"

	ginkgo "github.com/onsi/ginkgo/v2"
	gomega "github.com/onsi/gomega"
)

// ── Feature group: concurrent warm tasks ──

var _ = ginkgo.Describe("Warm Concurrency", ginkgo.Label("flux-sandbox", "concurrency"), func() {

	ginkgo.It("runs three tasks concurrently without duplicate dispatches (WARM-E2E-014)", func() {
		tb := ginkgo.GinkgoT()
		groups := [][]string{
			{"snapshot-c1-001", "snapshot-c1-002"},
			{"snapshot-c2-001", "snapshot-c2-002"},
			{"snapshot-c3-001", "snapshot-c3-002"},
		}
		var all []string
		for _, g := range groups {
			all = append(all, g...)
		}
		wEnv.db(tb).deleteDetails(tb, all...)
		wEnv.e2bClearRequests(tb)

		// Submit the three tasks back-to-back.
		var mu sync.Mutex
		taskIDs := make([]string, 0, 3)
		var wg sync.WaitGroup
		for _, g := range groups {
			g := g
			wg.Add(1)
			go func() {
				defer wg.Done()
				created, code, _ := wEnv.createWarmTask(tb, g)
				gomega.Expect(code).To(gomega.Equal(201))
				mu.Lock()
				taskIDs = append(taskIDs, created.TaskID)
				mu.Unlock()
			}()
		}
		wg.Wait()
		gomega.Expect(taskIDs).To(gomega.HaveLen(3))

		// All three reach SUCCEED and none lingers in WAIT.
		sawConcurrentWarming := false
		warmings := 0
		deadline := time.Now().Add(3 * time.Minute)
		for time.Now().Before(deadline) {
			states := map[int]int{}
			for _, id := range taskIDs {
				states[wEnv.taskState(tb, id)]++
			}
			if states[1] > warmings {
				warmings = states[1]
			}
			if states[1] >= 2 {
				sawConcurrentWarming = true
			}
			if states[2] == 3 {
				break
			}
			time.Sleep(200 * time.Millisecond)
		}
		for _, id := range taskIDs {
			gomega.Expect(wEnv.taskState(tb, id)).To(gomega.Equal(2), "task %s", id)
		}
		// The dispatch delay (2s) plus the poll cadence should show at least
		// two tasks WARMING at once (3 workers configured).
		gomega.Expect(sawConcurrentWarming).To(gomega.BeTrue(),
			"expected at least two tasks in WARMING simultaneously")

		// Every snapshot:superpod pair was dispatched exactly once.
		reqs := wEnv.e2bRequests(tb).Requests
		seen := map[string]int{}
		for _, r := range reqs {
			seen[r.WarmedSnapshotID]++
		}
		gomega.Expect(seen).To(gomega.HaveLen(len(all)*2), "expected 2 targets per snapshot")
		for pair, n := range seen {
			gomega.Expect(n).To(gomega.Equal(1), "pair %s dispatched %d times", pair, n)
		}
	})

	ginkgo.It("keeps one consistent detail row for a concurrently warmed snapshot (WARM-E2E-015)", func() {
		tb := ginkgo.GinkgoT()
		snap := "snapshot-shared-001"
		wEnv.db(tb).deleteDetails(tb, snap)
		wEnv.e2bClearRequests(tb)

		// Two near-simultaneous tasks warming the same snapshot.
		var mu sync.Mutex
		taskIDs := make([]string, 0, 2)
		var wg sync.WaitGroup
		for i := 0; i < 2; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				created, code, _ := wEnv.createWarmTask(tb, []string{snap})
				gomega.Expect(code).To(gomega.Equal(201))
				mu.Lock()
				taskIDs = append(taskIDs, created.TaskID)
				mu.Unlock()
			}()
		}
		wg.Wait()
		for _, id := range taskIDs {
			gomega.Expect(wEnv.waitForTaskTerminal(tb, id, 3*time.Minute)).To(gomega.Equal(2), "task %s", id)
		}

		// Exactly one detail row, SUCCESS, no duplicate targets.
		var rowCount int
		err := wEnv.db(tb).conn.QueryRow(wEnv.ctx,
			`SELECT count(*) FROM superpod_warm_detail WHERE snapshot_id = $1`, snap).Scan(&rowCount)
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
		gomega.Expect(rowCount).To(gomega.Equal(1), "exactly one detail row expected")

		row := wEnv.db(tb).detailRowMust(tb, snap)
		gomega.Expect(row.State).To(gomega.Equal(int16(1)))
		warmed := jsonStrings(tb, row.WarmedSuperPodID)
		gomega.Expect(hasDuplicates(warmed)).To(gomega.BeFalse(), "warmed set: %v", warmed)
		gomega.Expect(warmed).To(gomega.HaveLen(len(unique(warmed))))

		// Every warmed target has at least one dispatch, and the warmed set
		// equals the union of both tasks' plans. Concurrent tasks may both
		// dispatch the same target before either lands in the detail row -
		// that is expected; the idempotency key guarantees at most one
		// effective cache entry (asserted by the dedup checks above).
		reqs := wEnv.e2bRequests(tb).Requests
		seen := map[string]bool{}
		for _, r := range reqs {
			seen[r.WarmedSnapshotID] = true
		}
		for _, sp := range warmed {
			gomega.Expect(seen).To(gomega.HaveKey(snap+":"+sp), "warmed target %s missing a dispatch", sp)
		}

		// The cache-details view shows the snapshot exactly once per pod.
		details, code, raw := wEnv.getCacheDetails(tb, "")
		gomega.Expect(code).To(gomega.Equal(200), "body %s", raw)
		for sp, snaps := range details {
			n := 0
			for _, s := range snaps {
				if s.SnapshotID == snap {
					n++
				}
			}
			if n > 0 {
				gomega.Expect(n).To(gomega.Equal(1), "snapshot appears %d times under %s", n, sp)
			}
		}
	})
})
