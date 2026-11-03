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
	"sort"
	"time"

	ginkgo "github.com/onsi/ginkgo/v2"
	gomega "github.com/onsi/gomega"
)

// ── Feature group: load balancing and idempotent skip ──
//
// The fake MoonCakeStore reports strictly descending capacities
// (superpod1:4000 > superpod2:3000 > superpod3:2000), so the available-pool
// ordering is deterministic: the load balancer picks the top-2 capacity pods
// first and rotates a shared cursor across snapshots.

var _ = ginkgo.Describe("Warm Load Balancing", ginkgo.Label("flux-sandbox", "lb"), func() {

	ginkgo.It("spreads regular snapshots across SuperPods by capacity (WARM-E2E-008)", func() {
		tb := ginkgo.GinkgoT()
		ids := []string{"snapshot-lb-001", "snapshot-lb-002"}
		wEnv.db(tb).deleteDetails(tb, ids...)

		// Step 1: submit a two-snapshot task and wait for terminal state.
		created, code, raw := wEnv.createWarmTask(tb, ids)
		gomega.Expect(code).To(gomega.Equal(201), "body %s", raw)
		gomega.Expect(wEnv.waitForTaskTerminal(tb, created.TaskID, 2*time.Minute)).To(gomega.Equal(2))

		// Step 2: the first snapshot gets the top-2 capacity SuperPods; the
		// second rotates the shared cursor and includes superpod3.
		row1 := wEnv.db(tb).detailRowMust(tb, ids[0])
		row2 := wEnv.db(tb).detailRowMust(tb, ids[1])
		targets1 := jsonStrings(tb, row1.SuperPodID)
		targets2 := jsonStrings(tb, row2.SuperPodID)

		gomega.Expect(targets1).To(gomega.ConsistOf("superpod1", "superpod2"),
			"first snapshot must target the two highest-capacity SuperPods")
		gomega.Expect(targets2).To(gomega.ContainElement("superpod3"),
			"second snapshot must rotate the cursor to superpod3")
		gomega.Expect(targets2).To(gomega.HaveLen(2))
		// The two target sets must not be identical (spread, not concentration).
		gomega.Expect(targets1).NotTo(gomega.ConsistOf(targets2),
			"snapshots must not concentrate on the same SuperPod pair")

		// warmed == planned for both (SUCCEED task).
		gomega.Expect(jsonStrings(tb, row1.WarmedSuperPodID)).To(gomega.ConsistOf(targets1))
		gomega.Expect(jsonStrings(tb, row2.WarmedSuperPodID)).To(gomega.ConsistOf(targets2))

		// Step 3: verify against the etcd capacity view - the allocation
		// matches the descending available_bytes order.
		details, code, raw := wEnv.getStorageDetails(tb)
		gomega.Expect(code).To(gomega.Equal(200), "body %s", raw)
		type capEntry struct {
			id    string
			avail int64
		}
		caps := make([]capEntry, 0, len(wSuperPodIDs))
		for _, sp := range wSuperPodIDs {
			caps = append(caps, capEntry{id: sp, avail: details[sp].AvailableBytes})
		}
		sort.Slice(caps, func(i, j int) bool { return caps[i].avail > caps[j].avail })
		top2 := []string{caps[0].id, caps[1].id}
		gomega.Expect(targets1).To(gomega.ConsistOf(top2),
			"first snapshot targets must match the capacity-sorted top-2")
	})

	ginkgo.It("skips an already-cached snapshot without re-dispatching (WARM-E2E-009)", func() {
		tb := ginkgo.GinkgoT()
		snap := "snapshot-idem-001"
		wEnv.db(tb).deleteDetails(tb, snap)
		wEnv.e2bClearRequests(tb)

		// First warm-up: SUCCEED and record the baseline.
		created1, code, raw := wEnv.createWarmTask(tb, []string{snap})
		gomega.Expect(code).To(gomega.Equal(201), "body %s", raw)
		gomega.Expect(wEnv.waitForTaskTerminal(tb, created1.TaskID, 2*time.Minute)).To(gomega.Equal(2))

		row1 := wEnv.db(tb).detailRowMust(tb, snap)
		warmed1 := jsonStrings(tb, row1.WarmedSuperPodID)
		gomega.Expect(row1.State).To(gomega.Equal(int16(1)))
		baseDispatches := wEnv.e2bDispatchCount(tb, snap+":superpod1") +
			wEnv.e2bDispatchCount(tb, snap+":superpod2") +
			wEnv.e2bDispatchCount(tb, snap+":superpod3")
		gomega.Expect(baseDispatches).To(gomega.Equal(2), "baseline dispatch count")

		// Second task with the same snapshot: SUCCEED via the idempotent
		// skip (no new dispatches, detail row unchanged).
		created2, code2, raw2 := wEnv.createWarmTask(tb, []string{snap})
		gomega.Expect(code2).To(gomega.Equal(201), "body %s", raw2)
		gomega.Expect(wEnv.waitForTaskTerminal(tb, created2.TaskID, 2*time.Minute)).To(gomega.Equal(2))

		row2 := wEnv.db(tb).detailRowMust(tb, snap)
		gomega.Expect(row2.State).To(gomega.Equal(int16(1)), "detail state stays SUCCESS")
		gomega.Expect(jsonStrings(tb, row2.WarmedSuperPodID)).To(gomega.ConsistOf(warmed1),
			"warmed set must not change on the second task")

		postDispatches := wEnv.e2bDispatchCount(tb, snap+":superpod1") +
			wEnv.e2bDispatchCount(tb, snap+":superpod2") +
			wEnv.e2bDispatchCount(tb, snap+":superpod3")
		gomega.Expect(postDispatches).To(gomega.Equal(baseDispatches),
			"no new dispatches for an already-cached snapshot")
	})
})
