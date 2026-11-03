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
	"time"

	ginkgo "github.com/onsi/ginkgo/v2"
	gomega "github.com/onsi/gomega"
)

// ── Feature group: etcd staleness state transitions ──

var _ = ginkgo.Describe("Warm Storage Health", ginkgo.Label("flux-sandbox", "health"), func() {

	ginkgo.It("marks a stale MoonCake report UNAVAILABLE and recovers (WARM-E2E-020)", func() {
		tb := ginkgo.GinkgoT()

		// Step 1: all SuperPods READY to begin with.
		details, code, raw := wEnv.getStorageDetails(tb)
		gomega.Expect(code).To(gomega.Equal(200), "body %s", raw)
		for _, sp := range wSuperPodIDs {
			gomega.Expect(details[sp].Health).To(gomega.Equal("READY"), "%s initial health", sp)
		}

		// Step 2: backdate superpod3's report by 10 minutes. The fake
		// reporter writes the stale timestamp immediately.
		ginkgo.DeferCleanup(func() {
			wEnv.e2bAdmin(tb, "POST", "/_admin/report", map[string]any{"superpod": "superpod3", "backdate_ms": 0})
		})
		wEnv.e2bAdmin(tb, "POST", "/_admin/report", map[string]any{"superpod": "superpod3", "backdate_ms": 600000})

		// Step 3: past the staleness threshold (10s) the SuperPod flips to
		// UNAVAILABLE while the others stay READY, and the endpoint still
		// returns 200 overall (partial degradation is not 503).
		waitFor(tb, "superpod3 marked UNAVAILABLE", 30*time.Second, func() bool {
			d, _, _ := wEnv.getStorageDetails(tb)
			return d["superpod3"].Health == "UNAVAILABLE"
		})
		d2, c2, r2 := wEnv.getStorageDetails(tb)
		gomega.Expect(c2).To(gomega.Equal(200), "body %s", r2)
		gomega.Expect(d2["superpod1"].Health).To(gomega.Equal("READY"))
		gomega.Expect(d2["superpod2"].Health).To(gomega.Equal("READY"))

		// Step 4: restore a fresh report - the SuperPod returns to READY
		// (health is computed per observation, not sticky).
		wEnv.e2bAdmin(tb, "POST", "/_admin/report", map[string]any{"superpod": "superpod3", "backdate_ms": 0})
		waitFor(tb, "superpod3 recovered to READY", 30*time.Second, func() bool {
			d, _, _ := wEnv.getStorageDetails(tb)
			return d["superpod3"].Health == "READY"
		})
	})
})
