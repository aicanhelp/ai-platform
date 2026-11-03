/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package dataplane

import (
	ginkgo "github.com/onsi/ginkgo/v2"
	gomega "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/flux-sandbox/framework"
)

// ── Feature group: special-deployment agent ──
//
// These specs target an independently deployed agent with non-default config.
// Point E2E_SMALL_CAPACITY_AGENT_URL at that agent (FLUXSANDBOXAGENT_CAPACITY=2)
// to enable them.

var _ = ginkgo.Describe("Special Deployment", ginkgo.Label("flux-sandbox", "agent-api"), func() {

	ginkgo.Describe("capacity", func() {
		ginkgo.It("rejects create when capacity is exceeded (DP-E2E-005)", ginkgo.Label("needs-small-capacity"), func() {
			t := ginkgo.GinkgoT()
			smallAgent := smallCapacityAgent()
			if smallAgent == nil {
				ginkgo.Skip("skipped: set E2E_SMALL_CAPACITY_AGENT_URL to enable (capacity=2 agent)")
			}
			const capacity = 2
			createOnAgent(t, smallAgent, capacity)

			extraID := framework.NewID(t.Name()) + "-extra"
			resp, code, err := smallAgent.Create(framework.BasicSpec(extraID))
			gomega.Expect(err).NotTo(gomega.HaveOccurred())
			gomega.Expect(code).To(gomega.Equal(500), "over-capacity create: %+v", resp)
			gomega.Expect(resp.Success).To(gomega.BeFalse())
			gomega.Expect(resp.Message).To(gomega.ContainSubstring("capacity exceeded"))
		})
	})
})
