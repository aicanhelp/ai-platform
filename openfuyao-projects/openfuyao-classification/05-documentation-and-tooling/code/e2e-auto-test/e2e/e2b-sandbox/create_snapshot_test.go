/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 */

package e2b_sandbox

import (
	"sync"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/e2b-sandbox/utils"
)

var _ = Describe("创建快照",
	Label("e2b-sandbox", "with-workload-cluster", "26.01"), Label("P0"), func() {

		It("E2E-IT-004：创建快照成功——返回 image 和 teamId", func() {
			By("创建 sandbox")
			sandboxID := createSandboxAndAwaitRunning("it004")

			By("创建快照")
			snap, status, err := apiClient.CreateSnapshot(ctx, sandboxID, "my-snapshot-it004")
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(Equal(StatusCreated), "创建快照应返回 201")

			By("校验快照响应字段")
			Expect(snap.ID).NotTo(BeEmpty(), "快照 ID 不应为空")
			Expect(snap.Image).NotTo(BeEmpty(), "image 字段不应为空（TM snapshot_id）")
			Expect(snap.TeamID).NotTo(BeEmpty(), "teamId 字段不应为空（E2B team UUID）")
			Expect(snap.Status.State).To(Equal(SnapReady), "快照状态应为 Ready")
		})

		It("E2E-IT-005：创建快照后 sandbox 仍 Running", func() {
			By("创建 sandbox")
			sandboxID := createSandboxAndAwaitRunning("it005")

			By("创建快照")
			_, status, err := apiClient.CreateSnapshot(ctx, sandboxID, "keep-running-it005")
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(Equal(StatusCreated))

			By("校验 sandbox 仍为 Running")
			sb, _, err := apiClient.GetSandbox(ctx, sandboxID)
			Expect(err).NotTo(HaveOccurred())
			Expect(sb.Status.State).To(Equal(StateRunning), "创建快照后 sandbox 应仍为 Running")
		})

		It("E2E-IT-006：sandbox 不存在时创建快照返回 404", func() {
			By("对不存在的 sandbox 创建快照")
			_, status, err := apiClient.CreateSnapshot(ctx, "nonexistent-sandbox-id", "should-fail")
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(Equal(StatusNotFound), "sandbox 不存在应返回 404")
		})

		It("E2E-IT-007：sandbox 非 Running 时创建快照返回 409", func() {
			By("创建 sandbox 并暂停")
			sandboxID := createSandboxAndAwaitRunning("it007")
			pauseStatus, err := apiClient.PauseSandbox(ctx, sandboxID)
			Expect(err).NotTo(HaveOccurred())
			Expect(pauseStatus).To(Equal(StatusAccepted))
			awaitSandboxState(sandboxID, StatePaused)

			By("对 Paused sandbox 创建快照")
			_, status, err := apiClient.CreateSnapshot(ctx, sandboxID, "should-fail-paused")
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(Equal(StatusConflict), "Paused sandbox 创建快照应返回 409")
		})

		It("E2E-IT-008：并发创建快照——同一 sandbox 只允许一个", func() {
			By("创建 sandbox")
			sandboxID := createSandboxAndAwaitRunning("it008")

			By("并发发起 2 个快照创建请求")
			var wg sync.WaitGroup
			statuses := make([]int, 2)
			wg.Add(2)
			for i := 0; i < 2; i++ {
				go func(idx int) {
					defer wg.Done()
					_, s, _ := apiClient.CreateSnapshot(ctx, sandboxID, "concurrent-snap-it008")
					statuses[idx] = s
				}(i)
			}
			wg.Wait()

			By("校验一个 201 一个 409")
			passCount := 0
			conflictCount := 0
			for _, s := range statuses {
				if s == StatusCreated {
					passCount++
				} else if s == StatusConflict {
					conflictCount++
				}
			}
			Expect(passCount).To(Equal(1), "应有一个请求返回 201")
			Expect(conflictCount).To(Equal(1), "应有一个请求返回 409")
		})
	})
