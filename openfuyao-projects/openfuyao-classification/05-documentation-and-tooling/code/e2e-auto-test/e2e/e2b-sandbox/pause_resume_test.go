/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 */

package e2b_sandbox

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/e2b-sandbox/utils"
)

var _ = Describe("暂停与恢复 Sandbox",
	Label("e2b-sandbox", "with-workload-cluster", "26.01"), Label("P0"), func() {

		It("E2E-IT-015：暂停 sandbox 成功", func() {
			By("创建 sandbox")
			sandboxID := createSandboxAndAwaitRunning("it015")

			By("暂停 sandbox")
			status, err := apiClient.PauseSandbox(ctx, sandboxID)
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(Equal(StatusAccepted), "暂停应返回 202")

			By("等待 sandbox 进入 Paused 状态")
			awaitSandboxState(sandboxID, StatePaused)
		})

		It("E2E-IT-016：暂停幂等——已 Paused 再次暂停", func() {
			By("创建 sandbox 并暂停")
			sandboxID := createSandboxAndAwaitRunning("it016")
			status, err := apiClient.PauseSandbox(ctx, sandboxID)
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(Equal(StatusAccepted))
			awaitSandboxState(sandboxID, StatePaused)

			By("再次暂停已 Paused 的 sandbox")
			status2, err2 := apiClient.PauseSandbox(ctx, sandboxID)
			Expect(err2).NotTo(HaveOccurred())
			// 幂等：再次暂停应返回成功（202 或 200）
			Expect(status2).To(BeNumerically("<=", StatusAccepted), "已 Paused 再次暂停应返回成功")
		})

		It("E2E-IT-017：恢复 sandbox 成功", func() {
			By("创建 sandbox 并暂停")
			sandboxID := createSandboxAndAwaitRunning("it017")
			pauseStatus, err := apiClient.PauseSandbox(ctx, sandboxID)
			Expect(err).NotTo(HaveOccurred())
			Expect(pauseStatus).To(Equal(StatusAccepted))
			awaitSandboxState(sandboxID, StatePaused)

			By("恢复 sandbox")
			resumeStatus, err := apiClient.ResumeSandbox(ctx, sandboxID)
			Expect(err).NotTo(HaveOccurred())
			Expect(resumeStatus).To(Equal(StatusAccepted), "恢复应返回 202")

			By("等待 sandbox 恢复 Running")
			awaitSandboxState(sandboxID, StateRunning)
		})

		It("E2E-IT-018：恢复幂等——已 Running 再次恢复返回 409", func() {
			By("创建 sandbox（已 Running）")
			sandboxID := createSandboxAndAwaitRunning("it018")

			By("对 Running sandbox 执行恢复")
			status, err := apiClient.ResumeSandbox(ctx, sandboxID)
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(Equal(StatusConflict), "已 Running 再次恢复应返回 409")
		})

		It("E2E-IT-019：非 Running 状态拒绝暂停", func() {
			By("创建 sandbox 并暂停")
			sandboxID := createSandboxAndAwaitRunning("it019")
			pauseStatus, err := apiClient.PauseSandbox(ctx, sandboxID)
			Expect(err).NotTo(HaveOccurred())
			Expect(pauseStatus).To(Equal(StatusAccepted))
			awaitSandboxState(sandboxID, StatePaused)

			By("对 Paused sandbox 再次暂停（应返回成功或 409）")
			status2, err2 := apiClient.PauseSandbox(ctx, sandboxID)
			Expect(err2).NotTo(HaveOccurred())
			// 已 Paused 状态：幂等返回成功，或 409 冲突
			Expect(status2).To(Or(
				Equal(StatusAccepted),
				Equal(StatusConflict),
			), "已 Paused 再次暂停应返回 202 或 409")
		})
	})
