/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 */

package e2b_sandbox

import (
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/e2b-sandbox/utils"
)

var _ = Describe("E2B Sandbox 全流程",
	Label("e2b-sandbox", "with-workload-cluster", "26.01"), Label("P0"), func() {

		It("E2E-IT-022：创建→快照→从快照创建→暂停→恢复→删除 全流程", func() {
			By("Step 1: 创建 sandbox")
			sandboxID := createSandboxAndAwaitRunning("flow022")

			By("Step 2: 创建快照")
			snapshotID := createSnapshotAndAwaitReady(sandboxID, "flow-snap-022")

			By("Step 3: 从快照创建新 sandbox")
			resp, status, err := apiClient.CreateSandbox(ctx, &CreateSandboxRequest{
				SnapshotID: snapshotID,
				Entrypoint: []string{"sleep", "3600"},
				ResourceLimits: &ResourceLimits{
					CPU:    DefaultCPU,
					Memory: DefaultMemory,
				},
				Extensions: map[string]string{"sandboxGroup": sgName},
				Timeout:    DefaultTimeout,
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(Equal(StatusAccepted))
			newSandboxID := resp.ID
			awaitSandboxState(newSandboxID, StateRunning)

			By("等待 sandbox 完全就绪后再暂停")
			time.Sleep(5 * time.Second)

			By("Step 4: 暂停新 sandbox")
			pauseStatus, err := apiClient.PauseSandbox(ctx, newSandboxID)
			Expect(err).NotTo(HaveOccurred())
			Expect(pauseStatus).To(Equal(StatusAccepted))
			awaitSandboxState(newSandboxID, StatePaused)

			By("Step 5: 恢复 sandbox")
			resumeStatus, err := apiClient.ResumeSandbox(ctx, newSandboxID)
			Expect(err).NotTo(HaveOccurred())
			Expect(resumeStatus).To(Equal(StatusAccepted))
			awaitSandboxState(newSandboxID, StateRunning)

			By("Step 6: 删除快照")
			delStatus, err := apiClient.DeleteSnapshot(ctx, snapshotID)
			Expect(err).NotTo(HaveOccurred())
			Expect(delStatus).To(Equal(StatusNoContent))

			By("全流程完成")
		})

		It("E2E-IT-023：暂停→恢复→再暂停 全流程", func() {
			By("创建 sandbox")
			sandboxID := createSandboxAndAwaitRunning("flow023")

			By("Step 1: 暂停")
			pauseStatus1, err := apiClient.PauseSandbox(ctx, sandboxID)
			Expect(err).NotTo(HaveOccurred())
			Expect(pauseStatus1).To(Equal(StatusAccepted))
			awaitSandboxState(sandboxID, StatePaused)

			By("Step 2: 恢复")
			resumeStatus, err := apiClient.ResumeSandbox(ctx, sandboxID)
			Expect(err).NotTo(HaveOccurred())
			Expect(resumeStatus).To(Equal(StatusAccepted))
			awaitSandboxState(sandboxID, StateRunning)

			By("等待 sandbox 完全就绪后再暂停")
			time.Sleep(5 * time.Second)

			By("Step 3: 再暂停")
			pauseStatus2, err := apiClient.PauseSandbox(ctx, sandboxID)
			Expect(err).NotTo(HaveOccurred())
			Expect(pauseStatus2).To(Equal(StatusAccepted))
			awaitSandboxState(sandboxID, StatePaused)

			By("状态正确转换 Running→Paused→Running→Paused")
		})
	})
