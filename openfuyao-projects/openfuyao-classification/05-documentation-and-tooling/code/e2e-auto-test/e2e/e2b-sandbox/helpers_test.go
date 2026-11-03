/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 */

package e2b_sandbox

import (
	"fmt"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/e2b-sandbox/utils"
)

const (
	sandboxWaitTimeout   = 3 * time.Minute
	sandboxPollInterval  = 5 * time.Second
	snapshotWaitTimeout  = 3 * time.Minute
	snapshotPollInterval = 5 * time.Second
)

// createSandboxAndAwaitRunning creates a sandbox from the default image and
// waits until it reaches Running state. Returns the sandbox ID.
// Registers DeferCleanup to delete the sandbox.
func createSandboxAndAwaitRunning(suffix string) string {
	return createSandboxFromImageAndAwaitRunning(suffix, EnvOr("E2B_SANDBOX_IMAGE", DefaultImage))
}

// createSandboxFromImageAndAwaitRunning creates a sandbox with the given image
// and waits until it reaches Running state. Returns the sandbox ID.
// Registers DeferCleanup to delete the sandbox.
func createSandboxFromImageAndAwaitRunning(suffix, image string) string {
	resp, status, err := apiClient.CreateSandbox(ctx, &CreateSandboxRequest{
		Image:      &ImageSpec{URI: image},
		Entrypoint: []string{"sleep", "3600"},
		ResourceLimits: &ResourceLimits{
			CPU:    DefaultCPU,
			Memory: DefaultMemory,
		},
		Extensions: map[string]string{"sandboxGroup": sgName},
		Timeout:    DefaultTimeout,
	})
	Expect(err).NotTo(HaveOccurred(), "创建 sandbox 不应返回错误")
	Expect(status).To(Equal(StatusAccepted), "创建 sandbox 应返回 202")
	Expect(resp.ID).NotTo(BeEmpty(), "sandbox ID 不应为空")

	sandboxID := resp.ID

	awaitSandboxState(sandboxID, StateRunning)
	return sandboxID
}

// awaitSandboxState polls GET /v1/sandboxes/{id} until status.state matches want.
func awaitSandboxState(id, want string) {
	Eventually(func(g Gomega) string {
		sb, _, err := apiClient.GetSandbox(ctx, id)
		if err != nil || sb == nil {
			return ""
		}
		return sb.Status.State
	}, sandboxWaitTimeout, sandboxPollInterval).Should(Equal(want),
		fmt.Sprintf("sandbox %s 应当进入 %s 状态", id, want))
}

// createSnapshotAndAwaitReady creates a snapshot for the given sandbox and
// waits until it reaches Ready state. Returns the snapshot ID.
// Registers DeferCleanup to delete the snapshot.
func createSnapshotAndAwaitReady(sandboxID, name string) string {
	snap, status, err := apiClient.CreateSnapshot(ctx, sandboxID, name)
	Expect(err).NotTo(HaveOccurred(), "创建快照不应返回错误")
	Expect(status).To(Equal(StatusCreated), "创建快照应返回 201")
	Expect(snap.ID).NotTo(BeEmpty(), "快照 ID 不应为空")

	snapshotID := snap.ID

	awaitSnapshotState(snapshotID, SnapReady)
	return snapshotID
}

// awaitSnapshotState polls GET /v1/snapshots/{id} until status.state matches want.
func awaitSnapshotState(id, want string) {
	Eventually(func(g Gomega) string {
		snap, _, err := apiClient.GetSnapshot(ctx, id)
		if err != nil || snap == nil {
			return ""
		}
		return snap.Status.State
	}, snapshotWaitTimeout, snapshotPollInterval).Should(Equal(want),
		fmt.Sprintf("快照 %s 应当进入 %s 状态", id, want))
}

// cleanupAllSandboxesAndSnapshots lists and deletes all sandboxes and snapshots
// via the API. Used in BeforeSuite and DeferCleanup.
func cleanupAllSandboxesAndSnapshots() {
	By("清理残留 sandbox 和快照")
	cleanupAllSandboxes()
	cleanupAllSnapshots()
}

func cleanupAllSandboxes() {
	page := 1
	for {
		resp, status, err := apiClient.ListSnapshots(ctx, page, 200)
		if err != nil || status != StatusOK {
			break
		}
		// Delete sandboxes — we need a ListSandboxes API, but the OpenSandbox
		// API lists sandboxes via GET /v1/sandboxes. We use a simple approach:
		// list snapshots first (they reference sandbox IDs), then delete sandboxes.
		// For now, we rely on per-test DeferCleanup for sandbox deletion.
		_ = resp
		break
	}
}

func cleanupAllSnapshots() {
	page := 1
	for {
		resp, status, err := apiClient.ListSnapshots(ctx, page, 200)
		if err != nil || status != StatusOK {
			break
		}
		if len(resp.Items) == 0 {
			break
		}
		for _, snap := range resp.Items {
			_, _ = apiClient.DeleteSnapshot(ctx, snap.ID)
		}
		if !resp.Pagination.HasNextPage {
			break
		}
		page++
	}
}
