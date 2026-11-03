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
	"fmt"
	"net/http"
	"time"

	ginkgo "github.com/onsi/ginkgo/v2"
	gomega "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// wDestructive gates specs that stop dependencies (WARM_E2E_DESTRUCTIVE).
// It is on by default unless disabled explicitly ("0"/"false"/"no"/"off").
var wDestructive = gateEnabled("WARM_E2E_DESTRUCTIVE")

// scaleDeploy scales a suite dependency Deployment to 0 and back.
func (e *warmEnv) scaleDeploy(tb ginkgo.GinkgoTInterface, name string, replicas int32) {
	tb.Helper()
	repl, err := e.kube.AppsV1().Deployments(wNamespace).GetScale(e.ctx, name, metav1.GetOptions{})
	if err != nil {
		tb.Fatalf("get scale %s: %v", name, err)
	}
	repl.Spec.Replicas = replicas
	if _, err := e.kube.AppsV1().Deployments(wNamespace).UpdateScale(e.ctx, name, repl, metav1.UpdateOptions{}); err != nil {
		tb.Fatalf("update scale %s: %v", name, err)
	}
}

// waitForDeployScale waits until the Deployment reports the wanted replicas.
func (e *warmEnv) waitForDeployScale(tb ginkgo.GinkgoTInterface, name string, replicas int32) {
	tb.Helper()
	waitFor(tb, fmt.Sprintf("deployment %s at %d replicas", name, replicas), 3*time.Minute, func() bool {
		d, err := e.kube.AppsV1().Deployments(wNamespace).Get(e.ctx, name, metav1.GetOptions{})
		if err != nil {
			return false
		}
		return d.Status.Replicas == replicas && d.Status.ReadyReplicas == replicas
	})
}

// ── Feature group: dependency outage behavior ──

var _ = ginkgo.Describe("Warm Dependency Outage", ginkgo.Label("flux-sandbox", "outage"), func() {

	ginkgo.It("returns 503 on PG-dependent endpoints when PG is down (WARM-E2E-025)", func() {
		if !wDestructive {
			ginkgo.Skip("WARM_E2E_DESTRUCTIVE gate disabled; dependency-outage specs skipped")
		}
		tb := ginkgo.GinkgoT()

		// Stop PostgreSQL.
		wEnv.scaleDeploy(tb, "warm-postgres", 0)
		ginkgo.DeferCleanup(func() {
			wEnv.scaleDeploy(tb, "warm-postgres", 1)
			wEnv.waitForDeployScale(tb, "warm-postgres", 1)
			// The old PG port-forward points at the deleted pod; re-establish
			// it so subsequent specs can reach the database again.
			wEnv.restartPGPortForward(tb)
			// Give the controller pool time to notice the recovery.
			time.Sleep(10 * time.Second)
		})
		wEnv.waitForDeployScale(tb, "warm-postgres", 0)

		// Create: 503 failed to persist task, retryable, no fake task_id.
		resp, raw := wEnv.httpDo(tb, http.MethodPost, "/api/v1/snapshot-warm-tasks",
			map[string]any{"snapshot_ids": []string{"snapshot-pgdown-001"}})
		gomega.Expect(resp.StatusCode).To(gomega.Equal(503), "body %s", raw)
		apiErr := decodeError(tb, raw)
		gomega.Expect(apiErr.Code).To(gomega.Equal("UNAVAILABLE"))
		gomega.Expect(apiErr.Message).To(gomega.Equal("failed to persist task"))
		gomega.Expect(apiErr.Retryable).To(gomega.BeTrue())
		gomega.Expect(string(raw)).NotTo(gomega.ContainSubstring("task_id"))

		// Progress and cache-details: 503 database unavailable.
		for _, path := range []string{
			"/api/v1/snapshot-warm-tasks/warm-task-not-exist",
			"/api/v1/snapshots/cache-details",
		} {
			resp, raw := wEnv.httpDo(tb, http.MethodGet, path, nil)
			gomega.Expect(resp.StatusCode).To(gomega.Equal(503), "GET %s: %s", path, raw)
			gomega.Expect(decodeError(tb, raw).Message).To(gomega.Equal("database unavailable"))
		}

		// Storage-details only reads etcd: still 200.
		_, c, r := wEnv.getStorageDetails(tb)
		gomega.Expect(c).To(gomega.Equal(200), "body %s", r)

		// Restore PostgreSQL: task creation works again.
		wEnv.scaleDeploy(tb, "warm-postgres", 1)
		wEnv.waitForDeployScale(tb, "warm-postgres", 1)
		waitFor(tb, "create works again after PG recovery", 60*time.Second, func() bool {
			resp, _ := wEnv.httpDo(tb, http.MethodPost, "/api/v1/snapshot-warm-tasks",
				map[string]any{"snapshot_ids": []string{"snapshot-pgdown-001"}})
			return resp.StatusCode == 201
		})
	})

	ginkgo.It("returns 503 on storage-details when etcd is down (WARM-E2E-026)", func() {
		if !wDestructive {
			ginkgo.Skip("WARM_E2E_DESTRUCTIVE gate disabled; dependency-outage specs skipped")
		}
		tb := ginkgo.GinkgoT()

		// Stop MoonCake etcd.
		wEnv.scaleDeploy(tb, "warm-etcd", 0)
		ginkgo.DeferCleanup(func() {
			wEnv.scaleDeploy(tb, "warm-etcd", 1)
			wEnv.waitForDeployScale(tb, "warm-etcd", 1)
			// The old etcd port-forward and client point at the deleted pod.
			wEnv.reconnectEtcd(tb)
			time.Sleep(10 * time.Second)
		})
		wEnv.waitForDeployScale(tb, "warm-etcd", 0)

		// Storage-details: 503 mooncake etcd unavailable, retryable.
		resp, raw := wEnv.httpDo(tb, http.MethodGet, "/api/v1/superpods/storage-details", nil)
		gomega.Expect(resp.StatusCode).To(gomega.Equal(503), "body %s", raw)
		apiErr := decodeError(tb, raw)
		gomega.Expect(apiErr.Code).To(gomega.Equal("UNAVAILABLE"))
		gomega.Expect(apiErr.Message).To(gomega.Equal("mooncake etcd unavailable"))
		gomega.Expect(apiErr.Retryable).To(gomega.BeTrue())

		// PG-backed endpoints are unaffected: create 201, progress and
		// cache-details 200. Note the task itself may lag or fail while
		// etcd is down (the load balancer reads the capacity view from
		// etcd) - the outage spec only asserts enqueue/query isolation.
		created, c, r := wEnv.createWarmTask(tb, []string{"snapshot-etcd-001"})
		gomega.Expect(c).To(gomega.Equal(201), "body %s", r)

		_, c2, r2 := wEnv.getProgress(tb, created.TaskID, "")
		gomega.Expect(c2).To(gomega.Equal(200), "body %s", r2)
		_, c3, r3 := wEnv.getCacheDetails(tb, "")
		gomega.Expect(c3).To(gomega.Equal(200), "body %s", r3)

		// Restore etcd: storage-details recovers to 200 with per-pod
		// freshness-based health.
		wEnv.scaleDeploy(tb, "warm-etcd", 1)
		wEnv.waitForDeployScale(tb, "warm-etcd", 1)
		waitFor(tb, "storage-details recovers", 60*time.Second, func() bool {
			_, c, _ := wEnv.getStorageDetails(tb)
			return c == 200
		})
	})
})
