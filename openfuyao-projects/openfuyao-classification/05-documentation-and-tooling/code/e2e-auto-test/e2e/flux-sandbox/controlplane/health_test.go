/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package controlplane

import (
	ginkgo "github.com/onsi/ginkgo/v2"

	"encoding/json"
	"io"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// ── Scenario group: Health checks and readiness gates ──

// TestHealthz_Controller verifies FS-FUNC-028: Controller /healthz returns
// {"ok":true} and /readyz returns {"ready":true}.
func testHealthz_Controller(t ginkgo.GinkgoTInterface) {
	code, body := controllerHealthz(t)
	if code != 200 {
		t.Fatalf("/healthz: expected 200, got %d", code)
	}
	var healthz struct {
		OK bool `json:"ok"`
	}
	if err := json.Unmarshal([]byte(body), &healthz); err != nil {
		t.Fatalf("/healthz: unmarshal %q: %v", body, err)
	}
	if !healthz.OK {
		t.Fatalf("/healthz: expected ok=true, got %s", body)
	}

	code, body = controllerReadyz(t)
	if code != 200 {
		t.Fatalf("/readyz: expected 200, got %d", code)
	}
	var readyz struct {
		Ready bool `json:"ready"`
	}
	if err := json.Unmarshal([]byte(body), &readyz); err != nil {
		t.Fatalf("/readyz: unmarshal %q: %v", body, err)
	}
	if !readyz.Ready {
		t.Fatalf("/readyz: expected ready=true, got %s", body)
	}
}

// TestHealthz_Scheduler verifies FS-FUNC-029: a Scheduler's /healthz and
// /readyz both return 200.
func testHealthz_Scheduler(t ginkgo.GinkgoTInterface) {
	pods, err := schedulerPods()
	if err != nil || len(pods) == 0 {
		t.Skipf("no scheduler pods to probe: %v", err)
	}
	pod := pods[0]

	// Port-forward the scheduler pod's 9090 to a local port.
	const localPort = "19090"
	cmd := exec.Command("kubectl", "port-forward", "-n", cpNamespace, pod.Name, localPort+":9090")
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		t.Fatalf("start scheduler port-forward: %v", err)
	}
	t.Cleanup(func() {
		if cmd.Process != nil {
			if err := cmd.Process.Kill(); err != nil {
				t.Logf("kill scheduler port-forward: %v", err)
			}
			if err := cmd.Wait(); err != nil {
				t.Logf("reap scheduler port-forward: %v", err)
			}
		}
	})

	base := "http://127.0.0.1:" + localPort
	waitFor(t, "scheduler port-forward reachable", 30*time.Second, func() bool {
		code, _ := httpGetSilent(base + "/healthz")
		return code == 200
	})

	if code, _ := httpGetSilent(base + "/healthz"); code != 200 {
		t.Errorf("scheduler /healthz: expected 200, got %d", code)
	}
	if code, _ := httpGetSilent(base + "/readyz"); code != 200 {
		t.Errorf("scheduler /readyz: expected 200, got %d", code)
	}
}

// httpGetSilent performs a GET and returns (statusCode, body) without using
// t.Fatalf (used inside polling loops).
func httpGetSilent(url string) (int, string) {
	resp, err := http.Get(url)
	if err != nil {
		return 0, ""
	}
	defer resp.Body.Close()
	b, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		return resp.StatusCode, ""
	}
	return resp.StatusCode, string(b)
}

// TestReadyz_Gate verifies FS-FUNC-030: when a Scheduler Pod goes away, the
// Controller's Router stops routing new requests to it. The implementation
// removes the scheduler's endpoint from the router when its Pod is deleted
// (SchedulerPodReconciler deletes instead of keeping a ready=false entry), so
// the case accepts either the endpoint disappearing from the snapshot or being
// marked ready=false — the invariant under test is that no ready=true endpoint
// for the removed scheduler remains.
//
// This is a destructive case: it scales the scheduler StatefulSet down. It
// requires E2E_DESTRUCTIVE and restores the replica count in cleanup.
func testReadyz_Gate(t ginkgo.GinkgoTInterface) {
	requireDestructive(t)
	requireSchedulerReplicas(t, 2)

	// Capture the original replica count to restore in cleanup.
	origReplicas, err := getSchedReplicas()
	if err != nil {
		t.Fatalf("get scheduler replicas: %v", err)
	}
	t.Cleanup(func() {
		if err := exec.Command("kubectl", "scale", "sts", "flux-sandbox-scheduler",
			"-n", cpNamespace, "--replicas="+itoa(origReplicas)).Run(); err != nil {
			t.Logf("restore scheduler replicas to %d: %v", origReplicas, err)
		}
		// Give the controller time to observe the restored schedulers.
		time.Sleep(30 * time.Second)
	})

	// Identify scheduler-1 (the one to remove).
	pods, err := schedulerPods()
	if err != nil || len(pods) < 2 {
		t.Skipf("need >=2 scheduler pods, got %d", len(pods))
	}
	target := ""
	for i := range pods {
		if strings.HasSuffix(pods[i].Name, "-1") {
			target = pods[i].Name
			break
		}
	}
	if target == "" {
		t.Skip("could not find scheduler pod ending in -1")
	}

	// Scale down to remove scheduler-1.
	if out, err := exec.Command("kubectl", "scale", "sts", "flux-sandbox-scheduler",
		"-n", cpNamespace, "--replicas=1").CombinedOutput(); err != nil {
		t.Fatalf("scale down scheduler: %v: %s", err, out)
	}

	// Wait for the controller to stop routing scheduler-1 in the router
	// snapshot: its entry is either gone or marked ready=false.
	waitFor(t, "router stops routing scheduler-1 (gone or ready=false)", 120*time.Second, func() bool {
		snap := routerSnapshot(t)
		endpoints, _ := snap["schedulerEndpoints"].(map[string]any)
		for _, v := range endpoints {
			ep, _ := v.(map[string]any)
			name, _ := ep["scheduler"].(string)
			if !strings.HasSuffix(name, "-1") {
				continue
			}
			if ready, _ := ep["ready"].(bool); ready {
				return false
			}
		}
		return true
	})
}

// getSchedReplicas returns the scheduler StatefulSet's current replica count.
func getSchedReplicas() (int, error) {
	out, err := exec.Command("kubectl", "get", "sts", "flux-sandbox-scheduler",
		"-n", cpNamespace, "-o", "jsonpath={.spec.replicas}").Output()
	if err != nil {
		return 0, err
	}
	return atoiSafe(strings.TrimSpace(string(out)))
}

func itoa(n int) string {
	return strconv.Itoa(n)
}

func atoiSafe(s string) (int, error) {
	if s == "" {
		return 0, nil
	}
	return strconv.Atoi(s)
}
