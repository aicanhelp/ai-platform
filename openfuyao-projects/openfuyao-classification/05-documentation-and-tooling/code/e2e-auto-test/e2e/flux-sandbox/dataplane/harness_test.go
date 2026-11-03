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
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	ginkgo "github.com/onsi/ginkgo/v2"

	agentapi "gitcode.com/openFuyao/e2e-auto-test/e2e/flux-sandbox/agentapi"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/flux-sandbox/framework"
)

// env is the shared e2e environment. It is a value (never a nil pointer)
// populated by BeforeSuite via framework.Setup; specs only run afterwards.
var env framework.Env

// ── Ginkgo-facing agent API helpers ──

// sbID derives a unique sandbox ID from the current spec name.
func sbID(tb ginkgo.GinkgoTInterface) string {
	tb.Helper()
	return framework.NewID(tb.Name())
}

// createSandbox calls POST /create and fails the spec on transport errors.
func createSandbox(tb ginkgo.GinkgoTInterface, spec agentapi.SandboxSpec) (agentapi.CreateSandboxResponse, int) {
	tb.Helper()
	resp, code, err := env.Agent.Create(spec)
	if err != nil {
		tb.Fatalf("create request failed: %v", err)
	}
	return resp, code
}

// createSandboxRaw posts an arbitrary body to /create (malformed-payload tests).
func createSandboxRaw(tb ginkgo.GinkgoTInterface, rawBody string) (int, string) {
	tb.Helper()
	code, body, err := env.Agent.CreateRaw(rawBody)
	if err != nil {
		tb.Fatalf("raw create request failed: %v", err)
	}
	return code, body
}

// deleteSandbox calls POST /delete.
func deleteSandbox(tb ginkgo.GinkgoTInterface, sandboxID string) (agentapi.DeleteSandboxResponse, int) {
	tb.Helper()
	resp, code, err := env.Agent.Delete(sandboxID)
	if err != nil {
		tb.Fatalf("delete request failed: %v", err)
	}
	return resp, code
}

// getStatus calls GET /status.
func getStatus(tb ginkgo.GinkgoTInterface) (agentapi.FluxSandboxAgentStatus, int) {
	tb.Helper()
	resp, code, err := env.Agent.Status()
	if err != nil {
		tb.Fatalf("status request failed: %v", err)
	}
	return resp, code
}

// getLogs fetches the sandbox logs (no follow, no tail).
func getLogs(tb ginkgo.GinkgoTInterface, sandboxID string) (int, string) {
	tb.Helper()
	code, body, err := env.Agent.Logs(sandboxID, false, 0, 0)
	if err != nil {
		tb.Fatalf("logs request failed: %v", err)
	}
	return code, body
}

// getLogsWithOpts fetches logs with follow/tail options, reading a follow
// stream for readFor at most.
func getLogsWithOpts(tb ginkgo.GinkgoTInterface, sandboxID string, follow bool, tail int, readFor time.Duration) (int, string) {
	tb.Helper()
	code, body, err := env.Agent.Logs(sandboxID, follow, tail, readFor)
	if err != nil {
		tb.Fatalf("logs request failed: %v", err)
	}
	return code, body
}

// sandboxStatus returns the current status entry for sandboxID, if present.
func sandboxStatus(tb ginkgo.GinkgoTInterface, sandboxID string) (agentapi.SandboxStatus, bool) {
	tb.Helper()
	status, _ := getStatus(tb)
	for _, s := range status.SandboxStatuses {
		if s.SandboxID == sandboxID {
			return s, true
		}
	}
	return agentapi.SandboxStatus{}, false
}

// statusSandboxIDs returns all sandbox IDs reported by the agent.
func statusSandboxIDs() []string {
	status, _, err := env.Agent.Status()
	if err != nil {
		return nil
	}
	var ids []string
	for _, s := range status.SandboxStatuses {
		ids = append(ids, s.SandboxID)
	}
	return ids
}

// ── Polling wrappers (fail the spec on timeout) ──

// waitFor polls until check holds or timeout, failing the spec otherwise.
func waitFor(tb ginkgo.GinkgoTInterface, desc string, timeout time.Duration, check func() bool) {
	tb.Helper()
	if err := framework.WaitFor(desc, timeout, check); err != nil {
		tb.Fatal(err.Error())
	}
}

// waitForNot asserts check stays false for window, failing the spec otherwise.
func waitForNot(tb ginkgo.GinkgoTInterface, desc string, window time.Duration, check func() bool) {
	tb.Helper()
	if err := framework.WaitForNot(desc, window, check); err != nil {
		tb.Fatal(err.Error())
	}
}

// ── containerd node-level queries ──

// containerExists reports whether a container with the exact sandbox ID
// exists on the agent node.
func containerExists(tb ginkgo.GinkgoTInterface, sandboxID string) bool {
	tb.Helper()
	return containerCount(tb, sandboxID) > 0
}

// containerCount returns how many containerd containers exactly match the
// sandbox ID (used to prove idempotency does not duplicate).
func containerCount(tb ginkgo.GinkgoTInterface, sandboxID string) int {
	tb.Helper()
	out, err := env.Node.Ctr("containers", "ls", "-q")
	if err != nil {
		tb.Fatalf("ctr containers ls: %v", err)
	}
	count := 0
	for _, line := range strings.Split(out, "\n") {
		if strings.TrimSpace(line) == sandboxID {
			count++
		}
	}
	return count
}

// containerLabels returns the labels of the sandbox container.
func containerLabels(tb ginkgo.GinkgoTInterface, sandboxID string) map[string]string {
	tb.Helper()
	out, err := env.Node.Ctr("containers", "info", sandboxID, "--format", "{{json .}}")
	if err != nil {
		tb.Fatalf("ctr containers info %s: %v", sandboxID, err)
	}
	var raw struct {
		Labels map[string]string `json:"Labels"`
	}
	if err := json.Unmarshal([]byte(out), &raw); err != nil {
		tb.Fatalf("unmarshal container info: %v (raw head: %.200s)", err, out)
	}
	return raw.Labels
}

// snapshotExists reports whether the sandbox snapshot exists. containerd v2
// has no `snapshots ls -q`, so the table output is parsed.
func snapshotExists(tb ginkgo.GinkgoTInterface, sandboxID string) bool {
	tb.Helper()
	out, err := env.Node.Ctr("snapshots", "ls")
	if err != nil {
		tb.Fatalf("ctr snapshots ls: %v", err)
	}
	target := sandboxID + "-snapshot"
	for _, line := range strings.Split(out, "\n") {
		fields := strings.Fields(line)
		if len(fields) > 0 && fields[0] == target {
			return true
		}
	}
	return false
}

// logFileExists reports whether the sandbox log file exists on the node.
func logFileExists(tb ginkgo.GinkgoTInterface, sandboxID string) bool {
	tb.Helper()
	return env.Node.PathExists("/var/log/flux-sandbox/" + sandboxID + ".log")
}

// nodeFileGlobExists reports whether any node file matches the glob pattern.
func nodeFileGlobExists(tb ginkgo.GinkgoTInterface, pattern string) bool {
	tb.Helper()
	return env.Node.GlobExists(pattern)
}

// getContainerPID returns the host PID of the sandbox task init process
// (containerd v2 removed `ctr task info`, so `task ps` is parsed).
func getContainerPID(tb ginkgo.GinkgoTInterface, sandboxID string) int {
	tb.Helper()
	out, err := env.Node.Ctr("task", "ps", sandboxID)
	if err != nil {
		tb.Fatalf("ctr task ps %s: %v", sandboxID, err)
	}
	for _, line := range strings.Split(out, "\n") {
		fields := strings.Fields(line)
		if len(fields) > 0 && fields[0] != "PID" {
			if pid, perr := strconv.Atoi(fields[0]); perr == nil {
				return pid
			}
		}
	}
	tb.Fatalf("no PID found in ctr task ps output for %s: %q", sandboxID, out)
	return 0
}

// updateContainerLabel overwrites a container label via ctr.
func updateContainerLabel(tb ginkgo.GinkgoTInterface, sandboxID, key, val string) {
	tb.Helper()
	if _, err := env.Node.Ctr("containers", "label", sandboxID, key+"="+val); err != nil {
		tb.Fatalf("ctr containers label %s: %v", sandboxID, err)
	}
}

// ── Sandbox CR helpers ──

// applySandboxCR creates the Sandbox CR named after the sandbox.
func applySandboxCR(tb ginkgo.GinkgoTInterface, name string) {
	tb.Helper()
	if _, err := env.K8s.ApplySandboxCR(name); err != nil {
		tb.Fatalf("apply Sandbox CR %s: %v", name, err)
	}
}

// deleteSandboxCR deletes the Sandbox CR.
func deleteSandboxCR(tb ginkgo.GinkgoTInterface, name string) {
	tb.Helper()
	if err := env.K8s.DeleteSandboxCR(name); err != nil {
		tb.Fatalf("delete Sandbox CR %s: %v", name, err)
	}
}

// ── Lifecycle helpers ──

// forceDeleteContainer forcibly removes the sandbox container, task, and
// snapshot on the node, ignoring failures.
func forceDeleteContainer(sandboxID string) {
	env.Node.CtrBestEffort("task", "kill", sandboxID)
	env.Node.CtrBestEffort("task", "delete", sandboxID)
	env.Node.CtrBestEffort("containers", "delete", sandboxID)
	env.Node.CtrBestEffort("snapshots", "rm", "--", sandboxID+"-snapshot")
}

// cleanupSandbox is the standard DeferCleanup: API delete + CR delete +
// best-effort containerd removal.
func cleanupSandbox(tb ginkgo.GinkgoTInterface, sandboxID string) {
	tb.Helper()
	_, _, _ = env.Agent.Delete(sandboxID)
	_ = env.K8s.DeleteSandboxCR(sandboxID)
	forceDeleteContainer(sandboxID)
}

// createSandboxWithCR creates the Sandbox CR and the sandbox itself.
func createSandboxWithCR(tb ginkgo.GinkgoTInterface, sandboxID string) {
	tb.Helper()
	applySandboxCR(tb, sandboxID)
	resp, code := createSandbox(tb, framework.BasicSpec(sandboxID))
	if code != 200 || !resp.Success {
		tb.Fatalf("create sandbox %s with CR: code=%d resp=%+v", sandboxID, code, resp)
	}
}

// createSandboxNoCR creates only the sandbox (no CR), for orphan scenarios.
// It waits for any same-named leftover to fully disappear first, otherwise
// a leftover task causes "task already exists" errors.
func createSandboxNoCR(tb ginkgo.GinkgoTInterface, sandboxID string) {
	tb.Helper()
	_ = env.K8s.DeleteSandboxCR(sandboxID)
	_, _, _ = env.Agent.Delete(sandboxID)
	forceDeleteContainer(sandboxID)
	waitForSandboxGone(tb, sandboxID)

	resp, code := createSandbox(tb, framework.BasicSpec(sandboxID))
	if code != 200 || !resp.Success {
		tb.Fatalf("create sandbox %s without CR: code=%d resp=%+v", sandboxID, code, resp)
	}
}

// waitForSandboxGone polls agent status until the sandbox ID disappears.
func waitForSandboxGone(tb ginkgo.GinkgoTInterface, sandboxID string) {
	tb.Helper()
	waitFor(tb, "sandbox "+sandboxID+" gone from agent status", 30*time.Second, func() bool {
		_, ok := sandboxStatus(tb, sandboxID)
		return !ok
	})
}

// waitForSandboxPhase polls agent status until the sandbox reaches phase.
func waitForSandboxPhase(tb ginkgo.GinkgoTInterface, sandboxID, phase string, timeout time.Duration) {
	tb.Helper()
	waitFor(tb, "sandbox "+sandboxID+" phase="+phase, timeout, func() bool {
		s, ok := sandboxStatus(tb, sandboxID)
		return ok && s.Phase == phase
	})
}

// waitForContainer polls until the sandbox container exists in containerd.
func waitForContainer(tb ginkgo.GinkgoTInterface, sandboxID string, timeout time.Duration) {
	tb.Helper()
	waitFor(tb, "container "+sandboxID+" exists in containerd", timeout, func() bool {
		return containerExists(tb, sandboxID)
	})
}

// waitForContainerGone polls until the container disappears from containerd.
func waitForContainerGone(tb ginkgo.GinkgoTInterface, sandboxID string, timeout time.Duration) {
	tb.Helper()
	waitFor(tb, "container "+sandboxID+" gone from containerd", timeout, func() bool {
		return !containerExists(tb, sandboxID)
	})
}

// restartAgentPod deletes the agent pod and re-attaches (new identity,
// restarted port-forward, reconcile happens on startup).
func restartAgentPod(tb ginkgo.GinkgoTInterface) {
	tb.Helper()
	if err := env.RestartAgentPod(); err != nil {
		tb.Fatalf("restart agent pod: %v", err)
	}
}

// createRestartAndWaitReconcile creates the sandbox, waits for its container,
// restarts the agent pod, and waits for reconcile to re-admit the sandbox
// into status (shared by the DP-E2E-026/027 specs).
func createRestartAndWaitReconcile(tb ginkgo.GinkgoTInterface, id string) {
	tb.Helper()
	resp, code := createSandbox(tb, framework.BasicSpec(id))
	if code != 200 || !resp.Success {
		tb.Fatalf("create sandbox %s: code=%d resp=%+v", id, code, resp)
	}
	waitForContainer(tb, id, 30*time.Second)
	restartAgentPod(tb)
	waitFor(tb, "sandbox "+id+" reconciled after restart", 60*time.Second, func() bool {
		_, ok := sandboxStatus(tb, id)
		return ok
	})
}

// smallCapacityAgent returns a client for the independently deployed
// capacity=2 agent, or nil when E2E_SMALL_CAPACITY_AGENT_URL is unset.
func smallCapacityAgent() *framework.AgentClient {
	if env.Cfg == nil || env.Cfg.SmallCapacityAgentURL == "" {
		return nil
	}
	return framework.NewAgentClient(env.Cfg.SmallCapacityAgentURL, env.Cfg.AgentAuthToken)
}

// createOnAgent creates n sandboxes on the given agent with IDs derived from
// the current spec name and registers deletion cleanup (shared by the
// DP-E2E-005/029 specs).
func createOnAgent(tb ginkgo.GinkgoTInterface, agent *framework.AgentClient, n int) []string {
	tb.Helper()
	var ids []string
	for i := 0; i < n; i++ {
		id := framework.NewID(tb.Name()) + "-" + string(rune('a'+i))
		resp, code, err := agent.Create(framework.BasicSpec(id))
		if err != nil {
			tb.Fatalf("create %s on %s: %v", id, agent.BaseURL, err)
		}
		if code != 200 || !resp.Success {
			tb.Fatalf("create %s on %s: code=%d resp=%+v", id, agent.BaseURL, code, resp)
		}
		ids = append(ids, id)
	}
	ginkgo.DeferCleanup(func() {
		for _, id := range ids {
			_, _, _ = agent.Delete(id)
		}
	})
	return ids
}

// currentAgentPodIP returns the live pod IP of the agent pod.
func currentAgentPodIP(tb ginkgo.GinkgoTInterface) string {
	tb.Helper()
	pod, err := env.K8s.GetPod(env.AgentPodName)
	if err != nil {
		tb.Fatalf("get agent pod %s: %v", env.AgentPodName, err)
	}
	return pod.Status.PodIP
}

// skipIfMissing skips the spec when the env var is unset.
func skipIfMissing(tb ginkgo.GinkgoTInterface, envKey, reason string) {
	tb.Helper()
	if os.Getenv(envKey) != "" {
		return
	}
	tb.Skipf("skipped: set %s to enable (%s)", envKey, reason)
}

// ── Suite-level cleanup ──

// cleanSlate removes every sandbox the agent reports (API delete + node
// force delete) before the first spec. If entries remain stuck, the agent
// pod is restarted so the new process reconciles a fresh registry.
func cleanSlate() {
	for i := 0; i < 60; i++ {
		ids := statusSandboxIDs()
		if len(ids) == 0 {
			return
		}
		for _, id := range ids {
			_, _, _ = env.Agent.Delete(id)
			forceDeleteContainer(id)
		}
		time.Sleep(2 * time.Second)
	}

	log.Printf("cleanSlate: restarting agent pod to clear stale registry entries")
	if err := env.RestartAgentPod(); err != nil {
		log.Printf("cleanSlate: agent pod restart failed: %v", err)
		return
	}
	for i := 0; i < 30; i++ {
		if len(statusSandboxIDs()) == 0 {
			return
		}
		time.Sleep(2 * time.Second)
	}
	log.Printf("WARN: cleanSlate could not fully drain the agent registry")
}

// fmtSandbox returns a compact debug representation of a status entry.
func fmtSandbox(s agentapi.SandboxStatus) string {
	return fmt.Sprintf("{id:%s phase:%s msg:%q exit:%d}", s.SandboxID, s.Phase, s.Message, s.ExitCode)
}
