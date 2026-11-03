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
	"fmt"
	"strconv"
	"strings"
	"time"

	ginkgo "github.com/onsi/ginkgo/v2"
	gomega "github.com/onsi/gomega"

	agentapi "gitcode.com/openFuyao/e2e-auto-test/e2e/flux-sandbox/agentapi"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/flux-sandbox/framework"
)

// ── Feature group: Sandbox lifecycle (flux-sandbox-agent + containerd) ──

var _ = ginkgo.Describe("Sandbox Lifecycle", ginkgo.Label("flux-sandbox", "lifecycle"), func() {

	ginkgo.Describe("basic lifecycle", func() {
		ginkgo.It("create → running → delete (DP-E2E-014)", func() {
			t := ginkgo.GinkgoT()
			id := sbID(t)

			resp, code := createSandbox(t, framework.BasicSpec(id))
			gomega.Expect(code).To(gomega.Equal(200), "create: %+v", resp)
			gomega.Expect(resp.Success).To(gomega.BeTrue())

			waitForSandboxPhase(t, id, "running", 30*time.Second)
			gomega.Expect(containerExists(t, id)).To(gomega.BeTrue())

			delResp, delCode := deleteSandbox(t, id)
			gomega.Expect(delCode).To(gomega.Equal(200))
			gomega.Expect(delResp.Success).To(gomega.BeTrue())

			waitForSandboxGone(t, id)
			waitForContainerGone(t, id, 30*time.Second)
		})

		ginkgo.It("containers carry the full label set (DP-E2E-015)", func() {
			t := ginkgo.GinkgoT()
			id := sbID(t)
			ginkgo.DeferCleanup(cleanupSandbox, t, id)

			resp, code := createSandbox(t, framework.BasicSpec(id))
			gomega.Expect(code).To(gomega.Equal(200), "create: %+v", resp)

			waitForContainer(t, id, 30*time.Second)

			labels := containerLabels(t, id)
			requiredLabels := map[string]string{
				"flux-sandbox.io/runtime":      "containerd",
				"flux-sandbox.io/sandbox-id":   id,
				"flux-sandbox.io/sandbox-name": id,
				"flux-sandbox.io/agent-uid":    env.AgentPodUID,
				"flux-sandbox.io/agent-name":   env.AgentPodName,
				"flux-sandbox.io/namespace":    env.Cfg.Namespace,
				"flux-sandbox.io/image":        framework.DefaultImage,
			}
			for key, want := range requiredLabels {
				got, ok := labels[key]
				gomega.Expect(ok).To(gomega.BeTrue(), "label %s missing on container %s; got %v", key, id, labels)
				gomega.Expect(got).To(gomega.Equal(want), "label %s mismatch", key)
			}
		})
	})

	ginkgo.Context("when spec fields are set", func() {
		ginkgo.It("injects env vars into the sandbox (DP-E2E-016)", func() {
			t := ginkgo.GinkgoT()
			id := sbID(t)
			ginkgo.DeferCleanup(cleanupSandbox, t, id)

			spec := framework.BasicSpec(id)
			spec.Command = []string{"/bin/sh", "-c", "echo $MY_VAR; sleep 3600"}
			spec.Env = map[string]string{"MY_VAR": "hello"}
			resp, code := createSandbox(t, spec)
			gomega.Expect(code).To(gomega.Equal(200), "create: %+v", resp)

			waitFor(t, "env var value in logs", 30*time.Second, func() bool {
				_, body := getLogs(t, id)
				return strings.Contains(body, "hello")
			})
		})

		ginkgo.It("passes args alongside the command (DP-E2E-017)", func() {
			t := ginkgo.GinkgoT()
			id := sbID(t)
			ginkgo.DeferCleanup(cleanupSandbox, t, id)

			spec := framework.BasicSpec(id)
			spec.Command = []string{"/bin/sh"}
			spec.Args = []string{"-c", "echo args-ok; sleep 3600"}
			resp, code := createSandbox(t, spec)
			gomega.Expect(code).To(gomega.Equal(200), "create: %+v", resp)

			waitFor(t, "args output in logs", 30*time.Second, func() bool {
				_, body := getLogs(t, id)
				return strings.Contains(body, "args-ok")
			})
		})

		ginkgo.It("respects workingDir (DP-E2E-018)", func() {
			t := ginkgo.GinkgoT()
			id := sbID(t)
			ginkgo.DeferCleanup(cleanupSandbox, t, id)

			workDir := "/tmp/e2e-wd-" + id
			spec := framework.BasicSpec(id)
			spec.Command = []string{"/bin/sh", "-c", "pwd; sleep 3600"}
			spec.WorkingDir = workDir
			resp, code := createSandbox(t, spec)
			gomega.Expect(code).To(gomega.Equal(200), "create: %+v", resp)

			waitFor(t, "pwd output in logs", 30*time.Second, func() bool {
				_, body := getLogs(t, id)
				return strings.Contains(body, workDir)
			})
		})

		ginkgo.It("bind-mounts host volumeMounts (DP-E2E-019)", func() {
			t := ginkgo.GinkgoT()
			id := sbID(t)
			ginkgo.DeferCleanup(cleanupSandbox, t, id)

			hostDir := "/tmp/host-vm-e2e-" + id
			_, err := env.Node.Run("mkdir", "-p", hostDir)
			gomega.Expect(err).NotTo(gomega.HaveOccurred())
			ginkgo.DeferCleanup(func() { env.Node.BestEffort("rm", "-rf", hostDir) })

			spec := framework.BasicSpec(id)
			spec.Command = []string{"/bin/sh", "-c", "echo mounted > /data/test; cat /data/test; sleep 3600"}
			spec.VolumeMounts = []agentapi.VolumeMount{
				{Source: hostDir, Destination: "/data", ReadOnly: false},
			}
			resp, code := createSandbox(t, spec)
			gomega.Expect(code).To(gomega.Equal(200), "create: %+v", resp)

			waitFor(t, "mounted content in logs", 30*time.Second, func() bool {
				_, body := getLogs(t, id)
				return strings.Contains(body, "mounted")
			})
		})
	})

	ginkgo.Context("endpoints", func() {
		ginkgo.It("returns the endpoint as PodIP:44772 (DP-E2E-020)", func() {
			t := ginkgo.GinkgoT()
			id := sbID(t)
			ginkgo.DeferCleanup(cleanupSandbox, t, id)

			resp, code := createSandbox(t, framework.BasicSpec(id))
			gomega.Expect(code).To(gomega.Equal(200), "create: %+v", resp)

			gomega.Expect(resp.Endpoints).To(gomega.HaveLen(1))
			ep := resp.Endpoints[0]
			gomega.Expect(ep.Port).To(gomega.Equal(int32(44772)))
			gomega.Expect(ep.Type).To(gomega.Equal(agentapi.EndpointTypeExecd))
			gomega.Expect(ep.IP).To(gomega.Equal(currentAgentPodIP(t)))
			gomega.Expect(resp.EndpointSource).To(gomega.Equal(agentapi.EndpointSourceAgentPod))
		})

		ginkgo.It("reaches execd through the pod network (DP-E2E-021)", ginkgo.Label("needs-execd"), func() {
			t := ginkgo.GinkgoT()
			skipIfMissing(t, "E2E_EXECD_REACHABLE", "needs pod-network access to execd from the node")
			id := sbID(t)
			ginkgo.DeferCleanup(cleanupSandbox, t, id)

			resp, code := createSandbox(t, framework.BasicSpec(id))
			gomega.Expect(code).To(gomega.Equal(200), "create: %+v", resp)
			gomega.Expect(resp.Endpoints).NotTo(gomega.BeEmpty())
			ep := resp.Endpoints[0]
			addr := fmt.Sprintf("%s:%d", ep.IP, ep.Port)

			// execd starts asynchronously inside the sandbox (bootstrap.sh);
			// poll until it answers. --noproxy bypasses any ambient
			// http_proxy, which cannot route pod-IP traffic.
			var httpCode string
			waitFor(t, "execd answers at "+addr, 15*time.Second, func() bool {
				out, err := env.Node.RunShell(
					"curl --noproxy '*' -sS -o /dev/null -w '%{http_code}' --max-time 3 http://" + addr)
				if err != nil {
					return false
				}
				httpCode = strings.TrimSpace(out)
				return httpCode != "" && httpCode != "000"
			})
			gomega.Expect(httpCode).NotTo(gomega.BeEmpty(), "no HTTP response from execd at %s", addr)
		})
	})

	ginkgo.Context("after delete", func() {
		ginkgo.It("cleans the snapshot (DP-E2E-022)", func() {
			t := ginkgo.GinkgoT()
			id := sbID(t)

			resp, code := createSandbox(t, framework.BasicSpec(id))
			gomega.Expect(code).To(gomega.Equal(200), "create: %+v", resp)
			waitFor(t, "snapshot exists", 30*time.Second, func() bool {
				return snapshotExists(t, id)
			})

			delResp, delCode := deleteSandbox(t, id)
			gomega.Expect(delCode).To(gomega.Equal(200))
			gomega.Expect(delResp.Success).To(gomega.BeTrue())

			waitFor(t, "snapshot gone after delete", 30*time.Second, func() bool {
				return !snapshotExists(t, id)
			})
		})

		ginkgo.It("cleans the log file (DP-E2E-023)", func() {
			t := ginkgo.GinkgoT()
			id := sbID(t)

			resp, code := createSandbox(t, framework.BasicSpec(id))
			gomega.Expect(code).To(gomega.Equal(200), "create: %+v", resp)
			waitFor(t, "log file exists", 30*time.Second, func() bool {
				return logFileExists(t, id)
			})

			delResp, delCode := deleteSandbox(t, id)
			gomega.Expect(delCode).To(gomega.Equal(200))
			gomega.Expect(delResp.Success).To(gomega.BeTrue())

			waitFor(t, "log file gone after delete", 30*time.Second, func() bool {
				return !logFileExists(t, id)
			})
		})
	})

	ginkgo.Context("process exit", func() {
		ginkgo.It("reports a stopped process in the sandbox message (DP-E2E-024)", func() {
			t := ginkgo.GinkgoT()
			id := sbID(t)
			ginkgo.DeferCleanup(cleanupSandbox, t, id)

			spec := framework.BasicSpec(id)
			spec.Command = []string{"/bin/sh", "-c", "exit 42"}
			resp, code := createSandbox(t, spec)
			gomega.Expect(code).To(gomega.Equal(200), "create: %+v", resp)

			waitFor(t, "sandbox reports stopped", 30*time.Second, func() bool {
				s, ok := sandboxStatus(t, id)
				return ok && strings.Contains(s.Message, "stopped")
			})
			s, ok := sandboxStatus(t, id)
			gomega.Expect(ok).To(gomega.BeTrue())
			// The manager phase stays running; only the message reflects the
			// stopped process state (exitCode is not exposed over HTTP yet).
			gomega.Expect(s.Phase).To(gomega.Equal("running"), "status: %s", fmtSandbox(s))
		})
	})

	ginkgo.Context("resource limits", func() {
		ginkgo.It("enforces cgroup limits (DP-E2E-025)", ginkgo.Label("needs-ssh"), func() {
			t := ginkgo.GinkgoT()
			skipIfMissing(t, "E2E_NODE_SSH_READY", "needs node-level cgroup access")
			id := sbID(t)
			ginkgo.DeferCleanup(cleanupSandbox, t, id)

			spec := framework.BasicSpec(id) // cpu 100m, memory 64Mi
			resp, code := createSandbox(t, spec)
			gomega.Expect(code).To(gomega.Equal(200), "create: %+v", resp)
			waitForContainer(t, id, 30*time.Second)

			pid := getContainerPID(t, id)
			gomega.Expect(pid).NotTo(gomega.BeZero())

			out, err := env.Node.Run("cat", "/proc/"+strconv.Itoa(pid)+"/cgroup")
			gomega.Expect(err).NotTo(gomega.HaveOccurred())

			// Parse /proc/<pid>/cgroup. cgroup v1 lines are
			// "<hierarchy>:<controllers>:<path>" (e.g. "7:memory:/k8s.io/x",
			// "5:cpu,cpuacct:/k8s.io/x"); cgroup v2 has a single "0::<path>"
			// unified line. Hybrid hosts emit both, but the v2 control files
			// only exist when the unified hierarchy is actually mounted, so
			// prefer the v1 controller paths when present.
			memPath, cpuPath, v2Path := "", "", ""
			for _, line := range strings.Split(out, "\n") {
				fields := strings.SplitN(line, ":", 3)
				if len(fields) != 3 {
					continue
				}
				switch {
				case fields[1] == "memory":
					memPath = fields[2]
				case fields[1] == "cpu" || fields[1] == "cpu,cpuacct":
					cpuPath = fields[2]
				case fields[0] == "0" && fields[1] == "":
					v2Path = fields[2]
				}
			}

			if memPath != "" && cpuPath != "" {
				// cgroup v1: per-controller limit files.
				memOut, err := env.Node.Run("cat", "/sys/fs/cgroup/memory"+memPath+"/memory.limit_in_bytes")
				gomega.Expect(err).NotTo(gomega.HaveOccurred())
				memMax, err := strconv.ParseInt(strings.TrimSpace(memOut), 10, 64)
				gomega.Expect(err).NotTo(gomega.HaveOccurred())
				gomega.Expect(memMax).To(gomega.BeNumerically(">=", 60*1024*1024), "memory.limit_in_bytes: %s", memOut)
				gomega.Expect(memMax).To(gomega.BeNumerically("<=", 70*1024*1024), "memory.limit_in_bytes: %s", memOut)

				quotaOut, err := env.Node.Run("cat", "/sys/fs/cgroup/cpu"+cpuPath+"/cpu.cfs_quota_us")
				gomega.Expect(err).NotTo(gomega.HaveOccurred())
				periodOut, err := env.Node.Run("cat", "/sys/fs/cgroup/cpu"+cpuPath+"/cpu.cfs_period_us")
				gomega.Expect(err).NotTo(gomega.HaveOccurred())
				gomega.Expect(strings.TrimSpace(quotaOut)).To(gomega.Equal("10000"), "cpu.cfs_quota_us: %s", quotaOut)
				gomega.Expect(strings.TrimSpace(periodOut)).To(gomega.Equal("100000"), "cpu.cfs_period_us: %s", periodOut)
				return
			}

			// cgroup v2: unified hierarchy. Read through /proc/<pid>/root:
			// with private cgroupns the "0::" path (e.g. "/") is only valid
			// inside the sandbox's own cgroup namespace.
			gomega.Expect(v2Path).NotTo(gomega.BeEmpty(), "cgroup paths not found for pid %d: %s", pid, out)
			memOut, err := env.Node.Run("cat", "/proc/"+strconv.Itoa(pid)+"/root/sys/fs/cgroup"+v2Path+"/memory.max")
			gomega.Expect(err).NotTo(gomega.HaveOccurred())
			memMax, err := strconv.ParseInt(strings.TrimSpace(memOut), 10, 64)
			gomega.Expect(err).NotTo(gomega.HaveOccurred())
			gomega.Expect(memMax).To(gomega.BeNumerically(">=", 60*1024*1024), "memory.max: %s", memOut)
			gomega.Expect(memMax).To(gomega.BeNumerically("<=", 70*1024*1024), "memory.max: %s", memOut)

			cpuOut, err := env.Node.Run("cat", "/proc/"+strconv.Itoa(pid)+"/root/sys/fs/cgroup"+v2Path+"/cpu.max")
			gomega.Expect(err).NotTo(gomega.HaveOccurred())
			gomega.Expect(strings.TrimSpace(cpuOut)).To(gomega.ContainSubstring("10000 100000"), "cpu.max: %s", cpuOut)
		})
	})
})
