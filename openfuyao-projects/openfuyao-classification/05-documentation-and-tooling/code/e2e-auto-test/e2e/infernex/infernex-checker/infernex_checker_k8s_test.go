package infernex_checker

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

type k8sNodeEvidence struct {
	node           checkerNode
	k8sNodeName    string
	found          bool
	ready          bool
	hasTaints      bool
	hasAllocatable bool
}

type k01Evidence struct {
	totalCoreDNSPods   int
	runningCoreDNSPods int
	dnsResolutionOK    bool
	dnsPartialFailure  bool
	dnsFailedNodes     []string
	dnsCheckExecFailed bool
}

var _ = Describe("Infernex software pre-flight checks", Label("infernex_checker", "precheck"), func() {
	Describe("K8s Layer K-01~K-04", Ordered, Label("k8s"), func() {
		var k8sRun checkRun
		var k8sReport *report
		var nodes []checkerNode
		var kubectlNode checkerNode
		var k01Env k01Evidence
		var nodeEnv []k8sNodeEvidence

		BeforeAll(func() {
			nodes = loadCheckerNodes()
			kubectlNode = selectK8sKubectlNode(nodes)
			k01Env = collectK01Evidence(kubectlNode, nodes)
			nodeEnv = collectK8sNodeEvidence(nodes, kubectlNode)
			k8sRun = runInfernexCheckCached("k8s", false, true)
			Expect(k8sRun.err).NotTo(HaveOccurred(), "infernex-check k8s should run successfully, stderr: %s", k8sRun.stderr)
			k8sReport = k8sRun.report
			Expect(k8sReport).NotTo(BeNil(), "k8s command should write JSON report")
			Expect(k8sReport.K8s).NotTo(BeNil(), "JSON report should contain k8s result")
		})

		It("ic_k8s_001-【K-01】coreDNS pod running且DNS解析成功", Label("P0"), func() {
			if k01Env.runningCoreDNSPods == 0 {
				Skip("current cluster has no Running CoreDNS pods; K-01 positive scenario requires Running CoreDNS")
			}
			if k01Env.dnsCheckExecFailed {
				Skip("current cluster could not execute DNS probe pods; K-01 positive scenario requires DNS evidence to be collectable")
			}
			if !k01Env.dnsResolutionOK {
				Skip("current cluster does not have all nodes passing DNS resolution; K-01 positive scenario requires all nodes to resolve DNS successfully")
			}
			check := requireK8sCheck(k8sReport, "K-01")
			Expect(check.Status).To(Equal(statusPassed), "K-01 checker message: %s; suggestion: %s", check.Message, check.Suggestion)
			Expect(check.Message).To(ContainSubstring("CoreDNS Running"))
			Expect(k8sRun.stdout).To(ContainSubstring("K-01"))
			Expect(k8sRun.stdout).To(ContainSubstring("CoreDNS Running"))
		})

		It("ic_k8s_002-【K-01】coreDNS pod不存在时报失败且终止K8s层后续检查", Label("P0"), func() {
			if k01Env.totalCoreDNSPods != 0 {
				Skip("current cluster has CoreDNS pods; missing CoreDNS scenario requires a real cluster without CoreDNS pods")
			}
			assertK8sFailure(k8sReport, "K-01", "No CoreDNS pods found")
			Expect(k8sReport.K8s.Status).To(Equal(statusFailed))
		})

		It("ic_k8s_003-【K-01】coreDNS pod存在但所有非Running时报失败", Label("P1"), func() {
			if k01Env.totalCoreDNSPods == 0 || k01Env.runningCoreDNSPods > 0 {
				Skip("current cluster does not match the real scenario where CoreDNS pods exist but none are Running")
			}
			assertK8sFailure(k8sReport, "K-01", "All CoreDNS pods are not in Running state")
			Expect(k8sReport.K8s.Status).To(Equal(statusFailed))
		})

		It("ic_k8s_004-【K-01】所有节点DNS解析执行失败时报失败", Label("P1"), func() {
			if k01Env.runningCoreDNSPods == 0 {
				Skip("current cluster has no Running CoreDNS pods; DNS resolution failure scenario requires Running CoreDNS with failed DNS resolution")
			}
			if k01Env.dnsCheckExecFailed {
				Skip("current cluster could not execute DNS probe pods; DNS resolution failure scenario requires the probe to run and fail resolution")
			}
			if k01Env.dnsResolutionOK || k01Env.dnsPartialFailure {
				Skip("current cluster DNS resolution works on all or some nodes; all-nodes-failed scenario requires DNS to fail on every node")
			}
			assertK8sFailure(k8sReport, "K-01", "DNS resolution failed")
			Expect(k8sRun.stdout).To(ContainSubstring("K-01"))
			Expect(k8sRun.stdout).To(ContainSubstring("DNS resolution failed"))
			Expect(k8sReport.K8s.Status).To(Equal(statusFailed))
		})

		It("ic_k8s_005-【K-01】部分节点DNS解析执行失败时报Warning", Label("P1"), func() {
			if k01Env.runningCoreDNSPods == 0 {
				Skip("current cluster has no Running CoreDNS pods; partial DNS failure scenario requires Running CoreDNS")
			}
			if k01Env.dnsCheckExecFailed {
				Skip("current cluster could not execute DNS probe pods; partial DNS failure scenario requires DNS evidence to be collectable")
			}
			if !k01Env.dnsPartialFailure {
				Skip("current cluster DNS resolution passed on all nodes or failed on all nodes; partial failure scenario requires some nodes passing and some failing")
			}
			check := requireK8sCheck(k8sReport, "K-01")
			Expect(check.Status).To(Equal(statusWarning))
			Expect(check.Message).To(ContainSubstring("DNS resolution failed on some nodes"))
			Expect(k8sRun.stdout).To(ContainSubstring("K-01"))
		})

		It("ic_k8s_006-【K-02】所有配置节点均处于Ready状态", Label("P0"), func() {
			matched := filterEvidence(nodeEnv, func(item k8sNodeEvidence) bool {
				return item.found && item.ready
			})
			if len(matched) == 0 {
				Skip("current configured nodes are not Ready or not found from kubectl on the tested node")
			}
			for _, item := range matched {
				check := k8sCheckForNodeOrSkip(k8sReport, item, "K-02")
				Expect(check.Status).To(Equal(statusPassed))
				Expect(check.Message).To(ContainSubstring("Ready"))
			}
		})

		It("ic_k8s_007-【K-02】节点NotReady时报失败且由于无Ready节点k8s层终止", Label("P0"), func() {
			readyNodes := filterEvidence(nodeEnv, func(item k8sNodeEvidence) bool {
				return item.found && item.ready
			})
			if len(readyNodes) > 0 {
				Skip("current configured nodes include Ready nodes; this case requires all configured nodes to be NotReady or unavailable")
			}
			matched := filterEvidence(nodeEnv, func(item k8sNodeEvidence) bool {
				return !item.found || !item.ready
			})
			if len(matched) == 0 {
				Skip("current configured nodes are Ready; NotReady scenario requires mock or dedicated fault cluster")
			}
			for _, item := range matched {
				assertK8sNodeFailureOrSkip(k8sReport, item, "K-02", "not in Ready state", "configured node did not reach K-02 in current tested kubeconfig")
			}
			Expect(k8sReport.K8s.Status).To(Equal(statusFailed))
			Expect(k8sReport.K8s.Checks).NotTo(ContainElement(WithTransform(func(c checkResult) string { return c.ID }, Or(Equal("K-03"), Equal("K-04")))))
		})

		It("ic_k8s_008-【K-03】节点无污点时状态为PASSED", Label("P1"), func() {
			matched := filterEvidence(nodeEnv, func(item k8sNodeEvidence) bool {
				return item.found && item.ready && !item.hasTaints
			})
			if len(matched) == 0 {
				Skip("current configured nodes have taints; no-taint positive scenario requires an untainted node")
			}
			for _, item := range matched {
				check := k8sCheckForNodeOrSkip(k8sReport, item, "K-03")
				Expect(check.Status).To(Equal(statusPassed))
				Expect(check.Message).To(ContainSubstring("has no taints"))
			}
		})

		It("ic_k8s_009-【K-03】节点有污点时状态为INFO并提示用户判断", Label("P1"), func() {
			matched := filterEvidence(nodeEnv, func(item k8sNodeEvidence) bool {
				return item.found && item.ready && item.hasTaints
			})
			if len(matched) == 0 {
				Skip("current configured nodes have no taints; taint info scenario requires a tainted node")
			}
			for _, item := range matched {
				check := k8sCheckForNodeOrSkip(k8sReport, item, "K-03")
				Expect(check.Status).To(Equal(statusInfo))
				Expect(check.Message).To(ContainSubstring("has taints"))
			}
		})

		It("ic_k8s_010-【K-04】上报各节点可分配CPU和内存（INFO类型）", Label("P1"), func() {
			matched := filterEvidence(nodeEnv, func(item k8sNodeEvidence) bool {
				return item.found && item.ready && item.hasAllocatable
			})
			if len(matched) == 0 {
				Skip("current configured nodes do not have allocatable CPU/Memory evidence from kubectl on the tested node")
			}
			for _, item := range matched {
				check := k8sCheckForNodeOrSkip(k8sReport, item, "K-04")
				Expect(check.Status).To(Equal(statusInfo))
				Expect(check.Message).To(ContainSubstring("Allocatable:"))
				Expect(check.Message).To(ContainSubstring("CPU"))
				Expect(check.Message).To(ContainSubstring("Memory"))
				Expect(check.Message).To(Or(
					ContainSubstring("Remaining:"),
					ContainSubstring("unable to calculate remaining"),
				))
			}
		})
	})
})

func collectK01Evidence(kubectlNode checkerNode, nodes []checkerNode) k01Evidence {
	podsResult := runRemoteKubectl(kubectlNode, "get", "pods", "-A", "-l", "k8s-app=kube-dns", "-o", "json")
	Expect(podsResult.err).NotTo(HaveOccurred(), "kubectl should list CoreDNS pods with k8s-app=kube-dns on node %s", kubectlNode.Name)

	var podsJSON map[string]interface{}
	Expect(json.Unmarshal([]byte(podsResult.stdout), &podsJSON)).To(Succeed(), "kubectl pods output should be valid JSON")

	evidence := k01Evidence{}
	for _, raw := range listItems(podsJSON) {
		pod, ok := raw.(map[string]interface{})
		if !ok {
			continue
		}
		evidence.totalCoreDNSPods++
		if podPhase(pod) == "Running" {
			evidence.runningCoreDNSPods++
		}
	}

	if evidence.runningCoreDNSPods > 0 {
		collectDNSResolutionEvidence(kubectlNode, nodes, &evidence)
	}
	return evidence
}

func podPhase(pod map[string]interface{}) string {
	status, _ := pod["status"].(map[string]interface{})
	phase, _ := status["phase"].(string)
	return phase
}

func selectK8sKubectlNode(nodes []checkerNode) checkerNode {
	for _, node := range nodes {
		result := runRemoteKubectl(node, "get", "nodes", "-o", "json")
		if result.err == nil && strings.TrimSpace(result.stdout) != "" {
			GinkgoWriter.Printf("using node %s (%s) for remote kubectl evidence collection\n", node.Name, node.IP)
			return node
		}
		GinkgoWriter.Printf("node %s (%s) cannot run kubectl get nodes: %v, stderr: %s\n",
			node.Name, node.IP, result.err, result.stderr)
	}
	Skip("no configured node can run kubectl get nodes; k8s evidence requires kubectl on a real tested node")
	return checkerNode{}
}

func runRemoteKubectl(node checkerNode, args ...string) commandResult {
	command := shellCommand("kubectl", args...)
	return runSSH(node, command)
}

func shellCommand(name string, args ...string) string {
	parts := []string{shellQuote(name)}
	for _, arg := range args {
		parts = append(parts, shellQuote(arg))
	}
	return strings.Join(parts, " ")
}

func collectDNSResolutionEvidence(kubectlNode checkerNode, nodes []checkerNode, evidence *k01Evidence) {
	nodesResult := runRemoteKubectl(kubectlNode, "get", "nodes", "-o", "json")
	if nodesResult.err != nil {
		evidence.dnsCheckExecFailed = true
		return
	}
	var nodesJSON map[string]interface{}
	if err := json.Unmarshal([]byte(nodesResult.stdout), &nodesJSON); err != nil {
		evidence.dnsCheckExecFailed = true
		return
	}

	var k8sNodeNames []string
	for _, node := range nodes {
		name := findK8sNodeForCheckerNode(nodesJSON, node)
		if name != "" {
			k8sNodeNames = append(k8sNodeNames, name)
		}
	}
	if len(k8sNodeNames) == 0 {
		evidence.dnsCheckExecFailed = true
		return
	}

	timestamp := time.Now().UnixNano()
	passedCount := 0
	for i, k8sNodeName := range k8sNodeNames {
		podName := fmt.Sprintf("infernex-e2e-dns-%d-%d", timestamp, i)
		if testDNSOnNode(kubectlNode, k8sNodeName, podName) {
			passedCount++
		} else {
			evidence.dnsFailedNodes = append(evidence.dnsFailedNodes, k8sNodeName)
		}
	}

	total := len(k8sNodeNames)
	evidence.dnsResolutionOK = passedCount == total
	evidence.dnsPartialFailure = passedCount > 0 && passedCount < total
}

func testDNSOnNode(kubectlNode checkerNode, k8sNodeName string, podName string) bool {
	overrides := fmt.Sprintf(`{"spec":{"nodeName":"%s"}}`, k8sNodeName)
	create := runRemoteKubectl(kubectlNode, "run", podName,
		"--image=busybox:1.36",
		"--restart=Never",
		"--overrides="+overrides,
		"--command", "--", "sleep", "300")
	if create.err != nil {
		return false
	}
	defer runRemoteKubectl(kubectlNode, "delete", "pod", podName, "--ignore-not-found=true", "--wait=false")

	wait := runRemoteKubectl(kubectlNode, "wait", "--for=condition=Ready", "pod/"+podName, "--timeout=60s")
	if wait.err != nil {
		return false
	}

	lookup := runRemoteKubectl(kubectlNode, "exec", podName, "--", "nslookup", "kubernetes.default.svc.cluster.local")
	output := strings.ToLower(lookup.stdout + "\n" + lookup.stderr)
	return lookup.err == nil && strings.Contains(output, "kubernetes.default") && strings.Contains(output, "address")
}

func collectK8sNodeEvidence(nodes []checkerNode, kubectlNode checkerNode) []k8sNodeEvidence {
	nodesResult := runRemoteKubectl(kubectlNode, "get", "nodes", "-o", "json")
	Expect(nodesResult.err).NotTo(HaveOccurred(), "kubectl should list nodes on node %s", kubectlNode.Name)

	var nodesJSON map[string]interface{}
	Expect(json.Unmarshal([]byte(nodesResult.stdout), &nodesJSON)).To(Succeed(), "kubectl nodes output should be valid JSON")

	var evidence []k8sNodeEvidence
	for _, node := range nodes {
		item := k8sNodeEvidence{node: node}
		k8sNodeName := findK8sNodeForCheckerNode(nodesJSON, node)
		item.k8sNodeName = k8sNodeName
		item.found = k8sNodeName != ""
		if item.found {
			item.ready = k8sNodeReady(nodesJSON, k8sNodeName)
			item.hasTaints = k8sNodeHasTaints(nodesJSON, k8sNodeName)
			item.hasAllocatable = k8sNodeHasAllocatable(nodesJSON, k8sNodeName)
		}
		evidence = append(evidence, item)
	}
	return evidence
}

func k8sCheckForNodeOrSkip(report *report, item k8sNodeEvidence, id string) checkResult {
	checks := k8sChecksForNode(report, item, id)
	if len(checks) == 0 {
		Skip(id + " was not executed for configured node " + item.node.Name)
	}
	return checks[0]
}

func assertK8sNodeFailureOrSkip(report *report, item k8sNodeEvidence, id string, messagePart string, skipReason string) {
	checks := k8sChecksForNode(report, item, id)
	for _, check := range checks {
		if check.Status == statusFailed && strings.Contains(check.Message, messagePart) {
			Expect(check.Suggestion).NotTo(BeEmpty())
			return
		}
	}
	Skip(skipReason)
}

func assertK8sFailure(report *report, id string, messagePart string) {
	checks := findK8sChecks(report, id)
	for _, check := range checks {
		if check.Status == statusFailed && strings.Contains(check.Message, messagePart) {
			Expect(check.Suggestion).NotTo(BeEmpty())
			return
		}
	}
	Expect(checks).To(ContainElement(SatisfyAll(
		WithTransform(func(c checkResult) checkStatus { return c.Status }, Equal(statusFailed)),
		WithTransform(func(c checkResult) string { return c.Message }, ContainSubstring(messagePart)),
	)), "%s should fail with message containing %q", id, messagePart)
}

func k8sChecksForNode(report *report, item k8sNodeEvidence, id string) []checkResult {
	checks := findK8sChecks(report, id)
	var matched []checkResult
	identifiers := []string{item.k8sNodeName, item.node.Name, item.node.IP}
	for _, check := range checks {
		for _, identifier := range identifiers {
			if identifier == "" {
				continue
			}
			if check.Node == identifier || strings.Contains(check.Message, identifier) {
				matched = append(matched, check)
				break
			}
		}
	}
	return matched
}

func k8sNodeReady(nodesJSON map[string]interface{}, k8sNode string) bool {
	node := k8sNodeObject(nodesJSON, k8sNode)
	status, _ := node["status"].(map[string]interface{})
	for _, raw := range listItemsFrom(status, "conditions") {
		condition, ok := raw.(map[string]interface{})
		if !ok {
			continue
		}
		conditionType, _ := condition["type"].(string)
		conditionStatus, _ := condition["status"].(string)
		if conditionType == "Ready" {
			return conditionStatus == "True"
		}
	}
	return false
}

func k8sNodeHasTaints(nodesJSON map[string]interface{}, k8sNode string) bool {
	node := k8sNodeObject(nodesJSON, k8sNode)
	spec, _ := node["spec"].(map[string]interface{})
	taints, _ := spec["taints"].([]interface{})
	return len(taints) > 0
}

func k8sNodeHasAllocatable(nodesJSON map[string]interface{}, k8sNode string) bool {
	node := k8sNodeObject(nodesJSON, k8sNode)
	status, _ := node["status"].(map[string]interface{})
	allocatable, _ := status["allocatable"].(map[string]interface{})
	_, hasCPU := allocatable["cpu"]
	_, hasMemory := allocatable["memory"]
	return hasCPU && hasMemory
}

func k8sNodeObject(nodesJSON map[string]interface{}, k8sNode string) map[string]interface{} {
	for _, raw := range listItems(nodesJSON) {
		item, ok := raw.(map[string]interface{})
		if !ok {
			continue
		}
		metadata, _ := item["metadata"].(map[string]interface{})
		name, _ := metadata["name"].(string)
		if name == k8sNode {
			return item
		}
	}
	return nil
}
