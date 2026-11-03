package utils

import (
	"fmt"
	"os"
	"strings"
)

// InferenceE2ENodeIP picks a node IP for NodePort gateway access without hardcoding n2.
// If gatewaySvcNS/Name are set, prefers the node where the gateway Service's backend Pod runs.
func InferenceE2ENodeIP(fitID, gatewaySvcNS, gatewaySvcName string) string {
	for _, key := range []string{"INFERENCE_E2E_NODE_IP", "HERMES_NODE_IP", "INFERNEX_E2E_NODE_IP"} {
		if ip := strings.TrimSpace(os.Getenv(key)); ip != "" {
			LogStep(fitID, "gateway node IP from env %s=%s", key, ip)
			return ip
		}
	}
	if nodeName := strings.TrimSpace(os.Getenv("INFERNEX_E2E_NODE_NAME")); nodeName != "" {
		if ip := nodeIPByNodeName(fitID, nodeName); ip != "" {
			return ip
		}
	}

	if gatewaySvcNS != "" && gatewaySvcName != "" {
		if ip, node := nodeIPFromGatewayService(fitID, gatewaySvcNS, gatewaySvcName); ip != "" {
			LogStep(fitID, "gateway node IP=%s (node %s hosting gateway pod; from endpoints %s/%s)",
				ip, node, gatewaySvcNS, gatewaySvcName)
			return ip
		}
	}

	if host, err := os.Hostname(); err == nil && host != "" {
		if ip := nodeIPByNodeName(fitID, host); ip != "" {
			LogStep(fitID, "gateway node IP=%s (matched local hostname to node %s)", ip, host)
			return ip
		}
	}

	LogClusterNodeIPs(fitID)
	if ip, node := firstReadyNodeIP(fitID); ip != "" {
		LogStep(fitID, "gateway node IP=%s (first Ready node %s; set INFERENCE_E2E_NODE_IP if unreachable)",
			ip, node)
		return ip
	}
	return ""
}

// LogClusterNodeIPs prints kubectl get nodes -o wide equivalent for manual pick.
func LogClusterNodeIPs(fitID string) {
	kc := localKubeconfig()
	cmd := fmt.Sprintf(
		"KUBECONFIG=%q kubectl get nodes -o custom-columns=NAME:.metadata.name,READY:.status.conditions[?(@.type==\"Ready\")].status,INTERNAL-IP:.status.addresses[?(@.type==\"InternalIP\")].address,EXTERNAL-IP:.status.addresses[?(@.type==\"ExternalIP\")].address --no-headers",
		kc)
	out, stderr, err := ExecLocal(cmd)
	if err != nil {
		LogStep(fitID, "kubectl get nodes -o wide: %v %s", err, stderr)
		return
	}
	LogStep(fitID, "cluster nodes (pick INFERENCE_E2E_NODE_IP from INTERNAL-IP reachable by test host):")
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			LogStep(fitID, "  %s", line)
		}
	}
	LogStep(fitID, "manual: kubectl get nodes -o wide")
	LogStep(fitID, "manual: kubectl get node <name> -o jsonpath='{.status.addresses[?(@.type==\"InternalIP\")].address}'")
}

func nodeIPFromGatewayService(fitID, svcNS, svcName string) (ip, nodeName string) {
	kc := localKubeconfig()
	// endpoints → pod → node
	podCmd := fmt.Sprintf(
		"KUBECONFIG=%q kubectl get endpoints %q -n %q -o jsonpath='{.subsets[0].addresses[0].targetRef.name}'",
		kc, svcName, svcNS)
	pod, _, _ := ExecLocal(podCmd)
	pod = strings.TrimSpace(pod)
	if pod == "" {
		return "", ""
	}
	nodeCmd := fmt.Sprintf(
		"KUBECONFIG=%q kubectl get pod %q -n %q -o jsonpath='{.spec.nodeName}'",
		kc, pod, svcNS)
	node, _, _ := ExecLocal(nodeCmd)
	node = strings.TrimSpace(node)
	if node == "" {
		return "", ""
	}
	ip = nodeIPByNodeName(fitID, node)
	return ip, node
}

func nodeIPByNodeName(fitID, nodeName string) string {
	kc := localKubeconfig()
	for _, addrType := range []string{"InternalIP", "ExternalIP"} {
		cmd := fmt.Sprintf(
			"KUBECONFIG=%q kubectl get node %q -o jsonpath='{.status.addresses[?(@.type==%q)].address}'",
			kc, nodeName, addrType)
		out, stderr, err := ExecLocal(cmd)
		if err != nil {
			LogStep(fitID, "kubectl get node %s %s: %v %s", nodeName, addrType, err, stderr)
			continue
		}
		if ip := strings.TrimSpace(out); ip != "" {
			return ip
		}
	}
	return ""
}

func SchedulableNodeCount(fitID string) int {
	kc := localKubeconfig()
	out, stderr, err := ExecLocal(fmt.Sprintf("KUBECONFIG=%q kubectl get nodes --no-headers", kc))
	if err != nil {
		LogStep(fitID, "kubectl get nodes: %v %s", err, stderr)
		return -1
	}
	count := 0
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		fields := strings.Fields(strings.TrimSpace(line))
		// NAME STATUS ROLES AGE VERSION — Ready status may be "Ready" or "Ready,<more>"
		if len(fields) >= 2 && fields[1] == "Ready" {
			count++
		}
	}
	return count
}

func firstReadyNodeIP(fitID string) (ip, nodeName string) {
	kc := localKubeconfig()
	cmd := fmt.Sprintf(
		"KUBECONFIG=%q kubectl get nodes -o jsonpath='{range .items[?(@.status.conditions[?(@.type==\"Ready\")].status==\"True\")]}{.metadata.name}{\"\\t\"}{.status.addresses[?(@.type==\"InternalIP\")].address}{\"\\n\"}{end}'",
		kc)
	out, _, err := ExecLocal(cmd)
	if err != nil || strings.TrimSpace(out) == "" {
		return "", ""
	}
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		fields := strings.Fields(line)
		if len(fields) >= 2 && fields[1] != "" {
			return fields[1], fields[0]
		}
	}
	return "", ""
}
