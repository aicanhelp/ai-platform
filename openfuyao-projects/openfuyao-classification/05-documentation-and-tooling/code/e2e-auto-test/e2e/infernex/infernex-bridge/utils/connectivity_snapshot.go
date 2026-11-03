package utils

import (
	"fmt"
	"regexp"
	"strings"
)

var reMooncakeMemLine = regexp.MustCompile(`(?i)(mem storage|memory|segment|global_segment)`)

// LogE2EConnectivitySnapshot prints Mooncake / cache-indexer / inference Service endpoints before HTTP probes.
func LogE2EConnectivitySnapshot(fitID, ns, owner string) {
	LogPhaseBegin(fitID, fmt.Sprintf("connectivity snapshot owner=%s", owner))
	defer LogPhaseEnd(fitID, fmt.Sprintf("connectivity snapshot owner=%s", owner))

	paths := discoverChatCompletionPaths(fitID, ns, owner)
	gatewaySvcs := discoverGatewayServices(fitID, ns, owner)
	gwNS, gwSvc := "", ""
	if len(gatewaySvcs) > 0 {
		gwNS, gwSvc = gatewaySvcs[0].Namespace, gatewaySvcs[0].Name
	}
	LogStep(fitID, "gateway e2e: nodeIP=%s HTTPRoute path(s)=%v gateway services=%s",
		InferenceE2ENodeIP(fitID, gwNS, gwSvc), paths, formatGatewayServices(gatewaySvcs))
	for _, gs := range gatewaySvcs {
		for _, path := range paths {
			for _, c := range gatewayExternalAccessURLs(fitID, gs.Namespace, gs.Name, path) {
				LogStep(fitID, "  gateway URL candidate: %s (host=%s port=%d path=%s)", c.URL, c.ListenHost, c.ListenPort, c.Path)
			}
		}
	}

	logServiceRow(fitID, ns, owner+"-kserve-workload-svc")
	logServiceRow(fitID, ns, owner+"-cache-indexer")
	logServiceRow(fitID, ns, "mooncake-master-service")
	logServiceRow(fitID, ns, "redis-service")
	logServiceRow(fitID, ns, owner+"-epp-service")

	for _, deploy := range []string{
		BridgeWorkloadDeployName(owner, "mooncake-master"),
		BridgeWorkloadDeployName(owner, "mooncake-metadata"),
		DeployName(owner, "cache-indexer"),
		KServeAggregateWorkloadDeployName(owner),
	} {
		logDeployMemoryAndPods(fitID, ns, deploy)
	}

	logMooncakeMasterMemoryLines(fitID, ns, BridgeWorkloadDeployName(owner, "mooncake-master"))
}

func logServiceRow(fitID, ns, svcName string) {
	kc := localKubeconfig()
	existsCmd := fmt.Sprintf("KUBECONFIG=%q kubectl get svc %q -n %q --ignore-not-found -o name", kc, svcName, ns)
	nameOut, _, err := ExecLocal(existsCmd)
	if err != nil || strings.TrimSpace(nameOut) == "" {
		LogStep(fitID, "svc %s/%s: (not found)", ns, svcName)
		return
	}
	typePort := fmt.Sprintf(
		"KUBECONFIG=%q kubectl get svc %q -n %q -o jsonpath='{.spec.type}{\" \"}{.spec.clusterIP}{\" \"}{.spec.ports[0].port}{\" \"}{.spec.ports[0].nodePort}{\" \"}{.spec.ports[0].targetPort}'",
		kc, svcName, ns)
	row, _, _ := ExecLocal(typePort)
	fields := strings.Fields(strings.TrimSpace(row))
	if len(fields) < 3 {
		LogStep(fitID, "svc %s/%s: present (details unavailable)", ns, svcName)
		return
	}
	svcType, clusterIP, port := fields[0], fields[1], fields[2]
	nodePort := "-"
	if len(fields) > 3 && fields[3] != "0" {
		nodePort = fields[3]
	}
	targetPort := port
	if len(fields) > 4 {
		targetPort = fields[4]
	}
	clusterDNS := fmt.Sprintf("%s.%s.svc.cluster.local:%s", svcName, ns, port)
	LogStep(fitID, "svc %s/%s type=%s clusterIP=%s port=%s nodePort=%s targetPort=%s clusterDNS=%s",
		ns, svcName, svcType, clusterIP, port, nodePort, targetPort, clusterDNS)
}

func logDeployMemoryAndPods(fitID, ns, deploy string) {
	kc := localKubeconfig()
	check := fmt.Sprintf("KUBECONFIG=%q kubectl get deploy %q -n %q --ignore-not-found -o name", kc, deploy, ns)
	if out, _, err := ExecLocal(check); err != nil || strings.TrimSpace(out) == "" {
		return
	}
	memCmd := fmt.Sprintf(
		"KUBECONFIG=%q kubectl get deploy %q -n %q -o jsonpath='{range .spec.template.spec.containers[*]}{.name}{\": requests memory=\"}{.resources.requests.memory}{\" limits memory=\"}{.resources.limits.memory}{\"\\n\"}{end}'",
		kc, deploy, ns)
	memOut, _, _ := ExecLocal(memCmd)
	for _, line := range strings.Split(strings.TrimSpace(memOut), "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			LogStep(fitID, "deploy %s/%s container %s", ns, deploy, line)
		}
	}
	podCmd := fmt.Sprintf(
		"KUBECONFIG=%q kubectl get pods -n %q -o custom-columns=NAME:.metadata.name,POD_IP:.status.podIP,HOST_IP:.status.hostIP,PHASE:.status.phase --no-headers 2>/dev/null | grep '^%s' || true",
		kc, ns, deploy)
	podOut, _, _ := ExecLocal(podCmd)
	for _, line := range strings.Split(strings.TrimSpace(podOut), "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			LogStep(fitID, "  %s", line)
		}
	}
}

func logMooncakeMasterMemoryLines(fitID, ns, deploy string) {
	kc := localKubeconfig()
	cmd := fmt.Sprintf("KUBECONFIG=%q kubectl logs -n %q deploy/%q --tail=120 2>/dev/null", kc, ns, deploy)
	out, _, err := ExecLocal(cmd)
	if err != nil || strings.TrimSpace(out) == "" {
		LogStep(fitID, "mooncake-master %s/%s: no Mem Storage lines in recent logs", ns, deploy)
		return
	}
	var hits []string
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if reMooncakeMemLine.MatchString(line) {
			hits = append(hits, line)
		}
	}
	if len(hits) == 0 {
		LogStep(fitID, "mooncake-master %s/%s: no Mem Storage/memory lines in last 120 log lines", ns, deploy)
		return
	}
	LogStep(fitID, "mooncake-master %s/%s latest metrics: %s", ns, deploy, hits[len(hits)-1])
}
