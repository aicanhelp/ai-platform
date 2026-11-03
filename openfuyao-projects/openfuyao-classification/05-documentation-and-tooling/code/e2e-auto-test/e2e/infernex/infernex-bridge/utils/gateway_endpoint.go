package utils

import (
	"fmt"
	"net/url"
	"os"
	"strconv"
	"strings"
)

// AllowWorkloadDirectInference enables bypassing gateway (debug only).
func AllowWorkloadDirectInference() bool {
	return envBool("INFERENCE_ALLOW_WORKLOAD_DIRECT")
}

// discoverKServeEnvoyIngressService finds envoy-kserve-kserve-ingress-gateway-* in envoy-gateway-system.
func discoverKServeEnvoyIngressService(fitID string) (ns, name string, ok bool) {
	if override := strings.TrimSpace(os.Getenv("INFERENCE_GATEWAY_SVC")); override != "" {
		parts := strings.Split(override, "/")
		if len(parts) == 2 {
			return parts[0], parts[1], serviceExistsOK(parts[0], parts[1])
		}
	}
	gwNS := EnvoyGatewaySystemNS()
	prefix := KServeGatewaySvcPrefix()
	names := listServiceNamesInNamespace(fitID, gwNS, func(n string) bool {
		return strings.HasPrefix(n, prefix)
	})
	if len(names) == 0 {
		LogStep(fitID, "no svc in %s with prefix %q", gwNS, prefix)
		return "", "", false
	}
	name = pickStableName(names)
	LogStep(fitID, "KServe ingress gateway svc=%s/%s (prefix %q)", gwNS, name, prefix)
	logGatewayServiceEndpoints(fitID, gwNS, name)
	return gwNS, name, true
}

// infernexGatewayCRForDiscovery returns managed Gateway CR name or HTTPRoute parent ref (gateway.ref FITs).
func infernexGatewayCRForDiscovery(fitID, insvcNS, owner string) string {
	gwCR := infernexGatewayCRName(owner)
	if k8sResourceExists(fitID, "gateway", insvcNS, gwCR) {
		return gwCR
	}
	routeName := owner + "-infernex-route"
	if !k8sResourceExists(fitID, "httproute", insvcNS, routeName) {
		return gwCR
	}
	kc := localKubeconfig()
	parent := strings.TrimSpace(kubectlJSONPathQuiet(kc, fmt.Sprintf(
		"kubectl get httproute %q -n %q -o jsonpath='{.spec.parentRefs[0].name}'",
		routeName, insvcNS)))
	if parent != "" {
		LogStep(fitID, "InferNex gateway discovery: HTTPRoute %s/%s parent gateway=%s", insvcNS, routeName, parent)
		return parent
	}
	return gwCR
}

// discoverInfernexEnvoyIngressService finds the Envoy Service for the InferNex HTTPRoute parent Gateway.
// FIT 6/7: managed Gateway CR {owner}-infernex-gateway; FIT 36/42: external gateway via gateway.ref parentRefs.
func discoverInfernexEnvoyIngressService(fitID, insvcNS, owner string) (ns, name string, ok bool) {
	if override := strings.TrimSpace(os.Getenv("INFERNEX_GATEWAY_SVC")); override != "" {
		parts := strings.Split(override, "/")
		if len(parts) == 2 {
			return parts[0], parts[1], serviceExistsOK(parts[0], parts[1])
		}
	}
	gwCR := infernexGatewayCRForDiscovery(fitID, insvcNS, owner)
	if !k8sResourceExists(fitID, "gateway", insvcNS, gwCR) {
		LogStep(fitID, "InferNex Gateway CR %s/%s not found (skip infernex envoy svc discovery)", insvcNS, gwCR)
		return "", "", false
	}

	searchNSs := dedupeStrings([]string{EnvoyGatewaySystemNS(), insvcNS})
	for _, svcNS := range searchNSs {
		names := listEnvoyOwningGatewayServices(fitID, svcNS, insvcNS, gwCR)
		if len(names) > 0 {
			name = pickStableName(names)
			LogStep(fitID, "InferNex ingress gateway svc=%s/%s (Gateway CR %s/%s labels owning-gateway-name=%s owning-gateway-namespace=%s)",
				svcNS, name, insvcNS, gwCR, gwCR, insvcNS)
			logGatewayServiceEndpoints(fitID, svcNS, name)
			return svcNS, name, true
		}
	}

	// Some clusters still expose gateway.networking.k8s.io/gateway-name
	for _, svcNS := range searchNSs {
		names := listGatewayServiceNamesByK8sLabel(fitID, svcNS, gwCR)
		if len(names) > 0 {
			name = pickStableName(names)
			LogStep(fitID, "InferNex ingress gateway svc=%s/%s (Gateway CR %s/%s label gateway.networking.k8s.io/gateway-name=%s)",
				svcNS, name, insvcNS, gwCR, gwCR)
			logGatewayServiceEndpoints(fitID, svcNS, name)
			return svcNS, name, true
		}
	}

	prefix := infernexEnvoyServicePrefix(insvcNS, gwCR)
	for _, svcNS := range searchNSs {
		names := listServiceNamesInNamespace(fitID, svcNS, func(n string) bool {
			return strings.HasPrefix(n, prefix)
		})
		if len(names) == 0 {
			names = listServiceNamesInNamespace(fitID, svcNS, func(n string) bool {
				return strings.Contains(n, owner) && strings.Contains(n, "infernex-gateway")
			})
		}
		if len(names) > 0 {
			name = pickStableName(names)
			LogStep(fitID, "InferNex ingress gateway svc=%s/%s (name prefix %q)", svcNS, name, prefix)
			logGatewayServiceEndpoints(fitID, svcNS, name)
			return svcNS, name, true
		}
	}

	logInfernexGatewayProgrammedState(fitID, insvcNS, gwCR)
	LogStep(fitID, "no envoy svc for gateway CR %s/%s (searched ns=%v; label gateway.envoyproxy.io/owning-gateway-*)", insvcNS, gwCR, searchNSs)
	return "", "", false
}

func infernexGatewayCRName(owner string) string {
	return owner + "-infernex-gateway"
}

// infernexEnvoyServicePrefix matches envoy-{gateway-ns}-{gateway-name}-<hash> from Envoy Gateway.
func infernexEnvoyServicePrefix(gatewayNS, gatewayName string) string {
	return fmt.Sprintf("envoy-%s-%s-", gatewayNS, gatewayName)
}

func listEnvoyOwningGatewayServices(fitID, svcNS, owningGatewayNS, owningGatewayName string) []string {
	kc := localKubeconfig()
	cmd := fmt.Sprintf(
		"KUBECONFIG=%q kubectl get svc -n %q -l gateway.envoyproxy.io/owning-gateway-namespace=%q,gateway.envoyproxy.io/owning-gateway-name=%q -o jsonpath='{range .items[*]}{.metadata.name}{\"\\n\"}{end}'",
		kc, svcNS, owningGatewayNS, owningGatewayName)
	out, stderr, err := ExecLocal(cmd)
	if err != nil || strings.TrimSpace(out) == "" {
		LogPollThrottled(fitID, "envoy-owning-"+svcNS+"-"+owningGatewayName, defaultPollLogInterval,
			"svc ns=%s owning-gateway %s/%s: empty (%v %s)", svcNS, owningGatewayNS, owningGatewayName, err, stderr)
		return nil
	}
	return splitNonEmptyLines(out)
}

func listGatewayServiceNamesByK8sLabel(fitID, svcNS, gatewayCRName string) []string {
	kc := localKubeconfig()
	cmd := fmt.Sprintf(
		"KUBECONFIG=%q kubectl get svc -n %q -l gateway.networking.k8s.io/gateway-name=%q -o jsonpath='{range .items[*]}{.metadata.name}{\"\\n\"}{end}'",
		kc, svcNS, gatewayCRName)
	out, _, err := ExecLocal(cmd)
	if err != nil || strings.TrimSpace(out) == "" {
		return nil
	}
	return splitNonEmptyLines(out)
}

func splitNonEmptyLines(out string) []string {
	var names []string
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			names = append(names, line)
		}
	}
	return names
}

func logInfernexGatewayProgrammedState(fitID, insvcNS, gwCR string) {
	kc := localKubeconfig()
	cmd := fmt.Sprintf(
		"KUBECONFIG=%q kubectl get gateway %q -n %q -o jsonpath='programmed={.status.conditions[?(@.type==\"Programmed\")].status} reason={.status.conditions[?(@.type==\"Programmed\")].reason} accepted={.status.conditions[?(@.type==\"Accepted\")].status}'",
		kc, gwCR, insvcNS)
	out, _, _ := ExecLocal(cmd)
	if strings.TrimSpace(out) != "" {
		LogStep(fitID, "gateway %s/%s status: %s", insvcNS, gwCR, strings.TrimSpace(out))
	}
}

func listServiceNamesInNamespace(fitID, ns string, match func(string) bool) []string {
	kc := localKubeconfig()
	cmd := fmt.Sprintf(
		"KUBECONFIG=%q kubectl get svc -n %q -o jsonpath='{range .items[*]}{.metadata.name}{\"\\n\"}{end}'",
		kc, ns)
	out, _, err := ExecLocal(cmd)
	if err != nil {
		return nil
	}
	var names []string
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		line = strings.TrimSpace(line)
		if line != "" && match(line) {
			names = append(names, line)
		}
	}
	return names
}

func pickStableName(names []string) string {
	if len(names) == 0 {
		return ""
	}
	name := names[0]
	for _, m := range names[1:] {
		if m < name {
			name = m
		}
	}
	return name
}

func k8sResourceExists(fitID, resource, ns, name string) bool {
	kc := localKubeconfig()
	cmd := fmt.Sprintf("KUBECONFIG=%q kubectl get %s %q -n %q --ignore-not-found -o name", kc, resource, name, ns)
	out, _, err := ExecLocal(cmd)
	return err == nil && strings.TrimSpace(out) != ""
}

// logGatewayServiceEndpoints logs backing Pod IPs for the gateway Service (endpoints slice).
func logGatewayServiceEndpoints(fitID, svcNS, svcName string) {
	kc := localKubeconfig()
	cmd := fmt.Sprintf(
		"KUBECONFIG=%q kubectl get endpoints %q -n %q -o jsonpath='{range .subsets[*]}{range .addresses[*]}{.ip}{\" pod=\"}{.targetRef.name}{\"\\n\"}{end}{end}'",
		kc, svcName, svcNS)
	out, _, err := ExecLocal(cmd)
	if err != nil || strings.TrimSpace(out) == "" {
		LogStep(fitID, "gateway endpoints %s/%s: (none yet)", svcNS, svcName)
		return
	}
	LogStep(fitID, "gateway endpoints %s/%s backends:", svcNS, svcName)
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			LogStep(fitID, "  %s", line)
		}
	}
}

func serviceExistsOK(ns, name string) bool {
	kc := localKubeconfig()
	cmd := fmt.Sprintf("KUBECONFIG=%q kubectl get svc %q -n %q --ignore-not-found -o name", kc, name, ns)
	out, _, err := ExecLocal(cmd)
	return err == nil && strings.TrimSpace(out) != ""
}

// discoverGatewayServices returns gateway Services in envoy-gateway-system for e2e HTTP.
// FIT 4/5 (LLMISVC): envoy-kserve-kserve-ingress-gateway-* ; FIT 6/7 (InferNexService): label gateway-name={owner}-infernex-gateway.
func discoverGatewayServices(fitID, ns, owner string) []gatewayService {
	var out []gatewayService
	add := func(gs gatewayService) {
		for _, existing := range out {
			if existing.Namespace == gs.Namespace && existing.Name == gs.Name {
				return
			}
		}
		out = append(out, gs)
	}

	hasKServeRoute := k8sResourceExists(fitID, "httproute", ns, owner+"-kserve-route")
	hasInfernexGW := k8sResourceExists(fitID, "gateway", ns, owner+"-infernex-gateway") ||
		k8sResourceExists(fitID, "httproute", ns, owner+"-infernex-route")

	if hasKServeRoute {
		if gwNS, gwSvc, ok := discoverKServeEnvoyIngressService(fitID); ok {
			add(gatewayService{Namespace: gwNS, Name: gwSvc, Port: 80})
		}
	}
	if hasInfernexGW {
		if gwNS, gwSvc, ok := discoverInfernexEnvoyIngressService(fitID, ns, owner); ok {
			add(gatewayService{Namespace: gwNS, Name: gwSvc, Port: 80})
		}
	}

	if len(out) == 0 {
		LogStep(fitID, "gateway discovery: no KServe/InferNex envoy svc (kserve-route=%v infernex-gateway=%v)",
			hasKServeRoute, hasInfernexGW)
	}
	return out
}

// gatewayExternalAccessURLs builds http://<nodeIP>:<nodePort><path> for gateway e2e (HTTP port 80's nodePort).
func gatewayExternalAccessURLs(fitID, ns, svcName, path string) []inferenceCandidate {
	nodeIP := InferenceE2ENodeIP(fitID, ns, svcName)
	if nodeIP == "" {
		LogStep(fitID, "gateway %s/%s: no node IP for NodePort access", ns, svcName)
		return nil
	}

	var out []inferenceCandidate
	for _, np := range serviceHTTPNodePorts(fitID, ns, svcName) {
		u := fmt.Sprintf("http://%s:%d%s", nodeIP, np, path)
		out = append(out, enrichCandidateFromURL(inferenceCandidate{
			Name:       fmt.Sprintf("gateway-nodeport:%s/%s:%d", ns, svcName, np),
			URL:        u,
			Access:     "gateway-e2e-nodeport",
			Backend:    fmt.Sprintf("svc/%s/%s NodePort=%d (http)", ns, svcName, np),
			ListenHost: nodeIP,
			ListenPort: np,
			Path:       path,
		}))
	}
	if len(out) == 0 {
		LogStep(fitID, "gateway %s/%s: no HTTP nodePort (check svc type NodePort/LB and port 80)", ns, svcName)
	}
	return out
}

// serviceHTTPNodePorts returns nodePorts for Service ports named http or port 80/8080.
func serviceHTTPNodePorts(fitID, ns, svcName string) []int {
	kc := localKubeconfig()
	cmd := fmt.Sprintf(
		"KUBECONFIG=%q kubectl get svc %q -n %q -o jsonpath='{range .spec.ports[*]}{.name}{\" \"}{.port}{\" \"}{.nodePort}{\"\\n\"}{end}'",
		kc, svcName, ns)
	out, _, err := ExecLocal(cmd)
	if err != nil || strings.TrimSpace(out) == "" {
		return nil
	}
	var ports []int
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 3 {
			continue
		}
		pName := strings.ToLower(fields[0])
		svcPort, _ := strconv.Atoi(fields[1])
		np, err := strconv.Atoi(fields[2])
		if err != nil || np <= 0 {
			continue
		}
		if strings.Contains(pName, "http") || svcPort == 80 || svcPort == 8080 {
			ports = append(ports, np)
			LogStep(fitID, "gateway %s/%s HTTP nodePort=%d (port name=%s svcPort=%d)", ns, svcName, np, fields[0], svcPort)
		}
	}
	return ports
}

func gatewayCRAddresses(fitID, ns, svcName string) []string {
	kc := localKubeconfig()
	gwName := strings.TrimSpace(kubectlJSONPathQuiet(kc, fmt.Sprintf(
		"kubectl get svc %q -n %q -o jsonpath='{.metadata.labels.gateway\\.networking\\.k8s\\.io/gateway-name}'", svcName, ns)))
	if gwName == "" {
		gwName = svcName
	}
	cmd := fmt.Sprintf(
		"KUBECONFIG=%q kubectl get gateway %q -n %q --ignore-not-found -o jsonpath='{range .status.addresses[*]}{.value}{\"\\n\"}{end}'",
		kc, gwName, ns)
	out, _, _ := ExecLocal(cmd)
	var hosts []string
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			hosts = append(hosts, line)
		}
	}
	return hosts
}

func isExternalInferenceURL(raw string) bool {
	u, err := url.Parse(raw)
	if err != nil || u.Host == "" {
		return false
	}
	host := u.Hostname()
	if host == "127.0.0.1" || host == "localhost" {
		return false
	}
	return !strings.Contains(host, ".svc") && !strings.Contains(host, ".cluster.local")
}
