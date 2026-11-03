package utils

import (
	"fmt"
	"time"

	. "github.com/onsi/gomega"
)

// WaitKubectlResourceExists polls until kubectl get resource/name in ns succeeds.
func WaitKubectlResourceExists(fitID, resource, ns, name string, timeout time.Duration) {
	phase := "kubectl resource exists " + resource + "/" + name
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)
	Eventually(func(g Gomega) {
		exists := KubectlExists(fitID, resource, ns, name)
		if !exists {
			LogPollThrottled(fitID, "exists-"+resource+"-"+name, defaultPollLogInterval,
				"%s/%s not found yet in ns=%s", resource, name, ns)
		}
		g.Expect(exists).To(BeTrue(), "%s/%s in ns=%s", resource, name, ns)
	}).WithTimeout(timeout).WithPolling(5 * time.Second).Should(Succeed())
}

// WaitManagedGatewayTripleExists waits for Gateway, HTTPRoute, InferencePool of an InferNexService owner.
func WaitManagedGatewayTripleExists(fitID, ns, owner string, timeout time.Duration) {
	gw, route, pool := ManagedResourceNames(owner)
	LogPhaseBegin(fitID, "managed gateway triple owner="+owner)
	defer LogPhaseEnd(fitID, "managed gateway triple owner="+owner)
	Eventually(func(g Gomega) {
		gwOK := KubectlExists(fitID, "gateway", ns, gw)
		routeOK := KubectlExists(fitID, "httproute", ns, route)
		poolOK := KubectlExists(fitID, "inferencepool", ns, pool)
		if !gwOK || !routeOK || !poolOK {
			LogPollThrottled(fitID, "gw-triple-"+owner, defaultPollLogInterval,
				"gateway=%v httproute=%v inferencepool=%v (want all true)", gwOK, routeOK, poolOK)
		}
		g.Expect(gwOK).To(BeTrue())
		g.Expect(routeOK).To(BeTrue())
		g.Expect(poolOK).To(BeTrue())
	}).WithTimeout(timeout).WithPolling(5 * time.Second).Should(Succeed())
	LogStep(fitID, "managed triple ready gw=%s route=%s pool=%s", gw, route, pool)
}

// WaitInferNexEnvoyIngressServiceReady waits until Envoy Gateway publishes a NodePort/LB Service for the managed Gateway CR.
func WaitInferNexEnvoyIngressServiceReady(fitID, insvcNS, owner string, timeout time.Duration) {
	if SkipInferenceE2E() {
		return
	}
	phase := fmt.Sprintf("InferNex envoy ingress svc owner=%s", owner)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)
	Eventually(func(g Gomega) {
		svcNS, svcName, ok := discoverInfernexEnvoyIngressService(fitID, insvcNS, owner)
		if !ok {
			gwCR := infernexGatewayCRForDiscovery(fitID, insvcNS, owner)
			g.Expect(ok).To(BeTrue(), "envoy ingress service for gateway %s/%s", insvcNS, gwCR)
			return
		}
		nps := serviceHTTPNodePorts(fitID, svcNS, svcName)
		if len(nps) == 0 {
			LogPollThrottled(fitID, "infernex-envoy-np-"+owner, defaultPollLogInterval,
				"svc %s/%s exists but no HTTP nodePort yet", svcNS, svcName)
		}
		g.Expect(len(nps)).To(BeNumerically(">", 0), "svc %s/%s HTTP nodePort", svcNS, svcName)
		LogPollThrottled(fitID, "infernex-envoy-ready-"+owner, defaultPollLogInterval,
			"InferNex envoy ingress ready svc=%s/%s nodePorts=%v", svcNS, svcName, nps)
	}).WithTimeout(timeout).WithPolling(10 * time.Second).Should(Succeed())
}

// WaitKubectlResourceAbsent polls until resource/name is gone from ns.
func WaitKubectlResourceAbsent(fitID, resource, ns, name string, timeout time.Duration) {
	phase := "kubectl resource absent " + resource + "/" + name
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)
	Eventually(func(g Gomega) {
		exists := KubectlExists(fitID, resource, ns, name)
		if exists {
			LogPollThrottled(fitID, "absent-"+resource+"-"+name, defaultPollLogInterval,
				"%s/%s still exists in ns=%s", resource, name, ns)
		}
		g.Expect(exists).To(BeFalse())
	}).WithTimeout(timeout).WithPolling(5 * time.Second).Should(Succeed())
}

// WaitManagedGatewayTripleAbsent waits until managed Gateway/HTTPRoute/InferencePool are deleted.
func WaitManagedGatewayTripleAbsent(fitID, ns, owner string, timeout time.Duration) {
	gw, route, pool := ManagedResourceNames(owner)
	LogPhaseBegin(fitID, "managed gateway triple absent owner="+owner)
	defer LogPhaseEnd(fitID, "managed gateway triple absent owner="+owner)
	Eventually(func(g Gomega) {
		gwOK := !KubectlExists(fitID, "gateway", ns, gw)
		routeOK := !KubectlExists(fitID, "httproute", ns, route)
		poolOK := !KubectlExists(fitID, "inferencepool", ns, pool)
		if !gwOK || !routeOK || !poolOK {
			LogPollThrottled(fitID, "gw-triple-absent-"+owner, defaultPollLogInterval,
				"gateway gone=%v httproute gone=%v pool gone=%v", gwOK, routeOK, poolOK)
		}
		g.Expect(gwOK).To(BeTrue())
		g.Expect(routeOK).To(BeTrue())
		g.Expect(poolOK).To(BeTrue())
	}).WithTimeout(timeout).WithPolling(5 * time.Second).Should(Succeed())
}
