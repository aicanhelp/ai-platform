package utils

import (
	"fmt"
	"strings"

	. "github.com/onsi/gomega"
)

func KubectlJSONPath(fitID, resource, ns, name, path string) string {
	kc := localKubeconfig()
	var nsFlag string
	if ns != "" {
		nsFlag = fmt.Sprintf("-n %q", ns)
	}
	cmd := fmt.Sprintf("KUBECONFIG=%q kubectl get %s %q %s -o jsonpath=%q 2>/dev/null", kc, resource, name, nsFlag, path)
	out, stderr, err := ExecLocal(cmd)
	LogStep(fitID, "kubectl get %s/%s path=%s => %q err=%v stderr=%s", resource, name, path, strings.TrimSpace(out), err, stderr)
	Expect(err).NotTo(HaveOccurred())
	return strings.TrimSpace(out)
}

func KubectlExists(fitID, resource, ns, name string) bool {
	kc := localKubeconfig()
	nsFlag := ""
	if ns != "" {
		nsFlag = fmt.Sprintf("-n %q", ns)
	}
	cmd := fmt.Sprintf("KUBECONFIG=%q kubectl get %s %q %s --ignore-not-found -o name", kc, resource, name, nsFlag)
	out, _, err := ExecLocal(cmd)
	return err == nil && strings.TrimSpace(out) != ""
}

// GatewayAPIInstalled reports whether Gateway API + InferencePool CRDs are registered.
func GatewayAPIInstalled(fitID string) bool {
	return CRDExists(fitID, "gateway.networking.k8s.io", "gateways") &&
		CRDExists(fitID, "gateway.networking.k8s.io", "httproutes") &&
		CRDExists(fitID, "inference.networking.x-k8s.io", "inferencepools")
}

func ManagedResourceNames(owner string) (gateway, route, pool string) {
	return owner + "-infernex-gateway", owner + "-infernex-route", owner + "-inference-pool"
}
