package utils

import (
	"fmt"
	"os"
	"strings"

	. "github.com/onsi/gomega"
)

// ApplyManifest runs kubectl apply -f and returns combined output.
func ApplyManifest(path string) (stdout, stderr string, err error) {
	kc := localKubeconfig()
	cmd := fmt.Sprintf("KUBECONFIG=%q kubectl apply -f %q", kc, path)
	return ExecLocal(cmd)
}

// DeleteManifest runs kubectl delete -f (symmetric to ApplyManifest).
// Use after validating fixtures so multi-doc YAML (Config + InferNexService, etc.) is removed
// even when apply only partially succeeded.
func DeleteManifest(path string) (stdout, stderr string, err error) {
	kc := localKubeconfig()
	cmd := fmt.Sprintf("KUBECONFIG=%q kubectl delete -f %q --ignore-not-found --wait=false", kc, path)
	return ExecLocal(cmd)
}

// ApplyManifestString applies inline YAML via stdin.
func ApplyManifestString(yaml string) (stdout, stderr string, err error) {
	tmp, err := os.CreateTemp("", "infernex-bridge-*.yaml")
	if err != nil {
		return "", "", err
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.WriteString(yaml); err != nil {
		return "", "", err
	}
	tmp.Close()
	return ApplyManifest(tmp.Name())
}

// PatchInferNexService runs kubectl patch on InferNexService.
func PatchInferNexService(ns, name, patchJSON string) (stdout, stderr string, err error) {
	kc := localKubeconfig()
	cmd := fmt.Sprintf("KUBECONFIG=%q kubectl patch infernexservice %q -n %q --type=merge -p %q",
		kc, name, ns, patchJSON)
	stdout, stderr, err = ExecLocal(cmd)
	if err != nil {
		err = fmt.Errorf("%w: %s%s", err, stdout, stderr)
	}
	return stdout, stderr, err
}

// ExpectAdmissionDenied asserts kubectl apply failed with admission (or API) rejection.
// wantSubstr must appear in stdout/stderr (cluster message suffix may include
// "for direct InferNexService (set true or false explicitly)" etc.).
func ExpectAdmissionDenied(fitID, stdout, stderr string, applyErr error, wantSubstr string) {
	combined := strings.TrimSpace(stdout + "\n" + stderr)
	LogStep(fitID, "kubectl apply output:\n%s", combined)
	Expect(applyErr).To(HaveOccurred(), "expected apply to fail (admission or CRD validation)")
	Expect(combined).To(SatisfyAny(
		ContainSubstring(bridgeWebhookName()),
		ContainSubstring("admission webhook"),
		ContainSubstring("denied the request"),
		ContainSubstring("Error from server"),
	))
	Expect(combined).To(ContainSubstring(wantSubstr))
}

func bridgeWebhookName() string {
	return "vinfernexservice-v1alpha1.kb.io"
}

// RenderNamespace replaces __E2E_NS__ placeholder in fixture bytes.
func RenderNamespace(content, ns string) string {
	return strings.ReplaceAll(content, "__E2E_NS__", ns)
}
