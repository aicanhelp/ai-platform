package controller

import (
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestComponentPatchHelpers(t *testing.T) {
	t.Parallel()

	t.Run("openfuyao and llmisvc pod selector strings", func(t *testing.T) {
		got := openfuyaoPDPodSelectorString(openfuyaoPDRolePrefill, "demo")
		if !strings.Contains(got, "openfuyao.com/pdRole=prefill") || !strings.Contains(got, "openfuyao.com/pdGroupID=demo") {
			t.Fatalf("unexpected openfuyao selector: %q", got)
		}
		pref := llmisvcProxyPrefillSelector("demo")
		if !strings.Contains(pref, " in (") ||
			!strings.Contains(pref, kserveWorkloadComponentPrefill) ||
			!strings.Contains(pref, kserveWorkloadComponentLeaderPrefill) ||
			!strings.Contains(pref, kserveWorkloadComponentWorkerPrefill) {
			t.Fatalf("unexpected proxy prefill selector: %q", pref)
		}
		dec := llmisvcProxyDecodeSelector("demo")
		if !strings.Contains(dec, kserveWorkloadComponentLeader) ||
			!strings.Contains(dec, kserveWorkloadComponentWorker) {
			t.Fatalf("unexpected proxy decode selector: %q", dec)
		}
	})

	t.Run("mergePodTemplateLabelIfAbsent keeps user labels", func(t *testing.T) {
		tpl := &corev1.PodTemplateSpec{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"keep": "user"}}}
		mergePodTemplateLabelIfAbsent(tpl, "keep", "override")
		mergePodTemplateLabelIfAbsent(tpl, "new", "v")
		if tpl.Labels["keep"] != "user" || tpl.Labels["new"] != "v" {
			t.Fatalf("unexpected labels: %#v", tpl.Labels)
		}
	})

	t.Run("mergePDInferenceWorkloadPodTemplateLabels", func(t *testing.T) {
		tpl := &corev1.PodTemplateSpec{}
		mergePDInferenceWorkloadPodTemplateLabels(tpl, "infernex-bridge-system", "demo", true)
		wantGroup := infernexEngineGroupValue("infernex-bridge-system", "demo")
		if tpl.Labels[labelInfernexPDEngineGroup] != wantGroup ||
			tpl.Labels[labelOpenFuyaoPDRole] != openfuyaoPDRolePrefill ||
			tpl.Labels[labelOpenFuyaoPDGroup] != "demo" {
			t.Fatalf("unexpected pd labels: %#v", tpl.Labels)
		}
	})

	t.Run("mergeProxyPDLabelsAndEnv direct pd", func(t *testing.T) {
		tpl := &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main"}}}}
		mergeProxyPDLabelsAndEnv(tpl, "demo", "ns-a", defaultProxyDiscoverySec, false)
		if tpl.Labels[labelOpenFuyaoPDRole] != openfuyaoPDRoleLeader {
			t.Fatalf("expected leader pd role, got %#v", tpl.Labels)
		}
		found := false
		for _, e := range tpl.Spec.Containers[0].Env {
			if e.Name == envProxyWorkloadNS && e.Value == "ns-a" {
				found = true
			}
		}
		if !found {
			t.Fatalf("expected proxy workload ns env, got %#v", tpl.Spec.Containers[0].Env)
		}
	})

	t.Run("mergeProxyPDLabelsAndEnv kserve linked backend", func(t *testing.T) {
		tpl := &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "proxy-server"}}}}
		mergeProxyPDLabelsAndEnv(tpl, "demo", "ns-a", 0, true)
		if tpl.Labels[labelKServeComponent] != "workload" || tpl.Labels[labelAppKubernetesIOComponent] != valueKServeProxyComponent {
			t.Fatalf("expected kserve proxy labels, got %#v", tpl.Labels)
		}
		foundPref, foundDec := false, false
		for _, e := range tpl.Spec.Containers[0].Env {
			if e.Name == envProxyLegacyPrefillArg && strings.Contains(e.Value, " in (") &&
				strings.Contains(e.Value, kserveWorkloadComponentLeaderPrefill) {
				foundPref = true
			}
			if e.Name == envProxyLegacyDecoderArg && strings.Contains(e.Value, kserveWorkloadComponentLeader) {
				foundDec = true
			}
		}
		if !foundPref || !foundDec {
			t.Fatalf("expected kserve prefill/decode selector env, got %#v", tpl.Spec.Containers[0].Env)
		}
	})

	t.Run("mergeEnvVar updates existing and appends new", func(t *testing.T) {
		c := &corev1.Container{Name: "main", Env: []corev1.EnvVar{{Name: "KEEP", Value: "1"}}}
		mergeEnvVar(c, "KEEP", "override")
		mergeEnvVar(c, "NEW", "v")
		if len(c.Env) != 2 || c.Env[0].Value != "override" || c.Env[1].Name != "NEW" {
			t.Fatalf("unexpected env after mergeEnvVar: %#v", c.Env)
		}
	})

	t.Run("preferredContainer resolves named container", func(t *testing.T) {
		tpl := &corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "sidecar"}, {Name: "cache-indexer"}}},
		}
		if preferredContainer(nil) != nil {
			t.Fatal("nil template should return nil container")
		}
		got := preferredContainer(tpl, "cache-indexer", "main")
		if got == nil || got.Name != "cache-indexer" {
			t.Fatalf("expected cache-indexer container, got %#v", got)
		}
	})
}
