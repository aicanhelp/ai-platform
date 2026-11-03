package v1alpha1

import (
	"os"
	"testing"
)

func TestInfernexTemplateNamespaceForWebhook(t *testing.T) {
	t.Run("prefer INFERNEX_TEMPLATE_NAMESPACE", func(t *testing.T) {
		t.Setenv("INFERNEX_TEMPLATE_NAMESPACE", " tpl-ns ")
		t.Setenv("POD_NAMESPACE", "pod-ns")
		if got := infernexTemplateNamespaceForWebhook(); got != "tpl-ns" {
			t.Fatalf("expected tpl-ns, got %q", got)
		}
	})

	t.Run("fallback to POD_NAMESPACE", func(t *testing.T) {
		t.Setenv("INFERNEX_TEMPLATE_NAMESPACE", "")
		t.Setenv("POD_NAMESPACE", " pod-ns ")
		if got := infernexTemplateNamespaceForWebhook(); got != "pod-ns" {
			t.Fatalf("expected pod-ns, got %q", got)
		}
	})

	t.Run("empty when both missing", func(t *testing.T) {
		_ = os.Unsetenv("INFERNEX_TEMPLATE_NAMESPACE")
		_ = os.Unsetenv("POD_NAMESPACE")
		if got := infernexTemplateNamespaceForWebhook(); got != "" {
			t.Fatalf("expected empty namespace, got %q", got)
		}
	})
}

