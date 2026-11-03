package utils

import (
	"context"
	"fmt"
	"strings"
	"time"

	. "github.com/onsi/gomega"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/client-go/dynamic"
)

const (
	LLMConfigMutatedAnnotation = "infernex.io/llm-inference-service-config-mutated"
	LLMConfigMutatedValue      = "infernex"
)

// MissingLLMInferenceServiceConfigs returns config names not found in ns.
func MissingLLMInferenceServiceConfigs(ctx context.Context, dyn dynamic.Interface, ns string, names []string) []string {
	var missing []string
	for _, name := range names {
		if _, err := GetLLMInferenceServiceConfig(ctx, dyn, ns, name); err != nil {
			if apierrors.IsNotFound(err) {
				missing = append(missing, name)
				continue
			}
			missing = append(missing, name+" (get error: "+err.Error()+")")
		}
	}
	return missing
}

// SnapshotLLMConfigResourceVersions maps config name -> resourceVersion (empty if object missing).
func SnapshotLLMConfigResourceVersions(ctx context.Context, dyn dynamic.Interface, ns string, names []string) map[string]string {
	out := make(map[string]string, len(names))
	for _, name := range names {
		cfg, err := GetLLMInferenceServiceConfig(ctx, dyn, ns, name)
		if err != nil {
			out[name] = ""
			continue
		}
		out[name] = cfg.GetResourceVersion()
	}
	return out
}

// ConfigInferNexMutated reports whether the InferNex mutating webhook marked and patched the config.
func ConfigInferNexMutated(cfg *unstructured.Unstructured) (bool, string) {
	if cfg == nil {
		return false, "nil config"
	}
	if ConfigMutatedAnnotation(cfg) != LLMConfigMutatedValue {
		return false, fmt.Sprintf("annotation %q want %q", ConfigMutatedAnnotation(cfg), LLMConfigMutatedValue)
	}
	// Bridge clears llm-d init sidecars on spec.template; capabilities.drop ALL is preserved.
	if _, found, _ := unstructured.NestedMap(cfg.Object, "spec", "template"); found {
		inits, hasInits, err := unstructured.NestedSlice(cfg.Object, "spec", "template", "initContainers")
		if err != nil {
			return false, fmt.Sprintf("spec.template.initContainers: %v", err)
		}
		if hasInits && len(inits) > 0 {
			return false, "spec.template.initContainers should be cleared"
		}
	}
	return true, ""
}

// WaitKServeConfigsInferNexMutated waits until all named configs pass ConfigInferNexMutated checks.
func WaitKServeConfigsInferNexMutated(ctx context.Context, dyn dynamic.Interface, fitID, ns string, names []string, beforeRV, beforeAnn map[string]string, timeout time.Duration) {
	LogPhaseBegin(fitID, fmt.Sprintf("kserve configs mutated count=%d", len(names)))
	defer LogPhaseEnd(fitID, "kserve configs mutated")
	Eventually(func(g Gomega) {
		for _, cfgName := range names {
			cfg, err := GetLLMInferenceServiceConfig(ctx, dyn, ns, cfgName)
			g.Expect(err).NotTo(HaveOccurred())
			ok, reason := ConfigInferNexMutated(cfg)
			if !ok {
				LogPollThrottled(fitID, "mutate-"+cfgName, defaultPollLogInterval, "config %s not mutated: %s", cfgName, reason)
			}
			g.Expect(ok).To(BeTrue(), "config %s: %s", cfgName, reason)
			ann := ConfigMutatedAnnotation(cfg)
			if beforeAnn[cfgName] != LLMConfigMutatedValue {
				g.Expect(ann).To(Equal(LLMConfigMutatedValue), "config %s should be patched on first mutating run", cfgName)
				g.Expect(cfg.GetResourceVersion()).NotTo(Equal(beforeRV[cfgName]),
					"config %s resourceVersion should change when newly patched", cfgName)
			} else {
				g.Expect(cfg.GetResourceVersion()).To(Equal(beforeRV[cfgName]),
					"config %s should stay idempotent when already mutated", cfgName)
			}
		}
	}).WithTimeout(timeout).WithPolling(5 * time.Second).Should(Succeed())
}

// LLMISVCHasInfernexRuntimeLabel matches mutating webhook runtimeLabelKey/Value.
func LLMISVCHasInfernexRuntimeLabel(obj *unstructured.Unstructured) bool {
	if obj == nil {
		return false
	}
	labels := obj.GetLabels()
	if labels == nil {
		return false
	}
	return strings.EqualFold(strings.TrimSpace(labels["infernex.io/runtime"]), "true")
}
