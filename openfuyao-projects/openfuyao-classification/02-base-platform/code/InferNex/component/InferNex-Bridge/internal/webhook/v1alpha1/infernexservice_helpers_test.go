package v1alpha1

import (
	"testing"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

func TestValidationHelperConverters(t *testing.T) {
	t.Parallel()
	if got := componentSpecOrNil[infernexv1alpha1.CacheIndexerComponentSpec](nil, func(spec *infernexv1alpha1.CacheIndexerComponentSpec) *infernexv1alpha1.ComponentSpec {
		t.Fatalf("getter should not be called for nil input")
		return nil
	}); got != nil {
		t.Fatalf("expected nil componentSpecOrNil result, got %#v", got)
	}

	cacheEnabled := true
	cacheSpec := &infernexv1alpha1.CacheIndexerComponentSpec{Enabled: &cacheEnabled}
	c := componentSpecOrNil(cacheSpec, componentSpecFromCacheIndexer)
	if c == nil || c.Enabled == nil || !*c.Enabled {
		t.Fatalf("expected componentSpecOrNil converted enabled=true, got %#v", c)
	}

	if got := enabledComponentSpecOrNil[infernexv1alpha1.PDOrchestratorComponentSpec](nil, func(*infernexv1alpha1.PDOrchestratorComponentSpec) *infernexv1alpha1.EnabledComponentSpec {
		t.Fatalf("getter should not be called for nil input")
		return nil
	}); got != nil {
		t.Fatalf("expected nil enabledComponentSpecOrNil result for nil input, got %#v", got)
	}
	pd := &infernexv1alpha1.PDOrchestratorComponentSpec{}
	if got := enabledComponentSpecOrNil(pd, func(*infernexv1alpha1.PDOrchestratorComponentSpec) *infernexv1alpha1.EnabledComponentSpec {
		return nil
	}); got != nil {
		t.Fatalf("expected nil when getter returns nil enabled spec, got %#v", got)
	}
	enabledFalse := false
	got := enabledComponentSpecOrNil(pd, func(*infernexv1alpha1.PDOrchestratorComponentSpec) *infernexv1alpha1.EnabledComponentSpec {
		return &infernexv1alpha1.EnabledComponentSpec{Enabled: &enabledFalse}
	})
	if got == nil || got.Enabled == nil || *got.Enabled {
		t.Fatalf("expected enabled=false conversion, got %#v", got)
	}
}

func TestSetStringMap(t *testing.T) {
	t.Parallel()
	obj := map[string]interface{}{}
	setStringMap(obj, map[string]string{"a": "1", "b": "2"}, "metadata", "labels")
	labels, ok := obj["metadata"].(map[string]interface{})["labels"].(map[string]interface{})
	if !ok || labels["a"] != "1" || labels["b"] != "2" {
		t.Fatalf("expected labels set by setStringMap, got %#v", obj)
	}
}

