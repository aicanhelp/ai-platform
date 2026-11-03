package assets

import (
	"strings"
	"testing"
)

func TestLoadManagedComponent(t *testing.T) {
	t.Parallel()

	t.Run("known component loads with template", func(t *testing.T) {
		t.Parallel()
		spec, err := LoadManagedComponent("cache-indexer")
		if err != nil {
			t.Fatalf("LoadManagedComponent(cache-indexer) error: %v", err)
		}
		if spec == nil || spec.Template == nil {
			t.Fatal("expected non-nil component template")
		}
		if len(spec.Template.Spec.Containers) == 0 {
			t.Fatal("expected template to contain containers")
		}
	})

	t.Run("unknown component returns error", func(t *testing.T) {
		t.Parallel()
		_, err := LoadManagedComponent("unknown-component")
		if err == nil {
			t.Fatal("expected error for unknown component")
		}
	})

	t.Run("all bundled components load", func(t *testing.T) {
		t.Parallel()
		for name := range componentTemplateFiles {
			spec, err := LoadManagedComponent(name)
			if err != nil {
				t.Fatalf("LoadManagedComponent(%q) error: %v", name, err)
			}
			if spec == nil || spec.Template == nil || len(spec.Template.Spec.Containers) == 0 {
				t.Fatalf("expected valid template for %q, got %#v", name, spec)
			}
		}
	})

	t.Run("proxy-server template includes proxy container", func(t *testing.T) {
		t.Parallel()
		spec, err := LoadManagedComponent("proxy-server")
		if err != nil {
			t.Fatalf("LoadManagedComponent(proxy-server) error: %v", err)
		}
		found := false
		for _, c := range spec.Template.Spec.Containers {
			if c.Name == "proxy-server" {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("expected proxy-server container in template, got %#v", spec.Template.Spec.Containers)
		}
	})

	t.Run("cache-indexer and elastic-scaler bundle openfuyao images with webhook args", func(t *testing.T) {
		t.Parallel()
		ci, err := LoadManagedComponent("cache-indexer")
		if err != nil {
			t.Fatalf("LoadManagedComponent(cache-indexer) error: %v", err)
		}
		if tag, ok := openfuyaoImageTag(ci.Template.Spec.Containers[0].Image, "cache-indexer"); !ok {
			t.Fatalf("unexpected cache-indexer image: %s (tag %q)", ci.Template.Spec.Containers[0].Image, tag)
		}
		es, err := LoadManagedComponent("pd-orchestrator-elastic-scaler")
		if err != nil {
			t.Fatalf("LoadManagedComponent(elastic-scaler) error: %v", err)
		}
		c := es.Template.Spec.Containers[0]
		if tag, ok := openfuyaoImageTag(c.Image, "elastic-scaler"); !ok {
			t.Fatalf("unexpected elastic-scaler image: %s (tag %q)", c.Image, tag)
		}
		foundWebhook := false
		for _, arg := range c.Args {
			if arg == "--enable-admission-webhook" {
				foundWebhook = true
			}
		}
		if !foundWebhook {
			t.Fatalf("expected --enable-admission-webhook in args, got %#v", c.Args)
		}
	})

	t.Run("component template file map covers bundled assets", func(t *testing.T) {
		t.Parallel()
		if len(componentTemplateFiles) != 9 {
			t.Fatalf("expected 9 bundled components, got %d", len(componentTemplateFiles))
		}
		for _, file := range componentTemplateFiles {
			if file == "" {
				t.Fatal("expected non-empty template file path")
			}
		}
	})
}

// openfuyaoImageTag checks bundled asset images use the openfuyao registry/repo and a non-empty tag.
// Tag value is intentionally not pinned so master (e.g. 0.22.0) and release branches can diverge.
func openfuyaoImageTag(image, component string) (string, bool) {
	const prefix = "cr.openfuyao.cn/openfuyao/"
	if !strings.HasPrefix(image, prefix) {
		return "", false
	}
	rest := strings.TrimPrefix(image, prefix)
	name, tag, ok := strings.Cut(rest, ":")
	if !ok || name != component || strings.TrimSpace(tag) == "" {
		return "", false
	}
	return tag, true
}

