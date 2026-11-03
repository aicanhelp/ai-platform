package utils

import (
	"fmt"
	"os"
	"strings"
)

// FixtureTemplate is one of four canonical workload stacks (insvc/llmisvc × aggregate/pd).
type FixtureTemplate string

const (
	FixtureTemplateInsvcAggregate   FixtureTemplate = "insvc_aggregate"
	FixtureTemplateInsvcPD          FixtureTemplate = "insvc_pd"
	FixtureTemplateLLMISvcAggregate FixtureTemplate = "llmisvc_aggregate"
	FixtureTemplateLLMISvcPD        FixtureTemplate = "llmisvc_pd"
)

func (t FixtureTemplate) fileName() string {
	return string(t) + ".yaml"
}

// FixtureOverlay mutates parsed multi-document YAML before serialization.
type FixtureOverlay func(docs []map[string]interface{}) ([]map[string]interface{}, error)

// FixtureRecipe describes how to build a legacy fixture from a canonical template.
type FixtureRecipe struct {
	Template FixtureTemplate
	Overlays []FixtureOverlay
	// LegacyPath is the original fixture relative to fixtures/ (deploy/foo.yaml).
	LegacyPath string
	// TestName is the RenderOptions.Name used in equivalence tests when legacy uses __NAME__.
	TestName string
}

func loadTemplateDocs(template FixtureTemplate) ([]map[string]interface{}, error) {
	path, err := FixturePath("templates", template.fileName())
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return parseYAMLDocuments(data)
}

func loadSnippetDocs(parts ...string) ([]map[string]interface{}, error) {
	path, err := FixturePath(parts...)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return parseYAMLDocuments(data)
}

func loadLegacyDocs(parts ...string) ([]map[string]interface{}, error) {
	path, err := FixturePath(parts...)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return parseYAMLDocuments(data)
}

func cloneDocs(docs []map[string]interface{}) ([]map[string]interface{}, error) {
	raw, err := marshalYAMLDocuments(docs)
	if err != nil {
		return nil, err
	}
	return parseYAMLDocuments(raw)
}

// BuildFixtureFromRecipe renders a fixture from template + overlays, then applies placeholders.
func BuildFixtureFromRecipe(recipe FixtureRecipe, opts RenderOptions) ([]byte, error) {
	docs, err := loadTemplateDocs(recipe.Template)
	if err != nil {
		return nil, err
	}
	for _, overlay := range recipe.Overlays {
		docs, err = overlay(docs)
		if err != nil {
			return nil, fmt.Errorf("overlay on %s: %w", recipe.LegacyPath, err)
		}
	}
	raw, err := marshalYAMLDocuments(docs)
	if err != nil {
		return nil, err
	}
	return applyRenderOptions(raw, opts), nil
}

func applyRenderOptions(data []byte, opts RenderOptions) []byte {
	s := string(data)
	s = stringsReplaceRenderOptions(s, opts)
	return []byte(s)
}

func stringsReplaceRenderOptions(s string, opts RenderOptions) string {
	s = stringsReplaceAll(s, "__E2E_NS__", opts.ns())
	s = stringsReplaceAll(s, "__TEMPLATE_NS__", TemplateNS())
	if opts.Name != "" {
		s = stringsReplaceAll(s, "__NAME__", opts.Name)
	}
	if opts.ExternalGW != "" {
		s = stringsReplaceAll(s, "__EXTERNAL_GW__", opts.ExternalGW)
	} else {
		s = stringsReplaceAll(s, "__EXTERNAL_GW__", "fit-ext-gw")
	}
	if opts.ExternalRoute != "" {
		s = stringsReplaceAll(s, "__EXTERNAL_ROUTE__", opts.ExternalRoute)
	} else {
		s = stringsReplaceAll(s, "__EXTERNAL_ROUTE__", "fit-ext-route")
	}
	if opts.ExternalPool != "" {
		s = stringsReplaceAll(s, "__EXTERNAL_POOL__", opts.ExternalPool)
	} else {
		s = stringsReplaceAll(s, "__EXTERNAL_POOL__", "fit-ext-pool")
	}
	if opts.KServeNS != "" {
		s = stringsReplaceAll(s, "__KSERVE_NS__", opts.KServeNS)
	} else {
		s = stringsReplaceAll(s, "__KSERVE_NS__", KServeNS())
	}
	prefill := opts.PrefillDeploy
	if prefill == "" && opts.Name != "" {
		prefill = opts.Name + "-engine-pd-prefill"
	}
	decode := opts.DecodeDeploy
	if decode == "" && opts.Name != "" {
		decode = opts.Name + "-engine-pd-decode"
	}
	if prefill != "" {
		s = stringsReplaceAll(s, "__PREFILL_DEPLOY__", prefill)
	}
	if decode != "" {
		s = stringsReplaceAll(s, "__DECODE_DEPLOY__", decode)
	}
	return s
}

func stringsReplaceAll(s, old, new string) string {
	return strings.ReplaceAll(s, old, new)
}
