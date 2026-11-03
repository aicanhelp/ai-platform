package utils

import (
	"os"
	"path/filepath"
)

var fixtureRecipes = buildFixtureRegistry()

func buildFixtureRegistry() map[string]FixtureRecipe {
	reg := map[string]FixtureRecipe{}

	add := func(parts []string, recipe FixtureRecipe) {
		key := filepath.ToSlash(filepath.Join(parts...))
		recipe.LegacyPath = key
		reg[key] = recipe
	}

	// --- insvc aggregate (fit06 baseline) ---
	insvcAgg := FixtureRecipe{Template: FixtureTemplateInsvcAggregate}
	insvcAggNoop := insvcAgg
	insvcAggNoop.Overlays = []FixtureOverlay{overlayNoop}

	add([]string{"deploy", "fit06_insvc_aggregate_no_replicas.yaml"}, insvcAggNoop)
	add([]string{"deploy", "fit27_insvc_aggregate_scale.yaml"}, insvcAggNoop)
	add([]string{"deploy", "fit45_insvc_delete_lifecycle.yaml"}, insvcAggNoop)
	add([]string{"deploy", "fit50_mooncake_aggregate_insvc.yaml"}, insvcAggNoop)

	add([]string{"deploy", "fit24_pd_orchestrator_on.yaml"}, FixtureRecipe{
		Template: FixtureTemplateInsvcAggregate,
		Overlays: []FixtureOverlay{overlayPDOrchestrator(true, true, true)},
	})
	add([]string{"deploy", "fit25_pd_orchestrator_off.yaml"}, FixtureRecipe{
		Template: FixtureTemplateInsvcAggregate,
		Overlays: []FixtureOverlay{overlayPDOrchestrator(false, false, false)},
	})

	// --- insvc PD (fit07 baseline) ---
	insvcPDNoop := FixtureRecipe{Template: FixtureTemplateInsvcPD, Overlays: []FixtureOverlay{overlayNoop}}
	add([]string{"deploy", "fit07_insvc_pd_no_replicas.yaml"}, insvcPDNoop)
	add([]string{"deploy", "fit29_insvc_pd_scale.yaml"}, insvcPDNoop)
	add([]string{"deploy", "fit35_gateway_managed.yaml"}, insvcPDNoop)
	add([]string{"deploy", "fit48_mooncake_pd_insvc.yaml"}, insvcPDNoop)

	add([]string{"deploy", "fit56_insvc_aggregate_lws.yaml"}, FixtureRecipe{
		Template: FixtureTemplateInsvcAggregate,
		Overlays: []FixtureOverlay{overlayMoEOfflineInsvcAggregate, overlayInsvcAggregateLWS},
	})
	add([]string{"deploy", "fit58_insvc_pd_lws.yaml"}, FixtureRecipe{
		Template: FixtureTemplateInsvcPD,
		Overlays: []FixtureOverlay{overlayMoEOfflineInsvcPD, overlayInsvcPDLWS},
	})

	add([]string{"deploy", "fit34_router_disabled.yaml"}, FixtureRecipe{
		Template: FixtureTemplateInsvcPD,
		Overlays: []FixtureOverlay{overlayRouterDisabledOnly},
	})
	add([]string{"deploy", "fit36_gateway_ref.yaml"}, FixtureRecipe{
		Template: FixtureTemplateInsvcPD,
		Overlays: []FixtureOverlay{overlayMergeInsvcIGR(igrPatchGatewayRef)},
	})
	add([]string{"deploy", "fit42_mixed_gateway_ref.yaml"}, FixtureRecipe{
		Template: FixtureTemplateInsvcPD,
		Overlays: []FixtureOverlay{overlayMergeInsvcIGR(igrPatchGatewayRef)},
	})
	add([]string{"deploy", "fit37_gateway_custom_spec.yaml"}, FixtureRecipe{
		Template: FixtureTemplateInsvcPD,
		Overlays: []FixtureOverlay{overlayMergeInsvcIGR(igrPatchGatewaySpec)},
	})
	add([]string{"deploy", "fit39_pool_custom_spec.yaml"}, FixtureRecipe{
		Template: FixtureTemplateInsvcPD,
		Overlays: []FixtureOverlay{overlayMergeInsvcIGR(igrPatchInferencePoolSpec)},
	})
	add([]string{"deploy", "fit41_httproute_spec_parentrefs.yaml"}, FixtureRecipe{
		Template: FixtureTemplateInsvcPD,
		Overlays: []FixtureOverlay{overlayMergeInsvcIGR(igrPatchHTTPRouteSpec)},
	})
	add([]string{"deploy", "fit38_external_pool.yaml"}, FixtureRecipe{
		Template: FixtureTemplateInsvcPD,
		Overlays: []FixtureOverlay{
			overlayPrependSnippetDocKinds([]string{"snippets", "external_inference_pool.yaml"}, "InferencePool"),
			overlayMergeInsvcIGR(igrPatchInferencePoolRef),
			overlayReplaceRouterPoolNameArg("__EXTERNAL_POOL__"),
		},
	})
	add([]string{"deploy", "fit40_external_httproute.yaml"}, FixtureRecipe{
		Template: FixtureTemplateInsvcPD,
		Overlays: []FixtureOverlay{
			overlayPrependSnippetDocKinds([]string{"snippets", "external_httproute.yaml"}, "HTTPRoute"),
			overlayMergeInsvcIGR(igrPatchHTTPRouteRef),
		},
	})

	// --- llmisvc aggregate (fit04 baseline) ---
	llmAggNoop := FixtureRecipe{Template: FixtureTemplateLLMISvcAggregate, Overlays: []FixtureOverlay{overlayNoop}}
	add([]string{"llmisvc", "fit04_llmisvc_aggregate_no_storage_initializer.yaml"}, llmAggNoop)
	add([]string{"llmisvc", "fit44_llmisvc_delete_lifecycle.yaml"}, llmAggNoop)
	add([]string{"llmisvc", "fit46_llmisvc_runtime_label_lifecycle.yaml"}, llmAggNoop)
	add([]string{"llmisvc", "fit51_mooncake_aggregate_llmisvc.yaml"}, llmAggNoop)

	add([]string{"llmisvc", "fit26_llmisvc_aggregate_scale.yaml"}, FixtureRecipe{
		Template: FixtureTemplateLLMISvcAggregate,
		Overlays: []FixtureOverlay{overlayRemoveLLMISvcReplicas},
	})

	// --- llmisvc PD (fit05 baseline) ---
	llmPDNoop := FixtureRecipe{Template: FixtureTemplateLLMISvcPD, Overlays: []FixtureOverlay{overlayNoop}}
	add([]string{"llmisvc", "fit05_llmisvc_pd_no_storage_initializer.yaml"}, llmPDNoop)
	add([]string{"llmisvc", "fit49_mooncake_pd_llmisvc.yaml"}, llmPDNoop)

	add([]string{"llmisvc", "fit28_llmisvc_pd_scale.yaml"}, FixtureRecipe{
		Template: FixtureTemplateLLMISvcPD,
		Overlays: []FixtureOverlay{overlayRemoveLLMISvcPDScaleReplicas},
	})

	add([]string{"llmisvc", "fit57_llmisvc_aggregate_lws.yaml"}, FixtureRecipe{
		Template: FixtureTemplateLLMISvcAggregate,
		Overlays: []FixtureOverlay{overlayMoEOfflineLLMISvcAggregate, overlayLLMISvcAggregateLWS},
	})
	add([]string{"llmisvc", "fit59_llmisvc_pd_lws.yaml"}, FixtureRecipe{
		Template: FixtureTemplateLLMISvcPD,
		Overlays: []FixtureOverlay{overlayMoEOfflineLLMISvcPD, overlayLLMISvcPDLWS},
	})

	// --- validating (insvc aggregate + targeted violation) ---
	validatingName := func(fit string) string { return fit }
	add([]string{"validating", "fit11_no_engine.yaml"}, FixtureRecipe{
		Template: FixtureTemplateInsvcAggregate,
		Overlays: []FixtureOverlay{overlayDropInsvcConfigDoc, overlayRemoveInsvcBaseRefs},
		TestName: "fit11-no-engine",
	})
	add([]string{"validating", "fit12_cache_indexer_no_enabled.yaml"}, FixtureRecipe{
		Template: FixtureTemplateInsvcAggregate,
		Overlays: []FixtureOverlay{overlayMergeInsvcComponents(compPatchCacheIndexerNoEnabled)},
		TestName: validatingName("fit12-cache-indexer"),
	})
	add([]string{"validating", "fit13_mooncake_no_enabled.yaml"}, FixtureRecipe{
		Template: FixtureTemplateInsvcAggregate,
		Overlays: []FixtureOverlay{overlayMergeInsvcComponents(compPatchMooncakeNoEnabled)},
		TestName: validatingName("fit13-mooncake"),
	})
	add([]string{"validating", "fit14_tidal_no_enabled.yaml"}, FixtureRecipe{
		Template: FixtureTemplateInsvcAggregate,
		Overlays: []FixtureOverlay{overlayMergeInsvcComponents(compPatchTidalNoEnabled)},
		TestName: validatingName("fit14-tidal"),
	})
	add([]string{"validating", "fit15_eagleeye_no_enabled.yaml"}, FixtureRecipe{
		Template: FixtureTemplateInsvcAggregate,
		Overlays: []FixtureOverlay{overlayMergeInsvcComponents(compPatchEagleEyeNoEnabled)},
		TestName: validatingName("fit15-eagleeye"),
	})
	add([]string{"validating", "fit15_npe_no_enabled.yaml"}, FixtureRecipe{
		Template: FixtureTemplateInsvcAggregate,
		Overlays: []FixtureOverlay{overlayMergeInsvcComponents(compPatchNPENoEnabled)},
		TestName: validatingName("fit15-npe"),
	})
	add([]string{"validating", "fit16_router_no_enabled.yaml"}, FixtureRecipe{
		Template: FixtureTemplateInsvcAggregate,
		Overlays: []FixtureOverlay{overlayRouterNoEnabled},
		TestName: validatingName("fit16-router"),
	})
	add([]string{"validating", "fit17_router_no_template.yaml"}, FixtureRecipe{
		Template: FixtureTemplateInsvcAggregate,
		Overlays: []FixtureOverlay{overlayRouterNoTemplate},
		TestName: validatingName("fit17-router-template"),
	})
	add([]string{"validating", "fit18_httproute_ref_spec.yaml"}, FixtureRecipe{
		Template: FixtureTemplateInsvcAggregate,
		Overlays: []FixtureOverlay{overlayMergeInsvcIGR(igrPatchHTTPRouteRefSpecConflict)},
		TestName: validatingName("fit18-httproute"),
	})
	add([]string{"validating", "fit19_gateway_ref_spec.yaml"}, FixtureRecipe{
		Template: FixtureTemplateInsvcAggregate,
		Overlays: []FixtureOverlay{overlayMergeInsvcIGR(igrPatchGatewayRefSpecConflict)},
		TestName: validatingName("fit19-gateway"),
	})
	add([]string{"validating", "fit20_pool_ref_spec.yaml"}, FixtureRecipe{
		Template: FixtureTemplateInsvcAggregate,
		Overlays: []FixtureOverlay{overlayMergeInsvcIGR(igrPatchPoolRefSpecConflict)},
		TestName: validatingName("fit20-pool"),
	})
	add([]string{"validating", "fit21_engine_empty_image.yaml"}, FixtureRecipe{
		Template: FixtureTemplateInsvcAggregate,
		Overlays: []FixtureOverlay{overlaySetEngineMainImage("")},
		TestName: validatingName("fit21-empty-image"),
	})

	add([]string{"validating", "fit60_router_no_main_epp.yaml"}, FixtureRecipe{
		Template: FixtureTemplateInsvcAggregate,
		Overlays: []FixtureOverlay{overlayRouterMainRenamedToEPP},
		TestName: validatingName("fit60-router-no-main"),
	})
	add([]string{"validating", "fit61_router_main_no_grpc.yaml"}, FixtureRecipe{
		Template: FixtureTemplateInsvcAggregate,
		Overlays: []FixtureOverlay{overlayRouterMainNoGRPCPort},
		TestName: validatingName("fit61-router-no-grpc"),
	})
	add([]string{"validating", "fit62_engine_dp_indivisible.yaml"}, FixtureRecipe{
		Template: FixtureTemplateInsvcAggregate,
		Overlays: []FixtureOverlay{overlayEngineIndivisibleDataParallel},
		TestName: validatingName("fit62-dp-indivisible"),
	})
	add([]string{"validating", "fit63_engine_worker_group_size_one.yaml"}, FixtureRecipe{
		Template: FixtureTemplateInsvcAggregate,
		Overlays: []FixtureOverlay{overlayEngineWorkerWhenGroupSizeOne},
		TestName: validatingName("fit63-worker-group1"),
	})

	return reg
}

// LookupFixtureRecipe returns a recipe when parts identify a generated fixture.
func LookupFixtureRecipe(parts []string) (FixtureRecipe, bool) {
	key := filepath.ToSlash(filepath.Join(parts...))
	recipe, ok := fixtureRecipes[key]
	return recipe, ok
}

// FixtureUsesTemplate reports whether the fixture is built from a canonical template.
func FixtureUsesTemplate(parts []string) bool {
	_, ok := LookupFixtureRecipe(parts)
	return ok
}

// RenderFixtureBytes loads a fixture: generated from template when registered, else legacy file.
func RenderFixtureBytesFromTemplate(parts []string, opts RenderOptions) ([]byte, error) {
	if recipe, ok := LookupFixtureRecipe(parts); ok {
		return BuildFixtureFromRecipe(recipe, opts)
	}
	path, err := FixturePath(parts...)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return applyRenderOptions(data, opts), nil
}

// AllFixtureRecipesForTest exposes the registry for unit tests.
func AllFixtureRecipesForTest() map[string]FixtureRecipe {
	out := make(map[string]FixtureRecipe, len(fixtureRecipes))
	for k, v := range fixtureRecipes {
		out[k] = v
	}
	return out
}
