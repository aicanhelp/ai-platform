package utils

import (
	"fmt"
)

func overlayNoop(docs []map[string]interface{}) ([]map[string]interface{}, error) {
	return docs, nil
}

func overlayDropInsvcConfigDoc(docs []map[string]interface{}) ([]map[string]interface{}, error) {
	return dropDocsExceptKinds(docs, "InferNexService"), nil
}

func overlayRemoveInsvcBaseRefs(docs []map[string]interface{}) ([]map[string]interface{}, error) {
	insvc, err := insvcDoc(docs)
	if err != nil {
		return nil, err
	}
	deleteMapPath(insvc, "spec", "baseRefs")
	return docs, nil
}

func overlaySetEngineMainImage(image string) func([]map[string]interface{}) ([]map[string]interface{}, error) {
	return func(docs []map[string]interface{}) ([]map[string]interface{}, error) {
		cfg, err := insvcConfigDoc(docs)
		if err != nil {
			return nil, err
		}
		spec := mustMap(cfg["spec"])
		engine := mustMap(spec["engine"])
		tpl := mustMap(engine["template"])
		podSpec := mustMap(tpl["spec"])
		containers, ok := podSpec["containers"].([]interface{})
		if !ok || len(containers) == 0 {
			return nil, fmt.Errorf("engine template containers missing")
		}
		main := mustMap(containers[0].(map[string]interface{}))
		main["image"] = image
		containers[0] = main
		podSpec["containers"] = containers
		return docs, nil
	}
}

func overlayPDOrchestrator(elastic, tidal, rsg bool) func([]map[string]interface{}) ([]map[string]interface{}, error) {
	return func(docs []map[string]interface{}) ([]map[string]interface{}, error) {
		insvc, err := insvcDoc(docs)
		if err != nil {
			return nil, err
		}
		spec := mustMap(insvc["spec"])
		spec["components"] = map[string]interface{}{
			"pdOrchestrator": map[string]interface{}{
				"elasticScaler": map[string]interface{}{"enabled": elastic},
				"tidal":         map[string]interface{}{"enabled": tidal},
				"resourceScalingGroup": map[string]interface{}{
					"enabled": rsg,
				},
			},
		}
		insvc["spec"] = spec
		return docs, nil
	}
}

func overlayRouterDisabledOnly(docs []map[string]interface{}) ([]map[string]interface{}, error) {
	insvc, err := insvcDoc(docs)
	if err != nil {
		return nil, err
	}
	spec := mustMap(insvc["spec"])
	spec["intelligentGatewayRouting"] = map[string]interface{}{
		"router": map[string]interface{}{"enabled": false},
	}
	insvc["spec"] = spec
	return docs, nil
}

func overlayMergeInsvcIGR(patchYAML string) func([]map[string]interface{}) ([]map[string]interface{}, error) {
	return func(docs []map[string]interface{}) ([]map[string]interface{}, error) {
		insvc, err := insvcDoc(docs)
		if err != nil {
			return nil, err
		}
		spec := mustMap(insvc["spec"])
		igr := mustMap(spec["intelligentGatewayRouting"])
		if err := mergeYAMLPatch(igr, patchYAML); err != nil {
			return nil, err
		}
		spec["intelligentGatewayRouting"] = igr
		insvc["spec"] = spec
		return docs, nil
	}
}

func overlayMergeInsvcComponents(patchYAML string) func([]map[string]interface{}) ([]map[string]interface{}, error) {
	return func(docs []map[string]interface{}) ([]map[string]interface{}, error) {
		insvc, err := insvcDoc(docs)
		if err != nil {
			return nil, err
		}
		spec := mustMap(insvc["spec"])
		components := mustMap(spec["components"])
		if err := mergeYAMLPatch(components, patchYAML); err != nil {
			return nil, err
		}
		spec["components"] = components
		insvc["spec"] = spec
		return docs, nil
	}
}

func overlayRouterNoTemplate(docs []map[string]interface{}) ([]map[string]interface{}, error) {
	insvc, err := insvcDoc(docs)
	if err != nil {
		return nil, err
	}
	spec := mustMap(insvc["spec"])
	igr := mustMap(spec["intelligentGatewayRouting"])
	router := mustMap(igr["router"])
	delete(router, "template")
	igr["router"] = router
	spec["intelligentGatewayRouting"] = igr
	insvc["spec"] = spec
	return docs, nil
}

func overlayRouterNoEnabled(docs []map[string]interface{}) ([]map[string]interface{}, error) {
	insvc, err := insvcDoc(docs)
	if err != nil {
		return nil, err
	}
	spec := mustMap(insvc["spec"])
	igr := mustMap(spec["intelligentGatewayRouting"])
	router := mustMap(igr["router"])
	delete(router, "enabled")
	igr["router"] = router
	spec["intelligentGatewayRouting"] = igr
	insvc["spec"] = spec
	return docs, nil
}

func overlayRouterMutateMainContainer(docs []map[string]interface{}, mutate func(map[string]interface{})) ([]map[string]interface{}, error) {
	insvc, err := insvcDoc(docs)
	if err != nil {
		return nil, err
	}
	spec := mustMap(insvc["spec"])
	igr := mustMap(spec["intelligentGatewayRouting"])
	router := mustMap(igr["router"])
	tpl := mustMap(router["template"])
	podSpec := mustMap(tpl["spec"])
	containers, ok := podSpec["containers"].([]interface{})
	if !ok {
		return nil, fmt.Errorf("router template containers missing")
	}
	found := false
	for i, c := range containers {
		cm, ok := c.(map[string]interface{})
		if !ok {
			continue
		}
		if name, _ := cm["name"].(string); name != "main" {
			continue
		}
		mutate(cm)
		containers[i] = cm
		found = true
		break
	}
	if !found {
		return nil, fmt.Errorf("router template main container missing")
	}
	podSpec["containers"] = containers
	return docs, nil
}

func overlayRouterMainRenamedToEPP(docs []map[string]interface{}) ([]map[string]interface{}, error) {
	return overlayRouterMutateMainContainer(docs, func(main map[string]interface{}) {
		main["name"] = "epp"
	})
}

func overlayRouterMainNoGRPCPort(docs []map[string]interface{}) ([]map[string]interface{}, error) {
	return overlayRouterMutateMainContainer(docs, func(main map[string]interface{}) {
		ports, _ := main["ports"].([]interface{})
		out := make([]interface{}, 0, len(ports))
		for _, p := range ports {
			pm, ok := p.(map[string]interface{})
			if !ok {
				continue
			}
			if name, _ := pm["name"].(string); name == "grpc" {
				continue
			}
			out = append(out, pm)
		}
		main["ports"] = out
	})
}

func overlayEngineIndivisibleDataParallel(docs []map[string]interface{}) ([]map[string]interface{}, error) {
	cfg, err := insvcConfigDoc(docs)
	if err != nil {
		return nil, err
	}
	spec := mustMap(cfg["spec"])
	engine := mustMap(spec["engine"])
	engine["dataParallelSize"] = 3
	engine["dataParallelSizeLocal"] = 2
	spec["engine"] = engine
	cfg["spec"] = spec
	return docs, nil
}

func overlayEngineWorkerWhenGroupSizeOne(docs []map[string]interface{}) ([]map[string]interface{}, error) {
	cfg, err := insvcConfigDoc(docs)
	if err != nil {
		return nil, err
	}
	spec := mustMap(cfg["spec"])
	engine := mustMap(spec["engine"])
	engine["worker"] = map[string]interface{}{
		"spec": map[string]interface{}{
			"containers": []interface{}{
				map[string]interface{}{
					"name":  "worker",
					"image": "hub.oepkgs.net/openfuyao/ascend/vllm-ascend:v0.18.0",
				},
			},
		},
	}
	spec["engine"] = engine
	cfg["spec"] = spec
	return docs, nil
}

func overlayRemoveLLMISvcReplicas(docs []map[string]interface{}) ([]map[string]interface{}, error) {
	llm, err := llmisvcDoc(docs)
	if err != nil {
		return nil, err
	}
	deleteMapPath(llm, "spec", "replicas")
	return docs, nil
}

func overlayRemoveLLMISvcPDScaleReplicas(docs []map[string]interface{}) ([]map[string]interface{}, error) {
	cfg, err := llmisvcConfigDoc(docs)
	if err != nil {
		return nil, err
	}
	deleteMapPath(cfg, "spec", "prefill", "replicas")

	llm, err := llmisvcDoc(docs)
	if err != nil {
		return nil, err
	}
	spec := mustMap(llm["spec"])
	delete(spec, "replicas")
	delete(spec, "prefill")
	llm["spec"] = spec
	return docs, nil
}

func overlayInsvcAggregateLWS(docs []map[string]interface{}) ([]map[string]interface{}, error) {
	cfg, err := insvcConfigDoc(docs)
	if err != nil {
		return nil, err
	}
	spec := mustMap(cfg["spec"])
	engine := mustMap(spec["engine"])
	patchInsvcEngineSpecForLWS(engine, lwsNameInsvcAggregate, lwsNameInsvcAggregate)
	spec["engine"] = engine
	cfg["spec"] = spec
	return docs, nil
}

func overlayInsvcPDLWS(docs []map[string]interface{}) ([]map[string]interface{}, error) {
	cfg, err := insvcConfigDoc(docs)
	if err != nil {
		return nil, err
	}
	spec := mustMap(cfg["spec"])
	engine := mustMap(spec["engine"])
	patchInsvcEngineSpecForLWS(engine, lwsNameInsvcPDDecode, lwsNameInsvcPDPrefill)
	spec["engine"] = engine
	cfg["spec"] = spec
	return docs, nil
}

func overlayLLMISvcAggregateLWS(docs []map[string]interface{}) ([]map[string]interface{}, error) {
	cfg, err := llmisvcConfigDoc(docs)
	if err != nil {
		return nil, err
	}
	cfgSpec := mustMap(cfg["spec"])
	patchLLMISvcConfigSpecForLWS(cfgSpec, lwsNameLLMISvcDecode, lwsNameLLMISvcDecode)
	cfg["spec"] = cfgSpec

	llm, err := llmisvcDoc(docs)
	if err != nil {
		return nil, err
	}
	spec := mustMap(llm["spec"])
	for k, v := range llmisvcLWSParallelismFields() {
		spec[k] = v
	}
	llm["spec"] = spec
	return docs, nil
}

func overlayLLMISvcPDLWS(docs []map[string]interface{}) ([]map[string]interface{}, error) {
	cfg, err := llmisvcConfigDoc(docs)
	if err != nil {
		return nil, err
	}
	cfgSpec := mustMap(cfg["spec"])
	patchLLMISvcConfigSpecForLWS(cfgSpec, lwsNameLLMISvcDecode, lwsNameLLMISvcPrefill)
	cfg["spec"] = cfgSpec

	llm, err := llmisvcDoc(docs)
	if err != nil {
		return nil, err
	}
	spec := mustMap(llm["spec"])
	for k, v := range llmisvcLWSParallelismFields() {
		spec[k] = v
	}
	prefill := mustMap(spec["prefill"])
	prefill["parallelism"] = map[string]interface{}{"data": lwsDataParallelSize, "dataLocal": lwsDataParallelSizeLocal}
	prefill["worker"] = map[string]interface{}{}
	spec["prefill"] = prefill
	llm["spec"] = spec
	return docs, nil
}

func overlayPrependSnippetDocKinds(snippetParts []string, kinds ...string) func([]map[string]interface{}) ([]map[string]interface{}, error) {
	return func(docs []map[string]interface{}) ([]map[string]interface{}, error) {
		snippetDocs, err := loadSnippetDocs(snippetParts...)
		if err != nil {
			return nil, err
		}
		allowed := map[string]struct{}{}
		for _, k := range kinds {
			allowed[k] = struct{}{}
		}
		var prefix []map[string]interface{}
		for _, doc := range snippetDocs {
			if _, ok := allowed[docKind(doc)]; ok {
				cloned, err := cloneDocs([]map[string]interface{}{doc})
				if err != nil {
					return nil, err
				}
				prefix = append(prefix, cloned[0])
			}
		}
		if len(prefix) == 0 {
			return nil, fmt.Errorf("no snippet docs with kinds %v in %v", kinds, snippetParts)
		}
		return prependDocs(docs, prefix), nil
	}
}

func overlayReplaceRouterPoolNameArg(poolName string) func([]map[string]interface{}) ([]map[string]interface{}, error) {
	return func(docs []map[string]interface{}) ([]map[string]interface{}, error) {
		insvc, err := insvcDoc(docs)
		if err != nil {
			return nil, err
		}
		spec := mustMap(insvc["spec"])
		igr := mustMap(spec["intelligentGatewayRouting"])
		router := mustMap(igr["router"])
		tpl := mustMap(router["template"])
		podSpec := mustMap(tpl["spec"])
		containers, ok := podSpec["containers"].([]interface{})
		if !ok || len(containers) == 0 {
			return nil, fmt.Errorf("router template containers missing")
		}
		mainIdx := -1
		for i, c := range containers {
			cm := mustMap(c.(map[string]interface{}))
			if name, _ := cm["name"].(string); name == "main" {
				mainIdx = i
				break
			}
		}
		if mainIdx < 0 {
			return nil, fmt.Errorf("router main container not found")
		}
		main := mustMap(containers[mainIdx].(map[string]interface{}))
		args, ok := main["args"].([]interface{})
		if !ok {
			return nil, fmt.Errorf("router main container args missing")
		}
		patched := false
		for i := 0; i < len(args)-1; i++ {
			if s, ok := args[i].(string); ok && s == "--pool-name" {
				args[i+1] = poolName
				patched = true
				break
			}
		}
		if !patched {
			return nil, fmt.Errorf("router main container --pool-name arg missing")
		}
		main["args"] = args
		containers[mainIdx] = main
		podSpec["containers"] = containers
		tpl["spec"] = podSpec
		router["template"] = tpl
		igr["router"] = router
		spec["intelligentGatewayRouting"] = igr
		insvc["spec"] = spec
		return docs, nil
	}
}

// Gateway / validating IGR patch snippets (merged into InferNexService.spec.intelligentGatewayRouting).
const (
	igrPatchGatewayRef = `
gateway:
  ref:
    name: __EXTERNAL_GW__
`
	igrPatchGatewaySpec = `
gateway:
  spec:
    gatewayClassName: envoy
    listeners:
      - name: custom-http
        protocol: HTTP
        port: 8080
        allowedRoutes:
          namespaces:
            from: Same
`
	igrPatchInferencePoolSpec = `
inferencePool:
  spec:
    selector:
      matchLabels:
        openfuyao.com/pdGroupID: __NAME__
        infernex.io/owner: __NAME__
    targetPorts:
      - number: 8000
    endpointPickerRef:
      name: __NAME__-hermes-router
      port:
        number: 9002
`
	igrPatchInferencePoolRef = `
inferencePool:
  ref:
    name: __EXTERNAL_POOL__
`
	igrPatchHTTPRouteRef = `
httpRoute:
  ref:
    name: __EXTERNAL_ROUTE__
`
	igrPatchHTTPRouteSpec = `
httpRoute:
  spec:
    parentRefs: []
    rules:
      - matches:
          - path:
              type: PathPrefix
              value: /__E2E_NS__/__NAME__/v1/completions
        backendRefs:
          - group: inference.networking.k8s.io
            kind: InferencePool
            name: __NAME__-inference-pool
            namespace: __E2E_NS__
            port: 8000
            weight: 1
        filters:
          - type: URLRewrite
            urlRewrite:
              path:
                type: ReplacePrefixMatch
                replacePrefixMatch: /v1/completions
        timeouts:
          request: 0s
          backendRequest: 0s
      - matches:
          - path:
              type: PathPrefix
              value: /__E2E_NS__/__NAME__/v1/chat/completions
        backendRefs:
          - group: inference.networking.k8s.io
            kind: InferencePool
            name: __NAME__-inference-pool
            namespace: __E2E_NS__
            port: 8000
            weight: 1
        filters:
          - type: URLRewrite
            urlRewrite:
              path:
                type: ReplacePrefixMatch
                replacePrefixMatch: /v1/chat/completions
        timeouts:
          request: 0s
          backendRequest: 0s
      - matches:
          - path:
              type: PathPrefix
              value: /__E2E_NS__/__NAME__
        backendRefs:
          - kind: Service
            name: __NAME__-workload-svc
            port: 8000
            weight: 1
        filters:
          - type: URLRewrite
            urlRewrite:
              path:
                type: ReplacePrefixMatch
                replacePrefixMatch: /
        timeouts:
          request: 0s
          backendRequest: 0s
`
	igrPatchHTTPRouteRefSpecConflict = `
httpRoute:
  ref:
    name: external-route
  spec:
    parentRefs: []
    rules:
      - matches:
          - path:
              type: PathPrefix
              value: /test
        backendRefs:
          - name: __NAME__-inference-pool
            kind: InferencePool
            group: inference.networking.k8s.io
            port: 8000
`
	igrPatchGatewayRefSpecConflict = `
gateway:
  ref:
    name: external-gw
  spec:
    gatewayClassName: envoy
    listeners:
      - name: fit19-http
        protocol: HTTP
        port: 80
        allowedRoutes:
          namespaces:
            from: Same
`
	igrPatchPoolRefSpecConflict = `
inferencePool:
  ref:
    name: external-pool
  spec:
    selector:
      matchLabels:
        infernex.io/owner: __NAME__
    targetPorts:
      - number: 8000
    endpointPickerRef:
      name: __NAME__-hermes-router
      port:
        number: 9002
`
)

const (
	compPatchCacheIndexerNoEnabled = `
cacheIndexer:
  replicas: 1
`
	compPatchMooncakeNoEnabled = `
mooncake:
  master:
    replicas: 1
`
	compPatchTidalNoEnabled = `
pdOrchestrator:
  tidal: {}
`
	compPatchEagleEyeNoEnabled = `
eagleEye:
  hardwareMonitor: {}
`
	compPatchNPENoEnabled = `
eagleEye:
  networkPerformanceExporter: {}
`
)
