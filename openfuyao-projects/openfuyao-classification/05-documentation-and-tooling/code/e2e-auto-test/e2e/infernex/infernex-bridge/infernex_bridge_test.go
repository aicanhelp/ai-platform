package infernexbridge

import (
	"os"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	bridgeutils "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/infernex-bridge/utils"
)

// 特性级集成：Ordered 保证 FIT_1→2→3 先完成就绪检测；Bridge 安装由 BeforeSuite 负责（SKIP_BRIDGE_INSTALL 可跳过）。
var _ = Describe("InferNex-Bridge 特性级集成测试", Ordered, ContinueOnFailure, Label("infernex-bridge", "npu"), func() {

	AfterEach(func() {
		afterEachWorkloadCleanupFallback()
	})

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_1 | 部署 - InferNex-Bridge控制器部署就绪 | P0 | 可自动化
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_1-部署 - InferNex-Bridge控制器部署就绪",
		Label("infernex-bridge-deploy", "P0"),
		func() {
			fitID := "InferNexBridge_FIT_1"
			requireCluster(fitID)
			bridgeutils.WaitBridgeControllerReady(suiteCtx, k8sClient.Clientset, fitID, 3*time.Minute)
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_2 | 部署 - InferNex-Bridge Webhook与证书就绪 | P0 | 可自动化
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_2-部署 - InferNex-Bridge Webhook与证书就绪",
		Label("infernex-bridge-deploy", "P0"),
		func() {
			fitID := "InferNexBridge_FIT_2"
			requireCluster(fitID)
			bridgeutils.AssertWebhookReady(fitID)
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_3 | 部署 - 默认InferNexServiceConfig模板已下发 | P0 | 可自动化
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_3-部署 - 默认InferNexServiceConfig模板已下发",
		Label("infernex-bridge-deploy", "P0"),
		func() {
			fitID := "InferNexBridge_FIT_3"
			requireCluster(fitID)
			bridgeutils.AssertDefaultTemplatesExist(suiteCtx, k8sClient.DynamicClient, fitID)
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_4 | LLMInferenceService 聚合模式 | P0
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_4-LLMInferenceService部署方式 - LLMInferenceService 聚合模式部署与端到端联通",
		Label("infernex-bridge-llmisvc", "P0"),
		func() {
			fitID := "InferNexBridge_FIT_4"
			requireLLMISVC(fitID)
			name := "fit04-llm"
			deploySuffixes := bridgeutils.AggregateLLMISvcDeploySuffixes()
			h := beginLLMISvcWorkloadCleanup(fitID, name, deploySuffixes)
			path := applyLLMISvcFixtureNoDefer(fitID, "fit04_llmisvc_aggregate_no_storage_initializer.yaml", name)
			h.SetManifest(path, true)

			waitAggregateLLMISvcFullyReady(fitID, name)

			assertPostWorkloadConnectivity(fitID, name, workloadConnectivityAggregateLLMISVC)
			bridgeutils.LogStep(fitID, "FIT_4 aggregate LLMISVC ready; teardown via DeferCleanup/AfterEach/SIGINT")
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_5 | LLMInferenceService PD模式 | P0
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_5-LLMInferenceService部署方式 - LLMInferenceService PD模式部署与端到端联通",
		Label("infernex-bridge-llmisvc", "P0"),
		func() {
			fitID := "InferNexBridge_FIT_5"
			requireLLMISVC(fitID)
			name := "fit05-pd-llm"
			deploySuffixes := bridgeutils.PDLLMISvcDeploySuffixes()
			h := beginLLMISvcWorkloadCleanup(fitID, name, deploySuffixes)
			if manifest := os.Getenv("FIT05_PD_MANIFEST"); manifest != "" {
				stdout, stderr, err := bridgeutils.ApplyManifest(manifest)
				bridgeutils.LogStep(fitID, "apply FIT05_PD_MANIFEST: stdout=%s stderr=%s err=%v", stdout, stderr, err)
				Expect(err).NotTo(HaveOccurred())
				h.SetManifest(manifest, false)
			} else {
				path := applyLLMISvcFixtureNoDefer(fitID, "fit05_llmisvc_pd_no_storage_initializer.yaml", name)
				h.SetManifest(path, true)
			}

			waitPDLLMISvcFullyReady(fitID, name)

			bridgeutils.AssertProxyServerHelmAlignedArgs(suiteCtx, k8sClient.Clientset, fitID, bridgeutils.E2ENS(), name, bridgeutils.ProxyBackendKServe)
			assertPostWorkloadConnectivity(fitID, name, workloadConnectivityPDLLMISVC)
			bridgeutils.LogStep(fitID, "FIT_5 PD LLMISVC ready; teardown via DeferCleanup/AfterEach/SIGINT")
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_6 | InferNexService聚合模式部署 | P0
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_6-InferNexService部署方式 - InferNexService聚合模式部署与端到端联通",
		Label("infernex-bridge-deploy", "P0"),
		func() {
			fitID := "InferNexBridge_FIT_6"
			requireBridgeInstalled(fitID)
			name := "fit06-aggregate"
			engineCfg := name + "-vllm-aggregate-engine-config"
			deploySuffixes := bridgeutils.AggregateInsvcDeploySuffixes()
			h := beginInsvcWorkloadCleanup(fitID, name, engineCfg, deploySuffixes)
			path := applyDeployFixtureNoDefer(fitID, "fit06_insvc_aggregate_no_replicas.yaml", name)
			h.SetManifest(path, true)

			waitAggregateInsvcFullyReady(fitID, name)

			assertPostWorkloadConnectivity(fitID, name, workloadConnectivityAggregateInsvc)
			bridgeutils.LogStep(fitID, "FIT_6 aggregate InferNexService ready; teardown via DeferCleanup/AfterEach/SIGINT")
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_7 | InferNexService PD模式部署 | P0
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_7-InferNexService部署方式 - InferNexServicePD模式部署与端到端联通",
		Label("infernex-bridge-deploy", "P0"),
		func() {
			fitID := "InferNexBridge_FIT_7"
			requireBridgeInstalled(fitID)
			name := "fit07-pd"
			engineCfg := name + "-vllm-pd-engine-config"
			deploySuffixes := bridgeutils.PDInsvcDeploySuffixes()
			h := beginInsvcWorkloadCleanup(fitID, name, engineCfg, deploySuffixes)
			path := applyDeployFixtureNoDefer(fitID, "fit07_insvc_pd_no_replicas.yaml", name)
			h.SetManifest(path, true)

			waitPDInsvcFullyReady(fitID, name)

			bridgeutils.AssertProxyServerHelmAlignedArgs(suiteCtx, k8sClient.Clientset, fitID, bridgeutils.E2ENS(), name, bridgeutils.ProxyBackendInferNexService)
			assertPostWorkloadConnectivity(fitID, name, workloadConnectivityPDInsvc)
			bridgeutils.LogStep(fitID, "FIT_7 PD InferNexService ready; teardown via DeferCleanup/AfterEach/SIGINT")
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_8 | Mutating 无 runtime 标签 | P0
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_8-Mutating - Mutating无runtime标签跳过Config补丁",
		Label("infernex-bridge-mutating", "P0"),
		func() {
			fitID := "InferNexBridge_FIT_8"
			requireLLMISVC(fitID)
			kserveNS := requireKServeMutateTargetConfigs(fitID)
			names := bridgeutils.KServeConfigNames()
			beforeRV := bridgeutils.SnapshotLLMConfigResourceVersions(suiteCtx, k8sClient.DynamicClient, kserveNS, names)
			beforeAnn := map[string]string{}
			for _, cfgName := range names {
				cfg, err := bridgeutils.GetLLMInferenceServiceConfig(suiteCtx, k8sClient.DynamicClient, kserveNS, cfgName)
				Expect(err).NotTo(HaveOccurred(), "config %s", cfgName)
				beforeAnn[cfgName] = bridgeutils.ConfigMutatedAnnotation(cfg)
			}
			name := "fit08-llm"
			applyLLMISvcFixture(fitID, "fit08_no_runtime.yaml", name)
			llm, err := bridgeutils.GetLLMInferenceService(suiteCtx, k8sClient.DynamicClient, bridgeutils.E2ENS(), name)
			Expect(err).NotTo(HaveOccurred())
			Expect(bridgeutils.LLMISVCHasInfernexRuntimeLabel(llm)).To(BeFalse(),
				"fixture must not set infernex.io/runtime=true")
			for _, cfgName := range names {
				cfg, err := bridgeutils.GetLLMInferenceServiceConfig(suiteCtx, k8sClient.DynamicClient, kserveNS, cfgName)
				Expect(err).NotTo(HaveOccurred())
				Expect(bridgeutils.ConfigMutatedAnnotation(cfg)).To(Equal(beforeAnn[cfgName]),
					"mutating webhook must not change %s annotation without runtime label", cfgName)
				Expect(cfg.GetResourceVersion()).To(Equal(beforeRV[cfgName]),
					"mutating webhook must not update %s (resourceVersion) without runtime label", cfgName)
			}
			bridgeutils.LogStep(fitID, "all %d kserve configs unchanged (already mutated=%v ok)",
				len(names), beforeAnn[names[0]] == bridgeutils.LLMConfigMutatedValue)
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_9 | Mutating 补丁 kserve 六模板 | P0
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_9-Mutating - Mutating补丁kserve命名空间全部LLMInferenceServiceConfig模板",
		Label("infernex-bridge-mutating", "P0"),
		func() {
			fitID := "InferNexBridge_FIT_9"
			requireLLMISVC(fitID)
			kserveNS := requireKServeMutateTargetConfigs(fitID)
			names := bridgeutils.KServeConfigNames()
			beforeRV := bridgeutils.SnapshotLLMConfigResourceVersions(suiteCtx, k8sClient.DynamicClient, kserveNS, names)
			beforeAnn := map[string]string{}
			for _, cfgName := range names {
				cfg, err := bridgeutils.GetLLMInferenceServiceConfig(suiteCtx, k8sClient.DynamicClient, kserveNS, cfgName)
				Expect(err).NotTo(HaveOccurred())
				beforeAnn[cfgName] = bridgeutils.ConfigMutatedAnnotation(cfg)
			}
			name := "fit09-llm"
			applyLLMISvcFixture(fitID, "fit09_runtime_mutate.yaml", name)
			llm, err := bridgeutils.GetLLMInferenceService(suiteCtx, k8sClient.DynamicClient, bridgeutils.E2ENS(), name)
			Expect(err).NotTo(HaveOccurred())
			Expect(bridgeutils.LLMISVCHasInfernexRuntimeLabel(llm)).To(BeTrue())
			bridgeutils.WaitKServeConfigsInferNexMutated(suiteCtx, k8sClient.DynamicClient, fitID, kserveNS, names, beforeRV, beforeAnn, 2*time.Minute)
			bridgeutils.LogStep(fitID, "all %d kserve configs have infernex mutation markers", len(names))
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_11 | Validating - Validating拒绝InferNexService方式缺merged.engine | P0
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_11-Validating - Validating拒绝InferNexService方式缺merged.engine",
		Label("infernex-bridge-validating", "P0"),
		func() {
			expectValidatingAdmissionDeniedNoConfig("InferNexBridge_FIT_11", []string{"validating", "fit11_no_engine.yaml"}, ErrMergedEngine, "fit11-no-engine")
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_12 | Validating - Validating拒绝cacheIndexer缺enabled | P0
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_12-Validating - Validating拒绝cacheIndexer缺enabled",
		Label("infernex-bridge-validating", "P0"),
		func() {
			expectValidatingAdmissionDenied("InferNexBridge_FIT_12", []string{"validating", "fit12_cache_indexer_no_enabled.yaml"}, ErrCacheIndexerEnabled, "fit12-cache-indexer", true)
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_13 | Validating - Validating拒绝mooncake缺enabled | P1
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_13-Validating - Validating拒绝mooncake缺enabled",
		Label("infernex-bridge-validating"),
		func() {
			expectValidatingAdmissionDenied("InferNexBridge_FIT_13", []string{"validating", "fit13_mooncake_no_enabled.yaml"}, ErrMooncakeEnabled, "fit13-mooncake", true)
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_14 | Validating - Validating拒绝pdOrchestrator子项缺enabled | P1
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_14-Validating - Validating拒绝pdOrchestrator子项缺enabled",
		Label("infernex-bridge-validating"),
		func() {
			expectValidatingAdmissionDenied("InferNexBridge_FIT_14", []string{"validating", "fit14_tidal_no_enabled.yaml"}, ErrTidalEnabled, "fit14-tidal", true)
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_15 | Validating - Validating拒绝eagleEye子项缺enabled | P1
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_15-Validating - Validating拒绝eagleEye子项缺enabled",
		Label("infernex-bridge-validating"),
		func() {
			expectValidatingAdmissionDenied("InferNexBridge_FIT_15", []string{"validating", "fit15_eagleeye_no_enabled.yaml"}, ErrEagleMonitorEnabled, "fit15-eagleeye", true)
			expectValidatingAdmissionDenied("InferNexBridge_FIT_15", []string{"validating", "fit15_npe_no_enabled.yaml"}, ErrNPENetworkPerfEnabled, "fit15-npe", true)
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_16 | Validating - Validating拒绝router块缺enabled | P1
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_16-Validating - Validating拒绝router块缺enabled",
		Label("infernex-bridge-validating"),
		func() {
			expectValidatingAdmissionDenied("InferNexBridge_FIT_16", []string{"validating", "fit16_router_no_enabled.yaml"}, ErrRouterEnabled, "fit16-router", true)
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_17 | Validating - Validating拒绝router.enabled为true但router.template为空 | P1
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_17-Validating - Validating拒绝router.enabled为true但router.template为空",
		Label("infernex-bridge-validating"),
		func() {
			expectValidatingAdmissionDenied("InferNexBridge_FIT_17", []string{"validating", "fit17_router_no_template.yaml"}, ErrRouterTemplate, "fit17-router-template", true)
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_18 | Validating - Validating拒绝httpRoute的ref与spec并存 | P0
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_18-Validating - Validating拒绝httpRoute的ref与spec并存",
		Label("infernex-bridge-validating", "P0"),
		func() {
			expectValidatingAdmissionDenied("InferNexBridge_FIT_18", []string{"validating", "fit18_httproute_ref_spec.yaml"}, ErrHTTPRouteExclusive, "fit18-httproute", true)
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_19 | Validating - Validating拒绝gateway的ref与spec并存 | P0
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_19-Validating - Validating拒绝gateway的ref与spec并存",
		Label("infernex-bridge-validating", "P0"),
		func() {
			expectValidatingAdmissionDenied("InferNexBridge_FIT_19", []string{"validating", "fit19_gateway_ref_spec.yaml"}, ErrGatewayExclusive, "fit19-gateway", true)
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_20 | Validating - Validating拒绝inferencePool的ref与spec并存 | P0
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_20-Validating - Validating拒绝inferencePool的ref与spec并存",
		Label("infernex-bridge-validating", "P0"),
		func() {
			expectValidatingAdmissionDenied("InferNexBridge_FIT_20", []string{"validating", "fit20_pool_ref_spec.yaml"}, ErrPoolExclusive, "fit20-pool", true)
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_21 | Validating - Validating拒绝引擎模板空image | 暂不自动化
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_21-Validating - Validating拒绝引擎模板空image",
		Label("infernex-bridge-validating"),
		func() {
			expectValidatingAdmissionDenied("InferNexBridge_FIT_21", []string{"validating", "fit21_engine_empty_image.yaml"}, ErrImageRequired, "fit21-empty-image", true)
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_22 | Validating - Validating允许sourceRef跳过InferNexService校验 | P1 | 可自动化
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_22-Validating - Validating允许sourceRef跳过InferNexService校验",
		Label("infernex-bridge-validating"),
		func() {
			fitID := "InferNexBridge_FIT_22"
			requireBridgeInstalled(fitID)
			if bridgeutils.SkipLLMISVCE2E() {
				Skip("SKIP_LLMISVC_E2E=true")
			}
			if !bridgeutils.CRDExists(fitID, "serving.kserve.io", "llminferenceservices") {
				Skip("LLMInferenceService CRD not found (KServe 0.17?)")
			}
			ns := bridgeutils.E2ENS()
			bridgeutils.EnsureNamespace(suiteCtx, k8sClient.Clientset, ns)
			const insvcName = "fit22-source-ref"
			const llmName = "fit22-llmisvc"
			clean := beginValidatingFixtureClean(fitID, insvcName, false, llmName)

			llmPath, err := bridgeutils.WriteRenderedFixtureOpts([]string{"validating", "fit22_llmisvc_stub.yaml"}, bridgeutils.RenderOptions{Namespace: ns, Name: llmName})
			Expect(err).NotTo(HaveOccurred())
			clean.TrackManifest(llmPath)
			llmOut, llmErr, err := bridgeutils.ApplyManifest(llmPath)
			bridgeutils.LogStep(fitID, "apply llmisvc stub: stdout=%s stderr=%s err=%v", llmOut, llmErr, err)
			Expect(err).NotTo(HaveOccurred(), "fit22 LLMISVC stub required so controller does not delete sourceRef insvc")

			insvcPath, err := bridgeutils.WriteRenderedFixtureOpts([]string{"validating", "fit22_source_ref.yaml"}, bridgeutils.RenderOptions{Namespace: ns, Name: insvcName})
			Expect(err).NotTo(HaveOccurred())
			clean.TrackManifest(insvcPath)
			stdout, stderr, err := bridgeutils.ApplyManifest(insvcPath)
			bridgeutils.LogStep(fitID, "apply insvc: stdout=%s stderr=%s err=%v", stdout, stderr, err)
			Expect(err).NotTo(HaveOccurred(), "sourceRef-only InferNexService should be admitted")

			bridgeutils.WaitInferNexServiceExists(suiteCtx, k8sClient.DynamicClient, fitID, ns, insvcName, 30*time.Second)

			pOut, pErr, err := bridgeutils.PatchInferNexService(ns, insvcName, `{"metadata":{"annotations":{"fit-touch":"1"}}}`)
			bridgeutils.LogStep(fitID, "patch insvc: stdout=%s stderr=%s err=%v", pOut, pErr, err)
			Expect(err).NotTo(HaveOccurred(), "metadata patch should succeed while sourceRef skips validation")
			bridgeutils.LogStep(fitID, "patch succeeded for sourceRef InferNexService")
		})

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_60 | Validating - router.template 缺少 Hermes EPP main 容器 | P1
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_60-Validating - Validating拒绝router.template缺少Hermes EPP main容器",
		Label("infernex-bridge-validating"),
		func() {
			expectValidatingAdmissionDenied("InferNexBridge_FIT_60", []string{"validating", "fit60_router_no_main_epp.yaml"}, ErrRouterMainContainer, "fit60-router-no-main", true)
		})

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_61 | Validating - router main 缺少 grpc 命名端口 | P1
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_61-Validating - Validating拒绝router main缺少grpc命名端口",
		Label("infernex-bridge-validating"),
		func() {
			expectValidatingAdmissionDenied("InferNexBridge_FIT_61", []string{"validating", "fit61_router_main_no_grpc.yaml"}, ErrRouterGRPCPort, "fit61-router-no-grpc", true)
		})

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_62 | Validating - dataParallelSize 不能被 dataParallelSizeLocal 整除 | P1
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_62-Validating - Validating拒绝dataParallelSize不能被dataParallelSizeLocal整除",
		Label("infernex-bridge-validating"),
		func() {
			expectValidatingAdmissionDenied("InferNexBridge_FIT_62", []string{"validating", "fit62_engine_dp_indivisible.yaml"}, ErrDataParallelDivisible, "fit62-dp-indivisible", true)
		})

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_63 | Validating - groupSize为1时不得设置engine.worker | P1
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_63-Validating - Validating拒绝groupSize为1时设置engine.worker",
		Label("infernex-bridge-validating"),
		func() {
			expectValidatingAdmissionDenied("InferNexBridge_FIT_63", []string{"validating", "fit63_engine_worker_group_size_one.yaml"}, ErrWorkerWhenGroupSizeOne, "fit63-worker-group1", true)
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_24 | pdOrchestrator三子项全开 | P1
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_24-PD-Orchestrator - InferNexService方式pdOrchestrator三子项全开",
		Label("infernex-bridge-pd-orchestrator"),
		func() {
			fitID := "InferNexBridge_FIT_24"
			requireBridgeInstalled(fitID)
			name := "fit24-pdo-on"
			beginPDOrchestratorInsvcTest(fitID, name, "fit24_pd_orchestrator_on.yaml", true)

			workloadTimeout := 15 * time.Minute
			if bridgeutils.SkipNPUWorkloadE2E() {
				workloadTimeout = 5 * time.Minute
			}
			bridgeutils.WaitInferNexServiceMode(suiteCtx, k8sClient.DynamicClient, fitID, bridgeutils.E2ENS(), name, "aggregate", 5*time.Minute)
			bridgeutils.WaitPDOrchestratorDeploysReady(suiteCtx, k8sClient.Clientset, fitID, bridgeutils.E2ENS(), name, workloadTimeout)
			bridgeutils.WaitPDOrchestratorComponentReady(suiteCtx, k8sClient.DynamicClient, fitID, bridgeutils.E2ENS(), name, workloadTimeout)
			bridgeutils.WaitAggregateInsvcWorkloadsForPDOrchestratorTest(suiteCtx, k8sClient.Clientset, fitID, bridgeutils.E2ENS(), name, true, workloadTimeout)
			bridgeutils.WaitAggregateInsvcComponentsReadyForPDOrchestratorTest(suiteCtx, k8sClient.DynamicClient, fitID, bridgeutils.E2ENS(), name, true, workloadTimeout)
			waitStackReadyWithOwnerFallback(fitID, name, bridgeutils.StackInsvc, workloadTimeout)
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_25 | pdOrchestrator三子项全关 | P1
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_25-PD-Orchestrator - InferNexService方式pdOrchestrator三子项全关",
		Label("infernex-bridge-pd-orchestrator"),
		func() {
			fitID := "InferNexBridge_FIT_25"
			requireBridgeInstalled(fitID)
			name := "fit25-pdo-off"
			beginPDOrchestratorInsvcTest(fitID, name, "fit25_pd_orchestrator_off.yaml", false)

			workloadTimeout := 15 * time.Minute
			if bridgeutils.SkipNPUWorkloadE2E() {
				workloadTimeout = 5 * time.Minute
			}
			bridgeutils.WaitInferNexServiceMode(suiteCtx, k8sClient.DynamicClient, fitID, bridgeutils.E2ENS(), name, "aggregate", 5*time.Minute)
			bridgeutils.WaitNoPDOrchestratorWorkloads(suiteCtx, k8sClient.Clientset, fitID, bridgeutils.E2ENS(), name, 3*time.Minute)
			bridgeutils.WaitPDOrchestratorComponentNotReady(suiteCtx, k8sClient.DynamicClient, fitID, bridgeutils.E2ENS(), name, 3*time.Minute)
			bridgeutils.WaitAggregateInsvcWorkloadsForPDOrchestratorTest(suiteCtx, k8sClient.Clientset, fitID, bridgeutils.E2ENS(), name, false, workloadTimeout)
			bridgeutils.WaitAggregateInsvcComponentsReadyForPDOrchestratorTest(suiteCtx, k8sClient.DynamicClient, fitID, bridgeutils.E2ENS(), name, false, workloadTimeout)
			waitStackReadyWithOwnerFallback(fitID, name, bridgeutils.StackInsvc, workloadTimeout)
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_26 | LLMISVC 聚合扩缩 | P0
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_26-LLMInferenceService部署方式 - LLMInferenceService 聚合模式推理副本扩缩",
		Label("infernex-bridge-scaling", "P0"),
		func() {
			fitID := "InferNexBridge_FIT_26"
			requireLLMISVC(fitID)
			bridgeutils.RequireNPUWorkload(fitID)
			bridgeutils.RequireScalingCRDs(fitID, false)
			name := "fit26-llm"
			beginAggregateLLMISvcScalingTest(fitID, name, "fit26_llmisvc_aggregate_scale.yaml")

			workloadTimeout := 15 * time.Minute
			scaleTimeout := bridgeutils.DefaultBridgeScaleTimeout

			waitStackFullyReady(fitID, name, bridgeutils.StackLLMISvc, bridgeutils.StackAggregate, bridgeutils.StackFullyReadyOpts{
				WorkloadTimeout: workloadTimeout,
				ReadyTimeout:    workloadTimeout,
				SkipComponents:  true,
				SkipLogHealth:   true,
			})

			bridgeutils.RunLLMISvcAggregateTidalScaleRoundTrip(suiteCtx, k8sClient.Clientset, k8sClient.DynamicClient, fitID, bridgeutils.E2ENS(), name, scaleTimeout)
			bridgeutils.LogStep(fitID, "FIT_26 aggregate LLMISVC tidal scale round-trip done")
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_27 | InferNexService 聚合 ElasticScaler | P0
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_27-InferNexService部署方式 - InferNexService聚合模式推理副本扩缩",
		Label("infernex-bridge-scaling", "P0"),
		func() {
			fitID := "InferNexBridge_FIT_27"
			requireBridgeInstalled(fitID)
			bridgeutils.RequireNPUWorkload(fitID)
			bridgeutils.RequireScalingCRDs(fitID, false)
			name := "fit27-scale"
			beginAggregateInsvcScalingTest(fitID, name, "fit27_insvc_aggregate_scale.yaml")

			workloadTimeout := 15 * time.Minute
			scaleTimeout := bridgeutils.DefaultBridgeScaleTimeout

			waitStackWorkloadsOnly(fitID, name, bridgeutils.StackInsvc, bridgeutils.StackAggregate, workloadTimeout)
			bridgeutils.WaitAggregateInsvcComponentsReady(suiteCtx, k8sClient.DynamicClient, fitID, bridgeutils.E2ENS(), name, workloadTimeout)
			waitStackReadyWithOwnerFallback(fitID, name, bridgeutils.StackInsvc, workloadTimeout)

			bridgeutils.RunInsvcAggregateTidalScaleRoundTrip(suiteCtx, k8sClient.Clientset, k8sClient.DynamicClient, fitID, bridgeutils.E2ENS(), name, scaleTimeout)
			bridgeutils.LogStep(fitID, "FIT_27 aggregate InferNexService tidal scale round-trip done")
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_28 | LLMISVC PD 扩缩 | P0
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_28-LLMInferenceService部署方式 - LLMInferenceService PD模式推理副本扩缩",
		Label("infernex-bridge-scaling", "P0"),
		func() {
			fitID := "InferNexBridge_FIT_28"
			requireLLMISVC(fitID)
			bridgeutils.RequireNPUWorkload(fitID)
			bridgeutils.RequireScalingCRDs(fitID, true)
			name := "fit28-pd-llm"
			beginPDLLMISvcScalingTest(fitID, name, "fit28_llmisvc_pd_scale.yaml")

			workloadTimeout := 15 * time.Minute
			scaleTimeout := bridgeutils.DefaultBridgeScaleTimeout

			waitStackFullyReady(fitID, name, bridgeutils.StackLLMISvc, bridgeutils.StackPD, bridgeutils.StackFullyReadyOpts{
				WorkloadTimeout: workloadTimeout,
				ReadyTimeout:    workloadTimeout,
				SkipComponents:  true,
				SkipLogHealth:   true,
			})

			bridgeutils.RunLLMISvcPDTidalScaleRoundTrip(suiteCtx, k8sClient.Clientset, k8sClient.DynamicClient, fitID, bridgeutils.E2ENS(), name, scaleTimeout)
			bridgeutils.CleanupPDRSGCloneDeploymentsFallback(suiteCtx, k8sClient.Clientset, fitID, bridgeutils.E2ENS(), name, 5*time.Minute)
			bridgeutils.LogStep(fitID, "FIT_28 PD LLMISVC scale up/down + RSG clone fallback cleanup done")
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_29 | InferNexService PD ElasticScaler | P0
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_29-InferNexService部署方式 - InferNexServicePD模式推理副本扩缩",
		Label("infernex-bridge-scaling", "P0"),
		func() {
			fitID := "InferNexBridge_FIT_29"
			requireBridgeInstalled(fitID)
			bridgeutils.RequireNPUWorkload(fitID)
			bridgeutils.RequireScalingCRDs(fitID, true)
			name := "fit29-pd-scale"
			beginPDInsvcScalingTest(fitID, name, "fit29_insvc_pd_scale.yaml")

			workloadTimeout := 15 * time.Minute
			scaleTimeout := bridgeutils.DefaultBridgeScaleTimeout

			waitStackWorkloadsOnly(fitID, name, bridgeutils.StackInsvc, bridgeutils.StackPD, workloadTimeout)
			bridgeutils.WaitPDInsvcComponentsReady(suiteCtx, k8sClient.DynamicClient, fitID, bridgeutils.E2ENS(), name, workloadTimeout)
			waitStackReadyWithOwnerFallback(fitID, name, bridgeutils.StackInsvc, workloadTimeout)

			bridgeutils.RunInsvcPDTidalScaleRoundTrip(suiteCtx, k8sClient.Clientset, k8sClient.DynamicClient, fitID, bridgeutils.E2ENS(), name, scaleTimeout)
			bridgeutils.CleanupPDRSGCloneDeploymentsFallback(suiteCtx, k8sClient.Clientset, fitID, bridgeutils.E2ENS(), name, 5*time.Minute)
			bridgeutils.LogStep(fitID, "FIT_29 PD InferNexService scale up/down + RSG clone fallback cleanup done")
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_34 | 智能网关 - router未启用时GatewayRoutingReady为RouterDisabled | P0
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_34-智能网关 - router未启用时InferNexService的GatewayRoutingReady为True且Reason为RouterDisabled",
		Label("infernex-bridge-gateway", "P0"),
		func() {
			fitID := "InferNexBridge_FIT_34"
			requireBridgeInstalled(fitID)
			// FIT 34: router disabled — only InferNex status; no Gateway API objects required.
			name := "fit34-router-off"
			_ = beginPDInsvcRouterOffTest(fitID, name)
			assertFit34RouterDisabledStatus(fitID, name)
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_35 | 全托管Gateway路由三元组 | P0
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_35-智能网关 - 验证InferNexService全托管Gateway路由三元组",
		Label("infernex-bridge-gateway", "P0"),
		func() {
			fitID := "InferNexBridge_FIT_35"
			requireBridgeInstalled(fitID)
			requireGatewayAPI(fitID)
			if r := bridgeutils.SkipGate(false, true, true); r != "" {
				Skip(r)
			}
			name := "fit35-gw"
			_ = beginGatewayPDInsvcTest(fitID, name, fit35GatewayManagedPDFixture)
			waitGatewayManagedPDInsvcFullyReady(fitID, name)
			assertGatewayPDInference(fitID, name)
			bridgeutils.LogStep(fitID, "FIT_35 PD managed gateway triple + gateway inference OK owner=%s", name)
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_36 | gateway.ref引用已有Gateway | P0
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_36-智能网关 - InferNexService方式gateway.ref引用已有Gateway",
		Label("infernex-bridge-gateway", "P0"),
		func() {
			fitID := "InferNexBridge_FIT_36"
			requireBridgeInstalled(fitID)
			requireGatewayAPI(fitID)
			if r := bridgeutils.SkipGate(false, true, true); r != "" {
				Skip(r)
			}
			name := "fit36-gwref"
			extGW := "fit36-ext-gw"
			ns := bridgeutils.E2ENS()
			render := bridgeutils.RenderOptions{Namespace: ns, Name: name, ExternalGW: extGW}
			h := beginGatewayPDInsvcTest(fitID, name, fit36GatewayRefFixture, render)
			applyExternalManifest(fitID, "fit36_external_gateway.yaml", render, h)
			waitGatewayPDInsvcWorkloadsAndReady(fitID, name)
			managedGW, route, _ := bridgeutils.ManagedResourceNames(name)
			Expect(bridgeutils.KubectlExists(fitID, "gateway", ns, managedGW)).To(BeFalse())
			bridgeutils.WaitKubectlResourceExists(fitID, "httproute", ns, route, 5*time.Minute)
			parent := bridgeutils.KubectlJSONPath(fitID, "httproute", ns, route, "{.spec.parentRefs[0].name}")
			Expect(parent).To(Equal(extGW))
			assertGatewayPDInference(fitID, name)
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_37 | gateway.spec 自定义 | P1
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_37-智能网关 - InferNexService方式gateway.spec自定义托管Gateway",
		Label("infernex-bridge-gateway"),
		func() {
			fitID := "InferNexBridge_FIT_37"
			requireBridgeInstalled(fitID)
			requireGatewayAPI(fitID)
			if r := bridgeutils.SkipGate(false, true, true); r != "" {
				Skip(r)
			}
			name := "fit37-gw-spec"
			ns := bridgeutils.E2ENS()
			_ = beginGatewayPDInsvcTest(fitID, name, fit37GatewaySpecFixture)
			waitGatewayPDInsvcWorkloadsAndReady(fitID, name)
			gw, _, _ := bridgeutils.ManagedResourceNames(name)
			bridgeutils.WaitKubectlResourceExists(fitID, "gateway", ns, gw, 5*time.Minute)
			port := bridgeutils.KubectlJSONPath(fitID, "gateway", ns, gw, "{.spec.listeners[0].port}")
			Expect(port).To(Equal("8080"))
			assertGatewayPDInference(fitID, name)
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_38 | inferencePool.ref | P1
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_38-智能网关 - 验证InferNexService引用已有InferencePool",
		Label("infernex-bridge-gateway"),
		func() {
			fitID := "InferNexBridge_FIT_38"
			requireBridgeInstalled(fitID)
			requireGatewayAPI(fitID)
			if r := bridgeutils.SkipGate(false, true, true); r != "" {
				Skip(r)
			}
			name := "fit38-pool-ref"
			extPool := "fit38-ext-pool"
			ns := bridgeutils.E2ENS()
			_ = beginGatewayPDInsvcTest(fitID, name, fit38PoolRefFixture, bridgeutils.RenderOptions{ExternalPool: extPool})
			waitGatewayPDInsvcWorkloadsAndReady(fitID, name)
			_, _, pool := bridgeutils.ManagedResourceNames(name)
			Expect(bridgeutils.KubectlExists(fitID, "inferencepool", ns, pool)).To(BeFalse())
			bridgeutils.WaitKubectlResourceExists(fitID, "inferencepool", ns, extPool, 3*time.Minute)
			assertGatewayPDInference(fitID, name)
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_39 | inferencePool.spec 覆盖 | P1
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_39-智能网关 - InferNexService方式inferencePool.spec覆盖托管池",
		Label("infernex-bridge-gateway"),
		func() {
			fitID := "InferNexBridge_FIT_39"
			requireBridgeInstalled(fitID)
			requireGatewayAPI(fitID)
			if r := bridgeutils.SkipGate(false, true, true); r != "" {
				Skip(r)
			}
			name := "fit39-pool-spec"
			ns := bridgeutils.E2ENS()
			_ = beginGatewayPDInsvcTest(fitID, name, fit39PoolSpecFixture)
			waitGatewayPDInsvcWorkloadsAndReady(fitID, name)
			_, _, pool := bridgeutils.ManagedResourceNames(name)
			bridgeutils.WaitKubectlResourceExists(fitID, "inferencepool", ns, pool, 5*time.Minute)
			tp := bridgeutils.KubectlJSONPath(fitID, "inferencepool", ns, pool, "{.spec.targetPorts[0].number}")
			Expect(tp).To(Equal("8000"))
			assertGatewayPDInference(fitID, name)
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_40 | httpRoute.ref | P1
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_40-智能网关 - InferNexService方式httpRoute.ref仅引用外部HTTPRoute",
		Label("infernex-bridge-gateway"),
		func() {
			fitID := "InferNexBridge_FIT_40"
			requireBridgeInstalled(fitID)
			requireGatewayAPI(fitID)
			if r := bridgeutils.SkipGate(false, true, true); r != "" {
				Skip(r)
			}
			name := "fit40-route-ref"
			extRoute := "fit40-ext-route"
			ns := bridgeutils.E2ENS()
			_ = beginGatewayPDInsvcTest(fitID, name, fit40HTTPRouteRefFixture, bridgeutils.RenderOptions{
				ExternalRoute: extRoute,
			})
			waitGatewayPDInsvcWorkloadsAndReady(fitID, name)
			_, route, _ := bridgeutils.ManagedResourceNames(name)
			Expect(bridgeutils.KubectlExists(fitID, "httproute", ns, route)).To(BeFalse())
			bridgeutils.WaitKubectlResourceExists(fitID, "httproute", ns, extRoute, 3*time.Minute)
			assertGatewayPDInference(fitID, name)
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_41 | httpRoute.spec parentRefs 填充 | P1
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_41-智能网关 - InferNexService方式httpRoute.spec且parentRefs为空由控制器填充",
		Label("infernex-bridge-gateway"),
		func() {
			fitID := "InferNexBridge_FIT_41"
			requireBridgeInstalled(fitID)
			requireGatewayAPI(fitID)
			if r := bridgeutils.SkipGate(false, true, true); r != "" {
				Skip(r)
			}
			name := "fit41-route-spec"
			ns := bridgeutils.E2ENS()
			_ = beginGatewayPDInsvcTest(fitID, name, fit41HTTPRouteSpecFixture)
			waitGatewayPDInsvcWorkloadsAndReady(fitID, name)
			_, route, _ := bridgeutils.ManagedResourceNames(name)
			bridgeutils.WaitKubectlResourceExists(fitID, "httproute", ns, route, 5*time.Minute)
			parent := bridgeutils.KubectlJSONPath(fitID, "httproute", ns, route, "{.spec.parentRefs[0].name}")
			gw, _, _ := bridgeutils.ManagedResourceNames(name)
			Expect(parent).To(Equal(gw))
			assertGatewayPDInference(fitID, name)
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_42 | 混合 gateway.ref | P1
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_42-智能网关 - InferNexService方式混合gateway.ref与托管Pool与Route",
		Label("infernex-bridge-gateway"),
		func() {
			fitID := "InferNexBridge_FIT_42"
			requireBridgeInstalled(fitID)
			requireGatewayAPI(fitID)
			if r := bridgeutils.SkipGate(false, true, true); r != "" {
				Skip(r)
			}
			name := "fit42-mixed"
			extGW := "fit42-ext-gw"
			ns := bridgeutils.E2ENS()
			render := bridgeutils.RenderOptions{Namespace: ns, Name: name, ExternalGW: extGW}
			h := beginGatewayPDInsvcTest(fitID, name, fit42MixedGatewayRefFixture, render)
			applyExternalManifest(fitID, "fit36_external_gateway.yaml", render, h)
			waitGatewayPDInsvcWorkloadsAndReady(fitID, name)
			managedGW, route, pool := bridgeutils.ManagedResourceNames(name)
			bridgeutils.LogPhaseBegin(fitID, "mixed gateway: no managed GW, route+pool exist")
			bridgeutils.WaitKubectlResourceExists(fitID, "httproute", ns, route, 5*time.Minute)
			bridgeutils.WaitKubectlResourceExists(fitID, "inferencepool", ns, pool, 5*time.Minute)
			Expect(bridgeutils.KubectlExists(fitID, "gateway", ns, managedGW)).To(BeFalse())
			parent := bridgeutils.KubectlJSONPath(fitID, "httproute", ns, route, "{.spec.parentRefs[0].name}")
			Expect(parent).To(Equal(extGW))
			bridgeutils.LogPhaseEnd(fitID, "mixed gateway: no managed GW, route+pool exist")
			assertGatewayPDInference(fitID, name)
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_43 | router改为false清理托管网关 | P1
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_43-智能网关 - InferNexService方式router由true改为false清理托管网关对象",
		Label("infernex-bridge-gateway"),
		func() {
			fitID := "InferNexBridge_FIT_43"
			requireBridgeInstalled(fitID)
			requireGatewayAPI(fitID)
			if r := bridgeutils.SkipGate(false, true, false); r != "" {
				Skip(r)
			}
			name := "fit43-gw-off"
			ns := bridgeutils.E2ENS()
			_ = beginGatewayPDInsvcTest(fitID, name, fit35GatewayManagedPDFixture)
			waitGatewayManagedPDInsvcFullyReady(fitID, name)
			_, _, err := bridgeutils.PatchInferNexService(ns, name, `{"spec":{"intelligentGatewayRouting":{"router":{"enabled":false}}}}`)
			Expect(err).NotTo(HaveOccurred())
			bridgeutils.WaitManagedGatewayTripleAbsent(fitID, ns, name, 3*time.Minute)
			bridgeutils.WaitInferNexCondition(suiteCtx, k8sClient.DynamicClient, fitID, ns, name,
				"GatewayRoutingReady", "True", "RouterDisabled", 3*time.Minute)
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_44 | 删除 LLMISVC 联动清理 | P0
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_44-生命周期 - 删除 LLMInferenceService 联动删除InferNexService及归属组件",
		Label("infernex-bridge-lifecycle", "infernex-bridge-llmisvc", "P0"),
		func() {
			fitID := "InferNexBridge_FIT_44"
			requireLLMISVC(fitID)
			if r := bridgeutils.SkipGate(true, true, false); r != "" {
				Skip(r)
			}
			name := "fit44-llm"
			deploySuffixes := bridgeutils.AggregateLLMISvcDeploySuffixes()
			cleanupLLMISvcEntry(fitID, name)
			path := applyLLMISvcFixtureNoDefer(fitID, "fit44_llmisvc_delete_lifecycle.yaml", name)
			DeferCleanup(func() {
				finalizeLifecycleLLMISVC(fitID, name, path, deploySuffixes)
			})

			waitAggregateLLMISvcFullyReady(fitID, name)

			Expect(bridgeutils.DeleteLLMInferenceService(suiteCtx, k8sClient.DynamicClient, bridgeutils.E2ENS(), name)).To(Succeed())
			assertLifecycleAfterDeleteLLMISVC(fitID, name, deploySuffixes)
			bridgeutils.LogStep(fitID, "FIT_44 delete LLMISVC cascade cleanup verified")
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_45 | 删除InferNexService清理归属组件 | P0
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_45-生命周期 - 删除InferNexService清理归属组件与托管路由",
		Label("infernex-bridge-lifecycle", "infernex-bridge-deploy", "P0"),
		func() {
			fitID := "InferNexBridge_FIT_45"
			requireBridgeInstalled(fitID)
			if r := bridgeutils.SkipGate(false, true, false); r != "" {
				Skip(r)
			}
			name := "fit45-lifecycle"
			engineCfg := name + "-vllm-aggregate-engine-config"
			deploySuffixes := bridgeutils.AggregateInsvcDeploySuffixes()
			cleanupInsvcEntry(fitID, name, engineCfg, deploySuffixes, bridgeutils.GatewayCleanupRefs{}, true)
			path := applyDeployFixtureNoDefer(fitID, "fit45_insvc_delete_lifecycle.yaml", name)
			DeferCleanup(func() {
				finalizeLifecycleInsvc(fitID, name, path, engineCfg, deploySuffixes)
			})

			waitAggregateInsvcFullyReady(fitID, name)

			Expect(bridgeutils.DeleteInferNexService(suiteCtx, k8sClient.DynamicClient, bridgeutils.E2ENS(), name)).To(Succeed())
			assertLifecycleAfterDeleteInsvc(fitID, name, engineCfg, deploySuffixes)
			bridgeutils.LogStep(fitID, "FIT_45 delete InferNexService owner + gateway cleanup verified")
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_46 | 摘除 runtime 标签 | P1
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_46-生命周期 - 摘除 LLMInferenceService 的 runtime标签后Bridge自动清理InferNexService",
		Label("infernex-bridge-lifecycle", "infernex-bridge-llmisvc"),
		func() {
			fitID := "InferNexBridge_FIT_46"
			requireLLMISVC(fitID)
			if r := bridgeutils.SkipGate(true, true, false); r != "" {
				Skip(r)
			}
			name := "fit46-llm"
			deploySuffixes := bridgeutils.AggregateLLMISvcDeploySuffixes()
			cleanupLLMISvcEntry(fitID, name)
			path := applyLLMISvcFixtureNoDefer(fitID, "fit46_llmisvc_runtime_label_lifecycle.yaml", name)
			DeferCleanup(func() {
				finalizeLifecycleLLMISVC(fitID, name, path, deploySuffixes)
			})

			waitAggregateLLMISvcFullyReady(fitID, name)

			_, _, err := bridgeutils.PatchLLMInferenceServiceLabels(bridgeutils.E2ENS(), name, `{"metadata":{"labels":{"infernex.io/runtime":null}}}`)
			Expect(err).NotTo(HaveOccurred())
			bridgeutils.WaitInferNexServiceAbsent(suiteCtx, k8sClient.DynamicClient, fitID, bridgeutils.E2ENS(), name, 5*time.Minute)
			_, err = bridgeutils.GetLLMInferenceService(suiteCtx, k8sClient.DynamicClient, bridgeutils.E2ENS(), name)
			Expect(err).NotTo(HaveOccurred(), "LLMInferenceService should remain after runtime label removed")
			bridgeutils.LogStep(fitID, "FIT_46 runtime label removed; InferNexService gone, LLMISVC retained; DeferCleanup will remove workload")
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_48 | Mooncake PD | P0
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_48-Mooncake - InferNexService模式PD部署Mooncake就绪与KV缓存命中",
		Label("infernex-bridge-mooncake", "infernex-bridge-deploy", "P0"),
		func() {
			fitID := "InferNexBridge_FIT_48"
			requireBridgeInstalled(fitID)
			if r := bridgeutils.SkipGate(false, true, true); r != "" {
				Skip(r)
			}
			name := "fit48-mooncake-pd"
			engineCfg := name + "-vllm-pd-engine-config"
			deploySuffixes := bridgeutils.PDInsvcDeploySuffixes()
			h := beginInsvcWorkloadCleanup(fitID, name, engineCfg, deploySuffixes)
			path := applyDeployFixtureNoDefer(fitID, "fit48_mooncake_pd_insvc.yaml", name)
			h.SetManifest(path, true)

			waitPDInsvcFullyReady(fitID, name)
			assertMooncakeDeploymentsReady(fitID, name)
			assertMooncakeInferenceAndKVWithEngine(fitID, name, fitID, bridgeutils.DeployName(name, "engine-pd-prefill"))
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_49 | Mooncake LLMISVC PD | P0
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_49-Mooncake - LLMInferenceService模式PD部署Mooncake就绪与KV缓存命中",
		Label("infernex-bridge-mooncake", "infernex-bridge-llmisvc", "P0"),
		func() {
			fitID := "InferNexBridge_FIT_49"
			requireLLMISVC(fitID)
			if r := bridgeutils.SkipGate(true, true, true); r != "" {
				Skip(r)
			}
			name := "fit49-pd-llm"
			deploySuffixes := bridgeutils.PDLLMISvcDeploySuffixes()
			h := beginLLMISvcWorkloadCleanup(fitID, name, deploySuffixes)
			path := applyLLMISvcFixtureNoDefer(fitID, "fit49_mooncake_pd_llmisvc.yaml", name)
			h.SetManifest(path, true)

			waitPDLLMISvcFullyReady(fitID, name)
			assertMooncakeDeploymentsReady(fitID, name)
			assertMooncakeInferenceAndKVWithEngine(fitID, name, fitID, bridgeutils.KServePrefillWorkloadDeployName(name))
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_50 | Mooncake聚合部署就绪 | P0
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_50-Mooncake - InferNexService模式聚合部署Mooncake就绪与KV缓存命中",
		Label("infernex-bridge-mooncake", "infernex-bridge-deploy", "P0"),
		func() {
			fitID := "InferNexBridge_FIT_50"
			requireBridgeInstalled(fitID)
			if r := bridgeutils.SkipGate(false, true, true); r != "" {
				Skip(r)
			}
			name := "fit50-mooncake"
			engineCfg := name + "-vllm-aggregate-engine-config"
			deploySuffixes := bridgeutils.AggregateInsvcDeploySuffixes()
			h := beginInsvcWorkloadCleanup(fitID, name, engineCfg, deploySuffixes)
			path := applyDeployFixtureNoDefer(fitID, "fit50_mooncake_aggregate_insvc.yaml", name)
			h.SetManifest(path, true)

			waitAggregateInsvcFullyReady(fitID, name)
			assertMooncakeDeploymentsReady(fitID, name)
			assertMooncakeInferenceAndKVWithEngine(fitID, name, fitID, bridgeutils.DeployName(name, "engine-aggregate"))
		})

	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_51 | Mooncake LLMISVC 聚合 | P0
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_51-Mooncake - LLMInferenceService模式聚合部署Mooncake就绪与KV缓存命中",
		Label("infernex-bridge-mooncake", "infernex-bridge-llmisvc", "P0"),
		func() {
			fitID := "InferNexBridge_FIT_51"
			requireLLMISVC(fitID)
			if r := bridgeutils.SkipGate(true, true, true); r != "" {
				Skip(r)
			}
			name := "fit51-mooncake-llm"
			deploySuffixes := bridgeutils.AggregateLLMISvcDeploySuffixes()
			h := beginLLMISvcWorkloadCleanup(fitID, name, deploySuffixes)
			path := applyLLMISvcFixtureNoDefer(fitID, "fit51_mooncake_aggregate_llmisvc.yaml", name)
			h.SetManifest(path, true)

			waitAggregateLLMISvcFullyReady(fitID, name)
			assertMooncakeDeploymentsReady(fitID, name)
			assertMooncakeInferenceAndKVWithEngine(fitID, name, fitID, bridgeutils.KServeAggregateWorkloadDeployName(name))
		})

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_56 | InferNexService 聚合 LWS 多 DP | P1
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_56-推理引擎 - InferNexService方式聚合LWS多DP部署",
		Label("infernex-bridge-lws", "skip-temporarily"),
		func() {
			fitID := "InferNexBridge_FIT_56"
			if r := skipLWSReason(); r != "" {
				Skip(r)
			}
			requireBridgeInstalled(fitID)
			requireLWSOperator(fitID)
			if r := bridgeutils.SkipGate(false, true, false); r != "" {
				Skip(r)
			}
			name := "fit56-lws"
			_ = beginInsvcAggLWSTest(fitID, name, fit56InsvcAggLWSFixture)
			waitInsvcAggLWSFullyReady(fitID, name)
			assertInsvcAggLWSTopology(fitID, name)
			assertPostWorkloadConnectivity(fitID, name, workloadConnectivityAggregateInsvc, bridgeutils.MoEOfflineInferenceModel)
		})

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_57 | LLMInferenceService 聚合 LWS 多 DP | P1
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_57-推理引擎 - LLMInferenceService方式聚合LWS多DP部署",
		Label("infernex-bridge-lws", "infernex-bridge-llmisvc", "skip-temporarily"),
		func() {
			fitID := "InferNexBridge_FIT_57"
			if r := skipLWSReason(); r != "" {
				Skip(r)
			}
			requireLLMISVC(fitID)
			requireLWSOperator(fitID)
			if r := bridgeutils.SkipGate(true, true, false); r != "" {
				Skip(r)
			}
			name := "fit57-lws"
			_ = beginLLMISvcAggLWSTest(fitID, name, fit57LLMISvcAggLWSFixture)
			waitLLMISvcAggLWSFullyReady(fitID, name)
			assertLLMISvcAggLWSTopology(fitID, name)
			assertPostWorkloadConnectivity(fitID, name, workloadConnectivityAggregateLLMISVC, bridgeutils.MoEOfflineInferenceModel)
		})

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_58 | InferNexService PD LWS 多 DP | P1
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_58-推理引擎 - InferNexService方式跨机MoE PD LWS多DP部署",
		Label("infernex-bridge-lws", "skip-temporarily"),
		func() {
			fitID := "InferNexBridge_FIT_58"
			if r := skipLWSReason(); r != "" {
				Skip(r)
			}
			requireBridgeInstalled(fitID)
			requireLWSOperator(fitID)
			if r := bridgeutils.SkipGate(false, true, false); r != "" {
				Skip(r)
			}
			name := "fit58-lws"
			_ = beginInsvcPDLWSTest(fitID, name, fit58InsvcPDLWSFixture)
			waitInsvcPDLWSFullyReady(fitID, name)
			assertInsvcPDLWSTopology(fitID, name)
			assertPostWorkloadConnectivity(fitID, name, workloadConnectivityPDInsvc, bridgeutils.MoEOfflineInferenceModel)
		})

	// ---------------------------------------------------------------------------
	// InferNexBridge_FIT_59 | LLMInferenceService PD LWS 多 DP | P1
	// ---------------------------------------------------------------------------
	It("InferNexBridge_FIT_59-推理引擎 - LLMInferenceService方式跨机MoE PD LWS多DP部署",
		Label("infernex-bridge-lws", "infernex-bridge-llmisvc", "skip-temporarily"),
		func() {
			fitID := "InferNexBridge_FIT_59"
			if r := skipLWSReason(); r != "" {
				Skip(r)
			}
			requireLLMISVC(fitID)
			requireLWSOperator(fitID)
			if r := bridgeutils.SkipGate(true, true, false); r != "" {
				Skip(r)
			}
			name := "fit59-lws"
			_ = beginLLMISvcPDLWSTest(fitID, name, fit59LLMISvcPDLWSFixture)
			waitLLMISvcPDLWSFullyReady(fitID, name)
			assertLLMISvcPDLWSTopology(fitID, name)
			assertPostWorkloadConnectivity(fitID, name, workloadConnectivityPDLLMISVC, bridgeutils.MoEOfflineInferenceModel)
		})

})
