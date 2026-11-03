package weight_dispatcher

import (
	"fmt"
	"time"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/weight-dispatcher/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Weight-Dispatcher功能测试", Ordered, Label("weight-dispatcher", "npu", "rdma"), func() {
	ns := WeightDispatcherTestNamespace
	var modelWarmupConfig ModelWarmupJobE2EConfig
	var fit2CachePath string

	BeforeAll(func() {
		Expect(WaitForNamespaceDeletionIfTerminating(clientset, ctx, WeightDispatcherHelmNamespace, 3*time.Minute, 5*time.Second)).NotTo(HaveOccurred())
		Expect(InstallWeightDispatcher()).NotTo(HaveOccurred())
		VerifyHelmReleaseStatus(WeightDispatcherHelmReleaseName, WeightDispatcherHelmNamespace)
		VerifyModelWarmupJobCRDEstablished(apiExtensionsClient, ctx, 3*time.Minute, 5*time.Second)
		VerifyWeightDispatcherPodsRunning(clientset, ctx, 5*time.Minute, 10*time.Second)
		Expect(EnsureNodeAgentDefaultRuntimeEnv(clientset, ctx, 5*time.Minute)).NotTo(HaveOccurred())
		Expect(PrepareWeightDispatcherTestData(clientset, ctx)).NotTo(HaveOccurred())

		Expect(EnsureNamespace(clientset, ctx, ns)).NotTo(HaveOccurred())
		Expect(DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT2Name)).NotTo(HaveOccurred())
		Expect(DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT3Name)).NotTo(HaveOccurred())
		Expect(DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT4Name)).NotTo(HaveOccurred())
		Expect(DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT4Join)).NotTo(HaveOccurred())
		Expect(DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT5Name)).NotTo(HaveOccurred())
		Expect(DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT6Name)).NotTo(HaveOccurred())
		Expect(DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT7Name)).NotTo(HaveOccurred())
		Expect(DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT9Name)).NotTo(HaveOccurred())
		Expect(DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT11Name)).NotTo(HaveOccurred())
		Expect(DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT12Name)).NotTo(HaveOccurred())
		Expect(DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT13Name)).NotTo(HaveOccurred())
		Expect(DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT14Name)).NotTo(HaveOccurred())
		Expect(DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT16Name)).NotTo(HaveOccurred())
		Expect(DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT17Name)).NotTo(HaveOccurred())
		Expect(DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT18Name)).NotTo(HaveOccurred())
		Expect(DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT19Name)).NotTo(HaveOccurred())
		Expect(DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT20Name)).NotTo(HaveOccurred())
		Expect(DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT21Name)).NotTo(HaveOccurred())
		Expect(DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT22Name)).NotTo(HaveOccurred())
		Expect(DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT23Name)).NotTo(HaveOccurred())
		Expect(DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT24Name)).NotTo(HaveOccurred())
		Expect(DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT24RetryName)).NotTo(HaveOccurred())
		Expect(DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT25Name)).NotTo(HaveOccurred())
		Expect(DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT26Name)).NotTo(HaveOccurred())
		for i := 0; i < ModelWarmupJobFIT15BatchCount; i++ {
			Expect(DeleteModelWarmupJob(dynamicClient, ctx, ns, fmt.Sprintf("%s-%02d", ModelWarmupJobFIT15Base, i))).NotTo(HaveOccurred())
		}
	})

	BeforeEach(func() {
		specName := CurrentSpecReport().LeafNodeText
		Expect(EnsureNodeAgentDefaultRuntimeEnv(clientset, ctx, 5*time.Minute)).NotTo(HaveOccurred())
		readyRoot := GetSuiteRuntimeConfig().TargetPath
		if readyRoot == "" {
			readyRoot = ModelWarmupJobDefaultTargetPath
		}
		preservePath := ""
		if fit2CachePath != "" && ShouldPreserveReadyCacheForSpec(specName) {
			preservePath = fit2CachePath
		}
		Expect(CleanReadyCacheOnAllNodeAgentsExcept(clientset, ctx, readyRoot, preservePath)).NotTo(HaveOccurred())
	})

	It("ModelWarmupJob_FIT_1 控制器与节点代理部署健康验证", func() {
		VerifyHelmReleaseStatus(WeightDispatcherHelmReleaseName, WeightDispatcherHelmNamespace)
		VerifyModelWarmupJobCRDEstablished(apiExtensionsClient, ctx, 3*time.Minute, 5*time.Second)
		VerifyWeightDispatcherPodsRunning(clientset, ctx, 5*time.Minute, 10*time.Second)
		Expect(EnsureNodeAgentDefaultRuntimeEnv(clientset, ctx, 5*time.Minute)).NotTo(HaveOccurred())
	})

	It("ModelWarmupJob_FIT_2 单节点显式预热成功并发布 READY 缓存", func() {
		var err error
		modelWarmupConfig, err = ResolveModelWarmupJobE2EConfig(clientset, ctx)
		Expect(err).NotTo(HaveOccurred())

		job := BuildModelWarmupJob(ModelWarmupJobFIT2Name, ns, modelWarmupConfig)
		Expect(CreateModelWarmupJob(dynamicClient, ctx, ns, job)).NotTo(HaveOccurred())

		succeededJob := WaitForModelWarmupJobSucceeded(dynamicClient, ctx, ns, ModelWarmupJobFIT2Name, 30*time.Minute)
		fit2CachePath = VerifySingleNodeWarmupSucceeded(succeededJob, modelWarmupConfig)
		VerifyJobTransferredFilesMatchSource(clientset, ctx, succeededJob)
	})

	It("ModelWarmupJob_FIT_3 同规格新建作业命中 READY 缓存", func() {
		if modelWarmupConfig.ArtifactKey == "" {
			var err error
			modelWarmupConfig, err = ResolveModelWarmupJobE2EConfig(clientset, ctx)
			Expect(err).NotTo(HaveOccurred())
		}

		job := BuildModelWarmupJob(ModelWarmupJobFIT3Name, ns, modelWarmupConfig)
		Expect(CreateModelWarmupJob(dynamicClient, ctx, ns, job)).NotTo(HaveOccurred())

		succeededJob := WaitForModelWarmupJobSucceeded(dynamicClient, ctx, ns, ModelWarmupJobFIT3Name, 30*time.Minute)
		VerifyReadyCacheHitSucceeded(succeededJob, modelWarmupConfig, fit2CachePath)
		VerifyJobTransferredFilesMatchSource(clientset, ctx, succeededJob)
	})

	It("ModelWarmupJob_FIT_4 同 artifact 并发预热共享单次写入", func() {
		if modelWarmupConfig.ArtifactKey == "" {
			var err error
			modelWarmupConfig, err = ResolveModelWarmupJobE2EConfig(clientset, ctx)
			Expect(err).NotTo(HaveOccurred())
		}
		concurrentConfig := modelWarmupConfig
		concurrentConfig.ArtifactKey = modelWarmupConfig.ArtifactKey + "-fit4"

		jobA := BuildModelWarmupJob(ModelWarmupJobFIT4Name, ns, concurrentConfig)
		jobB := BuildModelWarmupJob(ModelWarmupJobFIT4Join, ns, concurrentConfig)
		errCh := make(chan error, 2)
		go func() { errCh <- CreateModelWarmupJob(dynamicClient, ctx, ns, jobA) }()
		go func() { errCh <- CreateModelWarmupJob(dynamicClient, ctx, ns, jobB) }()
		Expect(<-errCh).NotTo(HaveOccurred())
		Expect(<-errCh).NotTo(HaveOccurred())

		succeededA := WaitForModelWarmupJobSucceeded(dynamicClient, ctx, ns, ModelWarmupJobFIT4Name, 30*time.Minute)
		succeededB := WaitForModelWarmupJobSucceeded(dynamicClient, ctx, ns, ModelWarmupJobFIT4Join, 30*time.Minute)
		VerifyConcurrentWarmupSharedCache(succeededA, succeededB, concurrentConfig)
		VerifyJobTransferredFilesMatchSource(clientset, ctx, succeededA)
	})

	It("ModelWarmupJob_FIT_5 基于 nodeSelector 解析多目标节点成功", func() {
		if modelWarmupConfig.ArtifactKey == "" {
			var err error
			modelWarmupConfig, err = ResolveModelWarmupJobE2EConfig(clientset, ctx)
			Expect(err).NotTo(HaveOccurred())
		}
		targetNodes, err := ResolveTargetNodes(clientset, ctx, 2)
		Expect(err).NotTo(HaveOccurred())
		sourceNodes, err := ResolveSourceNodes(clientset, ctx, 1)
		Expect(err).NotTo(HaveOccurred())
		selectorKey := ModelWarmupJobFIT5LabelKey
		selectorValue := ModelWarmupJobFIT5LabelValue
		Expect(LabelNodes(clientset, ctx, targetNodes, selectorKey, selectorValue)).NotTo(HaveOccurred())

		cfg := modelWarmupConfig
		cfg.ArtifactKey = modelWarmupConfig.ArtifactKey + "-fit5"
		job := BuildModelWarmupJobWithNodeSelector(ModelWarmupJobFIT5Name, ns, cfg,
			[]ModelWarmupJobSource{{NodeName: sourceNodes[0], RootPath: cfg.SourcePath}},
			map[string]string{selectorKey: selectorValue})
		Expect(CreateModelWarmupJob(dynamicClient, ctx, ns, job)).NotTo(HaveOccurred())

		succeededJob := WaitForModelWarmupJobSucceeded(dynamicClient, ctx, ns, ModelWarmupJobFIT5Name, 30*time.Minute)
		VerifyNodeSelectorResolved(succeededJob, targetNodes)
		VerifyJobTransferredFilesMatchSource(clientset, ctx, succeededJob)
	})

	It("ModelWarmupJob_FIT_6 单源双目标权重分发成功", func() {
		if modelWarmupConfig.ArtifactKey == "" {
			var err error
			modelWarmupConfig, err = ResolveModelWarmupJobE2EConfig(clientset, ctx)
			Expect(err).NotTo(HaveOccurred())
		}
		targetNodes, err := ResolveTargetNodes(clientset, ctx, 2)
		Expect(err).NotTo(HaveOccurred())
		sourceNodes, err := ResolveSourceNodes(clientset, ctx, 1)
		Expect(err).NotTo(HaveOccurred())

		cfg := modelWarmupConfig
		cfg.ArtifactKey = modelWarmupConfig.ArtifactKey + "-fit6"
		job := BuildModelWarmupJobWithNodes(ModelWarmupJobFIT6Name, ns, cfg,
			[]ModelWarmupJobSource{{NodeName: sourceNodes[0], RootPath: cfg.SourcePath}},
			targetNodes)
		Expect(CreateModelWarmupJob(dynamicClient, ctx, ns, job)).NotTo(HaveOccurred())

		succeededJob := WaitForModelWarmupJobSucceeded(dynamicClient, ctx, ns, ModelWarmupJobFIT6Name, 30*time.Minute)
		VerifyMultiTargetWarmupSucceeded(succeededJob, 2, true)
		VerifyJobTransferredFilesMatchSource(clientset, ctx, succeededJob)
	})

	It("ModelWarmupJob_FIT_7 多源单目标分条带传输成功", func() {
		if modelWarmupConfig.ArtifactKey == "" {
			var err error
			modelWarmupConfig, err = ResolveModelWarmupJobE2EConfig(clientset, ctx)
			Expect(err).NotTo(HaveOccurred())
		}
		sourceNodes, err := ResolveSourceNodes(clientset, ctx, 2)
		Expect(err).NotTo(HaveOccurred())
		targetNodes, err := ResolveTargetNodes(clientset, ctx, 1)
		Expect(err).NotTo(HaveOccurred())

		cfg := modelWarmupConfig
		cfg.ArtifactKey = modelWarmupConfig.ArtifactKey + "-fit7"
		job := BuildModelWarmupJobWithNodes(ModelWarmupJobFIT7Name, ns, cfg,
			[]ModelWarmupJobSource{
				{NodeName: sourceNodes[0], RootPath: cfg.SourcePath},
				{NodeName: sourceNodes[1], RootPath: cfg.SourcePath},
			},
			targetNodes)
		Expect(CreateModelWarmupJob(dynamicClient, ctx, ns, job)).NotTo(HaveOccurred())

		succeededJob := WaitForModelWarmupJobSucceeded(dynamicClient, ctx, ns, ModelWarmupJobFIT7Name, 30*time.Minute)
		VerifyMultiTargetWarmupSucceeded(succeededJob, 1, false)
		VerifyJobTransferredFilesMatchSource(clientset, ctx, succeededJob)
	})

	It("ModelWarmupJob_FIT_9 异构源根路径对齐后预热成功", func() {
		if modelWarmupConfig.ArtifactKey == "" {
			var err error
			modelWarmupConfig, err = ResolveModelWarmupJobE2EConfig(clientset, ctx)
			Expect(err).NotTo(HaveOccurred())
		}
		sourceNodes, err := ResolveSourceNodes(clientset, ctx, 2)
		Expect(err).NotTo(HaveOccurred())
		targetNodes, err := ResolveTargetNodes(clientset, ctx, 1)
		Expect(err).NotTo(HaveOccurred())

		cfg := modelWarmupConfig
		cfg.ArtifactKey = modelWarmupConfig.ArtifactKey + "-fit9"
		secondSourcePath := ResolveSecondSourcePath(cfg)
		job := BuildModelWarmupJobWithNodes(ModelWarmupJobFIT9Name, ns, cfg,
			[]ModelWarmupJobSource{
				{NodeName: sourceNodes[0], RootPath: cfg.SourcePath},
				{NodeName: sourceNodes[1], RootPath: secondSourcePath},
			},
			targetNodes)
		Expect(CreateModelWarmupJob(dynamicClient, ctx, ns, job)).NotTo(HaveOccurred())

		succeededJob := WaitForModelWarmupJobSucceeded(dynamicClient, ctx, ns, ModelWarmupJobFIT9Name, 30*time.Minute)
		VerifyMultiTargetWarmupSucceeded(succeededJob, 1, false)
		VerifyNoManifestMismatch(succeededJob)
		VerifyJobTransferredFilesMatchSource(clientset, ctx, succeededJob)
	})

	It("ModelWarmupJob_FIT_10 更新策略后重建计划并重新调谐", func() {
		fit2Job, err := GetModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT10Name)
		Expect(err).NotTo(HaveOccurred())
		oldPlanID := LastPlanID(fit2Job)
		oldTaskID := FirstTaskID(fit2Job)

		_, err = PatchModelWarmupJobChunkSize(dynamicClient, ctx, ns, ModelWarmupJobFIT10Name, 32)
		Expect(err).NotTo(HaveOccurred())

		updatedJob := WaitForModelWarmupJobReplannedAndSucceeded(dynamicClient, ctx, ns, ModelWarmupJobFIT10Name, oldPlanID, oldTaskID, 30*time.Minute)
		VerifyReadyCacheHitSucceeded(updatedJob, modelWarmupConfig, fit2CachePath)
		VerifyJobTransferredFilesMatchSource(clientset, ctx, updatedJob)
	})

	It("ModelWarmupJob_FIT_11 运行中状态与节点统计信息可查询", func() {
		if modelWarmupConfig.ArtifactKey == "" {
			var err error
			modelWarmupConfig, err = ResolveModelWarmupJobE2EConfig(clientset, ctx)
			Expect(err).NotTo(HaveOccurred())
		}
		targetNodes, err := ResolveTargetNodes(clientset, ctx, 2)
		Expect(err).NotTo(HaveOccurred())

		cfg := modelWarmupConfig
		cfg.ArtifactKey = modelWarmupConfig.ArtifactKey + "-fit11"
		job := BuildModelWarmupJobWithNodes(ModelWarmupJobFIT11Name, ns, cfg,
			[]ModelWarmupJobSource{{NodeName: cfg.SourceNode, RootPath: cfg.SourcePath}},
			targetNodes)
		Expect(CreateModelWarmupJob(dynamicClient, ctx, ns, job)).NotTo(HaveOccurred())

		runningJob := WaitForModelWarmupJobRunning(dynamicClient, ctx, ns, ModelWarmupJobFIT11Name, 5*time.Minute)
		VerifyRunningStatusQueryable(runningJob)
		succeededJob := WaitForModelWarmupJobSucceeded(dynamicClient, ctx, ns, ModelWarmupJobFIT11Name, 30*time.Minute)
		VerifyNodeStateStatistics(succeededJob)
		VerifyJobTransferredFilesMatchSource(clientset, ctx, succeededJob)
	})

	It("ModelWarmupJob_FIT_12 控制器重启期间预热任务可继续收敛", func() {
		if modelWarmupConfig.ArtifactKey == "" {
			var err error
			modelWarmupConfig, err = ResolveModelWarmupJobE2EConfig(clientset, ctx)
			Expect(err).NotTo(HaveOccurred())
		}
		targetNodes, err := ResolveTargetNodes(clientset, ctx, 2)
		Expect(err).NotTo(HaveOccurred())

		cfg := modelWarmupConfig
		cfg.ArtifactKey = modelWarmupConfig.ArtifactKey + "-fit12"
		job := BuildModelWarmupJobWithNodes(ModelWarmupJobFIT12Name, ns, cfg,
			[]ModelWarmupJobSource{{NodeName: cfg.SourceNode, RootPath: cfg.SourcePath}},
			targetNodes)
		Expect(CreateModelWarmupJob(dynamicClient, ctx, ns, job)).NotTo(HaveOccurred())
		_ = WaitForModelWarmupJobRunning(dynamicClient, ctx, ns, ModelWarmupJobFIT12Name, 5*time.Minute)

		Expect(RestartWeightDispatcherController(clientset, ctx, 5*time.Minute)).NotTo(HaveOccurred())
		succeededJob := WaitForModelWarmupJobSucceeded(dynamicClient, ctx, ns, ModelWarmupJobFIT12Name, 30*time.Minute)
		VerifyNodeStateStatistics(succeededJob)
		VerifyJobTransferredFilesMatchSource(clientset, ctx, succeededJob)
	})

	It("ModelWarmupJob_FIT_13 启用 chunk CRC32C 与 post-write 校验后预热成功", func() {
		if modelWarmupConfig.ArtifactKey == "" {
			var err error
			modelWarmupConfig, err = ResolveModelWarmupJobE2EConfig(clientset, ctx)
			Expect(err).NotTo(HaveOccurred())
		}
		Expect(PatchWeightDispatcherNodeAgentDaemonSetEnv(clientset, ctx, map[string]string{
			"WD_ENABLE_POST_WRITE_CRC32C": "true",
		}, 5*time.Minute)).NotTo(HaveOccurred())
		DeferCleanup(func() {
			Expect(PatchWeightDispatcherNodeAgentDaemonSetEnvIfPresent(clientset, ctx, map[string]string{
				"WD_ENABLE_POST_WRITE_CRC32C": "false",
			}, 5*time.Minute)).NotTo(HaveOccurred())
		})

		cfg := modelWarmupConfig
		cfg.ArtifactKey = modelWarmupConfig.ArtifactKey + "-fit13"
		job := BuildModelWarmupJobWithPolicy(ModelWarmupJobFIT13Name, ns, cfg, func(policy map[string]interface{}) {
			policy["enableChunkCRC32C"] = true
			policy["chunkSizeMB"] = int64(16)
		})
		Expect(CreateModelWarmupJob(dynamicClient, ctx, ns, job)).NotTo(HaveOccurred())

		succeededJob := WaitForModelWarmupJobSucceeded(dynamicClient, ctx, ns, ModelWarmupJobFIT13Name, 30*time.Minute)
		VerifyChunkCRC32CSucceeded(succeededJob, cfg)
		VerifyJobTransferredFilesMatchSource(clientset, ctx, succeededJob)
	})

	It("ModelWarmupJob_FIT_14 artifactKey 特殊字符清洗后路径正确", func() {
		if modelWarmupConfig.ArtifactKey == "" {
			var err error
			modelWarmupConfig, err = ResolveModelWarmupJobE2EConfig(clientset, ctx)
			Expect(err).NotTo(HaveOccurred())
		}

		cfg := modelWarmupConfig
		cfg.ArtifactKey = ModelWarmupJobFIT14ArtifactKey
		job := BuildModelWarmupJob(ModelWarmupJobFIT14Name, ns, cfg)
		Expect(CreateModelWarmupJob(dynamicClient, ctx, ns, job)).NotTo(HaveOccurred())

		succeededJob := WaitForModelWarmupJobSucceeded(dynamicClient, ctx, ns, ModelWarmupJobFIT14Name, 30*time.Minute)
		VerifyArtifactKeySanitizedPath(succeededJob, cfg)
		VerifyJobTransferredFilesMatchSource(clientset, ctx, succeededJob)
	})

	It("ModelWarmupJob_FIT_15 批量作业计划与状态更新测试", func() {
		if modelWarmupConfig.ArtifactKey == "" {
			var err error
			modelWarmupConfig, err = ResolveModelWarmupJobE2EConfig(clientset, ctx)
			Expect(err).NotTo(HaveOccurred())
		}
		batchConfig := modelWarmupConfig
		batchConfig.SourcePath = ResolveSingleFileSourcePath(modelWarmupConfig)

		for i := 0; i < ModelWarmupJobFIT15BatchCount; i++ {
			cfg := batchConfig
			cfg.ArtifactKey = fmt.Sprintf("%s-fit15-%02d", modelWarmupConfig.ArtifactKey, i)
			name := fmt.Sprintf("%s-%02d", ModelWarmupJobFIT15Base, i)
			job := BuildModelWarmupJob(name, ns, cfg)
			Expect(CreateModelWarmupJob(dynamicClient, ctx, ns, job)).NotTo(HaveOccurred())
		}
		for i := 0; i < ModelWarmupJobFIT15BatchCount; i++ {
			name := fmt.Sprintf("%s-%02d", ModelWarmupJobFIT15Base, i)
			succeededJob := WaitForModelWarmupJobSucceeded(dynamicClient, ctx, ns, name, 30*time.Minute)
			VerifySummaryMatchesNodeStates(succeededJob)
			VerifyJobTransferredFilesMatchSource(clientset, ctx, succeededJob)
		}
	})

	It("ModelWarmupJob_FIT_16 缺少 artifact.key 时创建请求被 CRD 拒绝", func() {
		if modelWarmupConfig.ArtifactKey == "" {
			var err error
			modelWarmupConfig, err = ResolveModelWarmupJobE2EConfig(clientset, ctx)
			Expect(err).NotTo(HaveOccurred())
		}

		job := BuildModelWarmupJobMissingArtifactKey(ModelWarmupJobFIT16Name, ns, modelWarmupConfig)
		err := CreateModelWarmupJob(dynamicClient, ctx, ns, job)
		VerifyModelWarmupJobRejected(err, dynamicClient, ctx, ns, ModelWarmupJobFIT16Name)
	})

	It("ModelWarmupJob_FIT_17 缺少 targetPath 时创建请求被 CRD 拒绝", func() {
		if modelWarmupConfig.ArtifactKey == "" {
			var err error
			modelWarmupConfig, err = ResolveModelWarmupJobE2EConfig(clientset, ctx)
			Expect(err).NotTo(HaveOccurred())
		}

		job := BuildModelWarmupJobMissingTargetPath(ModelWarmupJobFIT17Name, ns, modelWarmupConfig)
		err := CreateModelWarmupJob(dynamicClient, ctx, ns, job)
		VerifyModelWarmupJobRejected(err, dynamicClient, ctx, ns, ModelWarmupJobFIT17Name)
	})

	It("ModelWarmupJob_FIT_18 缺少 sources 时校验失败", func() {
		if modelWarmupConfig.ArtifactKey == "" {
			var err error
			modelWarmupConfig, err = ResolveModelWarmupJobE2EConfig(clientset, ctx)
			Expect(err).NotTo(HaveOccurred())
		}

		job := BuildModelWarmupJobWithoutSources(ModelWarmupJobFIT18Name, ns, modelWarmupConfig)
		Expect(CreateModelWarmupJob(dynamicClient, ctx, ns, job)).NotTo(HaveOccurred())

		failedJob := WaitForModelWarmupJobFailed(dynamicClient, ctx, ns, ModelWarmupJobFIT18Name, 5*time.Minute)
		VerifyValidationFailed(failedJob, "spec.sources must not be empty in phase one")
	})

	It("ModelWarmupJob_FIT_19 node 源显式填写 endpoint 时校验失败", func() {
		if modelWarmupConfig.ArtifactKey == "" {
			var err error
			modelWarmupConfig, err = ResolveModelWarmupJobE2EConfig(clientset, ctx)
			Expect(err).NotTo(HaveOccurred())
		}

		job := BuildModelWarmupJobNodeSourceWithEndpoint(ModelWarmupJobFIT19Name, ns, modelWarmupConfig)
		Expect(CreateModelWarmupJob(dynamicClient, ctx, ns, job)).NotTo(HaveOccurred())

		failedJob := WaitForModelWarmupJobFailed(dynamicClient, ctx, ns, ModelWarmupJobFIT19Name, 5*time.Minute)
		VerifyValidationFailed(failedJob, "sourceType=node must not set endpoint explicitly")
	})

	It("ModelWarmupJob_FIT_20 node 源缺少 nodeName 时校验失败", func() {
		if modelWarmupConfig.ArtifactKey == "" {
			var err error
			modelWarmupConfig, err = ResolveModelWarmupJobE2EConfig(clientset, ctx)
			Expect(err).NotTo(HaveOccurred())
		}

		job := BuildModelWarmupJobNodeSourceMissingNodeName(ModelWarmupJobFIT20Name, ns, modelWarmupConfig)
		Expect(CreateModelWarmupJob(dynamicClient, ctx, ns, job)).NotTo(HaveOccurred())

		failedJob := WaitForModelWarmupJobFailed(dynamicClient, ctx, ns, ModelWarmupJobFIT20Name, 5*time.Minute)
		VerifyValidationFailed(failedJob, "sourceType=node requires nodeName")
	})

	It("ModelWarmupJob_FIT_21 重复 source 定义时校验失败", func() {
		if modelWarmupConfig.ArtifactKey == "" {
			var err error
			modelWarmupConfig, err = ResolveModelWarmupJobE2EConfig(clientset, ctx)
			Expect(err).NotTo(HaveOccurred())
		}

		job := BuildModelWarmupJobWithDuplicateSources(ModelWarmupJobFIT21Name, ns, modelWarmupConfig)
		Expect(CreateModelWarmupJob(dynamicClient, ctx, ns, job)).NotTo(HaveOccurred())

		failedJob := WaitForModelWarmupJobFailed(dynamicClient, ctx, ns, ModelWarmupJobFIT21Name, 5*time.Minute)
		VerifyValidationFailed(failedJob, "duplicated source")
	})

	It("ModelWarmupJob_FIT_22 多 node 源 manifest 布局不一致时阻断任务", func() {
		if modelWarmupConfig.ArtifactKey == "" {
			var err error
			modelWarmupConfig, err = ResolveModelWarmupJobE2EConfig(clientset, ctx)
			Expect(err).NotTo(HaveOccurred())
		}
		sourceNodes, err := ResolveSourceNodes(clientset, ctx, 2)
		Expect(err).NotTo(HaveOccurred())
		targetNodes, err := ResolveTargetNodes(clientset, ctx, 1)
		Expect(err).NotTo(HaveOccurred())

		cfg := modelWarmupConfig
		cfg.ArtifactKey = modelWarmupConfig.ArtifactKey + "-fit22"
		mismatchSourcePath := ResolveManifestMismatchSourcePath(cfg)
		job := BuildModelWarmupJobWithNodes(ModelWarmupJobFIT22Name, ns, cfg,
			[]ModelWarmupJobSource{
				{NodeName: sourceNodes[0], RootPath: cfg.SourcePath},
				{NodeName: sourceNodes[1], RootPath: mismatchSourcePath},
			},
			targetNodes)
		Expect(CreateModelWarmupJob(dynamicClient, ctx, ns, job)).NotTo(HaveOccurred())

		failedJob := WaitForModelWarmupJobFailed(dynamicClient, ctx, ns, ModelWarmupJobFIT22Name, 10*time.Minute)
		VerifyManifestMismatchBlocked(failedJob)
	})

	It("ModelWarmupJob_FIT_23 node-agent 不可达后恢复可继续推进", func() {
		if modelWarmupConfig.ArtifactKey == "" {
			var err error
			modelWarmupConfig, err = ResolveModelWarmupJobE2EConfig(clientset, ctx)
			Expect(err).NotTo(HaveOccurred())
		}

		cfg := modelWarmupConfig
		cfg.ArtifactKey = modelWarmupConfig.ArtifactKey + "-fit23"
		job := BuildModelWarmupJob(ModelWarmupJobFIT23Name, ns, cfg)
		Expect(CreateModelWarmupJob(dynamicClient, ctx, ns, job)).NotTo(HaveOccurred())

		runningJob := WaitForModelWarmupJobRunning(dynamicClient, ctx, ns, ModelWarmupJobFIT23Name, 5*time.Minute)
		VerifyModelWarmupJobNotFailed(runningJob)
		Expect(RestartWeightDispatcherNodeAgentOnNode(clientset, ctx, cfg.TargetNode, 5*time.Minute)).NotTo(HaveOccurred())

		succeededJob := WaitForModelWarmupJobSucceeded(dynamicClient, ctx, ns, ModelWarmupJobFIT23Name, 30*time.Minute)
		VerifySingleNodeWarmupSucceeded(succeededJob, cfg)
		VerifyJobTransferredFilesMatchSource(clientset, ctx, succeededJob)
	})

	It("ModelWarmupJob_FIT_24 超小 timeout 触发任务失败且不发布 READY", func() {
		if modelWarmupConfig.ArtifactKey == "" {
			var err error
			modelWarmupConfig, err = ResolveModelWarmupJobE2EConfig(clientset, ctx)
			Expect(err).NotTo(HaveOccurred())
		}

		timeoutCfg := modelWarmupConfig
		timeoutCfg.ArtifactKey = modelWarmupConfig.ArtifactKey + "-fit24"
		timeoutCfg.SourcePath = ResolveTimeoutSourcePath(modelWarmupConfig)
		timeoutJob := BuildModelWarmupJobWithPolicy(ModelWarmupJobFIT24Name, ns, timeoutCfg, func(policy map[string]interface{}) {
			policy["timeoutSeconds"] = int64(1)
		})
		Expect(CreateModelWarmupJob(dynamicClient, ctx, ns, timeoutJob)).NotTo(HaveOccurred())

		failedJob := WaitForModelWarmupJobFailed(dynamicClient, ctx, ns, ModelWarmupJobFIT24Name, 10*time.Minute)
		VerifyTimeoutFailedAndNoReadyCache(failedJob)

		retryJob := BuildModelWarmupJobWithPolicy(ModelWarmupJobFIT24RetryName, ns, timeoutCfg, func(policy map[string]interface{}) {
			policy["timeoutSeconds"] = int64(600)
		})
		Expect(CreateModelWarmupJob(dynamicClient, ctx, ns, retryJob)).NotTo(HaveOccurred())
		succeededJob := WaitForModelWarmupJobSucceeded(dynamicClient, ctx, ns, ModelWarmupJobFIT24RetryName, 30*time.Minute)
		VerifySingleNodeWarmupSucceeded(succeededJob, timeoutCfg)
		VerifyJobTransferredFilesMatchSource(clientset, ctx, succeededJob)
	})

	It("ModelWarmupJob_FIT_25 注入单次 chunk CRC32C 失配后自动重传并成功", func() {
		if modelWarmupConfig.ArtifactKey == "" {
			var err error
			modelWarmupConfig, err = ResolveModelWarmupJobE2EConfig(clientset, ctx)
			Expect(err).NotTo(HaveOccurred())
		}
		Expect(PatchWeightDispatcherNodeAgentDaemonSetEnv(clientset, ctx, map[string]string{
			"WD_ENABLE_POST_WRITE_CRC32C":       "true",
			"WD_INJECT_CHUNK_CRC_MISMATCH_ONCE": "true",
			"WD_FORCE_TCP_FALLBACK":             "false",
		}, 5*time.Minute)).NotTo(HaveOccurred())
		DeferCleanup(func() {
			Expect(PatchWeightDispatcherNodeAgentDaemonSetEnvIfPresent(clientset, ctx, map[string]string{
				"WD_ENABLE_POST_WRITE_CRC32C":       "false",
				"WD_INJECT_CHUNK_CRC_MISMATCH_ONCE": "false",
				"WD_FORCE_TCP_FALLBACK":             "false",
			}, 5*time.Minute)).NotTo(HaveOccurred())
		})

		cfg := modelWarmupConfig
		cfg.ArtifactKey = modelWarmupConfig.ArtifactKey + "-fit25"
		job := BuildModelWarmupJobWithPolicy(ModelWarmupJobFIT25Name, ns, cfg, func(policy map[string]interface{}) {
			policy["enableChunkCRC32C"] = true
		})
		Expect(CreateModelWarmupJob(dynamicClient, ctx, ns, job)).NotTo(HaveOccurred())

		succeededJob := WaitForModelWarmupJobSucceeded(dynamicClient, ctx, ns, ModelWarmupJobFIT25Name, 30*time.Minute)
		VerifyChunkCRCRetrySucceeded(succeededJob, cfg)
		VerifyJobTransferredFilesMatchSource(clientset, ctx, succeededJob)
	})

	It("ModelWarmupJob_FIT_26 强制 TCP fallback 路径后任务成功", func() {
		if modelWarmupConfig.ArtifactKey == "" {
			var err error
			modelWarmupConfig, err = ResolveModelWarmupJobE2EConfig(clientset, ctx)
			Expect(err).NotTo(HaveOccurred())
		}
		Expect(PatchWeightDispatcherNodeAgentDaemonSetEnv(clientset, ctx, map[string]string{
			"WD_INJECT_CHUNK_CRC_MISMATCH_ONCE": "false",
			"WD_FORCE_TCP_FALLBACK":             "true",
		}, 5*time.Minute)).NotTo(HaveOccurred())
		DeferCleanup(func() {
			Expect(PatchWeightDispatcherNodeAgentDaemonSetEnvIfPresent(clientset, ctx, map[string]string{
				"WD_FORCE_TCP_FALLBACK": "false",
			}, 5*time.Minute)).NotTo(HaveOccurred())
		})

		cfg := modelWarmupConfig
		cfg.ArtifactKey = modelWarmupConfig.ArtifactKey + "-fit26"
		job := BuildModelWarmupJob(ModelWarmupJobFIT26Name, ns, cfg)
		Expect(CreateModelWarmupJob(dynamicClient, ctx, ns, job)).NotTo(HaveOccurred())

		succeededJob := WaitForModelWarmupJobSucceeded(dynamicClient, ctx, ns, ModelWarmupJobFIT26Name, 30*time.Minute)
		VerifyTCPFallbackSucceeded(succeededJob)
		VerifyJobTransferredFilesMatchSource(clientset, ctx, succeededJob)
	})

	AfterAll(func() {
		_ = DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT26Name)
		_ = DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT25Name)
		_ = DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT24RetryName)
		_ = DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT24Name)
		_ = DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT23Name)
		_ = DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT22Name)
		_ = DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT21Name)
		_ = DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT20Name)
		_ = DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT19Name)
		_ = DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT18Name)
		_ = DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT17Name)
		_ = DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT16Name)
		for i := 0; i < ModelWarmupJobFIT15BatchCount; i++ {
			_ = DeleteModelWarmupJob(dynamicClient, ctx, ns, fmt.Sprintf("%s-%02d", ModelWarmupJobFIT15Base, i))
		}
		_ = DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT14Name)
		_ = DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT13Name)
		_ = DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT12Name)
		_ = DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT11Name)
		_ = DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT9Name)
		_ = DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT7Name)
		_ = DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT6Name)
		_ = DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT5Name)
		_ = DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT4Join)
		_ = DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT4Name)
		_ = DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT3Name)
		_ = DeleteModelWarmupJob(dynamicClient, ctx, ns, ModelWarmupJobFIT2Name)
		_ = UninstallWeightDispatcher()
	})
})
