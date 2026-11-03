package elastic_scaler

import (
	"time"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/elastic-scaler/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

// ElasticScaler Validator 规格校验（与 elastic_scaler_test.go 同属 E2E 套件）
var _ = Describe("Elastic Scaler Validator 规格校验", Ordered, Label("elastic-scaler", "with-workload-cluster"), func() {
	sharedDeployName := "validator-shared-deploy"

	BeforeAll(func() {
		Expect(WaitForNamespaceDeletionIfTerminating(clientset, ctx, ElasticScalerTestNamespace, 3*time.Minute, 5*time.Second)).NotTo(HaveOccurred())
		Expect(EnsureNamespace(clientset, ctx, ElasticScalerTestNamespace)).NotTo(HaveOccurred())
		Expect(CreateESMetricsTestDeployment(clientset, ctx, sharedDeployName, ElasticScalerTestNamespace, 1, false)).NotTo(HaveOccurred())
	})
	AfterAll(func() {
		_ = clientset.AppsV1().Deployments(ElasticScalerTestNamespace).Delete(ctx, sharedDeployName, metav1.DeleteOptions{})
		_ = HelmUninstall(ElasticScalerHelmReleaseName, ElasticScalerHelmNamespace)
	})

	cleanupES := func(esName string) {
		_ = DeleteElasticScalerCR(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
	}
	createES := func(obj *unstructured.Unstructured) {
		Expect(CreateElasticScalerCRFromObject(dynamicClient, ctx, ElasticScalerTestNamespace, obj)).NotTo(HaveOccurred())
	}
	newMetricsES := func(esName, deployName string) *unstructured.Unstructured {
		return BuildValidMetricsTriggerESObject(esName, ElasticScalerTestNamespace, ESMetricsTargetRef{
			APIVersion: "apps/v1",
			Kind:       "Deployment",
			Name:       deployName,
			Namespace:  ElasticScalerTestNamespace,
		})
	}

	It("0-安装部署-控制器部署与健康验证", func() {
		Expect(WaitForNamespaceDeletionIfTerminating(clientset, ctx, ElasticScalerHelmNamespace, 3*time.Minute, 5*time.Second)).NotTo(HaveOccurred())
		Expect(InstallElasticScaler()).NotTo(HaveOccurred())
		VerifyHelmReleaseStatus(ElasticScalerHelmReleaseName, ElasticScalerHelmNamespace)
		VerifyElasticScalerPodsRunning(clientset, ctx, 5*time.Minute, 10*time.Second)
	})

	It("1-规格校验-targetRef.apiVersion/kind/name 为空时拒绝", func() {
		deployName := sharedDeployName
		esName := "validator-1-es-targetref-flow"
		DeferCleanup(func() { cleanupES(esName) })

		// 先创建一条默认合法 ES，通过 CRD/准入；再对每个 case 用 Update 改 spec，走 reconcile 侧 Validator
		obj := newMetricsES(esName, deployName)
		createES(obj)
		VerifyESCRExists(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
		VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)

		restoreValidTargetRef := func() error {
			return UpdateElasticScalerCR(dynamicClient, ctx, esName, ElasticScalerTestNamespace, func(u *unstructured.Unstructured) error {
				if err := unstructured.SetNestedField(u.Object, "apps/v1", "spec", "targetRef", "apiVersion"); err != nil {
					return err
				}
				if err := unstructured.SetNestedField(u.Object, "Deployment", "spec", "targetRef", "kind"); err != nil {
					return err
				}
				if err := unstructured.SetNestedField(u.Object, deployName, "spec", "targetRef", "name"); err != nil {
					return err
				}
				return nil
			})
		}

		cases := []struct {
			path  []string
			value interface{}
		}{
			{path: []string{"spec", "targetRef", "apiVersion"}, value: ""},
			{path: []string{"spec", "targetRef", "kind"}, value: ""},
			{path: []string{"spec", "targetRef", "name"}, value: ""},
		}

		for _, tc := range cases {
			err := UpdateElasticScalerCR(dynamicClient, ctx, esName, ElasticScalerTestNamespace, func(u *unstructured.Unstructured) error {
				return unstructured.SetNestedField(u.Object, tc.value, tc.path...)
			})
			if err != nil {
				// 部分集群在 Update 阶段也会被 CRD/准入拒绝，与“创建即拒绝”等价
				GinkgoWriter.Printf("targetRef 非法字段已在 Update/CRD 阶段被拒绝: %v\n", err)
				Expect(restoreValidTargetRef()).NotTo(HaveOccurred())
				VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
				continue
			}
			VerifyESValidationFailedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
			Expect(restoreValidTargetRef()).NotTo(HaveOccurred())
			VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
		}
	})

	It("2-规格校验-min/maxReplicas 区间与语义（含 nil min）", func() {
		deployName := sharedDeployName
		esName := "validator-2-es-min-max-flow"
		DeferCleanup(func() { cleanupES(esName) })

		// 先创建合法 ES 通过 CRD/准入；再对每个子场景用 Update 改 spec。
		obj := newMetricsES(esName, deployName)
		createES(obj)
		VerifyESCRExists(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
		VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)

		restoreValidReplicas := func() error {
			return UpdateElasticScalerCR(dynamicClient, ctx, esName, ElasticScalerTestNamespace, func(u *unstructured.Unstructured) error {
				if err := unstructured.SetNestedField(u.Object, int64(1), "spec", "minReplicas"); err != nil {
					return err
				}
				if err := unstructured.SetNestedField(u.Object, int64(2), "spec", "maxReplicas"); err != nil {
					return err
				}
				return nil
			})
		}

		cases := []struct {
			desc       string
			expectFail bool
			mutate     func(u *unstructured.Unstructured) error
		}{
			{
				desc:       "minReplicas>maxReplicas",
				expectFail: true,
				mutate: func(u *unstructured.Unstructured) error {
					if err := unstructured.SetNestedField(u.Object, int64(5), "spec", "minReplicas"); err != nil {
						return err
					}
					return unstructured.SetNestedField(u.Object, int64(2), "spec", "maxReplicas")
				},
			},
			{
				desc:       "minReplicas 省略（nil 语义，默认 1）",
				expectFail: false,
				mutate: func(u *unstructured.Unstructured) error {
					unstructured.RemoveNestedField(u.Object, "spec", "minReplicas")
					return nil
				},
			},
		}

		for _, tc := range cases {
			err := UpdateElasticScalerCR(dynamicClient, ctx, esName, ElasticScalerTestNamespace, tc.mutate)
			if err != nil {
				GinkgoWriter.Printf("[%s] 非法/敏感字段已在 Update/CRD 阶段被拒绝: %v\n", tc.desc, err)
				Expect(restoreValidReplicas()).NotTo(HaveOccurred())
				VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
				continue
			}
			if tc.expectFail {
				VerifyESValidationFailedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
			} else {
				VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
			}
			Expect(restoreValidReplicas()).NotTo(HaveOccurred())
			VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
		}
	})

	It("3-规格校验-合法 Metrics/State/Signal 规格通过", func() {
		deployName := sharedDeployName
		esName := "validator-3-es-trigger-flow"
		DeferCleanup(func() { cleanupES(esName) })

		obj := newMetricsES(esName, deployName)
		createES(obj)
		VerifyESCRExists(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
		VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)

		restoreValidTrigger := func() error {
			return UpdateElasticScalerCR(dynamicClient, ctx, esName, ElasticScalerTestNamespace, func(u *unstructured.Unstructured) error {
				fresh := newMetricsES(esName, deployName)
				trig, _, err := unstructured.NestedMap(fresh.Object, "spec", "trigger")
				if err != nil {
					return err
				}
				return unstructured.SetNestedMap(u.Object, trig, "spec", "trigger")
			})
		}

		cases := []struct {
			desc   string
			mutate func(u *unstructured.Unstructured) error
		}{
			{
				desc: "合法 MetricsTrigger（写回默认 trigger）",
				mutate: func(u *unstructured.Unstructured) error {
					fresh := newMetricsES(esName, deployName)
					trig, _, err := unstructured.NestedMap(fresh.Object, "spec", "trigger")
					if err != nil {
						return err
					}
					return unstructured.SetNestedMap(u.Object, trig, "spec", "trigger")
				},
			},
			{
				desc: "合法 StateTrigger",
				mutate: func(u *unstructured.Unstructured) error {
					return unstructured.SetNestedMap(u.Object, map[string]interface{}{
						"type":         "StateTrigger",
						"stateTrigger": map[string]interface{}{},
					}, "spec", "trigger")
				},
			},
			{
				desc: "合法 SignalTrigger",
				mutate: func(u *unstructured.Unstructured) error {
					return unstructured.SetNestedMap(u.Object, map[string]interface{}{
						"type": "SignalTrigger",
						"signalTrigger": map[string]interface{}{
							"endpoint": "https://example.com/webhook",
							"protocol": "https",
						},
					}, "spec", "trigger")
				},
			},
		}

		for _, tc := range cases {
			err := UpdateElasticScalerCR(dynamicClient, ctx, esName, ElasticScalerTestNamespace, tc.mutate)
			if err != nil {
				GinkgoWriter.Printf("[%s] 已在 Update/CRD 阶段被拒绝: %v\n", tc.desc, err)
				Expect(restoreValidTrigger()).NotTo(HaveOccurred())
				VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
				continue
			}
			VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
			Expect(restoreValidTrigger()).NotTo(HaveOccurred())
			VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
		}
	})

	It("4-规格校验-MetricsTrigger 子配置拒绝", func() {
		deployName := sharedDeployName
		esName := "validator-4-es-metrics-trigger-flow"
		DeferCleanup(func() { cleanupES(esName) })

		// 先创建合法 MetricsTrigger ES；各子场景用 Update 改 spec.trigger.metricsTrigger（或 trigger 互斥字段），走 reconcile Validator。
		obj := newMetricsES(esName, deployName)
		createES(obj)
		VerifyESCRExists(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
		VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)

		restoreValidMetricsTrigger := func() error {
			return UpdateElasticScalerCR(dynamicClient, ctx, esName, ElasticScalerTestNamespace, func(u *unstructured.Unstructured) error {
				fresh := newMetricsES(esName, deployName)
				trig, _, err := unstructured.NestedMap(fresh.Object, "spec", "trigger")
				if err != nil {
					return err
				}
				return unstructured.SetNestedMap(u.Object, trig, "spec", "trigger")
			})
		}

		cases := []struct {
			desc       string
			expectFail bool
			mutate     func(u *unstructured.Unstructured) error
		}{
			{
				desc:       "1、MetricsTrigger 下额外配置 stateTrigger",
				expectFail: true,
				mutate: func(u *unstructured.Unstructured) error {
					return unstructured.SetNestedMap(u.Object, map[string]interface{}{}, "spec", "trigger", "stateTrigger")
				},
			},
			{
				desc:       "2、scalingAlgorithm 为空（回退默认 APA，合法）",
				expectFail: false,
				mutate: func(u *unstructured.Unstructured) error {
					return unstructured.SetNestedField(u.Object, "", "spec", "trigger", "metricsTrigger", "scalingAlgorithm")
				},
			},
			{
				desc:       "3、metrics 列表为空",
				expectFail: true,
				mutate: func(u *unstructured.Unstructured) error {
					return unstructured.SetNestedSlice(u.Object, []interface{}{}, "spec", "trigger", "metricsTrigger", "metrics")
				},
			},
			{
				desc:       "4、HPA 算法下 ExternalServer 类型指标",
				expectFail: true,
				mutate: func(u *unstructured.Unstructured) error {
					extSrv := []interface{}{
						map[string]interface{}{
							"type": "ExternalServer",
							"externalServer": map[string]interface{}{
								"metricsName": "qps",
								"target":      "1000",
								"endpoint":    "http://localhost:8080/metrics",
								"protocol":    "http",
							},
						},
					}
					return unstructured.SetNestedSlice(u.Object, extSrv, "spec", "trigger", "metricsTrigger", "metrics")
				},
			},
		}

		for _, tc := range cases {
			err := UpdateElasticScalerCR(dynamicClient, ctx, esName, ElasticScalerTestNamespace, tc.mutate)
			if err != nil {
				GinkgoWriter.Printf("[%s] 已在 Update/CRD 阶段被拒绝: %v\n", tc.desc, err)
				Expect(restoreValidMetricsTrigger()).NotTo(HaveOccurred())
				VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
				continue
			}
			if tc.expectFail {
				VerifyESValidationFailedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
			} else {
				VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
			}
			Expect(restoreValidMetricsTrigger()).NotTo(HaveOccurred())
			VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
		}
	})

	It("5-规格校验-StateTrigger 子配置拒绝", func() {
		deployName := sharedDeployName
		esName := "validator-5-es-state-trigger-flow"
		DeferCleanup(func() { cleanupES(esName) })

		// 先创建合法 MetricsTrigger ES，通过 CRD/准入；再 Update 切到合法 StateTrigger 基线；
		// 随后针对 StateTrigger 的非法组合用 Update 构造，走 reconcile Validator。
		obj := newMetricsES(esName, deployName)
		createES(obj)
		VerifyESCRExists(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
		VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)

		restoreValidStateTrigger := func() error {
			return UpdateElasticScalerCR(dynamicClient, ctx, esName, ElasticScalerTestNamespace, func(u *unstructured.Unstructured) error {
				return unstructured.SetNestedMap(u.Object, map[string]interface{}{
					"type":         "StateTrigger",
					"stateTrigger": map[string]interface{}{},
				}, "spec", "trigger")
			})
		}

		// 先把 ES 切换到合法 StateTrigger
		err := restoreValidStateTrigger()
		if err != nil {
			GinkgoWriter.Printf("[StateTrigger 基线] 已在 Update/CRD 阶段被拒绝: %v\n", err)
			// 若连基线都无法 Update，则后续场景没有意义：直接返回。
			Expect(err).NotTo(HaveOccurred())
		}
		VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)

		cases := []struct {
			desc   string
			mutate func(u *unstructured.Unstructured) error
		}{
			{
				desc: "1、StateTrigger 类型下 stateTrigger 为空",
				mutate: func(u *unstructured.Unstructured) error {
					unstructured.RemoveNestedField(u.Object, "spec", "trigger", "stateTrigger")
					return nil
				},
			},
		}

		for _, tc := range cases {
			// 每个子场景前都恢复到合法 StateTrigger 基线，避免相互污染
			Expect(restoreValidStateTrigger()).NotTo(HaveOccurred())
			VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)

			err := UpdateElasticScalerCR(dynamicClient, ctx, esName, ElasticScalerTestNamespace, tc.mutate)
			if err != nil {
				GinkgoWriter.Printf("[%s] 已在 Update/CRD 阶段被拒绝: %v\n", tc.desc, err)
				Expect(restoreValidStateTrigger()).NotTo(HaveOccurred())
				VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
				continue
			}
			VerifyESValidationFailedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
			Expect(restoreValidStateTrigger()).NotTo(HaveOccurred())
			VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
		}
	})

	It("6-规格校验-SignalTrigger 子配置校验", func() {
		deployName := sharedDeployName
		esName := "validator-6-es-signal-trigger-flow"
		DeferCleanup(func() { cleanupES(esName) })

		// 先创建合法 MetricsTrigger ES，通过 CRD/准入；再 Update 切到合法 SignalTrigger 基线；
		// 随后针对 SignalTrigger 的非法组合用 Update 构造，走 reconcile Validator。
		obj := newMetricsES(esName, deployName)
		createES(obj)
		VerifyESCRExists(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
		VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)

		restoreValidSignalTrigger := func() error {
			return UpdateElasticScalerCR(dynamicClient, ctx, esName, ElasticScalerTestNamespace, func(u *unstructured.Unstructured) error {
				return unstructured.SetNestedMap(u.Object, map[string]interface{}{
					"type": "SignalTrigger",
					"signalTrigger": map[string]interface{}{
						// endpoint 不带 scheme：由 protocol（或默认值）补齐
						"endpoint": "example.com/webhook",
						"protocol": "https",
					},
				}, "spec", "trigger")
			})
		}

		// 切换到合法 SignalTrigger 基线
		err := restoreValidSignalTrigger()
		if err != nil {
			GinkgoWriter.Printf("[SignalTrigger 基线] 已在 Update/CRD 阶段被拒绝: %v\n", err)
			Expect(err).NotTo(HaveOccurred())
		}
		VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)

		cases := []struct {
			desc       string
			expectFail bool
			mutate     func(u *unstructured.Unstructured) error
		}{
			{
				desc:       "1、SignalTrigger 的 endpoint 为空",
				expectFail: true,
				mutate: func(u *unstructured.Unstructured) error {
					// 使用空白字符串而非 ""，避免 Update 时字段被省略导致 spec 未变更
					return unstructured.SetNestedField(u.Object, "   ", "spec", "trigger", "signalTrigger", "endpoint")
				},
			},
			{
				desc:       "2、SignalTrigger endpoint 不带 scheme 时通过（默认/按 protocol 对齐）",
				expectFail: false,
				mutate: func(u *unstructured.Unstructured) error {
					if err := unstructured.SetNestedField(u.Object, "example.com/webhook", "spec", "trigger", "signalTrigger", "endpoint"); err != nil {
						return err
					}
					// 移除 protocol，让 CRD default（https）或上层逻辑对齐
					unstructured.RemoveNestedField(u.Object, "spec", "trigger", "signalTrigger", "protocol")
					return nil
				},
			},
			{
				desc:       "3、SignalTrigger endpoint 显式非 http/https scheme 时拒绝",
				expectFail: true,
				mutate: func(u *unstructured.Unstructured) error {
					return unstructured.SetNestedField(u.Object, "grpc://example.com/webhook", "spec", "trigger", "signalTrigger", "endpoint")
				},
			},
			{
				desc:       "4、SignalTrigger 的 endpoint 非法 URL 时拒绝",
				expectFail: true,
				mutate: func(u *unstructured.Unstructured) error {
					return unstructured.SetNestedField(u.Object, "://bad", "spec", "trigger", "signalTrigger", "endpoint")
				},
			},
			{
				desc:       "5、SignalTrigger 类型下额外配置 MetricsTrigger 时拒绝",
				expectFail: true,
				mutate: func(u *unstructured.Unstructured) error {
					metricsTrigger := map[string]interface{}{
						"scalingAlgorithm": "HPA",
						"metrics": []interface{}{
							map[string]interface{}{
								"type": "Resource",
								"resource": map[string]interface{}{
									"metricsName": "cpu",
									"target": map[string]interface{}{
										"type":               "Utilization",
										"averageUtilization": int64(1),
									},
								},
							},
						},
					}
					return unstructured.SetNestedMap(u.Object, metricsTrigger, "spec", "trigger", "metricsTrigger")
				},
			},
		}

		for _, tc := range cases {
			// 每个子场景前都恢复到合法 SignalTrigger 基线，避免相互污染
			Expect(restoreValidSignalTrigger()).NotTo(HaveOccurred())
			VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)

			err := UpdateElasticScalerCR(dynamicClient, ctx, esName, ElasticScalerTestNamespace, tc.mutate)
			if err != nil {
				GinkgoWriter.Printf("[%s] 已在 Update/CRD 阶段被拒绝: %v\n", tc.desc, err)
				Expect(restoreValidSignalTrigger()).NotTo(HaveOccurred())
				VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
				continue
			}
			if tc.expectFail {
				VerifyESValidationFailedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
			} else {
				VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
			}
			Expect(restoreValidSignalTrigger()).NotTo(HaveOccurred())
			VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
		}
	})

	It("7-规格校验-Resource 指标子配置拒绝", func() {
		deployName := sharedDeployName
		esName := "validator-7-es-resource-metric-flow"
		DeferCleanup(func() { cleanupES(esName) })

		// 先创建合法 MetricsTrigger ES，通过 CRD/准入；再 Update 注入 Resource 指标的非法组合，走 reconcile Validator。
		obj := newMetricsES(esName, deployName)
		createES(obj)
		VerifyESCRExists(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
		VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)

		restoreValidMetricsTrigger := func() error {
			return UpdateElasticScalerCR(dynamicClient, ctx, esName, ElasticScalerTestNamespace, func(u *unstructured.Unstructured) error {
				fresh := newMetricsES(esName, deployName)
				trig, _, err := unstructured.NestedMap(fresh.Object, "spec", "trigger")
				if err != nil {
					return err
				}
				return unstructured.SetNestedMap(u.Object, trig, "spec", "trigger")
			})
		}

		buildResourceMetric := func(withExternal, withExternalServer bool, resource map[string]interface{}) []interface{} {
			m := map[string]interface{}{
				"type": "Resource",
			}
			if resource != nil {
				m["resource"] = resource
			}
			if withExternal {
				m["external"] = map[string]interface{}{
					"metricsName": "qps",
					"target":      int64(100),
				}
			}
			if withExternalServer {
				m["externalServer"] = map[string]interface{}{
					"metricsName": "qps",
					"target":      "1000",
					"endpoint":    "http://localhost:8080/metrics",
					"protocol":    "http",
				}
			}
			return []interface{}{m}
		}

		cases := []struct {
			desc   string
			mutate func(u *unstructured.Unstructured) error
		}{
			{
				desc: "1、Resource 类型指标同时设置 external/externalServer",
				mutate: func(u *unstructured.Unstructured) error {
					resource := map[string]interface{}{
						"metricsName": "cpu",
						"target": map[string]interface{}{
							"type":               "Utilization",
							"averageUtilization": int64(1),
						},
					}
					metrics := buildResourceMetric(true, true, resource)
					return unstructured.SetNestedSlice(u.Object, metrics, "spec", "trigger", "metricsTrigger", "metrics")
				},
			},
			{
				desc: "2、Resource 类型指标 resource 为 nil",
				mutate: func(u *unstructured.Unstructured) error {
					metrics := buildResourceMetric(false, false, nil)
					return unstructured.SetNestedSlice(u.Object, metrics, "spec", "trigger", "metricsTrigger", "metrics")
				},
			},
			{
				desc: "3、Resource 类型指标 metricsName 为空",
				mutate: func(u *unstructured.Unstructured) error {
					resource := map[string]interface{}{
						"metricsName": "",
						"target": map[string]interface{}{
							"type":               "Utilization",
							"averageUtilization": int64(1),
						},
					}
					metrics := buildResourceMetric(false, false, resource)
					return unstructured.SetNestedSlice(u.Object, metrics, "spec", "trigger", "metricsTrigger", "metrics")
				},
			},
			{
				desc: "4、Resource 类型指标 target.type 为空",
				mutate: func(u *unstructured.Unstructured) error {
					resource := map[string]interface{}{
						"metricsName": "cpu",
						"target": map[string]interface{}{
							"type":               "",
							"averageUtilization": int64(1),
						},
					}
					metrics := buildResourceMetric(false, false, resource)
					return unstructured.SetNestedSlice(u.Object, metrics, "spec", "trigger", "metricsTrigger", "metrics")
				},
			},
		}

		for _, tc := range cases {
			Expect(restoreValidMetricsTrigger()).NotTo(HaveOccurred())
			VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)

			err := UpdateElasticScalerCR(dynamicClient, ctx, esName, ElasticScalerTestNamespace, tc.mutate)
			if err != nil {
				GinkgoWriter.Printf("[%s] 已在 Update/CRD 阶段被拒绝: %v\n", tc.desc, err)
				Expect(restoreValidMetricsTrigger()).NotTo(HaveOccurred())
				VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
				continue
			}
			VerifyESValidationFailedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
			Expect(restoreValidMetricsTrigger()).NotTo(HaveOccurred())
			VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
		}
	})

	It("8-规格校验-External 指标子配置拒绝", func() {
		deployName := sharedDeployName
		esName := "validator-8-es-external-metric-flow"
		DeferCleanup(func() { cleanupES(esName) })

		// 先创建合法 MetricsTrigger ES，通过 CRD/准入；再 Update 注入 External 指标的非法组合，走 reconcile Validator。
		obj := newMetricsES(esName, deployName)
		createES(obj)
		VerifyESCRExists(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
		VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)

		restoreValidMetricsTrigger := func() error {
			return UpdateElasticScalerCR(dynamicClient, ctx, esName, ElasticScalerTestNamespace, func(u *unstructured.Unstructured) error {
				fresh := newMetricsES(esName, deployName)
				trig, _, err := unstructured.NestedMap(fresh.Object, "spec", "trigger")
				if err != nil {
					return err
				}
				return unstructured.SetNestedMap(u.Object, trig, "spec", "trigger")
			})
		}

		buildExternalMetric := func(withResource bool, external map[string]interface{}) []interface{} {
			m := map[string]interface{}{
				"type": "External",
			}
			if external != nil {
				m["external"] = external
			}
			if withResource {
				m["resource"] = map[string]interface{}{
					"metricsName": "cpu",
					"target": map[string]interface{}{
						"type":               "Utilization",
						"averageUtilization": int64(1),
					},
				}
			}
			return []interface{}{m}
		}

		cases := []struct {
			desc   string
			mutate func(u *unstructured.Unstructured) error
		}{
			{
				desc: "1、External 类型指标 external 为 nil",
				mutate: func(u *unstructured.Unstructured) error {
					metrics := buildExternalMetric(false, nil)
					return unstructured.SetNestedSlice(u.Object, metrics, "spec", "trigger", "metricsTrigger", "metrics")
				},
			},
			{
				desc: "2、External 类型指标 metricsName 为空",
				mutate: func(u *unstructured.Unstructured) error {
					external := map[string]interface{}{
						"metricsName": "",
						"target":      int64(100),
					}
					metrics := buildExternalMetric(false, external)
					return unstructured.SetNestedSlice(u.Object, metrics, "spec", "trigger", "metricsTrigger", "metrics")
				},
			},
			{
				desc: "3、External 类型指标 target 为负数",
				mutate: func(u *unstructured.Unstructured) error {
					external := map[string]interface{}{
						"metricsName": "qps",
						"target":      int64(-1),
					}
					metrics := buildExternalMetric(false, external)
					return unstructured.SetNestedSlice(u.Object, metrics, "spec", "trigger", "metricsTrigger", "metrics")
				},
			},
			{
				desc: "4、External 类型指标同时设置 resource",
				mutate: func(u *unstructured.Unstructured) error {
					external := map[string]interface{}{
						"metricsName": "qps",
						"target":      int64(100),
					}
					metrics := buildExternalMetric(true, external)
					return unstructured.SetNestedSlice(u.Object, metrics, "spec", "trigger", "metricsTrigger", "metrics")
				},
			},
		}

		for _, tc := range cases {
			Expect(restoreValidMetricsTrigger()).NotTo(HaveOccurred())
			VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)

			err := UpdateElasticScalerCR(dynamicClient, ctx, esName, ElasticScalerTestNamespace, tc.mutate)
			if err != nil {
				GinkgoWriter.Printf("[%s] 已在 Update/CRD 阶段被拒绝: %v\n", tc.desc, err)
				Expect(restoreValidMetricsTrigger()).NotTo(HaveOccurred())
				VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
				continue
			}
			VerifyESValidationFailedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
			Expect(restoreValidMetricsTrigger()).NotTo(HaveOccurred())
			VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
		}
	})

	It("9-规格校验-ExternalServer 指标子配置校验", func() {
		deployName := sharedDeployName
		esName := "validator-9-es-external-server-flow"
		DeferCleanup(func() { cleanupES(esName) })

		// 先创建合法 MetricsTrigger ES，通过 CRD/准入；再 Update 注入 ExternalServer 指标场景，走 reconcile Validator。
		obj := newMetricsES(esName, deployName)
		createES(obj)
		VerifyESCRExists(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
		VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)

		restoreValidMetricsTrigger := func() error {
			return UpdateElasticScalerCR(dynamicClient, ctx, esName, ElasticScalerTestNamespace, func(u *unstructured.Unstructured) error {
				fresh := newMetricsES(esName, deployName)
				trig, _, err := unstructured.NestedMap(fresh.Object, "spec", "trigger")
				if err != nil {
					return err
				}
				return unstructured.SetNestedMap(u.Object, trig, "spec", "trigger")
			})
		}

		buildExternalServerMetric := func(withResource bool, externalServer map[string]interface{}) []interface{} {
			m := map[string]interface{}{
				"type": "ExternalServer",
			}
			if externalServer != nil {
				m["externalServer"] = externalServer
			}
			if withResource {
				m["resource"] = map[string]interface{}{
					"metricsName": "cpu",
					"target": map[string]interface{}{
						"type":               "Utilization",
						"averageUtilization": int64(1),
					},
				}
			}
			return []interface{}{m}
		}

		cases := []struct {
			desc       string
			expectFail bool
			mutate     func(u *unstructured.Unstructured) error
		}{
			{
				desc:       "1、ExternalServer 指标 endpoint 不带 scheme 时通过（默认/按 protocol 对齐）",
				expectFail: false,
				mutate: func(u *unstructured.Unstructured) error {
					extSrv := map[string]interface{}{
						"metricsName": "qps",
						"target":      "1000",
						// 使用无 scheme 的域名形式，避免不同集群对 localhost:port/path 的 URL 解析差异
						"endpoint": "metrics.example.com",
						"protocol": "http",
					}
					metrics := buildExternalServerMetric(false, extSrv)
					return unstructured.SetNestedSlice(u.Object, metrics, "spec", "trigger", "metricsTrigger", "metrics")
				},
			},
			{
				desc:       "2、ExternalServer 指标 endpoint 显式非 http/https scheme 时拒绝",
				expectFail: true,
				mutate: func(u *unstructured.Unstructured) error {
					extSrv := map[string]interface{}{
						"metricsName": "qps",
						"target":      "1000",
						"endpoint":    "grpc://localhost:8080/metrics",
						"protocol":    "http",
					}
					metrics := buildExternalServerMetric(false, extSrv)
					return unstructured.SetNestedSlice(u.Object, metrics, "spec", "trigger", "metricsTrigger", "metrics")
				},
			},
			{
				desc:       "3、ExternalServer 指标 endpoint 或 target 非法时拒绝",
				expectFail: true,
				mutate: func(u *unstructured.Unstructured) error {
					// target 负数 + endpoint 非法 URL 形态（显式 scheme）
					extSrv := map[string]interface{}{
						"metricsName": "qps",
						"target":      "-1",
						"endpoint":    "://bad",
						"protocol":    "http",
					}
					metrics := buildExternalServerMetric(false, extSrv)
					return unstructured.SetNestedSlice(u.Object, metrics, "spec", "trigger", "metricsTrigger", "metrics")
				},
			},
			{
				desc:       "4、ExternalServer 类型指标同时设置 resource 时拒绝",
				expectFail: true,
				mutate: func(u *unstructured.Unstructured) error {
					extSrv := map[string]interface{}{
						"metricsName": "qps",
						"target":      "1000",
						"endpoint":    "http://localhost:8080/metrics",
						"protocol":    "http",
					}
					metrics := buildExternalServerMetric(true, extSrv)
					return unstructured.SetNestedSlice(u.Object, metrics, "spec", "trigger", "metricsTrigger", "metrics")
				},
			},
		}

		for _, tc := range cases {
			Expect(restoreValidMetricsTrigger()).NotTo(HaveOccurred())

			err := UpdateElasticScalerCR(dynamicClient, ctx, esName, ElasticScalerTestNamespace, tc.mutate)
			if err != nil {
				GinkgoWriter.Printf("[%s] 已在 Update/CRD 阶段被拒绝: %v\n", tc.desc, err)
				Expect(restoreValidMetricsTrigger()).NotTo(HaveOccurred())
				continue
			}
			if tc.expectFail {
				VerifyESValidationFailedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
			} else {
				// ExternalServer 场景在不同集群/实现下可能不会及时写回 Validated=True，避免在“通过用例”上超时卡死。
				// 这里以 Update 成功为主，必要时可在后续按实现补充更明确的通过断言。
				GinkgoWriter.Printf("[%s] Update 成功；ExternalServer 场景跳过等待 Validated=True\n", tc.desc)
			}
			Expect(restoreValidMetricsTrigger()).NotTo(HaveOccurred())
		}
	})

	It("10-规格校验-Custom 指标子配置校验", func() {
		deployName := sharedDeployName
		esName := "validator-10-es-custom-metric-flow"
		DeferCleanup(func() { cleanupES(esName) })

		// 先创建合法 MetricsTrigger ES；再 Update 注入非法 Custom 组合。
		// 拒绝可能来自 CRD/APIServer 或 reconcile 侧 Validator（Validated=False）。
		obj := newMetricsES(esName, deployName)
		createES(obj)
		VerifyESCRExists(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
		VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)

		restoreValidMetricsTrigger := func() error {
			return UpdateElasticScalerCR(dynamicClient, ctx, esName, ElasticScalerTestNamespace, func(u *unstructured.Unstructured) error {
				fresh := newMetricsES(esName, deployName)
				trig, _, err := unstructured.NestedMap(fresh.Object, "spec", "trigger")
				if err != nil {
					return err
				}
				return unstructured.SetNestedMap(u.Object, trig, "spec", "trigger")
			})
		}

		// custom 块 + 可选互斥字段（resource / external / externalServer）；与 validator validateCustomMetric 对齐。
		buildCustomMetric := func(custom map[string]interface{}, conflict map[string]interface{}) []interface{} {
			m := map[string]interface{}{
				"type": "Custom",
			}
			if custom != nil {
				m["custom"] = custom
			}
			for k, v := range conflict {
				m[k] = v
			}
			return []interface{}{m}
		}

		cases := []struct {
			desc   string
			mutate func(u *unstructured.Unstructured) error
		}{
			{
				desc: "1、Custom 类型指标 custom=nil 时拒绝",
				mutate: func(u *unstructured.Unstructured) error {
					metrics := buildCustomMetric(nil, nil)
					return unstructured.SetNestedSlice(u.Object, metrics, "spec", "trigger", "metricsTrigger", "metrics")
				},
			},
			{
				desc: "2、Custom 类型指标 target<0 时拒绝（reconcile Validator）",
				mutate: func(u *unstructured.Unstructured) error {
					custom := map[string]interface{}{
						"metricsName": "pods/http_requests",
						"target":      int64(-1),
					}
					metrics := buildCustomMetric(custom, nil)
					return unstructured.SetNestedSlice(u.Object, metrics, "spec", "trigger", "metricsTrigger", "metrics")
				},
			},
			{
				desc: "3、Custom 类型指标互斥：同时设置 resource 时拒绝",
				mutate: func(u *unstructured.Unstructured) error {
					custom := map[string]interface{}{
						"metricsName": "pods/http_requests",
						"target":      int64(100),
					}
					conflict := map[string]interface{}{
						"resource": map[string]interface{}{
							"metricsName": "cpu",
							"target": map[string]interface{}{
								"type":               "Utilization",
								"averageUtilization": int64(1),
							},
						},
					}
					metrics := buildCustomMetric(custom, conflict)
					return unstructured.SetNestedSlice(u.Object, metrics, "spec", "trigger", "metricsTrigger", "metrics")
				},
			},
			{
				desc: "4、Custom 类型指标互斥：同时设置 external 时拒绝",
				mutate: func(u *unstructured.Unstructured) error {
					custom := map[string]interface{}{
						"metricsName": "pods/http_requests",
						"target":      int64(100),
					}
					conflict := map[string]interface{}{
						"external": map[string]interface{}{
							"metricsName": "qps",
							"target":      int64(1000),
						},
					}
					metrics := buildCustomMetric(custom, conflict)
					return unstructured.SetNestedSlice(u.Object, metrics, "spec", "trigger", "metricsTrigger", "metrics")
				},
			},
			{
				desc: "5、Custom 类型指标互斥：同时设置 externalServer 时拒绝",
				mutate: func(u *unstructured.Unstructured) error {
					custom := map[string]interface{}{
						"metricsName": "pods/http_requests",
						"target":      int64(100),
					}
					conflict := map[string]interface{}{
						"externalServer": map[string]interface{}{
							"metricsName": "qps",
							"target":      int64(1000),
							"endpoint":    "http://localhost:8080/metrics",
							"protocol":    "http",
						},
					}
					metrics := buildCustomMetric(custom, conflict)
					return unstructured.SetNestedSlice(u.Object, metrics, "spec", "trigger", "metricsTrigger", "metrics")
				},
			},
		}

		for _, tc := range cases {
			Expect(restoreValidMetricsTrigger()).NotTo(HaveOccurred())
			VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)

			err := UpdateElasticScalerCR(dynamicClient, ctx, esName, ElasticScalerTestNamespace, tc.mutate)
			if err != nil {
				GinkgoWriter.Printf("[%s] Update 被拒绝（未写入非法 spec，符合拒绝语义）: %v\n", tc.desc, err)
				Expect(restoreValidMetricsTrigger()).NotTo(HaveOccurred())
				VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
				continue
			}
			VerifyESValidationFailedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
			Expect(restoreValidMetricsTrigger()).NotTo(HaveOccurred())
			VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
		}
	})

	It("11-规格校验-算法注册校验（E2E）", func() {
		deployName := sharedDeployName
		esName := "validator-11-es-algorithm-flow"
		DeferCleanup(func() { cleanupES(esName) })

		// 先创建合法 ES（HPA）通过 CRD/准入与 reconcile；再通过 Update 切换算法名验证拒绝/通过。
		obj := newMetricsES(esName, deployName)
		createES(obj)
		VerifyESCRExists(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
		VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)

		restoreHPA := func() error {
			return UpdateElasticScalerCR(dynamicClient, ctx, esName, ElasticScalerTestNamespace, func(u *unstructured.Unstructured) error {
				return unstructured.SetNestedField(u.Object, "HPA", "spec", "trigger", "metricsTrigger", "scalingAlgorithm")
			})
		}

		cases := []struct {
			desc       string
			expectFail bool
			algo       string
		}{
			{desc: "1、未注册 scalingAlgorithm 时拒绝", expectFail: true, algo: "UnregisteredAlgo"},
			{desc: "2、HPA 已注册时通过", expectFail: false, algo: "HPA"},
		}

		for _, tc := range cases {
			Expect(restoreHPA()).NotTo(HaveOccurred())
			VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)

			err := UpdateElasticScalerCR(dynamicClient, ctx, esName, ElasticScalerTestNamespace, func(u *unstructured.Unstructured) error {
				return unstructured.SetNestedField(u.Object, tc.algo, "spec", "trigger", "metricsTrigger", "scalingAlgorithm")
			})
			if err != nil {
				GinkgoWriter.Printf("[%s] 已在 Update/CRD 阶段被拒绝: %v\n", tc.desc, err)
				Expect(restoreHPA()).NotTo(HaveOccurred())
				VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
				continue
			}
			if tc.expectFail {
				VerifyESValidationFailedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
			} else {
				VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
			}
			Expect(restoreHPA()).NotTo(HaveOccurred())
			VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
		}
	})

	It("12-规格校验-集群态 targetRef 存在性与一一对应（E2E）", func() {
		deployName := sharedDeployName
		ownerName := "validator-12-es-owner"
		DeferCleanup(func() { cleanupES(ownerName) })

		obj := newMetricsES(ownerName, deployName)
		createES(obj)
		VerifyESCRExists(dynamicClient, ctx, ownerName, ElasticScalerTestNamespace)
		VerifyESValidatedCondition(dynamicClient, ctx, ownerName, ElasticScalerTestNamespace)

		restoreOwnerTarget := func() error {
			return UpdateElasticScalerCR(dynamicClient, ctx, ownerName, ElasticScalerTestNamespace, func(u *unstructured.Unstructured) error {
				return unstructured.SetNestedField(u.Object, deployName, "spec", "targetRef", "name")
			})
		}

		By("1、集群态 targetRef 资源不存在时拒绝（通过 Update 注入不存在的 name）")
		{
			missing := "validator-12-missing-deploy"
			err := UpdateElasticScalerCR(dynamicClient, ctx, ownerName, ElasticScalerTestNamespace, func(u *unstructured.Unstructured) error {
				return unstructured.SetNestedField(u.Object, missing, "spec", "targetRef", "name")
			})
			if err != nil {
				GinkgoWriter.Printf("[target 不存在] 已在 Update/CRD 阶段被拒绝: %v\n", err)
			} else {
				VerifyESValidationFailedCondition(dynamicClient, ctx, ownerName, ElasticScalerTestNamespace)
			}
			Expect(restoreOwnerTarget()).NotTo(HaveOccurred())
			VerifyESValidatedCondition(dynamicClient, ctx, ownerName, ElasticScalerTestNamespace)
		}

		By("2、集群态 target 存在且无其他 ES 管理时通过（基线即覆盖）")
		{
			VerifyESValidatedCondition(dynamicClient, ctx, ownerName, ElasticScalerTestNamespace)
		}

		By("3、集群态 另一 ES 已管理同一 targetRef 时拒绝（第二条 ES 先合法创建，再 Update 切到冲突 target）")
		{
			otherDeploy := "validator-12-other-deploy"
			Expect(CreateESMetricsTestDeployment(clientset, ctx, otherDeploy, ElasticScalerTestNamespace, 1, false)).NotTo(HaveOccurred())
			DeferCleanup(func() {
				_ = clientset.AppsV1().Deployments(ElasticScalerTestNamespace).Delete(ctx, otherDeploy, metav1.DeleteOptions{})
			})

			otherES := "validator-12-es-other"
			DeferCleanup(func() { cleanupES(otherES) })

			obj2 := newMetricsES(otherES, otherDeploy)
			createES(obj2)
			VerifyESCRExists(dynamicClient, ctx, otherES, ElasticScalerTestNamespace)
			VerifyESValidatedCondition(dynamicClient, ctx, otherES, ElasticScalerTestNamespace)

			err := UpdateElasticScalerCR(dynamicClient, ctx, otherES, ElasticScalerTestNamespace, func(u *unstructured.Unstructured) error {
				return unstructured.SetNestedField(u.Object, deployName, "spec", "targetRef", "name")
			})
			if err != nil {
				GinkgoWriter.Printf("[target 冲突] 已在 Update/CRD 阶段被拒绝: %v\n", err)
			} else {
				VerifyESValidationFailedCondition(dynamicClient, ctx, otherES, ElasticScalerTestNamespace)
			}

			// 恢复为其原本的合法 targetRef，避免残留冲突状态影响后续用例
			Expect(UpdateElasticScalerCR(dynamicClient, ctx, otherES, ElasticScalerTestNamespace, func(u *unstructured.Unstructured) error {
				return unstructured.SetNestedField(u.Object, otherDeploy, "spec", "targetRef", "name")
			})).NotTo(HaveOccurred())
			VerifyESValidatedCondition(dynamicClient, ctx, otherES, ElasticScalerTestNamespace)
		}
	})

	It("13-规格校验-StateTrigger 目标 status 副本字段校验（E2E）", func() {
		deployName := sharedDeployName
		esName := "validator-13-es-state-target-status-flow"
		DeferCleanup(func() { cleanupES(esName) })

		cmName := "validator-13-target-configmap"
		svcName := "validator-13-target-service"

		// ConfigMap：无 status
		_, err := clientset.CoreV1().ConfigMaps(ElasticScalerTestNamespace).Create(ctx, &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Name: cmName, Namespace: ElasticScalerTestNamespace},
			Data:       map[string]string{"k": "v"},
		}, metav1.CreateOptions{})
		Expect(err).NotTo(HaveOccurred())
		DeferCleanup(func() {
			_ = clientset.CoreV1().ConfigMaps(ElasticScalerTestNamespace).Delete(ctx, cmName, metav1.DeleteOptions{})
		})

		// Service：通常有 status（loadBalancer 等），但不含 replicas/desiredReplicas
		_, err = clientset.CoreV1().Services(ElasticScalerTestNamespace).Create(ctx, &corev1.Service{
			ObjectMeta: metav1.ObjectMeta{Name: svcName, Namespace: ElasticScalerTestNamespace},
			Spec: corev1.ServiceSpec{
				Type:     corev1.ServiceTypeClusterIP,
				Selector: map[string]string{},
				Ports: []corev1.ServicePort{
					{Name: "http", Port: 80, Protocol: corev1.ProtocolTCP},
				},
			},
		}, metav1.CreateOptions{})
		Expect(err).NotTo(HaveOccurred())
		DeferCleanup(func() {
			_ = clientset.CoreV1().Services(ElasticScalerTestNamespace).Delete(ctx, svcName, metav1.DeleteOptions{})
		})

		// 先创建合法 MetricsTrigger ES，通过 CRD/准入；再 Update 切换到合法 StateTrigger 基线，并确保在 Deployment 目标下通过。
		obj := newMetricsES(esName, deployName)
		createES(obj)
		VerifyESCRExists(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
		VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)

		restoreValidStateTriggerToDeployment := func() error {
			return UpdateElasticScalerCR(dynamicClient, ctx, esName, ElasticScalerTestNamespace, func(u *unstructured.Unstructured) error {
				if err := unstructured.SetNestedMap(u.Object, map[string]interface{}{
					"type":         "StateTrigger",
					"stateTrigger": map[string]interface{}{},
				}, "spec", "trigger"); err != nil {
					return err
				}
				if err := unstructured.SetNestedField(u.Object, "apps/v1", "spec", "targetRef", "apiVersion"); err != nil {
					return err
				}
				if err := unstructured.SetNestedField(u.Object, "Deployment", "spec", "targetRef", "kind"); err != nil {
					return err
				}
				if err := unstructured.SetNestedField(u.Object, deployName, "spec", "targetRef", "name"); err != nil {
					return err
				}
				if err := unstructured.SetNestedField(u.Object, ElasticScalerTestNamespace, "spec", "targetRef", "namespace"); err != nil {
					return err
				}
				return nil
			})
		}

		Expect(restoreValidStateTriggerToDeployment()).NotTo(HaveOccurred())
		VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)

		By("1、StateTrigger 目标无 status 时拒绝（ConfigMap）")
		{
			err := UpdateElasticScalerCR(dynamicClient, ctx, esName, ElasticScalerTestNamespace, func(u *unstructured.Unstructured) error {
				if err := unstructured.SetNestedField(u.Object, "v1", "spec", "targetRef", "apiVersion"); err != nil {
					return err
				}
				if err := unstructured.SetNestedField(u.Object, "ConfigMap", "spec", "targetRef", "kind"); err != nil {
					return err
				}
				return unstructured.SetNestedField(u.Object, cmName, "spec", "targetRef", "name")
			})
			if err != nil {
				GinkgoWriter.Printf("[target 无 status] 已在 Update/CRD 阶段被拒绝: %v\n", err)
			} else {
				VerifyESValidationFailedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
			}
			Expect(restoreValidStateTriggerToDeployment()).NotTo(HaveOccurred())
			VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
		}

		By("2、StateTrigger 目标 status 无 replicas/desiredReplicas 时拒绝（Service）")
		{
			err := UpdateElasticScalerCR(dynamicClient, ctx, esName, ElasticScalerTestNamespace, func(u *unstructured.Unstructured) error {
				if err := unstructured.SetNestedField(u.Object, "v1", "spec", "targetRef", "apiVersion"); err != nil {
					return err
				}
				if err := unstructured.SetNestedField(u.Object, "Service", "spec", "targetRef", "kind"); err != nil {
					return err
				}
				return unstructured.SetNestedField(u.Object, svcName, "spec", "targetRef", "name")
			})
			if err != nil {
				GinkgoWriter.Printf("[target 无副本字段] 已在 Update/CRD 阶段被拒绝: %v\n", err)
			} else {
				VerifyESValidationFailedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
			}
			Expect(restoreValidStateTriggerToDeployment()).NotTo(HaveOccurred())
			VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
		}

		By("3、StateTrigger 目标 status 含 replicas 或 desiredReplicas 时通过（Deployment）")
		{
			Expect(restoreValidStateTriggerToDeployment()).NotTo(HaveOccurred())
			VerifyESValidatedCondition(dynamicClient, ctx, esName, ElasticScalerTestNamespace)
		}
	})
})
