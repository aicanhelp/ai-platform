package elastic_scaler

import (
	"fmt"
	"strings"
	"time"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/elastic-scaler/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

// RSG测试用例
const rsgMetricAnnotationKey = "autoscaling.rsg.metric.values"

var _ = Describe("resourcescalinggroup validation", Ordered, ContinueOnFailure, Label("resourcescalinggroup", "with-workload-cluster"), func() {
	BeforeAll(func() {
		Expect(EnsureNamespace(clientset, ctx, RSGTestNamespace)).NotTo(HaveOccurred())
		Expect(EnsureLeaderWorkerSetController(clientset, ctx)).NotTo(HaveOccurred())
		Expect(HelmUninstall(RSGHelmReleaseName, RSGTestNamespace)).NotTo(HaveOccurred())
		Expect(InstallResourceScalingGroupController()).NotTo(HaveOccurred())
		VerifyHelmReleaseStatus(RSGHelmReleaseName, RSGTestNamespace)
		VerifyResourceScalingGroupControllerReady(clientset, ctx)
	})

	BeforeEach(func() {
		CleanupRSGWebhookTestResources(dynamicClient, ctx)
		CleanupRSGBusinessResources(clientset, ctx)
		CleanupRSGLWSResources(clientset, dynamicClient, ctx, RSGTestNamespace)
		WaitForNoRSGTestPods(clientset, ctx, RSGTestNamespace)
		VerifyResourceScalingGroupControllerReady(clientset, ctx)
	})

	// 用例1：targetResources 非空校验
	It("1-targetResources-must-not-be-empty", Label("webhook"), func() {
		name := "rsg-webhook-01"
		obj := NewGroupReplicationRSG(name)
		Expect(unstructured.SetNestedSlice(obj.Object, []any{}, "spec", "targetResources")).To(Succeed())

		err := CreateRSG(dynamicClient, ctx, obj)
		Expect(err).To(HaveOccurred())
		AssertErrorContains(err, "should have at least 1 items")
		VerifyRSGWasNotCreated(dynamicClient, ctx, name)
	})

	// 用例2：TargetResource 名称唯一性
	It("2-targetResource-name-must-be-unique", Label("webhook"), func() {
		name := "rsg-webhook-02"
		obj := NewResourceScalingGroup(name, map[string]any{
			"targetResources": []any{
				NewTargetResource("shared", "apps/v1", "Deployment", name+"-a", ""),
				NewTargetResource("shared", "apps/v1", "Deployment", name+"-b", ""),
			},
			"scalingStrategy": NewGroupReplicationStrategy(1),
		})

		err := CreateRSG(dynamicClient, ctx, obj)
		Expect(err).To(HaveOccurred())
		AssertErrorContains(err, "Duplicate value")
		VerifyRSGWasNotCreated(dynamicClient, ctx, name)
	})

	// 用例3：GroupReplication 必须配置 groupConfig
	It("3-groupReplication-requires-groupConfig", Label("webhook"), func() {
		name := "rsg-webhook-03"
		obj := NewResourceScalingGroup(name, map[string]any{
			"targetResources": []any{
				NewTargetResource("primary", "apps/v1", "Deployment", name+"-primary", ""),
			},
			"scalingStrategy": map[string]any{
				"type": "GroupReplication",
			},
		})

		err := CreateRSG(dynamicClient, ctx, obj)
		Expect(err).To(HaveOccurred())
		AssertErrorContains(err, "spec.scalingStrategy.groupConfig: Required value")
		VerifyRSGWasNotCreated(dynamicClient, ctx, name)
	})

	// 用例4：InplaceScaling 必须配置 leaderFollowerConfig
	It("4-inplaceScaling-requires-leaderFollowerConfig", Label("webhook"), func() {
		name := "rsg-webhook-04"
		obj := NewResourceScalingGroup(name, map[string]any{
			"targetResources": []any{
				NewTargetResource("leader", "apps/v1", "Deployment", name+"-leader", ""),
				NewTargetResource("follower", "apps/v1", "Deployment", name+"-follower", ""),
			},
			"scalingStrategy": map[string]any{
				"type": "InplaceScaling",
			},
		})

		err := CreateRSG(dynamicClient, ctx, obj)
		Expect(err).To(HaveOccurred())
		AssertErrorContains(err, "spec.scalingStrategy.leaderFollowerConfig: Required value")
		VerifyRSGWasNotCreated(dynamicClient, ctx, name)
	})

	// 用例5：不支持的策略类型在准入阶段拒绝
	It("5-unsupported-strategy-rejected-on-admission", Label("webhook"), func() {
		name := "rsg-webhook-05"
		obj := NewResourceScalingGroup(name, map[string]any{
			"targetResources": []any{
				NewTargetResource("primary", "apps/v1", "Deployment", name+"-primary", ""),
			},
			"scalingStrategy": map[string]any{
				"type": "Cusotm",
			},
		})

		err := CreateRSG(dynamicClient, ctx, obj)
		Expect(err).To(HaveOccurred())
		AssertErrorContains(err, "Unsupported value")
		VerifyRSGWasNotCreated(dynamicClient, ctx, name)
	})

	// 用例6：target workload 不能被多个 GroupReplication RSG 同时引用
	It("6-targetResource-must-be-unique-across-rsgs", Label("webhook"), func() {
		existing := NewResourceScalingGroup("rsg-webhook-existing", map[string]any{
			"targetResources": []any{
				NewTargetResource("primary", "apps/v1", "Deployment", "shared-target", ""),
			},
			"scalingStrategy": NewGroupReplicationStrategy(1),
		})
		Expect(CreateRSG(dynamicClient, ctx, existing)).NotTo(HaveOccurred())

		name := "rsg-webhook-06"
		obj := NewResourceScalingGroup(name, map[string]any{
			"targetResources": []any{
				NewTargetResource("secondary", "apps/v1", "Deployment", "shared-target", ""),
			},
			"scalingStrategy": NewGroupReplicationStrategy(2),
		})

		err := CreateRSGExpectAdmissionFailureEventually(dynamicClient, ctx, obj, "multiple group-replication ResourceScalingGroups")
		Expect(err).To(HaveOccurred())
		AssertErrorContains(err, "multiple group-replication ResourceScalingGroups")
		VerifyRSGWasNotCreated(dynamicClient, ctx, name)
	})

	// 用例7：leader 必须引用已定义 target
	It("7-leader-must-reference-existing-target", Label("webhook"), func() {
		name := "rsg-webhook-07"
		obj := NewResourceScalingGroup(name, map[string]any{
			"targetResources": []any{
				NewTargetResource("follower", "apps/v1", "Deployment", name+"-follower", ""),
			},
			"scalingStrategy": map[string]any{
				"type": "InplaceScaling",
				"leaderFollowerConfig": map[string]any{
					"leader": map[string]any{
						"resourceName": "missing",
						"replicas":     int64(1),
					},
					"follower": []any{
						map[string]any{
							"resourceName": "follower",
							"followRatio":  int64(1),
						},
					},
				},
			},
		})

		err := CreateRSG(dynamicClient, ctx, obj)
		Expect(err).To(HaveOccurred())
		AssertErrorContains(err, "must reference targetResources")
		VerifyRSGWasNotCreated(dynamicClient, ctx, name)
	})

	// 用例8：Follower followRatio 必须 >= 1
	It("8-follower-followRatio-must-be-greater-or-equal-1", Label("webhook"), func() {
		name := "rsg-webhook-08"
		obj := NewResourceScalingGroup(name, map[string]any{
			"targetResources": []any{
				NewTargetResource("leader", "apps/v1", "Deployment", name+"-leader", ""),
				NewTargetResource("follower", "apps/v1", "Deployment", name+"-follower", ""),
			},
			"scalingStrategy": map[string]any{
				"type": "InplaceScaling",
				"leaderFollowerConfig": map[string]any{
					"leader": map[string]any{
						"resourceName": "leader",
						"replicas":     int64(1),
					},
					"follower": []any{
						map[string]any{
							"resourceName": "follower",
							"followRatio":  int64(0),
						},
					},
				},
			},
		})

		err := CreateRSG(dynamicClient, ctx, obj)
		Expect(err).To(HaveOccurred())
		AssertErrorContains(err, "followRatio in body should be greater than or equal to 1")
		VerifyRSGWasNotCreated(dynamicClient, ctx, name)
	})

	// 用例9：Follower 不能指向 Leader
	It("9-follower-must-not-reference-leader", Label("webhook"), func() {
		name := "rsg-webhook-09"
		obj := NewResourceScalingGroup(name, map[string]any{
			"targetResources": []any{
				NewTargetResource("leader", "apps/v1", "Deployment", name+"-leader", ""),
				NewTargetResource("follower", "apps/v1", "Deployment", name+"-follower", ""),
			},
			"scalingStrategy": map[string]any{
				"type": "InplaceScaling",
				"leaderFollowerConfig": map[string]any{
					"leader": map[string]any{
						"resourceName": "leader",
						"replicas":     int64(1),
					},
					"follower": []any{
						map[string]any{
							"resourceName": "leader",
							"followRatio":  int64(1),
						},
					},
				},
			},
		})

		err := CreateRSG(dynamicClient, ctx, obj)
		Expect(err).To(HaveOccurred())
		AssertErrorContains(err, "must not reference leader")
		VerifyRSGWasNotCreated(dynamicClient, ctx, name)
	})

	// 用例10：首轮创建快照和 ConfigMap
	It("10-create-snapshot-and-configmap-on-first-reconcile", Label("groupreplication"), func() {
		logStart := time.Now()
		rsgName, groupName, appName, targetResources, serviceNames := prepareRSGGroupReplicationTargets(10)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 1, targetResources))).NotTo(HaveOccurred())
		rsgUID := GetRSGUID(dynamicClient, ctx, rsgName)

		VerifyRSGControllerRevisionSnapshot(clientset, ctx, RSGTestNamespace, rsgName+"-prefill-snapshot", rsgName, rsgUID, "prefill", "apps/v1", "Deployment", targetResourceRefName(targetResources[0]), RSGTestNamespace, 1, "")
		VerifyRSGControllerRevisionSnapshot(clientset, ctx, RSGTestNamespace, rsgName+"-decode-snapshot", rsgName, rsgUID, "decode", "apps/v1", "Deployment", targetResourceRefName(targetResources[1]), RSGTestNamespace, 1, "")
		VerifyRSGReplicaConfigMapContent(clientset, ctx, RSGTestNamespace, rsgName+"-replica-config", rsgName, rsgUID, map[string]string{
			"prefill": "1",
			"decode":  "1",
		})
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 1, 1)
		VerifyDeploymentExists(clientset, ctx, RSGTestNamespace, targetResources[0]["resourceRef"].(map[string]any)["name"].(string))
		VerifyDeploymentExists(clientset, ctx, RSGTestNamespace, targetResources[1]["resourceRef"].(map[string]any)["name"].(string))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, serviceNames[0])
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, serviceNames[1])
		VerifyRSGStatusSelectorContains(dynamicClient, ctx, rsgName, []string{
			"app=" + appName,
			RSGWebhookTestLabelKey + "=" + RSGWebhookTestLabelValue,
		})
		VerifyRSGControllerLogsContainForRSG(clientset, ctx, logStart, rsgName, "snapshot created", "replica configmap created")
	})

	// 用例11：按 desired replicas 创建 group1
	It("11-create-group-by-desired-replicas", Label("groupreplication"), func() {
		rsgName, groupName, appName, targetResources, serviceNames := prepareRSGGroupReplicationTargets(11)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 1, targetResources))).NotTo(HaveOccurred())
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 1, 1)

		Expect(updateRSGGroupReplicasWithReadyCheck(rsgName, 2)).NotTo(HaveOccurred())

		VerifyDeploymentExists(clientset, ctx, RSGTestNamespace, groupName+"-"+targetResources[0]["resourceRef"].(map[string]any)["name"].(string)+"-1")
		VerifyDeploymentExists(clientset, ctx, RSGTestNamespace, groupName+"-"+targetResources[1]["resourceRef"].(map[string]any)["name"].(string)+"-1")
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, serviceNames[0]+"-"+groupName+"-1")
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, serviceNames[1]+"-"+groupName+"-1")
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 2, 2)
		VerifyRSGStatusSelectorContains(dynamicClient, ctx, rsgName, []string{
			"app=" + appName,
			RSGWebhookTestLabelKey + "=" + RSGWebhookTestLabelValue,
		})
	})

	// 用例12：删除 RSG 保留业务资源
	It("12-delete-rsg-and-keep-business-resources", Label("groupreplication"), func() {
		rsgName, groupName, _, targetResources, serviceNames := prepareRSGGroupReplicationTargets(12)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 2, targetResources))).NotTo(HaveOccurred())

		prefillClone := groupName + "-" + targetResources[0]["resourceRef"].(map[string]any)["name"].(string) + "-1"
		decodeClone := groupName + "-" + targetResources[1]["resourceRef"].(map[string]any)["name"].(string) + "-1"
		VerifyDeploymentExists(clientset, ctx, RSGTestNamespace, prefillClone)
		VerifyDeploymentExists(clientset, ctx, RSGTestNamespace, decodeClone)
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, serviceNames[0]+"-"+groupName+"-1")
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, serviceNames[1]+"-"+groupName+"-1")

		DeleteRSGAndWait(dynamicClient, ctx, rsgName)

		VerifyDeploymentExists(clientset, ctx, RSGTestNamespace, prefillClone)
		VerifyDeploymentExists(clientset, ctx, RSGTestNamespace, decodeClone)
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, serviceNames[0]+"-"+groupName+"-1")
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, serviceNames[1]+"-"+groupName+"-1")
		VerifyDeploymentHasNoOwnerReferences(clientset, ctx, RSGTestNamespace, prefillClone)
		VerifyDeploymentHasNoOwnerReferences(clientset, ctx, RSGTestNamespace, decodeClone)
		VerifyServiceHasNoOwnerReferences(clientset, ctx, RSGTestNamespace, serviceNames[0]+"-"+groupName+"-1")
		VerifyServiceHasNoOwnerReferences(clientset, ctx, RSGTestNamespace, serviceNames[1]+"-"+groupName+"-1")
		VerifyDeploymentPersists(clientset, ctx, RSGTestNamespace, prefillClone)
		VerifyDeploymentPersists(clientset, ctx, RSGTestNamespace, decodeClone)
		VerifyServicePersists(clientset, ctx, RSGTestNamespace, serviceNames[0]+"-"+groupName+"-1")
		VerifyServicePersists(clientset, ctx, RSGTestNamespace, serviceNames[1]+"-"+groupName+"-1")
		VerifyRSGReplicaConfigMapDeleted(clientset, ctx, RSGTestNamespace, rsgName+"-replica-config")
		VerifyRSGControllerRevisionDeleted(clientset, ctx, RSGTestNamespace, rsgName+"-prefill-snapshot")
		VerifyRSGControllerRevisionDeleted(clientset, ctx, RSGTestNamespace, rsgName+"-decode-snapshot")
	})

	// 用例13：同名 RSG 重建不接管旧资源
	It("13-recreate-same-rsg-name-without-taking-over-old-resources", Label("groupreplication"), func() {
		rsgName, groupName, _, targetResources, _ := prepareRSGGroupReplicationTargets(13)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 2, targetResources))).NotTo(HaveOccurred())

		cloneName := groupName + "-" + targetResources[0]["resourceRef"].(map[string]any)["name"].(string) + "-1"
		VerifyDeploymentExists(clientset, ctx, RSGTestNamespace, cloneName)
		oldUID := GetRSGUID(dynamicClient, ctx, rsgName)
		VerifyDeploymentLabelValue(clientset, ctx, RSGTestNamespace, cloneName, "rsg-uid", oldUID)

		DeleteRSGAndWait(dynamicClient, ctx, rsgName)

		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 2, targetResources))).NotTo(HaveOccurred())
		newUID := GetRSGUID(dynamicClient, ctx, rsgName)
		Expect(newUID).NotTo(Equal(oldUID))

		VerifyDeploymentLabelValue(clientset, ctx, RSGTestNamespace, cloneName, "rsg-uid", oldUID)
		VerifyDeploymentLabelValueNotEqual(clientset, ctx, RSGTestNamespace, cloneName, "rsg-uid", newUID)
	})

	// 用例14：目标资源 NotFound 时延迟重试，目标出现后成功纳管
	It("14-requeue-when-target-not-found-and-adopt-after-target-created", Label("groupreplication"), func() {
		logStart := time.Now()
		rsgName := "rsg-e2e-14"
		groupName := "pd14"
		appName := "rsg-app-14"
		targetResources := []map[string]any{
			NewTargetResource("prefill", "apps/v1", "Deployment", "rsg-14-prefill", ""),
			NewTargetResource("decode", "apps/v1", "Deployment", "rsg-14-decode", ""),
		}

		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 2, targetResources))).NotTo(HaveOccurred())
		Consistently(func() bool {
			_, err := clientset.CoreV1().ConfigMaps(RSGTestNamespace).Get(ctx, rsgName+"-replica-config", metav1.GetOptions{})
			return apierrors.IsNotFound(err)
		}, 8*time.Second, 2*time.Second).Should(BeTrue(), "replica config should not exist before targets are created")

		Expect(CreateRSGTestDeployment(clientset, ctx, "rsg-14-prefill", RSGTestNamespace, appName, "prefill", 1)).NotTo(HaveOccurred())
		Expect(CreateRSGTestDeployment(clientset, ctx, "rsg-14-decode", RSGTestNamespace, appName, "decode", 1)).NotTo(HaveOccurred())
		Expect(CreateRSGTestService(clientset, ctx, "rsg-14-prefill-svc", RSGTestNamespace, appName, "prefill")).NotTo(HaveOccurred())
		Expect(CreateRSGTestService(clientset, ctx, "rsg-14-decode-svc", RSGTestNamespace, appName, "decode")).NotTo(HaveOccurred())

		VerifyRSGReplicaConfigMapData(clientset, ctx, RSGTestNamespace, rsgName+"-replica-config", map[string]string{
			"prefill": "1",
			"decode":  "1",
		})
		VerifyDeploymentExists(clientset, ctx, RSGTestNamespace, "pd14-rsg-14-prefill-1")
		VerifyDeploymentExists(clientset, ctx, RSGTestNamespace, "pd14-rsg-14-decode-1")
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName("rsg-14-prefill-svc", groupName, 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName("rsg-14-decode-svc", groupName, 1))
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 2, 2)
		VerifyRSGControllerLogsContainForRSG(clientset, ctx, logStart, rsgName, "target resource not found, requeue")
	})

	// 用例15：group0 标签和 selector 标签补齐
	It("15-fill-group0-and-selector-labels", Label("groupreplication"), func() {
		rsgName, groupName, appName, targetResources, serviceNames := prepareRSGGroupReplicationTargets(15)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 2, targetResources))).NotTo(HaveOccurred())

		prefillTargetName := targetResources[0]["resourceRef"].(map[string]any)["name"].(string)
		decodeTargetName := targetResources[1]["resourceRef"].(map[string]any)["name"].(string)
		prefillClone := groupName + "-" + prefillTargetName + "-1"
		prefillServiceClone := serviceNames[0] + "-" + groupName + "-1"
		rsgUID := GetRSGUID(dynamicClient, ctx, rsgName)

		VerifyDeploymentLabelsContain(clientset, ctx, RSGTestNamespace, prefillTargetName, map[string]string{
			"rsg.io/name":        rsgName,
			"rsg.io/group-id":    "0",
			"rsg.io/group-name":  groupName,
			"rsg.io/target-name": "prefill",
			"rsg-uid":            rsgUID,
			"app":                appName,
		})
		VerifyDeploymentLabelsContain(clientset, ctx, RSGTestNamespace, decodeTargetName, map[string]string{
			"rsg.io/name":        rsgName,
			"rsg.io/group-id":    "0",
			"rsg.io/group-name":  groupName,
			"rsg.io/target-name": "decode",
			"rsg-uid":            rsgUID,
			"app":                appName,
		})
		VerifyDeploymentSelectorContains(clientset, ctx, RSGTestNamespace, prefillClone, map[string]string{
			"rsg.io/group-id": "1",
			"group_id":        "1",
			"app":             appName,
		})
		VerifyServiceSelectorContains(clientset, ctx, RSGTestNamespace, prefillServiceClone, map[string]string{
			"rsg.io/group-id": "1",
			"group_id":        "1",
			"app":             appName,
		})
		VerifyDeploymentSelectorContains(clientset, ctx, RSGTestNamespace, groupName+"-"+decodeTargetName+"-1", map[string]string{
			"rsg.io/group-id": "1",
			"group_id":        "1",
			"app":             appName,
		})
		VerifyServiceSelectorContains(clientset, ctx, RSGTestNamespace, serviceNames[1]+"-"+groupName+"-1", map[string]string{
			"rsg.io/group-id": "1",
			"group_id":        "1",
			"app":             appName,
		})
		VerifyRSGStatusSelectorContains(dynamicClient, ctx, rsgName, []string{
			"app=" + appName,
			RSGWebhookTestLabelKey + "=" + RSGWebhookTestLabelValue,
		})
	})

	It("16-replica-configmap-controls-new-group-replicas", Label("groupreplication"), func() {
		rsgName, groupName, _, targetResources, _ := prepareRSGGroupReplicationTargets(16)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 1, targetResources))).NotTo(HaveOccurred())
		rsgUID := GetRSGUID(dynamicClient, ctx, rsgName)

		configMapName := rsgName + "-replica-config"
		VerifyRSGReplicaConfigMapContent(clientset, ctx, RSGTestNamespace, configMapName, rsgName, rsgUID, map[string]string{
			"prefill": "1",
			"decode":  "1",
		})
		Expect(UpdateRSGReplicaConfigMapData(clientset, ctx, RSGTestNamespace, configMapName, map[string]string{
			"prefill": "3",
			"decode":  "5",
		})).NotTo(HaveOccurred())
		VerifyRSGReplicaConfigMapContent(clientset, ctx, RSGTestNamespace, configMapName, rsgName, rsgUID, map[string]string{
			"prefill": "3",
			"decode":  "5",
		})

		Expect(updateRSGGroupReplicasWithReadyCheck(rsgName, 2)).NotTo(HaveOccurred())

		VerifyDeploymentReplicas(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[0], 1), 3)
		VerifyDeploymentReplicas(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[1], 1), 5)
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 2, 2)
	})

	It("17-cross-namespace-target-replication-converges", Label("groupreplication"), func() {
		targetNamespace := "rsg-target-17"
		Expect(EnsureNamespace(clientset, ctx, targetNamespace)).NotTo(HaveOccurred())
		CleanupRSGBusinessResourcesInNamespace(clientset, ctx, targetNamespace)
		WaitForNoRSGTestPods(clientset, ctx, targetNamespace)
		DeferCleanup(func() {
			CleanupRSGBusinessResourcesInNamespace(clientset, ctx, targetNamespace)
			WaitForNoRSGTestPods(clientset, ctx, targetNamespace)
		})

		rsgName, groupName, appName, targetResources, serviceNames := prepareRSGGroupReplicationTargetsInNamespace(17, targetNamespace)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 1, targetResources))).NotTo(HaveOccurred())
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 1, 1)

		Expect(updateRSGGroupReplicasWithReadyCheck(rsgName, 2)).NotTo(HaveOccurred())

		prefillClone := groupCloneDeploymentName(groupName, targetResources[0], 1)
		decodeClone := groupCloneDeploymentName(groupName, targetResources[1], 1)
		VerifyDeploymentExists(clientset, ctx, RSGTestNamespace, prefillClone)
		VerifyDeploymentExists(clientset, ctx, RSGTestNamespace, decodeClone)
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyDeploymentDeleted(clientset, ctx, targetNamespace, prefillClone)
		VerifyDeploymentDeleted(clientset, ctx, targetNamespace, decodeClone)
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 2, 2)
		VerifyRSGStatusSelectorContains(dynamicClient, ctx, rsgName, []string{
			"app=" + appName,
			RSGWebhookTestLabelKey + "=" + RSGWebhookTestLabelValue,
		})
	})

	It("18-inplace-scaling-expands-followers-by-ratio", Label("inplacescaling"), func() {
		rsgName := "rsg-e2e-18"
		leaderDeploymentName := "rsg-18-leader"
		followerDeploymentName := "rsg-18-follower"

		Expect(CreateRSGTestDeployment(clientset, ctx, leaderDeploymentName, RSGTestNamespace, "rsg-app-18", "leader", 1)).NotTo(HaveOccurred())
		Expect(CreateRSGTestDeployment(clientset, ctx, followerDeploymentName, RSGTestNamespace, "rsg-app-18", "follower", 1)).NotTo(HaveOccurred())

		Expect(CreateRSG(dynamicClient, ctx, newInplaceScalingRSG(rsgName, leaderDeploymentName, followerDeploymentName, 2, 3))).NotTo(HaveOccurred())

		VerifyDeploymentReplicas(clientset, ctx, RSGTestNamespace, leaderDeploymentName, 2)
		VerifyDeploymentReplicas(clientset, ctx, RSGTestNamespace, followerDeploymentName, 6)
		VerifyRSGReadyCondition(dynamicClient, ctx, rsgName, "True", "Reconciled")
	})

	It("19-scale-down-to-zero-shrinks-group0", Label("groupreplication"), Label("scaledown"), func() {
		rsgName, groupName, _, targetResources, serviceNames := prepareRSGGroupReplicationTargets(19)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 2, targetResources))).NotTo(HaveOccurred())

		prefillClone := groupCloneDeploymentName(groupName, targetResources[0], 1)
		decodeClone := groupCloneDeploymentName(groupName, targetResources[1], 1)
		VerifyDeploymentExists(clientset, ctx, RSGTestNamespace, prefillClone)
		VerifyDeploymentExists(clientset, ctx, RSGTestNamespace, decodeClone)
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 2, 2)

		Expect(updateRSGGroupReplicasWithReadyCheck(rsgName, 0)).NotTo(HaveOccurred())

		VerifyDeploymentReplicas(clientset, ctx, RSGTestNamespace, targetResourceRefName(targetResources[0]), 0)
		VerifyDeploymentReplicas(clientset, ctx, RSGTestNamespace, targetResourceRefName(targetResources[1]), 0)
		VerifyDeploymentDeleted(clientset, ctx, RSGTestNamespace, prefillClone)
		VerifyDeploymentDeleted(clientset, ctx, RSGTestNamespace, decodeClone)
		VerifyServiceDeleted(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServiceDeleted(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 0, 0)
	})

	It("20-scale-from-zero-restores-group0-before-new-groups", Label("groupreplication"), Label("scaledown"), func() {
		logStart := time.Now()
		rsgName, groupName, _, targetResources, serviceNames := prepareRSGGroupReplicationTargets(20)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 2, targetResources))).NotTo(HaveOccurred())

		prefillTargetName := targetResourceRefName(targetResources[0])
		decodeTargetName := targetResourceRefName(targetResources[1])
		prefillClone := groupCloneDeploymentName(groupName, targetResources[0], 1)
		decodeClone := groupCloneDeploymentName(groupName, targetResources[1], 1)
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 2, 2)

		Expect(updateRSGGroupReplicasWithReadyCheck(rsgName, 0)).NotTo(HaveOccurred())
		VerifyDeploymentReplicas(clientset, ctx, RSGTestNamespace, prefillTargetName, 0)
		VerifyDeploymentReplicas(clientset, ctx, RSGTestNamespace, decodeTargetName, 0)
		VerifyDeploymentDeleted(clientset, ctx, RSGTestNamespace, prefillClone)
		VerifyDeploymentDeleted(clientset, ctx, RSGTestNamespace, decodeClone)
		VerifyServiceDeleted(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServiceDeleted(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 0, 0)

		observedRestoreBeforeClone := false
		Expect(updateRSGGroupReplicasWithReadyCheck(rsgName, 2)).NotTo(HaveOccurred())

		Eventually(func() bool {
			prefill, err := clientset.AppsV1().Deployments(RSGTestNamespace).Get(ctx, prefillTargetName, metav1.GetOptions{})
			if err != nil || prefill.Spec.Replicas == nil {
				return false
			}
			decode, err := clientset.AppsV1().Deployments(RSGTestNamespace).Get(ctx, decodeTargetName, metav1.GetOptions{})
			if err != nil || decode.Spec.Replicas == nil {
				return false
			}
			restored := *prefill.Spec.Replicas == 1 && *decode.Spec.Replicas == 1

			_, err = clientset.AppsV1().Deployments(RSGTestNamespace).Get(ctx, prefillClone, metav1.GetOptions{})
			prefillCloneExists := err == nil
			_, err = clientset.AppsV1().Deployments(RSGTestNamespace).Get(ctx, decodeClone, metav1.GetOptions{})
			decodeCloneExists := err == nil
			if restored && !prefillCloneExists && !decodeCloneExists {
				observedRestoreBeforeClone = true
			}
			return restored && prefillCloneExists && decodeCloneExists
		}, 2*time.Minute, 500*time.Millisecond).Should(BeTrue())

		if !observedRestoreBeforeClone {
			GinkgoWriter.Printf("did not directly observe restore-before-clone window for %s; rely on controller log and final state verification\n", rsgName)
		}
		VerifyDeploymentReplicas(clientset, ctx, RSGTestNamespace, prefillTargetName, 1)
		VerifyDeploymentReplicas(clientset, ctx, RSGTestNamespace, decodeTargetName, 1)
		VerifyDeploymentReplicas(clientset, ctx, RSGTestNamespace, prefillClone, 1)
		VerifyDeploymentReplicas(clientset, ctx, RSGTestNamespace, decodeClone, 1)
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 2, 2)
		VerifyRSGControllerLogsContainForRSG(clientset, ctx, logStart, rsgName, "group0 restored, requeue before creating replicas")
	})

	It("23-scale-down-falls-back-to-max-group-id-when-no-metric-or-annotation", Label("groupreplication"), Label("scaledown"), func() {
		logStart := time.Now()
		rsgName, groupName, _, targetResources, serviceNames := prepareRSGGroupReplicationTargets(23)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 3, targetResources))).NotTo(HaveOccurred())

		group1Prefill := groupCloneDeploymentName(groupName, targetResources[0], 1)
		group2Prefill := groupCloneDeploymentName(groupName, targetResources[0], 2)
		group1Decode := groupCloneDeploymentName(groupName, targetResources[1], 1)
		group2Decode := groupCloneDeploymentName(groupName, targetResources[1], 2)
		VerifyDeploymentExists(clientset, ctx, RSGTestNamespace, group1Prefill)
		VerifyDeploymentExists(clientset, ctx, RSGTestNamespace, group2Prefill)
		VerifyDeploymentExists(clientset, ctx, RSGTestNamespace, group1Decode)
		VerifyDeploymentExists(clientset, ctx, RSGTestNamespace, group2Decode)
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 2))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 2))
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 3, 3)

		Expect(updateRSGGroupReplicasWithReadyCheck(rsgName, 2)).NotTo(HaveOccurred())

		VerifyDeploymentExists(clientset, ctx, RSGTestNamespace, group1Prefill)
		VerifyDeploymentExists(clientset, ctx, RSGTestNamespace, group1Decode)
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyDeploymentDeleted(clientset, ctx, RSGTestNamespace, group2Prefill)
		VerifyDeploymentDeleted(clientset, ctx, RSGTestNamespace, group2Decode)
		VerifyServiceDeleted(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 2))
		VerifyServiceDeleted(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 2))
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 2, 2)
		VerifyRSGControllerLogsContainForRSG(clientset, ctx, logStart, rsgName, "fallback to max group-id")
	})

	It("25-invalid-annotation-falls-back-to-max-group-id", Label("groupreplication"), Label("scaledown"), func() {
		logStart := time.Now()
		rsgName, groupName, _, targetResources, serviceNames := prepareRSGGroupReplicationTargets(25)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 3, targetResources))).NotTo(HaveOccurred())

		prefillTargetName := targetResourceRefName(targetResources[0])
		group1Prefill := groupCloneDeploymentName(groupName, targetResources[0], 1)
		group2Prefill := groupCloneDeploymentName(groupName, targetResources[0], 2)
		group1Decode := groupCloneDeploymentName(groupName, targetResources[1], 1)
		group2Decode := groupCloneDeploymentName(groupName, targetResources[1], 2)

		VerifyDeploymentExists(clientset, ctx, RSGTestNamespace, group1Prefill)
		VerifyDeploymentExists(clientset, ctx, RSGTestNamespace, group2Prefill)
		VerifyDeploymentExists(clientset, ctx, RSGTestNamespace, group1Decode)
		VerifyDeploymentExists(clientset, ctx, RSGTestNamespace, group2Decode)
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 2))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 2))
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 3, 3)

		Expect(AnnotateRSGTestDeployment(clientset, ctx, RSGTestNamespace, prefillTargetName, map[string]string{
			rsgMetricAnnotationKey: "not-a-number",
		})).NotTo(HaveOccurred())
		Expect(AnnotateRSGTestDeployment(clientset, ctx, RSGTestNamespace, group1Prefill, map[string]string{
			rsgMetricAnnotationKey: "not-a-number",
		})).NotTo(HaveOccurred())
		Expect(AnnotateRSGTestDeployment(clientset, ctx, RSGTestNamespace, group2Prefill, map[string]string{
			rsgMetricAnnotationKey: "not-a-number",
		})).NotTo(HaveOccurred())

		Expect(updateRSGGroupReplicasWithReadyCheck(rsgName, 2)).NotTo(HaveOccurred())

		VerifyDeploymentExists(clientset, ctx, RSGTestNamespace, group1Prefill)
		VerifyDeploymentExists(clientset, ctx, RSGTestNamespace, group1Decode)
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyDeploymentDeleted(clientset, ctx, RSGTestNamespace, group2Prefill)
		VerifyDeploymentDeleted(clientset, ctx, RSGTestNamespace, group2Decode)
		VerifyServiceDeleted(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 2))
		VerifyServiceDeleted(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 2))
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 2, 2)
		VerifyRSGControllerLogsContainForRSG(clientset, ctx, logStart, rsgName, "annotation metric invalid", "fallback to max group-id")
	})

	It("26-statefulset-create-snapshot-and-configmap-on-first-reconcile", Label("groupreplication"), Label("statefulset"), func() {
		logStart := time.Now()
		rsgName, groupName, appName, targetResources, serviceNames := prepareRSGGroupReplicationStatefulSetTargets(26)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 1, targetResources))).NotTo(HaveOccurred())
		rsgUID := GetRSGUID(dynamicClient, ctx, rsgName)

		VerifyRSGControllerRevisionSnapshot(clientset, ctx, RSGTestNamespace, rsgName+"-prefill-snapshot", rsgName, rsgUID, "prefill", "apps/v1", "StatefulSet", targetResourceRefName(targetResources[0]), RSGTestNamespace, 1, serviceNames[0])
		VerifyRSGControllerRevisionSnapshot(clientset, ctx, RSGTestNamespace, rsgName+"-decode-snapshot", rsgName, rsgUID, "decode", "apps/v1", "StatefulSet", targetResourceRefName(targetResources[1]), RSGTestNamespace, 1, serviceNames[1])
		VerifyRSGReplicaConfigMapContent(clientset, ctx, RSGTestNamespace, rsgName+"-replica-config", rsgName, rsgUID, map[string]string{
			"prefill": "1",
			"decode":  "1",
		})
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 1, 1)
		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, targetResourceRefName(targetResources[0]))
		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, targetResourceRefName(targetResources[1]))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, serviceNames[0])
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, serviceNames[1])
		VerifyRSGStatusSelectorContains(dynamicClient, ctx, rsgName, []string{
			"app=" + appName,
			RSGWebhookTestLabelKey + "=" + RSGWebhookTestLabelValue,
		})
		VerifyRSGControllerLogsContainForRSG(clientset, ctx, logStart, rsgName, "snapshot created", "replica configmap created")
	})

	It("27-statefulset-create-group-by-desired-replicas", Label("groupreplication"), Label("statefulset"), func() {
		rsgName, groupName, appName, targetResources, serviceNames := prepareRSGGroupReplicationStatefulSetTargets(27)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 1, targetResources))).NotTo(HaveOccurred())
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 1, 1)

		Expect(updateRSGGroupReplicasWithReadyCheck(rsgName, 2)).NotTo(HaveOccurred())

		prefillClone := groupCloneDeploymentName(groupName, targetResources[0], 1)
		decodeClone := groupCloneDeploymentName(groupName, targetResources[1], 1)
		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, prefillClone)
		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, decodeClone)
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyStatefulSetServiceName(clientset, ctx, RSGTestNamespace, prefillClone, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyStatefulSetServiceName(clientset, ctx, RSGTestNamespace, decodeClone, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 2, 2)
		VerifyRSGStatusSelectorContains(dynamicClient, ctx, rsgName, []string{
			"app=" + appName,
			RSGWebhookTestLabelKey + "=" + RSGWebhookTestLabelValue,
		})
	})

	It("28-delete-statefulset-rsg-and-keep-business-resources", Label("groupreplication"), Label("statefulset"), func() {
		rsgName, groupName, _, targetResources, serviceNames := prepareRSGGroupReplicationStatefulSetTargets(28)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 2, targetResources))).NotTo(HaveOccurred())

		prefillClone := groupCloneDeploymentName(groupName, targetResources[0], 1)
		decodeClone := groupCloneDeploymentName(groupName, targetResources[1], 1)
		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, prefillClone)
		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, decodeClone)
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))

		DeleteRSGAndWait(dynamicClient, ctx, rsgName)

		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, prefillClone)
		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, decodeClone)
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyStatefulSetHasNoOwnerReferences(clientset, ctx, RSGTestNamespace, prefillClone)
		VerifyStatefulSetHasNoOwnerReferences(clientset, ctx, RSGTestNamespace, decodeClone)
		VerifyServiceHasNoOwnerReferences(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServiceHasNoOwnerReferences(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyStatefulSetPersists(clientset, ctx, RSGTestNamespace, prefillClone)
		VerifyStatefulSetPersists(clientset, ctx, RSGTestNamespace, decodeClone)
		VerifyServicePersists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServicePersists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyRSGReplicaConfigMapDeleted(clientset, ctx, RSGTestNamespace, rsgName+"-replica-config")
		VerifyRSGControllerRevisionDeleted(clientset, ctx, RSGTestNamespace, rsgName+"-prefill-snapshot")
		VerifyRSGControllerRevisionDeleted(clientset, ctx, RSGTestNamespace, rsgName+"-decode-snapshot")
	})

	It("29-recreate-same-statefulset-rsg-name-without-taking-over-old-resources", Label("groupreplication"), Label("statefulset"), func() {
		rsgName, groupName, _, targetResources, _ := prepareRSGGroupReplicationStatefulSetTargets(29)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 2, targetResources))).NotTo(HaveOccurred())

		cloneName := groupCloneDeploymentName(groupName, targetResources[0], 1)
		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, cloneName)
		oldUID := GetRSGUID(dynamicClient, ctx, rsgName)
		VerifyStatefulSetLabelValue(clientset, ctx, RSGTestNamespace, cloneName, "rsg-uid", oldUID)

		DeleteRSGAndWait(dynamicClient, ctx, rsgName)

		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 2, targetResources))).NotTo(HaveOccurred())
		newUID := GetRSGUID(dynamicClient, ctx, rsgName)
		Expect(newUID).NotTo(Equal(oldUID))

		VerifyStatefulSetLabelValue(clientset, ctx, RSGTestNamespace, cloneName, "rsg-uid", oldUID)
		VerifyStatefulSetLabelValueNotEqual(clientset, ctx, RSGTestNamespace, cloneName, "rsg-uid", newUID)
	})

	It("30-requeue-when-statefulset-target-not-found-and-adopt-after-target-created", Label("groupreplication"), Label("statefulset"), func() {
		logStart := time.Now()
		rsgName := "rsg-e2e-30"
		groupName := "pd30"
		appName := "rsg-app-30"
		targetResources := []map[string]any{
			NewTargetResource("prefill", "apps/v1", "StatefulSet", "rsg-30-prefill", ""),
			NewTargetResource("decode", "apps/v1", "StatefulSet", "rsg-30-decode", ""),
		}

		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 2, targetResources))).NotTo(HaveOccurred())
		Consistently(func() bool {
			_, err := clientset.CoreV1().ConfigMaps(RSGTestNamespace).Get(ctx, rsgName+"-replica-config", metav1.GetOptions{})
			return apierrors.IsNotFound(err)
		}, 8*time.Second, 2*time.Second).Should(BeTrue(), "replica config should not exist before statefulset targets are created")

		Expect(CreateRSGTestHeadlessService(clientset, ctx, "rsg-30-prefill-headless", RSGTestNamespace, appName, "prefill")).NotTo(HaveOccurred())
		Expect(CreateRSGTestHeadlessService(clientset, ctx, "rsg-30-decode-headless", RSGTestNamespace, appName, "decode")).NotTo(HaveOccurred())
		Expect(CreateRSGTestStatefulSet(clientset, ctx, "rsg-30-prefill", RSGTestNamespace, appName, "prefill", "rsg-30-prefill-headless", 1)).NotTo(HaveOccurred())
		Expect(CreateRSGTestStatefulSet(clientset, ctx, "rsg-30-decode", RSGTestNamespace, appName, "decode", "rsg-30-decode-headless", 1)).NotTo(HaveOccurred())
		rsgUID := GetRSGUID(dynamicClient, ctx, rsgName)

		VerifyRSGReplicaConfigMapContent(clientset, ctx, RSGTestNamespace, rsgName+"-replica-config", rsgName, rsgUID, map[string]string{
			"prefill": "1",
			"decode":  "1",
		})
		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, "pd30-rsg-30-prefill-1")
		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, "pd30-rsg-30-decode-1")
		VerifyStatefulSetServiceName(clientset, ctx, RSGTestNamespace, "pd30-rsg-30-prefill-1", "rsg-30-prefill-headless-pd30-1")
		VerifyStatefulSetServiceName(clientset, ctx, RSGTestNamespace, "pd30-rsg-30-decode-1", "rsg-30-decode-headless-pd30-1")
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName("rsg-30-prefill-headless", groupName, 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName("rsg-30-decode-headless", groupName, 1))
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 2, 2)
		VerifyRSGControllerLogsContainForRSG(clientset, ctx, logStart, rsgName, "target resource not found, requeue")
	})

	It("31-fill-statefulset-group0-and-selector-labels", Label("groupreplication"), Label("statefulset"), func() {
		rsgName, groupName, appName, targetResources, serviceNames := prepareRSGGroupReplicationStatefulSetTargets(31)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 2, targetResources))).NotTo(HaveOccurred())

		prefillTargetName := targetResourceRefName(targetResources[0])
		decodeTargetName := targetResourceRefName(targetResources[1])
		prefillClone := groupCloneDeploymentName(groupName, targetResources[0], 1)
		rsgUID := GetRSGUID(dynamicClient, ctx, rsgName)

		VerifyStatefulSetLabelValue(clientset, ctx, RSGTestNamespace, prefillTargetName, "rsg.io/name", rsgName)
		VerifyStatefulSetLabelValue(clientset, ctx, RSGTestNamespace, prefillTargetName, "rsg-uid", rsgUID)
		VerifyStatefulSetLabelValue(clientset, ctx, RSGTestNamespace, decodeTargetName, "rsg.io/name", rsgName)
		VerifyStatefulSetLabelValue(clientset, ctx, RSGTestNamespace, decodeTargetName, "rsg-uid", rsgUID)
		VerifyStatefulSetSelectorContains(clientset, ctx, RSGTestNamespace, prefillClone, map[string]string{
			"rsg.io/group-id": "1",
			"group_id":        "1",
			"app":             appName,
		})
		VerifyStatefulSetTemplateLabelsContain(clientset, ctx, RSGTestNamespace, prefillClone, map[string]string{
			"rsg.io/group-id": "1",
			"group_id":        "1",
			"app":             appName,
		})
		VerifyServiceSelectorContains(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1), map[string]string{
			"rsg.io/group-id": "1",
			"group_id":        "1",
			"app":             appName,
		})
		VerifyStatefulSetSelectorContains(clientset, ctx, RSGTestNamespace, groupName+"-"+decodeTargetName+"-1", map[string]string{
			"rsg.io/group-id": "1",
			"group_id":        "1",
			"app":             appName,
		})
		VerifyStatefulSetTemplateLabelsContain(clientset, ctx, RSGTestNamespace, groupName+"-"+decodeTargetName+"-1", map[string]string{
			"rsg.io/group-id": "1",
			"group_id":        "1",
			"app":             appName,
		})
		VerifyServiceSelectorContains(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1), map[string]string{
			"rsg.io/group-id": "1",
			"group_id":        "1",
			"app":             appName,
		})
		VerifyRSGStatusSelectorContains(dynamicClient, ctx, rsgName, []string{
			"app=" + appName,
			RSGWebhookTestLabelKey + "=" + RSGWebhookTestLabelValue,
		})
	})

	It("32-statefulset-replica-configmap-controls-new-group-replicas", Label("groupreplication"), Label("statefulset"), func() {
		rsgName, groupName, _, targetResources, _ := prepareRSGGroupReplicationStatefulSetTargets(32)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 1, targetResources))).NotTo(HaveOccurred())
		rsgUID := GetRSGUID(dynamicClient, ctx, rsgName)

		configMapName := rsgName + "-replica-config"
		VerifyRSGReplicaConfigMapContent(clientset, ctx, RSGTestNamespace, configMapName, rsgName, rsgUID, map[string]string{
			"prefill": "1",
			"decode":  "1",
		})
		Expect(UpdateRSGReplicaConfigMapData(clientset, ctx, RSGTestNamespace, configMapName, map[string]string{
			"prefill": "3",
			"decode":  "5",
		})).NotTo(HaveOccurred())
		VerifyRSGReplicaConfigMapContent(clientset, ctx, RSGTestNamespace, configMapName, rsgName, rsgUID, map[string]string{
			"prefill": "3",
			"decode":  "5",
		})

		Expect(updateRSGGroupReplicasWithReadyCheck(rsgName, 2)).NotTo(HaveOccurred())

		VerifyStatefulSetReplicas(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[0], 1), 3)
		VerifyStatefulSetReplicas(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[1], 1), 5)
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 2, 2)
	})

	It("33-cross-namespace-statefulset-target-replication-converges", Label("groupreplication"), Label("statefulset"), func() {
		targetNamespace := "rsg-target-33"
		Expect(EnsureNamespace(clientset, ctx, targetNamespace)).NotTo(HaveOccurred())
		CleanupRSGBusinessResourcesInNamespace(clientset, ctx, targetNamespace)
		WaitForNoRSGTestPods(clientset, ctx, targetNamespace)
		DeferCleanup(func() {
			CleanupRSGBusinessResourcesInNamespace(clientset, ctx, targetNamespace)
			WaitForNoRSGTestPods(clientset, ctx, targetNamespace)
		})

		rsgName, groupName, appName, targetResources, serviceNames := prepareRSGGroupReplicationStatefulSetTargetsInNamespace(33, targetNamespace)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 1, targetResources))).NotTo(HaveOccurred())
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 1, 1)

		Expect(updateRSGGroupReplicasWithReadyCheck(rsgName, 2)).NotTo(HaveOccurred())

		prefillClone := groupCloneDeploymentName(groupName, targetResources[0], 1)
		decodeClone := groupCloneDeploymentName(groupName, targetResources[1], 1)
		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, prefillClone)
		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, decodeClone)
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyStatefulSetDeleted(clientset, ctx, targetNamespace, prefillClone)
		VerifyStatefulSetDeleted(clientset, ctx, targetNamespace, decodeClone)
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 2, 2)
		VerifyRSGStatusSelectorContains(dynamicClient, ctx, rsgName, []string{
			"app=" + appName,
			RSGWebhookTestLabelKey + "=" + RSGWebhookTestLabelValue,
		})
	})

	It("34-statefulset-inplace-scaling-expands-followers-by-ratio", Label("inplacescaling"), Label("statefulset"), func() {
		rsgName := "rsg-e2e-34"
		appName := "rsg-app-34"
		leaderName := "rsg-34-leader"
		followerName := "rsg-34-follower"

		Expect(CreateRSGTestHeadlessService(clientset, ctx, "rsg-34-leader-headless", RSGTestNamespace, appName, "leader")).NotTo(HaveOccurred())
		Expect(CreateRSGTestHeadlessService(clientset, ctx, "rsg-34-follower-headless", RSGTestNamespace, appName, "follower")).NotTo(HaveOccurred())
		Expect(CreateRSGTestStatefulSet(clientset, ctx, leaderName, RSGTestNamespace, appName, "leader", "rsg-34-leader-headless", 1)).NotTo(HaveOccurred())
		Expect(CreateRSGTestStatefulSet(clientset, ctx, followerName, RSGTestNamespace, appName, "follower", "rsg-34-follower-headless", 1)).NotTo(HaveOccurred())

		targetResources := []map[string]any{
			NewTargetResource("leader", "apps/v1", "StatefulSet", leaderName, ""),
			NewTargetResource("follower", "apps/v1", "StatefulSet", followerName, ""),
		}
		Expect(CreateRSG(dynamicClient, ctx, newInplaceScalingRSGWithTargets(rsgName, targetResources, 2, 3))).NotTo(HaveOccurred())

		VerifyStatefulSetReplicas(clientset, ctx, RSGTestNamespace, leaderName, 2)
		VerifyStatefulSetReplicas(clientset, ctx, RSGTestNamespace, followerName, 6)
		VerifyRSGReadyCondition(dynamicClient, ctx, rsgName, "True", "Reconciled")
	})

	It("35-statefulset-clone-service-name-is-rewritten", Label("groupreplication"), Label("statefulset"), func() {
		rsgName, groupName, _, targetResources, serviceNames := prepareRSGGroupReplicationStatefulSetTargets(35)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 2, targetResources))).NotTo(HaveOccurred())

		VerifyStatefulSetServiceName(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[0], 1), groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyStatefulSetServiceName(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[1], 1), groupCloneServiceName(serviceNames[1], groupName, 1))
	})

	It("36-scale-down-statefulset-to-zero-shrinks-group0", Label("groupreplication"), Label("statefulset"), Label("scaledown"), func() {
		rsgName, groupName, _, targetResources, serviceNames := prepareRSGGroupReplicationStatefulSetTargets(36)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 2, targetResources))).NotTo(HaveOccurred())

		prefillClone := groupCloneDeploymentName(groupName, targetResources[0], 1)
		decodeClone := groupCloneDeploymentName(groupName, targetResources[1], 1)
		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, prefillClone)
		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, decodeClone)
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 2, 2)

		Expect(updateRSGGroupReplicasWithReadyCheck(rsgName, 0)).NotTo(HaveOccurred())

		VerifyStatefulSetReplicas(clientset, ctx, RSGTestNamespace, targetResourceRefName(targetResources[0]), 0)
		VerifyStatefulSetReplicas(clientset, ctx, RSGTestNamespace, targetResourceRefName(targetResources[1]), 0)
		VerifyStatefulSetDeleted(clientset, ctx, RSGTestNamespace, prefillClone)
		VerifyStatefulSetDeleted(clientset, ctx, RSGTestNamespace, decodeClone)
		VerifyServiceDeleted(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServiceDeleted(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 0, 0)
	})

	It("37-scale-statefulset-from-zero-restores-group0-before-new-groups", Label("groupreplication"), Label("statefulset"), Label("scaledown"), func() {
		logStart := time.Now()
		rsgName, groupName, _, targetResources, serviceNames := prepareRSGGroupReplicationStatefulSetTargets(37)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 2, targetResources))).NotTo(HaveOccurred())

		prefillTargetName := targetResourceRefName(targetResources[0])
		decodeTargetName := targetResourceRefName(targetResources[1])
		prefillClone := groupCloneDeploymentName(groupName, targetResources[0], 1)
		decodeClone := groupCloneDeploymentName(groupName, targetResources[1], 1)
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 2, 2)

		Expect(updateRSGGroupReplicasWithReadyCheck(rsgName, 0)).NotTo(HaveOccurred())
		VerifyStatefulSetReplicas(clientset, ctx, RSGTestNamespace, prefillTargetName, 0)
		VerifyStatefulSetReplicas(clientset, ctx, RSGTestNamespace, decodeTargetName, 0)
		VerifyStatefulSetDeleted(clientset, ctx, RSGTestNamespace, prefillClone)
		VerifyStatefulSetDeleted(clientset, ctx, RSGTestNamespace, decodeClone)
		VerifyServiceDeleted(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServiceDeleted(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 0, 0)

		observedRestoreBeforeClone := false
		Expect(updateRSGGroupReplicasWithReadyCheck(rsgName, 2)).NotTo(HaveOccurred())

		Eventually(func() bool {
			prefill, err := clientset.AppsV1().StatefulSets(RSGTestNamespace).Get(ctx, prefillTargetName, metav1.GetOptions{})
			if err != nil || prefill.Spec.Replicas == nil {
				return false
			}
			decode, err := clientset.AppsV1().StatefulSets(RSGTestNamespace).Get(ctx, decodeTargetName, metav1.GetOptions{})
			if err != nil || decode.Spec.Replicas == nil {
				return false
			}
			restored := *prefill.Spec.Replicas == 1 && *decode.Spec.Replicas == 1

			_, err = clientset.AppsV1().StatefulSets(RSGTestNamespace).Get(ctx, prefillClone, metav1.GetOptions{})
			prefillCloneExists := err == nil
			_, err = clientset.AppsV1().StatefulSets(RSGTestNamespace).Get(ctx, decodeClone, metav1.GetOptions{})
			decodeCloneExists := err == nil
			if restored && !prefillCloneExists && !decodeCloneExists {
				observedRestoreBeforeClone = true
			}
			return restored && prefillCloneExists && decodeCloneExists
		}, 2*time.Minute, 500*time.Millisecond).Should(BeTrue())

		if !observedRestoreBeforeClone {
			GinkgoWriter.Printf("did not directly observe restore-before-clone window for %s; rely on controller log and final state verification\n", rsgName)
		}
		VerifyStatefulSetReplicas(clientset, ctx, RSGTestNamespace, prefillTargetName, 1)
		VerifyStatefulSetReplicas(clientset, ctx, RSGTestNamespace, decodeTargetName, 1)
		VerifyStatefulSetReplicas(clientset, ctx, RSGTestNamespace, prefillClone, 1)
		VerifyStatefulSetReplicas(clientset, ctx, RSGTestNamespace, decodeClone, 1)
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 2, 2)
		VerifyRSGControllerLogsContainForRSG(clientset, ctx, logStart, rsgName, "group0 restored, requeue before creating replicas")
	})

	It("39-statefulset-prometheus-missing-falls-back-to-annotation", Label("groupreplication"), Label("statefulset"), Label("scaledown"), func() {
		logStart := time.Now()
		rsgName, groupName, _, targetResources, serviceNames := prepareRSGGroupReplicationStatefulSetTargets(39)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithScaleDown(rsgName, groupName, 3, targetResources, "ascending"))).NotTo(HaveOccurred())

		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[0], 1))
		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[0], 2))
		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[1], 1))
		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[1], 2))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 2))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 2))
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 3, 3)

		Expect(AnnotateRSGTestStatefulSet(clientset, ctx, RSGTestNamespace, targetResourceRefName(targetResources[0]), map[string]string{rsgMetricAnnotationKey: "30"})).NotTo(HaveOccurred())
		Expect(AnnotateRSGTestStatefulSet(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[0], 1), map[string]string{rsgMetricAnnotationKey: "10"})).NotTo(HaveOccurred())
		Expect(AnnotateRSGTestStatefulSet(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[0], 2), map[string]string{rsgMetricAnnotationKey: "20"})).NotTo(HaveOccurred())

		Expect(updateRSGGroupReplicasWithReadyCheck(rsgName, 2)).NotTo(HaveOccurred())

		VerifyStatefulSetDeleted(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[0], 1))
		VerifyStatefulSetDeleted(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[1], 1))
		VerifyServiceDeleted(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServiceDeleted(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[0], 2))
		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[1], 2))
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 2, 2)
		VerifyRSGControllerLogsContainForRSG(clientset, ctx, logStart, rsgName, "fallback to annotation")
	})

	It("40-statefulset-scale-down-falls-back-to-max-group-id-when-no-metric-or-annotation", Label("groupreplication"), Label("statefulset"), Label("scaledown"), func() {
		logStart := time.Now()
		rsgName, groupName, _, targetResources, serviceNames := prepareRSGGroupReplicationStatefulSetTargets(40)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 3, targetResources))).NotTo(HaveOccurred())

		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[0], 1))
		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[0], 2))
		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[1], 1))
		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[1], 2))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 2))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 2))
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 3, 3)

		Expect(updateRSGGroupReplicasWithReadyCheck(rsgName, 2)).NotTo(HaveOccurred())

		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[0], 1))
		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[1], 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyStatefulSetDeleted(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[0], 2))
		VerifyStatefulSetDeleted(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[1], 2))
		VerifyServiceDeleted(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 2))
		VerifyServiceDeleted(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 2))
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 2, 2)
		VerifyRSGControllerLogsContainForRSG(clientset, ctx, logStart, rsgName, "fallback to max group-id")
	})

	It("42-invalid-statefulset-annotation-falls-back-to-max-group-id", Label("groupreplication"), Label("statefulset"), Label("scaledown"), func() {
		logStart := time.Now()
		rsgName, groupName, _, targetResources, serviceNames := prepareRSGGroupReplicationStatefulSetTargets(42)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 3, targetResources))).NotTo(HaveOccurred())

		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[0], 1))
		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[0], 2))
		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[1], 1))
		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[1], 2))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 2))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 2))
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 3, 3)

		Expect(AnnotateRSGTestStatefulSet(clientset, ctx, RSGTestNamespace, targetResourceRefName(targetResources[0]), map[string]string{rsgMetricAnnotationKey: "bad"})).NotTo(HaveOccurred())
		Expect(AnnotateRSGTestStatefulSet(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[0], 1), map[string]string{rsgMetricAnnotationKey: "bad"})).NotTo(HaveOccurred())
		Expect(AnnotateRSGTestStatefulSet(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[0], 2), map[string]string{rsgMetricAnnotationKey: "bad"})).NotTo(HaveOccurred())

		Expect(updateRSGGroupReplicasWithReadyCheck(rsgName, 2)).NotTo(HaveOccurred())

		VerifyStatefulSetDeleted(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[0], 2))
		VerifyStatefulSetDeleted(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[1], 2))
		VerifyServiceDeleted(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 2))
		VerifyServiceDeleted(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 2))
		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[0], 1))
		VerifyStatefulSetExists(clientset, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[1], 1))
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 2, 2)
		VerifyRSGControllerLogsContainForRSG(clientset, ctx, logStart, rsgName, "annotation metric invalid", "fallback to max group-id")
	})

	It("43-lws-create-snapshot-and-configmap-on-first-reconcile", Label("groupreplication"), Label("lws"), func() {
		logStart := time.Now()
		rsgName, groupName, appName, targetResources, serviceNames := prepareRSGGroupReplicationLWSTargets(43)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 1, targetResources))).NotTo(HaveOccurred())
		rsgUID := GetRSGUID(dynamicClient, ctx, rsgName)

		VerifyRSGControllerRevisionSnapshot(clientset, ctx, RSGTestNamespace, rsgName+"-prefill-snapshot", rsgName, rsgUID, "prefill", LeaderWorkerSetAPIVersion, "LeaderWorkerSet", targetResourceRefName(targetResources[0]), RSGTestNamespace, 1, "")
		VerifyRSGControllerRevisionSnapshot(clientset, ctx, RSGTestNamespace, rsgName+"-decode-snapshot", rsgName, rsgUID, "decode", LeaderWorkerSetAPIVersion, "LeaderWorkerSet", targetResourceRefName(targetResources[1]), RSGTestNamespace, 1, "")
		VerifyRSGReplicaConfigMapContent(clientset, ctx, RSGTestNamespace, rsgName+"-replica-config", rsgName, rsgUID, map[string]string{
			"prefill": "1",
			"decode":  "1",
		})
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 1, 1)
		VerifyLWSExists(dynamicClient, ctx, RSGTestNamespace, targetResourceRefName(targetResources[0]))
		VerifyLWSExists(dynamicClient, ctx, RSGTestNamespace, targetResourceRefName(targetResources[1]))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, serviceNames[0])
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, serviceNames[1])
		VerifyRSGStatusSelectorContains(dynamicClient, ctx, rsgName, []string{
			"app=" + appName,
			RSGWebhookTestLabelKey + "=" + RSGWebhookTestLabelValue,
		})
		VerifyRSGControllerLogsContainForRSG(clientset, ctx, logStart, rsgName, "snapshot created", "replica configmap created")
	})

	It("44-lws-create-group-by-desired-replicas", Label("groupreplication"), Label("lws"), func() {
		rsgName, groupName, appName, targetResources, serviceNames := prepareRSGGroupReplicationLWSTargets(44)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 1, targetResources))).NotTo(HaveOccurred())
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 1, 1)

		Expect(updateRSGGroupReplicasWithReadyCheck(rsgName, 2)).NotTo(HaveOccurred())

		VerifyLWSExists(dynamicClient, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[0], 1))
		VerifyLWSExists(dynamicClient, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[1], 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 2, 2)
		VerifyRSGStatusSelectorContains(dynamicClient, ctx, rsgName, []string{
			"app=" + appName,
			RSGWebhookTestLabelKey + "=" + RSGWebhookTestLabelValue,
		})
	})

	It("45-delete-lws-rsg-and-keep-business-resources", Label("groupreplication"), Label("lws"), func() {
		rsgName, groupName, _, targetResources, serviceNames := prepareRSGGroupReplicationLWSTargets(45)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 2, targetResources))).NotTo(HaveOccurred())

		prefillClone := groupCloneDeploymentName(groupName, targetResources[0], 1)
		decodeClone := groupCloneDeploymentName(groupName, targetResources[1], 1)
		VerifyLWSExists(dynamicClient, ctx, RSGTestNamespace, prefillClone)
		VerifyLWSExists(dynamicClient, ctx, RSGTestNamespace, decodeClone)
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))

		DeleteRSGAndWait(dynamicClient, ctx, rsgName)

		VerifyLWSExists(dynamicClient, ctx, RSGTestNamespace, prefillClone)
		VerifyLWSExists(dynamicClient, ctx, RSGTestNamespace, decodeClone)
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyLWSHasNoOwnerReferences(dynamicClient, ctx, RSGTestNamespace, prefillClone)
		VerifyLWSHasNoOwnerReferences(dynamicClient, ctx, RSGTestNamespace, decodeClone)
		VerifyServiceHasNoOwnerReferences(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServiceHasNoOwnerReferences(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyLWSPersists(dynamicClient, ctx, RSGTestNamespace, prefillClone)
		VerifyLWSPersists(dynamicClient, ctx, RSGTestNamespace, decodeClone)
		VerifyServicePersists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServicePersists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyRSGReplicaConfigMapDeleted(clientset, ctx, RSGTestNamespace, rsgName+"-replica-config")
		VerifyRSGControllerRevisionDeleted(clientset, ctx, RSGTestNamespace, rsgName+"-prefill-snapshot")
		VerifyRSGControllerRevisionDeleted(clientset, ctx, RSGTestNamespace, rsgName+"-decode-snapshot")
	})

	It("46-recreate-same-lws-rsg-name-without-taking-over-old-resources", Label("groupreplication"), Label("lws"), func() {
		rsgName, groupName, _, targetResources, _ := prepareRSGGroupReplicationLWSTargets(46)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 2, targetResources))).NotTo(HaveOccurred())

		cloneName := groupCloneDeploymentName(groupName, targetResources[0], 1)
		VerifyLWSExists(dynamicClient, ctx, RSGTestNamespace, cloneName)
		oldUID := GetRSGUID(dynamicClient, ctx, rsgName)
		VerifyLWSLabelValue(dynamicClient, ctx, RSGTestNamespace, cloneName, "rsg-uid", oldUID)

		DeleteRSGAndWait(dynamicClient, ctx, rsgName)

		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 2, targetResources))).NotTo(HaveOccurred())
		newUID := GetRSGUID(dynamicClient, ctx, rsgName)
		Expect(newUID).NotTo(Equal(oldUID))

		VerifyLWSLabelValue(dynamicClient, ctx, RSGTestNamespace, cloneName, "rsg-uid", oldUID)
		VerifyLWSLabelValueNotEqual(dynamicClient, ctx, RSGTestNamespace, cloneName, "rsg-uid", newUID)
	})

	It("47-requeue-when-lws-target-not-found-and-adopt-after-target-created", Label("groupreplication"), Label("lws"), func() {
		logStart := time.Now()
		rsgName := "rsg-e2e-47"
		groupName := "pd47"
		appName := "rsg-app-47"
		targetResources := []map[string]any{
			NewTargetResource("prefill", LeaderWorkerSetAPIVersion, "LeaderWorkerSet", "rsg-47-prefill", ""),
			NewTargetResource("decode", LeaderWorkerSetAPIVersion, "LeaderWorkerSet", "rsg-47-decode", ""),
		}

		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 2, targetResources))).NotTo(HaveOccurred())
		Consistently(func() bool {
			_, err := clientset.CoreV1().ConfigMaps(RSGTestNamespace).Get(ctx, rsgName+"-replica-config", metav1.GetOptions{})
			return apierrors.IsNotFound(err)
		}, 8*time.Second, 2*time.Second).Should(BeTrue(), "replica config should not exist before lws targets are created")

		Expect(CreateRSGTestService(clientset, ctx, "rsg-47-prefill-svc", RSGTestNamespace, appName, "prefill")).NotTo(HaveOccurred())
		Expect(CreateRSGTestService(clientset, ctx, "rsg-47-decode-svc", RSGTestNamespace, appName, "decode")).NotTo(HaveOccurred())
		Expect(CreateRSGTestLeaderWorkerSet(dynamicClient, ctx, "rsg-47-prefill", RSGTestNamespace, appName, "prefill", 1)).NotTo(HaveOccurred())
		Expect(CreateRSGTestLeaderWorkerSet(dynamicClient, ctx, "rsg-47-decode", RSGTestNamespace, appName, "decode", 1)).NotTo(HaveOccurred())

		rsgUID := GetRSGUID(dynamicClient, ctx, rsgName)
		VerifyRSGReplicaConfigMapContent(clientset, ctx, RSGTestNamespace, rsgName+"-replica-config", rsgName, rsgUID, map[string]string{
			"prefill": "1",
			"decode":  "1",
		})
		VerifyLWSExists(dynamicClient, ctx, RSGTestNamespace, "pd47-rsg-47-prefill-1")
		VerifyLWSExists(dynamicClient, ctx, RSGTestNamespace, "pd47-rsg-47-decode-1")
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName("rsg-47-prefill-svc", groupName, 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName("rsg-47-decode-svc", groupName, 1))
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 2, 2)
		VerifyRSGControllerLogsContainForRSG(clientset, ctx, logStart, rsgName, "target resource not found, requeue")
	})

	It("48-fill-lws-group0-and-selector-labels", Label("groupreplication"), Label("lws"), func() {
		rsgName, groupName, appName, targetResources, serviceNames := prepareRSGGroupReplicationLWSTargets(48)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 2, targetResources))).NotTo(HaveOccurred())

		prefillTargetName := targetResourceRefName(targetResources[0])
		decodeTargetName := targetResourceRefName(targetResources[1])
		prefillClone := groupCloneDeploymentName(groupName, targetResources[0], 1)
		rsgUID := GetRSGUID(dynamicClient, ctx, rsgName)

		VerifyLWSLabelsContain(dynamicClient, ctx, RSGTestNamespace, prefillTargetName, map[string]string{
			"rsg.io/name":        rsgName,
			"rsg.io/group-id":    "0",
			"rsg.io/group-name":  groupName,
			"rsg.io/target-name": "prefill",
			"rsg-uid":            rsgUID,
			"app":                appName,
		})
		VerifyLWSLabelsContain(dynamicClient, ctx, RSGTestNamespace, decodeTargetName, map[string]string{
			"rsg.io/name":        rsgName,
			"rsg.io/group-id":    "0",
			"rsg.io/group-name":  groupName,
			"rsg.io/target-name": "decode",
			"rsg-uid":            rsgUID,
			"app":                appName,
		})
		VerifyLWSTemplateLabelsContain(dynamicClient, ctx, RSGTestNamespace, prefillClone, map[string]string{
			"rsg.io/group-id": "1",
			"group_id":        "1",
			"app":             appName,
		})
		VerifyServiceSelectorContains(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1), map[string]string{
			"rsg.io/group-id": "1",
			"group_id":        "1",
			"app":             appName,
		})
		VerifyLWSTemplateLabelsContain(dynamicClient, ctx, RSGTestNamespace, groupName+"-"+decodeTargetName+"-1", map[string]string{
			"rsg.io/group-id": "1",
			"group_id":        "1",
			"app":             appName,
		})
		VerifyServiceSelectorContains(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1), map[string]string{
			"rsg.io/group-id": "1",
			"group_id":        "1",
			"app":             appName,
		})
		VerifyRSGStatusSelectorContains(dynamicClient, ctx, rsgName, []string{
			"app=" + appName,
			RSGWebhookTestLabelKey + "=" + RSGWebhookTestLabelValue,
		})
	})

	It("49-lws-replica-configmap-controls-new-group-replicas", Label("groupreplication"), Label("lws"), func() {
		rsgName, groupName, _, targetResources, _ := prepareRSGGroupReplicationLWSTargets(49)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 1, targetResources))).NotTo(HaveOccurred())
		rsgUID := GetRSGUID(dynamicClient, ctx, rsgName)

		configMapName := rsgName + "-replica-config"
		VerifyRSGReplicaConfigMapContent(clientset, ctx, RSGTestNamespace, configMapName, rsgName, rsgUID, map[string]string{
			"prefill": "1",
			"decode":  "1",
		})
		Expect(UpdateRSGReplicaConfigMapData(clientset, ctx, RSGTestNamespace, configMapName, map[string]string{
			"prefill": "3",
			"decode":  "5",
		})).NotTo(HaveOccurred())
		VerifyRSGReplicaConfigMapContent(clientset, ctx, RSGTestNamespace, configMapName, rsgName, rsgUID, map[string]string{
			"prefill": "3",
			"decode":  "5",
		})

		Expect(updateRSGGroupReplicasWithReadyCheck(rsgName, 2)).NotTo(HaveOccurred())

		VerifyLWSReplicas(dynamicClient, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[0], 1), 3)
		VerifyLWSReplicas(dynamicClient, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[1], 1), 5)
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 2, 2)
	})

	It("50-cross-namespace-lws-target-replication-converges", Label("groupreplication"), Label("lws"), func() {
		targetNamespace := "rsg-target-50"
		Expect(EnsureNamespace(clientset, ctx, targetNamespace)).NotTo(HaveOccurred())
		CleanupRSGBusinessResourcesInNamespace(clientset, ctx, targetNamespace)
		CleanupRSGLWSResources(clientset, dynamicClient, ctx, targetNamespace)
		WaitForNoRSGTestPods(clientset, ctx, targetNamespace)
		DeferCleanup(func() {
			CleanupRSGBusinessResourcesInNamespace(clientset, ctx, targetNamespace)
			CleanupRSGLWSResources(clientset, dynamicClient, ctx, targetNamespace)
			WaitForNoRSGTestPods(clientset, ctx, targetNamespace)
		})

		rsgName, groupName, appName, targetResources, serviceNames := prepareRSGGroupReplicationLWSTargetsInNamespace(50, targetNamespace)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 1, targetResources))).NotTo(HaveOccurred())
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 1, 1)

		Expect(updateRSGGroupReplicasWithReadyCheck(rsgName, 2)).NotTo(HaveOccurred())

		prefillClone := groupCloneDeploymentName(groupName, targetResources[0], 1)
		decodeClone := groupCloneDeploymentName(groupName, targetResources[1], 1)
		VerifyLWSExists(dynamicClient, ctx, RSGTestNamespace, prefillClone)
		VerifyLWSExists(dynamicClient, ctx, RSGTestNamespace, decodeClone)
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyLWSDeleted(dynamicClient, ctx, targetNamespace, prefillClone)
		VerifyLWSDeleted(dynamicClient, ctx, targetNamespace, decodeClone)
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 2, 2)
		VerifyRSGStatusSelectorContains(dynamicClient, ctx, rsgName, []string{
			"app=" + appName,
			RSGWebhookTestLabelKey + "=" + RSGWebhookTestLabelValue,
		})
	})

	It("51-lws-inplace-scaling-expands-followers-by-ratio", Label("inplacescaling"), Label("lws"), func() {
		rsgName := "rsg-e2e-51"
		leaderName := "rsg-51-leader"
		followerName := "rsg-51-follower"

		Expect(CreateRSGTestLeaderWorkerSet(dynamicClient, ctx, leaderName, RSGTestNamespace, "rsg-app-51", "leader", 1)).NotTo(HaveOccurred())
		Expect(CreateRSGTestLeaderWorkerSet(dynamicClient, ctx, followerName, RSGTestNamespace, "rsg-app-51", "follower", 1)).NotTo(HaveOccurred())

		targetResources := []map[string]any{
			NewTargetResource("leader", LeaderWorkerSetAPIVersion, "LeaderWorkerSet", leaderName, ""),
			NewTargetResource("follower", LeaderWorkerSetAPIVersion, "LeaderWorkerSet", followerName, ""),
		}
		Expect(CreateRSG(dynamicClient, ctx, newInplaceScalingRSGWithTargets(rsgName, targetResources, 2, 3))).NotTo(HaveOccurred())

		VerifyLWSReplicas(dynamicClient, ctx, RSGTestNamespace, leaderName, 2)
		VerifyLWSReplicas(dynamicClient, ctx, RSGTestNamespace, followerName, 6)
		VerifyRSGReadyCondition(dynamicClient, ctx, rsgName, "True", "Reconciled")
	})

	It("52-scale-down-lws-to-zero-shrinks-group0", Label("groupreplication"), Label("lws"), Label("scaledown"), func() {
		rsgName, groupName, _, targetResources, serviceNames := prepareRSGGroupReplicationLWSTargets(52)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 2, targetResources))).NotTo(HaveOccurred())

		prefillClone := groupCloneDeploymentName(groupName, targetResources[0], 1)
		decodeClone := groupCloneDeploymentName(groupName, targetResources[1], 1)
		VerifyLWSExists(dynamicClient, ctx, RSGTestNamespace, prefillClone)
		VerifyLWSExists(dynamicClient, ctx, RSGTestNamespace, decodeClone)
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 2, 2)

		Expect(updateRSGGroupReplicasWithReadyCheck(rsgName, 0)).NotTo(HaveOccurred())

		VerifyLWSReplicas(dynamicClient, ctx, RSGTestNamespace, targetResourceRefName(targetResources[0]), 0)
		VerifyLWSReplicas(dynamicClient, ctx, RSGTestNamespace, targetResourceRefName(targetResources[1]), 0)
		VerifyLWSDeleted(dynamicClient, ctx, RSGTestNamespace, prefillClone)
		VerifyLWSDeleted(dynamicClient, ctx, RSGTestNamespace, decodeClone)
		VerifyServiceDeleted(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServiceDeleted(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 0, 0)
	})

	It("53-scale-lws-from-zero-restores-group0-before-new-groups", Label("groupreplication"), Label("lws"), Label("scaledown"), func() {
		logStart := time.Now()
		rsgName, groupName, _, targetResources, serviceNames := prepareRSGGroupReplicationLWSTargets(53)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 2, targetResources))).NotTo(HaveOccurred())

		prefillTargetName := targetResourceRefName(targetResources[0])
		decodeTargetName := targetResourceRefName(targetResources[1])
		prefillClone := groupCloneDeploymentName(groupName, targetResources[0], 1)
		decodeClone := groupCloneDeploymentName(groupName, targetResources[1], 1)
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 2, 2)

		Expect(updateRSGGroupReplicasWithReadyCheck(rsgName, 0)).NotTo(HaveOccurred())
		VerifyLWSReplicas(dynamicClient, ctx, RSGTestNamespace, prefillTargetName, 0)
		VerifyLWSReplicas(dynamicClient, ctx, RSGTestNamespace, decodeTargetName, 0)
		VerifyLWSDeleted(dynamicClient, ctx, RSGTestNamespace, prefillClone)
		VerifyLWSDeleted(dynamicClient, ctx, RSGTestNamespace, decodeClone)
		VerifyServiceDeleted(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServiceDeleted(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 0, 0)

		observedRestoreBeforeClone := false
		Expect(updateRSGGroupReplicasWithReadyCheck(rsgName, 2)).NotTo(HaveOccurred())

		Eventually(func() bool {
			targetObj, err := dynamicClient.Resource(LeaderWorkerSetGVR).Namespace(RSGTestNamespace).Get(ctx, prefillTargetName, metav1.GetOptions{})
			if err != nil {
				return false
			}
			replicas, found, err := unstructured.NestedInt64(targetObj.Object, "spec", "replicas")
			if err != nil || !found {
				return false
			}
			decodeObj, err := dynamicClient.Resource(LeaderWorkerSetGVR).Namespace(RSGTestNamespace).Get(ctx, decodeTargetName, metav1.GetOptions{})
			if err != nil {
				return false
			}
			decodeReplicas, found, err := unstructured.NestedInt64(decodeObj.Object, "spec", "replicas")
			if err != nil || !found {
				return false
			}
			restored := replicas == 1 && decodeReplicas == 1

			_, err = dynamicClient.Resource(LeaderWorkerSetGVR).Namespace(RSGTestNamespace).Get(ctx, prefillClone, metav1.GetOptions{})
			prefillCloneExists := err == nil
			_, err = dynamicClient.Resource(LeaderWorkerSetGVR).Namespace(RSGTestNamespace).Get(ctx, decodeClone, metav1.GetOptions{})
			decodeCloneExists := err == nil
			if restored && !prefillCloneExists && !decodeCloneExists {
				observedRestoreBeforeClone = true
			}
			return restored && prefillCloneExists && decodeCloneExists
		}, 2*time.Minute, 500*time.Millisecond).Should(BeTrue())

		if !observedRestoreBeforeClone {
			GinkgoWriter.Printf("did not directly observe restore-before-clone window for %s; rely on controller log and final state verification\n", rsgName)
		}
		VerifyLWSReplicas(dynamicClient, ctx, RSGTestNamespace, prefillTargetName, 1)
		VerifyLWSReplicas(dynamicClient, ctx, RSGTestNamespace, decodeTargetName, 1)
		VerifyLWSReplicas(dynamicClient, ctx, RSGTestNamespace, prefillClone, 1)
		VerifyLWSReplicas(dynamicClient, ctx, RSGTestNamespace, decodeClone, 1)
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 2, 2)
		VerifyRSGControllerLogsContainForRSG(clientset, ctx, logStart, rsgName, "group0 restored, requeue before creating replicas")
	})

	It("56-lws-scale-down-falls-back-to-max-group-id-when-no-metric-or-annotation", Label("groupreplication"), Label("lws"), Label("scaledown"), func() {
		logStart := time.Now()
		rsgName, groupName, _, targetResources, serviceNames := prepareRSGGroupReplicationLWSTargets(56)
		Expect(CreateRSG(dynamicClient, ctx, newGroupReplicationRSGWithTargets(rsgName, groupName, 3, targetResources))).NotTo(HaveOccurred())

		VerifyLWSExists(dynamicClient, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[0], 1))
		VerifyLWSExists(dynamicClient, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[0], 2))
		VerifyLWSExists(dynamicClient, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[1], 1))
		VerifyLWSExists(dynamicClient, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[1], 2))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 2))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 2))
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 3, 3)

		Expect(updateRSGGroupReplicasWithReadyCheck(rsgName, 2)).NotTo(HaveOccurred())

		VerifyLWSExists(dynamicClient, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[0], 1))
		VerifyLWSExists(dynamicClient, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[1], 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 1))
		VerifyServiceExists(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 1))
		VerifyLWSDeleted(dynamicClient, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[0], 2))
		VerifyLWSDeleted(dynamicClient, ctx, RSGTestNamespace, groupCloneDeploymentName(groupName, targetResources[1], 2))
		VerifyServiceDeleted(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[0], groupName, 2))
		VerifyServiceDeleted(clientset, ctx, RSGTestNamespace, groupCloneServiceName(serviceNames[1], groupName, 2))
		VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 2, 2)
		VerifyRSGControllerLogsContainForRSG(clientset, ctx, logStart, rsgName, "fallback to max group-id")
	})

	AfterAll(func() {
		CleanupRSGWebhookTestResources(dynamicClient, ctx)
		CleanupRSGBusinessResources(clientset, ctx)
		CleanupRSGLWSResources(clientset, dynamicClient, ctx, RSGTestNamespace)
		WaitForNoRSGTestPods(clientset, ctx, RSGTestNamespace)
		Expect(HelmUninstall(RSGHelmReleaseName, RSGTestNamespace)).NotTo(HaveOccurred())
		Expect(CleanupLeaderWorkerSetController(clientset, ctx)).NotTo(HaveOccurred())
		time.Sleep(5 * time.Second)
		Expect(DeleteNamespaceIfExists(clientset, ctx, RSGTestNamespace)).NotTo(HaveOccurred())
	})
})

func updateRSGGroupReplicasWithReadyCheck(name string, replicas int64) error {
	var lastErr error
	deadline := time.Now().Add(2 * time.Minute)

	for {
		VerifyResourceScalingGroupControllerReady(clientset, ctx)
		lastErr = UpdateRSGGroupReplicas(dynamicClient, ctx, name, replicas)
		if lastErr == nil {
			return nil
		}
		if !isTransientWebhookConnectionError(lastErr) || time.Now().After(deadline) {
			return lastErr
		}
		GinkgoWriter.Printf("retrying replica update after rechecking controller readiness: %v\n", lastErr)
		time.Sleep(2 * time.Second)
	}
}

func isTransientWebhookConnectionError(err error) bool {
	if err == nil {
		return false
	}
	message := err.Error()
	if !strings.Contains(message, "failed calling webhook") {
		return false
	}
	return strings.Contains(message, "connect: connection refused") ||
		strings.Contains(message, "no endpoints available") ||
		strings.Contains(message, "context deadline exceeded") ||
		strings.Contains(message, "i/o timeout") ||
		strings.Contains(message, "EOF")
}

func prepareRSGGroupReplicationTargets(caseID int) (string, string, string, []map[string]any, []string) {
	return prepareRSGGroupReplicationTargetsInNamespace(caseID, RSGTestNamespace)
}

func prepareRSGGroupReplicationTargetsInNamespace(caseID int, namespace string) (string, string, string, []map[string]any, []string) {
	rsgName := fmt.Sprintf("rsg-e2e-%02d", caseID)
	groupName := fmt.Sprintf("pd%02d", caseID)
	appName := fmt.Sprintf("rsg-app-%02d", caseID)
	prefillDeploymentName := fmt.Sprintf("rsg-%02d-prefill", caseID)
	decodeDeploymentName := fmt.Sprintf("rsg-%02d-decode", caseID)
	prefillServiceName := fmt.Sprintf("rsg-%02d-prefill-svc", caseID)
	decodeServiceName := fmt.Sprintf("rsg-%02d-decode-svc", caseID)

	Expect(CreateRSGTestDeployment(clientset, ctx, prefillDeploymentName, namespace, appName, "prefill", 1)).NotTo(HaveOccurred())
	Expect(CreateRSGTestDeployment(clientset, ctx, decodeDeploymentName, namespace, appName, "decode", 1)).NotTo(HaveOccurred())
	Expect(CreateRSGTestService(clientset, ctx, prefillServiceName, namespace, appName, "prefill")).NotTo(HaveOccurred())
	Expect(CreateRSGTestService(clientset, ctx, decodeServiceName, namespace, appName, "decode")).NotTo(HaveOccurred())
	VerifyDeploymentExists(clientset, ctx, namespace, prefillDeploymentName)
	VerifyDeploymentExists(clientset, ctx, namespace, decodeDeploymentName)
	VerifyServiceExists(clientset, ctx, namespace, prefillServiceName)
	VerifyServiceExists(clientset, ctx, namespace, decodeServiceName)

	targetResources := []map[string]any{
		NewTargetResource("prefill", "apps/v1", "Deployment", prefillDeploymentName, namespace),
		NewTargetResource("decode", "apps/v1", "Deployment", decodeDeploymentName, namespace),
	}
	return rsgName, groupName, appName, targetResources, []string{prefillServiceName, decodeServiceName}
}

func prepareRSGGroupReplicationStatefulSetTargets(caseID int) (string, string, string, []map[string]any, []string) {
	return prepareRSGGroupReplicationStatefulSetTargetsInNamespace(caseID, RSGTestNamespace)
}

func prepareRSGGroupReplicationStatefulSetTargetsInNamespace(caseID int, namespace string) (string, string, string, []map[string]any, []string) {
	rsgName := fmt.Sprintf("rsg-e2e-%02d", caseID)
	groupName := fmt.Sprintf("pd%02d", caseID)
	appName := fmt.Sprintf("rsg-app-%02d", caseID)
	prefillStatefulSetName := fmt.Sprintf("rsg-%02d-prefill", caseID)
	decodeStatefulSetName := fmt.Sprintf("rsg-%02d-decode", caseID)
	prefillServiceName := fmt.Sprintf("rsg-%02d-prefill-headless", caseID)
	decodeServiceName := fmt.Sprintf("rsg-%02d-decode-headless", caseID)

	Expect(CreateRSGTestHeadlessService(clientset, ctx, prefillServiceName, namespace, appName, "prefill")).NotTo(HaveOccurred())
	Expect(CreateRSGTestHeadlessService(clientset, ctx, decodeServiceName, namespace, appName, "decode")).NotTo(HaveOccurred())
	Expect(CreateRSGTestStatefulSet(clientset, ctx, prefillStatefulSetName, namespace, appName, "prefill", prefillServiceName, 1)).NotTo(HaveOccurred())
	Expect(CreateRSGTestStatefulSet(clientset, ctx, decodeStatefulSetName, namespace, appName, "decode", decodeServiceName, 1)).NotTo(HaveOccurred())
	VerifyStatefulSetExists(clientset, ctx, namespace, prefillStatefulSetName)
	VerifyStatefulSetExists(clientset, ctx, namespace, decodeStatefulSetName)
	VerifyServiceExists(clientset, ctx, namespace, prefillServiceName)
	VerifyServiceExists(clientset, ctx, namespace, decodeServiceName)

	targetResources := []map[string]any{
		NewTargetResource("prefill", "apps/v1", "StatefulSet", prefillStatefulSetName, namespace),
		NewTargetResource("decode", "apps/v1", "StatefulSet", decodeStatefulSetName, namespace),
	}
	return rsgName, groupName, appName, targetResources, []string{prefillServiceName, decodeServiceName}
}

func prepareRSGGroupReplicationLWSTargets(caseID int) (string, string, string, []map[string]any, []string) {
	return prepareRSGGroupReplicationLWSTargetsInNamespace(caseID, RSGTestNamespace)
}

func prepareRSGGroupReplicationLWSTargetsInNamespace(caseID int, namespace string) (string, string, string, []map[string]any, []string) {
	rsgName := fmt.Sprintf("rsg-e2e-%02d", caseID)
	groupName := fmt.Sprintf("pd%02d", caseID)
	appName := fmt.Sprintf("rsg-app-%02d", caseID)
	prefillLWSName := fmt.Sprintf("rsg-%02d-prefill", caseID)
	decodeLWSName := fmt.Sprintf("rsg-%02d-decode", caseID)
	prefillServiceName := fmt.Sprintf("rsg-%02d-prefill-svc", caseID)
	decodeServiceName := fmt.Sprintf("rsg-%02d-decode-svc", caseID)

	Expect(CreateRSGTestService(clientset, ctx, prefillServiceName, namespace, appName, "prefill")).NotTo(HaveOccurred())
	Expect(CreateRSGTestService(clientset, ctx, decodeServiceName, namespace, appName, "decode")).NotTo(HaveOccurred())
	Expect(CreateRSGTestLeaderWorkerSet(dynamicClient, ctx, prefillLWSName, namespace, appName, "prefill", 1)).NotTo(HaveOccurred())
	Expect(CreateRSGTestLeaderWorkerSet(dynamicClient, ctx, decodeLWSName, namespace, appName, "decode", 1)).NotTo(HaveOccurred())
	VerifyLWSExists(dynamicClient, ctx, namespace, prefillLWSName)
	VerifyLWSExists(dynamicClient, ctx, namespace, decodeLWSName)
	VerifyServiceExists(clientset, ctx, namespace, prefillServiceName)
	VerifyServiceExists(clientset, ctx, namespace, decodeServiceName)

	targetResources := []map[string]any{
		NewTargetResource("prefill", LeaderWorkerSetAPIVersion, "LeaderWorkerSet", prefillLWSName, namespace),
		NewTargetResource("decode", LeaderWorkerSetAPIVersion, "LeaderWorkerSet", decodeLWSName, namespace),
	}
	return rsgName, groupName, appName, targetResources, []string{prefillServiceName, decodeServiceName}
}

func newGroupReplicationRSGWithTargets(name, groupName string, replicas int64, targetResources []map[string]any) *unstructured.Unstructured {
	items := make([]any, 0, len(targetResources))
	for _, item := range targetResources {
		items = append(items, item)
	}
	return NewResourceScalingGroup(name, map[string]any{
		"targetResources": items,
		"scalingStrategy": map[string]any{
			"type": "GroupReplication",
			"groupConfig": map[string]any{
				"groupName": groupName,
				"replicas":  replicas,
			},
		},
	})
}

func newGroupReplicationRSGWithScaleDown(name, groupName string, replicas int64, targetResources []map[string]any, order string) *unstructured.Unstructured {
	items := make([]any, 0, len(targetResources))
	for _, item := range targetResources {
		items = append(items, item)
	}
	return NewResourceScalingGroup(name, map[string]any{
		"targetResources": items,
		"scalingStrategy": map[string]any{
			"type": "GroupReplication",
			"groupConfig": map[string]any{
				"groupName": groupName,
				"replicas":  replicas,
				"scaleDown": map[string]any{
					"metric": map[string]any{
						"expr":       "rsg_e2e_metric",
						"type":       "Gauge",
						"aggregator": "sum",
					},
					"order": order,
				},
			},
		},
	})
}

func newInplaceScalingRSG(name, leaderDeploymentName, followerDeploymentName string, leaderReplicas, followRatio int64) *unstructured.Unstructured {
	targets := []map[string]any{
		NewTargetResource("leader", "apps/v1", "Deployment", leaderDeploymentName, ""),
		NewTargetResource("follower", "apps/v1", "Deployment", followerDeploymentName, ""),
	}
	return newInplaceScalingRSGWithTargets(name, targets, leaderReplicas, followRatio)
}

func newInplaceScalingRSGWithTargets(name string, targetResources []map[string]any, leaderReplicas, followRatio int64) *unstructured.Unstructured {
	items := make([]any, 0, len(targetResources))
	for _, item := range targetResources {
		items = append(items, item)
	}
	return NewResourceScalingGroup(name, map[string]any{
		"targetResources": items,
		"scalingStrategy": map[string]any{
			"type": "InplaceScaling",
			"leaderFollowerConfig": map[string]any{
				"leader": map[string]any{
					"resourceName": "leader",
					"replicas":     leaderReplicas,
				},
				"follower": []any{
					map[string]any{
						"resourceName": "follower",
						"followRatio":  followRatio,
					},
				},
			},
		},
	})
}

func targetResourceRefName(target map[string]any) string {
	return target["resourceRef"].(map[string]any)["name"].(string)
}

func groupCloneDeploymentName(groupName string, target map[string]any, groupID int) string {
	return fmt.Sprintf("%s-%s-%d", groupName, targetResourceRefName(target), groupID)
}

func groupCloneServiceName(serviceName, groupName string, groupID int) string {
	return fmt.Sprintf("%s-%s-%d", serviceName, groupName, groupID)
}
