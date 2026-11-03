package colocation

import (
	"context"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"gopkg.in/yaml.v3"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/colocation/utils"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
)

var _ = SIGDescribe("混部 Pod 抢占测试", Label("colocation-pod-preemption", "skip-temporarily"), func() {
	var client *k8s.K8SClient
	var ctx context.Context
	var originalConfig string

	BeforeEach(func() {
		// 注意：抢占测试需要集群资源紧张才能触发抢占行为，在资源充足的自动化环境中不稳定
		// 如需启用，请移除下面的 Skip 并确保测试环境资源受限
		Skip("抢占测试依赖集群资源紧张状态，在自动化环境中不稳定，已跳过")

		var err error
		client, err = k8s.NewK8SClientFromLocalKubeconfig()
		Expect(err).NotTo(HaveOccurred())
		ctx = context.Background()

		if !CheckVolcanoExists(client, ctx) {
			Skip("Volcano 未安装，跳过此测试")
		}

		// 等待 Volcano webhook 就绪
		Expect(WaitVolcanoWebhookReady(client, ctx, 30*time.Second)).To(Succeed())

		cm, err := GetVolcanoConfigMap(client, ctx)
		Expect(err).NotTo(HaveOccurred())
		originalConfig = cm.Data["volcano-scheduler.conf"]

		cm2, err := GetVolcanoConfigMap(client, ctx)
		Expect(err).NotTo(HaveOccurred())

		var config map[string]interface{}
		err = yaml.Unmarshal([]byte(cm2.Data["volcano-scheduler.conf"]), &config)
		Expect(err).NotTo(HaveOccurred())

		tiers, ok := config["tiers"].([]interface{})
		Expect(ok).To(BeTrue())
		hasPriority := false
		for _, tier := range tiers {
			tierMap, ok := tier.(map[string]interface{})
			if !ok {
				continue
			}
			plugins, ok := tierMap["plugins"].([]interface{})
			if !ok {
				continue
			}
			for _, plugin := range plugins {
				pluginMap, ok := plugin.(map[string]interface{})
				if !ok {
					continue
				}
				if name, ok := pluginMap["name"].(string); ok && name == "priority" {
					hasPriority = true
					break
				}
			}
		}
		Expect(hasPriority).To(BeTrue(), "priority 插件必须存在")

		actionsStr, ok := config["actions"].(string)
		if ok {
			actions := make(map[string]bool)
			for _, a := range splitAndTrim(actionsStr, ",") {
				actions[a] = true
			}
			delete(actions, "enqueue")
			actions["preempt"] = true

			var newActions []string
			for a := range actions {
				newActions = append(newActions, a)
			}
			config["actions"] = joinStrings(newActions, ", ")
		}

		newConfig, err := yaml.Marshal(config)
		Expect(err).NotTo(HaveOccurred())

		err = UpdateVolcanoConfigMap(client, ctx, string(newConfig))
		Expect(err).NotTo(HaveOccurred())
	})

	AfterEach(func() {
		if client != nil && originalConfig != "" {
			_ = UpdateVolcanoConfigMap(client, ctx, originalConfig)
		}
	})

	Context("Pod 抢占测试", func() {
		It("测试 HLS 抢占 LS - HLS Pod 应该抢占低优先级的 LS Pod", func() {
			var lsPod1Name, lsPod1Ns string
			var lsPod2Name, lsPod2Ns string
			var hlsPodName, hlsPodNs string

			defer func() {
				if client == nil {
					return
				}
				if lsPod1Name != "" {
					_ = DeletePod(client, ctx, lsPod1Name, lsPod1Ns, 30*time.Second)
				}
				if lsPod2Name != "" {
					_ = DeletePod(client, ctx, lsPod2Name, lsPod2Ns, 30*time.Second)
				}
				if hlsPodName != "" {
					_ = DeletePod(client, ctx, hlsPodName, hlsPodNs, 30*time.Second)
				}
			}()

			lsPod1, err := InstallLSPodNamed(client, ctx, "ls-pod-1", "", "")
			Expect(err).NotTo(HaveOccurred())
			lsPod1Name, lsPod1Ns = lsPod1.Name, lsPod1.Namespace
			Expect(WaitPodCondition(client, ctx, lsPod1Name, lsPod1Ns, "Running", 10*time.Second)).To(BeTrue())

			lsPod2, err := InstallLSPodNamed(client, ctx, "ls-pod-2", "", "")
			Expect(err).NotTo(HaveOccurred())
			lsPod2Name, lsPod2Ns = lsPod2.Name, lsPod2.Namespace
			Expect(WaitPodCondition(client, ctx, lsPod2Name, lsPod2Ns, "Pending", 10*time.Second)).To(BeTrue())

			hlsPod, err := InstallHLSPod(client, ctx, "", "")
			Expect(err).NotTo(HaveOccurred())
			hlsPodName, hlsPodNs = hlsPod.Name, hlsPod.Namespace
			Expect(WaitPodCondition(client, ctx, hlsPodName, hlsPodNs, "Running", 10*time.Second)).To(BeTrue())

			ls1Exists := PodExists(client, ctx, lsPod1Name, lsPod1Ns)
			ls2Exists := PodExists(client, ctx, lsPod2Name, lsPod2Ns)
			Expect(ls1Exists && ls2Exists).To(BeFalse(), "应该至少有一个 LS Pod 被抢占删除")
		})

		It("测试 LS 抢占 BE - LS Pod 应该抢占低优先级的 BE Pod", func() {
			var bePod1Name, bePod1Ns string
			var bePod2Name, bePod2Ns string
			var lsPodName, lsPodNs string

			defer func() {
				if client == nil {
					return
				}
				if bePod1Name != "" {
					_ = DeletePod(client, ctx, bePod1Name, bePod1Ns, 30*time.Second)
				}
				if bePod2Name != "" {
					_ = DeletePod(client, ctx, bePod2Name, bePod2Ns, 30*time.Second)
				}
				if lsPodName != "" {
					_ = DeletePod(client, ctx, lsPodName, lsPodNs, 30*time.Second)
				}
			}()

			bePod1, err := InstallBEPodNamed(client, ctx, "be-pod-1", "", "")
			Expect(err).NotTo(HaveOccurred())
			bePod1Name, bePod1Ns = bePod1.Name, bePod1.Namespace
			Expect(WaitPodCondition(client, ctx, bePod1Name, bePod1Ns, "Running", 10*time.Second)).To(BeTrue())

			bePod2, err := InstallBEPodNamed(client, ctx, "be-pod-2", "", "")
			Expect(err).NotTo(HaveOccurred())
			bePod2Name, bePod2Ns = bePod2.Name, bePod2.Namespace
			Expect(WaitPodCondition(client, ctx, bePod2Name, bePod2Ns, "Pending", 10*time.Second)).To(BeTrue())

			lsPod, err := InstallLSPod(client, ctx, "", "")
			Expect(err).NotTo(HaveOccurred())
			lsPodName, lsPodNs = lsPod.Name, lsPod.Namespace
			Expect(WaitPodCondition(client, ctx, lsPodName, lsPodNs, "Running", 10*time.Second)).To(BeTrue())

			be1Exists := PodExists(client, ctx, bePod1Name, bePod1Ns)
			be2Exists := PodExists(client, ctx, bePod2Name, bePod2Ns)
			Expect(be1Exists && be2Exists).To(BeFalse(), "应该至少有一个 BE Pod 被抢占删除")
		})
	})
})

func splitAndTrim(s, sep string) []string {
	parts := strings.Split(s, sep)
	var result []string
	for _, p := range parts {
		trimmed := strings.TrimSpace(p)
		if trimmed != "" {
			result = append(result, trimmed)
		}
	}
	return result
}

func joinStrings(strs []string, sep string) string {
	return strings.Join(strs, sep)
}
