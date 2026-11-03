package declarative_upgrade

import (
	"fmt"
	"net"
	"strconv"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
)

const (
	upgradePathCRName              = "openfuyao-upgrade-paths"
	upgradePathOCIDigestAnnotation = "config.openfuyao.com/oci-digest"
	defaultOnlineUpgradePathRef    = "cr.openfuyao.cn/openfuyao/upgrade-path:latest"
	upgradePathImageName           = "upgrade-path"
	upgradePathImageTag            = "latest"
	upgradePathSyncTimeout         = 15 * time.Minute
	upgradePathSyncInterval        = 30 * time.Second

	duplicateUpgradePathCRName      = "duplicate-upgrade-paths"
	upgradePathWebhookDeniedMsg     = "only one UpgradePath CR is allowed per cluster"
	upgradePathDigestSkipPullLogMsg = "OCI digest unchanged, skip pulling image"
	defaultDigestCheckInterval      = 5 * time.Minute
)

var _ = Describe("UpgradePath OCI 镜像地址解析与拉取", Label("upgradepath"), func() {
	var (
		sshExecutor *executor.SSHExecutor
		guideNode   *config.GuideNodeConfig
	)

	BeforeEach(func() {
		guideNode = config.LoadGuideNodeFromEnv()
		var err error
		sshExecutor, err = executor.NewSSHExecutor(guideNode.Host, guideNode.Port, guideNode.Username, guideNode.Password)
		Expect(err).NotTo(HaveOccurred(), "连接引导节点失败")
	})

	// 用例名称：离线默认镜像源下 upgrade-path OCI 地址拼接与拉取
	// 用例步骤：1) 从 bke-config 推导 OCI 地址并校验 UpgradePath CR 与 capbke 日志。
	// 预期结果：UpgradePath CR 存在，oci-digest 注解非空，capbke 日志含期望的 kubernetes/upgrade-path OCI 地址。
	// Author: luolong817
	It("离线默认镜像源应正确拼接并拉取 upgrade-path OCI 镜像", Label("required_nodes:0", "post-init", "offline", "offline-default"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {

		checkBootstrapPodsReady(sshExecutor)

		domain := getBKEConfigValue(sshExecutor, "domain")
		host := getBKEConfigValue(sshExecutor, "host")
		port := getBKEConfigValue(sshExecutor, "imageRepoPort")
		Expect(domain).NotTo(BeEmpty(), "bke-config 应包含 domain")
		Expect(host).NotTo(BeEmpty(), "bke-config 应包含 host")

		expectedRefs := buildOfflineUpgradePathOCIRefs(domain, host, port)
		GinkgoWriter.Printf("离线模式期望 upgrade-path OCI 引用: %v\n", expectedRefs)

		waitAndVerifyUpgradePathSynced(sshExecutor, expectedRefs)
	})

	// 用例名称：otherRepo 镜像源下 upgrade-path OCI 地址拼接与拉取
	// 用例步骤：1) 校验 bke-config 与 UpgradePath 同步结果。
	// 预期结果：UpgradePath CR 存在，oci-digest 注解非空，capbke 日志含 otherRepo 拼接的 upgrade-path OCI 地址。
	// Author: luolong817
	It("otherRepo 镜像源应正确拼接并拉取 upgrade-path OCI 镜像", Label("required_nodes:0", "post-init", "custom-repo"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {

		checkBootstrapPodsReady(sshExecutor)

		expectedRefs := buildOtherRepoUpgradePathOCIRefs(getBKEConfigValue(sshExecutor, "otherRepo"), getBKEConfigValue(sshExecutor, "otherRepoIp"))
		GinkgoWriter.Printf("otherRepo 模式期望 upgrade-path OCI 引用: %v\n", expectedRefs)

		waitAndVerifyUpgradePathSynced(sshExecutor, expectedRefs)
	})

	// 用例名称：onlineImage 镜像源下 upgrade-path OCI 地址回退与拉取
	// 用例步骤：1) bke reset；2) bke init --onlineImage；3) 校验回退至默认 upgrade-path OCI 引用并成功同步 CR。
	// 预期结果：UpgradePath CR 存在，oci-digest 注解非空，capbke 日志含 cr.openfuyao.cn/openfuyao/upgrade-path:latest。
	// Author: luolong817
	It("onlineImage 镜像源应回退默认地址并拉取 upgrade-path OCI 镜像", Label("required_nodes:0", "post-init", "online"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {

		checkBootstrapPodsReady(sshExecutor)

		expectedRefs := []string{defaultOnlineUpgradePathRef}
		GinkgoWriter.Printf("onlineImage 模式期望 upgrade-path OCI 引用: %v\n", expectedRefs)

		waitAndVerifyUpgradePathSynced(sshExecutor, expectedRefs)
	})
})

var _ = Describe("UpgradePath 生命周期与 Webhook", Label("upgradepath"), func() {
	var (
		sshExecutor *executor.SSHExecutor
		guideNode   *config.GuideNodeConfig
	)

	BeforeEach(func() {
		guideNode = config.LoadGuideNodeFromEnv()
		var err error
		sshExecutor, err = executor.NewSSHExecutor(guideNode.Host, guideNode.Port, guideNode.Username, guideNode.Password)
		Expect(err).NotTo(HaveOccurred(), "连接引导节点失败")
	})

	// 用例名称：UpgradePath 删除重建、Webhook 单例拦截与 digest 未变跳过拉取
	// 用例步骤：
	//   1) Eventually 校验 UpgradePath CR 存在且 status.phase=Active
	//   2) 删除 CR，Eventually 等待 CR 被 DigestMonitor 重建
	//   3) 构造不同 name 的 CR yaml 并 kubectl apply，校验 Webhook 拒绝
	//   4) 等待第二次镜像同步周期，校验日志含 digest 未变跳过拉取
	// 预期结果：CR 恢复 Active；第二个 CR 创建失败；capbke 日志含 OCI digest unchanged, skip pulling image
	// Author: luolong817
	It("删除重建后 Webhook 应拒绝第二个 UpgradePath 且 digest 未变时跳过拉取",
		Label("required_nodes:0", "post-init", "upgradepath-lifecycle", "offline", "custom-repo", "online"),
		SpecTimeout(InstallationItTimeout),
		func(ctx SpecContext) {

			checkBootstrapPodsReady(sshExecutor)

			By("Eventually 校验 UpgradePath CR 存在且 status.phase=Active")
			waitAndVerifyUpgradePathActive(sshExecutor)

			By("删除 UpgradePath CR")
			deleteUpgradePathCR(sshExecutor)

			By("Eventually 等待 DigestMonitor 重建 UpgradePath CR")
			waitForUpgradePathCRRecreated(sshExecutor)

			By("再次确认重建后 CR 为 Active")
			waitAndVerifyUpgradePathActive(sshExecutor)

			By("构造不同 name 的 UpgradePath yaml 并 apply，期望 Webhook 拦截")
			yamlPath := writeDuplicateUpgradePathYAML(sshExecutor, duplicateUpgradePathCRName)
			defer cleanupRemoteUpgradePathYAML(sshExecutor, yamlPath)
			applyUpgradePathYAMLExpectWebhookDenied(sshExecutor, yamlPath, upgradePathCRName)

			By("确认集群内仍仅 1 个 UpgradePath CR")
			Expect(countUpgradePathCRs(sshExecutor)).To(Equal(1))

			By("等待第二次镜像同步周期，校验 digest 未变跳过拉取")
			logMarker := markCapbkeLogPosition(sshExecutor)
			waitForDigestUnchangedSkipPull(sshExecutor, logMarker)
		})
})

func checkBootstrapPodsReady(sshExecutor *executor.SSHExecutor) {
	By("检查引导集群 Pod 状态")
	Eventually(func() bool {
		cmd := "kubectl get pod -A --no-headers"
		res, err := sshExecutor.Exec(cmd)
		if err != nil || res.ExitCode != 0 {
			return false
		}

		lines := strings.Split(strings.TrimSpace(res.Stdout), "\n")
		if len(lines) == 0 {
			return false
		}

		allReady := true
		for _, line := range lines {
			if strings.TrimSpace(line) == "" {
				continue
			}
			fields := strings.Fields(line)
			if len(fields) < 4 {
				continue
			}
			status := fields[3]
			if status != "Running" && status != "Completed" {
				GinkgoWriter.Printf("Pod 未就绪: %s (Status: %s)\n", fields[1], status)
				allReady = false
			}
		}
		return allReady
	}, 20*time.Minute, 10*time.Second).Should(BeTrue(), "引导集群所有 Pod 应该处于 Running 或 Completed 状态")
}

func getBKEConfigValue(sshExecutor *executor.SSHExecutor, key string) string {
	cmd := fmt.Sprintf(`kubectl get cm -n cluster-system bke-config -o jsonpath='{.data.%s}' 2>/dev/null`, key)
	res, err := sshExecutor.Exec(cmd)
	if err != nil || res == nil || res.ExitCode != 0 {
		return ""
	}
	return strings.TrimSpace(res.Stdout)
}

func buildOfflineUpgradePathOCIRefs(domain, host, port string) []string {
	const prefix = "kubernetes"
	return buildUpgradePathOCIRefs([]string{domain, host}, port, prefix)
}

func buildOtherRepoUpgradePathOCIRefs(otherRepo, otherRepoIP string) []string {
	trimmed := strings.TrimSuffix(strings.TrimSpace(otherRepo), "/")
	parts := strings.SplitN(trimmed, "/", 2)
	hostPort := parts[0]
	prefix := ""
	if len(parts) > 1 {
		prefix = strings.Trim(parts[1], "/")
	}

	port := "443"
	domain := hostPort
	if hp := strings.Split(hostPort, ":"); len(hp) == 2 {
		domain = hp[0]
		port = hp[1]
	}

	hosts := []string{domain}
	if ip := strings.TrimSpace(otherRepoIP); ip != "" {
		hosts = append(hosts, ip)
	}
	return buildUpgradePathOCIRefs(hosts, port, prefix)
}

func buildUpgradePathOCIRefs(hosts []string, port, prefix string) []string {
	refs := make([]string, 0, len(hosts))
	seen := make(map[string]struct{})
	for _, host := range hosts {
		host = strings.TrimSpace(host)
		if host == "" {
			continue
		}
		addr := host
		if port != "" && port != "443" {
			addr = net.JoinHostPort(host, port)
		}
		ref := fmt.Sprintf("%s/%s/%s:%s", addr, strings.Trim(prefix, "/"), upgradePathImageName, upgradePathImageTag)
		if _, ok := seen[ref]; ok {
			continue
		}
		seen[ref] = struct{}{}
		refs = append(refs, ref)
	}
	return refs
}

func waitAndVerifyUpgradePathSynced(sshExecutor *executor.SSHExecutor, expectedRefs []string) {
	By("等待 UpgradePath CR 出现")
	Eventually(func() bool {
		exists := upgradePathCRExists(sshExecutor)
		if !exists {
			GinkgoWriter.Printf("UpgradePath CR 尚未创建，继续等待...\n")
		}
		return exists
	}, upgradePathSyncTimeout, upgradePathSyncInterval).Should(
		BeTrue(),
		"kubectl get upgradepath -A 应能查到 %s", upgradePathCRName,
	)

	By("校验 UpgradePath oci-digest 注解非空")
	Eventually(func() string {
		digest := getUpgradePathOCIDigest(sshExecutor)
		if digest == "" {
			GinkgoWriter.Printf("UpgradePath CR 已存在，oci-digest 注解仍为空，继续等待...\n")
		} else {
			GinkgoWriter.Printf("UpgradePath CR oci-digest 注解已同步: %s\n", digest)
		}
		return digest
	}, upgradePathSyncTimeout, upgradePathSyncInterval).Should(
		Not(BeEmpty()),
		"UpgradePath 应包含非空 %s 注解", upgradePathOCIDigestAnnotation,
	)

	By("校验 capbke 日志包含期望的 upgrade-path OCI 引用")
	verifyUpgradePathOCIRefInLogs(sshExecutor, expectedRefs)
}

func upgradePathCRExists(sshExecutor *executor.SSHExecutor) bool {
	cmd := "kubectl get upgradepath -A --no-headers 2>/dev/null"
	res, err := sshExecutor.Exec(cmd)
	if err != nil || res == nil || res.ExitCode != 0 {
		return false
	}
	for _, line := range strings.Split(strings.TrimSpace(res.Stdout), "\n") {
		fields := strings.Fields(line)
		if len(fields) == 0 {
			continue
		}
		if fields[0] == upgradePathCRName {
			return true
		}
	}
	return false
}

func getUpgradePathOCIDigest(sshExecutor *executor.SSHExecutor) string {
	if !upgradePathCRExists(sshExecutor) {
		return ""
	}

	// 通过 yaml 解析注解，避免 jsonpath 在 SSH 远程执行时引号转义失败。
	yamlCmd := fmt.Sprintf("kubectl get upgradepath %s -o yaml 2>/dev/null", upgradePathCRName)
	yamlRes, err := sshExecutor.Exec(yamlCmd)
	if err != nil || yamlRes == nil || yamlRes.ExitCode != 0 {
		return ""
	}

	annotationKey := upgradePathOCIDigestAnnotation + ":"
	for _, line := range strings.Split(yamlRes.Stdout, "\n") {
		trimmed := strings.TrimSpace(line)
		if !strings.HasPrefix(trimmed, annotationKey) {
			continue
		}
		digest := strings.TrimSpace(strings.TrimPrefix(trimmed, annotationKey))
		digest = strings.Trim(digest, `"'`)
		return digest
	}
	return ""
}

func verifyUpgradePathOCIRefInLogs(sshExecutor *executor.SSHExecutor, expectedRefs []string) {
	logCmd := `kubectl logs -n cluster-system deployment/bke-controller-manager --tail=5000 2>/dev/null`
	res, err := sshExecutor.Exec(logCmd)
	Expect(err).NotTo(HaveOccurred())
	output := res.Stdout + res.Stderr

	matched := false
	matchedRef := ""
	for _, ref := range expectedRefs {
		if strings.Contains(output, ref) {
			matched = true
			matchedRef = ref
			break
		}
	}
	Expect(matched).To(BeTrue(), "capbke 日志应包含期望的 upgrade-path OCI 地址之一: %v", expectedRefs)
	GinkgoWriter.Printf("capbke 日志匹配到 OCI 引用: %s\n", matchedRef)
}

func getUpgradePathPhase(sshExecutor *executor.SSHExecutor) string {
	cmd := fmt.Sprintf(
		`kubectl get upgradepath %s -o yaml 2>/dev/null | awk '/^  phase:/{print $2; exit}'`,
		upgradePathCRName,
	)
	res, err := sshExecutor.Exec(cmd)
	if err != nil || res == nil || res.ExitCode != 0 {
		return ""
	}
	return strings.TrimSpace(res.Stdout)
}

func waitAndVerifyUpgradePathActive(sshExecutor *executor.SSHExecutor) {
	Eventually(func() bool {
		if !upgradePathCRExists(sshExecutor) {
			GinkgoWriter.Printf("UpgradePath CR 尚未创建，继续等待...\n")
			return false
		}
		phase := getUpgradePathPhase(sshExecutor)
		if phase != "Active" {
			GinkgoWriter.Printf("UpgradePath CR phase=%q，等待 Active...\n", phase)
			return false
		}
		return true
	}, upgradePathSyncTimeout, upgradePathSyncInterval).Should(
		BeTrue(),
		"UpgradePath CR %s 应存在且 status.phase=Active", upgradePathCRName,
	)
}

func deleteUpgradePathCR(sshExecutor *executor.SSHExecutor) {
	cmd := fmt.Sprintf("kubectl delete upgradepath %s --ignore-not-found", upgradePathCRName)
	res, err := sshExecutor.Exec(cmd)
	Expect(err).NotTo(HaveOccurred())
	Expect(res.ExitCode).To(Equal(0), "删除 UpgradePath CR 失败: %s", res.Stderr)

	Eventually(func() bool {
		return !upgradePathCRExists(sshExecutor)
	}, 2*time.Minute, 5*time.Second).Should(
		BeTrue(),
		"UpgradePath CR 应已被删除",
	)
}

func waitForUpgradePathCRRecreated(sshExecutor *executor.SSHExecutor) {
	Eventually(func() bool {
		exists := upgradePathCRExists(sshExecutor)
		if !exists {
			GinkgoWriter.Printf("等待 DigestMonitor 重建 UpgradePath CR...\n")
		}
		return exists
	}, upgradePathSyncTimeout, upgradePathSyncInterval).Should(
		BeTrue(),
		"删除后 DigestMonitor 应重建 UpgradePath CR %s", upgradePathCRName,
	)

	Eventually(func() string {
		return getUpgradePathOCIDigest(sshExecutor)
	}, upgradePathSyncTimeout, upgradePathSyncInterval).Should(
		Not(BeEmpty()),
		"重建后 UpgradePath 应包含非空 %s 注解", upgradePathOCIDigestAnnotation,
	)
}

func duplicateUpgradePathYAML(name string) string {
	return fmt.Sprintf(`apiVersion: config.openfuyao.com/v1alpha1
kind: UpgradePath
metadata:
  name: %s
spec:
  paths:
    - from: v26.03
      to: v26.06-rc.2
      blocked: false
      deprecated: false
      notes: "Recommended upgrade path"
      preCheck:
        - name: backup-etcd
          required: true
        - name: check-node-health
          required: true
        - name: check-resource-quota
          required: false
  versions:
    - version: v26.06-rc.2
      installable: true
      deprecated: false
      notes: "Release candidate version, for community validation only. Not recommended for new installations."
`, name)
}

func writeDuplicateUpgradePathYAML(sshExecutor *executor.SSHExecutor, name string) string {
	remotePath := fmt.Sprintf("/tmp/upgradepath-duplicate-%s.yaml", name)
	cmd := fmt.Sprintf("cat > %s <<'EOF'\n%sEOF", remotePath, duplicateUpgradePathYAML(name))
	res, err := sshExecutor.Exec(cmd)
	Expect(err).NotTo(HaveOccurred())
	Expect(res.ExitCode).To(Equal(0), "写入 duplicate UpgradePath yaml 失败: %s", res.Stderr)
	return remotePath
}

func cleanupRemoteUpgradePathYAML(sshExecutor *executor.SSHExecutor, path string) {
	_, _ = sshExecutor.Exec(fmt.Sprintf("rm -f %s", path))
}

func applyUpgradePathYAMLExpectWebhookDenied(
	sshExecutor *executor.SSHExecutor,
	yamlPath, existingCRName string,
) {
	cmd := fmt.Sprintf("kubectl apply -f %s 2>&1", yamlPath)
	res, err := sshExecutor.Exec(cmd)
	combined := ""
	if res != nil {
		combined = res.Stdout + res.Stderr
	}
	GinkgoWriter.Printf("kubectl apply duplicate UpgradePath 输出:\n%s\n", combined)

	Expect(err).To(HaveOccurred(), "第二个 UpgradePath CR 应被 admission 拒绝")
	Expect(res.ExitCode).NotTo(Equal(0))
	Expect(combined).To(SatisfyAny(
		ContainSubstring("admission webhook"),
		ContainSubstring("denied the request"),
		ContainSubstring("Error from server"),
	))
	Expect(combined).To(ContainSubstring(upgradePathWebhookDeniedMsg))
	Expect(combined).To(ContainSubstring(existingCRName))
}

func countUpgradePathCRs(sshExecutor *executor.SSHExecutor) int {
	res, err := sshExecutor.Exec("kubectl get upgradepath -A --no-headers 2>/dev/null")
	if err != nil || res == nil || res.ExitCode != 0 {
		return 0
	}
	lines := strings.Split(strings.TrimSpace(res.Stdout), "\n")
	if len(lines) == 1 && lines[0] == "" {
		return 0
	}
	return len(lines)
}

func markCapbkeLogPosition(sshExecutor *executor.SSHExecutor) string {
	cmd := `kubectl logs -n cluster-system deployment/bke-controller-manager --tail=1 2>/dev/null`
	res, err := sshExecutor.Exec(cmd)
	Expect(err).NotTo(HaveOccurred())
	return strings.TrimSpace(res.Stdout)
}

func resolveDigestCheckInterval(sshExecutor *executor.SSHExecutor) time.Duration {
	cmd := `kubectl get deployment -n cluster-system bke-controller-manager -o yaml 2>/dev/null | grep -E 'oci-digest-check-interval' | awk '{print $2}'`
	res, err := sshExecutor.Exec(cmd)
	if err != nil || res == nil || res.ExitCode != 0 {
		return defaultDigestCheckInterval
	}
	seconds, convErr := strconv.Atoi(strings.TrimSpace(res.Stdout))
	if convErr != nil || seconds <= 0 {
		return defaultDigestCheckInterval
	}
	return time.Duration(seconds) * time.Second
}

func waitForDigestUnchangedSkipPull(sshExecutor *executor.SSHExecutor, logMarker string) {
	interval := resolveDigestCheckInterval(sshExecutor)
	waitBudget := interval + 2*upgradePathSyncInterval

	Eventually(func() bool {
		logCmd := `kubectl logs -n cluster-system deployment/bke-controller-manager --tail=8000 2>/dev/null`
		res, err := sshExecutor.Exec(logCmd)
		if err != nil || res == nil || res.ExitCode != 0 {
			return false
		}
		output := res.Stdout + res.Stderr

		idx := strings.LastIndex(output, logMarker)
		if idx >= 0 {
			output = output[idx:]
		}

		if !strings.Contains(output, upgradePathDigestSkipPullLogMsg) {
			GinkgoWriter.Printf("尚未观察到 digest 未变跳过拉取日志，继续等待...\n")
			return false
		}

		GinkgoWriter.Printf("已匹配 digest 未变跳过拉取日志\n")
		return true
	}, waitBudget, upgradePathSyncInterval).Should(
		BeTrue(),
		"capbke 日志应含 %q 且本轮未拉取镜像", upgradePathDigestSkipPullLogMsg,
	)
}
