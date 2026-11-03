package utils

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

const (
	// FITTestNamespace FIT yaml 组件默认命名空间。
	FITTestNamespace = "fit-test"
	// FIT9PruneCoreNamespaceName FIT_9 预置核心 Namespace 样本（不在 fit-yaml manifest 内）。
	FIT9PruneCoreNamespaceName = "fit-prune-core"
	// FITPreBaselineProbeConfigMap 用于判断 install 是否已将基线 yaml 落到目标集群。
	FITPreBaselineProbeConfigMap = "fit-yaml-ssa"
	// FITYamlSSAConfigMap YAML_ENGINE_FIT_2 观测 ConfigMap。
	FITYamlSSAConfigMap = "fit-yaml-ssa"
	// FITYamlSSADataField YAML_ENGINE_FIT_2 观测 ConfigMap data 字段（vfit-pre→vfit-cur 应变化）。
	FITYamlSSADataField = "version"
	// FITYamlReplaceConfigMap YAML_ENGINE_FIT_3 Replace 策略观测 ConfigMap。
	FITYamlReplaceConfigMap = "fit-yaml-replace"
	// FITYamlCreateOnlyConfigMap YAML_ENGINE_FIT_4 CreateOnly 策略观测 ConfigMap。
	FITYamlCreateOnlyConfigMap = "fit-yaml-createonly"
	// FITYamlCreateOnlyDataField YAML_ENGINE_FIT_4 观测 ConfigMap data 字段（CreateOnly 不应被覆盖）。
	FITYamlCreateOnlyDataField = "value"
)

var fitPreBaselineComponentDirs = []string{
	"fit-yaml-gate",
	"fit-yaml-ssa",
	"fit-yaml-replace",
	"fit-yaml-createonly",
	"fit-yaml-prune",
	"fit-yaml-removed",
	"fit-yaml-skip-a",
	"fit-yaml-upgrade-b",
}

// KubectlApplyYAMLContent 在目标集群 apply 一段 YAML（经引导节点 stdin 导入）。
func (h *DeclarativeUpgradeHelper) KubectlApplyYAMLContent(kubeconfigPath, yamlContent string) error {
	yamlContent = strings.TrimSpace(yamlContent)
	Expect(yamlContent).NotTo(BeEmpty())

	const delimiter = "FIT_YAML_APPLY_EOF"
	cmd := fmt.Sprintf("cat <<'%s' | %skubectl apply -f -\n%s\n%s",
		delimiter, h.kubeconfigPrefix(kubeconfigPath), yamlContent, delimiter)
	result, err := h.executor.Exec(cmd)
	if err != nil {
		return fmt.Errorf("kubectl apply yaml content: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("kubectl apply yaml content failed: %s", result.Stderr)
	}
	return nil
}

// TargetClusterConfigMapExists 判断目标集群命名空间内 ConfigMap 是否存在。
func (h *DeclarativeUpgradeHelper) TargetClusterConfigMapExists(kubeconfigPath, namespace, name string) bool {
	_, err := h.kubectlJSONPath(kubeconfigPath, "cm", name, namespace, "{.metadata.name}")
	return err == nil
}

// GetTargetClusterNamespaceUID 读取目标集群 Namespace UID。
func (h *DeclarativeUpgradeHelper) GetTargetClusterNamespaceUID(kubeconfigPath, name string) (string, error) {
	cmd := fmt.Sprintf("%skubectl get ns %s -o jsonpath=%q",
		h.kubeconfigPrefix(kubeconfigPath), name, "{.metadata.uid}")
	result, err := h.executor.Exec(cmd)
	if err != nil {
		return "", fmt.Errorf("kubectl get ns %s: %w", name, err)
	}
	if result.ExitCode != 0 {
		return "", fmt.Errorf("kubectl get ns %s failed: %s", name, result.Stderr)
	}
	return strings.TrimSpace(strings.Trim(result.Stdout, "'")), nil
}

// GetTargetClusterConfigMapUID 读取目标集群 ConfigMap UID。
func (h *DeclarativeUpgradeHelper) GetTargetClusterConfigMapUID(kubeconfigPath, namespace, name string) (string, error) {
	uid, err := h.kubectlJSONPath(kubeconfigPath, "cm", name, namespace, "{.metadata.uid}")
	if err != nil {
		return "", fmt.Errorf("kubectl get cm %s/%s uid: %w", namespace, name, err)
	}
	uid = strings.TrimSpace(strings.Trim(uid, "'"))
	if uid == "" {
		return "", fmt.Errorf("configmap %s/%s uid is empty", namespace, name)
	}
	return uid, nil
}

// WaitTargetClusterConfigMapUID 等待 ConfigMap 存在并返回 UID。
func (h *DeclarativeUpgradeHelper) WaitTargetClusterConfigMapUID(kubeconfigPath, namespace, name string) string {
	var uid string
	Eventually(func() string {
		got, err := h.GetTargetClusterConfigMapUID(kubeconfigPath, namespace, name)
		if err != nil {
			GinkgoWriter.Printf("等待 ConfigMap %s/%s UID: %v\n", namespace, name, err)
			return ""
		}
		uid = got
		return got
	}, 2*time.Minute, 5*time.Second).ShouldNot(BeEmpty(),
		"目标集群 ConfigMap %s/%s 应存在", namespace, name)
	return uid
}

// GetTargetClusterConfigMapDataField 读取目标集群 ConfigMap data 字段。
func (h *DeclarativeUpgradeHelper) GetTargetClusterConfigMapDataField(
	kubeconfigPath, namespace, name, key string,
) (string, error) {
	jsonPath := fmt.Sprintf("{.data['%s']}", key)
	val, err := h.kubectlJSONPath(kubeconfigPath, "cm", name, namespace, jsonPath)
	if err != nil {
		return "", fmt.Errorf("kubectl get cm %s/%s data[%s]: %w", namespace, name, key, err)
	}
	return strings.TrimSpace(val), nil
}

// VerifyTargetClusterConfigMapUIDUnchanged 断言 ConfigMap UID 与升级前一致（SSA / CreateOnly 幂等）。
func (h *DeclarativeUpgradeHelper) VerifyTargetClusterConfigMapUIDUnchanged(
	kubeconfigPath, namespace, name, uidBefore, purpose string,
) {
	Expect(uidBefore).NotTo(BeEmpty())
	uidAfter, err := h.GetTargetClusterConfigMapUID(kubeconfigPath, namespace, name)
	Expect(err).NotTo(HaveOccurred(), "升级后 ConfigMap %s/%s 应仍存在", namespace, name)
	Expect(uidAfter).To(Equal(uidBefore),
		"%s: ConfigMap %s/%s UID 应与升级前一致", purpose, namespace, name)
	GinkgoWriter.Printf("%s: ConfigMap %s/%s uid=%s（与升级前一致）\n", purpose, namespace, name, uidAfter)
}

// VerifyTargetClusterConfigMapUIDChanged 断言 ConfigMap UID 已变化（Replace 删除重建）。
func (h *DeclarativeUpgradeHelper) VerifyTargetClusterConfigMapUIDChanged(
	kubeconfigPath, namespace, name, uidBefore, purpose string,
) {
	Expect(uidBefore).NotTo(BeEmpty())
	uidAfter, err := h.GetTargetClusterConfigMapUID(kubeconfigPath, namespace, name)
	Expect(err).NotTo(HaveOccurred(), "Replace 后 ConfigMap %s/%s 应仍存在", namespace, name)
	Expect(uidAfter).NotTo(Equal(uidBefore),
		"%s: ConfigMap %s/%s Replace 后 UID 必须变化（before=%s after=%s）",
		purpose, namespace, name, uidBefore, uidAfter)
	GinkgoWriter.Printf("%s: ConfigMap %s/%s uid changed %s → %s\n", purpose, namespace, name, uidBefore, uidAfter)
}

// EnsureFITPreBaselineOnTargetCluster 若 install 未 apply yaml 到目标集群，按 FIT-TEST-NOTES 手工补导入 vfit-pre 基线。
func (h *DeclarativeUpgradeHelper) EnsureFITPreBaselineOnTargetCluster(kubeconfigPath, baselineRoot string) {
	Expect(baselineRoot).NotTo(BeEmpty())
	if h.TargetClusterConfigMapExists(kubeconfigPath, FITTestNamespace, FITPreBaselineProbeConfigMap) {
		GinkgoWriter.Printf("FIT 基线 ConfigMap %s/%s 已存在，跳过手工补导入\n",
			FITTestNamespace, FITPreBaselineProbeConfigMap)
		return
	}

	By("FIT: install 未落到目标集群，按 FIT-TEST-NOTES 手工补导入 vfit-pre 基线 fit-yaml manifest")
	for _, component := range fitPreBaselineComponentDirs {
		componentDir := filepath.Join(baselineRoot, component)
		entries, err := os.ReadDir(componentDir)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			Expect(err).NotTo(HaveOccurred(), "读取 FIT 基线目录 %s", componentDir)
		}
		names := make([]string, 0, len(entries))
		for _, entry := range entries {
			if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".yaml") {
				continue
			}
			if entry.Name() == "component.yaml" {
				continue
			}
			names = append(names, entry.Name())
		}
		sort.Strings(names)
		for _, name := range names {
			content, readErr := os.ReadFile(filepath.Join(componentDir, name))
			Expect(readErr).NotTo(HaveOccurred(), "读取 FIT 基线 manifest %s/%s", component, name)
			Expect(h.KubectlApplyYAMLContent(kubeconfigPath, string(content))).To(Succeed(),
				"apply FIT 基线 manifest %s/%s", component, name)
			GinkgoWriter.Printf("FIT 手工补导入: %s/%s\n", component, name)
		}
	}

	Eventually(func() bool {
		return h.TargetClusterConfigMapExists(kubeconfigPath, FITTestNamespace, FITPreBaselineProbeConfigMap)
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(),
		"手工补导入后目标集群应存在 %s/%s", FITTestNamespace, FITPreBaselineProbeConfigMap)
}

// ApplyFIT9PruneCoreNamespace 预置 FIT_9 核心 Namespace 样本并返回升级前 UID。
func (h *DeclarativeUpgradeHelper) ApplyFIT9PruneCoreNamespace(kubeconfigPath, fixturePath string) string {
	content, err := os.ReadFile(fixturePath)
	Expect(err).NotTo(HaveOccurred(), "读取 FIT_9 fixture %s", fixturePath)
	Expect(h.KubectlApplyYAMLContent(kubeconfigPath, string(content))).To(Succeed())

	var uidBefore string
	Eventually(func() string {
		uid, getErr := h.GetTargetClusterNamespaceUID(kubeconfigPath, FIT9PruneCoreNamespaceName)
		if getErr != nil {
			GinkgoWriter.Printf("等待 FIT_9 Namespace %s: %v\n", FIT9PruneCoreNamespaceName, getErr)
			return ""
		}
		uidBefore = uid
		return uid
	}, 2*time.Minute, 5*time.Second).ShouldNot(BeEmpty(),
		"FIT_9 预置 Namespace %s 应存在", FIT9PruneCoreNamespaceName)
	GinkgoWriter.Printf("FIT_9 升级前 Namespace %s uid=%s\n", FIT9PruneCoreNamespaceName, uidBefore)
	return uidBefore
}

// VerifyFIT9PruneCoreNamespacePreserved 升级后核心 Namespace 仍存在且 UID 不变。
func (h *DeclarativeUpgradeHelper) VerifyFIT9PruneCoreNamespacePreserved(kubeconfigPath, uidBefore string) {
	Expect(uidBefore).NotTo(BeEmpty())
	uidAfter, err := h.GetTargetClusterNamespaceUID(kubeconfigPath, FIT9PruneCoreNamespaceName)
	Expect(err).NotTo(HaveOccurred(), "升级后 Namespace %s 应仍存在", FIT9PruneCoreNamespaceName)
	Expect(uidAfter).To(Equal(uidBefore),
		"FIT_9 核心 Namespace %s UID 应与升级前一致（Prune 不得误删）", FIT9PruneCoreNamespaceName)
	GinkgoWriter.Printf("FIT_9 升级后 Namespace %s uid=%s（与升级前一致）\n", FIT9PruneCoreNamespaceName, uidAfter)
}
