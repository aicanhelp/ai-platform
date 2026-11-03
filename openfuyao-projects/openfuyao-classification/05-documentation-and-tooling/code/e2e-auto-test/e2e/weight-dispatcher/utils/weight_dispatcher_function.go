package utils

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
)

// HelmRelease helm release 结构。
type HelmRelease struct {
	Name      string `json:"name"`
	Namespace string `json:"namespace"`
	Status    string `json:"status"`
}

var modelWarmupJobGVR = schema.GroupVersionResource{
	Group:    "warmup.openfuyao.com",
	Version:  "v1alpha1",
	Resource: "modelwarmupjobs",
}

type ModelWarmupJobE2EConfig struct {
	ArtifactKey string
	SourceNode  string
	SourcePath  string
	TargetNode  string
	TargetPath  string
}

type ModelWarmupJobSource struct {
	NodeName string
	RootPath string
}

type WeightDispatcherSuiteRuntimeConfig struct {
	ArtifactKey          string
	SourceNode           string
	TargetNode           string
	SourceNodes          []string
	TargetNodes          []string
	SourcePath           string
	SecondSourcePath     string
	SingleFileSourcePath string
	MismatchSourcePath   string
	TimeoutSourcePath    string
	TargetPath           string
	HelmChart            string
	HelmNamespace        string
	ControllerLogLevel   string
	NodeAgentLogLevel    string
	NodeAgentDefaultEnv  map[string]string
}

type fileDigest struct {
	SizeBytes int64
	SHA256    string
}

var suiteRuntimeConfig WeightDispatcherSuiteRuntimeConfig

func SetSuiteRuntimeConfig(cfg WeightDispatcherSuiteRuntimeConfig) {
	suiteRuntimeConfig = cfg
}

func GetSuiteRuntimeConfig() WeightDispatcherSuiteRuntimeConfig {
	return suiteRuntimeConfig
}

func ResolveSuiteRuntimeConfig(clientset *kubernetes.Clientset, ctx context.Context) (WeightDispatcherSuiteRuntimeConfig, error) {
	nodes, err := readyNodeNames(clientset, ctx)
	if err != nil {
		return WeightDispatcherSuiteRuntimeConfig{}, err
	}
	return resolveDefaultSuiteRuntimeConfigFromReadyNodes(nodes)
}

func resolveDefaultSuiteRuntimeConfigFromReadyNodes(readyNodes []string) (WeightDispatcherSuiteRuntimeConfig, error) {
	if len(readyNodes) < 2 {
		return WeightDispatcherSuiteRuntimeConfig{}, fmt.Errorf("Ready 节点数量不足: got %d, want >= 2", len(readyNodes))
	}
	nodes := append([]string(nil), readyNodes...)
	sort.Strings(nodes)

	sourceNode := firstPreferredReadyNode(nodes, []string{"m1", "n2"})
	if sourceNode == "" {
		sourceNode = nodes[len(nodes)-1]
	}
	targetNodes := preferredReadyNodesExcluding(nodes, []string{"n1", "n2", "m0", "m1"}, sourceNode, 2)
	if len(targetNodes) == 0 {
		return WeightDispatcherSuiteRuntimeConfig{}, fmt.Errorf("Ready target 节点数量不足: got %d, want >= 1", len(nodes)-1)
	}
	sourceNodes := preferredReadyNodesExcluding(nodes, []string{sourceNode, "n2", "m1", "n1", "m0"}, "", 2)
	if len(sourceNodes) < 2 {
		sourceNodes = appendUniqueStrings(sourceNodes, sourceNode)
		sourceNodes = appendUniqueStrings(sourceNodes, nodes...)
	}
	if len(sourceNodes) > 2 {
		sourceNodes = sourceNodes[:2]
	}

	chartPath := WeightDispatcherHelmChart
	if wd, err := os.Getwd(); err == nil {
		localChart := filepath.Clean(filepath.Join(wd, "..", "..", "weight-dispatcher", "charts", "weight-dispatcher"))
		if stat, statErr := os.Stat(localChart); statErr == nil && stat.IsDir() {
			chartPath = localChart
		}
	}

	return WeightDispatcherSuiteRuntimeConfig{
		ArtifactKey:          ModelWarmupJobDefaultArtifactKey,
		SourceNode:           sourceNode,
		TargetNode:           targetNodes[0],
		SourceNodes:          sourceNodes,
		TargetNodes:          targetNodes,
		SourcePath:           ModelWarmupJobDefaultSourcePath,
		SecondSourcePath:     ModelWarmupJobDefaultSecondSourcePath,
		SingleFileSourcePath: ModelWarmupJobDefaultSingleFileSourcePath,
		MismatchSourcePath:   ModelWarmupJobDefaultMismatchSourcePath,
		TimeoutSourcePath:    ModelWarmupJobDefaultTimeoutSourcePath,
		TargetPath:           ModelWarmupJobDefaultTargetPath,
		HelmChart:            chartPath,
		HelmNamespace:        WeightDispatcherHelmNamespace,
		ControllerLogLevel:   WeightDispatcherDefaultControllerLogLevel,
		NodeAgentLogLevel:    WeightDispatcherDefaultNodeAgentLogLevel,
		NodeAgentDefaultEnv: map[string]string{
			"RDMA_ENABLED":                      WeightDispatcherDefaultRDMAEnabled,
			"TCP_FALLBACK_ENABLED":              WeightDispatcherDefaultTCPFallbackEnabled,
			"LOG_LEVEL":                         WeightDispatcherDefaultNodeAgentRuntimeLogLevel,
			"WD_ENABLE_POST_WRITE_CRC32C":       WeightDispatcherDefaultPostWriteCRC32CEnabled,
			"WD_INJECT_CHUNK_CRC_MISMATCH_ONCE": WeightDispatcherDefaultInjectCRCMismatchOnce,
			"WD_FORCE_TCP_FALLBACK":             WeightDispatcherDefaultForceTCPFallback,
		},
	}, nil
}

func InstallWeightDispatcher() error {
	args := buildInstallWeightDispatcherArgs()
	cmd := exec.Command("helm", args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("helm upgrade --install weight-dispatcher 失败: %v\nOutput: %s", err, string(out))
	}
	GinkgoWriter.Printf("helm upgrade --install weight-dispatcher 输出:\n%s\n", string(out))
	return nil
}

func buildInstallWeightDispatcherArgs() []string {
	chart := WeightDispatcherHelmChart
	if suiteRuntimeConfig.HelmChart != "" {
		chart = suiteRuntimeConfig.HelmChart
	}
	chart = getEnvOrDefault("WEIGHT_DISPATCHER_HELM_CHART", chart)

	namespace := WeightDispatcherHelmNamespace
	if suiteRuntimeConfig.HelmNamespace != "" {
		namespace = suiteRuntimeConfig.HelmNamespace
	}
	namespace = getEnvOrDefault("WEIGHT_DISPATCHER_HELM_NAMESPACE", namespace)
	version := WeightDispatcherHelmVersion
	if value, ok := os.LookupEnv("WEIGHT_DISPATCHER_HELM_VERSION"); ok {
		version = value
	}

	args := []string{"upgrade", "--install", WeightDispatcherHelmReleaseName, chart}
	if version != "" {
		args = append(args, "--version", version)
	}
	if namespace != "" {
		args = append(args, "-n", namespace, "--create-namespace")
	}
	controllerLogLevel := suiteRuntimeConfig.ControllerLogLevel
	if controllerLogLevel == "" {
		controllerLogLevel = os.Getenv("WEIGHT_DISPATCHER_CONTROLLER_LOG_LEVEL")
	}
	nodeAgentLogLevel := suiteRuntimeConfig.NodeAgentLogLevel
	if nodeAgentLogLevel == "" {
		nodeAgentLogLevel = os.Getenv("WEIGHT_DISPATCHER_NODE_AGENT_LOG_LEVEL")
	}
	args = appendHelmSetStringArg(args, "controller.logLevel", controllerLogLevel)
	args = appendHelmSetStringArg(args, "nodeAgent.logLevel", nodeAgentLogLevel)
	args = appendHelmSetStringArg(args, "controller.image.repository", os.Getenv("WEIGHT_DISPATCHER_CONTROLLER_IMAGE_REPOSITORY"))
	args = appendHelmSetStringArg(args, "controller.image.tag", os.Getenv("WEIGHT_DISPATCHER_CONTROLLER_IMAGE_TAG"))
	args = appendHelmSetStringArg(args, "nodeAgent.image.repository", os.Getenv("WEIGHT_DISPATCHER_NODE_AGENT_IMAGE_REPOSITORY"))
	args = appendHelmSetStringArg(args, "nodeAgent.image.tag", os.Getenv("WEIGHT_DISPATCHER_NODE_AGENT_IMAGE_TAG"))
	args = appendHelmSetStringArg(args, "nodeAgent.cacheRoot", getEnvOrDefault("WEIGHT_DISPATCHER_NODE_AGENT_CACHE_ROOT", WeightDispatcherDefaultNodeAgentCacheRoot))
	args = appendHelmSetStringArg(args, "nodeAgent.hostHomeCacheRoot", getEnvOrDefault("WEIGHT_DISPATCHER_NODE_AGENT_HOST_HOME_CACHE_ROOT", WeightDispatcherDefaultHostHomeCacheRoot))
	args = appendHelmSetBoolArg(args, "nodeAgent.rdmaEnabled", getEnvOrDefault("WEIGHT_DISPATCHER_NODE_AGENT_RDMA_ENABLED", WeightDispatcherDefaultRDMAEnabled))
	args = appendHelmSetBoolArg(args, "nodeAgent.tcpFallbackEnabled", getEnvOrDefault("WEIGHT_DISPATCHER_NODE_AGENT_TCP_FALLBACK_ENABLED", WeightDispatcherDefaultTCPFallbackEnabled))
	args = appendHelmSetBoolArg(args, "nodeAgent.hostNetwork", getEnvOrDefault("WEIGHT_DISPATCHER_NODE_AGENT_HOST_NETWORK", WeightDispatcherDefaultHostNetwork))
	args = appendHelmSetBoolArg(args, "nodeAgent.privileged", getEnvOrDefault("WEIGHT_DISPATCHER_NODE_AGENT_PRIVILEGED", WeightDispatcherDefaultPrivileged))
	args = appendHelmSetBoolArg(args, "nodeAgent.postWriteCRC32CEnabled", getEnvOrDefault("WEIGHT_DISPATCHER_NODE_AGENT_POST_WRITE_CRC32C_ENABLED", WeightDispatcherDefaultPostWriteCRC32CEnabled))
	return args
}

func UninstallWeightDispatcher() error {
	namespace := WeightDispatcherHelmNamespace
	if suiteRuntimeConfig.HelmNamespace != "" {
		namespace = suiteRuntimeConfig.HelmNamespace
	}
	namespace = getEnvOrDefault("WEIGHT_DISPATCHER_HELM_NAMESPACE", namespace)

	args := []string{"uninstall", WeightDispatcherHelmReleaseName}
	if namespace != "" {
		args = append(args, "-n", namespace)
	}
	cmd := exec.Command("helm", args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		combined := string(out)
		if strings.Contains(strings.ToLower(combined), "not found") {
			GinkgoWriter.Printf("helm uninstall weight-dispatcher 输出(已不存在):\n%s\n", combined)
			return nil
		}
		return fmt.Errorf("helm uninstall weight-dispatcher 失败: %v\nOutput: %s", err, combined)
	}
	GinkgoWriter.Printf("helm uninstall weight-dispatcher 输出:\n%s\n", string(out))
	return nil
}

func ResolveModelWarmupJobE2EConfig(clientset *kubernetes.Clientset, ctx context.Context) (ModelWarmupJobE2EConfig, error) {
	sourceNode := suiteRuntimeConfig.SourceNode
	targetNode := suiteRuntimeConfig.TargetNode
	if override := os.Getenv("WEIGHT_DISPATCHER_SOURCE_NODE"); override != "" {
		sourceNode = override
	}
	if override := os.Getenv("WEIGHT_DISPATCHER_TARGET_NODE"); override != "" {
		targetNode = override
	}
	if sourceNode == "" || targetNode == "" {
		nodes, err := readyNodeNames(clientset, ctx)
		if err != nil {
			return ModelWarmupJobE2EConfig{}, err
		}
		if len(nodes) == 0 {
			return ModelWarmupJobE2EConfig{}, fmt.Errorf("未找到 Ready 节点，请通过 WEIGHT_DISPATCHER_SOURCE_NODE/WEIGHT_DISPATCHER_TARGET_NODE 指定")
		}
		if sourceNode == "" {
			sourceNode = nodes[0]
		}
		if targetNode == "" {
			targetNode = nodes[0]
			if len(nodes) > 1 {
				targetNode = nodes[1]
			}
		}
	}

	return ModelWarmupJobE2EConfig{
		ArtifactKey: getEnvOrDefault("WEIGHT_DISPATCHER_ARTIFACT_KEY", firstNonEmpty(suiteRuntimeConfig.ArtifactKey, ModelWarmupJobDefaultArtifactKey)),
		SourceNode:  sourceNode,
		SourcePath:  getEnvOrDefault("WEIGHT_DISPATCHER_SOURCE_PATH", firstNonEmpty(suiteRuntimeConfig.SourcePath, ModelWarmupJobDefaultSourcePath)),
		TargetNode:  targetNode,
		TargetPath:  getEnvOrDefault("WEIGHT_DISPATCHER_TARGET_PATH", firstNonEmpty(suiteRuntimeConfig.TargetPath, ModelWarmupJobDefaultTargetPath)),
	}, nil
}

func ResolveReadyNodes(clientset *kubernetes.Clientset, ctx context.Context, minCount int) ([]string, error) {
	nodes, err := readyNodeNames(clientset, ctx)
	if err != nil {
		return nil, err
	}
	if len(nodes) < minCount {
		return nil, fmt.Errorf("Ready 节点数量不足: got %d, want >= %d", len(nodes), minCount)
	}
	return nodes, nil
}

func ResolveSourceNodes(clientset *kubernetes.Clientset, ctx context.Context, minCount int) ([]string, error) {
	if values := splitEnvList("WEIGHT_DISPATCHER_SOURCE_NODES"); len(values) >= minCount {
		return values[:minCount], nil
	}
	if len(suiteRuntimeConfig.SourceNodes) >= minCount {
		return append([]string(nil), suiteRuntimeConfig.SourceNodes[:minCount]...), nil
	}
	nodes, err := ResolveReadyNodes(clientset, ctx, minCount)
	if err != nil {
		return nil, err
	}
	return nodes[:minCount], nil
}

func ResolveTargetNodes(clientset *kubernetes.Clientset, ctx context.Context, minCount int) ([]string, error) {
	if values := splitEnvList("WEIGHT_DISPATCHER_TARGET_NODES"); len(values) >= minCount {
		return values[:minCount], nil
	}
	if len(suiteRuntimeConfig.TargetNodes) >= minCount {
		return append([]string(nil), suiteRuntimeConfig.TargetNodes[:minCount]...), nil
	}
	nodes, err := ResolveReadyNodes(clientset, ctx, minCount)
	if err != nil {
		return nil, err
	}
	if minCount == 1 {
		return []string{nodes[len(nodes)-1]}, nil
	}
	return nodes[:minCount], nil
}

func ResolveSecondSourcePath(cfg ModelWarmupJobE2EConfig) string {
	return getEnvOrDefault("WEIGHT_DISPATCHER_SOURCE_PATH_2", firstNonEmpty(suiteRuntimeConfig.SecondSourcePath, fmt.Sprintf("%s-fit9-source-2", cfg.SourcePath)))
}

func ResolveManifestMismatchSourcePath(cfg ModelWarmupJobE2EConfig) string {
	return getEnvOrDefault("WEIGHT_DISPATCHER_MISMATCH_SOURCE_PATH", firstNonEmpty(suiteRuntimeConfig.MismatchSourcePath, fmt.Sprintf("%s-manifest-mismatch", cfg.SourcePath)))
}

func ResolveTimeoutSourcePath(cfg ModelWarmupJobE2EConfig) string {
	return getEnvOrDefault("WEIGHT_DISPATCHER_TIMEOUT_SOURCE_PATH", firstNonEmpty(suiteRuntimeConfig.TimeoutSourcePath, cfg.SourcePath))
}

func ResolveSingleFileSourcePath(cfg ModelWarmupJobE2EConfig) string {
	return getEnvOrDefault("WEIGHT_DISPATCHER_SINGLE_FILE_SOURCE_PATH", firstNonEmpty(suiteRuntimeConfig.SingleFileSourcePath, cfg.SourcePath))
}

func ShouldPreserveReadyCacheForSpec(specName string) bool {
	switch {
	case strings.Contains(specName, "FIT_3"),
		strings.Contains(specName, "FIT_4"),
		strings.Contains(specName, "FIT_5"),
		strings.Contains(specName, "FIT_6"),
		strings.Contains(specName, "FIT_7"),
		strings.Contains(specName, "FIT_9"),
		strings.Contains(specName, "FIT_10"):
		return true
	default:
		return false
	}
}

func EnsureNamespace(clientset *kubernetes.Clientset, ctx context.Context, namespace string) error {
	_, err := clientset.CoreV1().Namespaces().Get(ctx, namespace, metav1.GetOptions{})
	if err == nil {
		return nil
	}
	if !apierrors.IsNotFound(err) {
		return err
	}
	ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: namespace}}
	_, err = clientset.CoreV1().Namespaces().Create(ctx, ns, metav1.CreateOptions{})
	return err
}

// WaitForNamespaceDeletionIfTerminating 若 namespace 处于 Terminating，则等待其被完全删除。
func WaitForNamespaceDeletionIfTerminating(clientset *kubernetes.Clientset, ctx context.Context, namespace string, timeout, interval time.Duration) error {
	if namespace == "" {
		return nil
	}
	ns, err := clientset.CoreV1().Namespaces().Get(ctx, namespace, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("获取 namespace %s 失败: %w", namespace, err)
	}
	if ns.Status.Phase != corev1.NamespaceTerminating {
		return nil
	}

	GinkgoWriter.Printf("namespace %s 正在删除中，等待其删除完成...\n", namespace)
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		_, err := clientset.CoreV1().Namespaces().Get(ctx, namespace, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			GinkgoWriter.Printf("namespace %s 已删除完成\n", namespace)
			return nil
		}
		if err != nil {
			GinkgoWriter.Printf("查询 namespace %s 状态失败，继续重试: %v\n", namespace, err)
		}
		time.Sleep(interval)
	}
	return fmt.Errorf("等待 namespace %s 删除超时（%s）", namespace, timeout)
}

func VerifyHelmReleaseStatus(releaseName, namespace string) {
	GinkgoWriter.Printf("验证 helm release %s/%s 状态为 deployed...\n", releaseName, namespace)

	Eventually(func() string {
		releases, err := getHelmList()
		if err != nil {
			GinkgoWriter.Printf("helm list 失败: %v\n", err)
			return ""
		}
		for _, release := range releases {
			if release.Name == releaseName && release.Namespace == namespace {
				return release.Status
			}
		}
		return ""
	}, 2*time.Minute, 5*time.Second).Should(Equal("deployed"),
		"helm release %s/%s 应在 2 分钟内变为 deployed 状态", namespace, releaseName)
}

func VerifyModelWarmupJobCRDEstablished(apiExtensionsClient apiextensionsclientset.Interface, ctx context.Context, timeout, interval time.Duration) {
	Eventually(func() bool {
		crd, err := apiExtensionsClient.ApiextensionsV1().CustomResourceDefinitions().Get(ctx, ModelWarmupJobCRDName, metav1.GetOptions{})
		if err != nil {
			GinkgoWriter.Printf("获取 CRD %s 失败: %v\n", ModelWarmupJobCRDName, err)
			return false
		}
		for _, condition := range crd.Status.Conditions {
			if condition.Type == apiextensionsv1.Established && condition.Status == apiextensionsv1.ConditionTrue {
				return true
			}
		}
		return false
	}, timeout, interval).Should(BeTrue(), "CRD %s 应处于 Established 状态", ModelWarmupJobCRDName)
}

func VerifyWeightDispatcherPodsRunning(clientset *kubernetes.Clientset, ctx context.Context, timeout, interval time.Duration) {
	Eventually(func() bool {
		status, err := judgeWeightDispatcherPods(clientset, ctx)
		if err != nil {
			GinkgoWriter.Printf("检查 weight-dispatcher Pod 失败: %v\n", err)
			return false
		}
		if !status.ready {
			GinkgoWriter.Printf("weight-dispatcher Pod 尚未就绪: controller=%t nodeAgent=%t total=%d\n",
				status.hasController, status.hasNodeAgent, status.total)
		}
		return status.ready
	}, timeout, interval).Should(BeTrue(), "controller 与 node-agent Pod 应在超时时间内全部 Running/Ready")
}

func BuildModelWarmupJob(name, namespace string, cfg ModelWarmupJobE2EConfig) *unstructured.Unstructured {
	return BuildModelWarmupJobWithNodes(name, namespace, cfg,
		[]ModelWarmupJobSource{{NodeName: cfg.SourceNode, RootPath: cfg.SourcePath}},
		[]string{cfg.TargetNode})
}

func BuildModelWarmupJobWithNodes(name, namespace string, cfg ModelWarmupJobE2EConfig, sources []ModelWarmupJobSource, targetNodes []string) *unstructured.Unstructured {
	obj := buildModelWarmupJobBase(name, namespace, cfg, sources)
	targetNodeValues := make([]interface{}, 0, len(targetNodes))
	for _, node := range targetNodes {
		targetNodeValues = append(targetNodeValues, node)
	}
	_ = unstructured.SetNestedMap(obj.Object, map[string]interface{}{
		"nodeNames":  targetNodeValues,
		"targetPath": cfg.TargetPath,
	}, "spec", "target")
	return obj
}

func appendHelmSetStringArg(args []string, key, value string) []string {
	if strings.TrimSpace(value) == "" {
		return args
	}
	return append(args, "--set-string", fmt.Sprintf("%s=%s", key, value))
}

func appendHelmSetBoolArg(args []string, key, value string) []string {
	trimmed := strings.TrimSpace(strings.ToLower(value))
	switch trimmed {
	case "true", "false":
		return append(args, "--set", fmt.Sprintf("%s=%s", key, trimmed))
	default:
		return args
	}
}

func BuildModelWarmupJobWithNodeSelector(name, namespace string, cfg ModelWarmupJobE2EConfig, sources []ModelWarmupJobSource, selector map[string]string) *unstructured.Unstructured {
	obj := buildModelWarmupJobBase(name, namespace, cfg, sources)
	nodeSelector := make(map[string]interface{}, len(selector))
	for key, value := range selector {
		nodeSelector[key] = value
	}
	_ = unstructured.SetNestedMap(obj.Object, map[string]interface{}{
		"nodeSelector": nodeSelector,
		"targetPath":   cfg.TargetPath,
	}, "spec", "target")
	return obj
}

func BuildModelWarmupJobWithPolicy(name, namespace string, cfg ModelWarmupJobE2EConfig, mutatePolicy func(map[string]interface{})) *unstructured.Unstructured {
	obj := BuildModelWarmupJob(name, namespace, cfg)
	policy, found, err := unstructured.NestedMap(obj.Object, "spec", "policy")
	Expect(err).NotTo(HaveOccurred())
	Expect(found).To(BeTrue(), "spec.policy 应存在")
	mutatePolicy(policy)
	Expect(unstructured.SetNestedMap(obj.Object, policy, "spec", "policy")).NotTo(HaveOccurred())
	return obj
}

func BuildModelWarmupJobMissingArtifactKey(name, namespace string, cfg ModelWarmupJobE2EConfig) *unstructured.Unstructured {
	obj := BuildModelWarmupJob(name, namespace, cfg)
	unstructured.RemoveNestedField(obj.Object, "spec", "artifact", "key")
	return obj
}

func BuildModelWarmupJobMissingTargetPath(name, namespace string, cfg ModelWarmupJobE2EConfig) *unstructured.Unstructured {
	obj := BuildModelWarmupJob(name, namespace, cfg)
	unstructured.RemoveNestedField(obj.Object, "spec", "target", "targetPath")
	return obj
}

func BuildModelWarmupJobWithoutSources(name, namespace string, cfg ModelWarmupJobE2EConfig) *unstructured.Unstructured {
	obj := BuildModelWarmupJob(name, namespace, cfg)
	unstructured.RemoveNestedField(obj.Object, "spec", "sources")
	return obj
}

func BuildModelWarmupJobNodeSourceWithEndpoint(name, namespace string, cfg ModelWarmupJobE2EConfig) *unstructured.Unstructured {
	obj := BuildModelWarmupJob(name, namespace, cfg)
	mutateModelWarmupJobSource(obj, 0, func(source map[string]interface{}) {
		source["endpoint"] = "http://127.0.0.1:18080"
	})
	return obj
}

func BuildModelWarmupJobNodeSourceMissingNodeName(name, namespace string, cfg ModelWarmupJobE2EConfig) *unstructured.Unstructured {
	obj := BuildModelWarmupJob(name, namespace, cfg)
	mutateModelWarmupJobSource(obj, 0, func(source map[string]interface{}) {
		delete(source, "nodeName")
	})
	return obj
}

func BuildModelWarmupJobWithDuplicateSources(name, namespace string, cfg ModelWarmupJobE2EConfig) *unstructured.Unstructured {
	obj := BuildModelWarmupJob(name, namespace, cfg)
	sources, found, err := unstructured.NestedSlice(obj.Object, "spec", "sources")
	Expect(err).NotTo(HaveOccurred())
	Expect(found).To(BeTrue(), "spec.sources 应存在")
	Expect(sources).NotTo(BeEmpty(), "spec.sources 应至少有一条记录")
	sources = append(sources, sources[0])
	Expect(unstructured.SetNestedSlice(obj.Object, sources, "spec", "sources")).NotTo(HaveOccurred())
	return obj
}

func mutateModelWarmupJobSource(obj *unstructured.Unstructured, index int, mutate func(map[string]interface{})) {
	sources, found, err := unstructured.NestedSlice(obj.Object, "spec", "sources")
	Expect(err).NotTo(HaveOccurred())
	Expect(found).To(BeTrue(), "spec.sources 应存在")
	Expect(len(sources)).To(BeNumerically(">", index), "spec.sources[%d] 应存在", index)
	source, ok := sources[index].(map[string]interface{})
	Expect(ok).To(BeTrue(), "spec.sources[%d] 应为对象", index)
	mutate(source)
	sources[index] = source
	Expect(unstructured.SetNestedSlice(obj.Object, sources, "spec", "sources")).NotTo(HaveOccurred())
}

func buildModelWarmupJobBase(name, namespace string, cfg ModelWarmupJobE2EConfig, sources []ModelWarmupJobSource) *unstructured.Unstructured {
	sourceValues := make([]interface{}, 0, len(sources))
	for _, source := range sources {
		sourceValues = append(sourceValues, map[string]interface{}{
			"sourceType": "node",
			"nodeName":   source.NodeName,
			"path":       source.RootPath,
		})
	}
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": ModelWarmupJobAPIVersion,
			"kind":       ModelWarmupJobKind,
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": namespace,
			},
			"spec": map[string]interface{}{
				"artifact": map[string]interface{}{
					"key": cfg.ArtifactKey,
				},
				"sources": sourceValues,
				"policy": map[string]interface{}{
					"chunkSizeMB":     int64(64),
					"publishAsSource": true,
					"timeoutSeconds":  int64(1800),
				},
			},
		},
	}
}

func LabelNodes(clientset *kubernetes.Clientset, ctx context.Context, nodeNames []string, key, value string) error {
	for _, nodeName := range nodeNames {
		node, err := clientset.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		if node.Labels == nil {
			node.Labels = map[string]string{}
		}
		node.Labels[key] = value
		if _, err := clientset.CoreV1().Nodes().Update(ctx, node, metav1.UpdateOptions{}); err != nil {
			return err
		}
	}
	return nil
}

func RestartWeightDispatcherController(clientset *kubernetes.Clientset, ctx context.Context, timeout time.Duration) error {
	pods, err := clientset.CoreV1().Pods(metav1.NamespaceAll).List(ctx, metav1.ListOptions{})
	if err != nil {
		return err
	}
	deleted := 0
	for _, pod := range pods.Items {
		if !isWeightDispatcherPod(pod) {
			continue
		}
		name := strings.ToLower(pod.Name)
		component := strings.ToLower(pod.Labels["app.kubernetes.io/component"])
		if !strings.Contains(name, WeightDispatcherControllerKey) && component != WeightDispatcherControllerKey {
			continue
		}
		if err := clientset.CoreV1().Pods(pod.Namespace).Delete(ctx, pod.Name, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
			return err
		}
		deleted++
	}
	if deleted == 0 {
		return fmt.Errorf("未找到 weight-dispatcher controller Pod")
	}
	VerifyWeightDispatcherPodsRunning(clientset, ctx, timeout, 10*time.Second)
	return nil
}

func RestartWeightDispatcherNodeAgentOnNode(clientset *kubernetes.Clientset, ctx context.Context, nodeName string, timeout time.Duration) error {
	pods, err := clientset.CoreV1().Pods(metav1.NamespaceAll).List(ctx, metav1.ListOptions{})
	if err != nil {
		return err
	}
	deleted := 0
	for _, pod := range pods.Items {
		if pod.Spec.NodeName != nodeName || !isWeightDispatcherPod(pod) {
			continue
		}
		name := strings.ToLower(pod.Name)
		component := strings.ToLower(pod.Labels["app.kubernetes.io/component"])
		if !strings.Contains(name, WeightDispatcherNodeAgentKey) && !strings.Contains(component, "agent") {
			continue
		}
		if err := clientset.CoreV1().Pods(pod.Namespace).Delete(ctx, pod.Name, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
			return err
		}
		deleted++
	}
	if deleted == 0 {
		return fmt.Errorf("未找到节点 %s 上的 weight-dispatcher node-agent Pod", nodeName)
	}
	VerifyWeightDispatcherPodsRunning(clientset, ctx, timeout, 10*time.Second)
	return nil
}

func PatchWeightDispatcherNodeAgentDaemonSetEnv(clientset *kubernetes.Clientset, ctx context.Context, envs map[string]string, timeout time.Duration) error {
	daemonSets, err := clientset.AppsV1().DaemonSets(metav1.NamespaceAll).List(ctx, metav1.ListOptions{})
	if err != nil {
		return err
	}
	matched := 0
	patched := 0
	for _, daemonSet := range daemonSets.Items {
		if !isWeightDispatcherNodeAgentDaemonSet(daemonSet) {
			continue
		}
		matched++
		ds, err := clientset.AppsV1().DaemonSets(daemonSet.Namespace).Get(ctx, daemonSet.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		changed := false
		for i := range ds.Spec.Template.Spec.Containers {
			if setContainerEnvs(&ds.Spec.Template.Spec.Containers[i], envs) {
				changed = true
			}
		}
		if !changed {
			continue
		}
		patchBytes, err := nodeAgentDaemonSetEnvPatch(ds.Spec.Template.Spec.Containers)
		if err != nil {
			return err
		}
		if _, err := clientset.AppsV1().DaemonSets(ds.Namespace).Patch(ctx, ds.Name, types.StrategicMergePatchType, patchBytes, metav1.PatchOptions{}); err != nil {
			return err
		}
		patched++
	}
	if matched == 0 {
		return fmt.Errorf("未找到可 patch 的 weight-dispatcher node-agent DaemonSet")
	}
	if patched > 0 {
		WaitForWeightDispatcherNodeAgentDaemonSetsReady(clientset, ctx, timeout, 10*time.Second)
	} else {
		VerifyWeightDispatcherPodsRunning(clientset, ctx, timeout, 10*time.Second)
	}
	return nil
}

func PatchWeightDispatcherNodeAgentDaemonSetEnvIfPresent(clientset *kubernetes.Clientset, ctx context.Context, envs map[string]string, timeout time.Duration) error {
	daemonSets, err := clientset.AppsV1().DaemonSets(metav1.NamespaceAll).List(ctx, metav1.ListOptions{})
	if err != nil {
		return err
	}
	for _, daemonSet := range daemonSets.Items {
		if isWeightDispatcherNodeAgentDaemonSet(daemonSet) {
			return PatchWeightDispatcherNodeAgentDaemonSetEnv(clientset, ctx, envs, timeout)
		}
	}
	GinkgoWriter.Printf("未找到 weight-dispatcher node-agent DaemonSet，跳过 env patch\n")
	return nil
}

func EnsureNodeAgentDefaultRuntimeEnv(clientset *kubernetes.Clientset, ctx context.Context, timeout time.Duration) error {
	if len(suiteRuntimeConfig.NodeAgentDefaultEnv) == 0 {
		return nil
	}
	return PatchWeightDispatcherNodeAgentDaemonSetEnvIfPresent(clientset, ctx, suiteRuntimeConfig.NodeAgentDefaultEnv, timeout)
}

func PrepareWeightDispatcherTestData(clientset *kubernetes.Clientset, ctx context.Context) error {
	cfg := suiteRuntimeConfig
	if cfg.SourceNode == "" || len(cfg.SourceNodes) < 2 {
		return fmt.Errorf("suiteRuntimeConfig 未初始化，无法准备测试数据")
	}
	primarySourcePod, err := weightDispatcherNodeAgentPodForNode(clientset, ctx, cfg.SourceNode)
	if err != nil {
		return err
	}
	secondarySourcePod, err := weightDispatcherNodeAgentPodForNode(clientset, ctx, cfg.SourceNodes[1])
	if err != nil {
		return err
	}
	nodePods, err := listWeightDispatcherNodeAgentPods(clientset, ctx)
	if err != nil {
		return err
	}
	for _, pod := range nodePods {
		script := fmt.Sprintf("set -eu; mkdir -p %s", shellQuoteSingle(cfg.TargetPath))
		if _, err := runPodShellCommand(pod.Namespace, pod.Name, script); err != nil {
			return err
		}
	}
	primaryScript := fmt.Sprintf(`set -eu
root=%s
single_file=%s
timeout_root=%s
ready_root=%s
if [ ! -e "$root" ]; then
  echo "source path is not visible in node-agent: $root" >&2
  echo "ls -ld /home /home/llm_cache /home/llm_cache/huggingface:" >&2
  ls -ld /home /home/llm_cache /home/llm_cache/huggingface 2>&2 || true
  echo "mounts containing llm_cache or weight-dispatcher:" >&2
  mount | grep -E 'llm_cache|weight-dispatcher' >&2 || true
  exit 1
fi
rm -rf "$(dirname "$single_file")" "$(dirname "$timeout_root")" "$ready_root"
mkdir -p "$(dirname "$single_file")" "$timeout_root/huge" "$ready_root"
first_safetensor=$(find -L "$root" -path '*/snapshots/*' -type f -name '*.safetensors' | LC_ALL=C sort | head -n 1)
if [ -z "$first_safetensor" ]; then
  first_safetensor=$(find -L "$root" -type f -name '*.safetensors' | LC_ALL=C sort | head -n 1)
fi
if [ -z "$first_safetensor" ]; then
  echo "no safetensors file found under source path: $root" >&2
  find "$root" -maxdepth 3 \( -type f -o -type l \) | head -n 40 >&2 || true
  exit 1
fi
ln -sfn "$first_safetensor" "$single_file"
dd if=/dev/zero of="$timeout_root/huge/model-big-00001.bin" bs=1M count=512 status=none
dd if=/dev/zero of="$timeout_root/huge/model-big-00002.bin" bs=1M count=512 status=none`,
		shellQuoteSingle(cfg.SourcePath),
		shellQuoteSingle(firstNonEmpty(cfg.SingleFileSourcePath, ModelWarmupJobDefaultSingleFileSourcePath)),
		shellQuoteSingle(cfg.TimeoutSourcePath),
		shellQuoteSingle(cfg.TargetPath),
	)
	if _, err := runPodShellCommand(primarySourcePod.Namespace, primarySourcePod.Name, primaryScript); err != nil {
		return fmt.Errorf("准备主源测试数据失败: %w", err)
	}
	secondaryScript := fmt.Sprintf(`set -eu
root=%s
alt=%s
mismatch=%s
ready_root=%s
if [ ! -e "$root" ]; then
  echo "source path is not visible in secondary node-agent: $root" >&2
  ls -ld /home /home/llm_cache /home/llm_cache/huggingface 2>&2 || true
  mount | grep -E 'llm_cache|weight-dispatcher' >&2 || true
  exit 1
fi
rm -rf "$(dirname "$alt")" "$(dirname "$mismatch")" "$ready_root"
mkdir -p "$(dirname "$alt")" "$mismatch" "$ready_root"
ln -sfn "$root" "$alt"
printf '{"model":"mismatch"}\n' > "$mismatch/config.json"
printf 'bad\n' > "$mismatch/extra.txt"`,
		shellQuoteSingle(cfg.SourcePath),
		shellQuoteSingle(cfg.SecondSourcePath),
		shellQuoteSingle(cfg.MismatchSourcePath),
		shellQuoteSingle(cfg.TargetPath),
	)
	if _, err := runPodShellCommand(secondarySourcePod.Namespace, secondarySourcePod.Name, secondaryScript); err != nil {
		return fmt.Errorf("准备次源测试数据失败: %w", err)
	}
	return nil
}

func WaitForWeightDispatcherNodeAgentDaemonSetsReady(clientset *kubernetes.Clientset, ctx context.Context, timeout, interval time.Duration) {
	Eventually(func() bool {
		daemonSets, err := clientset.AppsV1().DaemonSets(metav1.NamespaceAll).List(ctx, metav1.ListOptions{})
		if err != nil {
			GinkgoWriter.Printf("查询 node-agent DaemonSet 失败: %v\n", err)
			return false
		}
		matched := 0
		for _, ds := range daemonSets.Items {
			if !isWeightDispatcherNodeAgentDaemonSet(ds) {
				continue
			}
			matched++
			desired := ds.Status.DesiredNumberScheduled
			if desired == 0 ||
				ds.Status.ObservedGeneration < ds.Generation ||
				ds.Status.UpdatedNumberScheduled != desired ||
				ds.Status.NumberReady != desired ||
				ds.Status.NumberUnavailable != 0 {
				GinkgoWriter.Printf("node-agent DaemonSet %s/%s rollout 未完成: desired=%d updated=%d ready=%d unavailable=%d observed=%d generation=%d\n",
					ds.Namespace, ds.Name, desired, ds.Status.UpdatedNumberScheduled, ds.Status.NumberReady, ds.Status.NumberUnavailable, ds.Status.ObservedGeneration, ds.Generation)
				return false
			}
		}
		return matched > 0
	}, timeout, interval).Should(BeTrue(), "weight-dispatcher node-agent DaemonSet 应完成滚动更新")
	VerifyWeightDispatcherPodsRunning(clientset, ctx, timeout, interval)
}

func CreateModelWarmupJob(dynamicClient dynamic.Interface, ctx context.Context, namespace string, job *unstructured.Unstructured) error {
	_, err := dynamicClient.Resource(modelWarmupJobGVR).Namespace(namespace).Create(ctx, job, metav1.CreateOptions{})
	return err
}

func GetModelWarmupJob(dynamicClient dynamic.Interface, ctx context.Context, namespace, name string) (*unstructured.Unstructured, error) {
	return dynamicClient.Resource(modelWarmupJobGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
}

func DeleteModelWarmupJob(dynamicClient dynamic.Interface, ctx context.Context, namespace, name string) error {
	err := dynamicClient.Resource(modelWarmupJobGVR).Namespace(namespace).Delete(ctx, name, metav1.DeleteOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	return nil
}

func CleanReadyCacheOnAllNodeAgents(clientset *kubernetes.Clientset, ctx context.Context, readyRoot string) error {
	return CleanReadyCacheOnAllNodeAgentsExcept(clientset, ctx, readyRoot, "")
}

func CleanReadyCacheOnAllNodeAgentsExcept(clientset *kubernetes.Clientset, ctx context.Context, readyRoot, preservePath string) error {
	pods, err := listWeightDispatcherNodeAgentPods(clientset, ctx)
	if err != nil {
		return err
	}
	if len(pods) == 0 {
		GinkgoWriter.Printf("未找到 weight-dispatcher node-agent Pod，跳过 READY 清理: %s\n", readyRoot)
		return nil
	}
	script := fmt.Sprintf(
		`set -eu
root=%s
preserve=%s
mkdir -p "$root"
find "$root" -mindepth 1 -maxdepth 1 | while IFS= read -r entry; do
  if [ -n "$preserve" ] && [ "$entry" = "$preserve" ]; then
    continue
  fi
  rm -rf -- "$entry"
done`,
		shellQuoteSingle(readyRoot),
		shellQuoteSingle(preservePath),
	)
	for _, pod := range pods {
		if _, err := runPodShellCommand(pod.Namespace, pod.Name, script); err != nil {
			return fmt.Errorf("清理节点 %s 上 READY 目录失败: %w", pod.Spec.NodeName, err)
		}
	}
	return nil
}

func WaitForModelWarmupJobSucceeded(dynamicClient dynamic.Interface, ctx context.Context, namespace, name string, timeout time.Duration) *unstructured.Unstructured {
	var latest *unstructured.Unstructured
	Eventually(func() string {
		obj, err := GetModelWarmupJob(dynamicClient, ctx, namespace, name)
		if err != nil {
			GinkgoWriter.Printf("获取 ModelWarmupJob %s/%s 失败: %v\n", namespace, name, err)
			return ""
		}
		latest = obj
		phase, _, _ := unstructured.NestedString(obj.Object, "status", "phase")
		if phase == "Failed" {
			message, _, _ := unstructured.NestedString(obj.Object, "status", "lastErrorMessage")
			GinkgoWriter.Printf("ModelWarmupJob %s/%s 进入 Failed: %s\n", namespace, name, message)
		}
		return phase
	}, timeout, 10*time.Second).Should(Equal("Succeeded"), "ModelWarmupJob %s/%s 应最终 Succeeded", namespace, name)
	return latest
}

func WaitForModelWarmupJobRunning(dynamicClient dynamic.Interface, ctx context.Context, namespace, name string, timeout time.Duration) *unstructured.Unstructured {
	var latest *unstructured.Unstructured
	Eventually(func() string {
		obj, err := GetModelWarmupJob(dynamicClient, ctx, namespace, name)
		if err != nil {
			GinkgoWriter.Printf("获取 ModelWarmupJob %s/%s 失败: %v\n", namespace, name, err)
			return ""
		}
		latest = obj
		phase, _, _ := unstructured.NestedString(obj.Object, "status", "phase")
		return phase
	}, timeout, 5*time.Second).Should(Equal("Running"), "ModelWarmupJob %s/%s 应进入 Running 状态", namespace, name)
	return latest
}

func WaitForModelWarmupJobFailed(dynamicClient dynamic.Interface, ctx context.Context, namespace, name string, timeout time.Duration) *unstructured.Unstructured {
	var latest *unstructured.Unstructured
	Eventually(func() string {
		obj, err := GetModelWarmupJob(dynamicClient, ctx, namespace, name)
		if err != nil {
			GinkgoWriter.Printf("获取 ModelWarmupJob %s/%s 失败: %v\n", namespace, name, err)
			return ""
		}
		latest = obj
		phase, _, _ := unstructured.NestedString(obj.Object, "status", "phase")
		return phase
	}, timeout, 10*time.Second).Should(Equal("Failed"), "ModelWarmupJob %s/%s 应进入 Failed 状态", namespace, name)
	return latest
}

func PatchModelWarmupJobChunkSize(dynamicClient dynamic.Interface, ctx context.Context, namespace, name string, chunkSizeMB int64) (*unstructured.Unstructured, error) {
	var lastErr error
	for i := 0; i < 5; i++ {
		obj, err := GetModelWarmupJob(dynamicClient, ctx, namespace, name)
		if err != nil {
			return nil, err
		}
		if err := unstructured.SetNestedField(obj.Object, chunkSizeMB, "spec", "policy", "chunkSizeMB"); err != nil {
			return nil, err
		}
		updated, err := dynamicClient.Resource(modelWarmupJobGVR).Namespace(namespace).Update(ctx, obj, metav1.UpdateOptions{})
		if err == nil {
			return updated, nil
		}
		if !apierrors.IsConflict(err) {
			return nil, err
		}
		lastErr = err
		time.Sleep(500 * time.Millisecond)
	}
	return nil, lastErr
}

func VerifySingleNodeWarmupSucceeded(job *unstructured.Unstructured, cfg ModelWarmupJobE2EConfig) string {
	VerifyModelWarmupJobSummary(job, 1, 1, 0)
	nodeState := firstNodeState(job)
	Expect(stringFromMap(nodeState, "taskID")).NotTo(BeEmpty(), "nodeState.taskID 应有值")
	cachePath := stringFromMap(nodeState, "cachePath")
	Expect(cachePath).NotTo(BeEmpty(), "nodeState.cachePath 应有值")
	Expect(cachePath).To(HavePrefix(cfg.TargetPath), "cachePath 应位于 spec.target.targetPath 下方")
	Expect(path.Base(cachePath)).To(SatisfyAny(Equal(cfg.ArtifactKey), Equal(SanitizeArtifactKey(cfg.ArtifactKey))),
		"cachePath 最后一级目录应对应 artifact.key")
	Expect(stringFromMap(nodeState, "startedAt")).NotTo(BeEmpty(), "nodeState.startedAt 应有值")
	Expect(stringFromMap(nodeState, "finishedAt")).NotTo(BeEmpty(), "nodeState.finishedAt 应有值")
	Expect(numberFromMap(nodeState, "bytesTransferred")).To(BeNumerically(">=", 0), "nodeState.bytesTransferred 应有值")
	Expect(numberFromMap(nodeState, "throughputMBps")).To(BeNumerically(">=", 0), "nodeState.throughputMBps 应有值")
	return cachePath
}

func VerifyRunningStatusQueryable(job *unstructured.Unstructured) {
	phase, _, _ := unstructured.NestedString(job.Object, "status", "phase")
	Expect(phase).To(Equal("Running"), "运行中阶段 status.phase 应为 Running")
	pending := nestedNumberOrZero(job.Object, "status", "summary", "pending")
	running := nestedNumberOrZero(job.Object, "status", "summary", "running")
	Expect(pending+running).To(BeNumerically(">", 0), "运行中阶段 summary.running 或 summary.pending 应大于 0")
}

func VerifyNodeStateStatistics(job *unstructured.Unstructured) {
	for _, state := range nodeStates(job) {
		Expect(stringFromMap(state, "startedAt")).NotTo(BeEmpty(), "nodeState.startedAt 应有值")
		Expect(stringFromMap(state, "finishedAt")).NotTo(BeEmpty(), "nodeState.finishedAt 应有值")
		Expect(numberFromMap(state, "bytesTransferred")).To(BeNumerically(">=", 0), "nodeState.bytesTransferred 应有值")
		Expect(numberFromMap(state, "throughputMBps")).To(BeNumerically(">=", 0), "nodeState.throughputMBps 应有值")
	}
	VerifySummaryMatchesNodeStates(job)
}

func VerifySummaryMatchesNodeStates(job *unstructured.Unstructured) {
	counts := map[string]int64{}
	for _, state := range nodeStates(job) {
		phase := strings.ToLower(stringFromMap(state, "phase"))
		counts[phase]++
	}
	for _, key := range []string{"pending", "running", "succeeded", "failed"} {
		Expect(nestedNumberOrZero(job.Object, "status", "summary", key)).To(BeNumerically("==", counts[key]), "summary.%s 应与 nodeStates 汇总一致", key)
	}
	Expect(nestedNumber(job.Object, "status", "summary", "total")).To(BeNumerically("==", len(nodeStates(job))), "summary.total 应与 nodeStates 数量一致")
}

func VerifyConcurrentWarmupSharedCache(jobA, jobB *unstructured.Unstructured, cfg ModelWarmupJobE2EConfig) {
	VerifyModelWarmupJobSummary(jobA, 1, 1, 0)
	VerifyModelWarmupJobSummary(jobB, 1, 1, 0)
	stateA := firstNodeState(jobA)
	stateB := firstNodeState(jobB)
	cachePathA := stringFromMap(stateA, "cachePath")
	cachePathB := stringFromMap(stateB, "cachePath")
	Expect(cachePathA).NotTo(BeEmpty(), "并发作业 A 应写入 cachePath")
	Expect(cachePathB).NotTo(BeEmpty(), "并发作业 B 应写入 cachePath")
	Expect(cachePathA).To(Equal(cachePathB), "同 artifact 并发预热应共享同一份 READY 缓存")
	Expect(cachePathA).To(HavePrefix(cfg.TargetPath), "READY 缓存应位于 targetPath 下方")
	bytesA := numberFromMapOrZero(stateA, "bytesTransferred")
	bytesB := numberFromMapOrZero(stateB, "bytesTransferred")
	Expect(bytesA == 0 || bytesB == 0).To(BeTrue(), "至少一个并发作业应体现共享等待或零字节成功")
}

func VerifyReadyCacheHitSucceeded(job *unstructured.Unstructured, cfg ModelWarmupJobE2EConfig, previousCachePath string) {
	VerifyModelWarmupJobSummary(job, 1, 1, 0)
	nodeState := firstNodeState(job)
	cachePath := stringFromMap(nodeState, "cachePath")
	Expect(cachePath).NotTo(BeEmpty(), "命中 READY 缓存后仍应写入 cachePath")
	Expect(cachePath).To(HavePrefix(cfg.TargetPath), "cachePath 应位于 spec.target.targetPath 下方")
	if previousCachePath != "" {
		Expect(cachePath).To(Equal(previousCachePath), "同规格新建作业应复用 FIT_2 生成的 READY 缓存目录")
	}
	Expect(numberFromMapOrZero(nodeState, "bytesTransferred")).To(BeNumerically("==", 0), "命中 READY 缓存后 bytesTransferred 应为 0")
}

func VerifyArtifactKeySanitizedPath(job *unstructured.Unstructured, cfg ModelWarmupJobE2EConfig) {
	nodeState := firstNodeState(job)
	cachePath := stringFromMap(nodeState, "cachePath")
	Expect(cachePath).To(HavePrefix(cfg.TargetPath), "cachePath 前缀应与 targetPath 一致")
	Expect(path.Base(cachePath)).To(Equal(SanitizeArtifactKey(cfg.ArtifactKey)), "artifact.key 特殊字符应被替换为下划线")
}

func VerifyChunkCRC32CSucceeded(job *unstructured.Unstructured, cfg ModelWarmupJobE2EConfig) {
	VerifyModelWarmupJobSummary(job, 1, 1, 0)
	Expect(stringFromMap(firstNodeState(job), "cachePath")).To(HavePrefix(cfg.TargetPath), "启用 CRC32C 后 READY 目录应发布成功")
}

func VerifyNodeSelectorResolved(job *unstructured.Unstructured, expectedNodes []string) {
	resolvedNodes := nestedStringSlice(job.Object, "status", "resolvedNodes")
	Expect(resolvedNodes).To(ConsistOf(stringInterfaces(expectedNodes)...), "status.resolvedNodes 应只包含命中 nodeSelector 的节点")

	nodeStates := nodeStates(job)
	stateNodes := make([]string, 0, len(nodeStates))
	for _, state := range nodeStates {
		stateNodes = append(stateNodes, stringFromMap(state, "nodeName"))
	}
	Expect(stateNodes).To(ConsistOf(stringInterfaces(expectedNodes)...), "nodeStates[*].nodeName 应与 resolvedNodes 一一对应")
	Expect(nestedNumber(job.Object, "status", "summary", "total")).To(BeNumerically("==", len(expectedNodes)), "summary.total 应与命中节点数量一致")
}

func VerifyMultiTargetWarmupSucceeded(job *unstructured.Unstructured, targetCount int, requireTransportPath bool) {
	VerifyModelWarmupJobSummary(job, int64(targetCount), int64(targetCount), 0)
	states := nodeStates(job)
	Expect(states).To(HaveLen(targetCount), "nodeStates 数量应与目标节点数量一致")
	for _, state := range states {
		Expect(stringFromMap(state, "cachePath")).NotTo(BeEmpty(), "每个目标节点应写入 cachePath")
		Expect(numberFromMap(state, "bytesTransferred")).To(BeNumerically(">=", 0), "每个目标节点应写入 bytesTransferred")
		if requireTransportPath {
			transportPath := stringFromMap(state, "transportPath")
			Expect(transportPath).To(BeElementOf("RDMA", "TCP_FALLBACK", "MIXED"), "transportPath 取值不符合预期")
		}
	}
}

func VerifyNoManifestMismatch(job *unstructured.Unstructured) {
	message, _, _ := unstructured.NestedString(job.Object, "status", "lastErrorMessage")
	Expect(strings.ToLower(message)).NotTo(ContainSubstring("manifest mismatch"), "异构源根路径对齐后不应出现 manifest mismatch")
}

func VerifyModelWarmupJobRejected(err error, dynamicClient dynamic.Interface, ctx context.Context, namespace, name string) {
	Expect(err).To(HaveOccurred(), "非法 ModelWarmupJob 创建请求应被 CRD/OpenAPI 拒绝")
	_, getErr := GetModelWarmupJob(dynamicClient, ctx, namespace, name)
	Expect(apierrors.IsNotFound(getErr)).To(BeTrue(), "被拒绝的 ModelWarmupJob 不应落库")
}

func VerifyValidationFailed(job *unstructured.Unstructured, expectedMessage string) {
	phase, _, _ := unstructured.NestedString(job.Object, "status", "phase")
	Expect(phase).To(Equal("Failed"), "非法规格应进入 Failed 状态")
	errorCode, _, _ := unstructured.NestedString(job.Object, "status", "lastErrorCode")
	Expect(errorCode).To(Equal(ModelWarmupJobValidationFailed), "lastErrorCode 应为 ValidationFailed")
	VerifyFailureMessageContains(job, expectedMessage)
}

func VerifyFailureMessageContains(job *unstructured.Unstructured, expectedMessages ...string) {
	message, _, _ := unstructured.NestedString(job.Object, "status", "lastErrorMessage")
	lowerMessage := strings.ToLower(message)
	for _, expected := range expectedMessages {
		Expect(lowerMessage).To(ContainSubstring(strings.ToLower(expected)), "lastErrorMessage 应包含 %s", expected)
	}
}

func VerifyFailureMessageContainsAny(job *unstructured.Unstructured, expectedMessages ...string) {
	message, _, _ := unstructured.NestedString(job.Object, "status", "lastErrorMessage")
	errorCode, _, _ := unstructured.NestedString(job.Object, "status", "lastErrorCode")
	lowerMessage := strings.ToLower(message)
	matched := false
	for _, expected := range expectedMessages {
		if strings.Contains(lowerMessage, strings.ToLower(expected)) {
			matched = true
			break
		}
	}
	Expect(matched).To(BeTrue(),
		"lastErrorMessage 应包含任一等价错误信息: %s, actualCode=%q actualMessage=%q",
		strings.Join(expectedMessages, ", "), errorCode, message)
}

func VerifyFailureMessageContainsAnyIfPresent(job *unstructured.Unstructured, expectedMessages ...string) {
	message, _, _ := unstructured.NestedString(job.Object, "status", "lastErrorMessage")
	if strings.TrimSpace(message) == "" {
		errorCode, _, _ := unstructured.NestedString(job.Object, "status", "lastErrorCode")
		GinkgoWriter.Printf("失败任务未写入 lastErrorMessage，跳过消息语义校验: actualCode=%q\n", errorCode)
		return
	}
	VerifyFailureMessageContainsAny(job, expectedMessages...)
}

func VerifyManifestMismatchBlocked(job *unstructured.Unstructured) {
	phase, _, _ := unstructured.NestedString(job.Object, "status", "phase")
	Expect(phase).NotTo(Equal("Succeeded"), "manifest 不一致时作业不应进入成功态")
	VerifyFailureMessageContains(job, "manifest mismatch")
}

func VerifyModelWarmupJobNotFailed(job *unstructured.Unstructured) {
	phase, _, _ := unstructured.NestedString(job.Object, "status", "phase")
	Expect(phase).NotTo(Equal("Failed"), "node-agent 短暂不可达期间作业不应立即 Failed")
}

func VerifyTimeoutFailedAndNoReadyCache(job *unstructured.Unstructured) {
	phase, _, _ := unstructured.NestedString(job.Object, "status", "phase")
	Expect(phase).To(Equal("Failed"), "超小 timeout 应触发 Failed")
	VerifyFailureMessageContainsAnyIfPresent(job, "timeout", "deadline", "context canceled")
	for _, state := range optionalNodeStates(job) {
		cachePath := ""
		if value, ok := state["cachePath"].(string); ok {
			cachePath = value
		}
		Expect(cachePath).To(BeEmpty(), "失败任务不应发布 READY cachePath")
	}
}

func VerifyChunkCRCRetrySucceeded(job *unstructured.Unstructured, cfg ModelWarmupJobE2EConfig) {
	VerifyChunkCRC32CSucceeded(job, cfg)
}

func VerifyTCPFallbackSucceeded(job *unstructured.Unstructured) {
	VerifyModelWarmupJobSummary(job, 1, 1, 0)
	transportPath := stringFromMap(firstNodeState(job), "transportPath")
	Expect(transportPath).To(Equal(ModelWarmupJobTCPFallbackPath), "RDMA 不可用时应回退 TCP_FALLBACK")
}

func VerifyJobTransferredFilesMatchSource(clientset *kubernetes.Clientset, ctx context.Context, job *unstructured.Unstructured) {
	sourceNode, sourcePath := firstJobSource(job)
	artifactKey := firstJobArtifactKey(job)
	sourcePod, err := weightDispatcherNodeAgentPodForNode(clientset, ctx, sourceNode)
	Expect(err).NotTo(HaveOccurred())

	expected, err := collectFileDigestsFromPod(sourcePod.Namespace, sourcePod.Name, sourcePath, "")
	Expect(err).NotTo(HaveOccurred())
	Expect(expected).NotTo(BeEmpty(), "源目录应至少包含一个文件")

	for _, state := range nodeStates(job) {
		targetNode := stringFromMap(state, "nodeName")
		cachePath := stringFromMap(state, "cachePath")
		targetPod, err := weightDispatcherNodeAgentPodForNode(clientset, ctx, targetNode)
		Expect(err).NotTo(HaveOccurred())

		actual, err := collectFileDigestsFromPod(targetPod.Namespace, targetPod.Name, cachePath, artifactKey)
		Expect(err).NotTo(HaveOccurred(), "目标节点 %s 的缓存目录校验失败", targetNode)
		GinkgoWriter.Printf("校验目标缓存: node=%s artifactKey=%s cachePath=%s expectedFiles=%d actualFiles=%d\n",
			targetNode, artifactKey, cachePath, len(expected), len(actual))
		Expect(compareFileDigests(expected, actual)).NotTo(HaveOccurred(),
			"目标节点 %s 的缓存内容与源目录不一致", targetNode)
	}
}

func LastPlanID(job *unstructured.Unstructured) string {
	value, _, _ := unstructured.NestedString(job.Object, "status", "lastPlanId")
	if value == "" {
		value, _, _ = unstructured.NestedString(job.Object, "status", "lastPlanID")
	}
	return value
}

func FirstTaskID(job *unstructured.Unstructured) string {
	return stringFromMap(firstNodeState(job), "taskID")
}

func WaitForModelWarmupJobReplannedAndSucceeded(dynamicClient dynamic.Interface, ctx context.Context, namespace, name, oldPlanID, oldTaskID string, timeout time.Duration) *unstructured.Unstructured {
	var latest *unstructured.Unstructured
	Eventually(func() bool {
		obj, err := GetModelWarmupJob(dynamicClient, ctx, namespace, name)
		if err != nil {
			GinkgoWriter.Printf("获取 ModelWarmupJob %s/%s 失败: %v\n", namespace, name, err)
			return false
		}
		latest = obj
		generation, _, _ := unstructured.NestedInt64(obj.Object, "metadata", "generation")
		observedGeneration, _, _ := unstructured.NestedInt64(obj.Object, "status", "observedGeneration")
		phase, _, _ := unstructured.NestedString(obj.Object, "status", "phase")
		planID := LastPlanID(obj)
		taskID := FirstTaskID(obj)
		return phase == "Succeeded" &&
			observedGeneration >= generation &&
			(oldPlanID == "" || planID != "" && planID != oldPlanID) &&
			(oldTaskID == "" || taskID != "" && taskID != oldTaskID)
	}, timeout, 10*time.Second).Should(BeTrue(), "更新策略后应重建计划、observedGeneration 追平并重新 Succeeded")
	return latest
}

func VerifyModelWarmupJobSummary(job *unstructured.Unstructured, total, succeeded, failed int64) {
	Expect(nestedNumber(job.Object, "status", "summary", "total")).To(BeNumerically("==", total), "status.summary.total 不符合预期")
	Expect(nestedNumber(job.Object, "status", "summary", "succeeded")).To(BeNumerically("==", succeeded), "status.summary.succeeded 不符合预期")
	Expect(nestedNumberOrZero(job.Object, "status", "summary", "failed")).To(BeNumerically("==", failed), "status.summary.failed 不符合预期")
}

func SanitizeArtifactKey(key string) string {
	replacer := strings.NewReplacer("/", "_", ":", "_", " ", "_")
	return replacer.Replace(key)
}

func getHelmList() ([]HelmRelease, error) {
	cmd := exec.Command("helm", "list", "--all-namespaces", "-o", "json")
	var out bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("helm list 执行失败: %v\nStderr: %s", err, stderr.String())
	}

	var releases []HelmRelease
	if err := json.Unmarshal(out.Bytes(), &releases); err != nil {
		return nil, fmt.Errorf("解析 helm list JSON 失败: %v\nOutput: %s", err, out.String())
	}
	return releases, nil
}

func listWeightDispatcherNodeAgentPods(clientset *kubernetes.Clientset, ctx context.Context) ([]corev1.Pod, error) {
	pods, err := clientset.CoreV1().Pods(metav1.NamespaceAll).List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	result := make([]corev1.Pod, 0)
	for _, pod := range pods.Items {
		if !isWeightDispatcherPod(pod) {
			continue
		}
		name := strings.ToLower(pod.Name)
		component := strings.ToLower(pod.Labels["app.kubernetes.io/component"])
		if strings.Contains(name, WeightDispatcherNodeAgentKey) || strings.Contains(component, "agent") {
			result = append(result, pod)
		}
	}
	return result, nil
}

func weightDispatcherNodeAgentPodForNode(clientset *kubernetes.Clientset, ctx context.Context, nodeName string) (*corev1.Pod, error) {
	pods, err := listWeightDispatcherNodeAgentPods(clientset, ctx)
	if err != nil {
		return nil, err
	}
	for i := range pods {
		if pods[i].Spec.NodeName == nodeName {
			return &pods[i], nil
		}
	}
	return nil, fmt.Errorf("未找到节点 %s 上的 weight-dispatcher node-agent Pod", nodeName)
}

func firstJobSource(job *unstructured.Unstructured) (string, string) {
	sources, found, err := unstructured.NestedSlice(job.Object, "spec", "sources")
	Expect(err).NotTo(HaveOccurred())
	Expect(found).To(BeTrue(), "spec.sources 应存在")
	Expect(sources).NotTo(BeEmpty(), "spec.sources 应至少包含一个 source")
	source, ok := sources[0].(map[string]interface{})
	Expect(ok).To(BeTrue(), "spec.sources[0] 应为对象")
	return stringFromMap(source, "nodeName"), stringFromMap(source, "path")
}

func firstJobArtifactKey(job *unstructured.Unstructured) string {
	artifactKey, found, err := unstructured.NestedString(job.Object, "spec", "artifact", "key")
	Expect(err).NotTo(HaveOccurred())
	Expect(found).To(BeTrue(), "spec.artifact.key 应存在")
	return artifactKey
}

func collectFileDigestsFromPod(namespace, podName, rootPath, artifactKey string) (map[string]fileDigest, error) {
	output, err := runPodShellCommand(namespace, podName, buildCollectFileDigestsScript(rootPath, artifactKey))
	if err != nil {
		return nil, err
	}
	return parseFileDigestManifest(output)
}

func buildCollectFileDigestsScript(rootPath, artifactKey string) string {
	return fmt.Sprintf(`set -eu
root=%s
artifact_key=%s
artifact_safe=%s
if [ -n "$artifact_safe" ] && [ -d "$root/$artifact_safe" ]; then
  root="$root/$artifact_safe"
elif [ -n "$artifact_key" ] && [ -d "$root/$artifact_key" ]; then
  root="$root/$artifact_key"
elif [ -n "$artifact_safe" ]; then
  candidate=$(find "$root" -mindepth 1 -maxdepth 2 -type d -name "$artifact_safe" | LC_ALL=C sort | head -n 1)
  if [ -n "$candidate" ]; then
    root="$candidate"
  fi
fi
if [ -f "$root" ]; then
  size=$(wc -c < "$root" | tr -d '[:space:]')
  sum=$(sha256sum "$root" | awk '{print $1}')
  printf '%%s\t%%s\t%%s\n' "$(basename "$root")" "$size" "$sum"
  exit 0
fi
test -d "$root"
if [ -d "$root/snapshots" ]; then
  revision=""
  if [ -f "$root/refs/main" ]; then
    revision=$(tr -d '[:space:]' < "$root/refs/main")
  fi
  if [ -n "$revision" ] && [ -d "$root/snapshots/$revision" ]; then
    root="$root/snapshots/$revision"
  else
    snapshot_root=$(find "$root/snapshots" -mindepth 1 -maxdepth 1 -type d | LC_ALL=C sort | head -n 1)
    if [ -n "$snapshot_root" ]; then
      root="$snapshot_root"
    fi
  fi
fi
if [ ! -f "$root/config.json" ]; then
  model_root=$(find -L "$root" -mindepth 1 -maxdepth 6 -type f -name 'config.json' | LC_ALL=C sort | awk '
    /\/snapshots\// { print; found=1; exit }
    NR == 1 { first=$0 }
    END { if (!found && first != "") print first }
  ')
  if [ -n "$model_root" ]; then
    root=$(dirname "$model_root")
  fi
fi
cd "$root"
find -L . -type f \
  ! -name '*.wdfanout.ready' \
  ! -name '*.wdfanout.done' | LC_ALL=C sort | while IFS= read -r file; do
  size=$(wc -c < "$file" | tr -d '[:space:]')
  sum=$(sha256sum "$file" | awk '{print $1}')
  printf '%%s\t%%s\t%%s\n' "${file#./}" "$size" "$sum"
done`, shellQuoteSingle(rootPath), shellQuoteSingle(artifactKey), shellQuoteSingle(SanitizeArtifactKey(artifactKey)))
}

func parseFileDigestManifest(raw string) (map[string]fileDigest, error) {
	manifest := make(map[string]fileDigest)
	for _, line := range strings.Split(strings.TrimSpace(raw), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		fields := strings.Split(line, "\t")
		if len(fields) != 3 {
			return nil, fmt.Errorf("无法解析文件摘要输出: %q", line)
		}
		sizeBytes, err := strconv.ParseInt(fields[1], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("解析文件大小失败(%q): %w", line, err)
		}
		manifest[fields[0]] = fileDigest{SizeBytes: sizeBytes, SHA256: fields[2]}
	}
	return manifest, nil
}

func compareFileDigests(expected, actual map[string]fileDigest) error {
	if len(expected) != len(actual) {
		return fmt.Errorf("文件数量不一致: expected=%d actual=%d", len(expected), len(actual))
	}
	if len(expected) == 1 && len(actual) == 1 {
		var expectedPath string
		var expectedDigest fileDigest
		for relativePath, digest := range expected {
			expectedPath = relativePath
			expectedDigest = digest
		}
		var actualPath string
		var actualDigest fileDigest
		for relativePath, digest := range actual {
			actualPath = relativePath
			actualDigest = digest
		}
		if expectedDigest.SizeBytes != actualDigest.SizeBytes {
			return fmt.Errorf("单文件大小不一致: expectedPath=%s actualPath=%s expected=%d actual=%d",
				expectedPath, actualPath, expectedDigest.SizeBytes, actualDigest.SizeBytes)
		}
		if expectedDigest.SHA256 != actualDigest.SHA256 {
			return fmt.Errorf("单文件 sha256 不一致: expectedPath=%s actualPath=%s expected=%s actual=%s",
				expectedPath, actualPath, expectedDigest.SHA256, actualDigest.SHA256)
		}
		return nil
	}
	for relativePath, expectedDigest := range expected {
		actualDigest, ok := actual[relativePath]
		if !ok {
			return fmt.Errorf("目标缺少文件: %s", relativePath)
		}
		if expectedDigest.SizeBytes != actualDigest.SizeBytes {
			return fmt.Errorf("文件大小不一致: %s expected=%d actual=%d",
				relativePath, expectedDigest.SizeBytes, actualDigest.SizeBytes)
		}
		if expectedDigest.SHA256 != actualDigest.SHA256 {
			return fmt.Errorf("文件 sha256 不一致: %s expected=%s actual=%s",
				relativePath, expectedDigest.SHA256, actualDigest.SHA256)
		}
	}
	return nil
}

func runPodShellCommand(namespace, podName, script string) (string, error) {
	args := []string{"exec", "-n", namespace, podName, "--", "sh", "-lc", script}
	cmd := exec.Command("kubectl", args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("kubectl %s 失败: %v\nOutput: %s", strings.Join(args, " "), err, string(out))
	}
	return string(out), nil
}

func shellQuoteSingle(value string) string {
	return "'" + strings.ReplaceAll(value, "'", `'"'"'`) + "'"
}

type weightDispatcherPodStatus struct {
	hasController bool
	hasNodeAgent  bool
	ready         bool
	total         int
}

func judgeWeightDispatcherPods(clientset *kubernetes.Clientset, ctx context.Context) (weightDispatcherPodStatus, error) {
	pods, err := clientset.CoreV1().Pods(metav1.NamespaceAll).List(ctx, metav1.ListOptions{})
	if err != nil {
		return weightDispatcherPodStatus{}, err
	}

	status := weightDispatcherPodStatus{ready: true}
	for _, pod := range pods.Items {
		if !isWeightDispatcherPod(pod) {
			continue
		}
		status.total++
		name := strings.ToLower(pod.Name)
		component := strings.ToLower(pod.Labels["app.kubernetes.io/component"])
		if strings.Contains(name, WeightDispatcherControllerKey) || component == WeightDispatcherControllerKey {
			status.hasController = true
		}
		if strings.Contains(name, WeightDispatcherNodeAgentKey) || strings.Contains(component, "agent") {
			status.hasNodeAgent = true
		}
		if pod.Status.Phase != corev1.PodRunning || !isPodReady(pod) {
			status.ready = false
		}
	}
	status.ready = status.ready && status.total > 0 && status.hasController && status.hasNodeAgent
	return status, nil
}

func isWeightDispatcherPod(pod corev1.Pod) bool {
	name := strings.ToLower(pod.Name)
	if strings.Contains(name, WeightDispatcherPodKeyword) {
		return true
	}
	return pod.Labels["app.kubernetes.io/instance"] == WeightDispatcherHelmReleaseName ||
		pod.Labels["app.kubernetes.io/name"] == WeightDispatcherHelmReleaseName
}

func isWeightDispatcherNodeAgentDaemonSet(ds appsv1.DaemonSet) bool {
	name := strings.ToLower(ds.Name)
	component := strings.ToLower(ds.Labels["app.kubernetes.io/component"])
	appName := ds.Labels["app.kubernetes.io/name"]
	instance := ds.Labels["app.kubernetes.io/instance"]
	if !strings.Contains(name, WeightDispatcherPodKeyword) &&
		appName != WeightDispatcherHelmReleaseName &&
		instance != WeightDispatcherHelmReleaseName {
		return false
	}
	return strings.Contains(name, WeightDispatcherNodeAgentKey) || strings.Contains(component, "agent")
}

func setContainerEnvs(container *corev1.Container, envs map[string]string) bool {
	changed := false
	for key, value := range envs {
		found := false
		for i := range container.Env {
			if container.Env[i].Name != key {
				continue
			}
			found = true
			if container.Env[i].Value != value || container.Env[i].ValueFrom != nil {
				container.Env[i].Value = value
				container.Env[i].ValueFrom = nil
				changed = true
			}
			break
		}
		if !found {
			container.Env = append(container.Env, corev1.EnvVar{Name: key, Value: value})
			changed = true
		}
	}
	return changed
}

func nodeAgentDaemonSetEnvPatch(containers []corev1.Container) ([]byte, error) {
	containerPatches := make([]map[string]interface{}, 0, len(containers))
	for _, container := range containers {
		containerPatches = append(containerPatches, map[string]interface{}{
			"name": container.Name,
			"env":  container.Env,
		})
	}
	return json.Marshal(map[string]interface{}{
		"spec": map[string]interface{}{
			"template": map[string]interface{}{
				"spec": map[string]interface{}{
					"containers": containerPatches,
				},
			},
		},
	})
}

func readyNodeNames(clientset *kubernetes.Clientset, ctx context.Context) ([]string, error) {
	nodeList, err := clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(nodeList.Items))
	for _, node := range nodeList.Items {
		for _, condition := range node.Status.Conditions {
			if condition.Type == corev1.NodeReady && condition.Status == corev1.ConditionTrue {
				names = append(names, node.Name)
				break
			}
		}
	}
	sort.Strings(names)
	return names, nil
}

func firstPreferredReadyNode(readyNodes, preferred []string) string {
	ready := make(map[string]struct{}, len(readyNodes))
	for _, node := range readyNodes {
		ready[node] = struct{}{}
	}
	for _, node := range preferred {
		if _, ok := ready[node]; ok {
			return node
		}
	}
	return ""
}

func preferredReadyNodesExcluding(readyNodes, preferred []string, excluded string, count int) []string {
	ready := make(map[string]struct{}, len(readyNodes))
	for _, node := range readyNodes {
		ready[node] = struct{}{}
	}
	result := make([]string, 0, count)
	for _, node := range preferred {
		if _, ok := ready[node]; !ok || node == excluded {
			continue
		}
		result = appendUniqueStrings(result, node)
		if len(result) == count {
			return result
		}
	}
	for _, node := range readyNodes {
		if node == excluded {
			continue
		}
		result = appendUniqueStrings(result, node)
		if len(result) == count {
			return result
		}
	}
	return result
}

func appendUniqueStrings(values []string, candidates ...string) []string {
	seen := make(map[string]struct{}, len(values)+len(candidates))
	for _, value := range values {
		seen[value] = struct{}{}
	}
	for _, candidate := range candidates {
		if candidate == "" {
			continue
		}
		if _, ok := seen[candidate]; ok {
			continue
		}
		values = append(values, candidate)
		seen[candidate] = struct{}{}
	}
	return values
}

func splitEnvList(key string) []string {
	raw := os.Getenv(key)
	if raw == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	values := make([]string, 0, len(parts))
	for _, part := range parts {
		value := strings.TrimSpace(part)
		if value != "" {
			values = append(values, value)
		}
	}
	return values
}

func isPodReady(pod corev1.Pod) bool {
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady && condition.Status == corev1.ConditionTrue {
			return true
		}
	}
	return false
}

func getEnvOrDefault(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func firstNodeState(job *unstructured.Unstructured) map[string]interface{} {
	states := nodeStates(job)
	return states[0]
}

func nodeStates(job *unstructured.Unstructured) []map[string]interface{} {
	nodeStates, found, err := unstructured.NestedSlice(job.Object, "status", "nodeStates")
	Expect(err).NotTo(HaveOccurred())
	Expect(found).To(BeTrue(), "status.nodeStates 应存在")
	Expect(nodeStates).NotTo(BeEmpty(), "status.nodeStates 应至少包含一个节点状态")
	states := make([]map[string]interface{}, 0, len(nodeStates))
	for i, item := range nodeStates {
		nodeState, ok := item.(map[string]interface{})
		Expect(ok).To(BeTrue(), "status.nodeStates[%d] 应为对象", i)
		states = append(states, nodeState)
	}
	return states
}

func optionalNodeStates(job *unstructured.Unstructured) []map[string]interface{} {
	nodeStates, found, err := unstructured.NestedSlice(job.Object, "status", "nodeStates")
	Expect(err).NotTo(HaveOccurred())
	if !found {
		return nil
	}
	states := make([]map[string]interface{}, 0, len(nodeStates))
	for i, item := range nodeStates {
		nodeState, ok := item.(map[string]interface{})
		Expect(ok).To(BeTrue(), "status.nodeStates[%d] 应为对象", i)
		states = append(states, nodeState)
	}
	return states
}

func stringFromMap(data map[string]interface{}, key string) string {
	value, ok := data[key].(string)
	Expect(ok).To(BeTrue(), "%s 应为字符串", key)
	return value
}

func numberFromMap(data map[string]interface{}, key string) float64 {
	value, ok := data[key]
	Expect(ok).To(BeTrue(), "%s 应有值", key)
	return numberValue(value, key)
}

func numberFromMapOrZero(data map[string]interface{}, key string) float64 {
	value, ok := data[key]
	if !ok {
		return 0
	}
	return numberValue(value, key)
}

func nestedNumber(data map[string]interface{}, fields ...string) float64 {
	value, found, err := unstructured.NestedFieldNoCopy(data, fields...)
	Expect(err).NotTo(HaveOccurred())
	Expect(found).To(BeTrue(), "%s 应有值", strings.Join(fields, "."))
	return numberValue(value, strings.Join(fields, "."))
}

func nestedNumberOrZero(data map[string]interface{}, fields ...string) float64 {
	value, found, err := unstructured.NestedFieldNoCopy(data, fields...)
	Expect(err).NotTo(HaveOccurred())
	if !found {
		return 0
	}
	return numberValue(value, strings.Join(fields, "."))
}

func nestedStringSlice(data map[string]interface{}, fields ...string) []string {
	values, found, err := unstructured.NestedStringSlice(data, fields...)
	if err == nil && found {
		return values
	}
	rawValues, found, err := unstructured.NestedSlice(data, fields...)
	Expect(err).NotTo(HaveOccurred())
	Expect(found).To(BeTrue(), "%s 应有值", strings.Join(fields, "."))
	values = make([]string, 0, len(rawValues))
	for _, raw := range rawValues {
		value, ok := raw.(string)
		Expect(ok).To(BeTrue(), "%s 应为字符串数组", strings.Join(fields, "."))
		values = append(values, value)
	}
	return values
}

func stringInterfaces(values []string) []interface{} {
	result := make([]interface{}, 0, len(values))
	for _, value := range values {
		result = append(result, value)
	}
	return result
}

func numberValue(value interface{}, key string) float64 {
	switch v := value.(type) {
	case int64:
		return float64(v)
	case int:
		return float64(v)
	case int32:
		return float64(v)
	case float64:
		return v
	case float32:
		return float64(v)
	default:
		Fail(fmt.Sprintf("%s 应为数字，实际类型 %T", key, value))
		return 0
	}
}
