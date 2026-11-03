package utils

import (
	"fmt"
	"strings"
	"testing"

	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

func TestShouldPreserveReadyCacheForSpec(t *testing.T) {
	cases := []struct {
		name string
		want bool
	}{
		{name: "ModelWarmupJob_FIT_2 单节点显式预热成功并发布 READY 缓存", want: false},
		{name: "ModelWarmupJob_FIT_3 同规格新建作业命中 READY 缓存", want: true},
		{name: "ModelWarmupJob_FIT_6 单源双目标权重分发成功", want: true},
		{name: "ModelWarmupJob_FIT_10 更新策略后重建计划并重新调谐", want: true},
		{name: "ModelWarmupJob_FIT_24 超小 timeout 触发任务失败且不发布 READY", want: false},
	}
	for _, tc := range cases {
		if got := ShouldPreserveReadyCacheForSpec(tc.name); got != tc.want {
			t.Fatalf("ShouldPreserveReadyCacheForSpec(%q) = %v, want %v", tc.name, got, tc.want)
		}
	}
}

func TestParseFileDigestManifest(t *testing.T) {
	manifest, err := parseFileDigestManifest("a.bin\t4\tabc\nsub/b.bin\t8\tdef\n")
	if err != nil {
		t.Fatalf("parseFileDigestManifest returned error: %v", err)
	}
	if len(manifest) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(manifest))
	}
	if got := manifest["a.bin"]; got.SizeBytes != 4 || got.SHA256 != "abc" {
		t.Fatalf("unexpected digest for a.bin: %#v", got)
	}
	if got := manifest["sub/b.bin"]; got.SizeBytes != 8 || got.SHA256 != "def" {
		t.Fatalf("unexpected digest for sub/b.bin: %#v", got)
	}
}

func TestCompareFileDigests(t *testing.T) {
	expected := map[string]fileDigest{
		"a.bin":     {SizeBytes: 4, SHA256: "abc"},
		"sub/b.bin": {SizeBytes: 8, SHA256: "def"},
	}
	if err := compareFileDigests(expected, expected); err != nil {
		t.Fatalf("compareFileDigests returned unexpected error: %v", err)
	}

	actual := map[string]fileDigest{
		"a.bin":     {SizeBytes: 4, SHA256: "abc"},
		"sub/b.bin": {SizeBytes: 9, SHA256: "xyz"},
	}
	if err := compareFileDigests(expected, actual); err == nil {
		t.Fatal("expected compareFileDigests to fail on mismatched digest")
	}
}

func TestCompareFileDigestsAllowsSingleFileRename(t *testing.T) {
	expected := map[string]fileDigest{
		"model.safetensors": {SizeBytes: 4, SHA256: "abc"},
	}
	actual := map[string]fileDigest{
		"model-00001-of-00005.safetensors": {SizeBytes: 4, SHA256: "abc"},
	}
	if err := compareFileDigests(expected, actual); err != nil {
		t.Fatalf("single file rename with same digest should pass: %v", err)
	}
}

func TestResolveDefaultSuiteRuntimeNodeLayout(t *testing.T) {
	cfg, err := resolveDefaultSuiteRuntimeConfigFromReadyNodes([]string{"n2", "m1", "n1"})
	if err != nil {
		t.Fatalf("resolveDefaultSuiteRuntimeConfigFromReadyNodes returned error: %v", err)
	}
	if cfg.SourceNode != "m1" || cfg.TargetNode != "n1" {
		t.Fatalf("unexpected primary node layout: source=%s target=%s", cfg.SourceNode, cfg.TargetNode)
	}
	if len(cfg.SourceNodes) != 2 || cfg.SourceNodes[0] != "m1" || cfg.SourceNodes[1] != "n2" {
		t.Fatalf("unexpected source nodes: %#v", cfg.SourceNodes)
	}
	if len(cfg.TargetNodes) != 2 || cfg.TargetNodes[0] != "n1" || cfg.TargetNodes[1] != "n2" {
		t.Fatalf("unexpected target nodes: %#v", cfg.TargetNodes)
	}
	if cfg.ArtifactKey != ModelWarmupJobDefaultArtifactKey ||
		cfg.SourcePath != ModelWarmupJobDefaultSourcePath ||
		cfg.SingleFileSourcePath != ModelWarmupJobDefaultSingleFileSourcePath {
		t.Fatalf("unexpected qwen3-8b defaults: %#v", cfg)
	}
}

func TestBuildInstallWeightDispatcherArgsUsesConfigDefaults(t *testing.T) {
	t.Setenv("WEIGHT_DISPATCHER_NODE_AGENT_CACHE_ROOT", "")
	t.Setenv("WEIGHT_DISPATCHER_NODE_AGENT_HOST_HOME_CACHE_ROOT", "")
	t.Setenv("WEIGHT_DISPATCHER_NODE_AGENT_RDMA_ENABLED", "")
	t.Setenv("WEIGHT_DISPATCHER_NODE_AGENT_TCP_FALLBACK_ENABLED", "")
	t.Setenv("WEIGHT_DISPATCHER_NODE_AGENT_HOST_NETWORK", "")
	t.Setenv("WEIGHT_DISPATCHER_NODE_AGENT_PRIVILEGED", "")
	t.Setenv("WEIGHT_DISPATCHER_NODE_AGENT_POST_WRITE_CRC32C_ENABLED", "")

	oldCfg := suiteRuntimeConfig
	suiteRuntimeConfig = WeightDispatcherSuiteRuntimeConfig{}
	t.Cleanup(func() {
		suiteRuntimeConfig = oldCfg
	})

	args := strings.Join(buildInstallWeightDispatcherArgs(), "\n")
	expected := []string{
		fmt.Sprintf("nodeAgent.cacheRoot=%s", WeightDispatcherDefaultNodeAgentCacheRoot),
		fmt.Sprintf("nodeAgent.hostHomeCacheRoot=%s", WeightDispatcherDefaultHostHomeCacheRoot),
		fmt.Sprintf("nodeAgent.rdmaEnabled=%s", WeightDispatcherDefaultRDMAEnabled),
		fmt.Sprintf("nodeAgent.tcpFallbackEnabled=%s", WeightDispatcherDefaultTCPFallbackEnabled),
		fmt.Sprintf("nodeAgent.hostNetwork=%s", WeightDispatcherDefaultHostNetwork),
		fmt.Sprintf("nodeAgent.privileged=%s", WeightDispatcherDefaultPrivileged),
		fmt.Sprintf("nodeAgent.postWriteCRC32CEnabled=%s", WeightDispatcherDefaultPostWriteCRC32CEnabled),
	}
	for _, item := range expected {
		if !strings.Contains(args, item) {
			t.Fatalf("expected helm args to contain %q, got:\n%s", item, args)
		}
	}
}

func TestBuildCollectFileDigestsScriptUsesHuggingFaceSnapshot(t *testing.T) {
	script := buildCollectFileDigestsScript(ModelWarmupJobDefaultSourcePath, "")
	expectedSnippets := []string{
		`if [ -d "$root/snapshots" ]; then`,
		`revision=$(tr -d '[:space:]' < "$root/refs/main")`,
		`root="$root/snapshots/$revision"`,
		`find -L . -type f`,
		`! -name '*.wdfanout.ready'`,
		`! -name '*.wdfanout.done'`,
	}
	for _, snippet := range expectedSnippets {
		if !strings.Contains(script, snippet) {
			t.Fatalf("expected digest script to contain %q, got:\n%s", snippet, script)
		}
	}
}

func TestBuildCollectFileDigestsScriptSelectsArtifactSubdir(t *testing.T) {
	artifactKey := "qwen/3:70b alpha"
	script := buildCollectFileDigestsScript(ModelWarmupJobDefaultTargetPath, artifactKey)
	expectedSnippets := []string{
		`artifact_key='qwen/3:70b alpha'`,
		`artifact_safe='qwen_3_70b_alpha'`,
		`if [ -n "$artifact_safe" ] && [ -d "$root/$artifact_safe" ]; then`,
		`elif [ -n "$artifact_key" ] && [ -d "$root/$artifact_key" ]; then`,
		`find "$root" -mindepth 1 -maxdepth 2 -type d -name "$artifact_safe"`,
		`if [ ! -f "$root/config.json" ]; then`,
		`find -L "$root" -mindepth 1 -maxdepth 6 -type f -name 'config.json'`,
	}
	for _, snippet := range expectedSnippets {
		if !strings.Contains(script, snippet) {
			t.Fatalf("expected digest script to contain %q, got:\n%s", snippet, script)
		}
	}
}

func TestVerifyReadyCacheHitAllowsMissingBytesTransferred(t *testing.T) {
	RegisterTestingT(t)

	cachePath := ModelWarmupJobDefaultTargetPath + "/" + ModelWarmupJobDefaultArtifactKey
	job := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"status": map[string]interface{}{
				"summary": map[string]interface{}{
					"total":     int64(1),
					"succeeded": int64(1),
					"failed":    int64(0),
				},
				"nodeStates": []interface{}{
					map[string]interface{}{
						"phase":     "Succeeded",
						"nodeName":  ModelWarmupJobDefaultTargetPath,
						"cachePath": cachePath,
					},
				},
			},
		},
	}

	VerifyReadyCacheHitSucceeded(job, ModelWarmupJobE2EConfig{
		ArtifactKey: ModelWarmupJobDefaultArtifactKey,
		TargetPath:  ModelWarmupJobDefaultTargetPath,
	}, cachePath)
}

func TestVerifyConcurrentWarmupSharedCacheAllowsMissingBytesTransferred(t *testing.T) {
	RegisterTestingT(t)

	cachePath := ModelWarmupJobDefaultTargetPath + "/" + ModelWarmupJobDefaultArtifactKey
	jobA := buildSucceededJobWithNodeState(map[string]interface{}{
		"phase":            "Succeeded",
		"cachePath":        cachePath,
		"bytesTransferred": int64(1024),
	})
	jobB := buildSucceededJobWithNodeState(map[string]interface{}{
		"phase":     "Succeeded",
		"cachePath": cachePath,
	})

	VerifyConcurrentWarmupSharedCache(jobA, jobB, ModelWarmupJobE2EConfig{
		TargetPath: ModelWarmupJobDefaultTargetPath,
	})
}

func buildSucceededJobWithNodeState(state map[string]interface{}) *unstructured.Unstructured {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"status": map[string]interface{}{
				"summary": map[string]interface{}{
					"total":     int64(1),
					"succeeded": int64(1),
					"failed":    int64(0),
				},
				"nodeStates": []interface{}{state},
			},
		},
	}
}

func TestVerifyModelWarmupJobSummaryAllowsMissingZeroFailedField(t *testing.T) {
	RegisterTestingT(t)

	job := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"status": map[string]interface{}{
				"summary": map[string]interface{}{
					"total":     int64(1),
					"succeeded": int64(1),
				},
			},
		},
	}

	VerifyModelWarmupJobSummary(job, 1, 1, 0)
}
