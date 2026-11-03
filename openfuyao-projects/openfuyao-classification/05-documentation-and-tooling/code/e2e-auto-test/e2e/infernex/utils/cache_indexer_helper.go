package utils

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	frameworkk8s "gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
	"gopkg.in/yaml.v3"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

const CacheIndexerHitRatePath = "/kv-cache/hit-rate"

type cacheIndexerHealthResponse struct {
	Status string `json:"status"`
}

type CacheIndexerRuntimeConfig struct {
	ServiceName string
	ServicePort int
}

type AggregatedBackendRuntimeConfig struct {
	ServiceName string
	ServicePort int
	ModelName   string
	BlockSize   int
}

type PDBackendRuntimeConfig struct {
	ServiceName      string
	ServicePort      int
	ModelName        string
	PrefillBlockSize int
	DecodeBlockSize  int
	PDGroupID        string
}

type AggregatedBackendPodIdentity struct {
	Name string
	IP   string
}

type CacheIndexerPodSnapshot struct {
	PodName      string
	RestartCount int32
}

type CacheIndexerHitRateRequest struct {
	ServerIP []string                       `json:"server_ip"`
	Body     CacheIndexerHitRateRequestBody `json:"body"`
}

type CacheIndexerHitRateRequestBody struct {
	TokenIDs  []int32 `json:"token_ids"`
	CacheSalt string  `json:"cache_salt"`
	BlockSize int64   `json:"block_size"`
}

type CacheIndexerHitRateResponse struct {
	ServerScoreList []CacheIndexerServerScore `json:"server_score_list"`
	Message         string                    `json:"message"`
	Status          int                       `json:"status"`
}

type CacheIndexerServerScore struct {
	ServerIP   string  `json:"server_ip"`
	L1HitRatio float64 `json:"l1_hit_ratio"`
	L3HitRatio float64 `json:"l3_hit_ratio"`
}

func BuildCacheIndexerHitRateRequest(serverIPs []string, tokenIDs []int32, blockSize int64, cacheSalt string) CacheIndexerHitRateRequest {
	return CacheIndexerHitRateRequest{
		ServerIP: serverIPs,
		Body: CacheIndexerHitRateRequestBody{
			TokenIDs:  tokenIDs,
			CacheSalt: cacheSalt,
			BlockSize: blockSize,
		},
	}
}

func ParseCacheIndexerHitRateResponse(body []byte) (CacheIndexerHitRateResponse, error) {
	var resp CacheIndexerHitRateResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return CacheIndexerHitRateResponse{}, fmt.Errorf("parse cache-indexer hit-rate response failed: %w; body=%s", err, strings.TrimSpace(string(body)))
	}
	return resp, nil
}

func validateCacheIndexerRuntimeConfig(valuesFile string, runtimeCfg CacheIndexerRuntimeConfig) (CacheIndexerRuntimeConfig, error) {
	if runtimeCfg.ServiceName == "" {
		return CacheIndexerRuntimeConfig{}, fmt.Errorf("cache-indexer service.name is empty in %s", valuesFile)
	}
	if runtimeCfg.ServicePort <= 0 {
		return CacheIndexerRuntimeConfig{}, fmt.Errorf("cache-indexer service.port is invalid in %s", valuesFile)
	}
	return runtimeCfg, nil
}

func loadValuesFile(valuesFile string, out any) error {
	data, err := os.ReadFile(valuesFile)
	if err != nil {
		return fmt.Errorf("read values file %s failed: %w", valuesFile, err)
	}
	if err := yaml.Unmarshal(data, out); err != nil {
		return fmt.Errorf("unmarshal values file %s failed: %w", valuesFile, err)
	}
	return nil
}

func LoadStandaloneCacheIndexerRuntimeConfig(valuesFile string) (CacheIndexerRuntimeConfig, error) {
	var values struct {
		Service struct {
			Name string `yaml:"name"`
			Port int    `yaml:"port"`
		} `yaml:"service"`
	}

	if err := loadValuesFile(valuesFile, &values); err != nil {
		return CacheIndexerRuntimeConfig{}, err
	}

	return validateCacheIndexerRuntimeConfig(valuesFile, CacheIndexerRuntimeConfig{
		ServiceName: values.Service.Name,
		ServicePort: values.Service.Port,
	})
}

func LoadEmbeddedCacheIndexerRuntimeConfig(valuesFile string) (CacheIndexerRuntimeConfig, error) {
	var values struct {
		CacheIndexer struct {
			Enabled bool `yaml:"enabled"`
			Service struct {
				Name string `yaml:"name"`
				Port int    `yaml:"port"`
			} `yaml:"service"`
		} `yaml:"cache-indexer"`
	}

	if err := loadValuesFile(valuesFile, &values); err != nil {
		return CacheIndexerRuntimeConfig{}, err
	}
	if !values.CacheIndexer.Enabled {
		return CacheIndexerRuntimeConfig{}, fmt.Errorf("cache-indexer.enabled=false in %s", valuesFile)
	}

	return validateCacheIndexerRuntimeConfig(valuesFile, CacheIndexerRuntimeConfig{
		ServiceName: values.CacheIndexer.Service.Name,
		ServicePort: values.CacheIndexer.Service.Port,
	})
}

func LoadAggregatedBackendRuntimeConfig(valuesFile string) (AggregatedBackendRuntimeConfig, error) {
	var values struct {
		Global struct {
			ModelName string `yaml:"modelName"`
		} `yaml:"global"`
		InferenceBackend struct {
			Services []struct {
				Name    string `yaml:"name"`
				Enabled bool   `yaml:"enabled"`
				Service struct {
					Port int `yaml:"port"`
				} `yaml:"service"`
				Aggregated struct {
					BlockSize int      `yaml:"blockSize"`
					ExtraArgs []string `yaml:"extraArgs"`
				} `yaml:"aggregated"`
			} `yaml:"services"`
		} `yaml:"inference-backend"`
	}

	if err := loadValuesFile(valuesFile, &values); err != nil {
		return AggregatedBackendRuntimeConfig{}, err
	}

	for _, service := range values.InferenceBackend.Services {
		if !service.Enabled {
			continue
		}
		runtimeCfg := AggregatedBackendRuntimeConfig{
			ServiceName: service.Name,
			ServicePort: service.Service.Port,
			ModelName:   values.Global.ModelName,
			BlockSize:   blockSizeFromEngineNode(service.Aggregated.BlockSize, service.Aggregated.ExtraArgs),
		}
		if runtimeCfg.ServiceName == "" {
			return AggregatedBackendRuntimeConfig{}, fmt.Errorf("aggregated backend service.name is empty in %s", valuesFile)
		}
		if runtimeCfg.ServicePort <= 0 {
			return AggregatedBackendRuntimeConfig{}, fmt.Errorf("aggregated backend service.port is invalid in %s", valuesFile)
		}
		if runtimeCfg.ModelName == "" {
			return AggregatedBackendRuntimeConfig{}, fmt.Errorf("global.modelName is empty in %s", valuesFile)
		}
		if runtimeCfg.BlockSize <= 0 {
			return AggregatedBackendRuntimeConfig{}, fmt.Errorf("aggregated blockSize is invalid in %s", valuesFile)
		}
		return runtimeCfg, nil
	}

	return AggregatedBackendRuntimeConfig{}, fmt.Errorf("no enabled aggregated backend service found in %s", valuesFile)
}

func LoadPDBackendRuntimeConfigs(valuesFile string) ([]PDBackendRuntimeConfig, error) {
	var values struct {
		Global struct {
			ModelName string `yaml:"modelName"`
		} `yaml:"global"`
		InferenceBackend struct {
			Services []struct {
				Name      string `yaml:"name"`
				Enabled   bool   `yaml:"enabled"`
				Mode      string `yaml:"mode"`
				PDGroupID string `yaml:"pdGroupID"`
				Service   struct {
					Port int `yaml:"port"`
				} `yaml:"service"`
				PD struct {
					Prefill struct {
						BlockSize int      `yaml:"blockSize"`
						ExtraArgs []string `yaml:"extraArgs"`
					} `yaml:"prefill"`
					Decode struct {
						BlockSize int      `yaml:"blockSize"`
						ExtraArgs []string `yaml:"extraArgs"`
					} `yaml:"decode"`
				} `yaml:"pd"`
			} `yaml:"services"`
		} `yaml:"inference-backend"`
	}

	if err := loadValuesFile(valuesFile, &values); err != nil {
		return nil, err
	}

	runtimeCfgs := make([]PDBackendRuntimeConfig, 0)
	for _, service := range values.InferenceBackend.Services {
		if !service.Enabled || service.Mode != "pd" {
			continue
		}
		runtimeCfg := PDBackendRuntimeConfig{
			ServiceName:      service.Name,
			ServicePort:      service.Service.Port,
			ModelName:        values.Global.ModelName,
			PrefillBlockSize: blockSizeFromEngineNode(service.PD.Prefill.BlockSize, service.PD.Prefill.ExtraArgs),
			DecodeBlockSize:  blockSizeFromEngineNode(service.PD.Decode.BlockSize, service.PD.Decode.ExtraArgs),
			PDGroupID:        service.PDGroupID,
		}
		if runtimeCfg.ServiceName == "" {
			return nil, fmt.Errorf("pd backend service.name is empty in %s", valuesFile)
		}
		if runtimeCfg.ServicePort <= 0 {
			return nil, fmt.Errorf("pd backend service.port is invalid in %s", valuesFile)
		}
		if runtimeCfg.ModelName == "" {
			return nil, fmt.Errorf("global.modelName is empty in %s", valuesFile)
		}
		if runtimeCfg.PrefillBlockSize <= 0 {
			return nil, fmt.Errorf("pd prefill blockSize is invalid in %s", valuesFile)
		}
		if runtimeCfg.DecodeBlockSize <= 0 {
			runtimeCfg.DecodeBlockSize = runtimeCfg.PrefillBlockSize
		}
		if runtimeCfg.PDGroupID == "" {
			runtimeCfg.PDGroupID = runtimeCfg.ServiceName
		}
		runtimeCfgs = append(runtimeCfgs, runtimeCfg)
	}

	if len(runtimeCfgs) == 0 {
		return nil, fmt.Errorf("no enabled pd backend service found in %s", valuesFile)
	}

	sort.Slice(runtimeCfgs, func(i, j int) bool {
		return runtimeCfgs[i].ServiceName < runtimeCfgs[j].ServiceName
	})
	return runtimeCfgs, nil
}

func LoadPDBackendRuntimeConfig(valuesFile string) (PDBackendRuntimeConfig, error) {
	runtimeCfgs, err := LoadPDBackendRuntimeConfigs(valuesFile)
	if err != nil {
		return PDBackendRuntimeConfig{}, err
	}
	if len(runtimeCfgs) != 1 {
		serviceNames := make([]string, 0, len(runtimeCfgs))
		for _, runtimeCfg := range runtimeCfgs {
			serviceNames = append(serviceNames, runtimeCfg.ServiceName)
		}
		return PDBackendRuntimeConfig{}, fmt.Errorf("expected exactly 1 pd backend service, got %v", serviceNames)
	}
	return runtimeCfgs[0], nil
}

// blockSizeFromEngineNode reads --block-size from extraArgs (PR #222),
// falling back to the removed structured blockSize field if still present.
func blockSizeFromEngineNode(legacyBlockSize int, extraArgs []string) int {
	if n := extraArgInt(extraArgs, "--block-size"); n > 0 {
		return n
	}
	return legacyBlockSize
}

func extraArgInt(extraArgs []string, flag string) int {
	for _, raw := range extraArgs {
		arg := strings.TrimSpace(raw)
		if arg == "" {
			continue
		}
		var rest string
		switch {
		case strings.HasPrefix(arg, flag+" "):
			rest = strings.TrimSpace(arg[len(flag)+1:])
		case strings.HasPrefix(arg, flag+"="):
			rest = strings.TrimSpace(arg[len(flag)+1:])
		case strings.HasPrefix(arg, flag+"\t"):
			rest = strings.TrimSpace(arg[len(flag):])
		default:
			continue
		}
		n, err := strconv.Atoi(rest)
		if err != nil {
			continue
		}
		return n
	}
	return 0
}

func RemoveFirstInferenceBackendServiceKVTransferConfig(valuesFile string) (string, error) {
	data, err := os.ReadFile(valuesFile)
	if err != nil {
		return "", fmt.Errorf("read values file %s failed: %w", valuesFile, err)
	}

	var values map[string]any
	if err := yaml.Unmarshal(data, &values); err != nil {
		return "", fmt.Errorf("unmarshal values file %s failed: %w", valuesFile, err)
	}

	inferenceBackend, ok := values["inference-backend"].(map[string]any)
	if !ok {
		return "", fmt.Errorf("inference-backend is missing or invalid in %s", valuesFile)
	}
	services, ok := inferenceBackend["services"].([]any)
	if !ok || len(services) == 0 {
		return "", fmt.Errorf("inference-backend.services is missing or empty in %s", valuesFile)
	}
	firstService, ok := services[0].(map[string]any)
	if !ok {
		return "", fmt.Errorf("inference-backend.services[0] is invalid in %s", valuesFile)
	}
	if _, exists := firstService["kvTransferConfig"]; !exists {
		return "", fmt.Errorf("inference-backend.services[0].kvTransferConfig not found in %s", valuesFile)
	}
	delete(firstService, "kvTransferConfig")

	yamlData, err := yaml.Marshal(values)
	if err != nil {
		return "", fmt.Errorf("marshal values without kvTransferConfig failed: %w", err)
	}

	dir, err := getPackageDir()
	if err != nil {
		return "", fmt.Errorf("get package dir failed: %w", err)
	}
	tempDir := filepath.Join(dir, "tmp")
	if err := os.MkdirAll(tempDir, 0755); err != nil {
		return "", fmt.Errorf("create temp dir failed: %w", err)
	}
	tempFile := filepath.Join(tempDir, fmt.Sprintf("values-no-mooncake-%d.yaml", time.Now().UnixNano()))
	if err := os.WriteFile(tempFile, yamlData, 0644); err != nil {
		return "", fmt.Errorf("write temp values file failed: %w", err)
	}

	return tempFile, nil
}

func BuildValuesVariantFromFile(
	baseValuesFile, variantPrefix string,
	mutate func(map[string]any) error,
) (string, error) {
	data, err := os.ReadFile(baseValuesFile)
	if err != nil {
		return "", fmt.Errorf("read values file %s failed: %w", baseValuesFile, err)
	}

	var values map[string]any
	if err := yaml.Unmarshal(data, &values); err != nil {
		return "", fmt.Errorf("unmarshal values file %s failed: %w", baseValuesFile, err)
	}

	if mutate != nil {
		if err := mutate(values); err != nil {
			return "", err
		}
	}

	yamlData, err := yaml.Marshal(values)
	if err != nil {
		return "", fmt.Errorf("marshal variant values failed: %w", err)
	}

	tempFile := filepath.Join(filepath.Dir(baseValuesFile), fmt.Sprintf("%s-%d.yaml", variantPrefix, time.Now().UnixNano()))
	if err := os.WriteFile(tempFile, yamlData, 0644); err != nil {
		return "", fmt.Errorf("write variant values failed: %w", err)
	}
	return tempFile, nil
}

func NestedMap(root map[string]any, keys ...string) (map[string]any, error) {
	current := root
	for _, key := range keys {
		next, ok := current[key].(map[string]any)
		if !ok {
			return nil, fmt.Errorf("values path %q is missing or invalid", strings.Join(keys, "."))
		}
		current = next
	}
	return current, nil
}

func EnsureNestedMap(root map[string]any, keys ...string) map[string]any {
	current := root
	for _, key := range keys {
		next, ok := current[key].(map[string]any)
		if !ok {
			next = map[string]any{}
			current[key] = next
		}
		current = next
	}
	return current
}

func AssertAggregatedBackendNoManualTESource(ctx context.Context, k8sClient K8SClientInterface, namespace, podName, serviceName, valuesFile string) error {
	if err := assertAggregatedBackendPodHasNoManualTEAnnotation(ctx, k8sClient, namespace, podName); err != nil {
		return err
	}
	if err := assertAggregatedBackendValuesUseMasterResolvedTE(serviceName, valuesFile); err != nil {
		return err
	}
	return nil
}

func assertAggregatedBackendPodHasNoManualTEAnnotation(ctx context.Context, k8sClient K8SClientInterface, namespace, podName string) error {
	pods, err := k8sClient.ListPods(ctx, namespace, metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("list pods failed: %w", err)
	}

	for _, pod := range pods.Items {
		if pod.Name != podName {
			continue
		}
		annotationValue := strings.TrimSpace(pod.Annotations["openfuyao.com/mooncake-local-hostname"])
		if annotationValue != "" {
			return fmt.Errorf("aggregated backend pod %s still has manual TE annotation openfuyao.com/mooncake-local-hostname=%q", podName, annotationValue)
		}
		return nil
	}

	return fmt.Errorf("aggregated backend pod %s not found", podName)
}

func assertAggregatedBackendValuesUseMasterResolvedTE(serviceName, valuesFile string) error {
	data, err := os.ReadFile(valuesFile)
	if err != nil {
		return fmt.Errorf("read values file %s failed: %w", valuesFile, err)
	}
	if strings.Contains(string(data), "openfuyao.com/mooncake-local-hostname") {
		return fmt.Errorf("values file %s still contains legacy manual TE annotation source", valuesFile)
	}

	var values struct {
		InferenceBackend struct {
			Services []struct {
				Name             string `yaml:"name"`
				Enabled          bool   `yaml:"enabled"`
				KVTransferConfig struct {
					Mooncake struct {
						Config string `yaml:"config"`
					} `yaml:"mooncake"`
				} `yaml:"kvTransferConfig"`
			} `yaml:"services"`
		} `yaml:"inference-backend"`
	}
	if err := yaml.Unmarshal(data, &values); err != nil {
		return fmt.Errorf("unmarshal values file %s failed: %w", valuesFile, err)
	}

	for _, service := range values.InferenceBackend.Services {
		if service.Name != serviceName || !service.Enabled {
			continue
		}
		if strings.TrimSpace(service.KVTransferConfig.Mooncake.Config) == "" {
			return fmt.Errorf("aggregated backend service %s has empty mooncake config in %s", serviceName, valuesFile)
		}

		var mooncakeConfig struct {
			LocalHostname string `yaml:"local_hostname"`
		}
		if err := yaml.Unmarshal([]byte(service.KVTransferConfig.Mooncake.Config), &mooncakeConfig); err != nil {
			return fmt.Errorf("unmarshal mooncake config for service %s failed: %w", serviceName, err)
		}
		if mooncakeConfig.LocalHostname != "$POD_IP" {
			return fmt.Errorf("aggregated backend service %s local_hostname should be $POD_IP for Master-resolved TE, got %q", serviceName, mooncakeConfig.LocalHostname)
		}
		return nil
	}

	return fmt.Errorf("enabled aggregated backend service %s not found in %s", serviceName, valuesFile)
}

func BuildSequentialTokenIDs(count int) []int32 {
	tokenIDs := make([]int32, count)
	for idx := range tokenIDs {
		tokenIDs[idx] = int32(idx + 1)
	}
	return tokenIDs
}

func AssertCacheIndexerPodReady(ctx context.Context, k8sClient K8SClientInterface, namespace string) error {
	pods, err := k8sClient.ListPods(ctx, namespace, metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("list pods failed: %w", err)
	}

	var cachePods []string
	for _, pod := range pods.Items {
		if strings.Contains(pod.Name, "cache-indexer") {
			cachePods = append(cachePods, pod.Name)
			if pod.Status.Phase != "Running" {
				return fmt.Errorf("cache-indexer pod %s phase=%s", pod.Name, pod.Status.Phase)
			}

			ready := 0
			total := len(pod.Status.ContainerStatuses)
			for _, cs := range pod.Status.ContainerStatuses {
				if cs.Ready {
					ready++
				}
			}
			if ready != 1 || total != 1 {
				return fmt.Errorf("cache-indexer pod %s ready=%d/%d", pod.Name, ready, total)
			}
		}
	}

	if len(cachePods) != 1 {
		return fmt.Errorf("expected exactly 1 cache-indexer pod, got %d: %v", len(cachePods), cachePods)
	}

	return nil
}

func AssertCacheIndexerHealthEndpoints(ctx context.Context, namespace string, runtimeCfg CacheIndexerRuntimeConfig) error {
	for _, path := range []string{"/healthz", "/readyz"} {
		statusCode, body, err := DoServiceRequestViaPortForward(
			ctx,
			namespace,
			runtimeCfg.ServiceName,
			runtimeCfg.ServicePort,
			http.MethodGet,
			path,
			"",
			nil,
			10*time.Second,
		)
		if err != nil {
			return err
		}
		if statusCode != http.StatusOK {
			return fmt.Errorf("%s returned status %d body=%s", path, statusCode, string(body))
		}
		if err := assertHealthBodyOK(body); err != nil {
			return fmt.Errorf("%s body invalid: %w", path, err)
		}
	}
	return nil
}

func AssertCacheIndexerStartupLogs(ctx context.Context, k8sClient *frameworkk8s.K8SClient, namespace, logContainer string) error {
	logText, err := GetCacheIndexerLogText(ctx, k8sClient, namespace, logContainer)
	if err != nil {
		return err
	}
	for _, keyword := range []string{"cache-indexer starting", "discovery started", "server started"} {
		if !strings.Contains(logText, keyword) {
			return fmt.Errorf("cache-indexer log missing keyword %q", keyword)
		}
	}
	lower := strings.ToLower(logText)
	if strings.Contains(lower, `"level":"error"`) || strings.Contains(lower, `"level":"fatal"`) {
		return fmt.Errorf("cache-indexer log contains error-level entries")
	}
	return nil
}

func AssertCacheIndexerLogContains(ctx context.Context, k8sClient *frameworkk8s.K8SClient, namespace, logContainer, keyword string) error {
	logText, err := GetCacheIndexerLogText(ctx, k8sClient, namespace, logContainer)
	if err != nil {
		return err
	}
	if !strings.Contains(logText, keyword) {
		return fmt.Errorf("cache-indexer log missing keyword %q", keyword)
	}
	return nil
}

func AssertCacheIndexerLogNotContains(ctx context.Context, k8sClient *frameworkk8s.K8SClient, namespace, logContainer, keyword string) error {
	logText, err := GetCacheIndexerLogText(ctx, k8sClient, namespace, logContainer)
	if err != nil {
		return err
	}
	if strings.Contains(logText, keyword) {
		return fmt.Errorf("cache-indexer log unexpectedly contains keyword %q", keyword)
	}
	return nil
}

func AssertCacheIndexerLogContainsLineWithAll(ctx context.Context, k8sClient *frameworkk8s.K8SClient, namespace, logContainer string, keywords ...string) error {
	logText, err := GetCacheIndexerLogText(ctx, k8sClient, namespace, logContainer)
	if err != nil {
		return err
	}
	for _, line := range strings.Split(logText, "\n") {
		matched := true
		for _, keyword := range keywords {
			if !strings.Contains(line, keyword) {
				matched = false
				break
			}
		}
		if matched {
			return nil
		}
	}
	return fmt.Errorf("cache-indexer log missing line containing keywords %v", keywords)
}

func ParseLatestLoggedHitRateResponse(logText string) (CacheIndexerHitRateResponse, error) {
	lines := strings.Split(strings.TrimSpace(logText), "\n")
	for idx := len(lines) - 1; idx >= 0; idx-- {
		line := strings.TrimSpace(lines[idx])
		if line == "" || !strings.Contains(line, "hit-rate request completed") || !strings.Contains(line, "responseBody") {
			continue
		}

		var payload map[string]any
		if err := json.Unmarshal([]byte(line), &payload); err != nil {
			jsonStart := strings.Index(line, "{")
			if jsonStart < 0 {
				continue
			}
			if err := json.Unmarshal([]byte(line[jsonStart:]), &payload); err != nil {
				continue
			}
		}

		responseBody, ok := payload["responseBody"].(string)
		if !ok || strings.TrimSpace(responseBody) == "" {
			continue
		}
		return ParseCacheIndexerHitRateResponse([]byte(responseBody))
	}
	return CacheIndexerHitRateResponse{}, fmt.Errorf("no responseBody found in hit-rate completion logs")
}

func CompareHitRateResponses(expected, actual CacheIndexerHitRateResponse) error {
	expectedJSON, err := json.Marshal(expected)
	if err != nil {
		return err
	}
	actualJSON, err := json.Marshal(actual)
	if err != nil {
		return err
	}
	if string(expectedJSON) != string(actualJSON) {
		return fmt.Errorf("logged response mismatch: expected=%s actual=%s", string(expectedJSON), string(actualJSON))
	}
	return nil
}

func NewLogSegment(before, after string) string {
	if strings.HasPrefix(after, before) {
		return after[len(before):]
	}
	return after
}

func SnapshotCacheIndexerPod(ctx context.Context, k8sClient K8SClientInterface, namespace string) (CacheIndexerPodSnapshot, error) {
	pods, err := k8sClient.ListPods(ctx, namespace, metav1.ListOptions{})
	if err != nil {
		return CacheIndexerPodSnapshot{}, fmt.Errorf("list pods failed: %w", err)
	}

	var matched CacheIndexerPodSnapshot
	matchCount := 0
	for _, pod := range pods.Items {
		if !strings.Contains(pod.Name, "cache-indexer") {
			continue
		}
		matchCount++
		restartCount := int32(0)
		for _, containerStatus := range pod.Status.ContainerStatuses {
			restartCount += containerStatus.RestartCount
		}
		matched = CacheIndexerPodSnapshot{
			PodName:      pod.Name,
			RestartCount: restartCount,
		}
	}
	if matchCount != 1 {
		return CacheIndexerPodSnapshot{}, fmt.Errorf("expected exactly 1 cache-indexer pod, got %d", matchCount)
	}
	return matched, nil
}

func AssertCacheIndexerPodUnchanged(ctx context.Context, k8sClient K8SClientInterface, namespace string, before CacheIndexerPodSnapshot) error {
	after, err := SnapshotCacheIndexerPod(ctx, k8sClient, namespace)
	if err != nil {
		return err
	}
	if after.PodName != before.PodName {
		return fmt.Errorf("cache-indexer pod changed: before=%s after=%s", before.PodName, after.PodName)
	}
	if after.RestartCount != before.RestartCount {
		return fmt.Errorf("cache-indexer restart count changed: before=%d after=%d", before.RestartCount, after.RestartCount)
	}
	return nil
}

func AssertSingleZeroHitResponse(statusCode int, body []byte, expectedIP string) error {
	if statusCode != http.StatusOK {
		return fmt.Errorf("unexpected http status=%d body=%s", statusCode, string(body))
	}
	resp, err := ParseCacheIndexerHitRateResponse(body)
	if err != nil {
		return err
	}
	if resp.Status != 0 {
		return fmt.Errorf("unexpected response status=%d body=%s", resp.Status, string(body))
	}
	if len(resp.ServerScoreList) != 1 {
		return fmt.Errorf("expected 1 server_score item, got %d body=%s", len(resp.ServerScoreList), string(body))
	}
	score := resp.ServerScoreList[0]
	if score.ServerIP != expectedIP {
		return fmt.Errorf("unexpected server_ip=%s want=%s body=%s", score.ServerIP, expectedIP, string(body))
	}
	if score.L1HitRatio != 0.0 || score.L3HitRatio != 0.0 {
		return fmt.Errorf("expected 0.0/0.0, got l1=%v l3=%v body=%s", score.L1HitRatio, score.L3HitRatio, string(body))
	}
	return nil
}

func AssertRepeatedServerIPsZeroHitResponse(statusCode int, body []byte, expectedIPs []string) error {
	if statusCode != http.StatusOK {
		return fmt.Errorf("unexpected http status=%d body=%s", statusCode, string(body))
	}
	resp, err := ParseCacheIndexerHitRateResponse(body)
	if err != nil {
		return err
	}
	if resp.Status != 0 {
		return fmt.Errorf("unexpected response status=%d body=%s", resp.Status, string(body))
	}
	if len(resp.ServerScoreList) != len(expectedIPs) {
		return fmt.Errorf("expected %d server_score items, got %d body=%s", len(expectedIPs), len(resp.ServerScoreList), string(body))
	}
	for idx, expectedIP := range expectedIPs {
		score := resp.ServerScoreList[idx]
		if score.ServerIP != expectedIP {
			return fmt.Errorf("unexpected server_ip at %d: got=%s want=%s body=%s", idx, score.ServerIP, expectedIP, string(body))
		}
		if score.L1HitRatio != 0.0 || score.L3HitRatio != 0.0 {
			return fmt.Errorf("expected repeated entry %d to be 0.0/0.0, got l1=%v l3=%v body=%s", idx, score.L1HitRatio, score.L3HitRatio, string(body))
		}
	}
	return nil
}

func GetCacheIndexerLogText(ctx context.Context, k8sClient *frameworkk8s.K8SClient, namespace, logContainer string) (string, error) {
	pods, err := k8sClient.ListPods(ctx, namespace, metav1.ListOptions{})
	if err != nil {
		return "", fmt.Errorf("list pods failed: %w", err)
	}
	for _, pod := range pods.Items {
		if !strings.Contains(pod.Name, "cache-indexer") {
			continue
		}
		logResult, err := frameworkk8s.GetLogsByPod(ctx, k8sClient.Clientset, namespace, pod.Name, logContainer)
		if err != nil {
			return "", err
		}
		return logResult.Log, nil
	}
	return "", fmt.Errorf("cache-indexer pod not found for log collection")
}

func assertHealthBodyOK(body []byte) error {
	trimmed := strings.TrimSpace(string(body))
	if trimmed == "ok" {
		return nil
	}

	var payload cacheIndexerHealthResponse
	if err := json.Unmarshal(body, &payload); err != nil {
		return fmt.Errorf("parse health response failed: %w; body=%s", err, trimmed)
	}
	if payload.Status != "ok" {
		return fmt.Errorf("expected status=ok, got %q", payload.Status)
	}
	return nil
}

func CleanupCacheIndexerInfernexConflicts(ctx context.Context, k8sClient K8SClientInterface, targetRelease, targetNamespace string) error {
	_ = HelmUninstallIfExists(targetRelease, targetNamespace)

	for _, namespace := range []string{"istio-system", "scaling-system"} {
		saList, err := k8sClient.ListServiceAccounts(ctx, namespace, metav1.ListOptions{})
		if err != nil {
			continue
		}
		for _, sa := range saList.Items {
			ann := sa.Annotations
			if ann == nil {
				continue
			}
			releaseNamespace := ann["meta.helm.sh/release-namespace"]
			if releaseNamespace != "" && releaseNamespace != targetNamespace {
				_ = k8sClient.DeleteServiceAccount(ctx, namespace, sa.Name)
			}
		}
	}

	return nil
}

func getReadyPodIdentitiesByPrefix(ctx context.Context, k8sClient K8SClientInterface, namespace, prefix, kind string) ([]AggregatedBackendPodIdentity, error) {
	pods, err := k8sClient.ListPods(ctx, namespace, metav1.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("list pods failed: %w", err)
	}

	matched := make([]AggregatedBackendPodIdentity, 0)
	for _, pod := range pods.Items {
		if !strings.HasPrefix(pod.Name, prefix) {
			continue
		}
		if pod.Status.Phase != "Running" {
			return nil, fmt.Errorf("%s %s phase=%s", kind, pod.Name, pod.Status.Phase)
		}

		ready := 0
		total := len(pod.Status.ContainerStatuses)
		for _, cs := range pod.Status.ContainerStatuses {
			if cs.Ready {
				ready++
			}
		}
		if total == 0 || ready != total {
			return nil, fmt.Errorf("%s %s ready=%d/%d", kind, pod.Name, ready, total)
		}
		if pod.Status.PodIP == "" {
			return nil, fmt.Errorf("%s %s podIP is empty", kind, pod.Name)
		}
		matched = append(matched, AggregatedBackendPodIdentity{
			Name: pod.Name,
			IP:   pod.Status.PodIP,
		})
	}

	if len(matched) == 0 {
		return nil, fmt.Errorf("%s not found with prefix %s", kind, prefix)
	}

	sort.Slice(matched, func(i, j int) bool {
		return matched[i].Name < matched[j].Name
	})
	return matched, nil
}

func getSingleReadyPodIdentityByPrefix(ctx context.Context, k8sClient K8SClientInterface, namespace, prefix, kind string) (string, string, error) {
	matched, err := getReadyPodIdentitiesByPrefix(ctx, k8sClient, namespace, prefix, kind)
	if err != nil {
		return "", "", err
	}
	if len(matched) != 1 {
		matchedNames := make([]string, 0, len(matched))
		for _, pod := range matched {
			matchedNames = append(matchedNames, pod.Name)
		}
		return "", "", fmt.Errorf("expected exactly 1 %s with prefix %s, got %v", kind, prefix, matchedNames)
	}
	return matched[0].Name, matched[0].IP, nil
}

func GetAggregatedBackendPodIdentities(ctx context.Context, k8sClient K8SClientInterface, namespace, serviceName string) ([]AggregatedBackendPodIdentity, error) {
	return getReadyPodIdentitiesByPrefix(ctx, k8sClient, namespace, serviceName+"-", "aggregated backend pod")
}

func GetAggregatedBackendPodIdentity(ctx context.Context, k8sClient K8SClientInterface, namespace, serviceName string) (string, string, error) {
	return getSingleReadyPodIdentityByPrefix(ctx, k8sClient, namespace, serviceName+"-", "aggregated backend pod")
}

func GetPDBackendPodIdentity(ctx context.Context, k8sClient K8SClientInterface, namespace, serviceName, role string) (string, string, error) {
	prefix := fmt.Sprintf("%s-%s-", serviceName, role)
	return getSingleReadyPodIdentityByPrefix(ctx, k8sClient, namespace, prefix, fmt.Sprintf("pd %s pod", role))
}

func GetPDSingleGroupPodIdentity(ctx context.Context, k8sClient K8SClientInterface, namespace, serviceName, role string) (string, string, error) {
	return GetPDBackendPodIdentity(ctx, k8sClient, namespace, serviceName, role)
}

func AssertAggregatedBackendPodsAbsent(ctx context.Context, k8sClient K8SClientInterface, namespace, serviceName string) error {
	pods, err := k8sClient.ListPods(ctx, namespace, metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("list pods failed: %w", err)
	}

	for _, pod := range pods.Items {
		if strings.HasPrefix(pod.Name, serviceName+"-") {
			return fmt.Errorf("aggregated backend pod %s still exists phase=%s", pod.Name, pod.Status.Phase)
		}
	}

	return nil
}

func ScaleAggregatedBackendWorkload(ctx context.Context, k8sClient *frameworkk8s.K8SClient, namespace, podName string, replicas int32) error {
	if k8sClient == nil || k8sClient.Clientset == nil {
		return fmt.Errorf("k8s client is nil")
	}

	pod, err := k8sClient.Clientset.CoreV1().Pods(namespace).Get(ctx, podName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("get aggregated backend pod %s failed: %w", podName, err)
	}

	owner := metav1.GetControllerOf(pod)
	if owner == nil {
		return fmt.Errorf("aggregated backend pod %s has no controller owner", podName)
	}

	switch owner.Kind {
	case "StatefulSet":
		return scaleAggregatedBackendStatefulSet(ctx, k8sClient, namespace, owner.Name, replicas)
	case "Deployment":
		return scaleAggregatedBackendDeployment(ctx, k8sClient, namespace, owner.Name, replicas)
	case "ReplicaSet":
		rs, err := k8sClient.Clientset.AppsV1().ReplicaSets(namespace).Get(ctx, owner.Name, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("get aggregated backend replicaset %s failed: %w", owner.Name, err)
		}
		rsOwner := metav1.GetControllerOf(rs)
		if rsOwner != nil && rsOwner.Kind == "Deployment" {
			return scaleAggregatedBackendDeployment(ctx, k8sClient, namespace, rsOwner.Name, replicas)
		}
		rs = rs.DeepCopy()
		rs.Spec.Replicas = &replicas
		if _, err := k8sClient.Clientset.AppsV1().ReplicaSets(namespace).Update(ctx, rs, metav1.UpdateOptions{}); err != nil {
			return fmt.Errorf("scale aggregated backend replicaset %s to %d failed: %w", owner.Name, replicas, err)
		}
		return nil
	default:
		return fmt.Errorf("unsupported aggregated backend controller kind %s for pod %s", owner.Kind, podName)
	}
}

func scaleAggregatedBackendDeployment(ctx context.Context, k8sClient *frameworkk8s.K8SClient, namespace, deploymentName string, replicas int32) error {
	deploy, err := k8sClient.Clientset.AppsV1().Deployments(namespace).Get(ctx, deploymentName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("get aggregated backend deployment %s failed: %w", deploymentName, err)
	}

	deploy = deploy.DeepCopy()
	deploy.Spec.Replicas = &replicas
	if _, err := k8sClient.Clientset.AppsV1().Deployments(namespace).Update(ctx, deploy, metav1.UpdateOptions{}); err != nil {
		return fmt.Errorf("scale aggregated backend deployment %s to %d failed: %w", deploymentName, replicas, err)
	}
	return nil
}

func scaleAggregatedBackendStatefulSet(ctx context.Context, k8sClient *frameworkk8s.K8SClient, namespace, statefulSetName string, replicas int32) error {
	sts, err := k8sClient.Clientset.AppsV1().StatefulSets(namespace).Get(ctx, statefulSetName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("get aggregated backend statefulset %s failed: %w", statefulSetName, err)
	}

	if owner := metav1.GetControllerOf(sts); owner != nil && owner.Kind == "LeaderWorkerSet" {
		return scaleAggregatedBackendLeaderWorkerSet(ctx, k8sClient, namespace, owner.Name, owner.APIVersion, replicas)
	}

	sts = sts.DeepCopy()
	sts.Spec.Replicas = &replicas
	if _, err := k8sClient.Clientset.AppsV1().StatefulSets(namespace).Update(ctx, sts, metav1.UpdateOptions{}); err != nil {
		return fmt.Errorf("scale aggregated backend statefulset %s to %d failed: %w", statefulSetName, replicas, err)
	}
	return nil
}

func scaleAggregatedBackendLeaderWorkerSet(ctx context.Context, k8sClient *frameworkk8s.K8SClient, namespace, name, apiVersion string, replicas int32) error {
	if k8sClient == nil || k8sClient.DynamicClient == nil {
		return fmt.Errorf("k8s dynamic client is nil")
	}

	gvr := schema.GroupVersionResource{
		Group:    "leaderworkerset.x-k8s.io",
		Version:  "v1",
		Resource: "leaderworkersets",
	}
	if apiVersion == "scheduling.k8s.io/v1alpha1" {
		gvr = schema.GroupVersionResource{
			Group:    "scheduling.k8s.io",
			Version:  "v1alpha1",
			Resource: "leaderworkersets",
		}
	}

	obj, err := k8sClient.DynamicClient.Resource(gvr).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("get aggregated backend leaderworkerset %s failed: %w", name, err)
	}
	if err := unstructured.SetNestedField(obj.Object, int64(replicas), "spec", "replicas"); err != nil {
		return fmt.Errorf("set leaderworkerset %s spec.replicas=%d failed: %w", name, replicas, err)
	}
	if _, err := k8sClient.DynamicClient.Resource(gvr).Namespace(namespace).Update(ctx, obj, metav1.UpdateOptions{}); err != nil {
		return fmt.Errorf("scale aggregated backend leaderworkerset %s to %d failed: %w", name, replicas, err)
	}
	return nil
}

func WaitForMooncakeMasterPodReady(ctx context.Context, k8sClient K8SClientInterface, namespace string) error {
	pods, err := k8sClient.ListPods(ctx, namespace, metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("list pods failed: %w", err)
	}

	for _, pod := range pods.Items {
		if !strings.HasPrefix(pod.Name, "mooncake-master-deployment-") {
			continue
		}
		if pod.Status.Phase != "Running" {
			return fmt.Errorf("mooncake master pod %s phase=%s", pod.Name, pod.Status.Phase)
		}

		ready := 0
		total := len(pod.Status.ContainerStatuses)
		for _, cs := range pod.Status.ContainerStatuses {
			if cs.Ready {
				ready++
			}
		}
		if total == 0 || ready != total {
			return fmt.Errorf("mooncake master pod %s ready=%d/%d", pod.Name, ready, total)
		}
		return nil
	}

	return fmt.Errorf("mooncake master pod not found")
}

func ScaleMooncakeMasterDeployment(ctx context.Context, k8sClient *frameworkk8s.K8SClient, namespace string, replicas int32) error {
	if k8sClient == nil || k8sClient.Clientset == nil {
		return fmt.Errorf("k8s client is nil")
	}

	deploy, err := k8sClient.Clientset.AppsV1().Deployments(namespace).Get(ctx, "mooncake-master-deployment", metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("get mooncake master deployment failed: %w", err)
	}

	deploy = deploy.DeepCopy()
	deploy.Spec.Replicas = &replicas
	if _, err := k8sClient.Clientset.AppsV1().Deployments(namespace).Update(ctx, deploy, metav1.UpdateOptions{}); err != nil {
		return fmt.Errorf("scale mooncake master deployment to %d failed: %w", replicas, err)
	}

	return nil
}

func AssertMooncakeMasterPodAbsent(ctx context.Context, k8sClient K8SClientInterface, namespace string) error {
	pods, err := k8sClient.ListPods(ctx, namespace, metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("list pods failed: %w", err)
	}

	for _, pod := range pods.Items {
		if strings.HasPrefix(pod.Name, "mooncake-master-deployment-") {
			return fmt.Errorf("mooncake master pod still exists: %s phase=%s", pod.Name, pod.Status.Phase)
		}
	}

	return nil
}

func CreateTemporaryVLLMPod(
	ctx context.Context,
	k8sClient *frameworkk8s.K8SClient,
	namespace, name string,
	labels map[string]string,
	ready bool,
) (*corev1.Pod, error) {
	if k8sClient == nil || k8sClient.Clientset == nil {
		return nil, fmt.Errorf("k8s client is nil")
	}

	readinessProbe := &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			Exec: &corev1.ExecAction{
				Command: []string{"sh", "-c", "test -f /tmp/ready"},
			},
		},
		InitialDelaySeconds: 1,
		PeriodSeconds:       1,
	}

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels:    labels,
		},
		Spec: corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyNever,
			Containers: []corev1.Container{
				{
					Name:  "fake-vllm",
					Image: "busybox",
					Command: []string{
						"sh",
						"-c",
						"if [ \"$READY_POD\" = \"true\" ]; then touch /tmp/ready; fi; sleep 3600",
					},
					Env: []corev1.EnvVar{
						{Name: "READY_POD", Value: fmt.Sprintf("%t", ready)},
					},
					Ports: []corev1.ContainerPort{
						{Name: "zmq-pub", ContainerPort: 5557},
					},
					ReadinessProbe: readinessProbe,
				},
			},
		},
	}

	created, err := k8sClient.Clientset.CoreV1().Pods(namespace).Create(ctx, pod, metav1.CreateOptions{})
	if err != nil {
		return nil, fmt.Errorf("create temporary vllm pod %s failed: %w", name, err)
	}
	return created, nil
}

func DeletePodIfExists(ctx context.Context, k8sClient *frameworkk8s.K8SClient, namespace, name string) error {
	if k8sClient == nil || k8sClient.Clientset == nil {
		return fmt.Errorf("k8s client is nil")
	}
	gracePeriodSeconds := int64(0)
	propagationPolicy := metav1.DeletePropagationBackground
	err := k8sClient.Clientset.CoreV1().Pods(namespace).Delete(ctx, name, metav1.DeleteOptions{
		GracePeriodSeconds: &gracePeriodSeconds,
		PropagationPolicy:  &propagationPolicy,
	})
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("delete pod %s/%s failed: %w", namespace, name, err)
	}
	return nil
}

func WaitForPodGone(ctx context.Context, k8sClient *frameworkk8s.K8SClient, namespace, name string, timeout time.Duration) error {
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		_, err := k8sClient.Clientset.CoreV1().Pods(namespace).Get(ctx, name, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return nil
		}
		if err != nil {
			return fmt.Errorf("get pod %s/%s failed: %w", namespace, name, err)
		}
		time.Sleep(2 * time.Second)
	}
	return fmt.Errorf("temporary pod %s/%s still exists after %s", namespace, name, timeout)
}

func UniqueTemporaryPodName(prefix string) string {
	return fmt.Sprintf("%s-%d", prefix, time.Now().UnixNano())
}

func tokenizePromptViaService(ctx context.Context, namespace, serviceName string, servicePort int, prompt string) ([]int32, error) {
	body, err := json.Marshal(BuildTokenizeRequestBody(prompt))
	if err != nil {
		return nil, fmt.Errorf("marshal tokenize request failed: %w", err)
	}

	statusCode, respBody, err := DoServiceRequestViaPortForward(
		ctx,
		namespace,
		serviceName,
		servicePort,
		http.MethodPost,
		"/tokenize",
		"application/json",
		body,
		30*time.Second,
	)
	if err != nil {
		return nil, err
	}
	if statusCode != http.StatusOK {
		return nil, fmt.Errorf("/tokenize returned status %d body=%s", statusCode, string(respBody))
	}

	resp, err := ParseTokenizeResponse(respBody)
	if err != nil {
		return nil, err
	}
	return resp.Tokens, nil
}

func tokenizePromptViaPod(ctx context.Context, namespace, podName string, podPort int, prompt string) ([]int32, error) {
	body, err := json.Marshal(BuildTokenizeRequestBody(prompt))
	if err != nil {
		return nil, fmt.Errorf("marshal tokenize request failed: %w", err)
	}

	statusCode, respBody, err := DoPodRequestViaPortForward(
		ctx,
		namespace,
		podName,
		podPort,
		http.MethodPost,
		"/tokenize",
		"application/json",
		body,
		30*time.Second,
	)
	if err != nil {
		return nil, err
	}
	if statusCode != http.StatusOK {
		return nil, fmt.Errorf("/tokenize returned status %d body=%s", statusCode, string(respBody))
	}

	resp, err := ParseTokenizeResponse(respBody)
	if err != nil {
		return nil, err
	}
	return resp.Tokens, nil
}

func TokenizePromptViaAggregatedBackend(ctx context.Context, namespace string, runtimeCfg AggregatedBackendRuntimeConfig, prompt string) ([]int32, error) {
	return tokenizePromptViaService(ctx, namespace, runtimeCfg.ServiceName, runtimeCfg.ServicePort, prompt)
}

func TokenizePromptViaPDBackend(ctx context.Context, namespace, prefillPodName string, runtimeCfg PDBackendRuntimeConfig, prompt string) ([]int32, error) {
	return tokenizePromptViaPod(ctx, namespace, prefillPodName, runtimeCfg.ServicePort, prompt)
}

func buildBackendInferenceRequestBody(modelName, prompt string) ([]byte, error) {
	reqBody := map[string]any{
		"model":       modelName,
		"prompt":      prompt,
		"max_tokens":  2,
		"temperature": 0,
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal inference request failed: %w", err)
	}
	return body, nil
}

func doInferenceViaService(ctx context.Context, namespace, serviceName string, servicePort int, modelName, prompt string) error {
	body, err := buildBackendInferenceRequestBody(modelName, prompt)
	if err != nil {
		return err
	}

	statusCode, respBody, err := DoServiceRequestViaPortForward(
		ctx,
		namespace,
		serviceName,
		servicePort,
		http.MethodPost,
		"/v1/completions",
		"application/json",
		body,
		90*time.Second,
	)
	if err != nil {
		return err
	}
	return ValidateResponse(statusCode, respBody, false)
}

func doInferenceViaPod(ctx context.Context, namespace, podName string, podPort int, modelName, prompt string) error {
	body, err := buildBackendInferenceRequestBody(modelName, prompt)
	if err != nil {
		return err
	}

	statusCode, respBody, err := DoPodRequestViaPortForward(
		ctx,
		namespace,
		podName,
		podPort,
		http.MethodPost,
		"/v1/completions",
		"application/json",
		body,
		90*time.Second,
	)
	if err != nil {
		return err
	}
	return ValidateResponse(statusCode, respBody, false)
}

func DoAggregatedBackendInference(ctx context.Context, namespace string, runtimeCfg AggregatedBackendRuntimeConfig, prompt string) error {
	return doInferenceViaService(ctx, namespace, runtimeCfg.ServiceName, runtimeCfg.ServicePort, runtimeCfg.ModelName, prompt)
}

func DoAggregatedBackendInferenceViaPod(ctx context.Context, namespace, podName string, runtimeCfg AggregatedBackendRuntimeConfig, prompt string) error {
	return doInferenceViaPod(ctx, namespace, podName, runtimeCfg.ServicePort, runtimeCfg.ModelName, prompt)
}

func DoPDBackendInference(ctx context.Context, namespace string, runtimeCfg PDBackendRuntimeConfig, prompt string) error {
	return doInferenceViaService(ctx, namespace, runtimeCfg.ServiceName, runtimeCfg.ServicePort, runtimeCfg.ModelName, prompt)
}

func QueryCacheIndexerHitRateForServerIPs(ctx context.Context, namespace string, backendPodIPs []string, tokens []int32, cacheRuntime CacheIndexerRuntimeConfig, blockSize int) ([]CacheIndexerServerScore, []byte, error) {
	reqBody := BuildCacheIndexerHitRateRequest(backendPodIPs, tokens, int64(blockSize), "")
	statusCode, body, err := DoCacheIndexerHitRateRequest(
		ctx,
		namespace,
		cacheRuntime.ServiceName,
		cacheRuntime.ServicePort,
		reqBody,
		15*time.Second,
	)
	if err != nil {
		return nil, nil, err
	}
	if statusCode != http.StatusOK {
		return nil, nil, fmt.Errorf("/kv-cache/hit-rate returned status %d body=%s", statusCode, string(body))
	}

	resp, err := ParseCacheIndexerHitRateResponse(body)
	if err != nil {
		return nil, nil, err
	}
	if resp.Status != 0 {
		return nil, nil, fmt.Errorf("unexpected hit-rate status=%d body=%s", resp.Status, string(body))
	}
	if len(resp.ServerScoreList) != len(backendPodIPs) {
		return nil, nil, fmt.Errorf("expected %d server_score items, got %d body=%s", len(backendPodIPs), len(resp.ServerScoreList), string(body))
	}
	for idx, backendPodIP := range backendPodIPs {
		if resp.ServerScoreList[idx].ServerIP != backendPodIP {
			return nil, nil, fmt.Errorf("expected server_score_list[%d].server_ip=%s, got %s body=%s", idx, backendPodIP, resp.ServerScoreList[idx].ServerIP, string(body))
		}
	}
	return resp.ServerScoreList, body, nil
}

func QuerySingleBackendHitRate(ctx context.Context, namespace, backendPodIP string, tokens []int32, cacheRuntime CacheIndexerRuntimeConfig, blockSize int) (CacheIndexerServerScore, []byte, error) {
	scores, body, err := QueryCacheIndexerHitRateForServerIPs(ctx, namespace, []string{backendPodIP}, tokens, cacheRuntime, blockSize)
	if err != nil {
		return CacheIndexerServerScore{}, nil, err
	}
	return scores[0], body, nil
}

func QuerySingleBackendFullHit(ctx context.Context, namespace, backendPodIP string, tokens []int32, cacheRuntime CacheIndexerRuntimeConfig, blockSize int) ([]byte, error) {
	score, body, err := QuerySingleBackendHitRate(ctx, namespace, backendPodIP, tokens, cacheRuntime, blockSize)
	if err != nil {
		return nil, err
	}
	if score.L1HitRatio != 1.0 || score.L3HitRatio != 1.0 {
		return nil, fmt.Errorf("expected l1/l3 hit ratio both 1.0, got l1=%v l3=%v body=%s", score.L1HitRatio, score.L3HitRatio, string(body))
	}
	return body, nil
}

func QuerySingleBackendL1OnlyHit(ctx context.Context, namespace, backendPodIP string, tokens []int32, cacheRuntime CacheIndexerRuntimeConfig, blockSize int) ([]byte, error) {
	score, body, err := QuerySingleBackendHitRate(ctx, namespace, backendPodIP, tokens, cacheRuntime, blockSize)
	if err != nil {
		return nil, err
	}
	if score.L1HitRatio != 1.0 || score.L3HitRatio != 0.0 {
		return nil, fmt.Errorf("expected l1=1.0 and l3=0.0, got l1=%v l3=%v body=%s", score.L1HitRatio, score.L3HitRatio, string(body))
	}
	return body, nil
}

func QuerySingleBackendZeroHit(ctx context.Context, namespace, backendPodIP string, tokens []int32, cacheRuntime CacheIndexerRuntimeConfig, blockSize int) ([]byte, error) {
	score, body, err := QuerySingleBackendHitRate(ctx, namespace, backendPodIP, tokens, cacheRuntime, blockSize)
	if err != nil {
		return nil, err
	}
	if score.L1HitRatio != 0.0 || score.L3HitRatio != 0.0 {
		return nil, fmt.Errorf("expected l1=0.0 and l3=0.0, got l1=%v l3=%v body=%s", score.L1HitRatio, score.L3HitRatio, string(body))
	}
	return body, nil
}

func QuerySingleBackendExpectedHit(ctx context.Context, namespace, backendPodIP string, tokens []int32, cacheRuntime CacheIndexerRuntimeConfig, blockSize int, expectedL1, expectedL3, tolerance float64) ([]byte, error) {
	score, body, err := QuerySingleBackendHitRate(ctx, namespace, backendPodIP, tokens, cacheRuntime, blockSize)
	if err != nil {
		return nil, err
	}
	if math.Abs(score.L1HitRatio-expectedL1) > tolerance || math.Abs(score.L3HitRatio-expectedL3) > tolerance {
		return nil, fmt.Errorf("expected l1=%v and l3=%v, got l1=%v l3=%v body=%s", expectedL1, expectedL3, score.L1HitRatio, score.L3HitRatio, string(body))
	}
	return body, nil
}

func QueryAggregateSingleBackendHitRate(ctx context.Context, namespace, backendPodIP string, tokens []int32, cacheRuntime CacheIndexerRuntimeConfig, blockSize int) (CacheIndexerServerScore, []byte, error) {
	return QuerySingleBackendHitRate(ctx, namespace, backendPodIP, tokens, cacheRuntime, blockSize)
}

func QueryAggregateSingleBackendFullHit(ctx context.Context, namespace, backendPodIP string, tokens []int32, cacheRuntime CacheIndexerRuntimeConfig, blockSize int) ([]byte, error) {
	return QuerySingleBackendFullHit(ctx, namespace, backendPodIP, tokens, cacheRuntime, blockSize)
}

func QueryAggregateSingleBackendL1OnlyHit(ctx context.Context, namespace, backendPodIP string, tokens []int32, cacheRuntime CacheIndexerRuntimeConfig, blockSize int) ([]byte, error) {
	return QuerySingleBackendL1OnlyHit(ctx, namespace, backendPodIP, tokens, cacheRuntime, blockSize)
}

func QueryAggregateSingleBackendZeroHit(ctx context.Context, namespace, backendPodIP string, tokens []int32, cacheRuntime CacheIndexerRuntimeConfig, blockSize int) ([]byte, error) {
	return QuerySingleBackendZeroHit(ctx, namespace, backendPodIP, tokens, cacheRuntime, blockSize)
}

func QueryAggregateSingleBackendExpectedHit(ctx context.Context, namespace, backendPodIP string, tokens []int32, cacheRuntime CacheIndexerRuntimeConfig, blockSize int, expectedL1, expectedL3, tolerance float64) ([]byte, error) {
	return QuerySingleBackendExpectedHit(ctx, namespace, backendPodIP, tokens, cacheRuntime, blockSize, expectedL1, expectedL3, tolerance)
}

func DoCacheIndexerHitRateRequest(
	ctx context.Context,
	namespace, serviceName string,
	servicePort int,
	reqBody CacheIndexerHitRateRequest,
	timeout time.Duration,
) (int, []byte, error) {
	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return 0, nil, fmt.Errorf("marshal cache-indexer hit-rate request failed: %w", err)
	}
	return DoServiceRequestViaPortForward(
		ctx,
		namespace,
		serviceName,
		servicePort,
		http.MethodPost,
		CacheIndexerHitRatePath,
		"application/json",
		bodyBytes,
		timeout,
	)
}

func DoServiceRequestViaPortForward(
	ctx context.Context,
	namespace, serviceName string,
	servicePort int,
	method, path, contentType string,
	reqBody []byte,
	timeout time.Duration,
) (int, []byte, error) {
	return doRequestViaPortForward(
		ctx,
		namespace,
		"svc/"+serviceName,
		servicePort,
		method,
		path,
		contentType,
		reqBody,
		timeout,
	)
}

func DoPodRequestViaPortForward(
	ctx context.Context,
	namespace, podName string,
	podPort int,
	method, path, contentType string,
	reqBody []byte,
	timeout time.Duration,
) (int, []byte, error) {
	return doRequestViaPortForward(
		ctx,
		namespace,
		"pod/"+podName,
		podPort,
		method,
		path,
		contentType,
		reqBody,
		timeout,
	)
}

func doRequestViaPortForward(
	ctx context.Context,
	namespace, resourceName string,
	resourcePort int,
	method, path, contentType string,
	reqBody []byte,
	timeout time.Duration,
) (int, []byte, error) {
	localPort, err := freeLocalPortForPortForward()
	if err != nil {
		return 0, nil, err
	}
	if timeout <= 0 {
		timeout = 10 * time.Second
	}

	pfCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	target := fmt.Sprintf("%d:%d", localPort, resourcePort)
	cmd := exec.CommandContext(pfCtx, "kubectl", "port-forward", "-n", namespace, resourceName, target)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		return 0, nil, fmt.Errorf("start kubectl port-forward for %s in namespace %s: %w", resourceName, namespace, err)
	}
	defer func() {
		cancel()
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
		_ = cmd.Wait()
	}()

	client := &http.Client{Timeout: 2 * time.Second}
	deadline := time.Now().Add(timeout)
	url := fmt.Sprintf("http://127.0.0.1:%d%s", localPort, path)
	var lastErr error
	for time.Now().Before(deadline) {
		req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(reqBody))
		if err != nil {
			return 0, nil, err
		}
		if contentType != "" {
			req.Header.Set("Content-Type", contentType)
		}
		resp, err := client.Do(req)
		if err == nil {
			body, readErr := io.ReadAll(resp.Body)
			_ = resp.Body.Close()
			if readErr == nil {
				return resp.StatusCode, body, nil
			}
			lastErr = readErr
		} else {
			lastErr = err
		}
		time.Sleep(200 * time.Millisecond)
	}

	return 0, nil, fmt.Errorf("request %s %s via port-forward %s timed out: %w; port-forward stderr: %s", method, path, resourceName, lastErr, stderr.String())
}

func freeLocalPortForPortForward() (int, error) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer listener.Close()
	return listener.Addr().(*net.TCPAddr).Port, nil
}
