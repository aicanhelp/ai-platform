package utils

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// FlushThresholdValue is the requestTracking.persistence.flushThreshold value
// that ensures synchronous writes to completed.jsonl (each request is persisted immediately).
const FlushThresholdValue = 1

// CompletedRecord represents a single line in completed.jsonl.
// The struct uses pointer fields with omitempty for PD-specific nested objects
// so it can parse both aggregate and PD completed.jsonl schemas.
type CompletedRecord struct {
	SchemaVersion         int                    `json:"schemaVersion"`
	RequestId             string                 `json:"requestId"`
	Model                 string                 `json:"model"`
	CompletedAt           string                 `json:"completedAt"` // date-time string, NOT time.Time — avoids RFC3339 parsing failures
	PredictionMode        string                 `json:"predictionMode"`
	PrefixCacheScore      float64                `json:"prefixCacheScore"`
	InputTokenLength      int                    `json:"inputTokenLength"`
	OutputLength          int                    `json:"outputLength"`
	TTFTMillis            float64                `json:"ttftMillis"`
	TotalLatencyMillis    float64                `json:"totalLatencyMillis"`
	Prediction            *PredictionData        `json:"prediction,omitempty"`
	SelectedRoute         *SelectedRoute         `json:"selectedRoute,omitempty"`
	PredictionInputs      *PredictionInputs      `json:"predictionInputs,omitempty"`
	MissingFeatureReasons *MissingFeatureReasons `json:"missingFeatureReasons,omitempty"`
}

// SelectedRoute describes where a request was routed.
// Leader is always present; Prefill and Decode are PD-specific (pointer+omitempty).
type SelectedRoute struct {
	Leader  RouteEndpoint  `json:"leader"`
	Prefill *RouteEndpoint `json:"prefill,omitempty"`
	Decode  *RouteEndpoint `json:"decode,omitempty"`
}

// RouteEndpoint matches routeEndpointSnapshot in the PD JSON schema.
type RouteEndpoint struct {
	Id       string            `json:"id"`
	Metrics  *EndpointMetrics  `json:"metrics,omitempty"`
	Inflight *Load             `json:"inflight,omitempty"`
	NPU      *NPUStateSnapshot `json:"npu,omitempty"`
}

// EndpointMetrics matches endpointMetricsSnapshot in the PD JSON schema.
type EndpointMetrics struct {
	ActiveModels            map[string]int `json:"activeModels,omitempty"`
	WaitingModels           map[string]int `json:"waitingModels,omitempty"`
	MaxActiveModels         int            `json:"maxActiveModels,omitempty"`
	RunningRequestsSize     int            `json:"runningRequestsSize,omitempty"`
	WaitingQueueSize        int            `json:"waitingQueueSize,omitempty"`
	KVCacheUsagePercent     float64        `json:"kvCacheUsagePercent,omitempty"`
	KVCacheMaxTokenCapacity int            `json:"kvCacheMaxTokenCapacity,omitempty"`
	CacheBlockSize          int            `json:"cacheBlockSize,omitempty"`
	CacheNumBlocks          int            `json:"cacheNumBlocks,omitempty"`
	UpdateTime              string         `json:"updateTime,omitempty"` // date-time string, NOT time.Time
}

// Load matches load in the PD JSON schema. Note capital-T JSON tag "Tokens" per schema.
type Load struct {
	Tokens   int `json:"Tokens"`
	Requests int `json:"Requests"`
}

// NPUStateSnapshot matches npuStateSnapshot in the PD JSON schema.
type NPUStateSnapshot struct {
	ObservationTime            string              `json:"observationTime,omitempty"`
	DataStatus                 string              `json:"dataStatus,omitempty"`
	DeviceCount                int                 `json:"deviceCount,omitempty"`
	AICoreUtilizationAvg       float64             `json:"aiCoreUtilizationAvg,omitempty"`
	AICoreUtilizationMax       float64             `json:"aiCoreUtilizationMax,omitempty"`
	HBMUtilizationAvg          float64             `json:"hbmUtilizationAvg,omitempty"`
	HBMUtilizationMax          float64             `json:"hbmUtilizationMax,omitempty"`
	HBMBandwidthUtilizationAvg float64             `json:"hbmBandwidthUtilizationAvg,omitempty"`
	HBMBandwidthUtilizationMax float64             `json:"hbmBandwidthUtilizationMax,omitempty"`
	Devices                    []NPUDeviceSnapshot `json:"devices,omitempty"`
}

// NPUDeviceSnapshot matches npuDeviceSnapshot in the PD JSON schema.
type NPUDeviceSnapshot struct {
	Target                  string  `json:"target,omitempty"`
	DeviceID                string  `json:"deviceID,omitempty"`
	AICoreUtilization       float64 `json:"aiCoreUtilization,omitempty"`
	HBMUtilization          float64 `json:"hbmUtilization,omitempty"`
	HBMBandwidthUtilization float64 `json:"hbmBandwidthUtilization,omitempty"`
}

// PredictionData matches prediction in the PD JSON schema.
type PredictionData struct {
	TTFTMillis         float64 `json:"ttftMillis"`
	TPOTMillis         float64 `json:"tpotMillis"`
	TotalLatencyMillis float64 `json:"totalLatencyMillis"`
}

// PredictionInputs matches predictionInputs in aggregate/PD completed.jsonl schemas.
// Aggregate mode uses leader; PD mode uses prefill/decode.
type PredictionInputs struct {
	Leader  *PredictionInput `json:"leader,omitempty"`
	Prefill *PredictionInput `json:"prefill,omitempty"`
	Decode  *PredictionInput `json:"decode,omitempty"`
}

// PredictionInput matches predictionInput in the PD JSON schema.
type PredictionInput struct {
	InputId              int     `json:"inputId"`
	PodType              string  `json:"podType"`
	InputTokenLength     int     `json:"inputTokenLength"`
	PrefixCacheScore     float64 `json:"prefixCacheScore"`
	NumRequestWaiting    int     `json:"numRequestWaiting"`
	NumRequestRunning    int     `json:"numRequestRunning"`
	KVCacheUsage         float64 `json:"kvCacheUsage"`
	InflightTokens       int     `json:"inflightTokens"`
	HBMUtilization       float64 `json:"hbmUtilization"`
	AICoreUtilization    float64 `json:"aiCoreUtilization"`
	BandwidthUtilization float64 `json:"bandwidthUtilization"`
	ServerIP             string  `json:"serverIP"`
}

// MissingFeatureReasons matches missingFeatureReasons in the PD JSON schema.
type MissingFeatureReasons struct {
	Prefill string `json:"prefill,omitempty"`
	Decode  string `json:"decode,omitempty"`
}

// ReadCompletedJSONL polls the EPP pod for completed.jsonl until expectedCount records
// are found or timeout expires. It returns the first expectedCount records.
func ReadCompletedJSONL(k8sClient K8SClientInterface, ctx context.Context, namespace string, expectedCount int, timeout time.Duration) ([]CompletedRecord, error) {
	pollCtx, pollCancel := context.WithTimeout(ctx, timeout)
	defer pollCancel()

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	lastReadErr := ""

	for {
		select {
		case <-pollCtx.Done():
			if lastReadErr != "" {
				return nil, fmt.Errorf("timeout waiting for %d completed records (context: %v, last error: %s)", expectedCount, pollCtx.Err(), lastReadErr)
			}
			return nil, fmt.Errorf("timeout waiting for %d completed records (context: %v)", expectedCount, pollCtx.Err())
		case <-ticker.C:
			pods, err := k8sClient.ListPods(pollCtx, namespace, metav1.ListOptions{})
			if err != nil {
				lastReadErr = fmt.Sprintf("list pods failed: %v", err)
				GinkgoWriter.Printf("ReadCompletedJSONL: %s\n", lastReadErr)
				continue
			}

			foundRunningEPP := false
			for i := range pods.Items {
				if !strings.Contains(pods.Items[i].Name, "epp") {
					continue
				}
				// 跳过正在删除（Terminating）的旧 release 残留 pod：其 Phase 仍为 Running 但
				// jsonl 文件可能已不存在，会被误选导致 kubectl exec 持续失败直至超时。
				if pods.Items[i].DeletionTimestamp != nil {
					continue
				}
				if pods.Items[i].Status.Phase != corev1.PodRunning {
					continue
				}
				foundRunningEPP = true

				cmd := fmt.Sprintf("kubectl exec %s -n %s -c epp -- cat /tmp/hermes-inflight/completed.jsonl", pods.Items[i].Name, namespace)
				stdout, stderr, err := execLocal(cmd)
				if err != nil {
					lastReadErr = fmt.Sprintf("kubectl exec failed for pod %s: %v; stderr: %s", pods.Items[i].Name, err, strings.TrimSpace(stderr))
					GinkgoWriter.Printf("ReadCompletedJSONL: %s\n", lastReadErr)
					continue
				}

				var records []CompletedRecord
				scanner := bufio.NewScanner(strings.NewReader(stdout))
				for scanner.Scan() {
					line := scanner.Text()
					if strings.TrimSpace(line) == "" {
						continue
					}
					var rec CompletedRecord
					if parseErr := json.Unmarshal([]byte(line), &rec); parseErr != nil {
						continue
					}
					records = append(records, rec)
				}

				if len(records) >= expectedCount {
					return records[len(records)-expectedCount:], nil
				}
			}

			if !foundRunningEPP {
				lastReadErr = "no running epp pod found"
				GinkgoWriter.Printf("ReadCompletedJSONL: %s\n", lastReadErr)
			}
		}
	}
}
