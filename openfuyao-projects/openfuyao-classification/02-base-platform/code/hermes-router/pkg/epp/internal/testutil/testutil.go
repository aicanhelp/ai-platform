/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Package testutil provides shared test-only constructors and fakes for EPP plugin tests.
package testutil

import (
	"context"
	"fmt"
	"math"
	"net/url"
	"sync"
	"testing"

	"github.com/go-logr/logr"
	metav1types "k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkreq "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

const (
	// TestNamespace is an exported constant used by Hermes Router.
	TestNamespace = "default"
	// TestInferencePort is an exported constant used by Hermes Router.
	TestInferencePort = "8000"
	// TestMetricsPort is an exported constant used by Hermes Router.
	TestMetricsPort = "9090"
	floatTolerance  = 1e-9

	logKeyValuePairWidth = 2
)

// RecordedLogEntry describes RecordedLogEntry data used by Hermes Router.
type RecordedLogEntry struct {
	// Message stores the Message value.
	Message string
	// Error stores the Error value.
	Error string
	// Keys stores the Keys value.
	Keys map[string]any
}

// RecordingSink describes RecordingSink data used by Hermes Router.
type RecordingSink struct {
	mu     sync.Mutex
	infos  []RecordedLogEntry
	errors []RecordedLogEntry
}

// ContextWithRecordingLogger provides the exported ContextWithRecordingLogger operation.
func ContextWithRecordingLogger() (context.Context, *RecordingSink) {
	sink := &RecordingSink{}
	logger := logr.New(sink)
	return log.IntoContext(context.Background(), logger), sink
}

// Init provides the exported Init operation.
func (s *RecordingSink) Init(logr.RuntimeInfo) {
	// RecordingSink has no runtime state to initialize.
}

// Enabled provides the exported Enabled operation.
func (s *RecordingSink) Enabled(int) bool { return true }

// Info provides the exported Info operation.
func (s *RecordingSink) Info(_ int, msg string, keysAndValues ...any) {
	s.record(&s.infos, RecordedLogEntry{Message: msg, Keys: keyValuesMap(keysAndValues)})
}

// Error provides the exported Error operation.
func (s *RecordingSink) Error(err error, msg string, keysAndValues ...any) {
	entry := RecordedLogEntry{Message: msg, Keys: keyValuesMap(keysAndValues)}
	if err != nil {
		entry.Error = err.Error()
	}
	s.record(&s.errors, entry)
}

// WithValues provides the exported WithValues operation.
func (s *RecordingSink) WithValues(...any) logr.LogSink { return s }

// WithName provides the exported WithName operation.
func (s *RecordingSink) WithName(string) logr.LogSink { return s }

// LastInfo provides the exported LastInfo operation.
func (s *RecordingSink) LastInfo(message string) RecordedLogEntry {
	s.mu.Lock()
	defer s.mu.Unlock()
	for index := len(s.infos) - 1; index >= 0; index-- {
		if s.infos[index].Message == message {
			return s.infos[index]
		}
	}
	return RecordedLogEntry{}
}

// LastError provides the exported LastError operation.
func (s *RecordingSink) LastError(message string) RecordedLogEntry {
	s.mu.Lock()
	defer s.mu.Unlock()
	for index := len(s.errors) - 1; index >= 0; index-- {
		if s.errors[index].Message == message {
			return s.errors[index]
		}
	}
	return RecordedLogEntry{}
}

// NewHandle constructs a Handle instance.
func NewHandle() fwkplugin.Handle {
	return fwkplugin.NewEppHandle(context.Background(), func() []metav1types.NamespacedName { return nil })
}

// NewEndpoint constructs a Endpoint instance.
func NewEndpoint(name, address, port string, labels map[string]string) fwkscheduling.Endpoint {
	metrics := fwkdl.NewMetrics()
	metadata := &fwkdl.EndpointMetadata{
		NamespacedName: metav1types.NamespacedName{Namespace: TestNamespace, Name: name},
		PodName:        name,
		Address:        address,
		Port:           port,
		MetricsHost:    address + ":" + TestMetricsPort,
		Labels:         cloneLabels(labels),
	}
	return fwkscheduling.NewEndpoint(metadata, metrics, fwkdl.NewAttributes())
}

// ScoredEndpointMetrics carries the scored test endpoint metrics.
type ScoredEndpointMetrics struct {
	// Score is the initial scheduling score.
	Score float64
	// Waiting is the waiting queue size metric.
	Waiting int
	// KVUsage is the KV cache usage percent metric.
	KVUsage float64
	// Running is the running request count metric.
	Running int
}

// NewScoredEndpoint constructs a ScoredEndpoint instance.
func NewScoredEndpoint(
	name, address, port string,
	scoredMetrics ScoredEndpointMetrics,
	labels map[string]string,
) fwkscheduling.ScoredEndpoint {
	metrics := fwkdl.NewMetrics()
	metrics.WaitingQueueSize = scoredMetrics.Waiting
	metrics.KVCacheUsagePercent = scoredMetrics.KVUsage
	metrics.RunningRequestsSize = scoredMetrics.Running
	metadata := &fwkdl.EndpointMetadata{
		NamespacedName: metav1types.NamespacedName{Namespace: TestNamespace, Name: name},
		PodName:        name,
		Address:        address,
		Port:           port,
		MetricsHost:    address + ":" + TestMetricsPort,
		Labels:         cloneLabels(labels),
	}
	return fwkscheduling.ScoredEndpoint{
		Endpoint: fwkscheduling.NewEndpoint(metadata, metrics, fwkdl.NewAttributes()),
		Score:    scoredMetrics.Score,
	}
}

// NewCompletionsRequest constructs a CompletionsRequest instance.
func NewCompletionsRequest(requestID, model, prompt string) *fwkscheduling.LLMRequest {
	return &fwkscheduling.LLMRequest{
		RequestId:   requestID,
		TargetModel: model,
		Headers:     map[string]string{},
		Body: &fwkscheduling.LLMRequestBody{
			Completions: &fwkscheduling.CompletionsRequest{Prompt: fwkscheduling.Prompt{Raw: prompt}},
		},
	}
}

// NewChatRequest constructs a ChatRequest instance.
func NewChatRequest(requestID, model string, messages ...string) *fwkscheduling.LLMRequest {
	chatMessages := make([]fwkscheduling.Message, 0, len(messages))
	for _, message := range messages {
		chatMessages = append(chatMessages, fwkscheduling.Message{Content: fwkscheduling.Content{Raw: message}})
	}
	return &fwkscheduling.LLMRequest{
		RequestId:   requestID,
		TargetModel: model,
		Headers:     map[string]string{},
		Body: &fwkscheduling.LLMRequestBody{
			ChatCompletions: &fwkscheduling.ChatCompletionsRequest{Messages: chatMessages},
		},
	}
}

// MustSplitHostPort provides the exported MustSplitHostPort operation.
func MustSplitHostPort(t *testing.T, rawURL string) (string, int) {
	t.Helper()
	parsed, err := url.Parse(rawURL)
	if err != nil {
		t.Fatalf("parse url: %v", err)
	}
	hostname := parsed.Hostname()
	port := 0
	if parsed.Port() != "" {
		_, err = fmt.Sscanf(parsed.Port(), "%d", &port)
		if err != nil {
			t.Fatalf("parse port: %v", err)
		}
	}
	return hostname, port
}

// NewSchedulingResult constructs a SchedulingResult instance.
func NewSchedulingResult(targets ...fwkscheduling.Endpoint) *fwkscheduling.SchedulingResult {
	return &fwkscheduling.SchedulingResult{
		PrimaryProfileName: "primary",
		ProfileResults: map[string]*fwkscheduling.ProfileRunResult{
			"primary": {
				TargetEndpoints: targets,
			},
		},
	}
}

// EndOfStreamResponse provides the exported EndOfStreamResponse operation.
func EndOfStreamResponse() *fwkreq.Response {
	return &fwkreq.Response{EndOfStream: true}
}

// AssertFloatClose provides the exported AssertFloatClose operation.
func AssertFloatClose(t *testing.T, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > floatTolerance {
		t.Fatalf("expected %v, got %v", want, got)
	}
}

func cloneLabels(labels map[string]string) map[string]string {
	if labels == nil {
		return map[string]string{}
	}
	cloned := make(map[string]string, len(labels))
	for key, value := range labels {
		cloned[key] = value
	}
	return cloned
}

func (s *RecordingSink) record(target *[]RecordedLogEntry, entry RecordedLogEntry) {
	s.mu.Lock()
	defer s.mu.Unlock()
	*target = append(*target, entry)
}

func keyValuesMap(keysAndValues []any) map[string]any {
	result := map[string]any{}
	for index := 0; index < len(keysAndValues); index += logKeyValuePairWidth {
		if index+1 >= len(keysAndValues) {
			break
		}
		key, ok := keysAndValues[index].(string)
		if !ok {
			continue
		}
		result[key] = keysAndValues[index+1]
	}
	return result
}
