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

package tokenizer

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	pb "hermes-router/api/tokenizer/v1"
	"hermes-router/pkg/epp/internal/testutil"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

type initOutcome struct {
	result Backend
	err    error
	block  <-chan struct{}
}

type closeTrackingTokenizer struct {
	*FakeTokenizer
	closed atomic.Bool
}

func (t *closeTrackingTokenizer) Close() error {
	t.closed.Store(true)
	return nil
}

func newTestPlugin(tokenizer Backend, backend string) *Plugin {
	return &Plugin{
		tokenizer: tokenizer,
		backend:   backend,
		state:     initStateReady,
	}
}

func waitForCondition(t *testing.T, timeout time.Duration, check func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if check() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("expected condition to become true")
}

func newTestPluginWithInitSequence(outcomes ...initOutcome) *Plugin {
	if len(outcomes) == 0 {
		outcomes = []initOutcome{{result: &FakeTokenizer{}}}
	}

	var (
		mu          sync.Mutex
		nextOutcome int
	)

	plugin := &Plugin{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: "test-tokenizer"},
		backend:   defaultBackend,
		model:     "test-model",
		initCfg: GRPCTokenizerConfig{
			SocketPath:  "/tmp/test-tokenizer.sock",
			Model:       "test-model",
			Timeout:     defaultTimeout,
			InitTimeout: time.Second,
		},
		initCtx:     context.Background(),
		state:       initStateInitializing,
		initRetries: 2,
		// Instant-success probe: socket-wait has its own tests below.
		probeSocket: func(context.Context, string) error { return nil },
		sleep:       func(context.Context, time.Duration) error { return nil },
	}

	plugin.newTokenizer = func(ctx context.Context, _ GRPCTokenizerConfig) (Backend, error) {
		mu.Lock()
		index := nextOutcome
		if index < len(outcomes)-1 {
			nextOutcome++
		}
		outcome := outcomes[index]
		mu.Unlock()

		if outcome.block != nil {
			select {
			case <-outcome.block:
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}

		return outcome.result, outcome.err
	}

	return plugin
}

func TestPrepareRequestData_NilRequest(t *testing.T) {
	plugin := newTestPlugin(&FakeTokenizer{}, "hf")

	if err := plugin.PrepareRequestData(context.Background(), nil, nil); err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
}

func TestPrepareRequestData_AlreadyTokenized(t *testing.T) {
	existing := &fwkscheduling.TokenizedPrompt{TokenIDs: []uint32{99}}
	plugin := newTestPlugin(&FakeTokenizer{
		TokenizeFn: func(_ context.Context, _ Request) (*TokenizationResult, error) {
			t.Fatal("Tokenize should not be called when request.TokenizedPrompt is already set")
			return nil, nil
		},
	}, "hf")

	req := &fwkscheduling.LLMRequest{
		Body: &fwkscheduling.LLMRequestBody{
			Completions: &fwkscheduling.CompletionsRequest{Prompt: fwkscheduling.Prompt{Raw: "hello"}},
		},
		TokenizedPrompt: existing,
	}

	if err := plugin.PrepareRequestData(context.Background(), req, nil); err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if req.TokenizedPrompt != existing {
		t.Fatal("expected existing tokenized prompt to remain unchanged")
	}
}

func TestPrepareRequestData_CompletionHFUsesTokenize(t *testing.T) {
	plugin := newTestPlugin(&FakeTokenizer{
		TokenizeFn: func(_ context.Context, req Request) (*TokenizationResult, error) {
			if req.Model != "completion-model" {
				t.Fatalf("unexpected model %q", req.Model)
			}
			if req.Prompt != "hello world" {
				t.Fatalf("unexpected prompt %q", req.Prompt)
			}
			return &TokenizationResult{TokenIDs: []uint32{1, 2, 3}}, nil
		},
	}, "hf")

	req := &fwkscheduling.LLMRequest{
		TargetModel: "completion-model",
		Body: &fwkscheduling.LLMRequestBody{
			Completions: &fwkscheduling.CompletionsRequest{Prompt: fwkscheduling.Prompt{Raw: "hello world"}},
		},
	}

	if err := plugin.PrepareRequestData(context.Background(), req, nil); err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if req.TokenizedPrompt == nil {
		t.Fatal("expected tokenized prompt to be populated")
	}
	if got := len(req.TokenizedPrompt.TokenIDs); got != 3 {
		t.Fatalf("expected 3 token ids, got %d", got)
	}
}

func TestPrepareRequestData_CompletionArrayPromptUsesComplete(t *testing.T) {
	plugin := newTestPlugin(&FakeTokenizer{
		CompleteFn: func(_ context.Context, req CompleteRequest) (*TokenizationResult, error) {
			if req.Model != "completion-model" {
				t.Fatalf("unexpected model %q", req.Model)
			}
			if req.PromptText != "hello world" {
				t.Fatalf("unexpected prompt text %q", req.PromptText)
			}
			return &TokenizationResult{TokenIDs: []uint32{10, 20}}, nil
		},
	}, "hf")

	req := &fwkscheduling.LLMRequest{
		TargetModel: "completion-model",
		Body: &fwkscheduling.LLMRequestBody{
			Completions: &fwkscheduling.CompletionsRequest{Prompt: fwkscheduling.Prompt{Strings: []string{"hello", "world"}}},
		},
	}

	if err := plugin.PrepareRequestData(context.Background(), req, nil); err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if req.TokenizedPrompt == nil {
		t.Fatal("expected tokenized prompt to be populated")
	}
	if got := len(req.TokenizedPrompt.TokenIDs); got != 2 {
		t.Fatalf("expected 2 token ids, got %d", got)
	}
}

func TestPrepareRequestData_ChatCompletionUsesChatComplete(t *testing.T) {
	plugin := newTestPlugin(&FakeTokenizer{
		ChatCompleteFn: func(_ context.Context, req ChatCompleteRequest) (*TokenizationResult, error) {
			if req.Model != "chat-model" {
				t.Fatalf("unexpected model %q", req.Model)
			}
			if len(req.Messages) != 2 {
				t.Fatalf("expected 2 messages, got %d", len(req.Messages))
			}
			if len(req.Messages[1].ContentParts) != 2 {
				t.Fatalf("expected multimodal content parts, got %d", len(req.Messages[1].ContentParts))
			}
			return &TokenizationResult{
				TokenIDs: []uint32{5, 6, 7},
				MultiModalFeatures: []fwkscheduling.MultiModalFeature{{
					Modality: fwkscheduling.ModalityImage,
					Hash:     "hash-1",
					Offset:   1,
					Length:   2,
				}},
			}, nil
		},
	}, "vllm")

	req := &fwkscheduling.LLMRequest{
		TargetModel: "chat-model",
		Body: &fwkscheduling.LLMRequestBody{
			ChatCompletions: &fwkscheduling.ChatCompletionsRequest{
				Messages: []fwkscheduling.Message{
					{Role: "system", Content: fwkscheduling.Content{Raw: "be helpful"}},
					{Role: "user", Content: fwkscheduling.Content{Structured: []fwkscheduling.ContentBlock{
						{Type: "text", Text: "describe this"},
						{Type: "image_url", ImageURL: fwkscheduling.ImageBlock{Url: "https://example.test/image.png"}},
					}}},
				},
			},
		},
	}

	if err := plugin.PrepareRequestData(context.Background(), req, nil); err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if req.TokenizedPrompt == nil {
		t.Fatal("expected tokenized prompt to be populated")
	}
	if got := len(req.TokenizedPrompt.MultiModalFeatures); got != 1 {
		t.Fatalf("expected 1 multimodal feature, got %d", got)
	}
}

func TestPrepareRequestDataLogsTokenizationSummary(t *testing.T) {
	ctx, sink := testutil.ContextWithRecordingLogger()
	plugin := newTestPlugin(&FakeTokenizer{
		TokenizeFn: func(_ context.Context, req Request) (*TokenizationResult, error) {
			if req.Model != "summary-model" {
				t.Fatalf("unexpected model %q", req.Model)
			}
			return &TokenizationResult{TokenIDs: []uint32{1, 2, 3, 4}}, nil
		},
	}, "hf")

	req := &fwkscheduling.LLMRequest{
		TargetModel: "summary-model",
		Body: &fwkscheduling.LLMRequestBody{
			Completions: &fwkscheduling.CompletionsRequest{Prompt: fwkscheduling.Prompt{Raw: "hello"}},
		},
	}

	if err := plugin.PrepareRequestData(ctx, req, nil); err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	entry := sink.LastInfo("Tokenizer generated tokenized prompt")
	if entry.Message == "" {
		t.Fatal("expected tokenizer success log")
	}
	if entry.Keys["model"] != "summary-model" {
		t.Fatalf("unexpected model: %v", entry.Keys["model"])
	}
	if entry.Keys["tokenCount"] != 4 {
		t.Fatalf("unexpected tokenCount: %v", entry.Keys["tokenCount"])
	}
	if entry.Keys["multimodalFeatureCount"] != 0 {
		t.Fatalf("unexpected multimodalFeatureCount: %v", entry.Keys["multimodalFeatureCount"])
	}
	if _, ok := entry.Keys["durationMs"]; !ok {
		t.Fatal("expected durationMs field")
	}
}

func TestPrepareRequestDataAlreadyTokenizedLogsDebugSkip(t *testing.T) {
	ctx, sink := testutil.ContextWithRecordingLogger()
	plugin := newTestPlugin(&FakeTokenizer{}, "hf")

	req := &fwkscheduling.LLMRequest{
		Body: &fwkscheduling.LLMRequestBody{
			Completions: &fwkscheduling.CompletionsRequest{Prompt: fwkscheduling.Prompt{Raw: "hello"}},
		},
		TokenizedPrompt: &fwkscheduling.TokenizedPrompt{TokenIDs: []uint32{99}},
	}

	if err := plugin.PrepareRequestData(ctx, req, nil); err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	entry := sink.LastInfo("Tokenizer skipped tokenization")
	if entry.Message == "" {
		t.Fatal("expected tokenizer skip log")
	}
	if entry.Keys["reason"] != "already_tokenized" {
		t.Fatalf("unexpected skip reason: %v", entry.Keys["reason"])
	}
}

func TestPrepareRequestDataRuntimeFailureLogsAndFailsOpen(t *testing.T) {
	ctx, sink := testutil.ContextWithRecordingLogger()
	plugin := newTestPlugin(&FakeTokenizer{
		TokenizeFn: func(_ context.Context, _ Request) (*TokenizationResult, error) {
			return nil, fmt.Errorf("sidecar unavailable")
		},
	}, "hf")

	req := &fwkscheduling.LLMRequest{
		TargetModel: "runtime-model",
		Body: &fwkscheduling.LLMRequestBody{
			Completions: &fwkscheduling.CompletionsRequest{Prompt: fwkscheduling.Prompt{Raw: "hello"}},
		},
	}

	if err := plugin.PrepareRequestData(ctx, req, nil); err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if req.TokenizedPrompt != nil {
		t.Fatal("expected tokenized prompt to remain nil on runtime error")
	}
	entry := sink.LastError("Tokenizer request tokenization failed; continuing without tokenized prompt")
	if entry.Message == "" {
		t.Fatal("expected runtime fail-open error log")
	}
	if entry.Keys["requestKind"] != "completion" {
		t.Fatalf("unexpected requestKind: %v", entry.Keys["requestKind"])
	}
	if entry.Keys["model"] != "runtime-model" {
		t.Fatalf("unexpected model: %v", entry.Keys["model"])
	}
	if entry.Keys["backend"] != "hf" {
		t.Fatalf("unexpected backend: %v", entry.Keys["backend"])
	}
}

func TestFactoryStartsBackgroundInitWorker(t *testing.T) {
	initializeStarted := make(chan struct{}, 1)
	releaseInitialize := make(chan struct{})
	defer close(releaseInitialize)

	socketPath := startFakeTokenizationServer(t, &fakeTokenizationService{
		initializeFn: func(_ context.Context, req *pb.InitializeRequest) (*pb.InitializeResponse, error) {
			select {
			case initializeStarted <- struct{}{}:
			default:
			}
			<-releaseInitialize
			return &pb.InitializeResponse{
				Model:            req.Model,
				ResolvedProvider: pb.TokenizerProvider_TOKENIZER_PROVIDER_HUGGINGFACE,
			}, nil
		},
	})

	pluginValue, err := Factory(
		"async-tokenizer",
		[]byte(fmt.Sprintf(`{"model":"background-model","socketPath":%q}`, socketPath)),
		fwkplugin.NewEppHandle(context.Background(), nil),
	)
	if err != nil {
		t.Fatalf("expected Factory to start background init worker, got %v", err)
	}
	plugin := pluginValue.(*Plugin)

	select {
	case <-initializeStarted:
	case <-time.After(200 * time.Millisecond):
		t.Fatal("expected background initialization worker to start")
	}

	_, state, _ := plugin.snapshotInitState()
	if state != initStateInitializing {
		t.Fatalf("expected initializing state, got %v", state)
	}
}

func TestPrepareRequestDataWaitsForReadyTokenizerWithoutStartingInit(t *testing.T) {
	releaseInitialize := make(chan struct{})
	plugin := newTestPluginWithInitSequence(initOutcome{
		result: &FakeTokenizer{
			TokenizeFn: func(_ context.Context, req Request) (*TokenizationResult, error) {
				if req.Model != "wait-model" {
					t.Fatalf("unexpected model %q", req.Model)
				}
				if req.Prompt != "hello world" {
					t.Fatalf("unexpected prompt %q", req.Prompt)
				}
				return &TokenizationResult{TokenIDs: []uint32{1, 2, 3}}, nil
			},
		},
		block: releaseInitialize,
	})
	originalNewTokenizer := plugin.newTokenizer
	initStarted := make(chan struct{}, 1)
	unexpectedInitAttempt := make(chan struct{}, 1)
	var (
		initCallsMu sync.Mutex
		initCalls   int
	)
	plugin.newTokenizer = func(ctx context.Context, cfg GRPCTokenizerConfig) (Backend, error) {
		initCallsMu.Lock()
		initCalls++
		callNumber := initCalls
		initCallsMu.Unlock()

		if callNumber == 1 {
			select {
			case initStarted <- struct{}{}:
			default:
			}
		} else {
			select {
			case unexpectedInitAttempt <- struct{}{}:
			default:
			}
		}

		return originalNewTokenizer(ctx, cfg)
	}
	go plugin.runInitLoop()

	select {
	case <-initStarted:
	case <-time.After(time.Second):
		t.Fatal("expected background initialization attempt to start")
	}

	req := &fwkscheduling.LLMRequest{
		TargetModel: "wait-model",
		Body: &fwkscheduling.LLMRequestBody{
			Completions: &fwkscheduling.CompletionsRequest{Prompt: fwkscheduling.Prompt{Raw: "hello world"}},
		},
	}

	_, state, _ := plugin.snapshotInitState()
	if state != initStateInitializing {
		t.Fatalf("expected initializing state, got %v", state)
	}

	done := make(chan error, 1)
	go func() {
		done <- plugin.PrepareRequestData(context.Background(), req, nil)
	}()

	select {
	case err := <-done:
		t.Fatalf("expected request to wait for tokenizer readiness, returned early with %v", err)
	case <-unexpectedInitAttempt:
		t.Fatal("expected PrepareRequestData to wait for readiness without starting another initialization attempt")
	case <-time.After(50 * time.Millisecond):
	}
	close(releaseInitialize)

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("expected nil error after tokenizer becomes ready, got %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("expected request to finish once tokenizer becomes ready")
	}

	select {
	case <-unexpectedInitAttempt:
		t.Fatal("expected exactly one tokenizer initialization attempt while waiting for readiness")
	default:
	}

	initCallsMu.Lock()
	gotInitCalls := initCalls
	initCallsMu.Unlock()
	if gotInitCalls != 1 {
		t.Fatalf("expected exactly 1 tokenizer initialization attempt, got %d", gotInitCalls)
	}
	if req.TokenizedPrompt == nil {
		t.Fatal("expected request to use tokenizer once the background worker makes it ready")
	}
}

func TestPrepareRequestDataFailsOpenWithErrorLogWhenTokenizerStaysUnavailable(t *testing.T) {
	ctx, sink := testutil.ContextWithRecordingLogger()
	blockInit := make(chan struct{})
	defer close(blockInit)

	initCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	plugin := newTestPluginWithInitSequence(initOutcome{
		err:   fmt.Errorf("tokenizer still warming"),
		block: blockInit,
	})
	plugin.initCtx = initCtx
	go plugin.runInitLoop()

	req := &fwkscheduling.LLMRequest{
		TargetModel: "fail-open-model",
		Body: &fwkscheduling.LLMRequestBody{
			Completions: &fwkscheduling.CompletionsRequest{Prompt: fwkscheduling.Prompt{Raw: "hello"}},
		},
	}

	if err := plugin.PrepareRequestData(ctx, req, nil); err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if req.TokenizedPrompt != nil {
		t.Fatal("expected request to fail open while tokenizer is unavailable")
	}
	if sink.LastError("Tokenizer unavailable after bounded readiness wait; continuing without tokenized prompt").Message == "" {
		t.Fatal("expected error log when tokenizer never becomes ready")
	}
}

func TestBackgroundInitWorkerLogsFailureThenReady(t *testing.T) {
	ctx, sink := testutil.ContextWithRecordingLogger()
	readyTokenizer := &FakeTokenizer{}
	plugin := newTestPluginWithInitSequence(
		initOutcome{err: fmt.Errorf("dial tokenizer grpc server: not ready")},
		initOutcome{result: readyTokenizer},
	)
	plugin.initCtx = ctx

	go plugin.runInitLoop()

	waitForCondition(t, time.Second, func() bool {
		tokenizer, state, err := plugin.snapshotInitState()
		return tokenizer == readyTokenizer && state == initStateReady && err == nil
	})

	tokenizer, state, err := plugin.snapshotInitState()
	if tokenizer != readyTokenizer || state != initStateReady || err != nil {
		t.Fatalf("expected ready tokenizer after retries, got tokenizer=%v state=%v err=%v", tokenizer, state, err)
	}
	if sink.LastError("Tokenizer initialization failed; retrying").Message == "" {
		t.Fatal("expected retry error log before tokenizer becomes ready")
	}
	if sink.LastInfo("Tokenizer initialization succeeded").Message == "" {
		t.Fatal("expected ready info log once tokenizer becomes available")
	}
}

func TestRunInitLoopWaitsForSocketBeforeInitialize(t *testing.T) {
	ctx, sink := testutil.ContextWithRecordingLogger()
	var initCalls atomic.Int32
	plugin := newTestPluginWithInitSequence(initOutcome{result: &FakeTokenizer{}})
	plugin.initCtx = ctx
	plugin.socketWaitTimeout = 50 * time.Millisecond
	plugin.probeSocket = func(context.Context, string) error {
		return fmt.Errorf("socket not ready")
	}
	plugin.newTokenizer = func(context.Context, GRPCTokenizerConfig) (Backend, error) {
		initCalls.Add(1)
		return &FakeTokenizer{}, nil
	}

	go plugin.runInitLoop()

	waitForCondition(t, 2*time.Second, func() bool {
		_, state, _ := plugin.snapshotInitState()
		return state == initStateFailed
	})
	time.Sleep(100 * time.Millisecond)
	if got := int(initCalls.Load()); got != 0 {
		t.Fatalf("expected no Initialize attempts while socket is down, got %d", got)
	}
	if sink.LastInfo("Waiting for tokenizer socket").Message == "" {
		t.Fatal("expected socket wait log")
	}
}

func TestRunInitLoopRecoversAfterSocketWaitTimeout(t *testing.T) {
	testStart := time.Now()
	var initCalls atomic.Int32
	plugin := newTestPluginWithInitSequence(initOutcome{result: &FakeTokenizer{}})
	plugin.socketWaitTimeout = 50 * time.Millisecond
	plugin.probeSocket = func(context.Context, string) error {
		if time.Since(testStart) < 100*time.Millisecond {
			return fmt.Errorf("socket not ready")
		}
		return nil
	}
	plugin.sleep = func(context.Context, time.Duration) error { return nil }
	originalNewTokenizer := plugin.newTokenizer
	plugin.newTokenizer = func(ctx context.Context, cfg GRPCTokenizerConfig) (Backend, error) {
		initCalls.Add(1)
		return originalNewTokenizer(ctx, cfg)
	}

	go plugin.runInitLoop()

	waitForCondition(t, 2*time.Second, func() bool {
		_, state, _ := plugin.snapshotInitState()
		return state == initStateReady
	})
	if got := int(initCalls.Load()); got != 1 {
		t.Fatalf("expected exactly 1 Initialize attempt (attempt 1 consumed by socket-wait timeout), got %d", got)
	}
	if elapsed := time.Since(testStart); elapsed < 50*time.Millisecond {
		t.Fatalf("expected attempt 1 to burn its socket-wait budget, ready after only %v", elapsed)
	}
}

func TestRunInitLoopProbesSocketThenInitializesOnce(t *testing.T) {
	plugin := newTestPluginWithInitSequence(initOutcome{result: &FakeTokenizer{}})
	plugin.initCfg.InitTimeout = 5 * time.Second
	var deadline time.Duration
	originalNewTokenizer := plugin.newTokenizer
	plugin.newTokenizer = func(ctx context.Context, cfg GRPCTokenizerConfig) (Backend, error) {
		if d, ok := ctx.Deadline(); ok {
			deadline = time.Until(d)
		}
		return originalNewTokenizer(ctx, cfg)
	}

	go plugin.runInitLoop()

	waitForCondition(t, time.Second, func() bool {
		_, state, _ := plugin.snapshotInitState()
		return state == initStateReady
	})
	if deadline <= 4*time.Second || deadline > 5*time.Second {
		t.Fatalf("expected Initialize deadline ≈ initTimeout (5s), got %v", deadline)
	}
}

func TestRunInitLoopRetriesWithFixedInterval(t *testing.T) {
	var (
		delaysMu sync.Mutex
		delays   []time.Duration
	)
	plugin := newTestPluginWithInitSequence(
		initOutcome{err: fmt.Errorf("transient failure")},
		initOutcome{result: &FakeTokenizer{}},
	)
	plugin.sleep = func(_ context.Context, delay time.Duration) error {
		delaysMu.Lock()
		delays = append(delays, delay)
		delaysMu.Unlock()
		return nil
	}

	go plugin.runInitLoop()

	waitForCondition(t, time.Second, func() bool {
		_, state, _ := plugin.snapshotInitState()
		return state == initStateReady
	})

	delaysMu.Lock()
	gotDelays := append([]time.Duration(nil), delays...)
	delaysMu.Unlock()
	if len(gotDelays) != 1 {
		t.Fatalf("expected exactly 1 between-attempt sleep, got %d: %v", len(gotDelays), gotDelays)
	}
	if gotDelays[0] != initRetryInterval {
		t.Fatalf("expected fixed retry interval %v, got %v", initRetryInterval, gotDelays[0])
	}
}

func TestRunInitLoopStopsOnPermanentError(t *testing.T) {
	ctx, sink := testutil.ContextWithRecordingLogger()
	var initCalls atomic.Int32
	plugin := newTestPluginWithInitSequence(
		initOutcome{err: status.Error(codes.InvalidArgument, "bad model")},
		initOutcome{result: &FakeTokenizer{}},
	)
	plugin.initCtx = ctx
	originalNewTokenizer := plugin.newTokenizer
	plugin.newTokenizer = func(ctx context.Context, cfg GRPCTokenizerConfig) (Backend, error) {
		initCalls.Add(1)
		return originalNewTokenizer(ctx, cfg)
	}

	go plugin.runInitLoop()

	waitForCondition(t, time.Second, func() bool {
		_, state, _ := plugin.snapshotInitState()
		return state == initStateFailed
	})
	time.Sleep(100 * time.Millisecond)
	if got := int(initCalls.Load()); got != 1 {
		t.Fatalf("expected permanent error to stop after 1 attempt, got %d", got)
	}
	if sink.LastError("Tokenizer initialization failed; giving up: permanent error").Message == "" {
		t.Fatal("expected permanent give-up log")
	}
}

func TestRunInitLoopStopsOnClientContractError(t *testing.T) {
	plugin := newTestPluginWithInitSequence(
		initOutcome{err: fmt.Errorf("%w: got %q want %q", errModelMismatch, "a", "b")},
		initOutcome{result: &FakeTokenizer{}},
	)
	var initCalls atomic.Int32
	originalNewTokenizer := plugin.newTokenizer
	plugin.newTokenizer = func(ctx context.Context, cfg GRPCTokenizerConfig) (Backend, error) {
		initCalls.Add(1)
		return originalNewTokenizer(ctx, cfg)
	}

	go plugin.runInitLoop()

	waitForCondition(t, time.Second, func() bool {
		_, state, _ := plugin.snapshotInitState()
		return state == initStateFailed
	})
	time.Sleep(100 * time.Millisecond)
	if got := int(initCalls.Load()); got != 1 {
		t.Fatalf("expected client contract error to stop after 1 attempt, got %d", got)
	}
}

func TestRunInitLoopGivesUpAfterInitRetries(t *testing.T) {
	ctx, sink := testutil.ContextWithRecordingLogger()
	plugin := newTestPluginWithInitSequence(
		initOutcome{err: fmt.Errorf("dial tokenizer grpc server: not ready")},
	)
	plugin.initCtx = ctx
	plugin.initRetries = 2
	var initCalls atomic.Int32
	originalNewTokenizer := plugin.newTokenizer
	plugin.newTokenizer = func(ctx context.Context, cfg GRPCTokenizerConfig) (Backend, error) {
		initCalls.Add(1)
		return originalNewTokenizer(ctx, cfg)
	}

	go plugin.runInitLoop()

	waitForCondition(t, time.Second, func() bool {
		_, state, _ := plugin.snapshotInitState()
		return state == initStateFailed
	})
	time.Sleep(100 * time.Millisecond)
	if got := int(initCalls.Load()); got != 3 {
		t.Fatalf("expected 3 attempts (1 + 2 retries), got %d", got)
	}
	if sink.LastError("Tokenizer initialization failed; giving up after init retries").Message == "" {
		t.Fatal("expected give-up log after retries exhausted")
	}
}

func TestRunInitLoopReprobesSocketBetweenAttempts(t *testing.T) {
	var probeCalls atomic.Int32
	plugin := newTestPluginWithInitSequence(
		initOutcome{err: fmt.Errorf("transient failure")},
		initOutcome{result: &FakeTokenizer{}},
	)
	plugin.probeSocket = func(context.Context, string) error {
		probeCalls.Add(1)
		return nil
	}

	go plugin.runInitLoop()

	waitForCondition(t, time.Second, func() bool {
		_, state, _ := plugin.snapshotInitState()
		return state == initStateReady
	})
	if got := int(probeCalls.Load()); got != 2 {
		t.Fatalf("expected socket re-probe before the second attempt, got %d probes", got)
	}
}

func TestRunInitLoopKeepsInitializingStateBetweenAttempts(t *testing.T) {
	var initCalls atomic.Int32
	holdSecondAttempt := make(chan struct{})
	plugin := newTestPluginWithInitSequence(
		initOutcome{err: fmt.Errorf("transient failure")},
		initOutcome{result: &FakeTokenizer{}},
	)
	originalNewTokenizer := plugin.newTokenizer
	plugin.newTokenizer = func(ctx context.Context, cfg GRPCTokenizerConfig) (Backend, error) {
		if initCalls.Add(1) == 2 {
			<-holdSecondAttempt
		}
		return originalNewTokenizer(ctx, cfg)
	}

	go plugin.runInitLoop()

	waitForCondition(t, time.Second, func() bool {
		return initCalls.Load() == 2
	})

	_, state, _ := plugin.snapshotInitState()
	if state != initStateInitializing {
		t.Fatalf("expected initializing state during retry attempt, got %v", state)
	}

	close(holdSecondAttempt)
	waitForCondition(t, time.Second, func() bool {
		_, state, _ := plugin.snapshotInitState()
		return state == initStateReady
	})
}

func TestBackgroundInitWorkerStopsOnContextCancel(t *testing.T) {
	initCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var initCalls atomic.Int32
	plugin := newTestPluginWithInitSequence(
		initOutcome{err: fmt.Errorf("dial tokenizer grpc server: not ready")},
		initOutcome{result: &FakeTokenizer{}},
	)
	plugin.initCtx = initCtx
	plugin.sleep = func(ctx context.Context, _ time.Duration) error {
		<-ctx.Done()
		return ctx.Err()
	}
	originalNewTokenizer := plugin.newTokenizer
	plugin.newTokenizer = func(ctx context.Context, cfg GRPCTokenizerConfig) (Backend, error) {
		initCalls.Add(1)
		return originalNewTokenizer(ctx, cfg)
	}

	go plugin.runInitLoop()

	waitForCondition(t, time.Second, func() bool {
		return int(initCalls.Load()) == 1
	})

	cancel()

	waitForCondition(t, 200*time.Millisecond, func() bool {
		_, state, _ := plugin.snapshotInitState()
		return state == initStateInitializing
	})

	time.Sleep(100 * time.Millisecond)
	if got := int(initCalls.Load()); got != 1 {
		t.Fatalf("expected worker to stop after context cancel, attempts grew to %d", got)
	}
	tokenizer, state, err := plugin.snapshotInitState()
	if tokenizer != nil || state != initStateInitializing || err == nil {
		t.Fatalf("expected stopped worker holding recorded failure in initializing state, got tokenizer=%v state=%v err=%v", tokenizer, state, err)
	}
}

func TestRegisterShutdownCloseClosesTokenizer(t *testing.T) {
	initCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	closer := &closeTrackingTokenizer{FakeTokenizer: &FakeTokenizer{}}
	plugin := &Plugin{
		initCtx:   initCtx,
		tokenizer: closer,
		state:     initStateReady,
	}

	plugin.registerShutdownClose()
	cancel()

	waitForCondition(t, 200*time.Millisecond, func() bool {
		return closer.closed.Load()
	})
	if !closer.closed.Load() {
		t.Fatal("expected shutdown cleanup to close tokenizer")
	}
}

func TestParseAndValidateParams(t *testing.T) {
	tests := []struct {
		name    string
		raw     []byte
		wantErr string
	}{
		{name: "nil", wantErr: "parameters are required"},
		{name: "missing model", raw: []byte(`{"socketPath":"/tmp/t.sock"}`), wantErr: "model is required"},
		{name: "bad backend", raw: []byte(`{"model":"m","socketPath":"/tmp/t.sock","backend":"bad"}`), wantErr: "backend must be"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := parseAndValidateParams(test.raw)
			if err == nil {
				t.Fatalf("expected error containing %q", test.wantErr)
			}
			if got := err.Error(); got == "" || !strings.Contains(got, test.wantErr) {
				t.Fatalf("expected error containing %q, got %q", test.wantErr, got)
			}
		})
	}

	params, err := parseAndValidateParams([]byte(`{"model":"m","socketPath":"/tmp/t.sock"}`))
	if err != nil {
		t.Fatalf("expected defaults to parse, got %v", err)
	}
	if params.Backend != defaultBackend {
		t.Fatalf("expected default backend %q, got %q", defaultBackend, params.Backend)
	}
}

func TestParseAndValidateParamsSocketPath(t *testing.T) {
	params, err := parseAndValidateParams([]byte(`{"model":"m"}`))
	if err != nil {
		t.Fatalf("expected params without socketPath to parse, got %v", err)
	}
	if params.SocketPath != defaultSocketPath {
		t.Fatalf("expected default socketPath %q, got %q", defaultSocketPath, params.SocketPath)
	}

	params, err = parseAndValidateParams([]byte(`{"model":"m","socketPath":"/custom/t.sock"}`))
	if err != nil {
		t.Fatalf("expected explicit socketPath to parse, got %v", err)
	}
	if params.SocketPath != "/custom/t.sock" {
		t.Fatalf("expected explicit socketPath to be kept, got %q", params.SocketPath)
	}
}

func TestParseAndValidateParamsTokenizerSource(t *testing.T) {
	params, err := parseAndValidateParams([]byte(`{"model":"glm-51","socketPath":"/tmp/t.sock","tokenizerSource":"/models/glm-base-tok"}`))
	if err != nil {
		t.Fatalf("expected params to parse, got %v", err)
	}
	if params.TokenizerName != "/models/glm-base-tok" {
		t.Fatalf("expected TokenizerName to be populated from the tokenizerSource key, got %q", params.TokenizerName)
	}
}

func TestParseAndValidateParamsInitControls(t *testing.T) {
	params, err := parseAndValidateParams([]byte(`{"model":"m","socketPath":"/tmp/t.sock"}`))
	if err != nil {
		t.Fatalf("default parse: %v", err)
	}
	if params.InitTimeout.Std() != defaultInitTimeout {
		t.Fatalf("expected default initTimeout %v, got %v", defaultInitTimeout, params.InitTimeout.Std())
	}
	if params.InitRetries == nil || *params.InitRetries != defaultInitRetries {
		t.Fatalf("expected default initRetries %d, got %v", defaultInitRetries, params.InitRetries)
	}

	params, err = parseAndValidateParams([]byte(`{"model":"m","socketPath":"/tmp/t.sock","initTimeout":"30s","initRetries":0}`))
	if err != nil {
		t.Fatalf("explicit parse: %v", err)
	}
	if params.InitTimeout.Std() != 30*time.Second {
		t.Fatalf("expected initTimeout 30s, got %v", params.InitTimeout.Std())
	}
	if params.InitRetries == nil || *params.InitRetries != 0 {
		t.Fatalf("expected explicit initRetries 0 preserved, got %v", params.InitRetries)
	}

	if _, err := parseAndValidateParams([]byte(`{"model":"m","socketPath":"/tmp/t.sock","initRetries":-1}`)); err == nil {
		t.Fatal("expected error for negative initRetries")
	}
}

func TestIsPermanentInitError(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{name: "model mismatch sentinel", err: fmt.Errorf("%w: got %q want %q", errModelMismatch, "a", "b"), want: true},
		{name: "provider sentinel", err: errProviderUnspecified, want: true},
		{name: "invalid argument", err: status.Error(codes.InvalidArgument, "bad model"), want: true},
		{name: "unimplemented", err: status.Error(codes.Unimplemented, "render unavailable"), want: true},
		{name: "wrapped invalid argument", err: fmt.Errorf("initialize tokenizer: %w", status.Error(codes.InvalidArgument, "bad model")), want: true},
		{name: "deadline exceeded", err: status.Error(codes.DeadlineExceeded, "slow"), want: false},
		{name: "unavailable", err: status.Error(codes.Unavailable, "socket gone"), want: false},
		{name: "plain error", err: fmt.Errorf("dial tokenizer grpc server: nope"), want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isPermanentInitError(tt.err); got != tt.want {
				t.Fatalf("isPermanentInitError(%v) = %v, want %v", tt.err, got, tt.want)
			}
		})
	}
}
