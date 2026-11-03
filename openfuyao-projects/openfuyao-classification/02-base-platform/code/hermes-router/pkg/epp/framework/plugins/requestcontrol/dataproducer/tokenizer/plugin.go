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
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/go-logr/logr"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	tokenizedpromptattr "hermes-router/pkg/epp/framework/plugins/datalayer/attribute/tokenizedprompt"
	"hermes-router/pkg/epp/internal/utils"
)

const (
	// PluginType is the registered plugin type identifier.
	PluginType            = "tokenizer"
	defaultBackend        = "hf"
	defaultTimeout        = 5 * time.Second
	defaultMaxMessageSize = 4 * 1024 * 1024
	// Extra attempts after the first; explicit initRetries 0 means a single try.
	defaultInitRetries = 2
	// Cold start (transformers/vLLM import, optional HF fetch, warmup) measured
	// ~23s worst case; 2m is roughly 5x headroom.
	defaultInitTimeout = 2 * time.Minute
	// Per-attempt UDS wait. Sidecar usually binds in seconds; 2m covers slow start.
	defaultSocketWaitTimeout   = 2 * time.Minute
	prepareRequestWaitAttempts = 3
	prepareRequestWaitInterval = 100 * time.Millisecond
	// Routed through sleep so tests can skip the poll delay.
	socketProbeInterval = 200 * time.Millisecond
	initRetryInterval   = 1 * time.Second
	// defaultSocketPath matches the llm-d UDS tokenizer convention and the
	// kserve scheduler preset's tokenizer-uds volume mount (/tmp/tokenizer),
	// so an EPP config that omits socketPath still reaches the sidecar.
	defaultSocketPath = "/tmp/tokenizer/tokenizer-uds.socket"
)

type initState string

const (
	initStateInitializing initState = "initializing"
	initStateReady        initState = "ready"
	initStateFailed       initState = "failed"
)

type tokenizerInitializer func(context.Context, GRPCTokenizerConfig) (Backend, error)
type sleepFunc func(context.Context, time.Duration) error

func defaultTokenizerInitializer(ctx context.Context, cfg GRPCTokenizerConfig) (Backend, error) {
	return NewGRPCTokenizer(ctx, cfg)
}

type pluginParameters struct {
	Model      string `json:"model"`
	SocketPath string `json:"socketPath"`
	Backend    string `json:"backend"`
	// tokenizerSource overrides where the tokenizer is loaded from when it
	// differs from the served model name (e.g. a quantized model served as
	// "glm-51" whose tokenizer lives at a custom path or repo id). Empty falls
	// back to model. Only the operator-facing json key is renamed; the struct
	// field name stays TokenizerName to minimize churn (see design spec, Scope A).
	TokenizerName  string         `json:"tokenizerSource"`
	Timeout        utils.Duration `json:"timeout"`
	InitTimeout    utils.Duration `json:"initTimeout"`
	MaxMessageSize int            `json:"maxMessageSize"`
	// Pointer so explicit 0 (single attempt) is distinct from an omitted key.
	InitRetries *int `json:"initRetries"`
}

// Plugin populates request.TokenizedPrompt before scheduling.
type Plugin struct {
	typedName         fwkplugin.TypedName
	tokenizer         Backend
	backend           string
	model             string
	initCfg           GRPCTokenizerConfig
	initCtx           context.Context
	initMu            sync.RWMutex
	state             initState
	lastInitErr       error
	newTokenizer      tokenizerInitializer
	sleep             sleepFunc
	probeSocket       func(ctx context.Context, socketPath string) error
	socketWaitTimeout time.Duration
	// Retry policy for the plugin loop, not gRPC client config.
	initRetries int
}

var _ requestcontrol.PrepareDataPlugin = (*Plugin)(nil)

// Factory builds a tokenizer plugin and validates its config without blocking EPP startup.
func Factory(name string, parameters json.RawMessage, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params, err := parseAndValidateParams(parameters)
	if err != nil {
		return nil, fmt.Errorf("tokenizer plugin %q: %w", name, err)
	}

	plugin := &Plugin{
		typedName:         fwkplugin.TypedName{Type: PluginType, Name: name},
		backend:           params.Backend,
		model:             params.Model,
		initCtx:           initContext(handle),
		state:             initStateInitializing,
		newTokenizer:      defaultTokenizerInitializer,
		sleep:             waitWithContext,
		probeSocket:       probeTokenizerSocket,
		socketWaitTimeout: defaultSocketWaitTimeout,
		initRetries:       *params.InitRetries,
		initCfg: GRPCTokenizerConfig{
			SocketPath:     params.SocketPath,
			Model:          params.Model,
			TokenizerName:  params.TokenizerName,
			Timeout:        params.Timeout.Std(),
			InitTimeout:    params.InitTimeout.Std(),
			MaxMessageSize: params.MaxMessageSize,
		},
	}
	plugin.registerShutdownClose()
	go plugin.runInitLoop()
	return plugin, nil
}

func initContext(handle fwkplugin.Handle) context.Context {
	if handle == nil || handle.Context() == nil {
		return context.Background()
	}
	return handle.Context()
}

func parseAndValidateParams(raw []byte) (pluginParameters, error) {
	if len(raw) == 0 {
		return pluginParameters{}, fmt.Errorf("parameters are required")
	}

	params := pluginParameters{}
	if err := json.Unmarshal(raw, &params); err != nil {
		return pluginParameters{}, fmt.Errorf("parse parameters: %w", err)
	}
	if params.Model == "" {
		return pluginParameters{}, fmt.Errorf("model is required")
	}
	if params.SocketPath == "" {
		params.SocketPath = defaultSocketPath
	}
	if params.Backend == "" {
		params.Backend = defaultBackend
	}
	if params.Backend != defaultBackend && params.Backend != "vllm" {
		return pluginParameters{}, fmt.Errorf("backend must be 'hf' or 'vllm', got %q", params.Backend)
	}
	if params.Timeout.Std() <= 0 {
		params.Timeout = utils.Duration(defaultTimeout)
	}
	if params.InitTimeout.Std() <= 0 {
		params.InitTimeout = utils.Duration(defaultInitTimeout)
	}
	if params.InitRetries != nil && *params.InitRetries < 0 {
		return pluginParameters{}, fmt.Errorf("initRetries must be >= 0, got %d", *params.InitRetries)
	}
	if params.InitRetries == nil {
		retries := defaultInitRetries
		params.InitRetries = &retries
	}
	if params.MaxMessageSize <= 0 {
		params.MaxMessageSize = defaultMaxMessageSize
	}

	return params, nil
}

// TypedName returns the plugin identity.
func (p *Plugin) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// Produces declares the tokenized prompt contract while the runtime payload remains on LLMRequest.
func (p *Plugin) Produces() map[string]any {
	return map[string]any{
		tokenizedpromptattr.TokenizedPromptKey: tokenizedpromptattr.Info{},
	}
}

// Consumes declares no framework-managed upstream keys.
func (p *Plugin) Consumes() map[string]any {
	return map[string]any{}
}

// PrepareRequestData tokenizes supported request shapes and fails open on runtime errors.
func (p *Plugin) PrepareRequestData(
	ctx context.Context,
	request *fwkscheduling.LLMRequest,
	_ []fwkscheduling.Endpoint,
) error {
	if ctx == nil {
		ctx = context.Background()
	}

	logger := log.FromContext(ctx).WithName("Plugin.PrepareRequestData")
	start := time.Now()
	if request == nil || request.Body == nil {
		logger.V(logutil.DEBUG).Info(
			"Tokenizer skipped tokenization",
			"reason", "missing_request_body",
			"durationMs", utils.DurationMillisecondsSince(start),
		)
		return nil
	}
	if request.TokenizedPrompt != nil {
		logger.V(logutil.DEBUG).Info(
			"Tokenizer skipped tokenization",
			"reason", "already_tokenized",
			"durationMs", utils.DurationMillisecondsSince(start),
		)
		return nil
	}

	tokenizer, state, err := p.snapshotInitState()
	if state != initStateReady || tokenizer == nil {
		tokenizer, state, err = p.waitForReadyTokenizer(ctx)
	}
	if state != initStateReady || tokenizer == nil {
		logger.Error(
			unavailableError(state, err),
			"Tokenizer unavailable after bounded readiness wait; continuing without tokenized prompt",
			"requestId", request.RequestId,
			"state", state,
			"requestKind", requestKind(request),
			"model", p.requestModel(request),
			"backend", p.backend,
			"durationMs", utils.DurationMillisecondsSince(start),
		)
		return nil
	}

	result, err := p.prepareTokenization(ctx, tokenizer, request)
	if err != nil {
		logger.Error(
			err,
			"Tokenizer request tokenization failed; continuing without tokenized prompt",
			"requestId", request.RequestId,
			"requestKind", requestKind(request),
			"model", p.requestModel(request),
			"backend", p.backend,
			"durationMs", utils.DurationMillisecondsSince(start),
		)
		return nil
	}

	request.TokenizedPrompt = &fwkscheduling.TokenizedPrompt{
		TokenIDs:           result.TokenIDs,
		MultiModalFeatures: result.MultiModalFeatures,
	}
	logger.V(logutil.DEBUG).Info(
		"Tokenizer generated tokenized prompt",
		"model", p.requestModel(request),
		"tokenCount", len(result.TokenIDs),
		"multimodalFeatureCount", len(result.MultiModalFeatures),
		"durationMs", utils.DurationMillisecondsSince(start),
	)
	return nil
}

func (p *Plugin) snapshotInitState() (Backend, initState, error) {
	if p == nil {
		return nil, initStateFailed, fmt.Errorf("tokenizer plugin is nil")
	}

	p.initMu.RLock()
	defer p.initMu.RUnlock()
	return p.tokenizer, p.state, p.lastInitErr
}

func waitWithContext(ctx context.Context, delay time.Duration) error {
	if ctx == nil {
		ctx = context.Background()
	}

	timer := time.NewTimer(delay)
	defer timer.Stop()

	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func isPermanentInitError(err error) bool {
	if errors.Is(err, errModelMismatch) || errors.Is(err, errProviderUnspecified) {
		return true
	}
	// grpc-go >= v1.44: FromError unwraps via errors.As, so the %w wrapping
	// in GRPCTokenizer.Initialize is transparent here.
	if st, ok := status.FromError(err); ok {
		switch st.Code() {
		case codes.InvalidArgument, codes.Unimplemented:
			return true
		default:
			return false
		}
	}
	return false
}

func (p *Plugin) waitForReadyTokenizer(ctx context.Context) (Backend, initState, error) {
	for attempt := 0; attempt < prepareRequestWaitAttempts; attempt++ {
		tokenizer, state, err := p.snapshotInitState()
		if state == initStateReady && tokenizer != nil {
			return tokenizer, state, nil
		}
		if attempt == prepareRequestWaitAttempts-1 {
			return nil, state, err
		}
		if err := waitWithContext(ctx, prepareRequestWaitInterval); err != nil {
			return nil, state, err
		}
	}

	return nil, initStateFailed, nil
}

func unavailableError(state initState, err error) error {
	if err != nil {
		return err
	}
	return fmt.Errorf("tokenizer state %s", state)
}

func requestKind(request *fwkscheduling.LLMRequest) string {
	if request != nil && request.Body != nil && request.Body.ChatCompletions != nil {
		return "chat"
	}
	return "completion"
}

func (p *Plugin) initLogger() logr.Logger {
	if p == nil || p.initCtx == nil {
		return logr.Discard()
	}
	return log.FromContext(p.initCtx).WithName("Plugin.Init").WithValues(
		"plugin", p.typedName.Name,
		"model", p.model,
		"backend", p.backend,
		"socketPath", p.initCfg.SocketPath,
	)
}

func (p *Plugin) storeReadyTokenizer(tokenizer Backend) {
	p.initMu.Lock()
	defer p.initMu.Unlock()
	p.tokenizer = tokenizer
	p.state = initStateReady
	p.lastInitErr = nil
}

func (p *Plugin) storeInitFailure(err error) {
	p.initMu.Lock()
	defer p.initMu.Unlock()
	p.tokenizer = nil
	p.state = initStateFailed
	p.lastInitErr = err
}

func (p *Plugin) storeInitializing() {
	p.initMu.Lock()
	defer p.initMu.Unlock()
	p.state = initStateInitializing
}

func (p *Plugin) runInitLoop() {
	if p == nil {
		return
	}

	logger := p.initLogger()
	socketWait := p.socketWaitTimeout
	if socketWait <= 0 {
		socketWait = defaultSocketWaitTimeout
	}
	totalAttempts := p.initRetries + 1
	if totalAttempts < 1 {
		totalAttempts = 1
	}

	var lastErr error
	for attempt := 1; attempt <= totalAttempts; attempt++ {
		logger.Info("Tokenizer initialization attempt started",
			"attempt", attempt, "initTimeout", p.initCfg.InitTimeout)
		if attempt > 1 {
			if err := p.sleep(p.initCtx, initRetryInterval); err != nil {
				return
			}
		}

		tokenizer, err := p.runInitAttempt(socketWait)
		if err == nil {
			p.storeReadyTokenizer(tokenizer)
			logger.Info("Tokenizer initialization succeeded", "attempt", attempt)
			return
		}

		if p.initCtx.Err() != nil {
			return
		}

		lastErr = err
		p.storeInitFailure(err)
		if isPermanentInitError(err) {
			logger.Error(err, "Tokenizer initialization failed; giving up: permanent error", "attempt", attempt)
			return
		}
		if attempt < totalAttempts {
			p.storeInitializing()
			logger.Error(err, "Tokenizer initialization failed; retrying",
				"attempt", attempt, "initTimeout", p.initCfg.InitTimeout)
		}
	}

	logger.Error(lastErr, "Tokenizer initialization failed; giving up after init retries",
		"attempts", totalAttempts)
}

func (p *Plugin) runInitAttempt(socketWait time.Duration) (Backend, error) {
	if err := p.waitSocketReady(socketWait); err != nil {
		return nil, err
	}

	attemptCtx, cancel := context.WithTimeout(p.initCtx, p.initCfg.InitTimeout)
	defer cancel()
	return p.newTokenizer(attemptCtx, p.initCfg)
}

// Socket-wait timeout is retryable: the sidecar may still be coming up.
func (p *Plugin) waitSocketReady(timeout time.Duration) error {
	probe := p.probeSocket
	if probe == nil {
		probe = probeTokenizerSocket
	}

	logger := p.initLogger()
	logger.Info("Waiting for tokenizer socket", "socketPath", p.initCfg.SocketPath)
	waitStart := time.Now()
	deadline := waitStart.Add(timeout)
	for {
		if err := probe(p.initCtx, p.initCfg.SocketPath); err == nil {
			logger.Info("Tokenizer socket ready", "waitDuration", time.Since(waitStart))
			return nil
		}
		if ctxErr := p.initCtx.Err(); ctxErr != nil {
			return ctxErr
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("tokenizer socket %s not ready within %s", p.initCfg.SocketPath, timeout)
		}
		if err := p.sleep(p.initCtx, socketProbeInterval); err != nil {
			return err
		}
	}
}

func (p *Plugin) registerShutdownClose() {
	if p == nil || p.initCtx == nil {
		return
	}

	go func() {
		<-p.initCtx.Done()
		tokenizer, _, err := p.snapshotInitState()
		if err != nil {
			p.initLogger().Error(err, "Tokenizer shutdown snapshot failed")
			return
		}
		if tokenizer == nil {
			return
		}
		if err := tokenizer.Close(); err != nil {
			p.initLogger().Error(err, "Tokenizer shutdown close failed")
		}
	}()
}

func (p *Plugin) prepareTokenization(
	ctx context.Context,
	tokenizer Backend,
	request *fwkscheduling.LLMRequest,
) (*TokenizationResult, error) {
	if request.Body.Completions != nil {
		return p.tokenizeCompletion(ctx, tokenizer, request)
	}
	if request.Body.ChatCompletions != nil {
		return p.tokenizeChatCompletion(ctx, tokenizer, request)
	}
	return nil, nil
}

func (p *Plugin) tokenizeCompletion(
	ctx context.Context,
	tokenizer Backend,
	request *fwkscheduling.LLMRequest,
) (*TokenizationResult, error) {
	completions := request.Body.Completions
	if completions == nil || completions.Prompt.IsEmpty() {
		return nil, nil
	}

	if p.backend == defaultBackend && canUseDirectTokenize(completions) {
		prompt := completions.Prompt.PlainText()
		if prompt == "" {
			return nil, nil
		}
		return tokenizer.Tokenize(ctx, Request{
			Model:  p.requestModel(request),
			Prompt: prompt,
		})
	}

	return tokenizer.Complete(ctx, CompleteRequest{
		Model:      p.requestModel(request),
		PromptText: completions.Prompt.PlainText(),
	})
}

func canUseDirectTokenize(request *fwkscheduling.CompletionsRequest) bool {
	return request != nil && request.Prompt.Raw != ""
}

func (p *Plugin) tokenizeChatCompletion(
	ctx context.Context,
	tokenizer Backend,
	request *fwkscheduling.LLMRequest,
) (*TokenizationResult, error) {
	chat := request.Body.ChatCompletions
	if chat == nil || len(chat.Messages) == 0 {
		return nil, nil
	}

	return tokenizer.ChatComplete(ctx, ChatCompleteRequest{
		Model:                  p.requestModel(request),
		Messages:               convertMessages(chat.Messages),
		ChatTemplate:           chat.ChatTemplate,
		ChatTemplateKWargsJSON: marshalJSONString(chat.ChatTemplateKWArgs),
		ToolsJSON:              marshalJSONString(chat.Tools),
		AddGenerationPrompt:    chat.AddGenerationPrompt,
		ContinueFinalMessage:   chat.ContinueFinalMessage,
	})
}

func (p *Plugin) requestModel(request *fwkscheduling.LLMRequest) string {
	if request != nil && request.TargetModel != "" {
		return request.TargetModel
	}
	return p.model
}

func convertMessages(messages []fwkscheduling.Message) []ChatMessage {
	converted := make([]ChatMessage, 0, len(messages))
	for _, message := range messages {
		converted = append(converted, ChatMessage{
			Role:         message.Role,
			Content:      message.Content.Raw,
			ContentParts: convertContentParts(message.Content.Structured),
		})
	}
	return converted
}

func convertContentParts(blocks []fwkscheduling.ContentBlock) []ChatContentPart {
	parts := make([]ChatContentPart, 0, len(blocks))
	for _, block := range blocks {
		switch block.Type {
		case "text":
			parts = append(parts, ChatContentPart{Text: block.Text})
		case "image_url":
			parts = append(parts, ChatContentPart{Media: &MediaRef{
				Modality: string(fwkscheduling.ModalityImage),
				URL:      block.ImageURL.Url,
			}})
		default:
			continue
		}
	}
	return parts
}

func marshalJSONString(value any) string {
	if value == nil {
		return ""
	}
	b, err := json.Marshal(value)
	if err != nil {
		return ""
	}
	return string(b)
}
