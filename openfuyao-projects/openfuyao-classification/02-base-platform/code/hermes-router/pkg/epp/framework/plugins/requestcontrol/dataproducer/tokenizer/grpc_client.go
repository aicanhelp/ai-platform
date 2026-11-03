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
	"errors"
	"fmt"
	"net"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	pb "hermes-router/api/tokenizer/v1"
)

// GRPCTokenizerConfig configures the UDS gRPC tokenizer client.
type GRPCTokenizerConfig struct {
	SocketPath    string
	Model         string
	TokenizerName string
	Timeout       time.Duration
	// Whole Initialize RPC, including sidecar resolution and chat warmup.
	InitTimeout    time.Duration
	MaxMessageSize int
}

// Client-side Initialize contract violations. They carry no gRPC status code,
// so isPermanentInitError recognizes them via errors.Is instead.
var (
	errModelMismatch       = errors.New("initialize tokenizer: model mismatch")
	errProviderUnspecified = errors.New("initialize tokenizer: resolved provider is unspecified")
)

// GRPCTokenizer implements Backend over the protobuf gRPC service.
type GRPCTokenizer struct {
	conn          *grpc.ClientConn
	client        pb.TokenizationServiceClient
	timeout       time.Duration
	initTimeout   time.Duration
	model         string
	tokenizerName string
}

var _ Backend = (*GRPCTokenizer)(nil)

// NewGRPCTokenizer connects to the tokenizer sidecar and validates initialization.
func NewGRPCTokenizer(ctx context.Context, cfg GRPCTokenizerConfig) (*GRPCTokenizer, error) {
	if cfg.SocketPath == "" {
		return nil, fmt.Errorf("socketPath is required")
	}
	if cfg.Model == "" {
		return nil, fmt.Errorf("model is required")
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = defaultTimeout
	}
	if cfg.MaxMessageSize <= 0 {
		cfg.MaxMessageSize = defaultMaxMessageSize
	}
	if ctx == nil {
		ctx = context.Background()
	}

	return tryConnectAndInitialize(ctx, cfg)
}

func tryConnectAndInitialize(ctx context.Context, cfg GRPCTokenizerConfig) (*GRPCTokenizer, error) {
	conn, err := newTokenizerConnection(cfg)
	if err != nil {
		return nil, err
	}

	tokenizer := newGRPCTokenizer(conn, cfg)
	initReq := InitializeRequest{Model: cfg.Model, TokenizerName: cfg.TokenizerName}
	if err := tokenizer.Initialize(ctx, initReq); err != nil {
		_ = conn.Close()
		return nil, err
	}

	return tokenizer, nil
}

func probeTokenizerSocket(ctx context.Context, socketPath string) error {
	var dialer net.Dialer
	conn, err := dialer.DialContext(ctx, "unix", socketPath)
	if err != nil {
		return fmt.Errorf("dial tokenizer grpc server: %w", err)
	}
	return conn.Close()
}

func newTokenizerConnection(cfg GRPCTokenizerConfig) (*grpc.ClientConn, error) {
	conn, err := grpc.NewClient(
		"unix://"+cfg.SocketPath,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) {
			var dialer net.Dialer
			return dialer.DialContext(ctx, "unix", cfg.SocketPath)
		}),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(cfg.MaxMessageSize),
			grpc.MaxCallSendMsgSize(cfg.MaxMessageSize),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("create tokenizer grpc client: %w", err)
	}
	return conn, nil
}

func newGRPCTokenizer(conn *grpc.ClientConn, cfg GRPCTokenizerConfig) *GRPCTokenizer {
	initTimeout := cfg.InitTimeout
	if initTimeout <= 0 {
		initTimeout = defaultInitTimeout
	}
	return &GRPCTokenizer{
		conn:          conn,
		client:        pb.NewTokenizationServiceClient(conn),
		timeout:       cfg.Timeout,
		initTimeout:   initTimeout,
		model:         cfg.Model,
		tokenizerName: cfg.TokenizerName,
	}
}

// Close releases the underlying gRPC connection.
func (c *GRPCTokenizer) Close() error {
	if c == nil || c.conn == nil {
		return nil
	}
	return c.conn.Close()
}

// Initialize issues an Initialize RPC and validates the response.
func (c *GRPCTokenizer) Initialize(ctx context.Context, request InitializeRequest) error {
	callCtx, cancel := c.withTimeout(ctx, c.initTimeout)
	defer cancel()

	pbReq := &pb.InitializeRequest{
		Model: request.Model,
	}
	if request.TokenizerName != "" {
		pbReq.TokenizerName = &request.TokenizerName
	}

	response, err := c.client.Initialize(callCtx, pbReq)
	if err != nil {
		return fmt.Errorf("initialize tokenizer: %w", err)
	}
	if response.GetModel() != request.Model {
		return fmt.Errorf("%w: got %q want %q", errModelMismatch, response.GetModel(), request.Model)
	}
	if response.GetResolvedProvider() == pb.TokenizerProvider_TOKENIZER_PROVIDER_UNSPECIFIED {
		return errProviderUnspecified
	}
	return nil
}

// Tokenize sends a plain-text tokenization request.
func (c *GRPCTokenizer) Tokenize(ctx context.Context, request Request) (*TokenizationResult, error) {
	callCtx, cancel := c.withTimeout(ctx, c.timeout)
	defer cancel()

	response, err := c.client.Tokenize(callCtx, &pb.TokenizeRequest{
		Model:            firstNonEmpty(request.Model, c.model),
		Prompt:           request.Prompt,
		AddSpecialTokens: request.AddSpecialTokens,
	})
	if err != nil {
		return nil, fmt.Errorf("tokenize prompt: %w", err)
	}
	return convertPBResult(response.GetResult()), nil
}

// Complete sends a completion-render tokenization request.
func (c *GRPCTokenizer) Complete(ctx context.Context, request CompleteRequest) (*TokenizationResult, error) {
	callCtx, cancel := c.withTimeout(ctx, c.timeout)
	defer cancel()

	pbReq := &pb.RenderCompletionRequest{
		Model:                firstNonEmpty(request.Model, c.model),
		AddSpecialTokens:     request.AddSpecialTokens,
		TruncatePromptTokens: request.TruncatePromptTokens,
	}
	applyRenderCompletionPromptSource(pbReq, request)

	response, err := c.client.RenderCompletion(callCtx, pbReq)
	if err != nil {
		return nil, fmt.Errorf("render completion: %w", err)
	}
	return convertPBResult(response.GetResult()), nil
}

// ChatComplete sends a chat-render tokenization request.
func (c *GRPCTokenizer) ChatComplete(ctx context.Context, request ChatCompleteRequest) (*TokenizationResult, error) {
	callCtx, cancel := c.withTimeout(ctx, c.timeout)
	defer cancel()

	response, err := c.client.RenderChatCompletion(callCtx, &pb.RenderChatCompletionRequest{
		Model:                  firstNonEmpty(request.Model, c.model),
		Messages:               convertMessagesToProto(request.Messages),
		ChatTemplate:           request.ChatTemplate,
		ChatTemplateKwargsJson: request.ChatTemplateKWargsJSON,
		ToolsJson:              request.ToolsJSON,
		ToolChoiceJson:         request.ToolChoiceJSON,
		AddGenerationPrompt:    request.AddGenerationPrompt,
		ContinueFinalMessage:   request.ContinueFinalMessage,
		MmProcessorKwargsJson:  request.MMProcessorKWargsJSON,
		MediaIoKwargsJson:      request.MediaIOKWargsJSON,
	})
	if err != nil {
		return nil, fmt.Errorf("render chat completion: %w", err)
	}
	return convertPBResult(response.GetResult()), nil
}

func (c *GRPCTokenizer) withTimeout(ctx context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithTimeout(ctx, timeout)
}

func convertPBResult(result *pb.TokenizationResult) *TokenizationResult {
	if result == nil {
		return &TokenizationResult{}
	}

	converted := &TokenizationResult{TokenIDs: append([]uint32(nil), result.GetTokenIds()...)}
	if len(result.GetMultimodalFeatures()) == 0 {
		return converted
	}

	converted.MultiModalFeatures = make([]fwkscheduling.MultiModalFeature, 0, len(result.GetMultimodalFeatures()))
	for _, feature := range result.GetMultimodalFeatures() {
		if feature == nil {
			continue
		}
		converted.MultiModalFeatures = append(converted.MultiModalFeatures, fwkscheduling.MultiModalFeature{
			Modality: convertPBModality(feature.GetModality()),
			Hash:     feature.GetHash(),
			Offset:   int(feature.GetOffset()),
			Length:   int(feature.GetLength()),
		})
	}
	return converted
}

func convertPBModality(modality pb.Modality) fwkscheduling.Modality {
	switch modality {
	case pb.Modality_MODALITY_IMAGE:
		return fwkscheduling.ModalityImage
	default:
		return fwkscheduling.Modality("")
	}
}

func convertMessagesToProto(messages []ChatMessage) []*pb.ChatMessage {
	converted := make([]*pb.ChatMessage, 0, len(messages))
	for _, message := range messages {
		pbMessage := &pb.ChatMessage{
			Role:          message.Role,
			Name:          message.Name,
			ToolCallsJson: message.ToolCallsJSON,
			ToolCallId:    message.ToolCallID,
		}
		applyChatMessageContentForm(pbMessage, message)
		converted = append(converted, pbMessage)
	}
	return converted
}

func convertPartsToProto(parts []ChatContentPart) []*pb.ChatContentPart {
	converted := make([]*pb.ChatContentPart, 0, len(parts))
	for _, part := range parts {
		pbPart := &pb.ChatContentPart{}
		applyChatContentPartValue(pbPart, part)
		converted = append(converted, pbPart)
	}
	return converted
}

func applyRenderCompletionPromptSource(target *pb.RenderCompletionRequest, request CompleteRequest) {
	if len(request.PromptTokenIDs) > 0 {
		target.PromptSource = &pb.RenderCompletionRequest_PromptTokenIds{
			PromptTokenIds: &pb.TokenIdList{Values: append([]uint32(nil), request.PromptTokenIDs...)},
		}
		return
	}
	target.PromptSource = &pb.RenderCompletionRequest_PromptText{PromptText: request.PromptText}
}

func applyChatMessageContentForm(target *pb.ChatMessage, message ChatMessage) {
	if len(message.ContentParts) > 0 {
		target.ContentForm = &pb.ChatMessage_ContentParts{
			ContentParts: &pb.ChatContentParts{Values: convertPartsToProto(message.ContentParts)},
		}
		return
	}
	target.ContentForm = &pb.ChatMessage_Content{Content: message.Content}
}

func applyChatContentPartValue(target *pb.ChatContentPart, part ChatContentPart) {
	if part.Media != nil {
		target.Part = &pb.ChatContentPart_Media{Media: convertMediaToProto(part.Media)}
		return
	}
	target.Part = &pb.ChatContentPart_Text{Text: part.Text}
}

func convertMediaToProto(media *MediaRef) *pb.MediaRef {
	if media == nil {
		return nil
	}
	return &pb.MediaRef{
		Modality:   convertModalityToProto(media.Modality),
		Url:        media.URL,
		InlineData: media.InlineData,
		MimeType:   media.MIMEType,
		DetailJson: media.DetailJSON,
	}
}

func convertModalityToProto(modality string) pb.Modality {
	switch strings.ToLower(strings.TrimPrefix(modality, "MODALITY_")) {
	case "image":
		return pb.Modality_MODALITY_IMAGE
	default:
		if modality == pb.Modality_MODALITY_IMAGE.String() {
			return pb.Modality_MODALITY_IMAGE
		}
		return pb.Modality_MODALITY_UNSPECIFIED
	}
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}
