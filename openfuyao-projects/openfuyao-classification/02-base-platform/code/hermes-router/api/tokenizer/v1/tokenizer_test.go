/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *         http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package tokenizerv1

import (
	"context"
	"net"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/proto"
)

func TestEnumValues(t *testing.T) {
	providers := map[TokenizerProvider]string{
		TokenizerProvider_TOKENIZER_PROVIDER_UNSPECIFIED: "TOKENIZER_PROVIDER_UNSPECIFIED",
		TokenizerProvider_TOKENIZER_PROVIDER_AUTO:        "TOKENIZER_PROVIDER_AUTO",
		TokenizerProvider_TOKENIZER_PROVIDER_HUGGINGFACE: "TOKENIZER_PROVIDER_HUGGINGFACE",
		TokenizerProvider_TOKENIZER_PROVIDER_MODELSCOPE:  "TOKENIZER_PROVIDER_MODELSCOPE",
	}
	for value, name := range providers {
		if got := value.String(); got != name {
			t.Fatalf("expected enum name %q, got %q", name, got)
		}
		if got := value.Number(); int32(got) != int32(value) {
			t.Fatalf("unexpected enum number for %s: %v", name, got)
		}
		if got := value.Enum(); *got != value {
			t.Fatalf("unexpected enum pointer value for %s", name)
		}
		if value.Type() == nil {
			t.Fatalf("expected non-nil enum type for %s", name)
		}
		if value.Descriptor() == nil {
			t.Fatalf("expected non-nil enum descriptor for %s", name)
		}
		if file, _ := value.EnumDescriptor(); file == nil {
			t.Fatalf("expected non-nil legacy enum descriptor for %s", name)
		}
	}

	modalities := map[Modality]string{
		Modality_MODALITY_UNSPECIFIED: "MODALITY_UNSPECIFIED",
		Modality_MODALITY_IMAGE:       "MODALITY_IMAGE",
	}
	for value, name := range modalities {
		if got := value.String(); got != name {
			t.Fatalf("expected enum name %q, got %q", name, got)
		}
		if got := value.Number(); int32(got) != int32(value) {
			t.Fatalf("unexpected enum number for %s: %v", name, got)
		}
		if got := value.Enum(); *got != value {
			t.Fatalf("unexpected enum pointer value for %s", name)
		}
		if value.Type() == nil {
			t.Fatalf("expected non-nil enum type for %s", name)
		}
		if value.Descriptor() == nil {
			t.Fatalf("expected non-nil enum descriptor for %s", name)
		}
		if file, _ := value.EnumDescriptor(); file == nil {
			t.Fatalf("expected non-nil legacy enum descriptor for %s", name)
		}
	}
}

func newMultiModalFeature() *MultiModalFeature {
	return &MultiModalFeature{
		Modality: Modality_MODALITY_IMAGE,
		Hash:     "hash-1",
		Offset:   2,
		Length:   3,
	}
}

func newTokenizationResult() *TokenizationResult {
	return &TokenizationResult{
		TokenIds:           []uint32{1, 2, 3},
		MultimodalFeatures: []*MultiModalFeature{newMultiModalFeature()},
	}
}

func newRenderChatCompletionRequest() *RenderChatCompletionRequest {
	return &RenderChatCompletionRequest{
		Model:                  "chat-model",
		Messages:               []*ChatMessage{newTextChatMessage()},
		ChatTemplate:           "template",
		ChatTemplateKwargsJson: `{"k":"v"}`,
		ToolsJson:              "[]",
		ToolChoiceJson:         "auto",
		AddGenerationPrompt:    true,
		ContinueFinalMessage:   false,
		MmProcessorKwargsJson:  `{"mm":"1"}`,
		MediaIoKwargsJson:      `{"io":"1"}`,
	}
}

func newTextChatMessage() *ChatMessage {
	return &ChatMessage{
		Role:          "user",
		ContentForm:   &ChatMessage_Content{Content: "hello"},
		Name:          "alice",
		ToolCallsJson: "[]",
		ToolCallId:    "call-1",
	}
}

func newPartsChatMessage() *ChatMessage {
	return &ChatMessage{
		Role: "user",
		ContentForm: &ChatMessage_ContentParts{ContentParts: &ChatContentParts{Values: []*ChatContentPart{{
			Part: &ChatContentPart_Text{Text: "describe"},
		}, {
			Part: &ChatContentPart_Media{Media: &MediaRef{
				Modality:   Modality_MODALITY_IMAGE,
				Url:        "https://example.test/image.png",
				InlineData: []byte("image-bytes"),
				MimeType:   "image/png",
				DetailJson: `{"scale":2}`,
			}},
		}}}},
	}
}

func TestSimpleMessageAccessors(t *testing.T) {
	feature := newMultiModalFeature()
	if got := feature.GetModality(); got != Modality_MODALITY_IMAGE {
		t.Fatalf("unexpected modality: %v", got)
	}
	if got := feature.GetHash(); got != "hash-1" {
		t.Fatalf("unexpected hash: %q", got)
	}
	if got := feature.GetOffset(); got != 2 {
		t.Fatalf("unexpected offset: %v", got)
	}
	if got := feature.GetLength(); got != 3 {
		t.Fatalf("unexpected length: %v", got)
	}

	result := newTokenizationResult()
	if got := result.GetTokenIds(); len(got) != 3 {
		t.Fatalf("unexpected token ids: %v", got)
	}
	if got := result.GetMultimodalFeatures(); len(got) != 1 {
		t.Fatalf("unexpected multimodal features: %v", got)
	}

	initializeReq := &InitializeRequest{Model: "m-1", TokenizerName: proto.String("tn-1")}
	if got := initializeReq.GetModel(); got != "m-1" {
		t.Fatalf("unexpected model: %q", got)
	}
	if got := initializeReq.GetTokenizerName(); got != "tn-1" {
		t.Fatalf("unexpected tokenizer name: %q", got)
	}

	initializeResp := &InitializeResponse{
		Model:                 "m-1",
		ResolvedProvider:      TokenizerProvider_TOKENIZER_PROVIDER_HUGGINGFACE,
		ResolvedTokenizerName: "tn-1",
	}
	if got := initializeResp.GetModel(); got != "m-1" {
		t.Fatalf("unexpected model: %q", got)
	}
	if got := initializeResp.GetResolvedProvider(); got != TokenizerProvider_TOKENIZER_PROVIDER_HUGGINGFACE {
		t.Fatalf("unexpected provider: %v", got)
	}
	if got := initializeResp.GetResolvedTokenizerName(); got != "tn-1" {
		t.Fatalf("unexpected resolved tokenizer name: %q", got)
	}

	tokenizeReq := &TokenizeRequest{Model: "m-1", Prompt: "prompt", AddSpecialTokens: true}
	if got := tokenizeReq.GetModel(); got != "m-1" {
		t.Fatalf("unexpected model: %q", got)
	}
	if got := tokenizeReq.GetPrompt(); got != "prompt" {
		t.Fatalf("unexpected prompt: %q", got)
	}
	if !tokenizeReq.GetAddSpecialTokens() {
		t.Fatal("expected AddSpecialTokens to be true")
	}

	tokenizeResp := &TokenizeResponse{Result: newTokenizationResult()}
	if got := tokenizeResp.GetResult().GetTokenIds(); len(got) != 3 {
		t.Fatalf("unexpected token ids: %v", got)
	}

	idList := &TokenIdList{Values: []uint32{7, 8}}
	if got := idList.GetValues(); len(got) != 2 {
		t.Fatalf("unexpected values: %v", got)
	}
}

func TestRenderCompletionRequestAccessors(t *testing.T) {
	textRequest := &RenderCompletionRequest{
		Model:                "m-1",
		PromptSource:         &RenderCompletionRequest_PromptText{PromptText: "prompt text"},
		AddSpecialTokens:     true,
		TruncatePromptTokens: proto.Int32(16),
	}
	if _, ok := textRequest.GetPromptSource().(*RenderCompletionRequest_PromptText); !ok {
		t.Fatalf("unexpected prompt source type: %T", textRequest.GetPromptSource())
	}
	if got := textRequest.GetPromptText(); got != "prompt text" {
		t.Fatalf("unexpected prompt text: %q", got)
	}
	if textRequest.GetPromptTokenIds() != nil {
		t.Fatalf("expected nil prompt token ids, got %v", textRequest.GetPromptTokenIds())
	}
	if !textRequest.GetAddSpecialTokens() {
		t.Fatal("expected AddSpecialTokens to be true")
	}
	if got := textRequest.GetTruncatePromptTokens(); got != 16 {
		t.Fatalf("unexpected truncate prompt tokens: %v", got)
	}
	if got := textRequest.GetModel(); got != "m-1" {
		t.Fatalf("unexpected model: %q", got)
	}

	idsRequest := &RenderCompletionRequest{
		PromptSource: &RenderCompletionRequest_PromptTokenIds{PromptTokenIds: &TokenIdList{Values: []uint32{1}}},
	}
	if _, ok := idsRequest.GetPromptSource().(*RenderCompletionRequest_PromptTokenIds); !ok {
		t.Fatalf("unexpected prompt source type: %T", idsRequest.GetPromptSource())
	}
	if got := idsRequest.GetPromptTokenIds().GetValues(); len(got) != 1 {
		t.Fatalf("unexpected prompt token ids: %v", got)
	}
	if idsRequest.GetPromptText() != "" {
		t.Fatalf("expected empty prompt text, got %q", idsRequest.GetPromptText())
	}
	if got := idsRequest.GetTruncatePromptTokens(); got != 0 {
		t.Fatalf("expected zero truncate prompt tokens, got %v", got)
	}
}

func TestChatMessageAccessors(t *testing.T) {
	textMessage := newTextChatMessage()
	if got := textMessage.GetRole(); got != "user" {
		t.Fatalf("unexpected role: %q", got)
	}
	if _, ok := textMessage.GetContentForm().(*ChatMessage_Content); !ok {
		t.Fatalf("unexpected content form type: %T", textMessage.GetContentForm())
	}
	if got := textMessage.GetContent(); got != "hello" {
		t.Fatalf("unexpected content: %q", got)
	}
	if textMessage.GetContentParts() != nil {
		t.Fatalf("expected nil content parts, got %v", textMessage.GetContentParts())
	}
	if got := textMessage.GetName(); got != "alice" {
		t.Fatalf("unexpected name: %q", got)
	}
	if got := textMessage.GetToolCallsJson(); got != "[]" {
		t.Fatalf("unexpected tool calls json: %q", got)
	}
	if got := textMessage.GetToolCallId(); got != "call-1" {
		t.Fatalf("unexpected tool call id: %q", got)
	}

	partsMessage := newPartsChatMessage()
	if _, ok := partsMessage.GetContentForm().(*ChatMessage_ContentParts); !ok {
		t.Fatalf("unexpected content form type: %T", partsMessage.GetContentForm())
	}
	if partsMessage.GetContent() != "" {
		t.Fatalf("expected empty content, got %q", partsMessage.GetContent())
	}
	parts := partsMessage.GetContentParts().GetValues()
	if len(parts) != 2 {
		t.Fatalf("expected 2 content parts, got %d", len(parts))
	}
	if _, ok := parts[0].GetPart().(*ChatContentPart_Text); !ok {
		t.Fatalf("unexpected part type: %T", parts[0].GetPart())
	}
	if got := parts[0].GetText(); got != "describe" {
		t.Fatalf("unexpected text: %q", got)
	}
	if parts[0].GetMedia() != nil {
		t.Fatalf("expected nil media, got %v", parts[0].GetMedia())
	}
	mediaPart, ok := parts[1].GetPart().(*ChatContentPart_Media)
	if !ok {
		t.Fatalf("unexpected part type: %T", parts[1].GetPart())
	}
	if got := mediaPart.Media.GetUrl(); got != "https://example.test/image.png" {
		t.Fatalf("unexpected media url: %q", got)
	}
	if got := mediaPart.Media.GetModality(); got != Modality_MODALITY_IMAGE {
		t.Fatalf("unexpected media modality: %v", got)
	}
	if got := mediaPart.Media.GetInlineData(); string(got) != "image-bytes" {
		t.Fatalf("unexpected inline data: %q", got)
	}
	if got := mediaPart.Media.GetMimeType(); got != "image/png" {
		t.Fatalf("unexpected mime type: %q", got)
	}
	if got := mediaPart.Media.GetDetailJson(); got != `{"scale":2}` {
		t.Fatalf("unexpected detail json: %q", got)
	}
	if parts[1].GetText() != "" {
		t.Fatalf("expected empty text on media part, got %q", parts[1].GetText())
	}
}

func TestRenderChatCompletionRequestAccessors(t *testing.T) {
	request := newRenderChatCompletionRequest()
	if got := request.GetModel(); got != "chat-model" {
		t.Fatalf("unexpected model: %q", got)
	}
	if got := len(request.GetMessages()); got != 1 {
		t.Fatalf("expected 1 message, got %d", got)
	}
	if got := request.GetChatTemplate(); got != "template" {
		t.Fatalf("unexpected chat template: %q", got)
	}
	if got := request.GetChatTemplateKwargsJson(); got != `{"k":"v"}` {
		t.Fatalf("unexpected chat template kwargs json: %q", got)
	}
	if got := request.GetToolsJson(); got != "[]" {
		t.Fatalf("unexpected tools json: %q", got)
	}
	if got := request.GetToolChoiceJson(); got != "auto" {
		t.Fatalf("unexpected tool choice json: %q", got)
	}
	if !request.GetAddGenerationPrompt() {
		t.Fatal("expected AddGenerationPrompt to be true")
	}
	if request.GetContinueFinalMessage() {
		t.Fatal("expected ContinueFinalMessage to be false")
	}
	if got := request.GetMmProcessorKwargsJson(); got != `{"mm":"1"}` {
		t.Fatalf("unexpected mm processor kwargs json: %q", got)
	}
	if got := request.GetMediaIoKwargsJson(); got != `{"io":"1"}` {
		t.Fatalf("unexpected media io kwargs json: %q", got)
	}

	response := &RenderChatCompletionResponse{Result: newTokenizationResult()}
	if got := response.GetResult().GetTokenIds(); len(got) != 3 {
		t.Fatalf("unexpected token ids: %v", got)
	}
}

func TestNilReceiverAccessorsReturnZeroValues(t *testing.T) {
	var feature *MultiModalFeature
	if feature.GetModality() != Modality_MODALITY_UNSPECIFIED || feature.GetHash() != "" ||
		feature.GetOffset() != 0 || feature.GetLength() != 0 {
		t.Fatal("expected zero values for nil MultiModalFeature")
	}

	var result *TokenizationResult
	if result.GetTokenIds() != nil || result.GetMultimodalFeatures() != nil {
		t.Fatal("expected nil slices for nil TokenizationResult")
	}

	var initializeRequest *InitializeRequest
	if initializeRequest.GetModel() != "" || initializeRequest.GetTokenizerName() != "" {
		t.Fatal("expected empty strings for nil InitializeRequest")
	}

	var initializeResponse *InitializeResponse
	if initializeResponse.GetModel() != "" ||
		initializeResponse.GetResolvedProvider() != TokenizerProvider_TOKENIZER_PROVIDER_UNSPECIFIED ||
		initializeResponse.GetResolvedTokenizerName() != "" {
		t.Fatal("expected zero values for nil InitializeResponse")
	}

	var tokenizeRequest *TokenizeRequest
	if tokenizeRequest.GetModel() != "" || tokenizeRequest.GetPrompt() != "" || tokenizeRequest.GetAddSpecialTokens() {
		t.Fatal("expected zero values for nil TokenizeRequest")
	}

	var tokenizeResponse *TokenizeResponse
	if tokenizeResponse.GetResult() != nil {
		t.Fatal("expected nil result for nil TokenizeResponse")
	}

	var idList *TokenIdList
	if idList.GetValues() != nil {
		t.Fatal("expected nil values for nil TokenIdList")
	}

	var completionRequest *RenderCompletionRequest
	if completionRequest.GetModel() != "" || completionRequest.GetPromptSource() != nil ||
		completionRequest.GetPromptText() != "" || completionRequest.GetPromptTokenIds() != nil ||
		completionRequest.GetAddSpecialTokens() || completionRequest.GetTruncatePromptTokens() != 0 {
		t.Fatal("expected zero values for nil RenderCompletionRequest")
	}

	var completionResponse *RenderCompletionResponse
	if completionResponse.GetResult() != nil {
		t.Fatal("expected nil result for nil RenderCompletionResponse")
	}

	var mediaRef *MediaRef
	if mediaRef.GetModality() != Modality_MODALITY_UNSPECIFIED || mediaRef.GetUrl() != "" ||
		mediaRef.GetInlineData() != nil || mediaRef.GetMimeType() != "" || mediaRef.GetDetailJson() != "" {
		t.Fatal("expected zero values for nil MediaRef")
	}

	var part *ChatContentPart
	if part.GetPart() != nil || part.GetText() != "" || part.GetMedia() != nil {
		t.Fatal("expected zero values for nil ChatContentPart")
	}

	var parts *ChatContentParts
	if parts.GetValues() != nil {
		t.Fatal("expected nil values for nil ChatContentParts")
	}

	var message *ChatMessage
	if message.GetRole() != "" || message.GetContentForm() != nil || message.GetContent() != "" ||
		message.GetContentParts() != nil || message.GetName() != "" ||
		message.GetToolCallsJson() != "" || message.GetToolCallId() != "" {
		t.Fatal("expected zero values for nil ChatMessage")
	}

	var chatRequest *RenderChatCompletionRequest
	if chatRequest.GetModel() != "" || chatRequest.GetMessages() != nil ||
		chatRequest.GetChatTemplate() != "" || chatRequest.GetChatTemplateKwargsJson() != "" ||
		chatRequest.GetToolsJson() != "" || chatRequest.GetToolChoiceJson() != "" ||
		chatRequest.GetAddGenerationPrompt() || chatRequest.GetContinueFinalMessage() ||
		chatRequest.GetMmProcessorKwargsJson() != "" || chatRequest.GetMediaIoKwargsJson() != "" {
		t.Fatal("expected zero values for nil RenderChatCompletionRequest")
	}

	var chatResponse *RenderChatCompletionResponse
	if chatResponse.GetResult() != nil {
		t.Fatal("expected nil result for nil RenderChatCompletionResponse")
	}
}

// descriptorMessage is satisfied by every generated message type in this
// package; it adds the legacy accessors to proto.Message.
type descriptorMessage interface {
	proto.Message
	Reset()
	String() string
	ProtoMessage()
	Descriptor() ([]byte, []int)
}

func TestMessageLifecycleOperations(t *testing.T) {
	messages := []descriptorMessage{
		newMultiModalFeature(),
		newTokenizationResult(),
		&InitializeRequest{Model: "m"},
		&InitializeResponse{Model: "m"},
		&TokenizeRequest{Model: "m", Prompt: "p"},
		&TokenizeResponse{Result: newTokenizationResult()},
		&TokenIdList{Values: []uint32{1}},
		&RenderCompletionRequest{PromptSource: &RenderCompletionRequest_PromptText{PromptText: "p"}},
		&RenderCompletionRequest{
			PromptSource: &RenderCompletionRequest_PromptTokenIds{PromptTokenIds: &TokenIdList{Values: []uint32{1}}},
		},
		&RenderCompletionResponse{Result: newTokenizationResult()},
		&MediaRef{Url: "https://example.test"},
		&ChatContentPart{Part: &ChatContentPart_Text{Text: "t"}},
		&ChatContentParts{Values: []*ChatContentPart{{Part: &ChatContentPart_Text{Text: "t"}}}},
		newTextChatMessage(),
		newPartsChatMessage(),
		newRenderChatCompletionRequest(),
		&RenderChatCompletionResponse{Result: newTokenizationResult()},
	}
	for _, message := range messages {
		message.Reset()
		// prototext of an empty message may be empty; the calls must not panic.
		_ = message.String()
		message.ProtoMessage()
		if message.ProtoReflect() == nil {
			t.Fatalf("expected non-nil ProtoReflect result for %T", message)
		}
		if file, _ := message.Descriptor(); file == nil {
			t.Fatalf("expected non-nil Descriptor result for %T", message)
		}
	}
}

func TestMessageProtoRoundTrips(t *testing.T) {
	roundTrips := []proto.Message{
		newTokenizationResult(),
		&InitializeRequest{Model: "m", TokenizerName: proto.String("tn")},
		&InitializeResponse{
			Model:                 "m",
			ResolvedProvider:      TokenizerProvider_TOKENIZER_PROVIDER_MODELSCOPE,
			ResolvedTokenizerName: "tn",
		},
		&TokenizeRequest{Model: "m", Prompt: "p", AddSpecialTokens: true},
		&TokenizeResponse{Result: newTokenizationResult()},
		&RenderCompletionRequest{
			Model:                "m",
			PromptSource:         &RenderCompletionRequest_PromptText{PromptText: "p"},
			AddSpecialTokens:     true,
			TruncatePromptTokens: proto.Int32(8),
		},
		&RenderCompletionRequest{
			Model:        "m",
			PromptSource: &RenderCompletionRequest_PromptTokenIds{PromptTokenIds: &TokenIdList{Values: []uint32{1, 2}}},
		},
		&MediaRef{
			Modality:   Modality_MODALITY_IMAGE,
			Url:        "https://example.test",
			InlineData: []byte("bytes"),
			MimeType:   "image/png",
			DetailJson: "{}",
		},
		newPartsChatMessage(),
		newRenderChatCompletionRequest(),
	}

	for _, message := range roundTrips {
		encoded, err := proto.Marshal(message)
		if err != nil {
			t.Fatalf("marshal %T: %v", message, err)
		}
		decoded := message.ProtoReflect().New().Interface()
		if err := proto.Unmarshal(encoded, decoded); err != nil {
			t.Fatalf("unmarshal %T: %v", message, err)
		}
		if !proto.Equal(message, decoded) {
			t.Fatalf("round-trip mismatch for %T: %+v != %+v", message, message, decoded)
		}
	}
}

type echoTokenizationService struct {
	UnimplementedTokenizationServiceServer
	initializeRequest *InitializeRequest
	tokenizeRequest   *TokenizeRequest
	completionRequest *RenderCompletionRequest
	chatRequest       *RenderChatCompletionRequest
}

func (s *echoTokenizationService) Initialize(_ context.Context, req *InitializeRequest) (*InitializeResponse, error) {
	s.initializeRequest = req
	return &InitializeResponse{
		Model:            req.GetModel(),
		ResolvedProvider: TokenizerProvider_TOKENIZER_PROVIDER_HUGGINGFACE,
	}, nil
}

func (s *echoTokenizationService) Tokenize(_ context.Context, req *TokenizeRequest) (*TokenizeResponse, error) {
	s.tokenizeRequest = req
	return &TokenizeResponse{Result: &TokenizationResult{TokenIds: []uint32{1, 2, 3}}}, nil
}

func (s *echoTokenizationService) RenderCompletion(_ context.Context, req *RenderCompletionRequest) (*RenderCompletionResponse, error) {
	s.completionRequest = req
	return &RenderCompletionResponse{Result: &TokenizationResult{TokenIds: []uint32{9, 8}}}, nil
}

func (s *echoTokenizationService) RenderChatCompletion(_ context.Context, req *RenderChatCompletionRequest) (*RenderChatCompletionResponse, error) {
	s.chatRequest = req
	return &RenderChatCompletionResponse{Result: &TokenizationResult{TokenIds: []uint32{5, 6, 7}}}, nil
}

func TestTokenizationServiceGRPCRoundTrip(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen tcp: %v", err)
	}
	server := grpc.NewServer()
	service := &echoTokenizationService{}
	RegisterTokenizationServiceServer(server, service)
	go func() { _ = server.Serve(listener) }()
	t.Cleanup(server.Stop)

	conn, err := grpc.NewClient(
		listener.Addr().String(),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		t.Fatalf("create client: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	client := NewTokenizationServiceClient(conn)
	ctx := context.Background()

	initializeResponse, err := client.Initialize(ctx, &InitializeRequest{Model: "grpc-model"})
	if err != nil {
		t.Fatalf("initialize rpc: %v", err)
	}
	if got := initializeResponse.GetModel(); got != "grpc-model" {
		t.Fatalf("unexpected initialize model: %q", got)
	}
	if service.initializeRequest.GetModel() != "grpc-model" {
		t.Fatalf("server received unexpected initialize request: %+v", service.initializeRequest)
	}

	tokenizeResponse, err := client.Tokenize(ctx, &TokenizeRequest{Model: "grpc-model", Prompt: "hello"})
	if err != nil {
		t.Fatalf("tokenize rpc: %v", err)
	}
	if got := tokenizeResponse.GetResult().GetTokenIds(); len(got) != 3 {
		t.Fatalf("unexpected token ids: %v", got)
	}
	if service.tokenizeRequest.GetPrompt() != "hello" {
		t.Fatalf("server received unexpected tokenize request: %+v", service.tokenizeRequest)
	}

	completionResponse, err := client.RenderCompletion(ctx, &RenderCompletionRequest{
		Model:        "grpc-model",
		PromptSource: &RenderCompletionRequest_PromptText{PromptText: "prompt"},
	})
	if err != nil {
		t.Fatalf("render completion rpc: %v", err)
	}
	if got := completionResponse.GetResult().GetTokenIds(); len(got) != 2 {
		t.Fatalf("unexpected token ids: %v", got)
	}
	if service.completionRequest.GetPromptText() != "prompt" {
		t.Fatalf("server received unexpected completion request: %+v", service.completionRequest)
	}

	chatResponse, err := client.RenderChatCompletion(ctx, newRenderChatCompletionRequest())
	if err != nil {
		t.Fatalf("render chat completion rpc: %v", err)
	}
	if got := chatResponse.GetResult().GetTokenIds(); len(got) != 3 {
		t.Fatalf("unexpected token ids: %v", got)
	}
	if service.chatRequest.GetModel() != "chat-model" {
		t.Fatalf("server received unexpected chat request: %+v", service.chatRequest)
	}
}

func TestUnimplementedTokenizationServiceServerReturnsErrors(t *testing.T) {
	unimplemented := UnimplementedTokenizationServiceServer{}
	ctx := context.Background()
	if _, err := unimplemented.Initialize(ctx, &InitializeRequest{}); err == nil {
		t.Fatal("expected unimplemented Initialize to return an error")
	}
	if _, err := unimplemented.Tokenize(ctx, &TokenizeRequest{}); err == nil {
		t.Fatal("expected unimplemented Tokenize to return an error")
	}
	if _, err := unimplemented.RenderCompletion(ctx, &RenderCompletionRequest{}); err == nil {
		t.Fatal("expected unimplemented RenderCompletion to return an error")
	}
	if _, err := unimplemented.RenderChatCompletion(ctx, &RenderChatCompletionRequest{}); err == nil {
		t.Fatal("expected unimplemented RenderChatCompletion to return an error")
	}
	unimplemented.mustEmbedUnimplementedTokenizationServiceServer()
	unimplemented.testEmbeddedByValue()
}
