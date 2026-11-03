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

	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

// TokenizationResult holds the normalized output of any tokenization path.
type TokenizationResult struct {
	TokenIDs           []uint32
	MultiModalFeatures []fwkscheduling.MultiModalFeature
}

// InitializeRequest carries model and tokenizer initialization settings.
type InitializeRequest struct {
	Model         string
	TokenizerName string
}

// Request carries plain-text tokenization inputs.
type Request struct {
	Model            string
	Prompt           string
	AddSpecialTokens bool
}

// CompleteRequest carries completion-render tokenization inputs.
type CompleteRequest struct {
	Model                string
	PromptText           string
	PromptTokenIDs       []uint32
	AddSpecialTokens     bool
	TruncatePromptTokens *int32
}

// ChatMessage represents one chat message sent to the tokenizer backend.
type ChatMessage struct {
	Role          string
	Content       string
	ContentParts  []ChatContentPart
	Name          string
	ToolCallsJSON string
	ToolCallID    string
}

// ChatContentPart represents one structured chat content block.
type ChatContentPart struct {
	Text  string
	Media *MediaRef
}

// MediaRef describes one multimodal item referenced in chat content.
type MediaRef struct {
	Modality   string
	URL        string
	InlineData []byte
	MIMEType   string
	DetailJSON string
}

// ChatCompleteRequest carries chat-render tokenization inputs.
type ChatCompleteRequest struct {
	Model                  string
	Messages               []ChatMessage
	ChatTemplate           string
	ChatTemplateKWargsJSON string
	ToolsJSON              string
	ToolChoiceJSON         string
	AddGenerationPrompt    bool
	ContinueFinalMessage   bool
	MMProcessorKWargsJSON  string
	MediaIOKWargsJSON      string
}

// Backend abstracts the transport used by the plugin.
type Backend interface {
	Close() error
	Initialize(ctx context.Context, request InitializeRequest) error
	Tokenize(ctx context.Context, request Request) (*TokenizationResult, error)
	Complete(ctx context.Context, request CompleteRequest) (*TokenizationResult, error)
	ChatComplete(ctx context.Context, request ChatCompleteRequest) (*TokenizationResult, error)
}
