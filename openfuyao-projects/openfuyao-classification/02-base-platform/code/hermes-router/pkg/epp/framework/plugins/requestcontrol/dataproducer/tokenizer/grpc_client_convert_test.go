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

package tokenizer

import (
	"testing"

	pb "hermes-router/api/tokenizer/v1"
)

func TestApplyRenderCompletionPromptSourcePrefersTokenIDs(t *testing.T) {
	request := &pb.RenderCompletionRequest{}
	applyRenderCompletionPromptSource(request, CompleteRequest{
		PromptText:     "fallback text",
		PromptTokenIDs: []uint32{4, 5, 6},
	})

	ids, ok := request.GetPromptSource().(*pb.RenderCompletionRequest_PromptTokenIds)
	if !ok {
		t.Fatalf("expected token id prompt source, got %T", request.GetPromptSource())
	}
	if got := ids.PromptTokenIds.GetValues(); len(got) != 3 || got[0] != 4 {
		t.Fatalf("unexpected prompt token ids: %v", got)
	}
}

func TestApplyRenderCompletionPromptSourceFallsBackToText(t *testing.T) {
	request := &pb.RenderCompletionRequest{}
	applyRenderCompletionPromptSource(request, CompleteRequest{PromptText: "prompt text"})

	text, ok := request.GetPromptSource().(*pb.RenderCompletionRequest_PromptText)
	if !ok {
		t.Fatalf("expected text prompt source, got %T", request.GetPromptSource())
	}
	if text.PromptText != "prompt text" {
		t.Fatalf("unexpected prompt text: %q", text.PromptText)
	}
}

func TestConvertModalityToProto(t *testing.T) {
	cases := []struct {
		modality string
		want     pb.Modality
	}{
		{"image", pb.Modality_MODALITY_IMAGE},
		{"IMAGE", pb.Modality_MODALITY_IMAGE},
		{"MODALITY_image", pb.Modality_MODALITY_IMAGE},
		{"MODALITY_IMAGE", pb.Modality_MODALITY_IMAGE},
		{"", pb.Modality_MODALITY_UNSPECIFIED},
		{"audio", pb.Modality_MODALITY_UNSPECIFIED},
		{"MODALITY_VIDEO", pb.Modality_MODALITY_UNSPECIFIED},
	}
	for _, tc := range cases {
		if got := convertModalityToProto(tc.modality); got != tc.want {
			t.Fatalf("modality %q: expected %v, got %v", tc.modality, tc.want, got)
		}
	}
}

func TestConvertMediaToProtoHandlesNilMedia(t *testing.T) {
	if got := convertMediaToProto(nil); got != nil {
		t.Fatalf("expected nil media conversion, got %+v", got)
	}
}

func TestConvertMediaToProtoPreservesAllFields(t *testing.T) {
	media := &MediaRef{
		Modality:   "image",
		URL:        "https://example.test/img.png",
		InlineData: []byte("bytes"),
		MIMEType:   "image/png",
		DetailJSON: `{"scale":2}`,
	}
	converted := convertMediaToProto(media)
	if converted.GetModality() != pb.Modality_MODALITY_IMAGE {
		t.Fatalf("unexpected modality: %v", converted.GetModality())
	}
	if converted.GetUrl() != media.URL || string(converted.GetInlineData()) != "bytes" ||
		converted.GetMimeType() != media.MIMEType || converted.GetDetailJson() != media.DetailJSON {
		t.Fatalf("unexpected converted media: %+v", converted)
	}
}
