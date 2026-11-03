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

package predictionv1

import (
	"context"
	"net"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/proto"
)

func newPredictInput() *PredictInput {
	return &PredictInput{
		InputId:              3,
		PodType:              "prefill",
		InputTokenLength:     128,
		PrefixCacheScore:     0.25,
		NumRequestWaiting:    2,
		NumRequestRunning:    5,
		KvCacheUsage:         0.5,
		InflightTokens:       64,
		HbmUtilization:       0.75,
		AiCoreUtilization:    0.6,
		BandwidthUtilization: 0.4,
		ServerIp:             "10.0.0.2",
	}
}

func TestPredictInputAccessors(t *testing.T) {
	input := newPredictInput()
	if got := input.GetInputId(); got != 3 {
		t.Fatalf("unexpected InputId: %v", got)
	}
	if got := input.GetPodType(); got != "prefill" {
		t.Fatalf("unexpected PodType: %v", got)
	}
	if got := input.GetInputTokenLength(); got != 128 {
		t.Fatalf("unexpected InputTokenLength: %v", got)
	}
	if got := input.GetPrefixCacheScore(); got != 0.25 {
		t.Fatalf("unexpected PrefixCacheScore: %v", got)
	}
	if got := input.GetNumRequestWaiting(); got != 2 {
		t.Fatalf("unexpected NumRequestWaiting: %v", got)
	}
	if got := input.GetNumRequestRunning(); got != 5 {
		t.Fatalf("unexpected NumRequestRunning: %v", got)
	}
	if got := input.GetKvCacheUsage(); got != 0.5 {
		t.Fatalf("unexpected KvCacheUsage: %v", got)
	}
	if got := input.GetInflightTokens(); got != 64 {
		t.Fatalf("unexpected InflightTokens: %v", got)
	}
	if got := input.GetHbmUtilization(); got != 0.75 {
		t.Fatalf("unexpected HbmUtilization: %v", got)
	}
	if got := input.GetAiCoreUtilization(); got != 0.6 {
		t.Fatalf("unexpected AiCoreUtilization: %v", got)
	}
	if got := input.GetBandwidthUtilization(); got != 0.4 {
		t.Fatalf("unexpected BandwidthUtilization: %v", got)
	}
	if got := input.GetServerIp(); got != "10.0.0.2" {
		t.Fatalf("unexpected ServerIp: %v", got)
	}
}

func TestMessageNilReceiverAccessorsReturnZeroValues(t *testing.T) {
	var input *PredictInput
	if got := input.GetInputId(); got != 0 {
		t.Fatalf("expected zero InputId, got %v", got)
	}
	if got := input.GetPodType(); got != "" {
		t.Fatalf("expected empty PodType, got %q", got)
	}
	if got := input.GetServerIp(); got != "" {
		t.Fatalf("expected empty ServerIp, got %q", got)
	}

	var output *PredictOutput
	if got := output.GetInputId(); got != 0 {
		t.Fatalf("expected zero InputId, got %v", got)
	}
	if got := output.GetServerIp(); got != "" {
		t.Fatalf("expected empty ServerIp, got %q", got)
	}
	if got := output.GetPredictedTtftMillis(); got != 0 {
		t.Fatalf("expected zero PredictedTtftMillis, got %v", got)
	}
	if got := output.GetPredictedTpotMillis(); got != 0 {
		t.Fatalf("expected zero PredictedTpotMillis, got %v", got)
	}

	var request *PredictRequest
	if got := request.GetTargetModel(); got != "" {
		t.Fatalf("expected empty TargetModel, got %q", got)
	}
	if got := request.GetInputs(); got != nil {
		t.Fatalf("expected nil Inputs, got %v", got)
	}

	var metadata *ResponseMetadata
	if got := metadata.GetTargetModel(); got != "" {
		t.Fatalf("expected empty TargetModel, got %q", got)
	}
	if got := metadata.GetModelVersion(); got != "" {
		t.Fatalf("expected empty ModelVersion, got %q", got)
	}
	if got := metadata.GetOmittedInputs(); got != 0 {
		t.Fatalf("expected zero OmittedInputs, got %v", got)
	}

	var response *PredictResponse
	if got := response.GetOutputs(); got != nil {
		t.Fatalf("expected nil Outputs, got %v", got)
	}
	if got := response.GetMetadata(); got != nil {
		t.Fatalf("expected nil Metadata, got %v", got)
	}
}

// descriptorMessage is satisfied by every generated message type in this
// package; it adds the legacy Descriptor accessor to proto.Message.
type descriptorMessage interface {
	proto.Message
	Reset()
	String() string
	ProtoMessage()
	Descriptor() ([]byte, []int)
}

func TestMessageLifecycleOperations(t *testing.T) {
	request := &PredictRequest{
		TargetModel: "test-model",
		Inputs:      []*PredictInput{newPredictInput()},
	}
	request.Reset()
	if request.GetTargetModel() != "" || request.GetInputs() != nil {
		t.Fatalf("expected Reset to clear request, got %+v", request)
	}

	messages := []descriptorMessage{
		newPredictInput(),
		&PredictOutput{InputId: 1, ServerIp: "10.0.0.2", PredictedTtftMillis: 1, PredictedTpotMillis: 2},
		&PredictRequest{TargetModel: "model"},
		&ResponseMetadata{TargetModel: "model", ModelVersion: "v1", OmittedInputs: 1},
		&PredictResponse{Outputs: []*PredictOutput{{InputId: 1}}, Metadata: &ResponseMetadata{TargetModel: "model"}},
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

func TestPredictRequestProtoRoundTrip(t *testing.T) {
	request := &PredictRequest{
		TargetModel: "round-trip-model",
		Inputs:      []*PredictInput{newPredictInput(), {InputId: 4, ServerIp: "10.0.0.3"}},
	}

	encoded, err := proto.Marshal(request)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	decoded := &PredictRequest{}
	if err := proto.Unmarshal(encoded, decoded); err != nil {
		t.Fatalf("unmarshal request: %v", err)
	}
	if !proto.Equal(request, decoded) {
		t.Fatalf("round-trip mismatch: %+v != %+v", request, decoded)
	}
}

func TestPredictResponseProtoRoundTrip(t *testing.T) {
	response := &PredictResponse{
		Outputs: []*PredictOutput{{
			InputId:             1,
			ServerIp:            "10.0.0.4",
			PredictedTtftMillis: 120.5,
			PredictedTpotMillis: 15.25,
		}},
		Metadata: &ResponseMetadata{
			TargetModel:   "round-trip-model",
			ModelVersion:  "v2",
			OmittedInputs: 1,
		},
	}

	encoded, err := proto.Marshal(response)
	if err != nil {
		t.Fatalf("marshal response: %v", err)
	}
	decoded := &PredictResponse{}
	if err := proto.Unmarshal(encoded, decoded); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}
	if !proto.Equal(response, decoded) {
		t.Fatalf("round-trip mismatch: %+v != %+v", response, decoded)
	}
}

type stubPredictionService struct {
	UnimplementedPredictionServiceServer
	request *PredictRequest
}

func (s *stubPredictionService) Predict(_ context.Context, req *PredictRequest) (*PredictResponse, error) {
	s.request = req
	return &PredictResponse{
		Outputs: []*PredictOutput{{
			InputId:             req.GetInputs()[0].GetInputId(),
			ServerIp:            req.GetInputs()[0].GetServerIp(),
			PredictedTtftMillis: 100,
			PredictedTpotMillis: 10,
		}},
		Metadata: &ResponseMetadata{TargetModel: req.GetTargetModel()},
	}, nil
}

func TestPredictionServiceGRPCRoundTrip(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen tcp: %v", err)
	}
	server := grpc.NewServer()
	service := &stubPredictionService{}
	RegisterPredictionServiceServer(server, service)
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

	client := NewPredictionServiceClient(conn)
	request := &PredictRequest{
		TargetModel: "grpc-model",
		Inputs:      []*PredictInput{newPredictInput()},
	}
	response, err := client.Predict(context.Background(), request)
	if err != nil {
		t.Fatalf("predict rpc: %v", err)
	}
	if got := len(response.GetOutputs()); got != 1 {
		t.Fatalf("expected 1 output, got %d", got)
	}
	if response.GetOutputs()[0].GetInputId() != 3 {
		t.Fatalf("unexpected output input id: %v", response.GetOutputs()[0].GetInputId())
	}
	if got := response.GetMetadata().GetTargetModel(); got != "grpc-model" {
		t.Fatalf("unexpected metadata target model: %q", got)
	}
	if service.request.GetTargetModel() != "grpc-model" {
		t.Fatalf("server received unexpected request: %+v", service.request)
	}
}

func TestUnimplementedPredictionServiceServerReturnsError(t *testing.T) {
	unimplemented := UnimplementedPredictionServiceServer{}
	if _, err := unimplemented.Predict(context.Background(), &PredictRequest{}); err == nil {
		t.Fatal("expected unimplemented Predict to return an error")
	}
	unimplemented.mustEmbedUnimplementedPredictionServiceServer()
	unimplemented.testEmbeddedByValue()
}
