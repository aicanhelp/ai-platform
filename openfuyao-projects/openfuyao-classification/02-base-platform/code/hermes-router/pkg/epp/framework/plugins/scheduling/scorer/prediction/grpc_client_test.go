/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package prediction

import (
	"context"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc"

	pb "hermes-router/api/prediction/v1"
	"hermes-router/pkg/epp/internal/predictiondata"
)

func TestGRPCPredictorClientPredictsOverUDS(t *testing.T) {
	socketPath := udsSocketPath(t, "p.sock")
	server := grpc.NewServer()
	pb.RegisterPredictionServiceServer(server, &fakePredictionService{})
	listener, err := net.Listen("unix", socketPath)
	if err != nil {
		t.Fatalf("listen unix socket: %v", err)
	}
	go func() {
		_ = server.Serve(listener)
	}()
	t.Cleanup(server.Stop)

	client, err := NewGRPCPredictorClient(GRPCPredictorConfig{
		Address: "unix://" + socketPath,
		Timeout: time.Second,
	})
	if err != nil {
		t.Fatalf("create client: %v", err)
	}
	t.Cleanup(func() { _ = client.Close() })

	outputs, err := client.Predict(context.Background(), "test-model", []predictiondata.Input{
		{InputID: 0, PodType: predictiondata.PodTypeAggregate, ServerIP: "10.0.0.1"},
		{InputID: 1, PodType: predictiondata.PodTypePrefill, ServerIP: "10.0.0.2"},
	})
	if err != nil {
		t.Fatalf("predict: %v", err)
	}
	if len(outputs) != 2 || outputs[0].InputID != 0 || outputs[1].InputID != 1 {
		t.Fatalf("unexpected outputs: %+v", outputs)
	}
}

func TestGRPCPredictorClientConstructorDoesNotDial(t *testing.T) {
	socketPath := filepath.Join(t.TempDir(), "missing.sock")
	client, err := NewGRPCPredictorClient(GRPCPredictorConfig{
		Address: "unix://" + socketPath,
		Timeout: 25 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("constructor should not require socket readiness: %v", err)
	}
	t.Cleanup(func() { _ = client.Close() })

	_, err = client.Predict(context.Background(), "test-model", []predictiondata.Input{{InputID: 0, PodType: predictiondata.PodTypeAggregate}})
	if err == nil {
		t.Fatal("expected first predict call to return transport error")
	}
}

func TestGRPCPredictorClientValidation(t *testing.T) {
	if _, err := NewGRPCPredictorClient(GRPCPredictorConfig{Address: "tcp://127.0.0.1:9000"}); err == nil || !strings.Contains(err.Error(), "unix://") {
		t.Fatalf("expected unix address validation error, got %v", err)
	}
	client, err := NewGRPCPredictorClient(GRPCPredictorConfig{Address: "unix:///tmp/missing.sock", Timeout: 25 * time.Millisecond})
	if err != nil {
		t.Fatalf("create client: %v", err)
	}
	t.Cleanup(func() { _ = client.Close() })
	if _, err := client.Predict(context.Background(), "", []predictiondata.Input{}); err == nil || !strings.Contains(err.Error(), "targetModel") {
		t.Fatalf("expected target model validation error, got %v", err)
	}
}

const windowsSocketRoot = "/hermes-ut-socks"

// udsSocketPath returns a unique Unix-domain-socket path that is valid both
// as a filesystem path and as a grpc "unix://" dial target. Windows
// drive-letter paths are rejected by grpc target parsing, so on Windows the
// socket is created in a drive-root-relative directory instead; on Unix the
// default temp directory applies.
func udsSocketPath(t *testing.T, name string) string {
	t.Helper()
	root := ""
	if runtime.GOOS == "windows" {
		if err := os.MkdirAll(windowsSocketRoot, 0o755); err != nil {
			t.Fatalf("create windows socket root: %v", err)
		}
		root = windowsSocketRoot
	}
	dir, err := os.MkdirTemp(root, "hp-")
	if err != nil {
		t.Fatalf("create temp dir: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	return filepath.ToSlash(filepath.Join(dir, name))
}

type fakePredictionService struct {
	pb.UnimplementedPredictionServiceServer
}

func (fakePredictionService) Predict(_ context.Context, request *pb.PredictRequest) (*pb.PredictResponse, error) {
	outputs := make([]*pb.PredictOutput, 0, len(request.GetInputs()))
	for _, input := range request.GetInputs() {
		outputs = append(outputs, &pb.PredictOutput{
			InputId:             input.GetInputId(),
			ServerIp:            input.GetServerIp(),
			PredictedTtftMillis: 10 + float64(input.GetInputId()),
			PredictedTpotMillis: 0.5 + float64(input.GetInputId()),
		})
	}
	return &pb.PredictResponse{
		Outputs: outputs,
		Metadata: &pb.ResponseMetadata{
			TargetModel: request.GetTargetModel(),
		},
	}, nil
}
