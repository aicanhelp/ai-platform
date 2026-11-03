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

package prediction

import (
	"context"
	"fmt"
	"net"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "hermes-router/api/prediction/v1"
	"hermes-router/pkg/epp/internal/predictiondata"
)

const (
	defaultTimeout        = 500 * time.Millisecond
	defaultMaxBatchSize   = 128
	defaultMaxMessageSize = 16 * 1024 * 1024
)

// GRPCPredictorConfig describes GRPCPredictorConfig data used by Hermes Router.
type GRPCPredictorConfig struct {
	// Address stores the Address value.
	Address string
	// Timeout stores the Timeout value.
	Timeout time.Duration
	// MaxMessageSize stores the MaxMessageSize value.
	MaxMessageSize int
}

// GRPCPredictorClient describes GRPCPredictorClient data used by Hermes Router.
type GRPCPredictorClient struct {
	conn    *grpc.ClientConn
	client  pb.PredictionServiceClient
	timeout time.Duration
}

var _ PredictorClient = (*GRPCPredictorClient)(nil)

// NewGRPCPredictorClient constructs a GRPCPredictorClient instance.
func NewGRPCPredictorClient(config GRPCPredictorConfig) (*GRPCPredictorClient, error) {
	socketPath, err := socketPathFromAddress(config.Address)
	if err != nil {
		return nil, err
	}
	if config.Timeout <= 0 {
		config.Timeout = defaultTimeout
	}
	if config.MaxMessageSize <= 0 {
		config.MaxMessageSize = defaultMaxMessageSize
	}

	conn, err := grpc.NewClient(
		config.Address,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) {
			var dialer net.Dialer
			return dialer.DialContext(ctx, "unix", socketPath)
		}),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(config.MaxMessageSize),
			grpc.MaxCallSendMsgSize(config.MaxMessageSize),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("create prediction grpc client: %w", err)
	}
	return &GRPCPredictorClient{
		conn:    conn,
		client:  pb.NewPredictionServiceClient(conn),
		timeout: config.Timeout,
	}, nil
}

func socketPathFromAddress(address string) (string, error) {
	if strings.TrimSpace(address) == "" {
		return "", fmt.Errorf("predictorAddress is required")
	}
	if !strings.HasPrefix(address, "unix://") {
		return "", fmt.Errorf("predictorAddress must use unix://")
	}
	socketPath := strings.TrimPrefix(address, "unix://")
	if socketPath == "" {
		return "", fmt.Errorf("predictorAddress unix socket path is required")
	}
	return socketPath, nil
}

// Close releases resources held by the receiver.
func (c *GRPCPredictorClient) Close() error {
	if c == nil || c.conn == nil {
		return nil
	}
	return c.conn.Close()
}

// Predict provides the exported Predict operation.
func (c *GRPCPredictorClient) Predict(
	ctx context.Context,
	targetModel string,
	inputs []predictiondata.Input,
) ([]predictiondata.Output, error) {
	if c == nil || c.client == nil {
		return nil, fmt.Errorf("prediction grpc client is not initialized")
	}
	if strings.TrimSpace(targetModel) == "" {
		return nil, fmt.Errorf("targetModel is required")
	}
	callCtx, cancel := c.withTimeout(ctx)
	defer cancel()

	response, err := c.client.Predict(callCtx, &pb.PredictRequest{
		TargetModel: targetModel,
		Inputs:      inputsToProto(inputs),
	})
	if err != nil {
		return nil, fmt.Errorf("predict latency: %w", err)
	}
	return outputsFromProto(response.GetOutputs()), nil
}

// withTimeout applies the client's default per-RPC deadline only when the
// caller has not already supplied one. The scorer wraps the request context
// with its own total budget before reaching here; direct callers that pass a
// deadline-less context still get a defensive timeout.
func (c *GRPCPredictorClient) withTimeout(ctx context.Context) (context.Context, context.CancelFunc) {
	if ctx == nil {
		ctx = context.Background()
	}
	if _, hasDeadline := ctx.Deadline(); hasDeadline {
		return ctx, func() {
			// The caller owns cancellation when it already supplied a deadline.
		}
	}
	return context.WithTimeout(ctx, c.timeout)
}

func inputsToProto(inputs []predictiondata.Input) []*pb.PredictInput {
	converted := make([]*pb.PredictInput, 0, len(inputs))
	for _, input := range inputs {
		converted = append(converted, predictiondata.InputToProto(input))
	}
	return converted
}

func outputsFromProto(outputs []*pb.PredictOutput) []predictiondata.Output {
	converted := make([]predictiondata.Output, 0, len(outputs))
	for _, output := range outputs {
		converted = append(converted, predictiondata.OutputFromProto(output))
	}
	return converted
}
