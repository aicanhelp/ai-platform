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
	"net"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	pb "hermes-router/api/tokenizer/v1"
)

type fakeTokenizationService struct {
	pb.UnimplementedTokenizationServiceServer
	initializeFn       func(context.Context, *pb.InitializeRequest) (*pb.InitializeResponse, error)
	tokenizeFn         func(context.Context, *pb.TokenizeRequest) (*pb.TokenizeResponse, error)
	renderCompletionFn func(context.Context, *pb.RenderCompletionRequest) (*pb.RenderCompletionResponse, error)
	renderChatFn       func(context.Context, *pb.RenderChatCompletionRequest) (*pb.RenderChatCompletionResponse, error)
}

func (f *fakeTokenizationService) Initialize(ctx context.Context, req *pb.InitializeRequest) (*pb.InitializeResponse, error) {
	if f.initializeFn != nil {
		return f.initializeFn(ctx, req)
	}
	return &pb.InitializeResponse{
		Model:            req.Model,
		ResolvedProvider: pb.TokenizerProvider_TOKENIZER_PROVIDER_HUGGINGFACE,
	}, nil
}

func (f *fakeTokenizationService) Tokenize(ctx context.Context, req *pb.TokenizeRequest) (*pb.TokenizeResponse, error) {
	if f.tokenizeFn != nil {
		return f.tokenizeFn(ctx, req)
	}
	return &pb.TokenizeResponse{Result: &pb.TokenizationResult{TokenIds: []uint32{1, 2, 3}}}, nil
}

func (f *fakeTokenizationService) RenderCompletion(ctx context.Context, req *pb.RenderCompletionRequest) (*pb.RenderCompletionResponse, error) {
	if f.renderCompletionFn != nil {
		return f.renderCompletionFn(ctx, req)
	}
	return &pb.RenderCompletionResponse{Result: &pb.TokenizationResult{TokenIds: []uint32{10, 20}}}, nil
}

func (f *fakeTokenizationService) RenderChatCompletion(ctx context.Context, req *pb.RenderChatCompletionRequest) (*pb.RenderChatCompletionResponse, error) {
	if f.renderChatFn != nil {
		return f.renderChatFn(ctx, req)
	}
	return &pb.RenderChatCompletionResponse{Result: &pb.TokenizationResult{TokenIds: []uint32{5, 6, 7}}}, nil
}

func startFakeTokenizationServer(t *testing.T, service *fakeTokenizationService) string {
	t.Helper()
	socketPath := shortSocketPath(t)
	cleanup := startFakeTokenizationServerAtPath(t, socketPath, service)
	t.Cleanup(cleanup)
	return socketPath
}

func startFakeTokenizationServerAtPath(t *testing.T, socketPath string, service *fakeTokenizationService) func() {
	t.Helper()
	listener, err := net.Listen("unix", socketPath)
	if err != nil {
		t.Fatalf("listen unix socket: %v", err)
	}

	server := grpc.NewServer()
	pb.RegisterTokenizationServiceServer(server, service)
	go func() {
		_ = server.Serve(listener)
	}()

	return func() {
		server.GracefulStop()
		_ = listener.Close()
		_ = os.Remove(socketPath)
	}
}

func startFakeTokenizationServerAfterDelay(t *testing.T, delay time.Duration, service *fakeTokenizationService) string {
	t.Helper()
	socketPath := shortSocketPath(t)
	done := make(chan struct{})
	var (
		mu      sync.Mutex
		cleanup func()
	)

	go func() {
		timer := time.NewTimer(delay)
		defer timer.Stop()

		select {
		case <-timer.C:
		case <-done:
			return
		}

		mu.Lock()
		defer mu.Unlock()
		select {
		case <-done:
			return
		default:
		}
		cleanup = startFakeTokenizationServerAtPath(t, socketPath, service)
	}()

	t.Cleanup(func() {
		close(done)
		mu.Lock()
		defer mu.Unlock()
		if cleanup != nil {
			cleanup()
		}
	})

	return socketPath
}

const windowsSocketRoot = "/hermes-ut-socks"

// shortSocketPath returns a unique Unix-domain-socket path that is valid both
// as a filesystem path and as a grpc "unix://" dial target. Windows
// drive-letter paths are rejected by grpc target parsing, so on Windows the
// socket is created in a drive-root-relative directory instead; on Unix the
// default temp directory applies.
func shortSocketPath(t *testing.T) string {
	t.Helper()
	root := ""
	if runtime.GOOS == "windows" {
		if err := os.MkdirAll(windowsSocketRoot, 0o755); err != nil {
			t.Fatalf("create windows socket root: %v", err)
		}
		root = windowsSocketRoot
	}
	dir, err := os.MkdirTemp(root, "ht-")
	if err != nil {
		t.Fatalf("create short temp dir: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	return filepath.ToSlash(filepath.Join(dir, "t.sock"))
}

func newTestGRPCTokenizer(t *testing.T, service *fakeTokenizationService) *GRPCTokenizer {
	t.Helper()
	socketPath := startFakeTokenizationServer(t, service)
	client, err := NewGRPCTokenizer(context.Background(), GRPCTokenizerConfig{
		SocketPath:     socketPath,
		Model:          "test-model",
		TokenizerName:  "",
		Timeout:        time.Second,
		MaxMessageSize: 4 * 1024 * 1024,
	})
	if err != nil {
		t.Fatalf("create grpc tokenizer: %v", err)
	}
	t.Cleanup(func() {
		_ = client.Close()
	})
	return client
}

func TestNewGRPCTokenizerDoesNotRetryInitialize(t *testing.T) {
	var attempts atomic.Int32
	socketPath := startFakeTokenizationServer(t, &fakeTokenizationService{
		initializeFn: func(_ context.Context, _ *pb.InitializeRequest) (*pb.InitializeResponse, error) {
			attempts.Add(1)
			return nil, status.Error(codes.Unavailable, "warming")
		},
	})

	_, err := NewGRPCTokenizer(context.Background(), GRPCTokenizerConfig{
		SocketPath:     socketPath,
		Model:          "single-attempt-model",
		Timeout:        100 * time.Millisecond,
		MaxMessageSize: 4 * 1024 * 1024,
	})
	if err == nil {
		t.Fatal("expected constructor initialize failure")
	}
	if got := attempts.Load(); got != 1 {
		t.Fatalf("expected exactly one initialize attempt, got %d", got)
	}
}

func TestNewGRPCTokenizerMissingSocketSurfacesRetryableError(t *testing.T) {
	socketPath := shortSocketPath(t)

	start := time.Now()
	_, err := NewGRPCTokenizer(context.Background(), GRPCTokenizerConfig{
		SocketPath:     socketPath,
		Model:          "test-model",
		Timeout:        time.Second,
		InitTimeout:    500 * time.Millisecond,
		MaxMessageSize: 4 * 1024 * 1024,
	})
	if err == nil {
		t.Fatal("expected connection error for missing socket")
	}
	if status.Code(err) != codes.Unavailable {
		t.Fatalf("expected Unavailable connection error, got %v", err)
	}
	if elapsed := time.Since(start); elapsed >= 400*time.Millisecond {
		t.Fatalf("expected fast connection failure, took %v", elapsed)
	}
}

func TestNewGRPCTokenizerCancelStopsInitializeAttempt(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	initializeStarted := make(chan struct{}, 1)
	socketPath := startFakeTokenizationServer(t, &fakeTokenizationService{
		initializeFn: func(initCtx context.Context, _ *pb.InitializeRequest) (*pb.InitializeResponse, error) {
			select {
			case initializeStarted <- struct{}{}:
			default:
			}
			<-initCtx.Done()
			return nil, initCtx.Err()
		},
	})

	resultCh := make(chan error, 1)
	go func() {
		_, err := NewGRPCTokenizer(ctx, GRPCTokenizerConfig{
			SocketPath:     socketPath,
			Model:          "cancel-model",
			Timeout:        100 * time.Millisecond,
			MaxMessageSize: 4 * 1024 * 1024,
		})
		resultCh <- err
	}()

	select {
	case <-initializeStarted:
	case <-time.After(200 * time.Millisecond):
		t.Fatal("expected initialize attempt to start")
	}

	cancel()

	select {
	case err := <-resultCh:
		if err == nil {
			t.Fatal("expected startup to fail after context cancellation")
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("expected single initialize attempt to exit promptly after context cancellation")
	}
}

func TestNewGRPCTokenizerFailsOnModelMismatch(t *testing.T) {
	socketPath := startFakeTokenizationServer(t, &fakeTokenizationService{
		initializeFn: func(_ context.Context, _ *pb.InitializeRequest) (*pb.InitializeResponse, error) {
			return &pb.InitializeResponse{
				Model:            "wrong-model",
				ResolvedProvider: pb.TokenizerProvider_TOKENIZER_PROVIDER_HUGGINGFACE,
			}, nil
		},
	})

	_, err := NewGRPCTokenizer(context.Background(), GRPCTokenizerConfig{
		SocketPath:     socketPath,
		Model:          "expected-model",
		Timeout:        time.Second,
		MaxMessageSize: 4 * 1024 * 1024,
	})
	if err == nil {
		t.Fatal("expected initialize model mismatch to fail")
	}
	if !errors.Is(err, errModelMismatch) {
		t.Fatalf("expected errModelMismatch, got %v", err)
	}
}

func TestGRPCTokenizerTokenize(t *testing.T) {
	client := newTestGRPCTokenizer(t, &fakeTokenizationService{
		tokenizeFn: func(_ context.Context, req *pb.TokenizeRequest) (*pb.TokenizeResponse, error) {
			if req.GetModel() != "test-model" {
				t.Fatalf("expected model test-model, got %q", req.GetModel())
			}
			if req.GetPrompt() != "hello world" {
				t.Fatalf("expected prompt hello world, got %q", req.GetPrompt())
			}
			return &pb.TokenizeResponse{Result: &pb.TokenizationResult{TokenIds: []uint32{1, 2, 3}}}, nil
		},
	})

	result, err := client.Tokenize(context.Background(), Request{Model: "test-model", Prompt: "hello world"})
	if err != nil {
		t.Fatalf("tokenize failed: %v", err)
	}
	if got := len(result.TokenIDs); got != 3 {
		t.Fatalf("expected 3 token ids, got %d", got)
	}
}

func TestGRPCTokenizerComplete(t *testing.T) {
	client := newTestGRPCTokenizer(t, &fakeTokenizationService{})

	result, err := client.Complete(context.Background(), CompleteRequest{
		Model:      "completion-model",
		PromptText: "hello world",
	})
	if err != nil {
		t.Fatalf("complete failed: %v", err)
	}
	if got := len(result.TokenIDs); got != 2 {
		t.Fatalf("expected 2 token ids, got %d", got)
	}
}

func TestGRPCTokenizerChatCompletePreservesMultimodalFeatures(t *testing.T) {
	client := newTestGRPCTokenizer(t, &fakeTokenizationService{
		renderChatFn: func(_ context.Context, _ *pb.RenderChatCompletionRequest) (*pb.RenderChatCompletionResponse, error) {
			return &pb.RenderChatCompletionResponse{
				Result: &pb.TokenizationResult{
					TokenIds: []uint32{1, 2, 3, 4},
					MultimodalFeatures: []*pb.MultiModalFeature{{
						Modality: pb.Modality_MODALITY_IMAGE,
						Hash:     "hash-1",
						Offset:   1,
						Length:   2,
					}},
				},
			}, nil
		},
	})

	result, err := client.ChatComplete(context.Background(), ChatCompleteRequest{
		Model: "chat-model",
		Messages: []ChatMessage{{
			Role: "user",
			ContentParts: []ChatContentPart{
				{Text: "describe"},
				{Media: &MediaRef{Modality: string(pb.Modality_MODALITY_IMAGE.String()), URL: "https://example.test/image.png"}},
			},
		}},
	})
	if err != nil {
		t.Fatalf("chat complete failed: %v", err)
	}
	if got := len(result.MultiModalFeatures); got != 1 {
		t.Fatalf("expected 1 multimodal feature, got %d", got)
	}
	if result.MultiModalFeatures[0].Hash != "hash-1" {
		t.Fatalf("expected multimodal feature hash to be preserved, got %q", result.MultiModalFeatures[0].Hash)
	}
}

func TestInitializeDeadlineComesFromInitTimeout(t *testing.T) {
	deadlineObserved := make(chan time.Duration, 1)
	service := &fakeTokenizationService{
		initializeFn: func(ctx context.Context, _ *pb.InitializeRequest) (*pb.InitializeResponse, error) {
			if deadline, ok := ctx.Deadline(); ok {
				deadlineObserved <- time.Until(deadline)
			}
			<-ctx.Done()
			return nil, ctx.Err()
		},
	}
	socketPath := startFakeTokenizationServer(t, service)

	start := time.Now()
	_, err := NewGRPCTokenizer(context.Background(), GRPCTokenizerConfig{
		SocketPath:     socketPath,
		Model:          "test-model",
		Timeout:        time.Second,
		InitTimeout:    300 * time.Millisecond,
		MaxMessageSize: 4 * 1024 * 1024,
	})
	if err == nil {
		t.Fatal("expected deadline error from blocked Initialize")
	}
	if status.Code(err) != codes.DeadlineExceeded {
		t.Fatalf("expected DeadlineExceeded, got %v", err)
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Fatalf("expected Initialize to respect the 300ms InitTimeout, took %v", elapsed)
	}
	select {
	case observed := <-deadlineObserved:
		if observed <= 0 || observed > time.Second {
			t.Fatalf("expected server-observed deadline near the client deadline, got %v", observed)
		}
	default:
		t.Fatal("expected server to observe the Initialize deadline")
	}
}

func TestInitializeFallsBackToDefaultInitTimeout(t *testing.T) {
	deadlineObserved := make(chan time.Duration, 1)
	service := &fakeTokenizationService{
		initializeFn: func(ctx context.Context, _ *pb.InitializeRequest) (*pb.InitializeResponse, error) {
			if deadline, ok := ctx.Deadline(); ok {
				deadlineObserved <- time.Until(deadline)
			}
			return &pb.InitializeResponse{
				Model:            "test-model",
				ResolvedProvider: pb.TokenizerProvider_TOKENIZER_PROVIDER_HUGGINGFACE,
			}, nil
		},
	}
	socketPath := startFakeTokenizationServer(t, service)

	client, err := NewGRPCTokenizer(context.Background(), GRPCTokenizerConfig{
		SocketPath: socketPath,
		Model:      "test-model",
		Timeout:    time.Second,
	})
	if err != nil {
		t.Fatalf("create grpc tokenizer: %v", err)
	}
	defer func() { _ = client.Close() }()

	select {
	case observed := <-deadlineObserved:
		if observed < defaultInitTimeout-time.Second || observed > defaultInitTimeout {
			t.Fatalf("expected deadline ≈ defaultInitTimeout (%v), got %v", defaultInitTimeout, observed)
		}
	default:
		t.Fatal("expected server to observe the Initialize deadline")
	}
}
