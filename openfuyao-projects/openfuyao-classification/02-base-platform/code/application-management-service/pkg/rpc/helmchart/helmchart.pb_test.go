/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
package helmchart

import (
	"bytes"
	"context"
	"io"
	"net"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/test/bufconn"
	"google.golang.org/protobuf/proto"
)

func TestChartRequest_ProtoMethodsAndGetters(t *testing.T) {
	req := &ChartRequest{
		TargetSystem: 1,
		RepoName:     "r",
		ChartName:    "c",
		ChartVersion: "v",
	}

	// Exercise getters + nil receiver branches.
	if got := req.GetTargetSystem(); got != 1 {
		t.Fatalf("GetTargetSystem: expected 1, got %d", got)
	}
	if got := req.GetRepoName(); got != "r" {
		t.Fatalf("GetRepoName: expected r, got %s", got)
	}
	if got := req.GetChartName(); got != "c" {
		t.Fatalf("GetChartName: expected c, got %s", got)
	}
	if got := req.GetChartVersion(); got != "v" {
		t.Fatalf("GetChartVersion: expected v, got %s", got)
	}
	var nilReq *ChartRequest
	if got := nilReq.GetRepoName(); got != "" {
		t.Fatalf("nil GetRepoName: expected empty, got %s", got)
	}

	// Exercise protoimpl-generated methods.
	_ = req.String()
	req.ProtoMessage()
	_ = req.ProtoReflect()
	_, _ = req.Descriptor()

	// Exercise Marshal/Unmarshal paths.
	b, err := proto.Marshal(req)
	if err != nil {
		t.Fatalf("proto.Marshal: %v", err)
	}

	var roundTrip ChartRequest
	if err := proto.Unmarshal(b, &roundTrip); err != nil {
		t.Fatalf("proto.Unmarshal: %v", err)
	}

	// Exercise Reset on non-nil message.
	req.Reset()
	if req.GetRepoName() != "" {
		t.Fatalf("after Reset, expected empty RepoName")
	}
}

func TestChartResponse_ProtoMethodsAndGetters(t *testing.T) {
	resp := &ChartResponse{
		Success:    true,
		ChartBytes: []byte("abc"),
	}

	if got := resp.GetSuccess(); !got {
		t.Fatalf("GetSuccess: expected true")
	}
	if got := resp.GetChartBytes(); !bytes.Equal(got, []byte("abc")) {
		t.Fatalf("GetChartBytes: expected abc, got %v", got)
	}
	var nilResp *ChartResponse
	if got := nilResp.GetChartBytes(); got != nil {
		t.Fatalf("nil GetChartBytes: expected nil, got %v", got)
	}

	_ = resp.String()
	resp.ProtoMessage()
	_ = resp.ProtoReflect()
	_, _ = resp.Descriptor()

	b, err := proto.Marshal(resp)
	if err != nil {
		t.Fatalf("proto.Marshal: %v", err)
	}
	var roundTrip ChartResponse
	if err := proto.Unmarshal(b, &roundTrip); err != nil {
		t.Fatalf("proto.Unmarshal: %v", err)
	}

	// Reset covers generated Reset logic.
	resp.Reset()
	if resp.GetSuccess() {
		t.Fatalf("after Reset, expected Success=false")
	}
}

type fakeChartManagerServer struct {
	UnimplementedChartManagerServer
}

func (s *fakeChartManagerServer) GetHelmChart(req *ChartRequest, stream grpc.ServerStreamingServer[ChartResponse]) error {
	// Send a single response and finish.
	return stream.Send(&ChartResponse{
		Success:    true,
		ChartBytes: []byte("chart-bytes"),
	})
}

func TestChartManagerClientServer_BufconnStreaming(t *testing.T) {
	// bufconn avoids real network dependencies while executing generated grpc code paths.
	const bufSize = 1 << 20
	lis := bufconn.Listen(bufSize)

	s := grpc.NewServer()
	RegisterChartManagerServer(s, &fakeChartManagerServer{})

	go func() {
		_ = s.Serve(lis)
	}()
	t.Cleanup(func() {
		s.Stop()
		lis.Close()
	})

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	conn, err := grpc.DialContext(
		ctx,
		"bufnet",
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) {
			return lis.Dial()
		}),
		grpc.WithInsecure(),
	)
	if err != nil {
		t.Fatalf("DialContext: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })

	client := NewChartManagerClient(conn)
	stream, err := client.GetHelmChart(ctx, &ChartRequest{
		TargetSystem: 1,
		RepoName:     "repo",
		ChartName:    "chart",
		ChartVersion: "v1",
	})
	if err != nil {
		t.Fatalf("GetHelmChart: %v", err)
	}

	var got []byte
	for {
		msg, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("stream.Recv: %v", err)
		}
		if msg == nil || !msg.GetSuccess() {
			t.Fatalf("unexpected response: %#v", msg)
		}
		got = append(got, msg.GetChartBytes()...)
	}

	if string(got) != "chart-bytes" {
		t.Fatalf("unexpected chart bytes: %q", string(got))
	}
}

