package helmchart

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type testChartManagerServer struct {
	UnimplementedChartManagerServer
}

type fakeClientStream struct {
	sendErr      error
	closeSendErr error
}

func (f *fakeClientStream) Header() (metadata.MD, error) { return metadata.MD{}, nil }
func (f *fakeClientStream) Trailer() metadata.MD          { return metadata.MD{} }
func (f *fakeClientStream) CloseSend() error              { return f.closeSendErr }
func (f *fakeClientStream) Context() context.Context      { return context.Background() }
func (f *fakeClientStream) SendMsg(_ interface{}) error   { return f.sendErr }
func (f *fakeClientStream) RecvMsg(_ interface{}) error   { return nil }

type fakeClientConn struct {
	stream grpc.ClientStream
	err    error
}

func (f *fakeClientConn) Invoke(context.Context, string, interface{}, interface{}, ...grpc.CallOption) error {
	return nil
}
func (f *fakeClientConn) NewStream(context.Context, *grpc.StreamDesc, string, ...grpc.CallOption) (grpc.ClientStream, error) {
	return f.stream, f.err
}

func TestChartRequestMethods(t *testing.T) {
	req := &ChartRequest{
		TargetSystem: 1,
		RepoName:     "repo",
		ChartName:    "chart",
		ChartVersion: "1.0.0",
	}
	assert.Equal(t, int32(1), req.GetTargetSystem())
	assert.Equal(t, "repo", req.GetRepoName())
	assert.Equal(t, "chart", req.GetChartName())
	assert.Equal(t, "1.0.0", req.GetChartVersion())
	assert.NotEmpty(t, req.String())
	assert.NotNil(t, req.ProtoReflect())
	rawDesc, idx := req.Descriptor()
	assert.NotEmpty(t, rawDesc)
	assert.NotEmpty(t, idx)
	req.Reset()
	assert.Equal(t, "", req.GetRepoName())
}

func TestChartResponseMethods(t *testing.T) {
	resp := &ChartResponse{
		Success:    true,
		ChartBytes: []byte("abc"),
	}
	assert.True(t, resp.GetSuccess())
	assert.Equal(t, []byte("abc"), resp.GetChartBytes())
	assert.NotEmpty(t, resp.String())
	assert.NotNil(t, resp.ProtoReflect())
	rawDesc, idx := resp.Descriptor()
	assert.NotEmpty(t, rawDesc)
	assert.NotEmpty(t, idx)
	resp.Reset()
	assert.False(t, resp.GetSuccess())
}

func TestNilGetters(t *testing.T) {
	var req *ChartRequest
	assert.Equal(t, int32(0), req.GetTargetSystem())
	assert.Equal(t, "", req.GetRepoName())
	assert.Equal(t, "", req.GetChartName())
	assert.Equal(t, "", req.GetChartVersion())

	var resp *ChartResponse
	assert.False(t, resp.GetSuccess())
	assert.Nil(t, resp.GetChartBytes())
}

func TestUnimplementedServerAndRegister(t *testing.T) {
	unimplemented := UnimplementedChartManagerServer{}
	err := unimplemented.GetHelmChart(&ChartRequest{}, nil)
	assert.Error(t, err)
	assert.Equal(t, codes.Unimplemented, status.Code(err))

	grpcServer := grpc.NewServer()
	RegisterChartManagerServer(grpcServer, &testChartManagerServer{})
	assert.NotNil(t, ChartManager_ServiceDesc.Streams)
	assert.Equal(t, ChartManager_GetHelmChart_FullMethodName, "/rpc.ChartManager/GetHelmChart")
}

func TestNewChartManagerClientAndGetHelmChart(t *testing.T) {
	connErr := &fakeClientConn{err: errors.New("new stream failed")}
	client := NewChartManagerClient(connErr)
	assert.NotNil(t, client)

	stream, err := client.GetHelmChart(context.Background(), &ChartRequest{})
	assert.Error(t, err)
	assert.Nil(t, stream)

	connSendErr := &fakeClientConn{stream: &fakeClientStream{sendErr: errors.New("send failed")}}
	client = NewChartManagerClient(connSendErr)
	stream, err = client.GetHelmChart(context.Background(), &ChartRequest{})
	assert.Error(t, err)
	assert.Nil(t, stream)

	connCloseErr := &fakeClientConn{stream: &fakeClientStream{closeSendErr: errors.New("close failed")}}
	client = NewChartManagerClient(connCloseErr)
	stream, err = client.GetHelmChart(context.Background(), &ChartRequest{})
	assert.Error(t, err)
	assert.Nil(t, stream)

	connOK := &fakeClientConn{stream: &fakeClientStream{}}
	client = NewChartManagerClient(connOK)
	stream, err = client.GetHelmChart(context.Background(), &ChartRequest{})
	assert.NoError(t, err)
	assert.NotNil(t, stream)
}
