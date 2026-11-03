package rpc

import (
	"bytes"
	"context"
	"errors"
	"mime/multipart"
	"testing"

	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc/metadata"
	"k8s.io/client-go/rest"

	helmv1 "marketplace-service/pkg/api/marketplace/v1beta1"
	"marketplace-service/pkg/helm"
	helmModel "marketplace-service/pkg/models/helm"
	pb "marketplace-service/pkg/rpc/helmchart"
	"marketplace-service/pkg/server/param"
	"marketplace-service/pkg/utils/httputil"
)

type mockHelmOperation struct {
	getChartBytesByVersion func(repoName, chartName, version string) (*bytes.Buffer, error)
}

func (m *mockHelmOperation) CreateRepo(_ *helmModel.SafeRepoEntry) (*httputil.ResponseJson, int) {
	return nil, 0
}
func (m *mockHelmOperation) UpdateRepo(_ *helmModel.SafeRepoEntry) (*httputil.ResponseJson, int) {
	return nil, 0
}
func (m *mockHelmOperation) DeleteRepo(_ string) (*httputil.ResponseJson, int) { return nil, 0 }
func (m *mockHelmOperation) ListRepo(_ *param.Query, _ string) (*httputil.ResponseJson, int) {
	return nil, 0
}
func (m *mockHelmOperation) GetRepo(_ string) (*httputil.ResponseJson, int) { return nil, 0 }
func (m *mockHelmOperation) SyncAllRepos() (*httputil.ResponseJson, int)    { return nil, 0 }
func (m *mockHelmOperation) SyncRepo(_ string) (*httputil.ResponseJson, int) { return nil, 0 }
func (m *mockHelmOperation) GetRepoSyncStatus(_ string) (*httputil.ResponseJson, int) {
	return nil, 0
}
func (m *mockHelmOperation) GetLatestCharts(_ *helmModel.ChartSearchParam) (*httputil.ResponseJson, int) {
	return nil, 0
}
func (m *mockHelmOperation) GetChartsWithOfficialTags(_ []string, _ *param.Query) (*httputil.ResponseJson, int) {
	return nil, 0
}
func (m *mockHelmOperation) CountCharts() (*httputil.ResponseJson, int) { return nil, 0 }
func (m *mockHelmOperation) UploadChart(_ multipart.File, _ *multipart.FileHeader) (*httputil.ResponseJson, int) {
	return nil, 0
}
func (m *mockHelmOperation) DeleteChart(_ string) (*httputil.ResponseJson, int) { return nil, 0 }
func (m *mockHelmOperation) DeleteChartVersion(_, _ string) (*httputil.ResponseJson, int) {
	return nil, 0
}
func (m *mockHelmOperation) GetChartVersions(_, _ string) (*httputil.ResponseJson, int) {
	return nil, 0
}
func (m *mockHelmOperation) GetChartVersion(_, _, _ string) (*httputil.ResponseJson, int) {
	return nil, 0
}
func (m *mockHelmOperation) GetChartFiles(_, _, _, _ string) (*httputil.ResponseJson, int) {
	return nil, 0
}
func (m *mockHelmOperation) GetChartBytesByVersion(repoName, chartName, version string) (*bytes.Buffer, error) {
	if m.getChartBytesByVersion != nil {
		return m.getChartBytesByVersion(repoName, chartName, version)
	}
	return bytes.NewBuffer(nil), nil
}

type mockChartStream struct {
	sent       []*pb.ChartResponse
	sendErr    error
	sendCalled int
}

func (m *mockChartStream) Send(resp *pb.ChartResponse) error {
	m.sendCalled++
	if m.sendErr != nil {
		return m.sendErr
	}
	m.sent = append(m.sent, resp)
	return nil
}
func (m *mockChartStream) SetHeader(_ metadata.MD) error { return nil }
func (m *mockChartStream) SendHeader(_ metadata.MD) error { return nil }
func (m *mockChartStream) SetTrailer(_ metadata.MD)       {}
func (m *mockChartStream) Context() context.Context        { return context.Background() }
func (m *mockChartStream) SendMsg(_ any) error             { return nil }
func (m *mockChartStream) RecvMsg(_ any) error             { return nil }

func TestGetHelmChart_GetBytesFailed(t *testing.T) {
	s := &Server{
		Handler: &helmv1.Handler{
			HelmHandler: &mockHelmOperation{
				getChartBytesByVersion: func(_, _, _ string) (*bytes.Buffer, error) {
					return nil, errors.New("failed")
				},
			},
		},
	}

	stream := &mockChartStream{}
	err := s.GetHelmChart(&pb.ChartRequest{}, stream)
	assert.Error(t, err)
	assert.Equal(t, 0, stream.sendCalled)
}

func TestGetHelmChart_SendFailed(t *testing.T) {
	s := &Server{
		Handler: &helmv1.Handler{
			HelmHandler: &mockHelmOperation{
				getChartBytesByVersion: func(_, _, _ string) (*bytes.Buffer, error) {
					return bytes.NewBuffer(make([]byte, 20000)), nil
				},
			},
		},
	}

	stream := &mockChartStream{sendErr: errors.New("send failed")}
	err := s.GetHelmChart(&pb.ChartRequest{}, stream)
	assert.Error(t, err)
	assert.Equal(t, 1, stream.sendCalled)
}

func TestGetHelmChart_Success(t *testing.T) {
	s := &Server{
		Handler: &helmv1.Handler{
			HelmHandler: &mockHelmOperation{
				getChartBytesByVersion: func(_, _, _ string) (*bytes.Buffer, error) {
					return bytes.NewBuffer([]byte("chart-bytes")), nil
				},
			},
		},
	}

	stream := &mockChartStream{}
	err := s.GetHelmChart(&pb.ChartRequest{
		RepoName:     "repo",
		ChartName:    "chart",
		ChartVersion: "1.0.0",
	}, stream)
	assert.NoError(t, err)
	assert.Equal(t, 1, stream.sendCalled)
	assert.Len(t, stream.sent, 1)
	assert.True(t, stream.sent[0].Success)
	assert.Equal(t, []byte("chart-bytes"), stream.sent[0].ChartBytes)
}

func TestServerStructEmbed(t *testing.T) {
	var _ pb.ChartManagerServer = &Server{
		UnimplementedChartManagerServer: pb.UnimplementedChartManagerServer{},
		Handler:                         &helmv1.Handler{},
	}
	assert.True(t, true)
}

func TestMockOperationSatisfiesInterface(t *testing.T) {
	var _ helm.Operation = &mockHelmOperation{}
	assert.NotNil(t, &rest.Config{})
}

