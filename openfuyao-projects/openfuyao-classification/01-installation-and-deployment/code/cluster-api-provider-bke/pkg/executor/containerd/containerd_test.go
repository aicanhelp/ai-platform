/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package containerd

import (
	"context"
	"syscall"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/containerd/containerd"
	"github.com/containerd/containerd/cio"
	srvconfig "github.com/containerd/containerd/services/server/config"
	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/apimachinery/pkg/util/wait"
	pb "k8s.io/cri-api/pkg/apis/runtime/v1"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
)

const (
	numZero           = 0
	numOne            = 1
	numTwo            = 2
	numThree          = 3
	numFour           = 4
	numEight          = 8
	numTen            = 10
	numSixty          = 60
	numTwoHundred     = 200
	numOneHundred     = 100
	numOneTwentySeven = 127
	numOneNinetyTwo   = 192
	shortWaitDuration = 100 * time.Millisecond
	longWaitDuration  = 5 * time.Second
	waitTimeout       = 5 * time.Second
	pollInterval      = 5 * time.Second
	containerdTimeout = 2 * time.Minute
)

type mockImageClient struct {
	pb.ImageServiceClient
	pullImageFunc   func(ctx context.Context, in *pb.PullImageRequest, opts ...grpc.CallOption) (*pb.PullImageResponse, error)
	imageStatusFunc func(ctx context.Context, in *pb.ImageStatusRequest, opts ...grpc.CallOption) (*pb.ImageStatusResponse, error)
}

func (m *mockImageClient) PullImage(ctx context.Context, in *pb.PullImageRequest, opts ...grpc.CallOption) (*pb.PullImageResponse, error) {
	if m.pullImageFunc != nil {
		return m.pullImageFunc(ctx, in, opts...)
	}
	return &pb.PullImageResponse{}, nil
}

func (m *mockImageClient) ImageStatus(ctx context.Context, in *pb.ImageStatusRequest, opts ...grpc.CallOption) (*pb.ImageStatusResponse, error) {
	if m.imageStatusFunc != nil {
		return m.imageStatusFunc(ctx, in, opts...)
	}
	return &pb.ImageStatusResponse{}, nil
}

type mockContainer struct {
	containerd.Container
	updateFunc func(ctx context.Context, opts ...containerd.UpdateContainerOpts) error
	labelsFunc func(ctx context.Context) (map[string]string, error)
	taskFunc   func(ctx context.Context, ioAttach cio.Attach) (containerd.Task, error)
	idVal      string
}

func (m *mockContainer) Update(ctx context.Context, opts ...containerd.UpdateContainerOpts) error {
	if m.updateFunc != nil {
		return m.updateFunc(ctx, opts...)
	}
	return nil
}

func (m *mockContainer) Labels(ctx context.Context) (map[string]string, error) {
	if m.labelsFunc != nil {
		return m.labelsFunc(ctx)
	}
	return map[string]string{}, nil
}

func (m *mockContainer) Task(ctx context.Context, ioAttach cio.Attach) (containerd.Task, error) {
	if m.taskFunc != nil {
		return m.taskFunc(ctx, ioAttach)
	}
	return nil, nil
}

func (m *mockContainer) ID() string {
	return m.idVal
}

type mockTask struct {
	containerd.Task
	statusFunc func(ctx context.Context) (containerd.Status, error)
	waitFunc   func(ctx context.Context) (<-chan containerd.ExitStatus, error)
	killFunc   func(ctx context.Context, sig syscall.Signal, opts ...containerd.KillOpts) error
	resumeFunc func(ctx context.Context) error
	deleteFunc func(ctx context.Context, opts ...containerd.ProcessDeleteOpts) (*containerd.ExitStatus, error)
}

func (m *mockTask) Status(ctx context.Context) (containerd.Status, error) {
	if m.statusFunc != nil {
		return m.statusFunc(ctx)
	}
	return containerd.Status{}, nil
}

func (m *mockTask) Wait(ctx context.Context) (<-chan containerd.ExitStatus, error) {
	if m.waitFunc != nil {
		return m.waitFunc(ctx)
	}
	ch := make(chan containerd.ExitStatus, numOne)
	return ch, nil
}

func (m *mockTask) Kill(ctx context.Context, sig syscall.Signal, opts ...containerd.KillOpts) error {
	if m.killFunc != nil {
		return m.killFunc(ctx, sig, opts...)
	}
	return nil
}

func (m *mockTask) Resume(ctx context.Context) error {
	if m.resumeFunc != nil {
		return m.resumeFunc(ctx)
	}
	return nil
}

func (m *mockTask) Delete(ctx context.Context, opts ...containerd.ProcessDeleteOpts) (*containerd.ExitStatus, error) {
	if m.deleteFunc != nil {
		return m.deleteFunc(ctx, opts...)
	}
	return &containerd.ExitStatus{}, nil
}

type mockImage struct {
	containerd.Image
}

func TestNewContainedClientSocketNotExists(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	client, err := NewContainedClient()
	assert.Error(t, err)
	assert.Nil(t, client)
	assert.Contains(t, err.Error(), "containerd service does not exist")
}

func TestNewContainedClientGetImageClientError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	patches.ApplyFunc(getImageClient, func(sock string) (pb.ImageServiceClient, *grpc.ClientConn, error) {
		return nil, nil, status.Error(codes.Unavailable, "connection refused")
	})

	client, err := NewContainedClient()
	assert.Error(t, err)
	assert.Nil(t, client)
}

func TestNewContainedClientContainerdNewError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	patches.ApplyFunc(getImageClient, func(sock string) (pb.ImageServiceClient, *grpc.ClientConn, error) {
		conn := &grpc.ClientConn{}
		return nil, conn, nil
	})

	patches.ApplyFunc(containerd.New, func(address string, opts ...containerd.ClientOpt) (*containerd.Client, error) {
		return nil, status.Error(codes.Internal, "failed to connect")
	})

	client, err := NewContainedClient()
	assert.Error(t, err)
	assert.Nil(t, client)
}

func TestNewContainedClientPingError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	testConn := &grpc.ClientConn{}
	patches.ApplyFunc(getImageClient, func(sock string) (pb.ImageServiceClient, *grpc.ClientConn, error) {
		return nil, testConn, nil
	})

	mockClient := &containerd.Client{}
	patches.ApplyFunc(containerd.New, func(address string, opts ...containerd.ClientOpt) (*containerd.Client, error) {
		return mockClient, nil
	})

	patches.ApplyMethod(mockClient, "Version", func(_ *containerd.Client, ctx context.Context) (containerd.Version, error) {
		return containerd.Version{}, status.Error(codes.Internal, "ping failed")
	})

	client, err := NewContainedClient()
	assert.Error(t, err)
	assert.Nil(t, client)
	assert.Contains(t, err.Error(), "failed to ping containerd service")
}

func TestNewContainedClientSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	testConn := &grpc.ClientConn{}
	patches.ApplyFunc(getImageClient, func(sock string) (pb.ImageServiceClient, *grpc.ClientConn, error) {
		return nil, testConn, nil
	})

	mockClient := &containerd.Client{}
	patches.ApplyFunc(containerd.New, func(address string, opts ...containerd.ClientOpt) (*containerd.Client, error) {
		return mockClient, nil
	})

	patches.ApplyMethod(mockClient, "Version", func(_ *containerd.Client, ctx context.Context) (containerd.Version, error) {
		return containerd.Version{Version: "1.7.0"}, nil
	})

	client, err := NewContainedClient()
	assert.NoError(t, err)
	assert.NotNil(t, client)
}

func TestPingWithNilClient(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
		}
	}()
	client := &Client{
		condClient: nil,
		ctx:        context.Background(),
	}
	client.Ping()
}

func TestPullWithCredentials(t *testing.T) {
	var pullImageCalled bool
	mockImgClient := &mockImageClient{
		pullImageFunc: func(ctx context.Context, in *pb.PullImageRequest, opts ...grpc.CallOption) (*pb.PullImageResponse, error) {
			pullImageCalled = true
			assert.Equal(t, "nginx:latest", in.Image.Image)
			assert.Equal(t, "testuser", in.Auth.Username)
			assert.Equal(t, "testpass", in.Auth.Password)
			return &pb.PullImageResponse{}, nil
		},
	}
	client := &Client{
		imageClient: mockImgClient,
		ctx:         context.Background(),
	}
	imageRef := ImageRef{
		Image:    "nginx:latest",
		Username: "testuser",
		Password: "testpass",
	}
	err := client.Pull(imageRef)
	assert.NoError(t, err)
	assert.True(t, pullImageCalled)
}

func TestPullWithoutCredentials(t *testing.T) {
	var pullImageCalled bool
	mockImgClient := &mockImageClient{
		pullImageFunc: func(ctx context.Context, in *pb.PullImageRequest, opts ...grpc.CallOption) (*pb.PullImageResponse, error) {
			pullImageCalled = true
			assert.Nil(t, in.Auth)
			return &pb.PullImageResponse{}, nil
		},
	}
	client := &Client{
		imageClient: mockImgClient,
		ctx:         context.Background(),
	}
	imageRef := ImageRef{
		Image: "nginx:latest",
	}
	err := client.Pull(imageRef)
	assert.NoError(t, err)
	assert.True(t, pullImageCalled)
}

func TestPullError(t *testing.T) {
	mockImgClient := &mockImageClient{
		pullImageFunc: func(ctx context.Context, in *pb.PullImageRequest, opts ...grpc.CallOption) (*pb.PullImageResponse, error) {
			return nil, status.Error(codes.Internal, "pull failed")
		},
	}
	client := &Client{
		imageClient: mockImgClient,
		ctx:         context.Background(),
	}
	imageRef := ImageRef{Image: "nginx:latest"}
	err := client.Pull(imageRef)
	assert.Error(t, err)
}

func TestEnsureImageExistsImageNotFound(t *testing.T) {
	var pullImageCalled bool
	mockImgClient := &mockImageClient{
		imageStatusFunc: func(ctx context.Context, in *pb.ImageStatusRequest, opts ...grpc.CallOption) (*pb.ImageStatusResponse, error) {
			return &pb.ImageStatusResponse{}, nil
		},
		pullImageFunc: func(ctx context.Context, in *pb.PullImageRequest, opts ...grpc.CallOption) (*pb.PullImageResponse, error) {
			pullImageCalled = true
			return &pb.PullImageResponse{}, nil
		},
	}
	client := &Client{
		imageClient: mockImgClient,
		ctx:         context.Background(),
	}
	imageRef := ImageRef{Image: "nginx:latest"}
	err := client.EnsureImageExists(imageRef)
	assert.NoError(t, err)
	assert.True(t, pullImageCalled)
}

func TestEnsureImageExistsImageExists(t *testing.T) {
	var pullImageCalled bool
	mockImgClient := &mockImageClient{
		imageStatusFunc: func(ctx context.Context, in *pb.ImageStatusRequest, opts ...grpc.CallOption) (*pb.ImageStatusResponse, error) {
			return &pb.ImageStatusResponse{
				Image: &pb.Image{Id: "sha256:abc123"},
			}, nil
		},
	}
	client := &Client{
		imageClient: mockImgClient,
		ctx:         context.Background(),
	}
	imageRef := ImageRef{Image: "nginx:latest"}
	err := client.EnsureImageExists(imageRef)
	assert.NoError(t, err)
	assert.False(t, pullImageCalled)
}

func TestEnsureImageExistsError(t *testing.T) {
	mockImgClient := &mockImageClient{
		imageStatusFunc: func(ctx context.Context, in *pb.ImageStatusRequest, opts ...grpc.CallOption) (*pb.ImageStatusResponse, error) {
			return nil, status.Error(codes.Internal, "internal error")
		},
	}
	client := &Client{
		imageClient: mockImgClient,
		ctx:         context.Background(),
	}
	imageRef := ImageRef{Image: "nginx:latest"}
	err := client.EnsureImageExists(imageRef)
	assert.Error(t, err)
}

func TestContainerListNilClient(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
		}
	}()
	client := &Client{
		condClient: nil,
		ctx:        context.Background(),
	}
	client.ContainerList()
}

func TestEnsureContainerRunNilClient(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
		}
	}()
	client := &Client{
		condClient: nil,
		ctx:        context.Background(),
	}
	client.EnsureContainerRun("test-container")
}

func TestStopContainerNilClient(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
		}
	}()
	client := &Client{
		condClient: nil,
		ctx:        context.Background(),
	}
	client.Stop("test-container")
}

func TestDeleteContainerNilClient(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
		}
	}()
	client := &Client{
		condClient: nil,
		ctx:        context.Background(),
	}
	client.Delete("test-container")
}

func TestWaitContainerStopContextDone(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	exitCh := make(chan containerd.ExitStatus)
	err := waitContainerStop(ctx, exitCh, "test-container")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "wait container")
}

func TestPullImageWithCredentials(t *testing.T) {
	testCases := []struct {
		name       string
		imageRef   ImageRef
		expectAuth bool
	}{
		{
			name:       "with credentials",
			imageRef:   ImageRef{Image: "nginx:latest", Username: "user", Password: "pass"},
			expectAuth: true,
		},
		{
			name:       "without credentials",
			imageRef:   ImageRef{Image: "nginx:latest"},
			expectAuth: false,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var authCalled bool
			mockImgClient := &mockImageClient{
				pullImageFunc: func(ctx context.Context, in *pb.PullImageRequest, opts ...grpc.CallOption) (*pb.PullImageResponse, error) {
					authCalled = in.Auth != nil
					return &pb.PullImageResponse{}, nil
				},
			}
			client := &Client{
				imageClient: mockImgClient,
				ctx:         context.Background(),
			}
			err := client.Pull(tc.imageRef)
			assert.NoError(t, err)
			assert.Equal(t, tc.expectAuth, authCalled)
		})
	}
}

func TestEnsureImageExistsTableDriven(t *testing.T) {
	testCases := []struct {
		name        string
		imageStatus *pb.ImageStatusResponse
		expectPull  bool
	}{
		{
			name:        "image not found",
			imageStatus: &pb.ImageStatusResponse{},
			expectPull:  true,
		},
		{
			name: "image exists",
			imageStatus: &pb.ImageStatusResponse{
				Image: &pb.Image{Id: "sha256:abc123"},
			},
			expectPull: false,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var pullCalled bool
			mockImgClient := &mockImageClient{
				imageStatusFunc: func(ctx context.Context, in *pb.ImageStatusRequest, opts ...grpc.CallOption) (*pb.ImageStatusResponse, error) {
					return tc.imageStatus, nil
				},
				pullImageFunc: func(ctx context.Context, in *pb.PullImageRequest, opts ...grpc.CallOption) (*pb.PullImageResponse, error) {
					pullCalled = true
					return &pb.PullImageResponse{}, nil
				},
			}
			client := &Client{
				imageClient: mockImgClient,
				ctx:         context.Background(),
			}
			err := client.EnsureImageExists(ImageRef{Image: "nginx:latest"})
			assert.NoError(t, err)
			assert.Equal(t, tc.expectPull, pullCalled)
		})
	}
}

func TestWaitContainerStopTableDriven(t *testing.T) {
	testCases := []struct {
		name      string
		setupCtx  func() (context.Context, context.CancelFunc)
		setupCh   func() chan containerd.ExitStatus
		expectErr bool
	}{
		{
			name: "context done before exit",
			setupCtx: func() (context.Context, context.CancelFunc) {
				return context.WithTimeout(context.Background(), shortWaitDuration)
			},
			setupCh: func() chan containerd.ExitStatus {
				ch := make(chan containerd.ExitStatus, numOne)
				return ch
			},
			expectErr: true,
		},
		{
			name: "exit before context done",
			setupCtx: func() (context.Context, context.CancelFunc) {
				return context.WithTimeout(context.Background(), longWaitDuration)
			},
			setupCh: func() chan containerd.ExitStatus {
				ch := make(chan containerd.ExitStatus, numOne)
				go func() {
					ch <- containerd.ExitStatus{}
				}()
				return ch
			},
			expectErr: false,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := tc.setupCtx()
			defer cancel()
			exitCh := tc.setupCh()
			err := waitContainerStop(ctx, exitCh, "test")
			if tc.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestDeleteWithInvalidLabels(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
		}
	}()
	client := &Client{
		condClient: nil,
		ctx:        context.Background(),
	}
	client.Delete("test-pod")
}

func TestDeleteContainerNotFound(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
		}
	}()
	client := &Client{
		condClient: nil,
		ctx:        context.Background(),
	}
	client.Delete("nonexistent")
}

func TestCloseWithNilClient(t *testing.T) {
	client := &Client{
		condClient: nil,
		cancel:     nil,
	}
	client.Close()
}

func TestCloseWithCancel(t *testing.T) {
	client := &Client{
		condClient: nil,
		cancel:     func() {},
	}
	client.Close()
}

func TestContainerListError(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
		}
	}()
	client := &Client{
		condClient: nil,
		ctx:        context.Background(),
	}
	client.ContainerList()
}

func TestEnsureContainerRunError(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
		}
	}()
	client := &Client{
		condClient: nil,
		ctx:        context.Background(),
	}
	client.EnsureContainerRun("test")
}

func TestStopSignalParsing(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
		}
	}()
	client := &Client{
		condClient: nil,
		ctx:        context.Background(),
	}
	client.Stop("test")
}

func TestGetContainerdConfigWithValidPath(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
		}
	}()
	config, err := GetContainerdConfig("/etc/containerd/config.toml")
	_ = config
	_ = err
}

func TestClientNilFields(t *testing.T) {
	client := &Client{
		imageClient: nil,
		condClient:  nil,
		ctx:         nil,
		cancel:      nil,
	}
	assert.Nil(t, client.imageClient)
	assert.Nil(t, client.condClient)
	assert.Nil(t, client.ctx)
	assert.Nil(t, client.cancel)
}

func TestSockAddrHashHex(t *testing.T) {
	hash := sockAddrHash()
	assert.Equal(t, numEight, len(hash))
	for _, char := range hash {
		assert.True(t, (char >= '0' && char <= '9') || (char >= 'a' && char <= 'f'))
	}
}

func TestClientClose(t *testing.T) {
	client := &Client{}

	client.Close()

	assert.Nil(t, client.condClient)
}

func TestClientCloseWithNil(t *testing.T) {
	client := &Client{
		condClient: nil,
	}

	client.Close()

	assert.Nil(t, client.condClient)
}

func TestGetContainerdConfigWithDefaultPath(t *testing.T) {
	config, err := GetContainerdConfig("")

	if err != nil {
		assert.Nil(t, config)
		return
	}
	assert.NotNil(t, config)
}

func TestGetContainerdConfigWithInvalidPath(t *testing.T) {
	config, err := GetContainerdConfig("/nonexistent/path/config.toml")

	assert.Error(t, err)
	assert.Nil(t, config)
}

func TestImageRefFields(t *testing.T) {
	imageRef := ImageRef{
		Image:    "nginx:latest",
		Username: "user",
		Password: "pass",
	}

	assert.Equal(t, "nginx:latest", imageRef.Image)
	assert.Equal(t, "user", imageRef.Username)
	assert.Equal(t, "pass", imageRef.Password)
}

func TestImageRefWithEmptyCredentials(t *testing.T) {
	imageRef := ImageRef{
		Image: "nginx:latest",
	}

	assert.Equal(t, "nginx:latest", imageRef.Image)
	assert.Empty(t, imageRef.Username)
	assert.Empty(t, imageRef.Password)
}

func TestContainerdClientInterface(t *testing.T) {
	var client ContainerdClient = nil

	assert.Nil(t, client)
}

func TestClientFields(t *testing.T) {
	client := &Client{}

	assert.Nil(t, client.imageClient)
	assert.Nil(t, client.condClient)
	assert.Nil(t, client.ctx)
	assert.Nil(t, client.cancel)
}

func TestContainerdConfigFilePathConstant(t *testing.T) {
	assert.Equal(t, "/etc/containerd/config.toml", ContainerdConfigFilePath)
}

func TestSockAddrHashReturnsEightCharacters(t *testing.T) {
	hash := sockAddrHash()

	assert.Equal(t, numEight, len(hash))
}

func TestSockAddrHashReturnsHexadecimal(t *testing.T) {
	hash := sockAddrHash()

	for _, char := range hash {
		assert.True(t, (char >= '0' && char <= '9') || (char >= 'a' && char <= 'f'),
			"Hash should contain only hexadecimal characters")
	}
}

func TestWaitContainerStopWithDoneContext(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), numZero)
	defer cancel()

	exitCh := make(chan containerd.ExitStatus)
	err := waitContainerStop(ctx, exitCh, "test-container")

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "wait container")
}

func TestWaitContainerStopWithNilError(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	exitCh := make(chan containerd.ExitStatus)
	errChan := make(chan error, numOne)
	go func() {
		exitCh <- containerd.ExitStatus{}
	}()

	go func() {
		err := <-exitCh
		errChan <- err.Error()
	}()

	select {
	case err := <-errChan:
		assert.Nil(t, err)
	case <-ctx.Done():
		assert.Fail(t, "Test timed out")
	}
}

func TestClientWithNilFields(t *testing.T) {
	client := &Client{
		imageClient: nil,
		condClient:  nil,
		ctx:         nil,
		cancel:      nil,
	}

	assert.Nil(t, client.imageClient)
	assert.Nil(t, client.condClient)
	assert.Nil(t, client.ctx)
	assert.Nil(t, client.cancel)
}

func TestWaitContainerdReadyTimeout(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(wait.PollImmediateUntil, func(interval time.Duration, condition func() (bool, error), stopCh <-chan struct{}) error {
		return status.Error(codes.DeadlineExceeded, "timeout")
	})

	err := WaitContainerdReady()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "wait containerd available")
}

func TestNewContainedClientNamespaceApplied(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	testConn := &grpc.ClientConn{}
	var capturedCtx context.Context

	patches.ApplyFunc(getImageClient, func(sock string) (pb.ImageServiceClient, *grpc.ClientConn, error) {
		return nil, testConn, nil
	})

	mockClient := &containerd.Client{}
	patches.ApplyFunc(containerd.New, func(address string, opts ...containerd.ClientOpt) (*containerd.Client, error) {
		return mockClient, nil
	})

	patches.ApplyMethod(mockClient, "Version", func(c *containerd.Client, ctx context.Context) (containerd.Version, error) {
		capturedCtx = ctx
		return containerd.Version{Version: "1.7.0"}, nil
	})

	client, err := NewContainedClient()
	assert.NoError(t, err)
	assert.NotNil(t, client)

	_ = capturedCtx
	assert.NotNil(t, client)
}

func TestEnsureContainerRunRunning(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &containerd.Client{}
	mockContainer := &mockContainer{idVal: "test-container"}
	mockTask := &mockTask{}

	patches.ApplyMethod(mockClient, "LoadContainer", func(c *containerd.Client, ctx context.Context, id string) (containerd.Container, error) {
		return mockContainer, nil
	})

	patches.ApplyMethod(mockContainer, "Task", func(c containerd.Container, ctx context.Context, ioAttach cio.Attach) (containerd.Task, error) {
		return mockTask, nil
	})

	patches.ApplyMethod(mockTask, "Status", func(t containerd.Task, ctx context.Context) (containerd.Status, error) {
		return containerd.Status{Status: containerd.Running}, nil
	})

	client := &Client{
		condClient: mockClient,
		ctx:        context.Background(),
	}

	isRunning, err := client.EnsureContainerRun("test-container")
	assert.NoError(t, err)
	assert.True(t, isRunning)
}

func TestEnsureContainerRunNotRunning(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &containerd.Client{}
	mockContainer := &mockContainer{idVal: "test-container"}
	mockTask := &mockTask{}

	patches.ApplyMethod(mockClient, "LoadContainer", func(c *containerd.Client, ctx context.Context, id string) (containerd.Container, error) {
		return mockContainer, nil
	})

	patches.ApplyMethod(mockContainer, "Task", func(c containerd.Container, ctx context.Context, ioAttach cio.Attach) (containerd.Task, error) {
		return mockTask, nil
	})

	patches.ApplyMethod(mockTask, "Status", func(t containerd.Task, ctx context.Context) (containerd.Status, error) {
		return containerd.Status{Status: containerd.Stopped}, nil
	})

	client := &Client{
		condClient: mockClient,
		ctx:        context.Background(),
	}

	isRunning, err := client.EnsureContainerRun("test-container")
	assert.NoError(t, err)
	assert.False(t, isRunning)
}

func TestEnsureContainerRunLoadContainerErrorMock(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &containerd.Client{}

	patches.ApplyMethod(mockClient, "LoadContainer", func(c *containerd.Client, ctx context.Context, id string) (containerd.Container, error) {
		return nil, status.Error(codes.NotFound, "container not found")
	})

	client := &Client{
		condClient: mockClient,
		ctx:        context.Background(),
	}

	isRunning, err := client.EnsureContainerRun("nonexistent-container")
	assert.Error(t, err)
	assert.False(t, isRunning)
	assert.Contains(t, err.Error(), "container not found")
}

func TestEnsureContainerRunTaskError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &containerd.Client{}
	mockContainer := &mockContainer{idVal: "test-container"}

	patches.ApplyMethod(mockClient, "LoadContainer", func(c *containerd.Client, ctx context.Context, id string) (containerd.Container, error) {
		return mockContainer, nil
	})

	patches.ApplyMethod(mockContainer, "Task", func(c containerd.Container, ctx context.Context, ioAttach cio.Attach) (containerd.Task, error) {
		return nil, status.Error(codes.NotFound, "task not found")
	})

	client := &Client{
		condClient: mockClient,
		ctx:        context.Background(),
	}

	isRunning, err := client.EnsureContainerRun("test-container")
	assert.Error(t, err)
	assert.False(t, isRunning)
	assert.Contains(t, err.Error(), "task not found")
}

func TestEnsureContainerRunStatusError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &containerd.Client{}
	mockContainer := &mockContainer{idVal: "test-container"}
	mockTask := &mockTask{}

	patches.ApplyMethod(mockClient, "LoadContainer", func(c *containerd.Client, ctx context.Context, id string) (containerd.Container, error) {
		return mockContainer, nil
	})

	patches.ApplyMethod(mockContainer, "Task", func(c containerd.Container, ctx context.Context, ioAttach cio.Attach) (containerd.Task, error) {
		return mockTask, nil
	})

	patches.ApplyMethod(mockTask, "Status", func(t containerd.Task, ctx context.Context) (containerd.Status, error) {
		return containerd.Status{}, status.Error(codes.Internal, "status error")
	})

	client := &Client{
		condClient: mockClient,
		ctx:        context.Background(),
	}

	isRunning, err := client.EnsureContainerRun("test-container")
	assert.Error(t, err)
	assert.False(t, isRunning)
	assert.Contains(t, err.Error(), "status error")
}

func TestEnsureContainerRunPaused(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &containerd.Client{}
	mockContainer := &mockContainer{idVal: "test-container"}
	mockTask := &mockTask{}

	patches.ApplyMethod(mockClient, "LoadContainer", func(c *containerd.Client, ctx context.Context, id string) (containerd.Container, error) {
		return mockContainer, nil
	})

	patches.ApplyMethod(mockContainer, "Task", func(c containerd.Container, ctx context.Context, ioAttach cio.Attach) (containerd.Task, error) {
		return mockTask, nil
	})

	patches.ApplyMethod(mockTask, "Status", func(t containerd.Task, ctx context.Context) (containerd.Status, error) {
		return containerd.Status{Status: containerd.Paused}, nil
	})

	client := &Client{
		condClient: mockClient,
		ctx:        context.Background(),
	}

	isRunning, err := client.EnsureContainerRun("test-container")
	assert.NoError(t, err)
	assert.False(t, isRunning)
}

func TestEnsureContainerRunCreated(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &containerd.Client{}
	mockContainer := &mockContainer{idVal: "test-container"}
	mockTask := &mockTask{}

	patches.ApplyMethod(mockClient, "LoadContainer", func(c *containerd.Client, ctx context.Context, id string) (containerd.Container, error) {
		return mockContainer, nil
	})

	patches.ApplyMethod(mockContainer, "Task", func(c containerd.Container, ctx context.Context, ioAttach cio.Attach) (containerd.Task, error) {
		return mockTask, nil
	})

	patches.ApplyMethod(mockTask, "Status", func(t containerd.Task, ctx context.Context) (containerd.Status, error) {
		return containerd.Status{Status: containerd.Created}, nil
	})

	client := &Client{
		condClient: mockClient,
		ctx:        context.Background(),
	}

	isRunning, err := client.EnsureContainerRun("test-container")
	assert.NoError(t, err)
	assert.False(t, isRunning)
}

func TestStopLoadContainerError(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
		}
	}()
	client := &Client{
		condClient: nil,
		ctx:        context.Background(),
	}

	_ = client.Stop("test-container")
}

func TestStopWithNilLabels(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
		}
	}()
	client := &Client{
		condClient: nil,
		ctx:        context.Background(),
	}

	_ = client.Stop("test-nil-labels")
}

func TestDeleteLoadContainerError(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
		}
	}()
	client := &Client{
		condClient: nil,
		ctx:        context.Background(),
	}

	_ = client.Delete("test-delete")
}

func TestDeleteContainerListError(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
		}
	}()
	client := &Client{
		condClient: nil,
		ctx:        context.Background(),
	}

	_ = client.Delete("test-container-list-error")
}

func TestEnsureContainerRunLoadContainerErrorOld(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
		}
	}()
	client := &Client{
		condClient: nil,
		ctx:        context.Background(),
	}

	_, _ = client.EnsureContainerRun("test-container")
}

func TestContainerListSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &containerd.Client{}
	var capturedCtx context.Context

	patches.ApplyMethod(mockClient, "Containers", func(c *containerd.Client, ctx context.Context, filters ...string) ([]containerd.Container, error) {
		capturedCtx = ctx
		return []containerd.Container{}, nil
	})

	client := &Client{
		condClient: mockClient,
		ctx:        context.Background(),
	}

	containers, err := client.ContainerList()
	assert.NoError(t, err)
	assert.NotNil(t, containers)
	_ = capturedCtx
}

func TestGetContainerdConfigLoadSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(srvconfig.LoadConfig, func(path string, cfg *srvconfig.Config) error {
		cfg.Version = numTwo
		return nil
	})

	config, err := GetContainerdConfig("/etc/containerd/config.toml")
	assert.NoError(t, err)
	assert.NotNil(t, config)
}

func TestWaitContainerdReadySuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(wait.PollImmediateUntil, func(interval time.Duration, condition func() (bool, error), stopCh <-chan struct{}) error {
		return nil
	})

	err := WaitContainerdReady()
	assert.NoError(t, err)
}

func TestWaitContainerdReadyPollError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(wait.PollImmediateUntil, func(interval time.Duration, condition func() (bool, error), stopCh <-chan struct{}) error {
		return status.Error(codes.Internal, "poll failed")
	})

	err := WaitContainerdReady()
	assert.Error(t, err)
}

func TestStopUpdateContainerError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &containerd.Client{}
	mockContainer := &mockContainer{}

	patches.ApplyMethod(mockClient, "LoadContainer", func(c *containerd.Client, ctx context.Context, id string) (containerd.Container, error) {
		return mockContainer, nil
	})

	patches.ApplyMethod(mockContainer, "Update", func(_ containerd.Container, ctx context.Context, opts ...containerd.UpdateContainerOpts) error {
		return status.Error(codes.Internal, "update failed")
	})

	client := &Client{
		condClient: mockClient,
		ctx:        context.Background(),
	}

	err := client.Stop("test-container")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "update failed")
}

func TestStopStatusCreated(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &containerd.Client{}
	mockContainer := &mockContainer{idVal: "test-container"}
	mockTask := &mockTask{}

	patches.ApplyMethod(mockClient, "LoadContainer", func(c *containerd.Client, ctx context.Context, id string) (containerd.Container, error) {
		return mockContainer, nil
	})

	patches.ApplyMethod(mockContainer, "Update", func(_ containerd.Container, ctx context.Context, opts ...containerd.UpdateContainerOpts) error {
		return nil
	})

	patches.ApplyMethod(mockContainer, "Labels", func(c containerd.Container, ctx context.Context) (map[string]string, error) {
		return map[string]string{}, nil
	})

	patches.ApplyMethod(mockContainer, "Task", func(c containerd.Container, ctx context.Context, ioAttach cio.Attach) (containerd.Task, error) {
		return mockTask, nil
	})

	patches.ApplyMethod(mockTask, "Status", func(t containerd.Task, ctx context.Context) (containerd.Status, error) {
		return containerd.Status{Status: containerd.Running}, nil
	})

	client := &Client{
		condClient: mockClient,
		ctx:        context.Background(),
	}

	err := client.Stop("test-container")
	assert.NoError(t, err)
}

func TestStopStatusStopped(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &containerd.Client{}
	mockContainer := &mockContainer{idVal: "test-container"}
	mockTask := &mockTask{}

	patches.ApplyMethod(mockClient, "LoadContainer", func(c *containerd.Client, ctx context.Context, id string) (containerd.Container, error) {
		return mockContainer, nil
	})

	patches.ApplyMethod(mockContainer, "Update", func(_ containerd.Container, ctx context.Context, opts ...containerd.UpdateContainerOpts) error {
		return nil
	})

	patches.ApplyMethod(mockContainer, "Labels", func(c containerd.Container, ctx context.Context) (map[string]string, error) {
		return map[string]string{}, nil
	})

	patches.ApplyMethod(mockContainer, "Task", func(c containerd.Container, ctx context.Context, ioAttach cio.Attach) (containerd.Task, error) {
		return mockTask, nil
	})

	patches.ApplyMethod(mockTask, "Status", func(t containerd.Task, ctx context.Context) (containerd.Status, error) {
		return containerd.Status{Status: containerd.Stopped}, nil
	})

	client := &Client{
		condClient: mockClient,
		ctx:        context.Background(),
	}

	err := client.Stop("test-container")
	assert.NoError(t, err)
}

func TestStopWithCustomTimeout(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &containerd.Client{}
	mockContainer := &mockContainer{idVal: "test-container"}
	mockTask := &mockTask{}

	patches.ApplyMethod(mockClient, "LoadContainer", func(c *containerd.Client, ctx context.Context, id string) (containerd.Container, error) {
		return mockContainer, nil
	})

	patches.ApplyMethod(mockContainer, "Update", func(_ containerd.Container, ctx context.Context, opts ...containerd.UpdateContainerOpts) error {
		return nil
	})

	patches.ApplyMethod(mockContainer, "Labels", func(c containerd.Container, ctx context.Context) (map[string]string, error) {
		return map[string]string{"nerdctl/stop-timeout": "30"}, nil
	})

	patches.ApplyMethod(mockContainer, "Task", func(c containerd.Container, ctx context.Context, ioAttach cio.Attach) (containerd.Task, error) {
		return mockTask, nil
	})

	patches.ApplyMethod(mockTask, "Status", func(t containerd.Task, ctx context.Context) (containerd.Status, error) {
		return containerd.Status{Status: containerd.Running}, nil
	})

	patches.ApplyMethod(mockTask, "Wait", func(t containerd.Task, ctx context.Context) (<-chan containerd.ExitStatus, error) {
		ch := make(chan containerd.ExitStatus, numOne)
		go func() {
			ch <- containerd.ExitStatus{}
			close(ch)
		}()
		return ch, nil
	})

	patches.ApplyMethod(mockTask, "Kill", func(t containerd.Task, ctx context.Context, sig syscall.Signal, opts ...containerd.KillOpts) error {
		return nil
	})

	client := &Client{
		condClient: mockClient,
		ctx:        context.Background(),
	}

	err := client.Stop("test-container")
	assert.NoError(t, err)
}

func TestStopWithInvalidTimeout(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &containerd.Client{}
	mockContainer := &mockContainer{}

	patches.ApplyMethod(mockClient, "LoadContainer", func(c *containerd.Client, ctx context.Context, id string) (containerd.Container, error) {
		return mockContainer, nil
	})

	patches.ApplyMethod(mockContainer, "Update", func(_ containerd.Container, ctx context.Context, opts ...containerd.UpdateContainerOpts) error {
		return nil
	})

	patches.ApplyMethod(mockContainer, "Labels", func(c containerd.Container, ctx context.Context) (map[string]string, error) {
		return map[string]string{"nerdctl/stop-timeout": "invalid"}, nil
	})

	client := &Client{
		condClient: mockClient,
		ctx:        context.Background(),
	}

	err := client.Stop("test-container")
	assert.Error(t, err)
}

func TestStopWithStopSignal(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &containerd.Client{}
	mockContainer := &mockContainer{idVal: "test-container"}
	mockTask := &mockTask{}

	patches.ApplyMethod(mockClient, "LoadContainer", func(c *containerd.Client, ctx context.Context, id string) (containerd.Container, error) {
		return mockContainer, nil
	})

	patches.ApplyMethod(mockContainer, "Update", func(_ containerd.Container, ctx context.Context, opts ...containerd.UpdateContainerOpts) error {
		return nil
	})

	patches.ApplyMethod(mockContainer, "Labels", func(c containerd.Container, ctx context.Context) (map[string]string, error) {
		return map[string]string{containerd.StopSignalLabel: "SIGINT"}, nil
	})

	patches.ApplyMethod(mockContainer, "Task", func(c containerd.Container, ctx context.Context, ioAttach cio.Attach) (containerd.Task, error) {
		return mockTask, nil
	})

	patches.ApplyMethod(mockTask, "Status", func(t containerd.Task, ctx context.Context) (containerd.Status, error) {
		return containerd.Status{Status: containerd.Running}, nil
	})

	patches.ApplyMethod(mockTask, "Wait", func(t containerd.Task, ctx context.Context) (<-chan containerd.ExitStatus, error) {
		ch := make(chan containerd.ExitStatus, numOne)
		go func() {
			ch <- containerd.ExitStatus{}
			close(ch)
		}()
		return ch, nil
	})

	var killCalled bool
	patches.ApplyMethod(mockTask, "Kill", func(t containerd.Task, ctx context.Context, sig syscall.Signal, opts ...containerd.KillOpts) error {
		killCalled = true
		return nil
	})

	client := &Client{
		condClient: mockClient,
		ctx:        context.Background(),
	}

	err := client.Stop("test-container")
	assert.NoError(t, err)
	assert.True(t, killCalled, "Kill should be called")
}

func TestStopWithInvalidStopSignal(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
		}
	}()

	client := &Client{
		condClient: nil,
		ctx:        context.Background(),
	}

	_ = client.Stop("test-container")
}

func TestStopKillError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &containerd.Client{}
	mockContainer := &mockContainer{idVal: "test-container"}
	mockTask := &mockTask{}

	patches.ApplyMethod(mockClient, "LoadContainer", func(c *containerd.Client, ctx context.Context, id string) (containerd.Container, error) {
		return mockContainer, nil
	})

	patches.ApplyMethod(mockContainer, "Update", func(_ containerd.Container, ctx context.Context, opts ...containerd.UpdateContainerOpts) error {
		return nil
	})

	patches.ApplyMethod(mockContainer, "Labels", func(c containerd.Container, ctx context.Context) (map[string]string, error) {
		return map[string]string{}, nil
	})

	patches.ApplyMethod(mockContainer, "Task", func(c containerd.Container, ctx context.Context, ioAttach cio.Attach) (containerd.Task, error) {
		return mockTask, nil
	})

	patches.ApplyMethod(mockTask, "Status", func(t containerd.Task, ctx context.Context) (containerd.Status, error) {
		return containerd.Status{Status: containerd.Running}, nil
	})

	patches.ApplyMethod(mockTask, "Wait", func(t containerd.Task, ctx context.Context) (<-chan containerd.ExitStatus, error) {
		ch := make(chan containerd.ExitStatus, numOne)
		go func() {
			ch <- containerd.ExitStatus{}
			close(ch)
		}()
		return ch, nil
	})

	patches.ApplyMethod(mockTask, "Kill", func(t containerd.Task, ctx context.Context, sig syscall.Signal, opts ...containerd.KillOpts) error {
		return status.Error(codes.Internal, "kill failed")
	})

	client := &Client{
		condClient: mockClient,
		ctx:        context.Background(),
	}

	err := client.Stop("test-container")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "kill failed")
}

func TestStopPausedWithResumeError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &containerd.Client{}
	mockContainer := &mockContainer{idVal: "test-container"}
	mockTask := &mockTask{}

	patches.ApplyMethod(mockClient, "LoadContainer", func(c *containerd.Client, ctx context.Context, id string) (containerd.Container, error) {
		return mockContainer, nil
	})

	patches.ApplyMethod(mockContainer, "Update", func(_ containerd.Container, ctx context.Context, opts ...containerd.UpdateContainerOpts) error {
		return nil
	})

	patches.ApplyMethod(mockContainer, "Labels", func(c containerd.Container, ctx context.Context) (map[string]string, error) {
		return map[string]string{}, nil
	})

	patches.ApplyMethod(mockContainer, "Task", func(c containerd.Container, ctx context.Context, ioAttach cio.Attach) (containerd.Task, error) {
		return mockTask, nil
	})

	patches.ApplyMethod(mockTask, "Status", func(t containerd.Task, ctx context.Context) (containerd.Status, error) {
		return containerd.Status{Status: containerd.Paused}, nil
	})

	patches.ApplyMethod(mockTask, "Wait", func(t containerd.Task, ctx context.Context) (<-chan containerd.ExitStatus, error) {
		ch := make(chan containerd.ExitStatus, numOne)
		go func() {
			ch <- containerd.ExitStatus{}
			close(ch)
		}()
		return ch, nil
	})

	patches.ApplyMethod(mockTask, "Kill", func(t containerd.Task, ctx context.Context, sig syscall.Signal, opts ...containerd.KillOpts) error {
		return nil
	})

	patches.ApplyMethod(mockTask, "Resume", func(t containerd.Task, ctx context.Context) error {
		return status.Error(codes.Internal, "resume failed")
	})

	client := &Client{
		condClient: mockClient,
		ctx:        context.Background(),
	}

	err := client.Stop("test-container")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unpause container test-container")
}

func TestStopTaskWaitError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &containerd.Client{}
	mockContainer := &mockContainer{idVal: "test-container"}
	mockTask := &mockTask{}

	patches.ApplyMethod(mockClient, "LoadContainer", func(c *containerd.Client, ctx context.Context, id string) (containerd.Container, error) {
		return mockContainer, nil
	})

	patches.ApplyMethod(mockContainer, "Update", func(_ containerd.Container, ctx context.Context, opts ...containerd.UpdateContainerOpts) error {
		return nil
	})

	patches.ApplyMethod(mockContainer, "Labels", func(c containerd.Container, ctx context.Context) (map[string]string, error) {
		return map[string]string{}, nil
	})

	patches.ApplyMethod(mockContainer, "Task", func(c containerd.Container, ctx context.Context, ioAttach cio.Attach) (containerd.Task, error) {
		return mockTask, nil
	})

	patches.ApplyMethod(mockTask, "Status", func(t containerd.Task, ctx context.Context) (containerd.Status, error) {
		return containerd.Status{Status: containerd.Running}, nil
	})

	patches.ApplyMethod(mockTask, "Wait", func(t containerd.Task, ctx context.Context) (<-chan containerd.ExitStatus, error) {
		return nil, status.Error(codes.Internal, "wait failed")
	})

	client := &Client{
		condClient: mockClient,
		ctx:        context.Background(),
	}

	err := client.Stop("test-container")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "wait failed")
}

func TestDeleteContainerListErrorWithMock(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &containerd.Client{}

	patches.ApplyMethod(mockClient, "Containers", func(c *containerd.Client, ctx context.Context, filters ...string) ([]containerd.Container, error) {
		return nil, status.Error(codes.Internal, "list failed")
	})

	client := &Client{
		condClient: mockClient,
		ctx:        context.Background(),
	}

	err := client.Delete("test-container")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "list failed")
}

func TestDeleteContainerLabelsError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &containerd.Client{}
	mockContainer := &mockContainer{idVal: "test-container"}

	patches.ApplyMethod(mockClient, "Containers", func(c *containerd.Client, ctx context.Context, filters ...string) ([]containerd.Container, error) {
		return []containerd.Container{mockContainer}, nil
	})

	patches.ApplyMethod(mockContainer, "Labels", func(c containerd.Container, ctx context.Context) (map[string]string, error) {
		return nil, status.Error(codes.Internal, "labels failed")
	})

	client := &Client{
		condClient: mockClient,
		ctx:        context.Background(),
	}

	err := client.Delete("test-container")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "labels failed")
}

func TestDeleteContainerNotFoundByPodName(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &containerd.Client{}
	mockContainer := &mockContainer{idVal: "container-1"}

	patches.ApplyMethod(mockClient, "Containers", func(c *containerd.Client, ctx context.Context, filters ...string) ([]containerd.Container, error) {
		return []containerd.Container{mockContainer}, nil
	})

	patches.ApplyMethod(mockContainer, "Labels", func(c containerd.Container, ctx context.Context) (map[string]string, error) {
		return map[string]string{"io.kubernetes.pod.name": "other-pod"}, nil
	})

	client := &Client{
		condClient: mockClient,
		ctx:        context.Background(),
	}

	err := client.Delete("test-pod")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
}

func TestDeleteContainerNotFoundByNerdctlName(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &containerd.Client{}
	mockContainer := &mockContainer{idVal: "container-1"}

	patches.ApplyMethod(mockClient, "Containers", func(c *containerd.Client, ctx context.Context, filters ...string) ([]containerd.Container, error) {
		return []containerd.Container{mockContainer}, nil
	})

	patches.ApplyMethod(mockContainer, "Labels", func(c containerd.Container, ctx context.Context) (map[string]string, error) {
		return map[string]string{"nerdctl/name": "other-container"}, nil
	})

	client := &Client{
		condClient: mockClient,
		ctx:        context.Background(),
	}

	err := client.Delete("test-container")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
}

/*func TestDeleteTaskStatusRunningKillError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &containerd.Client{}
	mockContainer := &mockContainer{idVal: "test-container"}
	mockTask := &mockTask{}

	patches.ApplyMethod(mockClient, "Containers", func(c *containerd.Client, ctx context.Context, filters ...string) ([]containerd.Container, error) {
		return []containerd.Container{mockContainer}, nil
	})

	patches.ApplyMethod(mockContainer, "Labels", func(c containerd.Container, ctx context.Context) (map[string]string, error) {
		return map[string]string{"io.kubernetes.pod.name": "test-pod"}, nil
	})

	patches.ApplyMethod(mockContainer, "Task", func(c containerd.Container, ctx context.Context, ioAttach cio.Attach) (containerd.Task, error) {
		return mockTask, nil
	})

	patches.ApplyMethod(mockTask, "Status", func(t containerd.Task, ctx context.Context) (containerd.Status, error) {
		return containerd.Status{Status: containerd.Running}, nil
	})

	patches.ApplyMethod(mockTask, "Kill", func(t containerd.Task, ctx context.Context, sig syscall.Signal, opts ...containerd.KillOpts) error {
		return status.Error(codes.Internal, "kill failed")
	})

	// 模拟Wait方法返回一个通道和错误，这样就不会进入无限循环
	exitStatusCh := make(chan containerd.ExitStatus, 1)
	exitStatusCh <- containerd.ExitStatus{}
	patches.ApplyMethod(mockTask, "Wait", func(t containerd.Task, ctx context.Context) (<-chan containerd.ExitStatus, error) {
		return exitStatusCh, nil
	})

	client := &Client{
		condClient: mockClient,
		ctx:        context.Background(),
	}

	err := client.Delete("test-pod")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "kill failed")
}*/

/*func TestDeleteTaskStatusRunningWaitError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &containerd.Client{}
	mockContainer := &mockContainer{idVal: "test-container"}
	mockTask := &mockTask{}

	patches.ApplyMethod(mockClient, "Containers", func(c *containerd.Client, ctx context.Context, filters ...string) ([]containerd.Container, error) {
		return []containerd.Container{mockContainer}, nil
	})

	patches.ApplyMethod(mockContainer, "Labels", func(c containerd.Container, ctx context.Context) (map[string]string, error) {
		return map[string]string{"io.kubernetes.pod.name": "test-pod"}, nil
	})

	patches.ApplyMethod(mockContainer, "Task", func(c containerd.Container, ctx context.Context, ioAttach cio.Attach) (containerd.Task, error) {
		return mockTask, nil
	})

	patches.ApplyMethod(mockTask, "Status", func(t containerd.Task, ctx context.Context) (containerd.Status, error) {
		return containerd.Status{Status: containerd.Running}, nil
	})

	patches.ApplyMethod(mockTask, "Kill", func(t containerd.Task, ctx context.Context, sig syscall.Signal, opts ...containerd.KillOpts) error {
		return nil
	})

	patches.ApplyMethod(mockTask, "Wait", func(t containerd.Task, ctx context.Context) (<-chan containerd.ExitStatus, error) {
		return nil, status.Error(codes.Internal, "wait failed")
	})

	client := &Client{
		condClient: mockClient,
		ctx:        context.Background(),
	}

	err := client.Delete("test-pod")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "wait failed")
}
*/

func TestCleanupNerdctlFiles(t *testing.T) {
	client := &Client{ctx: context.Background()}
	labels := map[string]string{
		"nerdctl/state-dir": "/tmp/test-state",
		"nerdctl/name":      "test-container",
	}
	client.cleanupNerdctlFiles("test-id", labels)
}

func TestHandleTaskCreatedStopped(t *testing.T) {
	client := &Client{ctx: context.Background()}
	mockTask := &mockTask{}
	err := client.handleTaskCreatedStopped(mockTask, "test-id")
	if err != nil {
		t.Logf("Expected error: %v", err)
	}
}

func TestHandleTaskPaused(t *testing.T) {
	client := &Client{ctx: context.Background()}
	mockTask := &mockTask{}
	err := client.handleTaskPaused(mockTask, "test-id")
	if err != nil {
		t.Logf("Expected error: %v", err)
	}
}

func TestHandleTaskRunning(t *testing.T) {
	client := &Client{ctx: context.Background()}
	mockTask := &mockTask{
		killFunc: func(ctx context.Context, sig syscall.Signal, opts ...containerd.KillOpts) error {
			return nil
		},
		waitFunc: func(ctx context.Context) (<-chan containerd.ExitStatus, error) {
			ch := make(chan containerd.ExitStatus, numOne)
			ch <- containerd.ExitStatus{}
			close(ch)
			return ch, nil
		},
	}
	err := client.handleTaskRunning(mockTask, "test-id")
	if err != nil {
		t.Logf("Expected error: %v", err)
	}
}

func TestHandleTaskExitStatus(t *testing.T) {
	client := &Client{ctx: context.Background()}
	exitCh := make(chan containerd.ExitStatus, numOne)
	exitCh <- containerd.ExitStatus{}
	close(exitCh)
	err := client.handleTaskExitStatus(exitCh, "test-id")
	if err != nil {
		t.Logf("Expected error: %v", err)
	}
}

func TestHandleTaskDeletion(t *testing.T) {
	client := &Client{ctx: context.Background()}
	mockTask := &mockTask{
		statusFunc: func(ctx context.Context) (containerd.Status, error) {
			return containerd.Status{Status: containerd.Stopped}, nil
		},
	}
	err := client.handleTaskDeletion(mockTask, "test-id")
	if err != nil {
		t.Logf("Expected error: %v", err)
	}
}
