/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          <http://license.coscl.org.cn/MulanPSL2>
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package docker

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net"
	"os"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/image"
	"github.com/docker/docker/api/types/network"
	dockerapi "github.com/docker/docker/client"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/bkeadm/utils"
)

const (
	testNumericZero  = 0
	testNumericOne   = 1
	testNumericTwo   = 2
	testNumericThree = 3
	testRetryCount   = 3
	testDelaySeconds = 1
	testContainerID  = "test-container-id-12345"
	testImageID      = "sha256:test-image-id-123456"
	testNumericPort  = 8080

	testIPv4SegmentA = 192
	testIPv4SegmentB = 168
	testIPv4SegmentC = 1
	testIPv4SegmentD = 100
)

const (
	testShortTimeout  = 1 * time.Second
	testMediumTimeout = 5 * time.Second
	testFileMode0644  = 0644
)

var (
	testRetryOptions = utils.RetryOptions{
		MaxRetry: testRetryCount,
		Delay:    testDelaySeconds,
	}
	testLoopbackIP = net.IPv4(
		testIPv4SegmentA,
		testIPv4SegmentB,
		testIPv4SegmentC,
		testIPv4SegmentD,
	).String()
	testTimeout = 5 * time.Second
)

func TestNewDockerClient(t *testing.T) {
	tests := []struct {
		name          string
		socketExists  bool
		newClientErr  error
		expectedError bool
	}{
		{
			name:          "socket exists and client created successfully",
			socketExists:  true,
			newClientErr:  nil,
			expectedError: false,
		},
		{
			name:          "socket does not exist",
			socketExists:  false,
			newClientErr:  nil,
			expectedError: true,
		},
		{
			name:          "socket exists but client creation fails",
			socketExists:  true,
			newClientErr:  errors.New("client creation error"),
			expectedError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			patches.ApplyFunc(utils.Exists, func(path string) bool {
				return tt.socketExists
			})

			patches.ApplyFunc(dockerapi.NewClientWithOpts, func(opts ...dockerapi.Opt) (*dockerapi.Client, error) {
				return nil, tt.newClientErr
			})

			client, err := NewDockerClient()

			if tt.expectedError {
				assert.Error(t, err)
				assert.Nil(t, client)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, client)
			}
		})
	}
}

func TestGetClient(t *testing.T) {
	t.Run("get client returns the underlying client", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		mockDockerClient := &dockerapi.Client{}
		patches.ApplyFunc(utils.Exists, func(path string) bool {
			return true
		})

		patches.ApplyFunc(dockerapi.NewClientWithOpts, func(opts ...dockerapi.Opt) (*dockerapi.Client, error) {
			return mockDockerClient, nil
		})

		dockerClient, err := NewDockerClient()
		assert.NoError(t, err)
		assert.NotNil(t, dockerClient)

		client := dockerClient.GetClient()
		assert.NotNil(t, client)
	})
}

func TestImageRefStruct(t *testing.T) {
	t.Run("image ref struct initialization", func(t *testing.T) {
		imageRef := ImageRef{
			Image:    "nginx:latest",
			Username: "testuser",
			Password: "testpass",
			Platform: "linux/amd64",
		}

		assert.Equal(t, "nginx:latest", imageRef.Image)
		assert.Equal(t, "testuser", imageRef.Username)
		assert.Equal(t, "testpass", imageRef.Password)
		assert.Equal(t, "linux/amd64", imageRef.Platform)
	})

	t.Run("image ref struct with empty credentials", func(t *testing.T) {
		imageRef := ImageRef{
			Image:    "public-image:latest",
			Username: "",
			Password: "",
			Platform: "",
		}

		assert.Equal(t, "public-image:latest", imageRef.Image)
		assert.Empty(t, imageRef.Username)
		assert.Empty(t, imageRef.Password)
		assert.Empty(t, imageRef.Platform)
	})
}

func TestContainerRefStruct(t *testing.T) {
	t.Run("container ref struct initialization", func(t *testing.T) {
		containerRef := ContainerRef{
			Id:   testContainerID,
			Name: "test-container",
		}

		assert.Equal(t, testContainerID, containerRef.Id)
		assert.Equal(t, "test-container", containerRef.Name)
	})
}

func TestImageList(t *testing.T) {
	t.Run("image list returns local images", func(t *testing.T) {
		client := &Client{
			Client: &dockerapi.Client{},
			ctx:    context.Background(),
		}

		patches := gomonkey.NewPatches()

		defer patches.Reset()

		patches.ApplyFunc((*dockerapi.Client).ImageList,
			func(_ *dockerapi.Client, _ context.Context, opts image.ListOptions) ([]image.Summary, error) {
				return []image.Summary{
					{
						ID:       testImageID,
						RepoTags: []string{"nginx:latest", "nginx:alpine"},
					},
					{
						ID:       "sha256:alpine-image-id",
						RepoTags: []string{"alpine:latest"},
					},
				}, nil
			})

		images, err := client.ImageList()
		assert.NoError(t, err)
		assert.Len(t, images, testNumericThree)
	})
}

func TestHasImage(t *testing.T) {
	tests := []struct {
		name           string
		image          string
		imageList      []ImageRef
		expectedResult bool
	}{
		{
			name:           "image exists in the list",
			image:          "nginx:latest",
			imageList:      []ImageRef{{Image: "nginx:latest"}, {Image: "alpine:latest"}},
			expectedResult: true,
		},
		{
			name:           "image does not exist in the list",
			image:          "postgres:latest",
			imageList:      []ImageRef{{Image: "nginx:latest"}, {Image: "alpine:latest"}},
			expectedResult: false,
		},
		{
			name:           "empty image list",
			image:          "nginx:latest",
			imageList:      []ImageRef{},
			expectedResult: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &Client{
				Client: &dockerapi.Client{},
				ctx:    context.Background(),
			}

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc((*dockerapi.Client).ImageList,
				func(_ *dockerapi.Client, _ context.Context, opts image.ListOptions) ([]image.Summary, error) {
					imageSummaries := make([]image.Summary, len(tt.imageList))
					for i, img := range tt.imageList {
						imageSummaries[i] = image.Summary{
							RepoTags: []string{img.Image},
						}
					}
					return imageSummaries, nil
				})

			result := client.HasImage(tt.image)
			assert.Equal(t, tt.expectedResult, result)
		})
	}
}

func TestContainerExists(t *testing.T) {
	tests := []struct {
		name             string
		containerInspect types.ContainerJSON
		expectedResult   bool
	}{
		{
			name: "container exists",
			containerInspect: types.ContainerJSON{
				ContainerJSONBase: &container.ContainerJSONBase{
					ID:   testContainerID,
					Name: "/test-container",
				},
			},
			expectedResult: true,
		},
		{
			name:             "container does not exist",
			containerInspect: types.ContainerJSON{},
			expectedResult:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &Client{
				Client: &dockerapi.Client{},
				ctx:    context.Background(),
			}

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc((*dockerapi.Client).ContainerInspect,
				func(_ *dockerapi.Client, _ context.Context, containerName string) (types.ContainerJSON, error) {
					return tt.containerInspect, nil
				})

			result, exists := client.ContainerExists("test-container")
			assert.Equal(t, tt.expectedResult, exists)
			if tt.expectedResult {
				assert.Equal(t, testContainerID, result.ID)
			}
		})
	}
}

func TestContainerRemove(t *testing.T) {
	tests := []struct {
		name          string
		removeErr     error
		expectedError bool
	}{
		{
			name:          "container removed successfully",
			removeErr:     nil,
			expectedError: false,
		},
		{
			name:          "container removal fails",
			removeErr:     errors.New("remove error"),
			expectedError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &Client{
				Client: &dockerapi.Client{},
				ctx:    context.Background(),
			}

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc((*dockerapi.Client).ContainerRemove,
				func(_ *dockerapi.Client, _ context.Context, containerID string, options container.RemoveOptions) error {
					return tt.removeErr
				})

			err := client.ContainerRemove(testContainerID)
			if tt.expectedError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestContainerStop(t *testing.T) {
	tests := []struct {
		name          string
		stopErr       error
		expectedError bool
	}{
		{
			name:          "container stopped successfully",
			stopErr:       nil,
			expectedError: false,
		},
		{
			name:          "container stop fails",
			stopErr:       errors.New("stop error"),
			expectedError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &Client{
				Client: &dockerapi.Client{},
				ctx:    context.Background(),
			}

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc((*dockerapi.Client).ContainerStop,
				func(_ *dockerapi.Client, _ context.Context, containerID string, options container.StopOptions) error {
					return tt.stopErr
				})

			err := client.ContainerStop(testContainerID)
			if tt.expectedError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestTag(t *testing.T) {
	tests := []struct {
		name          string
		tagErr        error
		expectedError bool
	}{
		{
			name:          "image tagged successfully",
			tagErr:        nil,
			expectedError: false,
		},
		{
			name:          "image tagging fails",
			tagErr:        errors.New("tag error"),
			expectedError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &Client{
				Client: &dockerapi.Client{},
				ctx:    context.Background(),
			}

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc((*dockerapi.Client).ImageTag,
				func(_ *dockerapi.Client, _ context.Context, sourceImage, targetImage string) error {
					return tt.tagErr
				})

			err := client.Tag("nginx:latest", "my-registry.com/nginx:latest")
			if tt.expectedError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestSave(t *testing.T) {
	tests := []struct {
		name          string
		imageSaveErr  error
		expectedError bool
	}{
		{
			name:          "image saved successfully",
			imageSaveErr:  nil,
			expectedError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &Client{
				Client: &dockerapi.Client{},
				ctx:    context.Background(),
			}

			mockBody := io.NopCloser(bytes.NewReader([]byte("test image data")))

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc((*dockerapi.Client).ImageSave,
				func(_ *dockerapi.Client, _ context.Context, imageIDs []string) (io.ReadCloser, error) {
					return mockBody, tt.imageSaveErr
				})

			patches.ApplyFunc(os.WriteFile, func(name string, data []byte, perm os.FileMode) error {
				return nil
			})

			err := client.Save("nginx:latest", "/tmp/test-image.tar")
			assert.NoError(t, err)
		})
	}
}

func TestLoad(t *testing.T) {
	tests := []struct {
		name          string
		loadErr       error
		mockBody      string
		expectedImage string
		expectedError bool
	}{
		{
			name:          "image loaded from stream message",
			loadErr:       nil,
			mockBody:      `{"stream":"Loaded image: registry:2.8.1\n"}`,
			expectedImage: "registry:2.8.1",
			expectedError: false,
		},
		{
			name:    "image loaded with progress detail",
			loadErr: nil,
			mockBody: `{"status":"Loading layer","progressDetail":{"current":98304,"total":196608},"progress":"[================>                                  ]"}
{"stream":"Loaded image: registry:2.8.1\n"}`,
			expectedImage: "registry:2.8.1",
			expectedError: false,
		},
		{
			name:          "image loaded from status message",
			loadErr:       nil,
			mockBody:      `{"status":"Loaded","id":"sha256:abc123","ProgressDetail":{}}`,
			expectedImage: "sha256:abc123",
			expectedError: false,
		},
		{
			name:          "load fails",
			loadErr:       errors.New("load error"),
			mockBody:      "",
			expectedError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &Client{
				Client: &dockerapi.Client{},
				ctx:    context.Background(),
			}

			mockBody := io.NopCloser(bytes.NewReader([]byte(tt.mockBody)))

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc((*dockerapi.Client).ImageLoad,
				func(_ *dockerapi.Client, _ context.Context, input io.Reader) (image.LoadResponse, error) {
					return image.LoadResponse{
						Body: mockBody,
					}, tt.loadErr
				})

			patches.ApplyFunc(os.OpenFile, func(name string, flag int, perm os.FileMode) (*os.File, error) {
				return &os.File{}, nil
			})

			image, err := client.Load("/tmp/test-image.tar")
			if tt.expectedError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expectedImage, image)
			}
		})
	}
}

func TestParseLoadedImage(t *testing.T) {
	tests := []struct {
		name     string
		body     string
		expected string
	}{
		{
			name:     "progress detail only returns empty",
			body:     `{"status":"Loading layer","progressDetail":{"current":98304,"total":196608}}`,
			expected: "",
		},
		{
			name:     "named image takes precedence over sha",
			body:     `{"status":"Loaded","id":"sha256:abc123"}
{"stream":"Loaded image: registry:2.8.1\n"}`,
			expected: "registry:2.8.1",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected, parseLoadedImage([]byte(tt.body)))
		})
	}
}

func TestRemove(t *testing.T) {
	tests := []struct {
		name          string
		removeErr     error
		expectedError bool
	}{
		{
			name:          "image removed successfully",
			removeErr:     nil,
			expectedError: false,
		},
		{
			name:          "image removal fails",
			removeErr:     errors.New("remove error"),
			expectedError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &Client{
				Client: &dockerapi.Client{},
				ctx:    context.Background(),
			}

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc((*dockerapi.Client).ImageRemove,
				func(_ *dockerapi.Client, _ context.Context, imageID string, options image.RemoveOptions) ([]image.DeleteResponse, error) {
					return []image.DeleteResponse{}, tt.removeErr
				})

			err := client.Remove(ImageRef{Image: "nginx:latest"})
			if tt.expectedError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestRun(t *testing.T) {
	tests := []struct {
		name          string
		createErr     error
		startErr      error
		expectedError bool
	}{
		{
			name:          "container created and started successfully",
			createErr:     nil,
			startErr:      nil,
			expectedError: false,
		},
		{
			name:          "container creation fails",
			createErr:     errors.New("create error"),
			startErr:      nil,
			expectedError: true,
		},
		{
			name:          "container start fails",
			createErr:     nil,
			startErr:      errors.New("start error"),
			expectedError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &Client{
				Client: &dockerapi.Client{},
				ctx:    context.Background(),
			}

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc((*dockerapi.Client).ContainerCreate,
				func(_ *dockerapi.Client, _ context.Context, config *container.Config, hostConfig *container.HostConfig,
					networkingConfig *network.NetworkingConfig, platform *specs.Platform, containerName string) (container.CreateResponse, error) {
					return container.CreateResponse{ID: testContainerID}, tt.createErr
				})

			patches.ApplyFunc((*dockerapi.Client).ContainerStart,
				func(_ *dockerapi.Client, _ context.Context, containerID string, options container.StartOptions) error {
					return tt.startErr
				})

			config := &container.Config{
				Image: "nginx:latest",
			}
			hostConfig := &container.HostConfig{}
			networkingConfig := &network.NetworkingConfig{}

			err := client.Run(config, hostConfig, networkingConfig, nil, "test-container")
			if tt.expectedError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestEnsureImageExists(t *testing.T) {
	sleepPatches := gomonkey.ApplyFunc(time.Sleep, func(time.Duration) {})
	defer sleepPatches.Reset()

	tests := []struct {
		name          string
		imageInspect  types.ImageInspect
		inspectErr    error
		pullErr       error
		expectedError bool
	}{
		{
			name: "image already exists",
			imageInspect: types.ImageInspect{
				ID: testImageID,
			},
			inspectErr:    nil,
			pullErr:       nil,
			expectedError: false,
		},
		{
			name:          "image does not exist, pull succeeds",
			imageInspect:  types.ImageInspect{},
			inspectErr:    errors.New("not found"),
			pullErr:       nil,
			expectedError: false,
		},
		{
			name:          "image does not exist, pull fails",
			imageInspect:  types.ImageInspect{},
			inspectErr:    errors.New("not found"),
			pullErr:       errors.New("pull error"),
			expectedError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &Client{
				Client: &dockerapi.Client{},
				ctx:    context.Background(),
			}

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc((*dockerapi.Client).ImageInspectWithRaw,
				func(_ *dockerapi.Client, _ context.Context, imageID string) (types.ImageInspect, []byte, error) {
					return tt.imageInspect, nil, tt.inspectErr
				})

			if tt.inspectErr != nil || tt.imageInspect.ID == "" {
				mockReader := io.NopCloser(bytes.NewReader([]byte("pulling")))
				patches.ApplyFunc((*dockerapi.Client).ImagePull,
					func(_ *dockerapi.Client, _ context.Context, image string, options image.PullOptions) (io.ReadCloser, error) {
						if tt.pullErr != nil {
							return nil, tt.pullErr
						}
						return mockReader, nil
					})

				patches.ApplyFunc(io.Copy, func(dst io.Writer, src io.Reader) (int64, error) {
					return 0, nil
				})

			}

			err := client.EnsureImageExists(ImageRef{Image: "nginx:latest"}, testRetryOptions)
			if tt.expectedError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestEnsureContainerRun(t *testing.T) {
	tests := []struct {
		name             string
		containerInspect types.ContainerJSON
		startErr         error
		removeErr        error
		expectedResult   bool
		expectedError    bool
	}{
		{
			name: "container exists and is running",
			containerInspect: types.ContainerJSON{
				ContainerJSONBase: &container.ContainerJSONBase{
					ID:   testContainerID,
					Name: "/test-container",
					State: &container.State{
						Running: true,
					},
				},
				Config: &container.Config{},
			},
			startErr:       nil,
			removeErr:      nil,
			expectedResult: true,
			expectedError:  false,
		},
		{
			name: "container exists but is not running, start succeeds",
			containerInspect: types.ContainerJSON{
				ContainerJSONBase: &container.ContainerJSONBase{
					ID:   testContainerID,
					Name: "/test-container",
					State: &container.State{
						Running: false,
					},
				},
				Config: &container.Config{},
			},
			startErr:       nil,
			removeErr:      nil,
			expectedResult: true,
			expectedError:  false,
		},
		{
			name: "container does not exist",
			containerInspect: types.ContainerJSON{
				ContainerJSONBase: nil,
			},
			startErr:       nil,
			removeErr:      nil,
			expectedResult: false,
			expectedError:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &Client{
				Client: &dockerapi.Client{},
				ctx:    context.Background(),
			}

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc((*dockerapi.Client).ContainerInspect,
				func(_ *dockerapi.Client, _ context.Context, containerID string) (types.ContainerJSON, error) {
					return tt.containerInspect, nil
				})

			if tt.containerInspect.ContainerJSONBase != nil && tt.containerInspect.State != nil && !tt.containerInspect.State.Running {
				patches.ApplyFunc((*dockerapi.Client).ContainerStart,
					func(_ *dockerapi.Client, _ context.Context, containerID string, options container.StartOptions) error {
						return tt.startErr
					})

				if tt.startErr != nil {
					patches.ApplyFunc((*dockerapi.Client).ContainerRemove,
						func(_ *dockerapi.Client, _ context.Context, containerID string, options container.RemoveOptions) error {
							return tt.removeErr
						})
				}
			}

			result, err := client.EnsureContainerRun(testContainerID)
			assert.Equal(t, tt.expectedResult, result)
			if tt.expectedError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestCopyFromContainer(t *testing.T) {
	tests := []struct {
		name          string
		copyErr       error
		expectedError bool
	}{
		{
			name:          "copy from container successfully",
			copyErr:       nil,
			expectedError: true,
		},
		{
			name:          "copy from container fails",
			copyErr:       errors.New("copy error"),
			expectedError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &Client{
				Client: &dockerapi.Client{},
				ctx:    context.Background(),
			}

			mockContent := io.NopCloser(bytes.NewReader([]byte("test content")))
			mockStat := container.PathStat{
				Mode: testFileMode0644,
			}

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc((*dockerapi.Client).CopyFromContainer,
				func(_ *dockerapi.Client, _ context.Context, containerID, srcPath string) (io.ReadCloser, container.PathStat, error) {
					return mockContent, mockStat, tt.copyErr
				})

			err := client.CopyFromContainer(testContainerID, "/tmp/test", "/tmp/dest")
			if tt.expectedError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestClientStruct(t *testing.T) {
	t.Run("client struct fields are properly initialized", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		mockDockerClient := &dockerapi.Client{}
		patches.ApplyFunc(utils.Exists, func(path string) bool {
			return true
		})

		patches.ApplyFunc(dockerapi.NewClientWithOpts, func(opts ...dockerapi.Opt) (*dockerapi.Client, error) {
			return mockDockerClient, nil
		})

		dockerClient, err := NewDockerClient()
		assert.NoError(t, err)
		assert.NotNil(t, dockerClient)

		client, ok := dockerClient.(*Client)
		assert.True(t, ok)
		assert.NotNil(t, client.Client)
		assert.NotNil(t, client.ctx)
	})
}

func TestPull(t *testing.T) {
	sleepPatches := gomonkey.ApplyFunc(time.Sleep, func(time.Duration) {})
	defer sleepPatches.Reset()

	tests := []struct {
		name          string
		pullErr       error
		expectedError bool
	}{
		{
			name:          "image pulled successfully",
			pullErr:       nil,
			expectedError: false,
		},
		{
			name:          "image pull fails",
			pullErr:       errors.New("pull error"),
			expectedError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &Client{
				Client: &dockerapi.Client{},
				ctx:    context.Background(),
			}

			mockReader := io.NopCloser(bytes.NewReader([]byte("pulling...")))
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc((*dockerapi.Client).ImagePull,
				func(_ *dockerapi.Client, _ context.Context, image string, options image.PullOptions) (io.ReadCloser, error) {
					return mockReader, tt.pullErr
				})

			patches.ApplyFunc(io.Copy, func(dst io.Writer, src io.Reader) (int64, error) {
				return 0, nil
			})

			err := client.Pull(ImageRef{Image: "nginx:latest"}, testRetryOptions)
			if tt.expectedError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestPullNormalizesPlatform(t *testing.T) {
	sleepPatches := gomonkey.ApplyFunc(time.Sleep, func(time.Duration) {})
	defer sleepPatches.Reset()

	client := &Client{
		Client: &dockerapi.Client{},
		ctx:    context.Background(),
	}

	var capturedPlatform string
	mockReader := io.NopCloser(bytes.NewReader([]byte("pulling...")))
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*dockerapi.Client).ImagePull,
		func(_ *dockerapi.Client, _ context.Context, imageName string, options image.PullOptions) (io.ReadCloser, error) {
			capturedPlatform = options.Platform
			return mockReader, nil
		})
	patches.ApplyFunc(io.Copy, func(dst io.Writer, src io.Reader) (int64, error) {
		return 0, nil
	})

	err := client.Pull(ImageRef{Image: "nginx:latest", Platform: "arm64"}, testRetryOptions)
	assert.NoError(t, err)
	assert.Equal(t, "linux/arm64", capturedPlatform)
}

func TestPush(t *testing.T) {
	tests := []struct {
		name          string
		pushErr       error
		expectedError bool
	}{
		{
			name:          "image pushed successfully",
			pushErr:       nil,
			expectedError: false,
		},
		{
			name:          "image push fails",
			pushErr:       errors.New("push error"),
			expectedError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &Client{
				Client: &dockerapi.Client{},
				ctx:    context.Background(),
			}

			mockReader := io.NopCloser(bytes.NewReader([]byte("pushing...")))
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc((*dockerapi.Client).ImagePush,
				func(_ *dockerapi.Client, _ context.Context, image string, options image.PushOptions) (io.ReadCloser, error) {
					return mockReader, tt.pushErr
				})

			patches.ApplyFunc(io.Copy, func(dst io.Writer, src io.Reader) (int64, error) {
				return 0, nil
			})

			err := client.Push(ImageRef{Image: "nginx:latest"})
			if tt.expectedError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestClose(t *testing.T) {
	tests := []struct {
		name     string
		closeErr error
	}{
		{
			name:     "close succeeds",
			closeErr: nil,
		},
		{
			name:     "close fails",
			closeErr: errors.New("close error"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &Client{
				Client: &dockerapi.Client{},
				ctx:    context.Background(),
			}

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc((*dockerapi.Client).Close,
				func(_ *dockerapi.Client) error {
					return tt.closeErr
				})

			client.Close()
		})
	}
}
