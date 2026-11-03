/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package registry

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/containers/image/v5/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gopkg.openfuyao.cn/bkeadm/utils"
)

func TestCreateMultiArchImage_PreferOCIIndexAndOSLowercase(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	// Stub: first child is OCI manifest, second is docker manifest
	call := 0
	patches.ApplyFunc(getImageSha, func(name string) (imageManifestInfo, error) {
		call++
		if call == 1 {
			return imageManifestInfo{digest: "sha256:child1", size: 123, mediaType: DockerV1Schema2MediaType}, nil
		}
		return imageManifestInfo{digest: "sha256:child2", size: 456, mediaType: DockerV2Schema2MediaType}, nil
	})

	var captured string
	patches.ApplyFunc(putManifests, func(manifests string, name string) error {
		captured = manifests
		assert.Equal(t, "example.com/repo/img:latest", name)
		return nil
	})

	err := CreateMultiArchImage([]ImageArch{
		{Name: "example.com/repo/img:latest-amd64", OS: "linux", Architecture: "amd64"},
		{Name: "example.com/repo/img:latest-arm64", OS: "linux", Architecture: "arm64"},
	}, "example.com/repo/img:latest")
	assert.NoError(t, err)
	assert.NotEmpty(t, captured)

	var out map[string]any
	assert.NoError(t, json.Unmarshal([]byte(captured), &out))

	// Top-level should be OCI index if any child is OCI.
	assert.Equal(t, float64(2), out["schemaVersion"])
	assert.Equal(t, DockerV1ListMediaType, out["mediaType"])

	manifests, ok := out["manifests"].([]any)
	assert.True(t, ok)
	assert.Len(t, manifests, 2)

	first := manifests[0].(map[string]any)
	platform := first["platform"].(map[string]any)

	// Ensure field is "os" (lowercase) in the serialized JSON.
	_, hasOSUpper := platform["OS"]
	_, hasOSLower := platform["os"]
	assert.False(t, hasOSUpper)
	assert.True(t, hasOSLower)
	assert.Equal(t, "linux", platform["os"])

	// Child mediaType should be whatever getImageSha returned for that image.
	assert.Equal(t, DockerV1Schema2MediaType, first["mediaType"])
}

func TestCreateMultiArchImage_KeepDockerListWhenAllChildrenDocker(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(getImageSha, func(name string) (imageManifestInfo, error) {
		return imageManifestInfo{digest: "sha256:child", size: 123, mediaType: DockerV2Schema2MediaType}, nil
	})

	var captured string
	patches.ApplyFunc(putManifests, func(manifests string, name string) error {
		captured = manifests
		return nil
	})

	err := CreateMultiArchImage([]ImageArch{
		{Name: "img:latest-amd64", OS: "linux", Architecture: "amd64"},
	}, "img:latest")
	assert.NoError(t, err)

	var out map[string]any
	assert.NoError(t, json.Unmarshal([]byte(captured), &out))
	assert.Equal(t, DockerV2ListMediaType, out["mediaType"])
}

func TestHandleMultiArchManifest(t *testing.T) {
	op := &Options{}

	t.Run("multi-arch manifest", func(t *testing.T) {
		raw, err := json.Marshal(DockerV2List{
			Manifests: []Manifest{{Digest: "sha256:abc"}},
		})
		require.NoError(t, err)

		handled, err := op.handleMultiArchManifest(raw)
		assert.NoError(t, err)
		assert.True(t, handled)
	})

	t.Run("single-arch schema", func(t *testing.T) {
		raw := []byte(`{"schemaVersion":2,"config":{"digest":"sha256:cfg"}}`)
		handled, err := op.handleMultiArchManifest(raw)
		assert.NoError(t, err)
		assert.False(t, handled)
	})

	t.Run("invalid json", func(t *testing.T) {
		handled, err := op.handleMultiArchManifest([]byte("not-json"))
		assert.Error(t, err)
		assert.False(t, handled)
	})
}

func TestOptionsManifests(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		var captured []ImageArch
		var target string
		patches.ApplyFunc(CreateMultiArchImage, func(ias []ImageArch, tgt string) error {
			captured = ias
			target = tgt
			return nil
		})

		op := &Options{
			Image: "registry.example.com/app",
			Args: []string{
				"registry.example.com/app-amd64",
				"registry.example.com/app-arm64",
			},
		}
		op.Manifests()

		require.Len(t, captured, 2)
		assert.Equal(t, "registry.example.com/app", target)
		assert.Equal(t, "amd64", captured[0].Architecture)
		assert.Equal(t, "arm64", captured[1].Architecture)
	})

	t.Run("invalid image prefix", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		called := false
		patches.ApplyFunc(CreateMultiArchImage, func([]ImageArch, string) error {
			called = true
			return nil
		})

		op := &Options{
			Image: "registry.example.com/app",
			Args:  []string{"other.example.com/app-amd64"},
		}
		op.Manifests()
		assert.False(t, called)
	})
}

func TestCreateMultiArchImageGetImageShaError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(getImageSha, func(string) (imageManifestInfo, error) {
		return imageManifestInfo{}, assert.AnError
	})

	err := CreateMultiArchImage([]ImageArch{
		{Name: "img:latest-amd64", OS: "linux", Architecture: "amd64"},
	}, "img:latest")
	assert.Error(t, err)
}

func TestCreateMultiArchImageEmptyChildMediaType(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(getImageSha, func(string) (imageManifestInfo, error) {
		return imageManifestInfo{digest: "sha256:child", size: 100}, nil
	})

	var captured string
	patches.ApplyFunc(putManifests, func(manifests string, _ string) error {
		captured = manifests
		return nil
	})

	err := CreateMultiArchImage([]ImageArch{
		{Name: "img:latest-amd64", OS: "linux", Architecture: "amd64"},
	}, "img:latest")
	assert.NoError(t, err)

	var out map[string]any
	assert.NoError(t, json.Unmarshal([]byte(captured), &out))
	manifests := out["manifests"].([]any)
	first := manifests[0].(map[string]any)
	assert.Equal(t, DockerV2Schema2MediaType, first["mediaType"])
}

func TestOptionsManifestsMissingArchSuffix(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	called := false
	patches.ApplyFunc(CreateMultiArchImage, func([]ImageArch, string) error {
		called = true
		return nil
	})

	op := &Options{
		Image: "registry.example.com/app",
		Args:  []string{"registry.example.com/app"},
	}
	op.Manifests()
	assert.False(t, called)
}

func TestIsMultiArchManifestsGetSourceFails(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(getNewImageSource, func(types.ImageReference, context.Context, *types.SystemContext,
		utils.RetryOptions) (types.ImageSource, error) {
		return nil, assert.AnError
	})

	assert.False(t, IsMultiArchManifests(true, "example.com/img:tag"))
}

func TestCreateMultiArchImagePutManifestsFails(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(getImageSha, func(string) (imageManifestInfo, error) {
		return imageManifestInfo{digest: "sha256:child", size: 123, mediaType: DockerV2Schema2MediaType}, nil
	})
	patches.ApplyFunc(putManifests, func(string, string) error {
		return assert.AnError
	})

	err := CreateMultiArchImage([]ImageArch{
		{Name: "img:latest-amd64", OS: "linux", Architecture: "amd64"},
	}, "img:latest")
	assert.Error(t, err)
}

func TestOptionsManifestsCreateFails(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(CreateMultiArchImage, func([]ImageArch, string) error {
		return assert.AnError
	})

	op := &Options{
		Image: "registry.example.com/app",
		Args:  []string{"registry.example.com/app-amd64"},
	}
	op.Manifests()
}

func TestHandleMultiArchManifestEmptyManifests(t *testing.T) {
	op := &Options{}
	raw, err := json.Marshal(DockerV2List{Manifests: []Manifest{}})
	require.NoError(t, err)

	handled, err := op.handleMultiArchManifest(raw)
	assert.NoError(t, err)
	assert.False(t, handled)
}

func TestIsMultiArchManifestsGetManifestFails(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(getNewImageSource, func(types.ImageReference, context.Context, *types.SystemContext,
		utils.RetryOptions) (types.ImageSource, error) {
		return &stubImageSource{}, nil
	})
	patches.ApplyFunc(getManifest, func(types.ImageSource, context.Context,
		utils.RetryOptions) ([]byte, string, error) {
		return nil, "", assert.AnError
	})

	assert.False(t, IsMultiArchManifests(true, "example.com/img:tag"))
}

func TestGetManifestRetrySuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(time.Sleep, func(time.Duration) {})

	src := &stubImageSource{
		manifest:         []byte(`{"schemaVersion":2}`),
		mediaType:        DockerV2Schema2MediaType,
		failUntilAttempt: 1,
	}

	got, _, err := getManifest(src, context.Background(), utils.RetryOptions{MaxRetry: 2, Delay: 0})
	assert.NoError(t, err)
	assert.NotEmpty(t, got)
}

func TestIsMultiArchManifestsInvalidManifestJSON(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(getNewImageSource, func(types.ImageReference, context.Context, *types.SystemContext,
		utils.RetryOptions) (types.ImageSource, error) {
		return &stubImageSource{}, nil
	})
	patches.ApplyFunc(getManifest, func(types.ImageSource, context.Context,
		utils.RetryOptions) ([]byte, string, error) {
		return []byte("not-json"), "", nil
	})

	assert.False(t, IsMultiArchManifests(true, "example.com/img:tag"))
}
