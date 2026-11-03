/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package registry

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/containers/image/v5/copy"
	"github.com/containers/image/v5/signature"
	"github.com/containers/image/v5/types"
	digest "github.com/opencontainers/go-digest"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gopkg.openfuyao.cn/bkeadm/utils"
)

func TestDebugReportWriter(t *testing.T) {
	w := &debugReportWriter{}
	n, err := w.Write([]byte("line1\nline2\npartial"))
	assert.NoError(t, err)
	assert.Equal(t, len("line1\nline2\npartial"), n)
	assert.Equal(t, []byte("partial"), w.remainder)

	n, err = w.Write([]byte(" rest\n"))
	assert.NoError(t, err)
	assert.Equal(t, len(" rest\n"), n)
	assert.Len(t, w.remainder, 0)

	w.remainder = []byte(" trailing ")
	w.flush()
	assert.Len(t, w.remainder, 0)
}

func TestExecuteCopyWithRetrySuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(copy.Image, func(context.Context, *signature.PolicyContext, types.ImageReference,
		types.ImageReference, *copy.Options) ([]byte, error) {
		return nil, nil
	})

	srcRef, destRef, err := parseImageReferences(
		"docker://src.example.com/img:tag",
		"docker://dest.example.com/img:tag",
		"src.example.com/img:tag",
	)
	require.NoError(t, err)

	policyCtx, err := createPolicyContext()
	require.NoError(t, err)

	sourceCtx, destCtx, err := createSystemContexts(Options{Arch: "amd64"})
	require.NoError(t, err)

	err = executeCopyWithRetry(copyParams{
		srcRef:         srcRef,
		destRef:        destRef,
		policyContext:  policyCtx,
		sourceCtx:      sourceCtx,
		destinationCtx: destCtx,
		imageAddress:   "docker://src.example.com/img:tag",
		targetAddress:  "docker://dest.example.com/img:tag",
		multiArch:      false,
	})
	assert.NoError(t, err)
}

func TestExecuteCopyWithRetryMultiArch(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	var selection copy.ImageListSelection
	patches.ApplyFunc(copy.Image, func(_ context.Context, _ *signature.PolicyContext, _ types.ImageReference,
		_ types.ImageReference, opts *copy.Options) ([]byte, error) {
		selection = opts.ImageListSelection
		return nil, nil
	})

	srcRef, destRef, err := parseImageReferences(
		"docker://src.example.com/img:tag",
		"docker://dest.example.com/img:tag",
		"src.example.com/img:tag",
	)
	require.NoError(t, err)
	policyCtx, err := createPolicyContext()
	require.NoError(t, err)
	sourceCtx, destCtx, err := createSystemContexts(Options{})
	require.NoError(t, err)

	err = executeCopyWithRetry(copyParams{
		srcRef: srcRef, destRef: destRef, policyContext: policyCtx,
		sourceCtx: sourceCtx, destinationCtx: destCtx,
		imageAddress: "src", targetAddress: "dest", multiArch: true,
	})
	assert.NoError(t, err)
	assert.Equal(t, copy.CopyAllImages, selection)
}

func TestExecuteCopyWithRetryFailure(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(copy.Image, func(context.Context, *signature.PolicyContext, types.ImageReference,
		types.ImageReference, *copy.Options) ([]byte, error) {
		return nil, errors.New("copy failed")
	})

	srcRef, destRef, err := parseImageReferences(
		"docker://src.example.com/img:tag",
		"docker://dest.example.com/img:tag",
		"src.example.com/img:tag",
	)
	require.NoError(t, err)
	policyCtx, err := createPolicyContext()
	require.NoError(t, err)
	sourceCtx, destCtx, err := createSystemContexts(Options{})
	require.NoError(t, err)

	err = executeCopyWithRetry(copyParams{
		srcRef: srcRef, destRef: destRef, policyContext: policyCtx,
		sourceCtx: sourceCtx, destinationCtx: destCtx,
		imageAddress: "src", targetAddress: "dest",
	})
	assert.Error(t, err)
}

func TestCopyRegistryDelegates(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	called := false
	patches.ApplyFunc(executeCopyWithRetry, func(copyParams) error {
		called = true
		return nil
	})

	err := CopyRegistry(Options{
		Source: "src.example.com/img:tag",
		Target: "dest.example.com/img:tag",
		Arch:   "amd64",
	})
	assert.NoError(t, err)
	assert.True(t, called)
}

func TestSyncSingleArchRepoImageSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(CopyRegistry, func(Options) error { return nil })

	err := syncSingleArchRepoImage(Options{
		Source: "src.example.com/img:tag",
		Target: "dest.example.com/img:tag",
	}, "amd64")
	assert.NoError(t, err)
}

func TestSyncSingleArchRepoImageFallback(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	call := 0
	patches.ApplyFunc(CopyRegistry, func(op Options) error {
		call++
		if call == 1 {
			return errors.New("first fail")
		}
		assert.Contains(t, op.Source, "-amd64")
		return nil
	})

	err := syncSingleArchRepoImage(Options{
		Source: "src.example.com/img:tag",
		Target: "dest.example.com/img:tag",
	}, "amd64")
	assert.NoError(t, err)
	assert.Equal(t, 2, call)
}

func TestSyncMultiArchRepoImageDirectSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(CopyRegistry, func(Options) error { return nil })

	err := syncMultiArchRepoImage(Options{
		Source: "src.example.com/img:tag",
		Target: "dest.example.com/img:tag",
	}, []string{"amd64", "arm64"})
	assert.NoError(t, err)
}

func TestSyncArchImagesAndCreateManifest(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(CopyRegistry, func(Options) error { return nil })
	patches.ApplyFunc(CreateMultiArchImage, func([]ImageArch, string) error { return nil })

	err := syncArchImagesAndCreateManifest(
		Options{SrcTLSVerify: true, DestTLSVerify: true},
		"src.example.com/img",
		"dest.example.com/img",
		[]string{"amd64", "arm64"},
	)
	assert.NoError(t, err)
}

func TestSyncArchImagesAndCreateManifestCopyFails(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(CopyRegistry, func(Options) error { return errors.New("copy failed") })

	err := syncArchImagesAndCreateManifest(Options{}, "src", "dest", []string{"amd64"})
	assert.Error(t, err)
}

func TestSyncMultiArchRepoImageFallback(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(CopyRegistry, func(Options) error { return errors.New("direct fail") })
	patches.ApplyFunc(syncArchImagesAndCreateManifest, func(Options, string, string, []string) error {
		return nil
	})

	err := syncMultiArchRepoImage(Options{
		Source: "src.example.com/img:tag",
		Target: "dest.example.com/img:tag",
	}, []string{"amd64", "arm64"})
	assert.NoError(t, err)
}

func newRegistryTestServer(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.Contains(r.URL.Path, "/_catalog"):
			json.NewEncoder(w).Encode(repo{Repositories: []string{"app", "lib/base"}})
		case strings.Contains(r.URL.Path, "/tags/list"):
			json.NewEncoder(w).Encode(tagResponse{Tags: []string{"v1", "latest"}})
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
}

func TestSyncRepo(t *testing.T) {
	server := newRegistryTestServer(t)
	defer server.Close()

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(syncRepoImage, func(Options, []string) error { return nil })

	err := syncRepo(Options{
		Source: server.URL,
		Target: "dest.example.com",
	})
	assert.NoError(t, err)
}

func TestSyncRepoTagFetchFails(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "/_catalog") {
			json.NewEncoder(w).Encode(repo{Repositories: []string{"broken"}})
			return
		}
		http.Error(w, "unavailable", http.StatusInternalServerError)
	}))
	defer server.Close()

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(syncRepoImage, func(Options, []string) error { return nil })

	err := syncRepo(Options{Source: server.URL, Target: "dest.example.com"})
	assert.NoError(t, err)
}

func TestOptionsSyncSingleImage(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(syncRepoImage, func(Options, []string) error { return nil })

	op := &Options{
		Source: "src.example.com/img:tag",
		Target: "dest.example.com/img:tag",
		Arch:   "amd64",
	}
	op.Sync()
}

func TestOptionsSyncFromFile(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	var syncCount int
	var syncedSources []string
	patches.ApplyFunc(syncRepoImage, func(op Options, _ []string) error {
		syncCount++
		syncedSources = append(syncedSources, op.Source)
		return nil
	})

	tmpDir := t.TempDir()
	listFile := filepath.Join(tmpDir, "list.txt")
	require.NoError(t, os.WriteFile(listFile, []byte("app:v1\napp:v2\n"), 0o644))

	op := &Options{
		Source: "src.example.com/",
		Target: "dest.example.com/",
		File:   listFile,
	}
	op.Sync()
	assert.Equal(t, 2, syncCount)
	assert.Len(t, syncedSources, 2)
	assert.Contains(t, syncedSources[0], "app:v1")
	assert.Contains(t, syncedSources[1], "app:v2")
}

func TestOptionsSyncFromFileReadError(t *testing.T) {
	op := &Options{
		Source: "src.example.com/",
		Target: "dest.example.com/",
		File:   "/nonexistent/list.txt",
	}
	op.Sync()
}

func TestOptionsSyncRepoMode(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(syncRepo, func(Options) error { return nil })

	op := &Options{
		SyncRepo: true,
		Source:   "src.example.com",
		Target:   "dest.example.com",
	}
	op.Sync()
}

func TestOptionsSyncRepoModeFails(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(syncRepo, func(Options) error { return errors.New("repo sync failed") })

	op := &Options{
		SyncRepo: true,
		Source:   "src.example.com",
		Target:   "dest.example.com",
	}
	op.Sync()
}

func TestOptionsSyncImageFails(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(syncRepoImage, func(Options, []string) error { return errors.New("sync failed") })

	op := &Options{
		Source: "src.example.com/img:tag",
		Target: "dest.example.com/img:tag",
	}
	op.Sync()
}

func TestGetManifestSuccess(t *testing.T) {
	raw := []byte(`{"schemaVersion":2}`)
	src := &stubImageSource{manifest: raw, mediaType: DockerV2Schema2MediaType}

	got, mediaType, err := getManifest(src, context.Background(), utils.RetryOptions{MaxRetry: 1, Delay: 0})
	assert.NoError(t, err)
	assert.Equal(t, raw, got)
	assert.Equal(t, DockerV2Schema2MediaType, mediaType)
}

func TestGetManifestRetriesExhausted(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(time.Sleep, func(time.Duration) {})

	src := &stubImageSource{manifestErr: errors.New("manifest unavailable")}
	_, _, err := getManifest(src, context.Background(), utils.RetryOptions{MaxRetry: 2, Delay: 0})
	assert.Error(t, err)
}

func TestIsMultiArchManifestsTrue(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	raw, err := json.Marshal(DockerV2List{
		Manifests: []Manifest{{Digest: "sha256:abc"}},
	})
	require.NoError(t, err)

	patches.ApplyFunc(getNewImageSource, func(types.ImageReference, context.Context, *types.SystemContext,
		utils.RetryOptions) (types.ImageSource, error) {
		return &stubImageSource{manifest: raw}, nil
	})
	patches.ApplyFunc(getManifest, func(types.ImageSource, context.Context,
		utils.RetryOptions) ([]byte, string, error) {
		return raw, DockerV2ListMediaType, nil
	})

	assert.True(t, IsMultiArchManifests(false, "example.com/img:tag"))
}

func TestIsMultiArchManifestsFalse(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	raw := []byte(`{"schemaVersion":2}`)
	patches.ApplyFunc(getNewImageSource, func(types.ImageReference, context.Context, *types.SystemContext,
		utils.RetryOptions) (types.ImageSource, error) {
		return &stubImageSource{}, nil
	})
	patches.ApplyFunc(getManifest, func(types.ImageSource, context.Context,
		utils.RetryOptions) ([]byte, string, error) {
		return raw, DockerV2Schema2MediaType, nil
	})

	assert.False(t, IsMultiArchManifests(true, "example.com/img:tag"))
}

func TestIsMultiArchManifestsInvalidName(t *testing.T) {
	assert.False(t, IsMultiArchManifests(true, "://bad"))
}

func TestCreateInspectImageSourceInvalidName(t *testing.T) {
	op := &Options{Image: "://bad", DestTLSVerify: true}
	_, _, err := op.createInspectImageSource(context.Background())
	assert.Error(t, err)
}

func TestInspectMultiArch(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	raw, err := json.Marshal(DockerV2List{
		Manifests: []Manifest{{Digest: "sha256:abc"}},
	})
	require.NoError(t, err)

	patches.ApplyPrivateMethod(&Options{}, "createInspectImageSource",
		func(_ *Options, _ context.Context) (types.ImageSource, *types.SystemContext, error) {
			return &stubImageSource{manifest: raw}, &types.SystemContext{}, nil
		})

	op := &Options{Image: "example.com/img:tag"}
	op.Inspect()
}

func TestInspectGetManifestFails(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyPrivateMethod(&Options{}, "createInspectImageSource",
		func(_ *Options, _ context.Context) (types.ImageSource, *types.SystemContext, error) {
			return &stubImageSource{manifestErr: errors.New("manifest error")}, &types.SystemContext{}, nil
		})

	op := &Options{Image: "example.com/img:tag"}
	op.Inspect()
}

func TestInspectSingleArchDelegatesPrint(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	raw := []byte(`{"schemaVersion":2,"config":{"digest":"sha256:cfg"}}`)
	called := false
	patches.ApplyPrivateMethod(&Options{}, "createInspectImageSource",
		func(_ *Options, _ context.Context) (types.ImageSource, *types.SystemContext, error) {
			return &stubImageSource{manifest: raw}, &types.SystemContext{}, nil
		})
	patches.ApplyPrivateMethod(&Options{}, "printSingleArchConfig",
		func(_ *Options, _ context.Context, _ *types.SystemContext, _ types.ImageSource) {
			called = true
		})

	op := &Options{Image: "example.com/img:tag"}
	op.Inspect()
	assert.True(t, called)
}

func TestInspectCreateSourceFails(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyPrivateMethod(&Options{}, "createInspectImageSource",
		func(_ *Options, _ context.Context) (types.ImageSource, *types.SystemContext, error) {
			return nil, nil, errors.New("source error")
		})

	op := &Options{Image: "example.com/img:tag"}
	op.Inspect()
}

func TestSyncRepoSyncImageFails(t *testing.T) {
	server := newRegistryTestServer(t)
	defer server.Close()

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(syncRepoImage, func(Options, []string) error {
		return errors.New("sync failed")
	})

	err := syncRepo(Options{Source: server.URL, Target: "dest.example.com"})
	assert.NoError(t, err)
}

func TestSyncArchImagesAndCreateManifestFailsOnManifest(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(CopyRegistry, func(Options) error { return nil })
	patches.ApplyFunc(CreateMultiArchImage, func([]ImageArch, string) error {
		return errors.New("manifest failed")
	})

	err := syncArchImagesAndCreateManifest(Options{}, "src", "dest", []string{"amd64"})
	assert.Error(t, err)
}

func registryViewMockHandler(w http.ResponseWriter, r *http.Request) {
	switch {
	case strings.Contains(r.URL.Path, "/_catalog"):
		json.NewEncoder(w).Encode(repo{Repositories: []string{"app", "demo/app"}})
	case strings.Contains(r.URL.Path, "/tags/list"):
		json.NewEncoder(w).Encode(tagResponse{Tags: []string{"latest", "v1"}})
	case strings.Contains(r.URL.Path, "/manifests/"):
		accept := r.Header.Get("Accept")
		if strings.Contains(accept, DockerV2ListMediaType) || strings.Contains(accept, DockerV1ListMediaType) {
			json.NewEncoder(w).Encode(DockerV2List{
				SchemaVersion: 2,
				Manifests: []Manifest{{
					Digest: "sha256:abc",
					Platform: struct {
						Architecture string `json:"architecture"`
						OS           string `json:"os"`
					}{Architecture: "amd64", OS: "linux"},
				}},
			})
			return
		}
		json.NewEncoder(w).Encode(DockerV2Schema{
			SchemaVersion: 2,
			Config: struct {
				MediaType string `json:"mediaType"`
				Size      int    `json:"size"`
				Digest    string `json:"digest"`
			}{Digest: "sha256:cfg"},
			Layers: []struct {
				MediaType string `json:"mediaType"`
				Size      int    `json:"size"`
				Digest    string `json:"digest"`
			}{{Size: 4096}},
		})
	case strings.Contains(r.URL.Path, "/blobs/"):
		json.NewEncoder(w).Encode(blobResponse{
			Created:      "2024-06-01T08:00:00.123456789Z",
			Architecture: "amd64",
		})
	}
}

func TestProcessSingleTag(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(registryViewMockHandler))
	defer server.Close()

	row, archKey := processSingleTag(&http.Client{}, server.URL, "demo/app", "v1")
	require.NotNil(t, row)
	assert.Equal(t, "demo/app", row[0])
	assert.Equal(t, "v1", row[1])
	assert.Contains(t, archKey, "amd64")
}

func TestProcessRepositoriesSkipsPrefix(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(tagResponse{Tags: []string{"latest"}})
	}))
	defer server.Close()

	rows, exports := processRepositories(&http.Client{}, server.URL, []string{"other/app"}, "team/", 1)
	assert.Empty(t, rows)
	assert.Empty(t, exports)
}

func TestViewExportMode(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.Contains(r.URL.Path, "/_catalog"):
			json.NewEncoder(w).Encode(repo{Repositories: []string{"demo/app"}})
		case strings.Contains(r.URL.Path, "/tags/list"):
			json.NewEncoder(w).Encode(tagResponse{Tags: []string{"v1"}})
		default:
			registryViewMockHandler(w, r)
		}
	}))
	defer server.Close()

	op := Options{Args: []string{server.URL}, Tags: 1, Export: true}
	op.View()

	files, err := filepath.Glob("*_image-list.txt")
	require.NoError(t, err)
	for _, f := range files {
		_ = os.Remove(f)
	}
}

func TestSetupHTTPClientUnreachable(t *testing.T) {
	_, _, err := setupHTTPClient("127.0.0.1:1")
	assert.Error(t, err)
}

func TestSyncSingleArchRepoImageBothAttemptsFail(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(CopyRegistry, func(Options) error {
		return errors.New("always fail")
	})

	err := syncSingleArchRepoImage(Options{
		Source: "src.example.com/img:tag",
		Target: "dest.example.com/img:tag",
	}, "amd64")
	assert.Error(t, err)
}

func TestSyncMultiArchRepoImageFallbackFails(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(CopyRegistry, func(Options) error { return errors.New("direct fail") })
	patches.ApplyFunc(syncArchImagesAndCreateManifest, func(Options, string, string, []string) error {
		return errors.New("fallback fail")
	})

	err := syncMultiArchRepoImage(Options{
		Source: "src.example.com/img:tag",
		Target: "dest.example.com/img:tag",
	}, []string{"amd64", "arm64"})
	assert.Error(t, err)
}

func TestOptionsSyncFromFilePartialFailure(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	call := 0
	patches.ApplyFunc(syncRepoImage, func(Options, []string) error {
		call++
		if call == 1 {
			return errors.New("first failed")
		}
		return nil
	})

	tmpDir := t.TempDir()
	listFile := filepath.Join(tmpDir, "list.txt")
	require.NoError(t, os.WriteFile(listFile, []byte("app:v1\napp:v2\n"), 0o644))

	op := &Options{
		Source: "src.example.com/",
		Target: "dest.example.com/",
		File:   listFile,
	}
	op.Sync()
	assert.Equal(t, 2, call)
}

func TestGetImageShaInvalidName(t *testing.T) {
	_, err := getImageSha("://bad")
	assert.Error(t, err)
}

func TestPutManifestsInvalidName(t *testing.T) {
	assert.Error(t, putManifests("{}", "://bad"))
}

func TestProcessRepositoriesWithPrefix(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "/tags/list") {
			json.NewEncoder(w).Encode(tagResponse{Tags: []string{"latest"}})
			return
		}
		registryViewMockHandler(w, r)
	}))
	defer server.Close()

	client := &http.Client{}
	rows, exports := processRepositories(client, server.URL, []string{"app", "other"}, "app", 5)
	assert.NotEmpty(t, rows)
	assert.NotEmpty(t, exports)
}

func TestProcessImageTagsError(t *testing.T) {
	client := &http.Client{}
	rows, exports := processImageTags(client, "http://invalid.local", "missing", 5)
	assert.Empty(t, rows)
	assert.Empty(t, exports)
}

func TestViewRepoImageWithServer(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(registryViewMockHandler))
	defer server.Close()

	host := strings.TrimPrefix(server.URL, "http://")
	result, err := ViewRepoImage(host, map[string][]string{"app": {"v1"}})
	assert.NoError(t, err)
	assert.Contains(t, result, "app:v1")
	assert.Equal(t, "amd64", result["app:v1"][2])
}

func TestSetupHTTPClientHTTPSFallback(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(repo{Repositories: []string{"app"}})
	}))
	defer server.Close()

	host := strings.TrimPrefix(server.URL, "http://")
	client, prefix, err := setupHTTPClient(host)
	require.NoError(t, err)
	assert.NotNil(t, client)
	assert.Contains(t, prefix, host)
}

// stubImageSource implements types.ImageSource for unit tests.
type stubImageSource struct {
	manifest         []byte
	mediaType        string
	manifestErr      error
	failUntilAttempt int
	manifestAttempts int
}

func (s *stubImageSource) Reference() types.ImageReference { return nil }
func (s *stubImageSource) Close() error                    { return nil }
func (s *stubImageSource) GetBlob(context.Context, types.BlobInfo, types.BlobInfoCache) (io.ReadCloser, int64, error) {
	return io.NopCloser(bytes.NewReader(nil)), 0, nil
}
func (s *stubImageSource) GetManifest(_ context.Context, _ *digest.Digest) ([]byte, string, error) {
	if s.manifestErr != nil {
		return nil, "", s.manifestErr
	}
	s.manifestAttempts++
	if s.failUntilAttempt > 0 && s.manifestAttempts <= s.failUntilAttempt {
		return nil, "", errors.New("temporary manifest error")
	}
	return s.manifest, s.mediaType, nil
}
func (s *stubImageSource) HasThreadSafeGetBlob() bool { return true }
func (s *stubImageSource) LayerInfosForCopy(context.Context, *digest.Digest) ([]types.BlobInfo, error) {
	return nil, nil
}
func (s *stubImageSource) GetSignatures(context.Context, *digest.Digest) ([][]byte, error) {
	return nil, nil
}
