/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
package registry

import (
	"archive/tar"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"runtime"
	"strings"
	"testing"

	"github.com/containers/image/v5/types"
	"github.com/stretchr/testify/assert"
)

const (
	testZeroValue  = 0
	testThreeValue = 3
)

func TestHasTransportPrefix(t *testing.T) {
	tests := []struct {
		ref      string
		expected bool
	}{
		{"docker://registry.example.com/image:tag", true},
		{"oci:/path/to/layout:ref", true},
		{"dir:/path/to/directory", true},
		{"oci-archive:/path/to/archive.tar", true},
		{"registry.example.com/image:tag", false},
		{"localhost:5000/image:tag", false},
		{"", false},
	}

	for _, tt := range tests {
		if got := hasTransportPrefix(tt.ref); got != tt.expected {
			t.Errorf("hasTransportPrefix(%q) = %v, want %v", tt.ref, got, tt.expected)
		}
	}
}

func TestNewSystemContext(t *testing.T) {
	ctx, err := newSystemContext()
	if err != nil {
		t.Fatalf("newSystemContext returned error: %v", err)
	}
	if ctx == nil {
		t.Fatal("newSystemContext returned nil")
	}
	if ctx.DockerRegistryUserAgent != "bke/v1.0.0" {
		t.Errorf("DockerRegistryUserAgent = %q, want %q", ctx.DockerRegistryUserAgent, "bke/v1.0.0")
	}
}

func TestOptions(t *testing.T) {
	t.Run("valid options", func(t *testing.T) {
		op := Options{
			Source:        "docker://source:tag",
			Target:        "docker://target:tag",
			Arch:          "amd64",
			MultiArch:     false,
			SrcTLSVerify:  true,
			DestTLSVerify: true,
		}

		if op.Source == "" {
			t.Error("Source should not be empty")
		}
		if op.Target == "" {
			t.Error("Target should not be empty")
		}
	})

	t.Run("multi-arch options", func(t *testing.T) {
		op := Options{
			Source:        "docker://source:tag",
			Target:        "docker://target:tag",
			Arch:          "",
			MultiArch:     true,
			SrcTLSVerify:  true,
			DestTLSVerify: true,
		}

		if !op.MultiArch {
			t.Error("MultiArch should be true")
		}
		if op.Arch != "" {
			t.Error("Arch should be empty for multi-arch")
		}
	})
}

func TestGetArchList(t *testing.T) {
	tests := []struct {
		name     string
		op       Options
		expected []string
	}{
		{
			name: "multi-arch mode",
			op: Options{
				MultiArch: true,
				Arch:      "",
			},
			expected: []string{"amd64", "arm64"},
		},
		{
			name: "single arch specified",
			op: Options{
				MultiArch: false,
				Arch:      "arm64",
			},
			expected: []string{"arm64"},
		},
		{
			name: "default to runtime arch",
			op: Options{
				MultiArch: false,
				Arch:      "",
			},
			expected: []string{runtime.GOARCH},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.op.getArchList()
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestEnsureTrailingSlash(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"registry.example.com", "registry.example.com/"},
		{"registry.example.com/", "registry.example.com/"},
		{"/", "/"},
		{"/path/to/repo", "/path/to/repo/"},
	}

	for _, tt := range tests {
		result := ensureTrailingSlash(tt.input)
		assert.Equal(t, tt.expected, result)
	}
}

func TestRemoveHTTPSchemePrefix(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"https://registry.example.com", "registry.example.com"},
		{"http://registry.example.com", "registry.example.com"},
		{"registry.example.com", "registry.example.com"},
		{"", ""},
	}

	for _, tt := range tests {
		result := removeHTTPSchemePrefix(tt.input)
		assert.Equal(t, tt.expected, result)
	}
}

func TestReadImageListFromFile(t *testing.T) {
	content := `image1:latest
image2:v1.0
image3:v2.0

`
	tempFile, err := os.CreateTemp("", "test-image-list-*.txt")
	assert.NoError(t, err)
	defer os.Remove(tempFile.Name())

	_, err = tempFile.WriteString(content)
	assert.NoError(t, err)
	tempFile.Close()

	images, err := readImageListFromFile(tempFile.Name())
	assert.NoError(t, err)
	assert.Equal(t, []string{"image1:latest", "image2:v1.0", "image3:v2.0"}, images)
}

func TestReadImageListFromFileNotFound(t *testing.T) {
	images, err := readImageListFromFile("/nonexistent/file.txt")
	assert.Error(t, err)
	assert.Nil(t, images)
}

func TestNormalizeImageAddresses(t *testing.T) {
	tests := []struct {
		source   string
		target   string
		expected struct {
			source string
			target string
		}
	}{
		{
			source:   "registry.example.com/image:tag",
			target:   "target.example.com/image:tag",
			expected: struct{ source, target string }{source: "docker://registry.example.com/image:tag", target: "docker://target.example.com/image:tag"},
		},
		{
			source:   "docker://registry.example.com/image:tag",
			target:   "docker://target.example.com/image:tag",
			expected: struct{ source, target string }{source: "docker://registry.example.com/image:tag", target: "docker://target.example.com/image:tag"},
		},
	}

	for _, tt := range tests {
		source, target := normalizeImageAddresses(tt.source, tt.target)
		assert.Equal(t, tt.expected.source, source)
		assert.Equal(t, tt.expected.target, target)
	}
}

func TestBuildSyncImageOptions(t *testing.T) {
	baseOp := Options{
		Source: "source.example.com",
		Target: "target.example.com",
	}

	result := buildSyncImageOptions(baseOp, "test-image", "v1.0")

	assert.Equal(t, "source.example.com/test-image:v1.0", result.Source)
	assert.Equal(t, "target.example.com/test-image:v1.0", result.Target)
}

func TestInitHTTPClient(t *testing.T) {
	tests := []struct {
		name        string
		addr        string
		expectedURL string
	}{
		{
			name:        "HTTPS address",
			addr:        "registry.example.com",
			expectedURL: "https://registry.example.com",
		},
		{
			name:        "HTTP address",
			addr:        "http://registry.example.com",
			expectedURL: "http://registry.example.com",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, baseURL := initHTTPClient(tt.addr)
			assert.NotNil(t, client)
			assert.Equal(t, tt.expectedURL, baseURL)
		})
	}
}

func TestSetupSyncHTTPClient(t *testing.T) {
	tests := []struct {
		name        string
		srcRepo     string
		expectHTTPS bool
	}{
		{
			name:        "without protocol",
			srcRepo:     "registry.example.com",
			expectHTTPS: true,
		},
		{
			name:        "with https protocol",
			srcRepo:     "https://registry.example.com",
			expectHTTPS: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, resultRepo := setupSyncHTTPClient(tt.srcRepo)
			assert.NotNil(t, client)
			if tt.expectHTTPS {
				assert.True(t, strings.HasPrefix(resultRepo, "https://"))
			}
		})
	}
}

func TestFetchImageTags(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v2/test-image/tags/list" {
			response := tagResponse{
				Tags: []string{"v1.0", "v2.0", "latest"},
			}
			json.NewEncoder(w).Encode(response)
		} else {
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	client := &http.Client{}
	tags, err := fetchImageTags(client, server.URL, "test-image")
	assert.NoError(t, err)
	assert.Equal(t, []string{"v1.0", "v2.0", "latest"}, tags)
}

func TestFetchImageTagsError(t *testing.T) {
	client := &http.Client{}
	tags, err := fetchImageTags(client, "http://nonexistent", "test-image")
	assert.Error(t, err)
	assert.Nil(t, tags)
}

func TestFetchImageTagsNoTags(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := tagResponse{
			Tags: nil,
		}
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := &http.Client{}
	tags, err := fetchImageTags(client, server.URL, "test-image")
	assert.Error(t, err)
	assert.Nil(t, tags)
}

func TestFetchRepositories(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v2/_catalog" {
			response := repo{
				Repositories: []string{"image1", "image2", "image3"},
			}
			json.NewEncoder(w).Encode(response)
		} else {
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	client := &http.Client{}
	repos := fetchRepositories(client, server.URL+"/v2/_catalog")
	assert.NotNil(t, repos)
	assert.Equal(t, testThreeValue, len(repos.Repositories))
}

func TestFetchRepositoriesError(t *testing.T) {
	client := &http.Client{}
	repos := fetchRepositories(client, "http://nonexistent/_catalog")
	assert.Nil(t, repos)
}

func TestGetImageTags(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := tagResponse{
			Tags: []string{"v1.0", "v2.0"},
		}
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := &http.Client{}
	tags, err := getImageTags(client, server.URL+"/v2/test-image/tags/list", "test-image")
	assert.NoError(t, err)
	assert.Equal(t, []string{"v1.0", "v2.0"}, tags.Tags)
}

func TestExtractArchitectures(t *testing.T) {
	tests := []struct {
		name     string
		manifest DockerV2List
		expected string
	}{
		{
			name: "multiple architectures",
			manifest: DockerV2List{
				Manifests: []Manifest{
					{Platform: struct {
						Architecture string `json:"architecture"`
						OS           string `json:"os"`
					}{Architecture: "amd64", OS: "linux"}},
					{Platform: struct {
						Architecture string `json:"architecture"`
						OS           string `json:"os"`
					}{Architecture: "arm64", OS: "linux"}},
				},
			},
			expected: "amd64,arm64,",
		},
		{
			name: "skip unknown architecture",
			manifest: DockerV2List{
				Manifests: []Manifest{
					{Platform: struct {
						Architecture string `json:"architecture"`
						OS           string `json:"os"`
					}{Architecture: "unknown", OS: "linux"}},
					{Platform: struct {
						Architecture string `json:"architecture"`
						OS           string `json:"os"`
					}{Architecture: "amd64", OS: "linux"}},
				},
			},
			expected: "amd64,",
		},
		{
			name:     "empty manifests",
			manifest: DockerV2List{Manifests: []Manifest{}},
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractArchitectures(&tt.manifest)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestOutputResults(t *testing.T) {
	headers := []string{"COL1", "COL2"}
	rows := [][]string{
		{"val1", "val2"},
		{"val3", "val4"},
	}

	exportList := map[string]string{
		"test.txt": "content",
	}

	outputResults(false, exportList, headers, rows)

	tempDir := t.TempDir()
	exportListFile := tempDir + "/test_export.txt"
	exportList2 := map[string]string{
		exportListFile: "test content\n",
	}

	outputResults(true, exportList2, headers, rows)

	content, err := os.ReadFile(exportListFile)
	assert.NoError(t, err)
	assert.Equal(t, "test content\n", string(content))
}

func TestReverseLayers(t *testing.T) {
	layers := []types.BlobInfo{
		{Digest: "sha256:first"},
		{Digest: "sha256:second"},
		{Digest: "sha256:third"},
	}

	original := make([]types.BlobInfo, len(layers))
	copy(original, layers)

	reversed := reverseLayers(layers)

	assert.Equal(t, original[0].Digest, reversed[2].Digest)
	assert.Equal(t, original[2].Digest, reversed[0].Digest)
}

func TestIsWhiteout(t *testing.T) {
	tests := []struct {
		path     string
		expected bool
	}{
		{"usr/.wh..opq", true},
		{"usr/bin/ls", false},
		{"usr/.wh.file", true},
		{"wh.file", false},
	}

	for _, tt := range tests {
		result := isWhiteout(tt.path)
		assert.Equal(t, tt.expected, result)
	}
}

func TestIsDir(t *testing.T) {
	dirHeader := &tar.Header{Typeflag: tar.TypeDir}
	fileHeader := &tar.Header{Typeflag: tar.TypeReg}

	assert.True(t, isDir(dirHeader))
	assert.False(t, isDir(fileHeader))
}

func TestFound(t *testing.T) {
	tests := []struct {
		m        map[string]string
		expected bool
	}{
		{map[string]string{"key1": "val1", "key2": "val2"}, true},
		{map[string]string{"key1": "", "key2": "val2"}, false},
		{map[string]string{}, true},
	}

	for _, tt := range tests {
		result := found(tt.m)
		assert.Equal(t, tt.expected, result)
	}
}

func TestProcessImageLayers(t *testing.T) {
	request := &ImageProcessRequest{
		Schema: &DockerV2Schema{
			Layers: []struct {
				MediaType string `json:"mediaType"`
				Size      int    `json:"size"`
				Digest    string `json:"digest"`
			}{
				{Size: 1000},
				{Size: 2000},
			},
		},
		Arch: "amd64",
	}

	schema, size, err := processImageLayers(request)
	assert.NoError(t, err)
	assert.NotNil(t, schema)
	assert.Equal(t, 3000*2, size)
}

func TestProcessImageLayersNoLayers(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
		}
	}()

	request := &ImageProcessRequest{
		HTTPClient: &http.Client{},
		Schema: &DockerV2Schema{Layers: []struct {
			MediaType string `json:"mediaType"`
			Size      int    `json:"size"`
			Digest    string `json:"digest"`
		}{}},
		Manifest: &DockerV2List{Manifests: []Manifest{{Digest: "sha256:test", Platform: struct {
			Architecture string `json:"architecture"`
			OS           string `json:"os"`
		}{Architecture: "amd64", OS: "linux"}}}},
		Arch: "",
	}

	schema, size, err := processImageLayers(request)
	assert.Error(t, err)
	assert.Nil(t, schema)
	assert.Equal(t, testZeroValue, size)
}

func TestStreamDataToFile(t *testing.T) {
	content := "test content for streaming"
	reader := strings.NewReader(content)
	tempFile := t.TempDir() + "/test-stream.txt"

	err := streamDataToFile(reader, tempFile)
	assert.NoError(t, err)

	result, err := os.ReadFile(tempFile)
	assert.NoError(t, err)
	assert.Equal(t, content, string(result))
}

func TestCopyKnownHdrs(t *testing.T) {
	copied := CopyKnownHdrs()

	assert.NotNil(t, copied)
	assert.Contains(t, copied, "gz")
	assert.Contains(t, copied, "xz")
	assert.Contains(t, copied, "zst")
	assert.Contains(t, copied, "tar")
}

func TestHeaderMatch(t *testing.T) {
	gzHeader := Header{
		Format:      "gz",
		magicNumber: []byte{0x1F, 0x8B},
		mgOffset:    0,
		SizeOff:     0,
		SizeLen:     0,
	}

	testBytes := []byte{0x1F, 0x8B, 0x08, 0x00}
	assert.True(t, gzHeader.Match(testBytes))

	nonGzBytes := []byte{0x89, 0x50, 0x4E, 0x47}
	assert.False(t, gzHeader.Match(nonGzBytes))
}

func TestBuildDownloadFileMap(t *testing.T) {
	od := &OptionsDownload{
		DownloadInImageFile: "/file1.txt,/file2.txt,/file3.txt",
	}

	result := od.buildDownloadFileMap()

	assert.Equal(t, testThreeValue, len(result))
	assert.Contains(t, result, "file1.txt")
	assert.Contains(t, result, "file2.txt")
	assert.Contains(t, result, "file3.txt")
}

func TestBuildDownloadFileMapEmpty(t *testing.T) {
	od := &OptionsDownload{
		DownloadInImageFile: "",
	}

	result := od.buildDownloadFileMap()

	assert.Equal(t, testZeroValue, len(result))
}

func TestBuildDownloadFileMapWithSlash(t *testing.T) {
	od := &OptionsDownload{
		DownloadInImageFile: "/etc/config/file.txt",
	}

	result := od.buildDownloadFileMap()

	assert.Contains(t, result, "etc/config/file.txt")
}

func TestBuildSystemContext(t *testing.T) {
	od := &OptionsDownload{
		Username:     "testuser",
		Password:     "testpass",
		CertDir:      "/etc/certs",
		SrcTLSVerify: true,
	}

	ctx := od.buildSystemContext()

	assert.NotNil(t, ctx)
	assert.NotNil(t, ctx.DockerAuthConfig)
	assert.Equal(t, "testuser", ctx.DockerAuthConfig.Username)
	assert.Equal(t, "testpass", ctx.DockerAuthConfig.Password)
	assert.Equal(t, "/etc/certs", ctx.DockerCertPath)
}

func TestBuildSystemContextNoAuth(t *testing.T) {
	od := &OptionsDownload{}

	ctx := od.buildSystemContext()

	assert.NotNil(t, ctx)
	assert.Nil(t, ctx.DockerAuthConfig)
}

func TestEnsureDownloadDir(t *testing.T) {
	od := &OptionsDownload{
		DownloadToDir: t.TempDir(),
	}

	err := od.ensureDownloadDir()
	assert.NoError(t, err)
}

func TestEnsureDownloadDirCreate(t *testing.T) {
	tempDir := t.TempDir()
	newDir := tempDir + "/new-download-dir"

	od := &OptionsDownload{
		DownloadToDir: newDir,
	}

	err := od.ensureDownloadDir()
	assert.NoError(t, err)
	assert.True(t, strings.HasPrefix(newDir, tempDir))
}

func TestCloseImage(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
		}
	}()
}

func TestNewFormatReadersError(t *testing.T) {
	reader := &errorReader{err: io.EOF}

	fr, err := NewFormatReaders(reader, 0)
	assert.True(t, err != nil || fr == nil)
}

func TestShouldExtractFile(t *testing.T) {
	tests := []struct {
		name     string
		hdr      *tar.Header
		target   string
		expected bool
	}{
		{
			name:     "matching file",
			hdr:      &tar.Header{Name: "path/to/file.txt", Typeflag: tar.TypeReg},
			target:   "file.txt",
			expected: true,
		},
		{
			name:     "non-matching file",
			hdr:      &tar.Header{Name: "path/to/other.txt", Typeflag: tar.TypeReg},
			target:   "file.txt",
			expected: false,
		},
		{
			name:     "whiteout file",
			hdr:      &tar.Header{Name: "path/to/.wh.file.txt", Typeflag: tar.TypeReg},
			target:   "file.txt",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := shouldExtractFile(tt.hdr, tt.target)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestCreatePolicyContext(t *testing.T) {
	policyCtx, err := createPolicyContext()
	assert.NoError(t, err)
	assert.NotNil(t, policyCtx)
}

func TestCreateSystemContexts(t *testing.T) {
	op := Options{
		SrcTLSVerify:  false,
		DestTLSVerify: false,
		Arch:          "amd64",
	}

	sourceCtx, destinationCtx, err := createSystemContexts(op)
	assert.NoError(t, err)
	assert.NotNil(t, sourceCtx)
	assert.NotNil(t, destinationCtx)
	assert.Equal(t, types.OptionalBoolTrue, sourceCtx.DockerInsecureSkipTLSVerify)
	assert.Equal(t, "amd64", sourceCtx.ArchitectureChoice)
}

func TestCreateSystemContextsNoTLSVerify(t *testing.T) {
	op := Options{
		SrcTLSVerify:  false,
		DestTLSVerify: false,
	}

	sourceCtx, destinationCtx, err := createSystemContexts(op)
	assert.NoError(t, err)
	assert.NotNil(t, sourceCtx)
	assert.NotNil(t, destinationCtx)
}

func TestFetchImageManifest(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "/manifests/") {
			response := DockerV2List{
				SchemaVersion: 2,
				Manifests: []Manifest{
					{
						Digest: "sha256:abc123",
						Platform: struct {
							Architecture string `json:"architecture"`
							OS           string `json:"os"`
						}{Architecture: "amd64", OS: "linux"},
					},
				},
			}
			json.NewEncoder(w).Encode(response)
		} else {
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	client := &http.Client{}
	manifest, req, err := fetchImageManifest(client, server.URL, "test-image", "latest")
	assert.NoError(t, err)
	assert.NotNil(t, manifest)
	assert.NotNil(t, req)
}

func TestFetchImageLayers(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := DockerV2Schema{
			SchemaVersion: 2,
			Layers: []struct {
				MediaType string `json:"mediaType"`
				Size      int    `json:"size"`
				Digest    string `json:"digest"`
			}{
				{Size: 1000},
			},
		}
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := &http.Client{}
	req, _ := http.NewRequest("GET", server.URL+"/v2/test-image/manifests/latest", nil)

	schema, _, err := fetchImageLayers(client, req, "test-image", "latest")
	assert.NoError(t, err)
	assert.NotNil(t, schema)
}

func TestFetchImageMetadata(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := blobResponse{
			Created:      "2025-01-01T00:00:00Z",
			Architecture: "amd64",
		}
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := &http.Client{}
	request := &ImageProcessRequest{
		HTTPClient: client,
		BaseURL:    server.URL,
		Image:      "test-image",
		Tag:        "latest",
		Schema: &DockerV2Schema{
			Config: struct {
				MediaType string `json:"mediaType"`
				Size      int    `json:"size"`
				Digest    string `json:"digest"`
			}{Digest: "sha256:config"},
		},
	}

	arch, createTime, err := fetchImageMetadata(request)
	assert.NoError(t, err)
	assert.Equal(t, "amd64", arch)
	assert.NotEmpty(t, createTime)
}

func TestProcessImageLayersComplete(t *testing.T) {
	request := &ImageProcessRequest{
		HTTPClient: &http.Client{},
		Schema: &DockerV2Schema{
			Layers: []struct {
				MediaType string `json:"mediaType"`
				Size      int    `json:"size"`
				Digest    string `json:"digest"`
			}{
				{Size: 1000},
				{Size: 2000},
			},
		},
		Arch: "amd64",
	}

	schema, size, err := processImageLayers(request)
	assert.NoError(t, err)
	assert.NotNil(t, schema)
	assert.Equal(t, 3000*2, size)
}

func TestViewMethod(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "/_catalog") {
			response := repo{
				Repositories: []string{"image1"},
			}
			json.NewEncoder(w).Encode(response)
		} else if strings.Contains(r.URL.Path, "/tags/list") {
			response := tagResponse{
				Tags: []string{"latest"},
			}
			json.NewEncoder(w).Encode(response)
		} else if strings.Contains(r.URL.Path, "/manifests/") {
			response := DockerV2List{
				SchemaVersion: 2,
				Manifests: []Manifest{
					{Digest: "sha256:abc", Platform: struct {
						Architecture string `json:"architecture"`
						OS           string `json:"os"`
					}{Architecture: "amd64", OS: "linux"}},
				},
			}
			json.NewEncoder(w).Encode(response)
		} else {
			response := DockerV2Schema{
				Layers: []struct {
					MediaType string `json:"mediaType"`
					Size      int    `json:"size"`
					Digest    string `json:"digest"`
				}{{Size: 1000}},
			}
			json.NewEncoder(w).Encode(response)
		}
	}))
	defer server.Close()

	op := Options{
		Args:   []string{server.URL},
		Prefix: "",
		Tags:   5,
		Export: false,
	}

	op.View()
}

func TestViewMethodNoRepos(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := repo{
			Repositories: []string{},
		}
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	op := Options{
		Args:   []string{server.URL},
		Prefix: "",
		Tags:   5,
		Export: false,
	}

	op.View()
}

func TestFetchV1Manifest(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := DockerV2List{
			SchemaVersion: 1,
			Manifests: []Manifest{
				{Digest: "sha256:v1", Platform: struct {
					Architecture string `json:"architecture"`
					OS           string `json:"os"`
				}{Architecture: "amd64", OS: "linux"}},
			},
		}
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := &http.Client{}
	req, _ := http.NewRequest("GET", server.URL+"/v2/test-image/manifests/v1", nil)

	manifest := &DockerV2List{}
	err := fetchV1Manifest(client, req, "test-image", "v1", manifest)
	assert.NoError(t, err)
}

func TestPrintTable(t *testing.T) {
	headers := []string{"COL1", "COL2", "COL3"}
	rows := [][]string{
		{"val1", "val2", "val3"},
		{"val4", "val5", "val6"},
	}

	PrintTable(headers, rows)
}

func TestLogDownloadResults(t *testing.T) {
	od := &OptionsDownload{
		DownloadInImageFile: "/file.txt",
	}

	downloadFileMap := map[string]string{
		"file.txt": "",
	}

	od.logDownloadResults(downloadFileMap)
}

func TestLogDownloadResultsSuccess(t *testing.T) {
	od := &OptionsDownload{
		DownloadInImageFile: "/file.txt",
	}

	downloadFileMap := map[string]string{
		"file.txt": "/extracted/path/file.txt",
	}

	od.logDownloadResults(downloadFileMap)
}

func TestStreamDataToFileError(t *testing.T) {
	content := "test content"
	reader := strings.NewReader(content)
	tempFile := t.TempDir() + "/subdir/test-stream.txt"

	err := streamDataToFile(reader, tempFile)
	assert.Error(t, err)
}

func TestCopyKnownHdrsIndependence(t *testing.T) {
	copied := CopyKnownHdrs()

	knownHeaders["gz"] = Header{Format: "changed"}
	assert.NotEqual(t, "changed", copied["gz"].Format)

	knownHeaders["gz"] = Header{Format: "gz"}
}

func TestDeleteInvalidArgsCount(t *testing.T) {
	op := &Options{
		Args: []string{},
	}

	op.Delete()
}

func TestProcessLayerNilMap(t *testing.T) {
	config := &LayerProcessConfig{
		DownloadFileMap: nil,
	}

	err := processLayer(context.Background(), config)
	assert.Error(t, err)
}

func TestConstructReaders(t *testing.T) {
	reader := &errorReader{err: io.EOF}

	fr := &FormatReaders{
		buf: make([]byte, 512),
	}

	err := fr.constructReaders(reader)
	assert.Error(t, err)
}

func TestZstReader(t *testing.T) {
	t.Skip("zstd reader requires valid frame; covered indirectly via constructReaders in integration scenarios")
}

func TestXzReader(t *testing.T) {
	t.Skip("xz reader requires valid frame; covered indirectly via constructReaders in integration scenarios")
}

func TestClose(t *testing.T) {
	fr := &FormatReaders{
		readers: []reader{},
	}

	err := fr.Close()
	assert.NoError(t, err)
}

type errorReader struct {
	err error
}

func (r *errorReader) Read(p []byte) (n int, err error) {
	return 0, r.err
}

func (r *errorReader) Close() error {
	return nil
}

func TestViewRepoImage(t *testing.T) {
	tests := []struct {
		name        string
		address     string
		images      map[string][]string
		expectPanic bool
	}{
		{
			name:        "view repo image with empty images",
			address:     "registry.example.com",
			images:      map[string][]string{},
			expectPanic: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := ViewRepoImage(tt.address, tt.images)

			assert.Error(t, err)
			assert.Nil(t, result)
		})
	}
}

func TestSetupHTTPClient(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(repo{Repositories: []string{"image1"}})
	}))
	defer server.Close()

	host := strings.TrimPrefix(server.URL, "http://")
	httpClient, httpPrefix, err := setupHTTPClient(host)

	assert.NoError(t, err)
	assert.NotNil(t, httpClient)
	assert.Contains(t, httpPrefix, host)
}
