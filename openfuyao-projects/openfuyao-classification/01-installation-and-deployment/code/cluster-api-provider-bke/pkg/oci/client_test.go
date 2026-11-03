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

package oci

import (
	"archive/tar"
	"bytes"
	"io"
	"testing"

	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/empty"
	"github.com/google/go-containerregistry/pkg/v1/mutate"
	"github.com/google/go-containerregistry/pkg/v1/tarball"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewClient(t *testing.T) {
	client := NewClient()
	assert.NotNil(t, client)
}

func TestRemoteOptions_InsecureAndAuth(t *testing.T) {
	opts := RemoteOptions(ClientConfig{
		InsecureSkipTLSVerify: true,
		Username:              "user",
		Password:              "pass",
	})
	assert.Len(t, opts, 2)
}

func TestNewClientFromConfig(t *testing.T) {
	client := NewClientFromConfig(ClientConfig{InsecureSkipTLSVerify: true})
	assert.NotNil(t, client)
}

func TestParseRefValid(t *testing.T) {
	ref, err := parseRef("registry.example.com/openfuyao-upgradepath:latest")
	require.NoError(t, err)
	assert.Equal(t, "registry.example.com/openfuyao-upgradepath:latest", ref.String())
}

func TestParseRefEmptyString(t *testing.T) {
	_, err := parseRef("")
	require.Error(t, err)
}

func TestLayerUnmarshalYAML(t *testing.T) {
	content := []byte("key: value\n")
	layer := &Layer{Path: "test.yaml", Content: content}

	var result map[string]string
	err := layer.UnmarshalYAML(&result)
	require.NoError(t, err)
	assert.Equal(t, "value", result["key"])
}

func TestFindFileInTar(t *testing.T) {
	tarData := createTarWithFiles(map[string]string{
		"paths.yaml":   "paths:\n  - from: v1\n    to: v2\n",
		"release.yaml": "version: v2.6.0\n",
	})

	layer, err := findFileInTar(tarData, "paths.yaml")
	require.NoError(t, err)
	assert.Equal(t, "paths.yaml", layer.Path)
	assert.Contains(t, string(layer.Content), "from: v1")

	_, err = findFileInTar(tarData, "nonexistent.yaml")
	require.Error(t, err)
}

func TestFindFileInTarLeadingSlash(t *testing.T) {
	tarData := createTarWithFiles(map[string]string{
		"/paths.yaml": "paths:\n  - from: v1\n    to: v2\n",
	})

	layer, err := findFileInTar(tarData, "paths.yaml")
	require.NoError(t, err)
	assert.Equal(t, "paths.yaml", layer.Path)
}

func TestFindFilesInTarByPrefix(t *testing.T) {
	tarData := createTarWithFiles(map[string]string{
		"kubernetes/v1.29.0/component.yaml": "name: kubernetes\n",
		"kubernetes/v1.29.0/01-crd.yaml":    "kind: CustomResourceDefinition\n",
		"etcd/v3.5.12/component.yaml":       "name: etcd\n",
	})

	layers, err := findFilesInTarByPrefix(tarData, "kubernetes/")
	require.NoError(t, err)
	assert.Len(t, layers, 2)
}

func TestGetLayerByPathTarFallback(t *testing.T) {
	img := createDockerImageWithTarLayer(t, map[string]string{
		"paths.yaml": "apiVersion: config.openfuyao.com/v1alpha1\nkind: UpgradePath\n",
	})

	ociImg := &Image{inner: img}
	layer, err := ociImg.GetLayerByPath("paths.yaml")
	require.NoError(t, err)
	assert.Contains(t, string(layer.Content), "apiVersion")
}

func TestGetLayerByPathNotFound(t *testing.T) {
	img := createDockerImageWithTarLayer(t, map[string]string{
		"other.yaml": "key: value\n",
	})

	ociImg := &Image{inner: img}
	_, err := ociImg.GetLayerByPath("paths.yaml")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
}

func TestGetLayerByPathWithLeadingSlash(t *testing.T) {
	img := createDockerImageWithTarLayer(t, map[string]string{
		"/paths.yaml": "key: value\n",
	})

	ociImg := &Image{inner: img}
	layer, err := ociImg.GetLayerByPath("paths.yaml")
	require.NoError(t, err)
	assert.Contains(t, string(layer.Content), "key: value")
}

func TestGetFilesByExtensionsTarFallback(t *testing.T) {
	img := createDockerImageWithTarLayer(t, map[string]string{
		"release.yaml":                         "kind: ReleaseImage\n",
		"components/kubernetes/component.yaml": "kind: ComponentVersion\n",
		"README.md":                            "ignore me\n",
	})

	ociImg := &Image{inner: img}
	layers, err := ociImg.GetFilesByExtensions(".yaml", ".yml")

	require.NoError(t, err)
	require.Len(t, layers, 2)
	paths := []string{layers[0].Path, layers[1].Path}
	assert.Contains(t, paths, "release.yaml")
	assert.Contains(t, paths, "components/kubernetes/component.yaml")
}

func TestGetLayersByPrefixAnnotatedAndTarFallback(t *testing.T) {
	img := createImageWithAnnotatedLayer(t, "components/etcd/component.yaml", "kind: ComponentVersion\n")
	tarLayer, err := tarball.LayerFromReader(io.NopCloser(bytes.NewReader(createTarWithFiles(map[string]string{
		"components/kubernetes/component.yaml": "kind: ComponentVersion\n",
		"notes/readme.md":                      "ignore\n",
	}))))
	require.NoError(t, err)
	img, err = mutate.AppendLayers(img, tarLayer)
	require.NoError(t, err)

	ociImg := &Image{inner: img}
	layers, err := ociImg.GetLayersByPrefix("components/")

	require.NoError(t, err)
	require.Len(t, layers, 2)
	paths := []string{layers[0].Path, layers[1].Path}
	assert.Contains(t, paths, "components/etcd/component.yaml")
	assert.Contains(t, paths, "components/kubernetes/component.yaml")
}

func TestGetFilesByExtensionsAnnotatedLayer(t *testing.T) {
	img := createImageWithAnnotatedLayer(t, "release.yaml", "kind: ReleaseImage\n")

	ociImg := &Image{inner: img}
	layers, err := ociImg.GetFilesByExtensions(".yaml")

	require.NoError(t, err)
	require.Len(t, layers, 1)
	assert.Equal(t, "release.yaml", layers[0].Path)
	assert.Contains(t, string(layers[0].Content), "ReleaseImage")
}

func createTarWithFiles(files map[string]string) []byte {
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	for name, content := range files {
		hdr := &tar.Header{
			Name: name,
			Size: int64(len(content)),
			Mode: 0644,
		}
		tw.WriteHeader(hdr)
		tw.Write([]byte(content))
	}
	tw.Close()
	return buf.Bytes()
}

func createDockerImageWithTarLayer(t *testing.T, files map[string]string) v1.Image {
	t.Helper()

	tarData := createTarWithFiles(files)
	layer, err := tarball.LayerFromReader(io.NopCloser(bytes.NewReader(tarData)))
	require.NoError(t, err)

	img, err := mutate.AppendLayers(empty.Image, layer)
	require.NoError(t, err)

	return img
}

func createImageWithAnnotatedLayer(t *testing.T, path, content string) v1.Image {
	t.Helper()

	layer, err := tarball.LayerFromReader(io.NopCloser(bytes.NewReader([]byte(content))))
	require.NoError(t, err)
	img, err := mutate.Append(empty.Image, mutate.Addendum{
		Layer: layer,
		Annotations: map[string]string{
			"org.opencontainers.image.title": path,
		},
	})
	require.NoError(t, err)
	return img
}
