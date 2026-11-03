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

// Package oci provides a client for interacting with OCI (Open Container Initiative)
// image registries. It supports fetching image digests, pulling images, and extracting
// specific file layers from OCI artifacts. This package is used by the upgrade path
// system to read upgrade path definitions (paths.yaml) stored in OCI images.
package oci

import (
	"archive/tar"
	"bytes"
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/google/go-containerregistry/pkg/authn"
	"github.com/google/go-containerregistry/pkg/name"
	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/remote"
	"sigs.k8s.io/yaml"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
)

type Client struct {
	// opts holds remote.Option values such as auth, transport, and platform selectors
	// that are passed to every remote operation.
	opts []remote.Option
}

func NewClient(opts ...remote.Option) *Client {
	return &Client{opts: opts}
}

// ClientConfig holds OCI registry client options shared by all OCI pull paths.
type ClientConfig struct {
	InsecureSkipTLSVerify bool
	Username              string
	Password              string
}

// RemoteOptions builds go-containerregistry remote options from ClientConfig.
func RemoteOptions(cfg ClientConfig) []remote.Option {
	var opts []remote.Option

	if cfg.InsecureSkipTLSVerify {
		tr := http.DefaultTransport.(*http.Transport).Clone()
		tr.TLSClientConfig = &tls.Config{
			MinVersion:         tls.VersionTLS12,
			InsecureSkipVerify: true, // #nosec G402 -- explicit opt-in via configuration
		}
		opts = append(opts, remote.WithTransport(tr))
	}

	if cfg.Username != "" && cfg.Password != "" {
		opts = append(opts, remote.WithAuth(&authn.Basic{
			Username: cfg.Username,
			Password: cfg.Password,
		}))
	}
	return opts
}

// NewClientFromConfig creates a Client with TLS and auth options from ClientConfig.
func NewClientFromConfig(cfg ClientConfig) *Client {
	return NewClient(RemoteOptions(cfg)...)
}

// parseRef parses an OCI reference string (e.g. "registry/repo:tag" or "registry/repo@sha256:...")
// into a name.Reference suitable for remote operations.
func parseRef(ociRef string) (name.Reference, error) {
	return name.ParseReference(ociRef)
}

// GetDigest queries the OCI registry for the current digest of the referenced image.
// Returns the digest string (e.g. "sha256:abcdef...") without pulling the full image.
// This is used by DigestMonitor to detect whether the OCI artifact has changed.
func (c *Client) GetDigest(ctx context.Context, ociRef string) (string, error) {
	ref, err := parseRef(ociRef)
	if err != nil {
		return "", fmt.Errorf("failed to parse reference %s: %w", ociRef, err)
	}

	desc, err := remote.Get(ref, c.opts...)
	if err != nil {
		return "", fmt.Errorf("failed to get descriptor for %s: %w", ociRef, err)
	}

	return desc.Descriptor.Digest.String(), nil
}

// Pull downloads the referenced OCI image and returns an Image wrapper that supports
// layer extraction by file path or path prefix.
func (c *Client) Pull(ctx context.Context, ociRef string) (*Image, error) {
	ref, err := parseRef(ociRef)
	if err != nil {
		return nil, fmt.Errorf("failed to parse reference %s: %w", ociRef, err)
	}

	img, err := remote.Image(ref, c.opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to pull image %s: %w", ociRef, err)
	}

	return &Image{inner: img}, nil
}

type Image struct {
	// inner is the underlying go-containerregistry v1.Image.
	inner v1.Image
}

type Layer struct {
	// Path is the file path annotation (org.opencontainers.image.title) of the layer,
	// e.g. "paths.yaml".
	Path string
	// Content holds the raw uncompressed bytes of the layer file.
	Content []byte
}

func (l *Layer) UnmarshalYAML(obj interface{}) error {
	return yaml.Unmarshal(l.Content, obj)
}

// GetLayerByPath first searches for a layer whose "org.opencontainers.image.title"
// annotation matches the given path exactly (OCI artifact style). If no annotated
// layer is found, it falls back to iterating all layers, treating each as a tar
// archive and looking for a file whose name matches the requested path (Docker
// container image style). Annotation-based lookup is always preferred.
func (i *Image) GetLayerByPath(path string) (*Layer, error) {
	manifest, err := i.inner.Manifest()
	if err != nil {
		return nil, fmt.Errorf("failed to get manifest: %w", err)
	}

	for _, layerDesc := range manifest.Layers {
		filePath, ok := annotatedPath(layerDesc)
		if !ok || filePath != path {
			continue
		}
		return i.readAnnotatedLayer(layerDesc, path)
	}

	return i.extractFileFromLayers(path)
}

// GetLayersByPrefix first searches for layers whose "org.opencontainers.image.title"
// annotation starts with the given prefix (OCI artifact style), then additionally
// searches inside tar-formatted layers for files whose path starts with the prefix
// (Docker container image style). Duplicate entries (same path found via both
// annotation and tar) are deduplicated, keeping the annotation-based result.
func (i *Image) GetLayersByPrefix(prefix string) ([]Layer, error) {
	manifest, err := i.inner.Manifest()
	if err != nil {
		return nil, fmt.Errorf("failed to get manifest: %w", err)
	}

	layers, seen, err := i.annotatedLayersByPrefix(manifest.Layers, prefix)
	if err != nil {
		return nil, err
	}

	tarLayers, err := i.extractFilesFromLayersByPrefix(prefix)
	if err != nil {
		return nil, fmt.Errorf("failed to extract files from layers by prefix %s: %w", prefix, err)
	}
	for _, l := range tarLayers {
		if _, exists := seen[l.Path]; !exists {
			seen[l.Path] = struct{}{}
			layers = append(layers, l)
		}
	}

	return layers, nil
}

func annotatedPath(layerDesc v1.Descriptor) (string, bool) {
	if layerDesc.Annotations == nil {
		return "", false
	}
	path, ok := layerDesc.Annotations[annotation.OCIImageTitleAnnotationKey]
	return path, ok
}

func (i *Image) readAnnotatedLayer(layerDesc v1.Descriptor, path string) (*Layer, error) {
	layer, err := i.inner.LayerByDigest(layerDesc.Digest)
	if err != nil {
		return nil, fmt.Errorf("failed to get layer by digest: %w", err)
	}
	content, err := layer.Uncompressed()
	if err != nil {
		return nil, fmt.Errorf("failed to get uncompressed content: %w", err)
	}
	defer content.Close()
	data, err := io.ReadAll(content)
	if err != nil {
		return nil, fmt.Errorf("failed to read layer content: %w", err)
	}
	return &Layer{Path: path, Content: data}, nil
}

func (i *Image) annotatedLayersByPrefix(layerDescs []v1.Descriptor, prefix string) ([]Layer, map[string]struct{}, error) {
	layers := make([]Layer, 0)
	seen := make(map[string]struct{})
	for _, layerDesc := range layerDescs {
		filePath, ok := annotatedPath(layerDesc)
		if !ok || !strings.HasPrefix(filePath, prefix) {
			continue
		}
		layer, err := i.readAnnotatedLayer(layerDesc, filePath)
		if err != nil {
			return nil, nil, err
		}
		seen[filePath] = struct{}{}
		layers = append(layers, *layer)
	}
	return layers, seen, nil
}

// GetFilesByExtensions extracts every file whose path ends with one of the supplied
// extensions. It supports both ORAS-style annotated layers and Docker image tar
// layers, so callers do not need the external oras binary at runtime.
func (i *Image) GetFilesByExtensions(exts ...string) ([]Layer, error) {
	manifest, err := i.inner.Manifest()
	if err != nil {
		return nil, fmt.Errorf("failed to get manifest: %w", err)
	}

	var layers []Layer
	seen := make(map[string]struct{})

	for _, layerDesc := range manifest.Layers {
		if layerDesc.Annotations == nil {
			continue
		}
		filePath, ok := layerDesc.Annotations[annotation.OCIImageTitleAnnotationKey]
		if !ok || !hasAnySuffix(filePath, exts) {
			continue
		}

		layer, err := i.inner.LayerByDigest(layerDesc.Digest)
		if err != nil {
			return nil, fmt.Errorf("failed to get layer by digest: %w", err)
		}
		content, err := layer.Uncompressed()
		if err != nil {
			return nil, fmt.Errorf("failed to get uncompressed content: %w", err)
		}
		data, readErr := io.ReadAll(content)
		content.Close()
		if readErr != nil {
			return nil, fmt.Errorf("failed to read layer content: %w", readErr)
		}

		normalizedPath := strings.TrimPrefix(filePath, "/")
		seen[normalizedPath] = struct{}{}
		layers = append(layers, Layer{Path: normalizedPath, Content: data})
	}

	tarLayers, err := i.extractFilesFromLayersByExtensions(exts...)
	if err != nil {
		return nil, fmt.Errorf("failed to extract files from layers by extensions %v: %w", exts, err)
	}
	for _, l := range tarLayers {
		if _, exists := seen[l.Path]; !exists {
			seen[l.Path] = struct{}{}
			layers = append(layers, l)
		}
	}

	return layers, nil
}

func readLayerContent(layer v1.Layer) ([]byte, error) {
	rc, err := layer.Uncompressed()
	if err != nil {
		return nil, err
	}
	content, err := io.ReadAll(rc)
	rc.Close()
	return content, err
}

// extractFileFromLayers iterates all image layers, treats each as a tar archive,
// and searches for a file whose name matches the given path. Path matching normalizes
// both the tar entry name and the requested path by stripping leading "/" so that
// "/paths.yaml" and "paths.yaml" both match. Returns the first matching file found.
func (i *Image) extractFileFromLayers(path string) (*Layer, error) {
	layers, err := i.inner.Layers()
	if err != nil {
		return nil, fmt.Errorf("file %s not found in image: %w", path, err)
	}

	normalizedPath := strings.TrimPrefix(path, "/")

	for _, layer := range layers {
		content, err := readLayerContent(layer)
		if err != nil {
			continue
		}

		result, err := findFileInTar(content, normalizedPath)
		if err == nil && result != nil {
			return result, nil
		}
	}

	return nil, fmt.Errorf("file %s not found in any layer", path)
}

// extractFilesFromLayersByPrefix iterates all image layers, treats each as a tar
// archive, and collects files whose name starts with the given prefix. Prefix matching
// normalizes tar entry names by stripping leading "/".
func (i *Image) extractFilesFromLayersByPrefix(prefix string) ([]Layer, error) {
	layers, err := i.inner.Layers()
	if err != nil {
		return nil, fmt.Errorf("failed to list layers: %w", err)
	}

	normalizedPrefix := strings.TrimPrefix(prefix, "/")
	var result []Layer

	for _, layer := range layers {
		content, err := readLayerContent(layer)
		if err != nil {
			continue
		}

		found, err := findFilesInTarByPrefix(content, normalizedPrefix)
		if err != nil {
			continue
		}
		result = append(result, found...)
	}

	return result, nil
}

func (i *Image) extractFilesFromLayersByExtensions(exts ...string) ([]Layer, error) {
	layers, err := i.inner.Layers()
	if err != nil {
		return nil, fmt.Errorf("failed to list layers: %w", err)
	}

	var result []Layer
	for _, layer := range layers {
		content, err := readLayerContent(layer)
		if err != nil {
			continue
		}

		found, err := findFilesInTarByExtensions(content, exts...)
		if err != nil {
			continue
		}
		result = append(result, found...)
	}

	return result, nil
}

// findFileInTar searches a tar archive for a single file matching the given
// normalized path. Returns a Layer with the file content if found.
func findFileInTar(tarData []byte, normalizedPath string) (*Layer, error) {
	tr := tar.NewReader(bytes.NewReader(tarData))
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			break
		}
		hdrName := strings.TrimPrefix(hdr.Name, "/")
		if hdrName == normalizedPath {
			data, err := io.ReadAll(tr)
			if err != nil {
				continue
			}
			return &Layer{Path: normalizedPath, Content: data}, nil
		}
	}
	return nil, fmt.Errorf("file %s not found in tar", normalizedPath)
}

// findFilesInTarByPrefix searches a tar archive for all files whose names start
// with the given normalized prefix. Returns a slice of Layers with file contents.
func findFilesInTarByPrefix(tarData []byte, normalizedPrefix string) ([]Layer, error) {
	return findFilesInTar(tarData, func(path string, hdr *tar.Header) bool {
		return strings.HasPrefix(path, normalizedPrefix)
	})
}

func findFilesInTarByExtensions(tarData []byte, exts ...string) ([]Layer, error) {
	return findFilesInTar(tarData, func(path string, hdr *tar.Header) bool {
		return !hdr.FileInfo().IsDir() && hasAnySuffix(path, exts)
	})
}

func findFilesInTar(tarData []byte, match func(string, *tar.Header) bool) ([]Layer, error) {
	tr := tar.NewReader(bytes.NewReader(tarData))
	var result []Layer
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			break
		}
		hdrName := strings.TrimPrefix(hdr.Name, "/")
		if !match(hdrName, hdr) {
			continue
		}
		data, err := io.ReadAll(tr)
		if err != nil {
			continue
		}
		result = append(result, Layer{Path: hdrName, Content: data})
	}
	return result, nil
}

func hasAnySuffix(path string, suffixes []string) bool {
	for _, suffix := range suffixes {
		if strings.HasSuffix(path, suffix) {
			return true
		}
	}
	return false
}
