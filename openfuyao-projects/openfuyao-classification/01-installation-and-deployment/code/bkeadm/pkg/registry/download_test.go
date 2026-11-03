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
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/containers/image/v5/types"
	digest "github.com/opencontainers/go-digest"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func buildLayerTar(t *testing.T, name, body string) []byte {
	t.Helper()
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	hdr := &tar.Header{
		Name:     name,
		Mode:     0o644,
		Size:     int64(len(body)),
		Typeflag: tar.TypeReg,
	}
	require.NoError(t, tw.WriteHeader(hdr))
	_, err := tw.Write([]byte(body))
	require.NoError(t, err)
	require.NoError(t, tw.Close())
	return buf.Bytes()
}

func buildGzipLayer(t *testing.T, tarPayload []byte) []byte {
	t.Helper()
	var gz bytes.Buffer
	gw := gzip.NewWriter(&gz)
	_, err := gw.Write(tarPayload)
	require.NoError(t, err)
	require.NoError(t, gw.Close())
	return gz.Bytes()
}

// readerWithMinSize pads payload so FormatReaders.matchHeader can ReadFull(512).
func readerWithMinSize(payload []byte) io.ReadCloser {
	if len(payload) >= MaxExpectedHdrSize {
		return io.NopCloser(bytes.NewReader(payload))
	}
	padded := make([]byte, MaxExpectedHdrSize)
	copy(padded, payload)
	return io.NopCloser(bytes.NewReader(padded))
}

func buildLargeLayerTar(t *testing.T, name, body string) []byte {
	t.Helper()
	// constructReaders may ReadFull(512) multiple times on decompressed tar.
	if len(body) < MaxExpectedHdrSize {
		body = body + strings.Repeat("x", MaxExpectedHdrSize-len(body))
	}
	return buildLayerTar(t, name, body)
}

func TestExtractFilesFromTar(t *testing.T) {
	tarBytes := buildLayerTar(t, "usr/bin/target.txt", "layer-payload")
	destDir := t.TempDir()

	config := &LayerProcessConfig{
		DownloadToDir:   destDir,
		DownloadFileMap: map[string]string{"target.txt": ""},
	}

	err := extractFilesFromTar(tar.NewReader(bytes.NewReader(tarBytes)), config)
	require.NoError(t, err)
	assert.Equal(t, "usr/bin/target.txt", config.DownloadFileMap["target.txt"])

	out, err := os.ReadFile(filepath.Join(destDir, "target.txt"))
	require.NoError(t, err)
	assert.Equal(t, "layer-payload", string(out))
}

func TestExtractFilesFromTarStopsWhenAllFound(t *testing.T) {
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	for _, entry := range []struct{ name, body string }{
		{"a/first.txt", "one"},
		{"b/second.txt", "two"},
	} {
		hdr := &tar.Header{Name: entry.name, Mode: 0o644, Size: int64(len(entry.body)), Typeflag: tar.TypeReg}
		require.NoError(t, tw.WriteHeader(hdr))
		_, err := tw.Write([]byte(entry.body))
		require.NoError(t, err)
	}
	require.NoError(t, tw.Close())

	destDir := t.TempDir()
	config := &LayerProcessConfig{
		DownloadToDir: destDir,
		DownloadFileMap: map[string]string{
			"first.txt":  "",
			"second.txt": "",
		},
	}

	err := extractFilesFromTar(tar.NewReader(&buf), config)
	require.NoError(t, err)
	assert.NotEmpty(t, config.DownloadFileMap["first.txt"])
	assert.NotEmpty(t, config.DownloadFileMap["second.txt"])
}

func TestNewFormatReadersGzipTar(t *testing.T) {
	tarPayload := buildLargeLayerTar(t, "etc/config.yaml", "key: value")
	gzPayload := buildGzipLayer(t, tarPayload)

	fr, err := NewFormatReaders(readerWithMinSize(gzPayload), 0)
	require.NoError(t, err)
	defer fr.Close()

	assert.True(t, fr.ArchiveGz)

	tr := tar.NewReader(fr.TopReader())
	hdr, err := tr.Next()
	require.NoError(t, err)
	assert.Equal(t, "etc/config.yaml", hdr.Name)
}

func TestMatchHeaderDetectsGzip(t *testing.T) {
	gzPayload := buildGzipLayer(t, []byte("plain"))

	fr := &FormatReaders{buf: make([]byte, MaxExpectedHdrSize)}
	fr.appendReader(rdrStream, readerWithMinSize(gzPayload))

	known := CopyKnownHdrs()
	hdr, err := fr.matchHeader(&known)
	require.NoError(t, err)
	require.NotNil(t, hdr)
	assert.Equal(t, "gz", hdr.Format)
}

func TestGzReaderDecompressesPayload(t *testing.T) {
	gzPayload := buildGzipLayer(t, []byte("hello-gzip"))

	fr := &FormatReaders{buf: make([]byte, MaxExpectedHdrSize)}
	fr.appendReader(rdrStream, readerWithMinSize(gzPayload))
	known := CopyKnownHdrs()
	hdr, err := fr.matchHeader(&known)
	require.NoError(t, err)
	require.NotNil(t, hdr)

	fr.fileFormatSelector(hdr)
	assert.True(t, fr.ArchiveGz)

	// Read exact decompressed length; ReadAll would consume trailing zero-padding
	// from readerWithMinSize and trigger "gzip: invalid header" on a second member.
	out := make([]byte, len("hello-gzip"))
	_, err = io.ReadFull(fr.TopReader(), out)
	require.NoError(t, err)
	assert.Equal(t, "hello-gzip", string(out))
}

func TestMatchHeaderPlainTar(t *testing.T) {
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	require.NoError(t, tw.WriteHeader(&tar.Header{Name: "x", Mode: 0o644, Size: 0, Typeflag: tar.TypeReg}))
	require.NoError(t, tw.Close())

	fr := &FormatReaders{buf: make([]byte, MaxExpectedHdrSize)}
	fr.appendReader(rdrStream, io.NopCloser(bytes.NewReader(buf.Bytes())))

	known := CopyKnownHdrs()
	hdr, err := fr.matchHeader(&known)
	require.NoError(t, err)
	require.NotNil(t, hdr)
	assert.Equal(t, "tar", hdr.Format)
}

func TestExtractFile(t *testing.T) {
	tarBytes := buildLayerTar(t, "opt/bin/tool", "binary-data")
	destDir := t.TempDir()

	config := &LayerProcessConfig{
		DownloadToDir:   destDir,
		DownloadFileMap: map[string]string{"tool": ""},
	}
	tr := tar.NewReader(bytes.NewReader(tarBytes))
	hdr, err := tr.Next()
	require.NoError(t, err)

	err = extractFile(tr, hdr, "tool", config)
	require.NoError(t, err)
	assert.Equal(t, "opt/bin/tool", config.DownloadFileMap["tool"])

	out, err := os.ReadFile(filepath.Join(destDir, "tool"))
	require.NoError(t, err)
	assert.Equal(t, "binary-data", string(out))
}

func TestProcessTargetFilesSkipsExtracted(t *testing.T) {
	tarBytes := buildLayerTar(t, "etc/keep.txt", "done")
	destDir := t.TempDir()
	config := &LayerProcessConfig{
		DownloadToDir: destDir,
		DownloadFileMap: map[string]string{
			"keep.txt": "already/extracted",
		},
	}
	tr := tar.NewReader(bytes.NewReader(tarBytes))
	hdr, err := tr.Next()
	require.NoError(t, err)

	assert.NoError(t, processTargetFiles(tr, hdr, config))
	assert.Equal(t, "already/extracted", config.DownloadFileMap["keep.txt"])
}

type stubBlobSource struct {
	payload  []byte
	err      error
	closeErr error
}

func (s *stubBlobSource) Reference() types.ImageReference { return nil }
func (s *stubBlobSource) GetManifest(context.Context, *digest.Digest) ([]byte, string, error) {
	return nil, "", nil
}
func (s *stubBlobSource) GetBlob(context.Context, types.BlobInfo, types.BlobInfoCache) (io.ReadCloser, int64, error) {
	if s.err != nil {
		return nil, 0, s.err
	}
	return readerWithMinSize(s.payload), int64(len(s.payload)), nil
}
func (s *stubBlobSource) HasThreadSafeGetBlob() bool { return true }
func (s *stubBlobSource) LayerInfosForCopy(context.Context, *digest.Digest) ([]types.BlobInfo, error) {
	return nil, nil
}
func (s *stubBlobSource) GetSignatures(context.Context, *digest.Digest) ([][]byte, error) {
	return nil, nil
}

func TestOpenLayerTarReader(t *testing.T) {
	tarBytes := buildLargeLayerTar(t, "app/config.yaml", "key: value")
	payload := buildGzipLayer(t, tarBytes)

	config := &LayerProcessConfig{
		ImageSource: &stubBlobSource{payload: payload},
		Layer:       types.BlobInfo{Digest: "sha256:test"},
		Cache:       nil,
	}

	tr, fr, err := openLayerTarReader(context.Background(), config)
	require.NoError(t, err)
	require.NotNil(t, tr)
	require.NotNil(t, fr)
	defer fr.Close()

	hdr, err := tr.Next()
	require.NoError(t, err)
	assert.Equal(t, "app/config.yaml", hdr.Name)
}

func TestOpenLayerTarReaderGetBlobFails(t *testing.T) {
	config := &LayerProcessConfig{
		ImageSource: &stubBlobSource{err: assert.AnError},
		Layer:       types.BlobInfo{Digest: "sha256:test"},
	}
	_, _, err := openLayerTarReader(context.Background(), config)
	assert.Error(t, err)
}

func TestProcessLayerEndToEnd(t *testing.T) {
	tarBytes := buildLargeLayerTar(t, "workspace/readme.md", "hello layer")
	payload := buildGzipLayer(t, tarBytes)
	destDir := t.TempDir()

	config := &LayerProcessConfig{
		ImageSource:     &stubBlobSource{payload: payload},
		Layer:           types.BlobInfo{Digest: "sha256:test"},
		DownloadToDir:   destDir,
		DownloadFileMap: map[string]string{"readme.md": ""},
	}

	err := processLayer(context.Background(), config)
	require.NoError(t, err)
	assert.NotEmpty(t, config.DownloadFileMap["readme.md"])
}

func TestOptionsDownloadEnsureDirAndMap(t *testing.T) {
	destDir := t.TempDir()
	od := &OptionsDownload{
		DownloadToDir:       destDir,
		DownloadInImageFile: "/etc/app.yaml,/bin/tool",
	}
	assert.NoError(t, od.ensureDownloadDir())
	m := od.buildDownloadFileMap()
	assert.Len(t, m, 2)
	assert.Contains(t, m, "etc/app.yaml")
	assert.Contains(t, m, "bin/tool")
}

func TestOptionsDownloadSetupImageSourceInvalid(t *testing.T) {
	od := &OptionsDownload{Image: "://bad"}
	_, _, err := od.setupImageSource(context.Background())
	assert.Error(t, err)
}

func TestOptionsDownloadMocked(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	destDir := t.TempDir()
	od := &OptionsDownload{
		DownloadToDir:       destDir,
		DownloadInImageFile: "readme.md",
		Image:               "example.com/img:tag",
	}

	patches.ApplyPrivateMethod(reflect.TypeOf(&OptionsDownload{}), "setupImageSource",
		func(_ *OptionsDownload, _ context.Context) (types.ImageSource, *types.SystemContext, error) {
			return &stubBlobSource{}, &types.SystemContext{}, nil
		})
	patches.ApplyPrivateMethod(reflect.TypeOf(&OptionsDownload{}), "downloadFilesFromLayers",
		func(_ *OptionsDownload, _ context.Context, _ types.ImageSource, _ *types.SystemContext,
			m map[string]string) error {
			m["readme.md"] = "layer/readme.md"
			return nil
		})

	assert.NoError(t, od.Download())
}

func TestCloseImageWithError(t *testing.T) {
	src := &stubBlobSource{closeErr: errors.New("close failed")}
	closeImage(src)
}

func TestExtractFilesFromTarReadError(t *testing.T) {
	config := &LayerProcessConfig{
		DownloadToDir:   t.TempDir(),
		DownloadFileMap: map[string]string{"missing.txt": ""},
	}
	broken := tar.NewReader(bytes.NewReader([]byte("not-a-tar")))
	err := extractFilesFromTar(broken, config)
	assert.Error(t, err)
}

func TestAppendReaderNilInterface(t *testing.T) {
	fr := &FormatReaders{buf: make([]byte, MaxExpectedHdrSize)}
	fr.appendReader(rdrStream, nil)
	assert.Empty(t, fr.readers)
}

func TestAppendReaderNonReadCloser(t *testing.T) {
	fr := &FormatReaders{buf: make([]byte, MaxExpectedHdrSize)}
	fr.appendReader(rdrStream, struct{}{})
	assert.Empty(t, fr.readers)
}

func (s *stubBlobSource) Close() error {
	if s.closeErr != nil {
		return s.closeErr
	}
	return nil
}
