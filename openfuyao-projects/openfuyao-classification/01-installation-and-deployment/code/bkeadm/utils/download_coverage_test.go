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

package utils

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type errorReadCloser struct {
	readErr error
}

func (e errorReadCloser) Read([]byte) (int, error) {
	return 0, e.readErr
}

func (e errorReadCloser) Close() error {
	return nil
}

func TestHttpGetSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(http.Get, func(url string) (*http.Response, error) {
		assert.Equal(t, "http://example.com/index.html", url)
		return &http.Response{
			StatusCode: HTTPStatusOK,
			Body:       io.NopCloser(strings.NewReader("<html>ok</html>")),
		}, nil
	})

	body, err := httpGet("http://example.com/index.html")
	require.NoError(t, err)
	assert.Equal(t, "<html>ok</html>", body)
}

func TestHttpGetNetworkError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(http.Get, func(string) (*http.Response, error) {
		return nil, errors.New("network down")
	})

	_, err := httpGet("http://example.com/index.html")
	assert.Error(t, err)
}

func TestHttpGetNonOKStatus(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(http.Get, func(string) (*http.Response, error) {
		return &http.Response{
			StatusCode: testFourHundredFourValue,
			Body:       io.NopCloser(strings.NewReader("not found")),
		}, nil
	})

	_, err := httpGet("http://example.com/missing")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "status code")
}

func TestHttpGetReadBodyError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(http.Get, func(string) (*http.Response, error) {
		return &http.Response{
			StatusCode: HTTPStatusOK,
			Body:       errorReadCloser{readErr: errors.New("read failed")},
		}, nil
	})

	_, err := httpGet("http://example.com/index.html")
	assert.Error(t, err)
}

func TestDownloadAllFilesHttpGetError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(http.Get, func(string) (*http.Response, error) {
		return nil, errors.New("index unavailable")
	})

	err := DownloadAllFiles("http://example.com/repo/", t.TempDir())
	assert.Error(t, err)
}

func TestDownloadAllFilesEmptyHTML(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(http.Get, func(string) (*http.Response, error) {
		return &http.Response{
			StatusCode: HTTPStatusOK,
			Body:       io.NopCloser(strings.NewReader("")),
		}, nil
	})

	err := DownloadAllFiles("http://example.com/repo/", t.TempDir())
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "Failed to get download list")
}

func TestDownloadAllFilesSkipsNonRPM(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(http.Get, func(url string) (*http.Response, error) {
		if strings.HasSuffix(url, "repo/") {
			return &http.Response{
				StatusCode: HTTPStatusOK,
				Body: io.NopCloser(strings.NewReader(
					`<a href="readme.txt">readme</a>`,
				)),
			}, nil
		}
		t.Fatalf("unexpected download url: %s", url)
		return nil, nil
	})

	err := DownloadAllFiles("http://example.com/repo/", t.TempDir())
	assert.NoError(t, err)
}

func TestDownloadAllFilesDownloadsRPMs(t *testing.T) {
	targetDir := t.TempDir()
	baseURL := "http://example.com/repo/"

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(http.Get, func(url string) (*http.Response, error) {
		switch url {
		case baseURL:
			return &http.Response{
				StatusCode: HTTPStatusOK,
				Body: io.NopCloser(strings.NewReader(
					`<a href="pkg-1.rpm">pkg-1</a>` +
						`<a href="notes.md">notes</a>` +
						`<a href="pkg-2.rpm">pkg-2</a>`,
				)),
			}, nil
		case baseURL + "pkg-1.rpm":
			return &http.Response{
				StatusCode: HTTPStatusOK,
				Body:       io.NopCloser(strings.NewReader("rpm-one")),
			}, nil
		case baseURL + "pkg-2.rpm":
			return &http.Response{
				StatusCode: HTTPStatusOK,
				Body:       io.NopCloser(strings.NewReader("rpm-two")),
			}, nil
		default:
			return nil, fmt.Errorf("unexpected url: %s", url)
		}
	})

	err := DownloadAllFiles(baseURL, targetDir)
	require.NoError(t, err)

	content1, err := os.ReadFile(filepath.Join(targetDir, "pkg-1.rpm"))
	require.NoError(t, err)
	assert.Equal(t, "rpm-one", string(content1))

	content2, err := os.ReadFile(filepath.Join(targetDir, "pkg-2.rpm"))
	require.NoError(t, err)
	assert.Equal(t, "rpm-two", string(content2))
}

func TestDownloadAllFilesDownloadError(t *testing.T) {
	baseURL := "http://example.com/repo/"

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(http.Get, func(url string) (*http.Response, error) {
		if url == baseURL {
			return &http.Response{
				StatusCode: HTTPStatusOK,
				Body: io.NopCloser(strings.NewReader(
					`<a href="broken.rpm">broken</a>`,
				)),
			}, nil
		}
		return nil, errors.New("rpm download failed")
	})

	err := DownloadAllFiles(baseURL, t.TempDir())
	assert.Error(t, err)
}

func TestDownloadSignalFileSuccess(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, http.MethodGet, r.Method)
		assert.Equal(t, "Mozilla/5.0 (compatible; Go-Client/1.0)", r.Header.Get("User-Agent"))
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("signal-data"))
	}))
	defer server.Close()

	targetPath := filepath.Join(t.TempDir(), "nested", "signal.yaml")
	err := DownloadSignalFile(server.URL, targetPath)
	require.NoError(t, err)

	content, err := os.ReadFile(targetPath)
	require.NoError(t, err)
	assert.Equal(t, "signal-data", string(content))
}

func TestDownloadSignalFileNewRequestError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(http.NewRequest, func(string, string, io.Reader) (*http.Request, error) {
		return nil, errors.New("bad request")
	})

	err := DownloadSignalFile("http://example.com/signal.yaml", filepath.Join(t.TempDir(), "signal.yaml"))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to create HTTP request")
}

func TestDownloadSignalFileDoError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyMethod(http.DefaultClient, "Do", func(_ *http.Client, _ *http.Request) (*http.Response, error) {
		return nil, errors.New("connection refused")
	})

	err := DownloadSignalFile("http://example.com/signal.yaml", filepath.Join(t.TempDir(), "signal.yaml"))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to fetch remote content")
}

func TestDownloadSignalFileNonOKStatus(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte("missing"))
	}))
	defer server.Close()

	err := DownloadSignalFile(server.URL, filepath.Join(t.TempDir(), "signal.yaml"))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "HTTP request failed")
}

func TestDownloadSignalFileReadBodyError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyMethod(http.DefaultClient, "Do", func(_ *http.Client, _ *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       errorReadCloser{readErr: errors.New("body read failed")},
		}, nil
	})

	err := DownloadSignalFile("http://example.com/signal.yaml", filepath.Join(t.TempDir(), "signal.yaml"))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to read response body")
}

func TestDownloadSignalFileMkdirError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("data"))
	}))
	defer server.Close()

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(os.MkdirAll, func(string, os.FileMode) error {
		return errors.New("mkdir failed")
	})

	err := DownloadSignalFile(server.URL, filepath.Join(t.TempDir(), "nested", "signal.yaml"))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to create directory")
}

func TestDownloadSignalFileWriteError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("data"))
	}))
	defer server.Close()

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(os.WriteFile, func(string, []byte, os.FileMode) error {
		return errors.New("write failed")
	})

	err := DownloadSignalFile(server.URL, filepath.Join(t.TempDir(), "signal.yaml"))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to write file")
}
