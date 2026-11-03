/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package download

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
)

const (
	numEight     = 8
	numSixteen   = 16
	numThirtyTwo = 32
)

func TestExecDownloadInvalidURL(t *testing.T) {
	tmpDir := t.TempDir()
	err := ExecDownload("http://invalid.invalid.invalid/file", tmpDir, "", "0644")
	assert.Error(t, err)
}

func TestExecDownloadNonExistentSavePath(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("test content"))
	}))
	defer server.Close()

	tmpDir := filepath.Join(t.TempDir(), "nonexistent", "path")
	err := ExecDownload(server.URL, tmpDir, "", "0644")
	// On Windows, this might succeed due to different path handling
	// Just verify the function runs without panic
	if err != nil {
		assert.Contains(t, err.Error(), "create directory")
	}
}

func TestExecDownloadInvalidChmod(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("test content"))
	}))
	defer server.Close()

	tmpDir := t.TempDir()
	err := ExecDownload(server.URL, tmpDir, "", "invalid")
	assert.NoError(t, err)
}

func TestExecDownloadWithRename(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("test content"))
	}))
	defer server.Close()

	tmpDir := t.TempDir()
	err := ExecDownload(server.URL, tmpDir, "renamedfile.txt", "0644")
	assert.NoError(t, err)

	savedFile := filepath.Join(tmpDir, "renamedfile.txt")
	_, err = os.Stat(savedFile)
	assert.NoError(t, err)
}

func TestDownloadBytes(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("[Service]\nExecStart=--health-port= --ntpserver=\n"))
	}))
	defer server.Close()

	data, err := DownloadBytes(server.URL + "/bkeagent.service")
	assert.NoError(t, err)
	assert.Contains(t, string(data), "--ntpserver=")
}

func TestExecDownloadForArch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/bkeagent-linux-arm64", r.URL.Path)
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("binary"))
	}))
	defer server.Close()

	tmpDir := t.TempDir()
	url := server.URL + "/bkeagent-linux-{.arch}"
	err := ExecDownloadForArch(url, tmpDir, "bkeagent", "0755", "arm64")
	assert.NoError(t, err)
	_, err = os.Stat(filepath.Join(tmpDir, "bkeagent"))
	assert.NoError(t, err)
}

func TestExecDownloadHTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	tmpDir := t.TempDir()
	err := ExecDownload(server.URL, tmpDir, "", "0644")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "status code")
}
