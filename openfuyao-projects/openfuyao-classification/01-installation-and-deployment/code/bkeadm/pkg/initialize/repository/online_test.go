/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          <http://license.coscl.org.cn/MulanPSL2>
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package repository

import (
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/bkeadm/pkg/build"
	"gopkg.openfuyao.cn/bkeadm/pkg/common"
	econd "gopkg.openfuyao.cn/bkeadm/pkg/executor/containerd"
	"gopkg.openfuyao.cn/bkeadm/pkg/registry"
	"gopkg.openfuyao.cn/bkeadm/pkg/server"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

const (
	testZeroValue    = 0
	testOneValue     = 1
	testTwoValue     = 2
	testThreeValue   = 3
	testDefaultPort  = 443
	testRegistryPort = "5000"
)

const (
	testIPv4SegmentA = 192
	testIPv4SegmentB = 168
	testIPv4SegmentC = 1
	testIPv4SegmentD = 100
)

const (
	testIPv4SegmentD2 = 102
)

var (
	testIPAddress  = fmt.Sprintf("%d.%d.%d.%d", testIPv4SegmentA, testIPv4SegmentB, testIPv4SegmentC, testIPv4SegmentD)
	testIPAddress2 = fmt.Sprintf("%d.%d.%d.%d", testIPv4SegmentA, testIPv4SegmentB, testIPv4SegmentC, testIPv4SegmentD2)
)

func TestParseOnlineConfig(t *testing.T) {
	tests := []struct {
		name        string
		domain      string
		image       string
		repo        string
		source      string
		chartRepo   string
		mockLoopIP  func(string) ([]string, error)
		expectError bool
	}{
		{
			name:        "empty repo and chartRepo",
			domain:      "example.com",
			image:       "",
			repo:        "",
			source:      "",
			chartRepo:   "",
			mockLoopIP:  func(s string) ([]string, error) { return nil, nil },
			expectError: false,
		},
		{
			name:        "repo with image tag and IP resolution success",
			domain:      "example.com",
			image:       "",
			repo:        "registry.example.com/test/image:v1.0",
			source:      "",
			chartRepo:   "",
			mockLoopIP:  func(s string) ([]string, error) { return []string{testIPAddress}, nil },
			expectError: false,
		},
		{
			name:        "repo with IP address",
			domain:      "example.com",
			image:       "",
			repo:        testIPAddress + ":5000/test/repo",
			source:      "http://" + testIPAddress + ":5000/source",
			chartRepo:   "",
			mockLoopIP:  func(s string) ([]string, error) { return nil, nil },
			expectError: false,
		},
		{
			name:        "repo with domain name resolution success",
			domain:      "example.com",
			image:       "",
			repo:        "registry.example.com/test/repo",
			source:      "",
			chartRepo:   "",
			mockLoopIP:  func(s string) ([]string, error) { return []string{testIPAddress}, nil },
			expectError: false,
		},
		{
			name:        "repo with domain name resolution failure",
			domain:      "example.com",
			image:       "",
			repo:        "nonexistent.example.com/test/repo",
			source:      "",
			chartRepo:   "",
			mockLoopIP:  func(s string) ([]string, error) { return []string{}, errors.New("resolution failed") },
			expectError: true,
		},
		{
			name:        "chartRepo with IP address",
			domain:      "example.com",
			image:       "",
			repo:        "",
			source:      "",
			chartRepo:   testIPAddress2 + ":8443/charts/repo",
			mockLoopIP:  func(s string) ([]string, error) { return nil, nil },
			expectError: false,
		},
		{
			name:        "chartRepo with domain name and resolution",
			domain:      "example.com",
			image:       "",
			repo:        "",
			source:      "",
			chartRepo:   "chart.example.com/charts/repo",
			mockLoopIP:  func(s string) ([]string, error) { return []string{testIPAddress}, nil },
			expectError: false,
		},
		{
			name:        "source is preserved when provided",
			domain:      "example.com",
			image:       "",
			repo:        "",
			source:      "http://source.example.com/data",
			chartRepo:   "",
			mockLoopIP:  func(s string) ([]string, error) { return nil, nil },
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(utils.LoopIP, tt.mockLoopIP)

			_, err := ParseOnlineConfig(tt.domain, tt.image, tt.repo, tt.source, tt.chartRepo)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestSetupCACertificate(t *testing.T) {
	tests := []struct {
		name         string
		config       CertificateConfig
		mockMkdirAll func(string, os.FileMode) error
		mockCopyFile func(string, string) error
		expectError  bool
	}{
		{
			name:         "nil CA file path",
			config:       CertificateConfig{CAFile: ""},
			mockMkdirAll: nil,
			mockCopyFile: nil,
			expectError:  false,
		},
		{
			name: "successful certificate setup",
			config: CertificateConfig{
				CAFile:       "/tmp/ca.crt",
				RegistryHost: "registry.example.com",
				RegistryPort: testRegistryPort,
			},
			mockMkdirAll: func(s string, m os.FileMode) error { return nil },
			mockCopyFile: func(s string, s2 string) error { return nil },
			expectError:  false,
		},
		{
			name: "mkdir fails for first directory",
			config: CertificateConfig{
				CAFile:       "/tmp/ca.crt",
				RegistryHost: "registry.example.com",
				RegistryPort: testRegistryPort,
			},
			mockMkdirAll: func(s string, m os.FileMode) error { return errors.New("mkdir error") },
			mockCopyFile: nil,
			expectError:  true,
		},
		{
			name: "copy file fails",
			config: CertificateConfig{
				CAFile:       "/tmp/ca.crt",
				RegistryHost: "registry.example.com",
				RegistryPort: testRegistryPort,
			},
			mockMkdirAll: func(s string, m os.FileMode) error { return nil },
			mockCopyFile: func(s string, s2 string) error { return errors.New("copy error") },
			expectError:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			if tt.mockMkdirAll != nil {
				patches.ApplyFunc(os.MkdirAll, tt.mockMkdirAll)
			}
			if tt.mockCopyFile != nil {
				patches.ApplyFunc(copyFile, tt.mockCopyFile)
			}

			err := SetupCACertificate(&tt.config)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestCopyFile(t *testing.T) {
	tests := []struct {
		name        string
		mockOpen    func(string) (*os.File, error)
		mockCreate  func(string) (*os.File, error)
		expectError bool
	}{
		{
			name: "source file open error",
			mockOpen: func(s string) (*os.File, error) {
				return nil, errors.New("open error")
			},
			expectError: true,
		},
		{
			name: "destination file create error",
			mockOpen: func(s string) (*os.File, error) {
				tmpFile, _ := os.CreateTemp("", "source")
				return tmpFile, nil
			},
			mockCreate: func(s string) (*os.File, error) {
				return nil, errors.New("create error")
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			if tt.mockOpen != nil {
				patches.ApplyFunc(os.Open, tt.mockOpen)
			}
			if tt.mockCreate != nil {
				patches.ApplyFunc(os.Create, tt.mockCreate)
			}

			err := copyFile("/tmp/source.txt", "/tmp/dest.txt")

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestParseRegistryHostPort(t *testing.T) {
	tests := []struct {
		name       string
		imageRepo  string
		expectHost string
		expectPort string
	}{
		{
			name:       "empty input",
			imageRepo:  "",
			expectHost: "",
			expectPort: "",
		},
		{
			name:       "registry with https prefix",
			imageRepo:  "https://registry.example.com/repo/image:v1",
			expectHost: "registry.example.com",
			expectPort: "443",
		},
		{
			name:       "registry with http prefix",
			imageRepo:  "http://registry.example.com:8080/repo/image:v1",
			expectHost: "registry.example.com",
			expectPort: "8080",
		},
		{
			name:       "registry without prefix and custom port",
			imageRepo:  "registry.example.com:5000/repo/image:v1",
			expectHost: "registry.example.com",
			expectPort: "5000",
		},
		{
			name:       "registry without port uses default 443",
			imageRepo:  "registry.example.com/repo/image:v1",
			expectHost: "registry.example.com",
			expectPort: "443",
		},
		{
			name:       "IP address with port",
			imageRepo:  testIPAddress + ":5000/repo/image:v1",
			expectHost: testIPAddress,
			expectPort: "5000",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			host, port := ParseRegistryHostPort(tt.imageRepo)
			assert.Equal(t, tt.expectHost, host)
			assert.Equal(t, tt.expectPort, port)
		})
	}
}

func TestCleanTempYumDataFile(t *testing.T) {
	tests := []struct {
		name        string
		mockExists  bool
		expectError bool
	}{
		{
			name:        "temp file does not exist",
			mockExists:  false,
			expectError: false,
		},
		{
			name:        "temp file removed successfully",
			mockExists:  true,
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			patches.ApplyFunc(utils.Exists, func(s string) bool {
				return tt.mockExists
			})

			patches.ApplyFunc(os.RemoveAll, func(s string) error {
				return nil
			})

			err := cleanTempYumDataFile()

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestBuildDownloadOptions(t *testing.T) {
	tests := []struct {
		name        string
		oc          OtherRepo
		certConfig  *CertificateConfig
		expectImage string
		expectUser  string
	}{
		{
			name: "build options with all fields",
			oc: OtherRepo{
				Image: "registry.example.com/image:v1",
			},
			certConfig: &CertificateConfig{
				TLSVerify: true,
				Username:  "user",
				Password:  "pass",
				CAFile:    "/tmp/ca.crt",
			},
			expectImage: "registry.example.com/image:v1",
			expectUser:  "user",
		},
		{
			name: "build options with empty config",
			oc: OtherRepo{
				Image: "registry.example.com/image:v1",
			},
			certConfig:  &CertificateConfig{},
			expectImage: "registry.example.com/image:v1",
			expectUser:  "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := buildDownloadOptions(tt.oc, tt.certConfig)
			assert.Equal(t, tt.expectImage, result.Image)
			assert.Equal(t, tt.expectUser, result.Username)
			assert.Equal(t, tt.certConfig.TLSVerify, result.SrcTLSVerify)
		})
	}
}

func TestFinalizeYumDataFile(t *testing.T) {
	tests := []struct {
		name        string
		mockRename  func(string, string) error
		expectError bool
	}{
		{
			name: "rename error",
			mockRename: func(s string, s2 string) error {
				return errors.New("rename error")
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			patches.ApplyFunc(os.Rename, tt.mockRename)

			patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

			err := finalizeYumDataFile()

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestSourceBaseFile(t *testing.T) {
	tests := []struct {
		name         string
		httpRepo     string
		mockExists   func(string) bool
		mockDownload func(string, string) error
		expectError  bool
	}{
		{
			name:        "all files already exist",
			httpRepo:    "http://example.com",
			mockExists:  func(s string) bool { return true },
			expectError: false,
		},
		{
			name:     "download chart fails but continues",
			httpRepo: "http://example.com",
			mockExists: func(s string) bool {
				if strings.Contains(s, "charts.tar.gz") {
					return false
				}
				return true
			},
			mockDownload: func(s string, s2 string) error {
				if strings.Contains(s, "charts.tar.gz") {
					return errors.New("download error")
				}
				return nil
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			patches.ApplyFunc(utils.Exists, tt.mockExists)

			if tt.mockDownload != nil {
				patches.ApplyFunc(utils.DownloadFile, tt.mockDownload)
			}

			patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

			err := sourceBaseFile(tt.httpRepo)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestCheckLocalRuntimeFilesExist(t *testing.T) {
	tests := []struct {
		name              string
		mockReadDir       func(string) ([]os.DirEntry, error)
		mockValidateExtra func(map[string]string) error
		expectResult      bool
		expectError       bool
	}{
		{
			name: "all files exist",
			mockReadDir: func(s string) ([]os.DirEntry, error) {
				return []os.DirEntry{
					&mockDirEntry{name: "containerd-1.7.0-linux-amd64.tar.gz"},
					&mockDirEntry{name: "cni-plugins-linux-amd64-v1.3.0.tgz"},
					&mockDirEntry{name: "kubectl-v1.27.0-linux-amd64"},
				}, nil
			},
			mockValidateExtra: func(m map[string]string) error { return nil },
			expectResult:      true,
			expectError:       false,
		},
		{
			name: "missing containerd files",
			mockReadDir: func(s string) ([]os.DirEntry, error) {
				return []os.DirEntry{
					&mockDirEntry{name: "cni-plugins-linux-amd64-v1.3.0.tgz"},
					&mockDirEntry{name: "kubectl-v1.27.0-linux-amd64"},
				}, nil
			},
			mockValidateExtra: func(m map[string]string) error { return nil },
			expectResult:      false,
			expectError:       false,
		},
		{
			name: "read dir error",
			mockReadDir: func(s string) ([]os.DirEntry, error) {
				return nil, errors.New("read dir error")
			},
			expectResult: false,
			expectError:  true,
		},
		{
			name: "invalid containerd files are skipped",
			mockReadDir: func(s string) ([]os.DirEntry, error) {
				return []os.DirEntry{
					&mockDirEntry{name: "invalid-file"},
					&mockDirEntry{name: "cni-plugins-linux-amd64-v1.3.0.tgz"},
					&mockDirEntry{name: "kubectl-v1.27.0-linux-amd64"},
				}, nil
			},
			mockValidateExtra: func(m map[string]string) error { return errors.New("invalid") },
			expectResult:      false,
			expectError:       false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(os.ReadDir, tt.mockReadDir)

			if tt.mockValidateExtra != nil {
				patches.ApplyFunc(validateCustomExtra, tt.mockValidateExtra)
			}

			result, err := checkLocalRuntimeFilesExist()

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expectResult, result)
			}
		})
	}
}

type testHTTPHandler struct {
	content string
	status  int
}

func (h *testHTTPHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(h.status)
	w.Write([]byte(h.content))
}

func TestFetchRemoteFileList(t *testing.T) {
	tests := []struct {
		name        string
		handler     *testHTTPHandler
		expectError bool
	}{
		{
			name: "HTTP error response",
			handler: &testHTTPHandler{
				status: http.StatusNotFound,
			},
			expectError: true,
		},
		{
			name:        "HTTP request error",
			handler:     nil,
			expectError: true,
		},
		{
			name: "empty HTML content",
			handler: &testHTTPHandler{
				content: "",
				status:  http.StatusOK,
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var server *httptest.Server
			if tt.handler != nil {
				server = httptest.NewServer(tt.handler)
				defer server.Close()
			}

			var url string
			if server != nil {
				url = server.URL + "/files/"
			} else {
				url = "http://invalid.example.com/files/"
			}

			_, err := fetchRemoteFileList(url)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestParseFileListFromHTML(t *testing.T) {
	tests := []struct {
		name     string
		htmlData string
		check    func(*runtimeFiles) bool
	}{
		{
			name: "parse all file types",
			htmlData: `<html>
			<body>
			<a href="containerd-1.7.0-linux-amd64.tar.gz">containerd-1.7.0-linux-amd64.tar.gz</a>
			<a href="containerd-1.6.0-linux-amd64.tar.gz">containerd-1.6.0-linux-amd64.tar.gz</a>
			<a href="cni-plugins-linux-amd64-v1.3.0.tgz">cni-plugins-linux-amd64-v1.3.0.tgz</a>
			<a href="cni-plugins-linux-amd64-v1.2.0.tgz">cni-plugins-linux-amd64-v1.2.0.tgz</a>
			<a href="kubectl-v1.27.0-linux-amd64">kubectl-v1.27.0-linux-amd64</a>
			<a href="kubectl-v1.26.0-linux-amd64">kubectl-v1.26.0-linux-amd64</a>
			</body>
			</html>`,
			check: func(rf *runtimeFiles) bool {
				return len(rf.containerd) == testTwoValue &&
					len(rf.cni) == testTwoValue &&
					len(rf.kubectl) == testTwoValue
			},
		},
		{
			name:     "empty HTML",
			htmlData: "",
			check: func(rf *runtimeFiles) bool {
				return len(rf.containerd) == testZeroValue &&
					len(rf.cni) == testZeroValue &&
					len(rf.kubectl) == testZeroValue
			},
		},
		{
			name: "no matching files",
			htmlData: `<html><body>
			<a href="readme.txt">readme.txt</a>
			<a href="config.yaml">config.yaml</a>
			</body></html>`,
			check: func(rf *runtimeFiles) bool {
				return len(rf.containerd) == testZeroValue &&
					len(rf.cni) == testZeroValue &&
					len(rf.kubectl) == testZeroValue
			},
		},
		{
			name: "mixed content with some matches",
			htmlData: `<html><body>
			<a href="containerd-1.7.0.tar.gz">containerd-1.7.0.tar.gz</a>
			<a href="random-file.txt">random-file.txt</a>
			<a href="kubectl-v1.27.0">kubectl-v1.27.0</a>
			</body></html>`,
			check: func(rf *runtimeFiles) bool {
				return len(rf.containerd) == testOneValue &&
					len(rf.cni) == testZeroValue &&
					len(rf.kubectl) == testOneValue
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := parseFileListFromHTML(tt.htmlData)
			assert.NoError(t, err)
			assert.True(t, tt.check(result), "check failed for test case: %s", tt.name)
		})
	}
}

func TestDownloadRuntimeFiles(t *testing.T) {
	tests := []struct {
		name         string
		files        *runtimeFiles
		mockDownload func(string, string) error
		expectError  bool
	}{
		{
			name: "download containerd fails",
			files: &runtimeFiles{
				containerd: []string{"containerd-1.7.0.tar.gz"},
				cni:        []string{"cni-plugins-v1.3.0.tgz"},
				kubectl:    []string{"kubectl-v1.27.0"},
			},
			mockDownload: func(s string, s2 string) error {
				if strings.Contains(s, "containerd") {
					return errors.New("download containerd error")
				}
				return nil
			},
			expectError: true,
		},
		{
			name: "download cni fails",
			files: &runtimeFiles{
				containerd: []string{"containerd-1.7.0.tar.gz"},
				cni:        []string{"cni-plugins-v1.3.0.tgz"},
				kubectl:    []string{"kubectl-v1.27.0"},
			},
			mockDownload: func(s string, s2 string) error {
				if strings.Contains(s, "cni") {
					return errors.New("download cni error")
				}
				return nil
			},
			expectError: true,
		},
		{
			name: "no kubectl files error",
			files: &runtimeFiles{
				containerd: []string{"containerd-1.7.0.tar.gz"},
				cni:        []string{"cni-plugins-v1.3.0.tgz"},
				kubectl:    []string{},
			},
			expectError: true,
		},
		{
			name: "download kubectl fails",
			files: &runtimeFiles{
				containerd: []string{"containerd-1.7.0.tar.gz"},
				cni:        []string{"cni-plugins-v1.3.0.tgz"},
				kubectl:    []string{"kubectl-v1.27.0"},
			},
			mockDownload: func(s string, s2 string) error {
				if strings.Contains(s, "kubectl") {
					return errors.New("download kubectl error")
				}
				return nil
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			if tt.mockDownload != nil {
				patches.ApplyFunc(utils.DownloadFile, tt.mockDownload)
			}
			patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

			err := downloadRuntimeFiles("http://example.com/files/", tt.files.containerd, tt.files.cni, tt.files.kubectl)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestSourceRuntimeKylin(t *testing.T) {
	tests := []struct {
		name         string
		httpRepo     string
		mockExists   func(string) bool
		mockDownload func(string, string) error
		expectError  bool
	}{
		{
			name:     "files already exist",
			httpRepo: "http://example.com",
			mockExists: func(s string) bool {
				return true
			},
			expectError: false,
		},
		{
			name:     "download kylin docker files",
			httpRepo: "http://example.com",
			mockExists: func(s string) bool {
				return false
			},
			mockDownload: func(s string, s2 string) error { return nil },
			expectError:  false,
		},
		{
			name:     "download fails for one file",
			httpRepo: "http://example.com",
			mockExists: func(s string) bool {
				return false
			},
			mockDownload: func(s string, s2 string) error {
				if strings.Contains(s, "arm64") {
					return nil
				}
				return errors.New("download error")
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			patches.ApplyFunc(utils.Exists, tt.mockExists)

			if tt.mockDownload != nil {
				patches.ApplyFunc(utils.DownloadFile, tt.mockDownload)
			}

			patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})
			patches.ApplyFunc(log.Errorf, func(format string, args ...interface{}) {})

			err := sourceRuntimeKylin(tt.httpRepo)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestSetupTLSCertificate(t *testing.T) {
	tests := []struct {
		name          string
		oc            OtherRepo
		certConfig    *CertificateConfig
		mockSetupCA   func(*CertificateConfig) error
		mockSetClient func(string) error
		expectError   bool
	}{
		{
			name: "TLSVerify is false",
			oc: OtherRepo{
				Image: "registry.example.com/image:v1",
			},
			certConfig: &CertificateConfig{
				TLSVerify: false,
			},
			expectError: false,
		},
		{
			name: "TLSVerify true with SetupCA error",
			oc: OtherRepo{
				Image: "registry.example.com/image:v1",
			},
			certConfig: &CertificateConfig{
				TLSVerify: true,
				CAFile:    "/tmp/ca.crt",
			},
			mockSetupCA: func(c *CertificateConfig) error {
				return errors.New("setup CA error")
			},
			expectError: true,
		},
		{
			name: "default image repo with SetClientCertificate error",
			oc: OtherRepo{
				Image: "deploy.bocloud.k8s:5000/image:v1",
			},
			certConfig: &CertificateConfig{
				TLSVerify: true,
				CAFile:    "/tmp/ca.crt",
			},
			mockSetClient: func(s string) error {
				return errors.New("set client error")
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			if tt.mockSetupCA != nil {
				patches.ApplyFunc(SetupCACertificate, tt.mockSetupCA)
			}

			if tt.mockSetClient != nil {
				patches.ApplyFunc(warehouseSetClientCertificate, tt.mockSetClient)
			}

			patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

			od := &registry.OptionsDownload{}
			err := setupTLSCertificate(tt.oc, tt.certConfig, od)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestSourceInit(t *testing.T) {
	tests := []struct {
		name             string
		oc               OtherRepo
		mockExists       func(string) bool
		mockMkdirAll     func(string, os.FileMode) error
		mockSourceBase   func(string) error
		mockSourceRun    func(string) error
		mockSourceRunKyl func(string) error
		expectError      bool
	}{
		{
			name: "empty source",
			oc: OtherRepo{
				Source: "",
			},
			expectError: false,
		},
		{
			name: "source init with all steps successful",
			oc: OtherRepo{
				Source: "http://example.com/source",
			},
			mockExists:       func(s string) bool { return true },
			mockMkdirAll:     func(s string, m os.FileMode) error { return nil },
			mockSourceBase:   func(s string) error { return nil },
			mockSourceRun:    func(s string) error { return nil },
			mockSourceRunKyl: func(s string) error { return nil },
			expectError:      false,
		},
		{
			name: "source init with mkdir error",
			oc: OtherRepo{
				Source: "http://example.com/source",
			},
			mockExists:   func(s string) bool { return false },
			mockMkdirAll: func(s string, m os.FileMode) error { return errors.New("mkdir error") },
			expectError:  true,
		},
		{
			name: "source init with sourceBaseFile error",
			oc: OtherRepo{
				Source: "http://example.com/source",
			},
			mockExists:     func(s string) bool { return true },
			mockSourceBase: func(s string) error { return errors.New("base error") },
			expectError:    true,
		},
		{
			name: "source init with sourceRuntime error",
			oc: OtherRepo{
				Source: "http://example.com/source",
			},
			mockExists:     func(s string) bool { return true },
			mockSourceBase: func(s string) error { return nil },
			mockSourceRun:  func(s string) error { return errors.New("runtime error") },
			expectError:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			patches.ApplyFunc(utils.Exists, tt.mockExists)

			if tt.mockMkdirAll != nil {
				patches.ApplyFunc(os.MkdirAll, tt.mockMkdirAll)
			}

			if tt.mockSourceBase != nil {
				patches.ApplyFunc(sourceBaseFile, tt.mockSourceBase)
			}

			if tt.mockSourceRun != nil {
				patches.ApplyFunc(sourceRuntime, tt.mockSourceRun)
			}

			if tt.mockSourceRunKyl != nil {
				patches.ApplyFunc(sourceRuntimeKylin, tt.mockSourceRunKyl)
			}

			patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

			err := SourceInit(tt.oc)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestRepoInit(t *testing.T) {
	tests := []struct {
		name          string
		oc            OtherRepo
		certConfig    *CertificateConfig
		mockExistsYum func(string) bool
		mockCleanTemp func() error
		mockSetupTLS  func(OtherRepo, *CertificateConfig, *registry.OptionsDownload) error
		expectError   bool
	}{
		{
			name: "empty image",
			oc: OtherRepo{
				Image: "",
			},
			certConfig:  &CertificateConfig{},
			expectError: false,
		},
		{
			name: "yum data file already exists",
			oc: OtherRepo{
				Image: "registry.example.com/image:v1",
			},
			certConfig:    &CertificateConfig{},
			mockExistsYum: func(s string) bool { return true },
			expectError:   false,
		},
		{
			name: "clean temp file error",
			oc: OtherRepo{
				Image: "registry.example.com/image:v1",
			},
			certConfig:    &CertificateConfig{},
			mockExistsYum: func(s string) bool { return false },
			mockCleanTemp: func() error { return errors.New("clean error") },
			expectError:   true,
		},
		{
			name: "setup TLS error",
			oc: OtherRepo{
				Image: "registry.example.com/image:v1",
			},
			certConfig: &CertificateConfig{
				TLSVerify: true,
			},
			mockExistsYum: func(s string) bool { return false },
			mockCleanTemp: func() error { return nil },
			mockSetupTLS: func(o OtherRepo, c *CertificateConfig, od *registry.OptionsDownload) error {
				return errors.New("TLS error")
			},
			expectError: true,
		},
		{
			name: "download error",
			oc: OtherRepo{
				Image: "registry.example.com/image:v1",
			},
			certConfig:    &CertificateConfig{},
			mockExistsYum: func(s string) bool { return false },
			mockCleanTemp: func() error { return nil },
			mockSetupTLS: func(o OtherRepo, c *CertificateConfig, od *registry.OptionsDownload) error {
				return nil
			},
			expectError: true,
		},
		{
			name: "successful repo init",
			oc: OtherRepo{
				Image: "registry.example.com/image:v1",
			},
			certConfig:    &CertificateConfig{},
			mockExistsYum: func(s string) bool { return false },
			mockCleanTemp: func() error { return nil },
			mockSetupTLS:  func(o OtherRepo, c *CertificateConfig, od *registry.OptionsDownload) error { return nil },
			expectError:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			patches.ApplyFunc(utils.Exists, tt.mockExistsYum)

			if tt.mockCleanTemp != nil {
				patches.ApplyFunc(cleanTempYumDataFile, tt.mockCleanTemp)
			}

			if tt.mockSetupTLS != nil {
				patches.ApplyFunc(setupTLSCertificate, tt.mockSetupTLS)
			}

			if tt.expectError {
				patches.ApplyFunc((*registry.OptionsDownload).Download, func(o *registry.OptionsDownload) error {
					return errors.New("download error")
				})
			} else {
				patches.ApplyFunc((*registry.OptionsDownload).Download, func(o *registry.OptionsDownload) error {
					return nil
				})
			}

			patches.ApplyFunc(finalizeYumDataFile, func() error { return nil })

			patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

			err := RepoInit(tt.oc, tt.certConfig)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestSourceRuntime(t *testing.T) {
	tests := []struct {
		name                string
		httpRepo            string
		mockCheckLocalExist func() (bool, error)
		mockFetchRemote     func(string) (*runtimeFiles, error)
		mockDownloadRun     func(string, []string, []string, []string) error
		expectError         bool
	}{
		{
			name:     "local files already exist",
			httpRepo: "http://example.com",
			mockCheckLocalExist: func() (bool, error) {
				return true, nil
			},
			expectError: false,
		},
		{
			name:     "check local exist error",
			httpRepo: "http://example.com",
			mockCheckLocalExist: func() (bool, error) {
				return false, errors.New("check error")
			},
			expectError: true,
		},
		{
			name:     "fetch remote file list error",
			httpRepo: "http://example.com",
			mockCheckLocalExist: func() (bool, error) {
				return false, nil
			},
			mockFetchRemote: func(s string) (*runtimeFiles, error) {
				return nil, errors.New("fetch error")
			},
			expectError: true,
		},
		{
			name:     "download runtime files error",
			httpRepo: "http://example.com",
			mockCheckLocalExist: func() (bool, error) {
				return false, nil
			},
			mockFetchRemote: func(s string) (*runtimeFiles, error) {
				return &runtimeFiles{}, nil
			},
			mockDownloadRun: func(s string, c []string, cn []string, k []string) error {
				return errors.New("download error")
			},
			expectError: true,
		},
		{
			name:     "successful runtime download",
			httpRepo: "http://example.com",
			mockCheckLocalExist: func() (bool, error) {
				return false, nil
			},
			mockFetchRemote: func(s string) (*runtimeFiles, error) {
				return &runtimeFiles{}, nil
			},
			mockDownloadRun: func(s string, c []string, cn []string, k []string) error {
				return nil
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			if tt.mockCheckLocalExist != nil {
				patches.ApplyFunc(checkLocalRuntimeFilesExist, tt.mockCheckLocalExist)
			}

			if tt.mockFetchRemote != nil {
				patches.ApplyFunc(fetchRemoteFileList, tt.mockFetchRemote)
			}

			if tt.mockDownloadRun != nil {
				patches.ApplyFunc(downloadRuntimeFiles, tt.mockDownloadRun)
			}

			err := sourceRuntime(tt.httpRepo)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

type mockDirEntry struct {
	name string
}

func (m *mockDirEntry) Name() string      { return m.name }
func (m *mockDirEntry) IsDir() bool       { return false }
func (m *mockDirEntry) Type() os.FileMode { return 0 }
func (m *mockDirEntry) Info() (os.FileInfo, error) {
	return nil, nil
}

func validateCustomExtra(m map[string]string) error {
	return nil
}

func warehouseSetClientCertificate(s string) error {
	return nil
}

func TestPrepareTempDirectory(t *testing.T) {
	tests := []struct {
		name        string
		targetTemp  string
		mockExists  func(string) bool
		mockRemove  func(string) error
		expectError bool
	}{
		{
			name:       "temp directory does not exist",
			targetTemp: "/tmp/nonexistent",
			mockExists: func(s string) bool {
				return false
			},
			expectError: false,
		},
		{
			name:       "temp directory exists and removed successfully",
			targetTemp: "/tmp/existing",
			mockExists: func(s string) bool {
				return true
			},
			mockRemove: func(s string) error {
				return nil
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(utils.Exists, tt.mockExists)

			if tt.mockRemove != nil {
				patches = patches.ApplyFunc(os.RemoveAll, tt.mockRemove)
			} else {
				patches = patches.ApplyFunc(os.RemoveAll, func(_ string) error { return nil })
			}

			err := prepareTempDirectory(tt.targetTemp)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestVerifyDirectoryExists(t *testing.T) {
	tests := []struct {
		name        string
		dir         string
		mockExists  func(string) bool
		expectError bool
	}{
		{
			name:        "directory exists",
			dir:         "/tmp/existing",
			mockExists:  func(s string) bool { return true },
			expectError: false,
		},
		{
			name:        "directory does not exist",
			dir:         "/tmp/nonexistent",
			mockExists:  func(s string) bool { return false },
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(utils.Exists, tt.mockExists)

			err := verifyDirectoryExists(tt.dir)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestRemoveArchiveExtensions(t *testing.T) {
	tests := []struct {
		name     string
		filename string
		expected string
	}{
		{
			name:     "tar.gz extension",
			filename: "image_amd64.tar.gz",
			expected: "image_amd64",
		},
		{
			name:     "tgz extension",
			filename: "image_amd64.tgz",
			expected: "image_amd64",
		},
		{
			name:     "no extension",
			filename: "image_amd64",
			expected: "image_amd64",
		},
		{
			name:     "multiple extensions only remove first",
			filename: "image.tar.gz.bak",
			expected: "image.tar.gz.bak",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := removeArchiveExtensions(tt.filename)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestCleanYumDataDirectory(t *testing.T) {
	tests := []struct {
		name        string
		mockRemove  func(string) error
		expectError bool
	}{
		{
			name:        "remove succeeds",
			mockRemove:  func(s string) error { return nil },
			expectError: false,
		},
		{
			name:        "remove fails with non-notexist error",
			mockRemove:  func(s string) error { return errors.New("remove error") },
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			patches.ApplyFunc(os.Remove, tt.mockRemove)

			err := cleanYumDataDirectory()

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestDecompressYumDataFile(t *testing.T) {
	t.Skip("Skipping due to gomonkey issues on this platform")
}

func TestPrepareChartData(t *testing.T) {
	t.Skip("Skipping due to gomonkey issues on this platform")
}

func TestVerifyContainerdFile(t *testing.T) {
	tests := []struct {
		name              string
		localImage        string
		mockReadDir       func(string) ([]os.DirEntry, error)
		mockValidateExtra func(map[string]string) error
		expectError       bool
		checkResult       func(result ContainerdFileResult) bool
	}{
		{
			name:       "all files exist",
			localImage: "",
			mockReadDir: func(s string) ([]os.DirEntry, error) {
				return []os.DirEntry{
					&mockDirEntry{name: "containerd-1.7.0-linux-amd64.tar.gz"},
					&mockDirEntry{name: "cni-plugins-linux-amd64-v1.3.0.tgz"},
				}, nil
			},
			mockValidateExtra: func(m map[string]string) error { return nil },
			expectError:       false,
			checkResult: func(result ContainerdFileResult) bool {
				return len(result.ContainerdList) > 0 && len(result.CniPluginList) > 0
			},
		},
		{
			name:       "no containerd files",
			localImage: "",
			mockReadDir: func(s string) ([]os.DirEntry, error) {
				return []os.DirEntry{
					&mockDirEntry{name: "cni-plugins-linux-amd64-v1.3.0.tgz"},
				}, nil
			},
			mockValidateExtra: func(m map[string]string) error { return nil },
			expectError:       true,
			checkResult:       nil,
		},
		{
			name:       "no cni plugin files",
			localImage: "",
			mockReadDir: func(s string) ([]os.DirEntry, error) {
				return []os.DirEntry{
					&mockDirEntry{name: "containerd-1.7.0-linux-amd64.tar.gz"},
				}, nil
			},
			mockValidateExtra: func(m map[string]string) error { return nil },
			expectError:       true,
			checkResult:       nil,
		},
		{
			name:       "read dir error",
			localImage: "",
			mockReadDir: func(s string) ([]os.DirEntry, error) {
				return nil, errors.New("read dir error")
			},
			expectError: true,
			checkResult: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			patches.ApplyFunc(os.ReadDir, tt.mockReadDir)

			if tt.mockValidateExtra != nil {
				patches.ApplyFunc(validateCustomExtra, tt.mockValidateExtra)
			}

			result, err := VerifyContainerdFile(tt.localImage)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.True(t, tt.checkResult(result), "checkResult failed")
			}
		})
	}
}

func TestDecompressDataPackage(t *testing.T) {
	tests := []struct {
		name                 string
		mockExistsDir        func(string) bool
		mockDirectoryIsEmpty func(string) bool
		mockMkdirAll         func(string, os.FileMode) error
		cfg                  decompressConfig
		expectError          bool
	}{
		{
			name:                 "directory already exists and not empty skip",
			mockExistsDir:        func(s string) bool { return true },
			mockDirectoryIsEmpty: func(s string) bool { return false },
			cfg: decompressConfig{
				dataFile:      "/test/data.tar.gz",
				dataDirectory: "/test/data",
				name:          "test",
				logMessage:    "Decompressing...",
				skipMessage:   "Skip if exists",
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			patches.ApplyFunc(utils.Exists, func(s string) bool {
				if strings.HasSuffix(s, ".tar.gz") {
					return true
				}
				return tt.mockExistsDir(s)
			})
			patches.ApplyFunc(utils.DirectoryIsEmpty, tt.mockDirectoryIsEmpty)

			if tt.mockMkdirAll != nil {
				patches.ApplyFunc(os.MkdirAll, tt.mockMkdirAll)
			}

			patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})
			patches.ApplyFunc(verifyDirectoryExists, func(s string) error { return nil })

			err := decompressDataPackage(tt.cfg)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestPrepareRepositoryDependOn(t *testing.T) {
	tests := []struct {
		name             string
		imageFilePath    string
		mockDecompress   func(decompressConfig) error
		mockPrepareChart func() error
		expectError      bool
	}{
		{
			name:             "successful preparation without local image",
			imageFilePath:    "",
			mockDecompress:   func(cfg decompressConfig) error { return nil },
			mockPrepareChart: func() error { return nil },
			expectError:      false,
		},
		{
			name:             "successful preparation with local image",
			imageFilePath:    "/test/local.tar.gz",
			mockDecompress:   func(cfg decompressConfig) error { return nil },
			mockPrepareChart: func() error { return nil },
			expectError:      false,
		},
		{
			name:             "prepare chart data fails",
			imageFilePath:    "",
			mockDecompress:   func(cfg decompressConfig) error { return nil },
			mockPrepareChart: func() error { return errors.New("chart prep error") },
			expectError:      true,
		},
		{
			name:          "decompress image data fails",
			imageFilePath: "",
			mockDecompress: func(cfg decompressConfig) error {
				if cfg.name == "image" {
					return errors.New("image decompress error")
				}
				return nil
			},
			mockPrepareChart: func() error { return nil },
			expectError:      true,
		},
		{
			name:          "decompress nfs data fails",
			imageFilePath: "",
			mockDecompress: func(cfg decompressConfig) error {
				if cfg.name == "NFS" {
					return errors.New("nfs decompress error")
				}
				return nil
			},
			mockPrepareChart: func() error { return nil },
			expectError:      true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			decompressCallCount := 0
			patches.ApplyFunc(decompressDataPackage, func(cfg decompressConfig) error {
				decompressCallCount++
				return tt.mockDecompress(cfg)
			})
			patches.ApplyFunc(prepareChartData, tt.mockPrepareChart)

			err := PrepareRepositoryDependOn(tt.imageFilePath)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestPrepareChartDataFunctional(t *testing.T) {
	tests := []struct {
		name                 string
		mockExistsDir        func(string) bool
		mockDirectoryIsEmpty func(string) bool
		expectError          bool
	}{
		{
			name:                 "directory already exists and not empty skip",
			mockExistsDir:        func(s string) bool { return true },
			mockDirectoryIsEmpty: func(s string) bool { return false },
			expectError:          false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			patches.ApplyFunc(utils.Exists, tt.mockExistsDir)
			patches.ApplyFunc(utils.DirectoryIsEmpty, tt.mockDirectoryIsEmpty)
			patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

			err := prepareChartData()

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestNfsPostProcessFunctional(t *testing.T) {
	tests := []struct {
		name          string
		targetTemp    string
		mockExists    func(string) bool
		mockRename    func(string, string) error
		mockRemoveAll func(string) error
		expectError   bool
	}{
		{
			name:          "nfsshare subdir rename fails",
			targetTemp:    "/tmp/nfs.tmp",
			mockExists:    func(s string) bool { return strings.Contains(s, "nfsshare") },
			mockRename:    func(s string, s2 string) error { return errors.New("rename error") },
			mockRemoveAll: func(s string) error { return nil },
			expectError:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			patches.ApplyFunc(utils.Exists, tt.mockExists)
			patches.ApplyFunc(os.Rename, tt.mockRename)
			if tt.mockRemoveAll != nil {
				patches.ApplyFunc(os.RemoveAll, tt.mockRemoveAll)
			}
			patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

			err := nfsPostProcess(tt.targetTemp)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestCreateImageLocalPostProcessFunctional(t *testing.T) {
	tests := []struct {
		name          string
		imageFilePath string
		mockExists    func(string) bool
		mockRename    func(string, string) error
		mockRemoveAll func(string) error
		expectError   bool
	}{
		{
			name:          "expected subdir exists rename successfully",
			imageFilePath: "/test/image_amd64.tar.gz",
			mockExists: func(s string) bool {
				return strings.Contains(s, "image_amd64")
			},
			mockRename:    func(s string, s2 string) error { return nil },
			mockRemoveAll: func(s string) error { return nil },
			expectError:   false,
		},
		{
			name:          "expected subdir does not exist rename temp",
			imageFilePath: "/test/image_amd64.tar.gz",
			mockExists: func(s string) bool {
				return strings.Contains(s, "image_amd64.tar.gz.tmp")
			},
			mockRename:    func(s string, s2 string) error { return nil },
			mockRemoveAll: func(s string) error { return nil },
			expectError:   false,
		},
		{
			name:          "subdir rename fails",
			imageFilePath: "/test/image_amd64.tar.gz",
			mockExists: func(s string) bool {
				return strings.Contains(s, "image_amd64")
			},
			mockRename:    func(s string, s2 string) error { return errors.New("rename error") },
			mockRemoveAll: func(s string) error { return nil },
			expectError:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			patches.ApplyFunc(utils.Exists, tt.mockExists)
			patches.ApplyFunc(os.Rename, tt.mockRename)
			if tt.mockRemoveAll != nil {
				patches.ApplyFunc(os.RemoveAll, tt.mockRemoveAll)
			}
			patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

			postProcess := createImageLocalPostProcess(tt.imageFilePath)
			err := postProcess(tt.imageFilePath + ".tmp")

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestLoadLocalRepository(t *testing.T) {
	tests := []struct {
		name              string
		mockLoadLocalRepo func(string) error
		expectError       bool
	}{
		{
			name:              "load local repository success",
			mockLoadLocalRepo: func(s string) error { return nil },
			expectError:       false,
		},
		{
			name:              "load local repository fails",
			mockLoadLocalRepo: func(s string) error { return errors.New("load error") },
			expectError:       true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(common.LoadLocalRepositoryFromFile, tt.mockLoadLocalRepo)

			err := LoadLocalRepository()

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestLoadLocalImage(t *testing.T) {
	tests := []struct {
		name              string
		mockLoadLocalRepo func(string) error
		expectError       bool
	}{
		{
			name:              "load local image success",
			mockLoadLocalRepo: func(s string) error { return nil },
			expectError:       false,
		},
		{
			name:              "load local image fails",
			mockLoadLocalRepo: func(s string) error { return errors.New("load error") },
			expectError:       true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(common.LoadLocalRepositoryFromFile, tt.mockLoadLocalRepo)

			err := LoadLocalImage()

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestContainerServer(t *testing.T) {
	tests := []struct {
		name               string
		localImage         string
		imageRegistryPort  string
		otherRepo          string
		onlineImage        string
		mockStartImageReg  func(int) error
		mockRemoveImageReg func(string) error
		mockSleep          func(time.Duration)
		expectError        bool
	}{
		{
			name:               "start image registry success on first try",
			localImage:         "",
			imageRegistryPort:  "5000",
			otherRepo:          "",
			onlineImage:        "",
			mockStartImageReg:  func(callCount int) error { return nil },
			mockRemoveImageReg: func(s string) error { return nil },
			mockSleep:          func(d time.Duration) {},
			expectError:        false,
		},
		{
			name:               "start image registry fails after retries",
			localImage:         "",
			imageRegistryPort:  "5000",
			otherRepo:          "",
			onlineImage:        "",
			mockStartImageReg:  func(callCount int) error { return errors.New("start error") },
			mockRemoveImageReg: func(s string) error { return nil },
			mockSleep:          func(d time.Duration) {},
			expectError:        true,
		},
		{
			name:              "start image registry success on retry",
			localImage:        "",
			imageRegistryPort: "5000",
			otherRepo:         "",
			onlineImage:       "",
			mockStartImageReg: func(callCount int) error {
				if callCount > 1 {
					return nil
				}
				return errors.New("start error")
			},
			mockRemoveImageReg: func(s string) error { return nil },
			mockSleep:          func(d time.Duration) {},
			expectError:        false,
		},
		{
			name:               "with local image uses default registry",
			localImage:         "/test/image.tar.gz",
			imageRegistryPort:  "5000",
			otherRepo:          "",
			onlineImage:        "",
			mockStartImageReg:  func(callCount int) error { return nil },
			mockRemoveImageReg: func(s string) error { return nil },
			mockSleep:          func(d time.Duration) {},
			expectError:        false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			startImageRegCallCount := 0
			patches.ApplyFunc(server.StartImageRegistry, func(s1 string, s2 string, s3 string, s4 string) error {
				startImageRegCallCount++
				return tt.mockStartImageReg(startImageRegCallCount)
			})
			patches.ApplyFunc(server.RemoveImageRegistry, tt.mockRemoveImageReg)
			patches.ApplyFunc(time.Sleep, tt.mockSleep)
			patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

			err := ContainerServer(tt.localImage, tt.imageRegistryPort, tt.otherRepo, tt.onlineImage)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestSyncLocalImage(t *testing.T) {
	tests := []struct {
		name              string
		imageRegistryPort string
		mockSpecificSync  func(string, string)
		mockEnsureImage   func(string) error
		expectError       bool
	}{
		{
			name:              "sync local image success",
			imageRegistryPort: "5000",
			mockSpecificSync:  func(s1 string, s2 string) {},
			mockEnsureImage:   func(s string) error { return nil },
			expectError:       false,
		},
		{
			name:              "sync local image ensure image fails",
			imageRegistryPort: "5000",
			mockSpecificSync:  func(s1 string, s2 string) {},
			mockEnsureImage:   func(s string) error { return errors.New("ensure error") },
			expectError:       true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			patches.ApplyFunc(build.SpecificSync, tt.mockSpecificSync)
			patches.ApplyFunc(econd.EnsureImageExists, tt.mockEnsureImage)
			patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

			err := SyncLocalImage(tt.imageRegistryPort)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestYumServer(t *testing.T) {
	tests := []struct {
		name              string
		localImage        string
		imageRegistryPort string
		yumRegistryPort   string
		otherRepo         string
		onlineImage       string
		mockStartYumReg   func(int) error
		mockRemoveYumReg  func(string) error
		mockSleep         func(time.Duration)
		expectError       bool
	}{
		{
			name:              "start yum registry success",
			localImage:        "",
			imageRegistryPort: "5000",
			yumRegistryPort:   "6000",
			otherRepo:         "",
			mockStartYumReg:   func(callCount int) error { return nil },
			mockRemoveYumReg:  func(s string) error { return nil },
			mockSleep:         func(d time.Duration) {},
			expectError:       false,
		},
		{
			name:              "start yum registry fails",
			localImage:        "",
			imageRegistryPort: "5000",
			yumRegistryPort:   "6000",
			otherRepo:         "",
			mockStartYumReg:   func(callCount int) error { return errors.New("start error") },
			mockRemoveYumReg:  func(s string) error { return nil },
			mockSleep:         func(d time.Duration) {},
			expectError:       true,
		},
		{
			name:              "start yum registry success on retry",
			localImage:        "",
			imageRegistryPort: "5000",
			yumRegistryPort:   "6000",
			otherRepo:         "",
			mockStartYumReg: func(callCount int) error {
				if callCount > 1 {
					return nil
				}
				return errors.New("start error")
			},
			mockRemoveYumReg: func(s string) error { return nil },
			mockSleep:        func(d time.Duration) {},
			expectError:      false,
		},
		{
			name:              "with other repo and no local image uses other repo",
			localImage:        "",
			imageRegistryPort: "5000",
			yumRegistryPort:   "6000",
			otherRepo:         "myrepo.com/",
			mockStartYumReg:   func(callCount int) error { return nil },
			mockRemoveYumReg:  func(s string) error { return nil },
			mockSleep:         func(d time.Duration) {},
			expectError:       false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			startYumRegCallCount := 0
			patches.ApplyFunc(server.StartYumRegistry, func(s1 string, s2 string, s3 string, s4 string) error {
				startYumRegCallCount++
				return tt.mockStartYumReg(startYumRegCallCount)
			})
			patches.ApplyFunc(server.RemoveYumRegistry, tt.mockRemoveYumReg)
			patches.ApplyFunc(time.Sleep, tt.mockSleep)
			patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

			err := YumServer(tt.localImage, tt.imageRegistryPort, tt.yumRegistryPort, tt.otherRepo, tt.onlineImage)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestChartServer(t *testing.T) {
	tests := []struct {
		name               string
		localImage         string
		imageRegistryPort  string
		chartRegistryPort  string
		otherRepo          string
		onlineImage        string
		mockStartChartReg  func(int) error
		mockRemoveChartReg func(string) error
		mockSleep          func(time.Duration)
		expectError        bool
	}{
		{
			name:               "start chart registry success",
			localImage:         "",
			imageRegistryPort:  "5000",
			chartRegistryPort:  "7000",
			otherRepo:          "",
			mockStartChartReg:  func(callCount int) error { return nil },
			mockRemoveChartReg: func(s string) error { return nil },
			mockSleep:          func(d time.Duration) {},
			expectError:        false,
		},
		{
			name:               "start chart registry fails",
			localImage:         "",
			imageRegistryPort:  "5000",
			chartRegistryPort:  "7000",
			otherRepo:          "",
			mockStartChartReg:  func(callCount int) error { return errors.New("start error") },
			mockRemoveChartReg: func(s string) error { return nil },
			mockSleep:          func(d time.Duration) {},
			expectError:        true,
		},
		{
			name:              "start chart registry success on retry",
			localImage:        "",
			imageRegistryPort: "5000",
			chartRegistryPort: "7000",
			otherRepo:         "",
			mockStartChartReg: func(callCount int) error {
				if callCount > 1 {
					return nil
				}
				return errors.New("start error")
			},
			mockRemoveChartReg: func(s string) error { return nil },
			mockSleep:          func(d time.Duration) {},
			expectError:        false,
		},
		{
			name:               "with other repo and no local image uses other repo",
			localImage:         "",
			imageRegistryPort:  "5000",
			chartRegistryPort:  "7000",
			otherRepo:          "myrepo.com/",
			mockStartChartReg:  func(callCount int) error { return nil },
			mockRemoveChartReg: func(s string) error { return nil },
			mockSleep:          func(d time.Duration) {},
			expectError:        false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			startChartRegCallCount := 0
			patches.ApplyFunc(server.StartChartRegistry, func(s1 string, s2 string, s3 string, s4 string) error {
				startChartRegCallCount++
				return tt.mockStartChartReg(startChartRegCallCount)
			})
			patches.ApplyFunc(server.RemoveChartRegistry, tt.mockRemoveChartReg)
			patches.ApplyFunc(time.Sleep, tt.mockSleep)
			patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

			err := ChartServer(tt.localImage, tt.imageRegistryPort, tt.chartRegistryPort, tt.otherRepo, tt.onlineImage)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestNFSServer(t *testing.T) {
	tests := []struct {
		name              string
		imageRegistryPort string
		otherRepo         string
		onlineImage       string
		mockStartNFSServ  func(int) error
		mockRemoveNFSServ func(string) error
		mockSleep         func(time.Duration)
		expectError       bool
	}{
		{
			name:              "start nfs server success",
			imageRegistryPort: "5000",
			otherRepo:         "",
			mockStartNFSServ:  func(callCount int) error { return nil },
			mockRemoveNFSServ: func(s string) error { return nil },
			mockSleep:         func(d time.Duration) {},
			expectError:       false,
		},
		{
			name:              "start nfs server fails",
			imageRegistryPort: "5000",
			otherRepo:         "",
			mockStartNFSServ:  func(callCount int) error { return errors.New("start error") },
			mockRemoveNFSServ: func(s string) error { return nil },
			mockSleep:         func(d time.Duration) {},
			expectError:       true,
		},
		{
			name:              "start nfs server success on retry",
			imageRegistryPort: "5000",
			otherRepo:         "",
			mockStartNFSServ: func(callCount int) error {
				if callCount > 1 {
					return nil
				}
				return errors.New("start error")
			},
			mockRemoveNFSServ: func(s string) error { return nil },
			mockSleep:         func(d time.Duration) {},
			expectError:       false,
		},
		{
			name:              "with other repo uses other repo",
			imageRegistryPort: "5000",
			otherRepo:         "myrepo.com/",
			mockStartNFSServ:  func(callCount int) error { return nil },
			mockRemoveNFSServ: func(s string) error { return nil },
			mockSleep:         func(d time.Duration) {},
			expectError:       false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			startNFSServCallCount := 0
			patches.ApplyFunc(server.StartNFSServer, func(s1 string, s2 string, s3 string) error {
				startNFSServCallCount++
				return tt.mockStartNFSServ(startNFSServCallCount)
			})
			patches.ApplyFunc(server.RemoveNFSServer, tt.mockRemoveNFSServ)
			patches.ApplyFunc(time.Sleep, tt.mockSleep)
			patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

			err := NFSServer(tt.imageRegistryPort, tt.otherRepo, tt.onlineImage)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestDecompressionSystemSourceFileFunctional(t *testing.T) {
	tests := []struct {
		name                 string
		mockExistsDir        func(string) bool
		mockDirectoryIsEmpty func(string) bool
		expectError          bool
	}{
		{
			name:                 "directory already exists and not empty skip",
			mockExistsDir:        func(s string) bool { return true },
			mockDirectoryIsEmpty: func(s string) bool { return false },
			expectError:          false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			patches.ApplyFunc(utils.Exists, tt.mockExistsDir)
			patches.ApplyFunc(utils.DirectoryIsEmpty, tt.mockDirectoryIsEmpty)
			patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

			err := DecompressionSystemSourceFile()

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestParseConsoleToolFilesFromHTML(t *testing.T) {
	htmlData := `<html><body>
	<a href="yq_linux_amd64">yq_linux_amd64</a>
	<a href="jq-linux-amd64">jq-linux-amd64</a>
	<a href="helm-v3.14.2-linux-amd64.tar.gz">helm-v3.14.2-linux-amd64.tar.gz</a>
	<a href="cfssl_1.6.4_linux_amd64">cfssl_1.6.4_linux_amd64</a>
	<a href="cfssl-certinfo_1.6.4_linux_amd64">cfssl-certinfo_1.6.4_linux_amd64</a>
	<a href="cfssljson_1.6.4_linux_amd64">cfssljson_1.6.4_linux_amd64</a>
	</body></html>`

	result, err := parseFileListFromHTML(htmlData)
	assert.NoError(t, err)
	assert.Len(t, result.consoleTools, 6)
}

func TestDownloadConsoleTools(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(string) bool { return false })
	patches.ApplyFunc(utils.DownloadFile, func(string, string) error { return nil })

	err := downloadConsoleTools("http://example.com/files/", []string{"yq_linux_amd64"}, "amd64")
	assert.NoError(t, err)

	err = downloadConsoleTools("http://example.com/files/", []string{}, "amd64")
	assert.Error(t, err)
}

func TestFilterConsoleToolsByArch(t *testing.T) {
	files := []string{
		"yq_linux_amd64",
		"yq_linux_arm64",
		"jq-linux-amd64",
		"helm-v3.14.2-linux-amd64.tar.gz",
		"cfssl_1.6.4_linux_amd64",
	}
	result := filterConsoleToolsByArch(files, "amd64")
	assert.Len(t, result, 4)
}

func TestEnsureConsoleTools(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(checkLocalConsoleToolsExist, func(string) (bool, error) { return true, nil })
	err := EnsureConsoleTools("http://example.com", "amd64")
	assert.NoError(t, err)

	patches.Reset()
	patches = gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(checkLocalConsoleToolsExist, func(string) (bool, error) { return false, nil })
	patches.ApplyFunc(utils.Exists, func(string) bool { return true })
	patches.ApplyFunc(fetchRemoteFileList, func(string) (*runtimeFiles, error) {
		return &runtimeFiles{consoleTools: []string{"yq_linux_amd64"}}, nil
	})
	patches.ApplyFunc(downloadConsoleTools, func(string, []string, string) error { return nil })

	err = EnsureConsoleTools("http://example.com", "amd64")
	assert.NoError(t, err)
}

func TestHttpFilesBase(t *testing.T) {
	tests := []struct {
		name     string
		source   string
		wantBase string
	}{
		{
			name:     "no prefix defaults to files",
			source:   "http://192.168.1.120:40080",
			wantBase: "http://192.168.1.120:40080/files",
		},
		{
			name:     "no prefix with trailing slash",
			source:   "http://192.168.1.120:40080/",
			wantBase: "http://192.168.1.120:40080/files",
		},
		{
			name:     "use existing prefix",
			source:   "http://repo.example.com:40080/openFuyao/binary",
			wantBase: "http://repo.example.com:40080/openFuyao/binary",
		},
		{
			name:     "existing prefix with trailing slash",
			source:   "http://repo.example.com:40080/openFuyao/binary/",
			wantBase: "http://repo.example.com:40080/openFuyao/binary",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.wantBase, httpFilesBase(tt.source))
		})
	}
}

