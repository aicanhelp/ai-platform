/******************************************************************
 * Copyright (c) 2026 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package dispatch

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"env-check/pkg/config"
	"env-check/pkg/logger"
)

func TestNewDispatcher(t *testing.T) {
	dispatchCfg := &config.DispatchConfig{Timeout: 600, WorkDir: "/tmp/envcheck"}
	appCfg := &config.AppConfig{Hosts: []config.Host{{IP: "192.168.1.1"}}}
	log := logger.NewLogger("")

	d := NewDispatcher(dispatchCfg, appCfg, log)
	assert.NotNil(t, d)
	assert.Equal(t, dispatchCfg, d.cfg)
	assert.Equal(t, appCfg, d.appCfg)
	assert.Equal(t, log, d.logger)
	assert.NotNil(t, d.sshClients)
}

func TestNewDispatchResult(t *testing.T) {
	result := newResult()
	assert.NotNil(t, result)
	assert.NotZero(t, result.Timestamp)
	assert.Empty(t, result.Nodes)
	assert.Empty(t, result.Duration)
}

func TestGetChecksToRun(t *testing.T) {
	tests := []struct {
		name       string
		checks     []string
		skipChecks []string
		want       []string
	}{
		{"defaultEmpty", []string{}, []string{}, []string{"kernel", "port", "disk", "clock", "fileQuery", "programCheck", "route", "iptables"}},
		{"defaultWithSkip", []string{}, []string{"fileQuery", "programCheck"}, []string{"kernel", "port", "disk", "clock", "route", "iptables"}},
		{"withChecksNoSkip", []string{"kernel", "port"}, []string{}, []string{"kernel", "port"}},
		{"withChecksAndSkip", []string{"kernel", "port", "disk"}, []string{"port"}, []string{"kernel", "disk"}},
		{"skipAll", []string{"kernel", "port"}, []string{"kernel", "port"}, nil},
		{"skipNotInChecks", []string{"kernel"}, []string{"port"}, []string{"kernel"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := NewDispatcher(&config.DispatchConfig{
				Checks:     tt.checks,
				SkipChecks: tt.skipChecks,
			}, &config.AppConfig{}, logger.NewLogger(""))
			got := d.getChecksToRun()
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestGetBinaryPathForHostFallback(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{}, &config.AppConfig{}, logger.NewLogger(""))
	host := config.Host{Arch: ""}
	got := d.getBinaryPathForHost(host)
	assert.NotEmpty(t, got)
}

func TestSummarizeResults(t *testing.T) {
	tests := []struct {
		name        string
		nodes       []config.NodeResult
		wantTotal   int
		wantSuccess int
		wantFailed  int
		wantRunning int
		wantResult  string
		wantError   string
	}{
		{
			name:      "allSuccess",
			nodes:     []config.NodeResult{{Status: "success", Results: []config.CheckResult{{Status: "pass"}}}},
			wantTotal: 1, wantSuccess: 1, wantFailed: 0, wantRunning: 0, wantResult: "PASS",
		},
		{
			name:      "failedStatus",
			nodes:     []config.NodeResult{{Status: "failed", Error: "ssh failed"}},
			wantTotal: 1, wantSuccess: 0, wantFailed: 1, wantRunning: 0, wantResult: "FAIL",
		},
		{
			name:      "successWithFailResult",
			nodes:     []config.NodeResult{{Status: "success", Results: []config.CheckResult{{CheckType: "fileQuery", Status: "fail"}, {Status: "pass"}}}},
			wantTotal: 1, wantSuccess: 0, wantFailed: 1, wantRunning: 0, wantResult: "FAIL", wantError: "failed checks: fileQuery",
		},
		{
			name: "successWithMultipleFailResults",
			nodes: []config.NodeResult{{Status: "success", Results: []config.CheckResult{
				{CheckType: "fileQuery", Status: "fail"},
				{CheckType: "programCheck", Status: "fail"},
				{CheckType: "kernel", Status: "pass"},
			}}},
			wantTotal: 1, wantSuccess: 0, wantFailed: 1, wantRunning: 0, wantResult: "FAIL", wantError: "failed checks: fileQuery, programCheck",
		},
		{
			name:      "runningStatus",
			nodes:     []config.NodeResult{{Status: "running"}},
			wantTotal: 1, wantSuccess: 0, wantFailed: 0, wantRunning: 1, wantResult: "PASS",
		},
		{
			name: "mixed",
			nodes: []config.NodeResult{
				{Status: "success", Results: []config.CheckResult{{Status: "pass"}}},
				{Status: "failed", Error: "error"},
				{Status: "running"},
			},
			wantTotal: 3, wantSuccess: 1, wantFailed: 1, wantRunning: 1, wantResult: "FAIL",
		},
		{
			name:      "unknownStatus",
			nodes:     []config.NodeResult{{Status: "unknown"}},
			wantTotal: 1, wantSuccess: 0, wantFailed: 0, wantRunning: 0, wantResult: "PASS",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := NewDispatcher(&config.DispatchConfig{}, &config.AppConfig{}, logger.NewLogger(""))
			result := &config.DispatchResult{Nodes: tt.nodes}
			d.summarizeResults(result)
			assert.Equal(t, tt.wantTotal, result.Summary.TotalNodes)
			assert.Equal(t, tt.wantSuccess, result.Summary.SuccessNodes)
			assert.Equal(t, tt.wantFailed, result.Summary.FailedNodes)
			assert.Equal(t, tt.wantRunning, result.Summary.RunningNodes)
			assert.Equal(t, tt.wantResult, result.Summary.Result)
			if tt.wantError != "" {
				assert.Equal(t, tt.wantError, result.Nodes[0].Error)
			}
		})
	}
}

func TestSaveResults(t *testing.T) {
	dir := t.TempDir()
	resultPath := filepath.Join(dir, "result.json")
	results := []config.CheckResult{
		{CheckType: "kernel", Status: "pass", Detail: "ok"},
	}

	err := SaveResults(results, resultPath)
	assert.NoError(t, err)

	data, err := os.ReadFile(resultPath)
	assert.NoError(t, err)
	assert.Contains(t, string(data), "kernel")
}

func TestSaveResultsCreateDir(t *testing.T) {
	dir := t.TempDir()
	resultPath := filepath.Join(dir, "subdir", "result.json")
	results := []config.CheckResult{}

	err := SaveResults(results, resultPath)
	assert.NoError(t, err)

	_, err = os.Stat(resultPath)
	assert.NoError(t, err)
}

func TestGenerateNodeConfig(t *testing.T) {
	appCfg := &config.AppConfig{
		PortCheck: &config.PortCheckConfig{
			Ports: map[string][]string{
				"master":    {"6443"},
				"worker":    {"10250"},
				"bootstrap": {"80"},
			},
		},
	}

	d := NewDispatcher(&config.DispatchConfig{}, appCfg, logger.NewLogger(""))
	nodeConfig, err := d.generateNodeConfig([]string{"master"})
	assert.NoError(t, err)

	var parsed config.AppConfig
	err = json.Unmarshal(nodeConfig, &parsed)
	assert.NoError(t, err)
	assert.Len(t, parsed.PortCheck.Ports, 1)
	assert.Contains(t, parsed.PortCheck.Ports, "master")
}

func TestMarkTimeoutFailures(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{}, &config.AppConfig{}, logger.NewLogger(""))
	nodeStatuses := map[string]*config.NodeResult{
		"192.168.1.1": {IP: "192.168.1.1", Status: "running"},
		"192.168.1.2": {IP: "192.168.1.2", Status: "running"},
	}
	completed := map[string]bool{
		"192.168.1.1": true,
	}
	result := &config.DispatchResult{}

	d.markTimeoutFailures(nodeStatuses, completed, result)
	assert.Len(t, result.Nodes, 1)
	assert.Equal(t, "failed", result.Nodes[0].Status)
	assert.Equal(t, "timeout waiting for result", result.Nodes[0].Error)
}

func TestDispatcherExecuteNoHosts(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{}, &config.AppConfig{Hosts: []config.Host{}}, logger.NewLogger(""))
	result, err := d.Execute()
	assert.Error(t, err)
	assert.Nil(t, result)
}

func TestCollectFromNodeNoSSHClient(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{WorkDir: "/tmp"}, &config.AppConfig{}, logger.NewLogger(""))
	nodeResult := &config.NodeResult{IP: "192.168.1.1", Status: "running"}
	result := &config.DispatchResult{}
	completed := make(map[string]bool)

	got := d.collectFromNode("192.168.1.1", nodeResult, result, completed)
	assert.True(t, got)
	assert.Equal(t, "failed", nodeResult.Status)
	assert.Equal(t, "SSH client not available", nodeResult.Error)
	assert.True(t, completed["192.168.1.1"])
	assert.Len(t, result.Nodes, 1)
}

func TestGenerateNodeConfigEmptyRoles(t *testing.T) {
	appCfg := &config.AppConfig{PortCheck: nil}

	d := NewDispatcher(&config.DispatchConfig{}, appCfg, logger.NewLogger(""))
	nodeConfig, err := d.generateNodeConfig([]string{})
	assert.NoError(t, err)
	assert.NotNil(t, nodeConfig)
}

func TestGetBinaryPathForHostWithArch(t *testing.T) {
	oldWd, err := os.Getwd()
	assert.NoError(t, err)
	defer func() {
		_ = os.Chdir(oldWd)
	}()

	tmpDir := t.TempDir()
	err = os.Chdir(tmpDir)
	assert.NoError(t, err)

	err = os.MkdirAll("build", 0755)
	assert.NoError(t, err)
	amd64Path := "./build/envCheck_amd64"
	err = os.WriteFile(amd64Path, []byte("test"), 0644)
	assert.NoError(t, err)

	d := NewDispatcher(&config.DispatchConfig{}, &config.AppConfig{}, logger.NewLogger(""))
	host := config.Host{Arch: "amd64"}
	got := d.getBinaryPathForHost(host)
	assert.Equal(t, amd64Path, got)
}

func TestSaveResultsInvalidPath(t *testing.T) {
	dir := t.TempDir()
	filePath := filepath.Join(dir, "file")
	err := os.WriteFile(filePath, []byte("x"), 0644)
	assert.NoError(t, err)

	results := []config.CheckResult{{CheckType: "kernel", Status: "pass"}}
	err = SaveResults(results, filepath.Join(filePath, "result.json"))
	assert.Error(t, err)
}

// Mock implementations for SSH/SFTP interfaces

type mockSSHConn struct {
	newSessionFunc func() (sshSession, error)
	closeFunc      func() error
}

func (m *mockSSHConn) NewSession() (sshSession, error) {
	if m.newSessionFunc != nil {
		return m.newSessionFunc()
	}
	return nil, fmt.Errorf("no newSessionFunc")
}
func (m *mockSSHConn) Close() error {
	if m.closeFunc != nil {
		return m.closeFunc()
	}
	return nil
}

type mockSSHSession struct {
	closeFunc  func() error
	outputFunc func(cmd string) ([]byte, error)
	runFunc    func(cmd string) error
	stdin      io.Reader
}

func (m *mockSSHSession) Close() error {
	if m.closeFunc != nil {
		return m.closeFunc()
	}
	return nil
}
func (m *mockSSHSession) Output(cmd string) ([]byte, error) {
	if m.outputFunc != nil {
		return m.outputFunc(cmd)
	}
	return nil, fmt.Errorf("no outputFunc")
}
func (m *mockSSHSession) Run(cmd string) error {
	if m.runFunc != nil {
		return m.runFunc(cmd)
	}
	return fmt.Errorf("no runFunc")
}
func (m *mockSSHSession) SetStdin(r io.Reader) { m.stdin = r }

type mockSFTPClient struct {
	closeFunc  func() error
	statFunc   func(path string) (os.FileInfo, error)
	removeFunc func(path string) error
	createFunc func(path string) (sftpFile, error)
	openFunc   func(path string) (sftpFile, error)
}

func (m *mockSFTPClient) Close() error {
	if m.closeFunc != nil {
		return m.closeFunc()
	}
	return nil
}
func (m *mockSFTPClient) Stat(path string) (os.FileInfo, error) {
	if m.statFunc != nil {
		return m.statFunc(path)
	}
	return nil, fmt.Errorf("no statFunc")
}
func (m *mockSFTPClient) Remove(path string) error {
	if m.removeFunc != nil {
		return m.removeFunc(path)
	}
	return nil
}
func (m *mockSFTPClient) Create(path string) (sftpFile, error) {
	if m.createFunc != nil {
		return m.createFunc(path)
	}
	return nil, fmt.Errorf("no createFunc")
}
func (m *mockSFTPClient) Open(path string) (sftpFile, error) {
	if m.openFunc != nil {
		return m.openFunc(path)
	}
	return nil, fmt.Errorf("no openFunc")
}

type mockSFTPFile struct {
	bytes.Buffer
	closeFunc func() error
}

func (m *mockSFTPFile) Close() error { return m.closeFunc() }

type mockFileInfo struct {
	size int64
}

func (m *mockFileInfo) Name() string       { return "mock" }
func (m *mockFileInfo) Size() int64        { return m.size }
func (m *mockFileInfo) Mode() os.FileMode  { return 0644 }
func (m *mockFileInfo) ModTime() time.Time { return time.Now() }
func (m *mockFileInfo) IsDir() bool        { return false }
func (m *mockFileInfo) Sys() interface{}   { return nil }

type mockFailingSFTPFile struct {
	closeFunc func() error
}

func (m *mockFailingSFTPFile) Write(p []byte) (int, error) { return 0, fmt.Errorf("write error") }
func (m *mockFailingSFTPFile) Read(p []byte) (int, error)  { return 0, io.EOF }
func (m *mockFailingSFTPFile) Close() error                { return m.closeFunc() }

func TestCloseAllConnectionsSuccess(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{}, &config.AppConfig{}, logger.NewLogger(""))
	sshClosed := false
	sftpClosed := false
	d.sshClients["192.168.1.1"] = &SSHClient{
		client:     &mockSSHConn{closeFunc: func() error { sshClosed = true; return nil }},
		sftpClient: &mockSFTPClient{closeFunc: func() error { sftpClosed = true; return nil }},
	}
	d.closeAllConnections()
	assert.True(t, sshClosed)
	assert.True(t, sftpClosed)
}

func TestExecuteCommandWithOutputSuccess(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{}, &config.AppConfig{}, logger.NewLogger(""))
	session := &mockSSHSession{
		closeFunc:  func() error { return nil },
		outputFunc: func(cmd string) ([]byte, error) { return []byte("output"), nil },
	}
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return session, nil },
		},
	}
	stdout, stderr, err := d.executeCommandWithOutput(client, "uname")
	assert.NoError(t, err)
	assert.Equal(t, "output", string(stdout))
	assert.Nil(t, stderr)
}

func TestExecuteCommandWithOutputSessionError(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{}, &config.AppConfig{}, logger.NewLogger(""))
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return nil, fmt.Errorf("session error") },
		},
	}
	_, _, err := d.executeCommandWithOutput(client, "uname")
	assert.Error(t, err)
}

func TestExecuteCommandWithOutputRunError(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{}, &config.AppConfig{}, logger.NewLogger(""))
	session := &mockSSHSession{
		closeFunc:  func() error { return nil },
		outputFunc: func(cmd string) ([]byte, error) { return nil, fmt.Errorf("run error") },
	}
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return session, nil },
		},
	}
	_, _, err := d.executeCommandWithOutput(client, "uname")
	assert.Error(t, err)
}

func TestExecuteCommandSuccess(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{}, &config.AppConfig{}, logger.NewLogger(""))
	session := &mockSSHSession{
		closeFunc: func() error { return nil },
		runFunc:   func(cmd string) error { return nil },
	}
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return session, nil },
		},
	}
	err := d.executeCommand(client, "ls")
	assert.NoError(t, err)
}

func TestExecuteCommandSessionError(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{}, &config.AppConfig{}, logger.NewLogger(""))
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return nil, fmt.Errorf("session error") },
		},
	}
	err := d.executeCommand(client, "ls")
	assert.Error(t, err)
}

func TestCheckFileExistsTrue(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{}, &config.AppConfig{}, logger.NewLogger(""))
	session := &mockSSHSession{
		closeFunc: func() error { return nil },
		runFunc:   func(cmd string) error { return nil },
	}
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return session, nil },
		},
	}
	assert.True(t, d.checkFileExists(client, "/tmp/result.json"))
}

func TestCheckFileExistsFalse(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{}, &config.AppConfig{}, logger.NewLogger(""))
	session := &mockSSHSession{
		closeFunc: func() error { return nil },
		runFunc:   func(cmd string) error { return fmt.Errorf("not found") },
	}
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return session, nil },
		},
	}
	assert.False(t, d.checkFileExists(client, "/tmp/result.json"))
}

func TestCheckFileExistsSessionError(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{}, &config.AppConfig{}, logger.NewLogger(""))
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return nil, fmt.Errorf("session error") },
		},
	}
	assert.False(t, d.checkFileExists(client, "/tmp/result.json"))
}

func TestReadRemoteFileSuccess(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{}, &config.AppConfig{}, logger.NewLogger(""))
	session := &mockSSHSession{
		closeFunc:  func() error { return nil },
		outputFunc: func(cmd string) ([]byte, error) { return []byte("error data"), nil },
	}
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return session, nil },
		},
	}
	data, err := d.readRemoteFile(client, "/tmp/error.log")
	assert.NoError(t, err)
	assert.Equal(t, "error data", string(data))
}

func TestReadRemoteFileSessionError(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{}, &config.AppConfig{}, logger.NewLogger(""))
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return nil, fmt.Errorf("session error") },
		},
	}
	_, err := d.readRemoteFile(client, "/tmp/error.log")
	assert.Error(t, err)
}

func TestDownloadFileSuccess(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{}, &config.AppConfig{}, logger.NewLogger(""))
	file := &mockSFTPFile{}
	file.WriteString("file content")
	file.closeFunc = func() error { return nil }
	client := &SSHClient{
		sftpClient: &mockSFTPClient{
			openFunc: func(path string) (sftpFile, error) { return file, nil },
		},
	}
	data, err := d.downloadFile(client, "/tmp/result.json")
	assert.NoError(t, err)
	assert.Equal(t, "file content", string(data))
}

func TestDownloadFileOpenError(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{}, &config.AppConfig{}, logger.NewLogger(""))
	client := &SSHClient{
		sftpClient: &mockSFTPClient{
			openFunc: func(path string) (sftpFile, error) { return nil, fmt.Errorf("open error") },
		},
	}
	_, err := d.downloadFile(client, "/tmp/result.json")
	assert.Error(t, err)
}

func TestUploadConfigBytesSuccess(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{WorkDir: "/tmp"}, &config.AppConfig{}, logger.NewLogger(""))
	file := &mockSFTPFile{closeFunc: func() error { return nil }}
	client := &SSHClient{
		sftpClient: &mockSFTPClient{
			removeFunc: func(path string) error { return nil },
			createFunc: func(path string) (sftpFile, error) { return file, nil },
		},
	}
	err := d.uploadConfigBytes(client, []byte("config"), "/tmp/config.json")
	assert.NoError(t, err)
	assert.Equal(t, "config", file.String())
}

func TestUploadConfigBytesCreateError(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{WorkDir: "/tmp"}, &config.AppConfig{}, logger.NewLogger(""))
	client := &SSHClient{
		sftpClient: &mockSFTPClient{
			removeFunc: func(path string) error { return nil },
			createFunc: func(path string) (sftpFile, error) { return nil, fmt.Errorf("create error") },
		},
	}
	err := d.uploadConfigBytes(client, []byte("config"), "/tmp/config.json")
	assert.Error(t, err)
}

func TestUploadConfigBytesWriteError(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{WorkDir: "/tmp"}, &config.AppConfig{}, logger.NewLogger(""))
	client := &SSHClient{
		sftpClient: &mockSFTPClient{
			removeFunc: func(path string) error { return nil },
			createFunc: func(path string) (sftpFile, error) {
				return &mockFailingSFTPFile{closeFunc: func() error { return nil }}, nil
			},
		},
	}
	err := d.uploadConfigBytes(client, []byte("config"), "/tmp/config.json")
	assert.Error(t, err)
}

func TestUploadViaSFTPSuccess(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{}, &config.AppConfig{}, logger.NewLogger(""))
	dir := t.TempDir()
	localPath := filepath.Join(dir, "testfile")
	err := os.WriteFile(localPath, []byte("data"), 0644)
	assert.NoError(t, err)

	file := &mockSFTPFile{closeFunc: func() error { return nil }}
	client := &SSHClient{
		sftpClient: &mockSFTPClient{
			removeFunc: func(path string) error { return nil },
			createFunc: func(path string) (sftpFile, error) { return file, nil },
			statFunc: func(path string) (os.FileInfo, error) {
				return &mockFileInfo{size: 4}, nil
			},
		},
	}
	err = d.uploadViaSFTP(client, localPath, "/remote/testfile")
	assert.NoError(t, err)
}

func TestUploadViaSFTPFileOpenError(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{}, &config.AppConfig{}, logger.NewLogger(""))
	client := &SSHClient{
		sftpClient: &mockSFTPClient{},
	}
	err := d.uploadViaSFTP(client, "/nonexistent/file", "/remote/testfile")
	assert.Error(t, err)
}

func TestUploadViaSFTPRemoteCreateError(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{}, &config.AppConfig{}, logger.NewLogger(""))
	dir := t.TempDir()
	localPath := filepath.Join(dir, "testfile")
	err := os.WriteFile(localPath, []byte("data"), 0644)
	assert.NoError(t, err)

	client := &SSHClient{
		sftpClient: &mockSFTPClient{
			removeFunc: func(path string) error { return nil },
			createFunc: func(path string) (sftpFile, error) { return nil, fmt.Errorf("create error") },
		},
	}
	err = d.uploadViaSFTP(client, localPath, "/remote/testfile")
	assert.Error(t, err)
}

func TestUploadViaSFTPSizeMismatch(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{}, &config.AppConfig{}, logger.NewLogger(""))
	dir := t.TempDir()
	localPath := filepath.Join(dir, "testfile")
	err := os.WriteFile(localPath, []byte("data"), 0644)
	assert.NoError(t, err)

	file := &mockSFTPFile{closeFunc: func() error { return nil }}
	client := &SSHClient{
		sftpClient: &mockSFTPClient{
			removeFunc: func(path string) error { return nil },
			createFunc: func(path string) (sftpFile, error) { return file, nil },
			statFunc: func(path string) (os.FileInfo, error) {
				return &mockFileInfo{size: 999}, nil
			},
		},
	}
	err = d.uploadViaSFTP(client, localPath, "/remote/testfile")
	assert.Error(t, err)
}

func TestUploadFileCompressedSuccess(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{}, &config.AppConfig{}, logger.NewLogger(""))
	dir := t.TempDir()
	localPath := filepath.Join(dir, "testfile")
	err := os.WriteFile(localPath, []byte("data"), 0644)
	assert.NoError(t, err)

	session := &mockSSHSession{
		closeFunc: func() error { return nil },
		runFunc:   func(cmd string) error { return nil },
	}
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return session, nil },
		},
	}
	err = d.uploadFileCompressed(client, localPath, "/remote/dir/file")
	assert.NoError(t, err)
	assert.NotNil(t, session.stdin)
}

func TestUploadFileCompressedOpenError(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{}, &config.AppConfig{}, logger.NewLogger(""))
	client := &SSHClient{}
	err := d.uploadFileCompressed(client, "/nonexistent/file", "/remote/dir/file")
	assert.Error(t, err)
}

func TestUploadFileCompressedSessionError(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{}, &config.AppConfig{}, logger.NewLogger(""))
	dir := t.TempDir()
	localPath := filepath.Join(dir, "testfile")
	err := os.WriteFile(localPath, []byte("data"), 0644)
	assert.NoError(t, err)

	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return nil, fmt.Errorf("session error") },
		},
	}
	err = d.uploadFileCompressed(client, localPath, "/remote/dir/file")
	assert.Error(t, err)
}

func TestUploadFileCompressedRunError(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{}, &config.AppConfig{}, logger.NewLogger(""))
	dir := t.TempDir()
	localPath := filepath.Join(dir, "testfile")
	err := os.WriteFile(localPath, []byte("data"), 0644)
	assert.NoError(t, err)

	session := &mockSSHSession{
		closeFunc: func() error { return nil },
		runFunc:   func(cmd string) error { return fmt.Errorf("run error") },
	}
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return session, nil },
		},
	}
	err = d.uploadFileCompressed(client, localPath, "/remote/dir/file")
	assert.Error(t, err)
}

func TestDispatchToNodeBinaryExists(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{WorkDir: "/tmp"}, &config.AppConfig{}, logger.NewLogger(""))
	dir := t.TempDir()
	binaryPath := filepath.Join(dir, "envCheck_amd64")
	err := os.WriteFile(binaryPath, []byte("binary"), 0644)
	assert.NoError(t, err)

	oldWd, _ := os.Getwd()
	os.Chdir(dir)
	defer os.Chdir(oldWd)

	session := &mockSSHSession{
		closeFunc: func() error { return nil },
		runFunc:   func(cmd string) error { return nil },
	}
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return session, nil },
		},
		sftpClient: &mockSFTPClient{
			statFunc: func(path string) (os.FileInfo, error) {
				return &mockFileInfo{size: 6}, nil
			},
			createFunc: func(path string) (sftpFile, error) {
				return &mockSFTPFile{closeFunc: func() error { return nil }}, nil
			},
			removeFunc: func(path string) error { return nil },
		},
		host: config.Host{Arch: "amd64", Role: []string{"master"}},
	}
	err = d.transferToNode("192.168.1.1", client)
	assert.NoError(t, err)
}

func TestDispatchToNodeUploadSuccess(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{WorkDir: "/tmp"}, &config.AppConfig{}, logger.NewLogger(""))
	dir := t.TempDir()
	binaryPath := filepath.Join(dir, "envCheck_amd64")
	err := os.WriteFile(binaryPath, []byte("binary"), 0644)
	assert.NoError(t, err)

	oldWd, _ := os.Getwd()
	os.Chdir(dir)
	defer os.Chdir(oldWd)

	compressSession := &mockSSHSession{
		closeFunc: func() error { return nil },
		runFunc:   func(cmd string) error { return nil },
	}
	chmodSession := &mockSSHSession{
		closeFunc: func() error { return nil },
		runFunc:   func(cmd string) error { return nil },
	}
	sessionIndex := 0
	sessions := []sshSession{compressSession, chmodSession}
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) {
				s := sessions[sessionIndex]
				sessionIndex++
				return s, nil
			},
		},
		sftpClient: &mockSFTPClient{
			statFunc: func(path string) (os.FileInfo, error) {
				return nil, fmt.Errorf("not found")
			},
			createFunc: func(path string) (sftpFile, error) {
				return &mockSFTPFile{closeFunc: func() error { return nil }}, nil
			},
			removeFunc: func(path string) error { return nil },
		},
		host: config.Host{Arch: "amd64", Role: []string{"master"}},
	}
	err = d.transferToNode("192.168.1.1", client)
	assert.NoError(t, err)
}

func TestDispatchToNodeUploadFails(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{WorkDir: "/tmp"}, &config.AppConfig{}, logger.NewLogger(""))
	dir := t.TempDir()
	binaryPath := filepath.Join(dir, "envCheck_amd64")
	err := os.WriteFile(binaryPath, []byte("binary"), 0644)
	assert.NoError(t, err)

	oldWd, _ := os.Getwd()
	os.Chdir(dir)
	defer os.Chdir(oldWd)

	session := &mockSSHSession{
		closeFunc: func() error { return nil },
		runFunc:   func(cmd string) error { return fmt.Errorf("upload failed") },
	}
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return session, nil },
		},
		sftpClient: &mockSFTPClient{
			statFunc: func(path string) (os.FileInfo, error) {
				return nil, fmt.Errorf("not found")
			},
		},
		host: config.Host{Arch: "amd64", Role: []string{"master"}},
	}
	err = d.transferToNode("192.168.1.1", client)
	assert.Error(t, err)
}

func TestDispatchToNodeChmodFails(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{WorkDir: "/tmp"}, &config.AppConfig{}, logger.NewLogger(""))
	dir := t.TempDir()
	binaryPath := filepath.Join(dir, "envCheck_amd64")
	err := os.WriteFile(binaryPath, []byte("binary"), 0644)
	assert.NoError(t, err)

	oldWd, _ := os.Getwd()
	os.Chdir(dir)
	defer os.Chdir(oldWd)

	compressSession := &mockSSHSession{
		closeFunc: func() error { return nil },
		runFunc:   func(cmd string) error { return nil },
	}
	chmodSession := &mockSSHSession{
		closeFunc: func() error { return nil },
		runFunc:   func(cmd string) error { return fmt.Errorf("chmod failed") },
	}
	sessionIndex := 0
	sessions := []sshSession{compressSession, chmodSession}
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) {
				s := sessions[sessionIndex]
				sessionIndex++
				return s, nil
			},
		},
		sftpClient: &mockSFTPClient{
			statFunc: func(path string) (os.FileInfo, error) {
				return nil, fmt.Errorf("not found")
			},
		},
		host: config.Host{Arch: "amd64", Role: []string{"master"}},
	}
	err = d.transferToNode("192.168.1.1", client)
	assert.Error(t, err)
}

func TestDetectNodeArchSuccess(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{ConcurrentLimit: 1}, &config.AppConfig{}, logger.NewLogger(""))
	session := &mockSSHSession{
		closeFunc:  func() error { return nil },
		outputFunc: func(cmd string) ([]byte, error) { return []byte("arm64\n"), nil },
	}
	d.sshClients["192.168.1.1"] = &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return session, nil },
		},
		host: config.Host{IP: "192.168.1.1"},
	}
	err := d.detectNodeArch()
	assert.NoError(t, err)
	assert.Equal(t, "arm64", d.sshClients["192.168.1.1"].host.Arch)
}

func TestDetectNodeArchError(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{ConcurrentLimit: 1}, &config.AppConfig{}, logger.NewLogger(""))
	session := &mockSSHSession{
		closeFunc:  func() error { return nil },
		outputFunc: func(cmd string) ([]byte, error) { return nil, fmt.Errorf("cmd failed") },
	}
	d.sshClients["192.168.1.1"] = &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return session, nil },
		},
		host: config.Host{IP: "192.168.1.1"},
	}
	err := d.detectNodeArch()
	assert.Error(t, err)
}

func TestDetectNodeArchEmptyOutput(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{ConcurrentLimit: 1}, &config.AppConfig{}, logger.NewLogger(""))
	session := &mockSSHSession{
		closeFunc:  func() error { return nil },
		outputFunc: func(cmd string) ([]byte, error) { return []byte("\n"), nil },
	}
	d.sshClients["192.168.1.1"] = &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return session, nil },
		},
		host: config.Host{IP: "192.168.1.1"},
	}
	err := d.detectNodeArch()
	assert.NoError(t, err)
	assert.Equal(t, "amd64", d.sshClients["192.168.1.1"].host.Arch)
}

func TestCleanRemoteEnvSuccess(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{WorkDir: "/tmp", ConcurrentLimit: 1}, &config.AppConfig{}, logger.NewLogger(""))
	runCount := 0
	session := &mockSSHSession{
		closeFunc: func() error { return nil },
		runFunc: func(cmd string) error {
			runCount++
			return nil
		},
	}
	d.sshClients["192.168.1.1"] = &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return session, nil },
		},
	}
	err := d.cleanRemoteEnv()
	assert.NoError(t, err)
	assert.Equal(t, 3, runCount)
}

func TestExecuteOnNodeSuccess(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{WorkDir: "/tmp"}, &config.AppConfig{}, logger.NewLogger(""))
	session := &mockSSHSession{
		closeFunc:  func() error { return nil },
		outputFunc: func(cmd string) ([]byte, error) { return []byte("1234\n"), nil },
	}
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return session, nil },
		},
	}
	nodeResult := &config.NodeResult{Status: "running"}
	d.executeOnNode(nodeExecRequest{ip: "192.168.1.1", client: client, nodeResult: nodeResult, checkArg: "kernel,port", mu: &sync.Mutex{}})
	assert.Equal(t, "running", nodeResult.Status)
}

func TestExecuteOnNodeCloseErrorDoesNotFailStartedCheck(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{WorkDir: "/tmp"}, &config.AppConfig{}, logger.NewLogger(""))
	session := &mockSSHSession{
		closeFunc:  func() error { return io.EOF },
		outputFunc: func(cmd string) ([]byte, error) { return []byte("1234\n"), nil },
	}
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return session, nil },
		},
	}
	nodeResult := &config.NodeResult{Status: "running"}

	d.executeOnNode(nodeExecRequest{
		ip:         "192.168.1.1",
		client:     client,
		nodeResult: nodeResult,
		checkArg:   "kernel",
		mu:         &sync.Mutex{},
	})

	assert.Equal(t, "running", nodeResult.Status)
	assert.Empty(t, nodeResult.Error)
}

func TestExecuteOnNodeSessionError(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{WorkDir: "/tmp"}, &config.AppConfig{}, logger.NewLogger(""))
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return nil, fmt.Errorf("session error") },
		},
	}
	nodeResult := &config.NodeResult{Status: "running"}
	d.executeOnNode(nodeExecRequest{ip: "192.168.1.1", client: client, nodeResult: nodeResult, checkArg: "kernel", mu: &sync.Mutex{}})
	assert.Equal(t, "failed", nodeResult.Status)
	assert.Contains(t, nodeResult.Error, "failed to create session")
}

func TestExecuteOnNodeCommandError(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{WorkDir: "/tmp"}, &config.AppConfig{}, logger.NewLogger(""))
	session := &mockSSHSession{
		closeFunc:  func() error { return nil },
		outputFunc: func(cmd string) ([]byte, error) { return nil, fmt.Errorf("cmd error") },
	}
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return session, nil },
		},
	}
	nodeResult := &config.NodeResult{Status: "running"}
	d.executeOnNode(nodeExecRequest{ip: "192.168.1.1", client: client, nodeResult: nodeResult, checkArg: "kernel", mu: &sync.Mutex{}})
	assert.Equal(t, "failed", nodeResult.Status)
	assert.Contains(t, nodeResult.Error, "failed to start check")
}

func TestCollectFromNodeResultFileSuccess(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{WorkDir: "/tmp"}, &config.AppConfig{}, logger.NewLogger(""))
	checkSession := &mockSSHSession{
		closeFunc: func() error { return nil },
		runFunc:   func(cmd string) error { return nil },
	}
	downloadFile := &mockSFTPFile{}
	downloadFile.WriteString(`[{"checkType":"kernel","status":"pass"}]`)
	downloadFile.closeFunc = func() error { return nil }
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return checkSession, nil },
		},
		sftpClient: &mockSFTPClient{
			openFunc: func(path string) (sftpFile, error) { return downloadFile, nil },
		},
	}
	d.sshClients["192.168.1.1"] = client

	nodeResult := &config.NodeResult{IP: "192.168.1.1", Status: "running"}
	result := &config.DispatchResult{}
	completed := make(map[string]bool)

	handled := d.collectFromNode("192.168.1.1", nodeResult, result, completed)
	assert.True(t, handled)
	assert.True(t, completed["192.168.1.1"])
	assert.Equal(t, "success", nodeResult.Status)
	assert.Len(t, nodeResult.Results, 1)
	assert.Len(t, result.Nodes, 1)
}

func TestCollectFromNodeResultFileSuccessClearsStartError(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{WorkDir: "/tmp"}, &config.AppConfig{}, logger.NewLogger(""))
	checkSession := &mockSSHSession{
		closeFunc: func() error { return nil },
		runFunc:   func(cmd string) error { return nil },
	}
	downloadFile := &mockSFTPFile{}
	downloadFile.WriteString(`[{"checkType":"kernel","status":"pass"}]`)
	downloadFile.closeFunc = func() error { return nil }
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return checkSession, nil },
		},
		sftpClient: &mockSFTPClient{
			openFunc: func(path string) (sftpFile, error) { return downloadFile, nil },
		},
	}
	d.sshClients["192.168.1.1"] = client

	nodeResult := &config.NodeResult{
		IP:        "192.168.1.1",
		Status:    "failed",
		Error:     "failed to close session: EOF",
		ErrorFile: "failed to close session: EOF",
	}
	result := &config.DispatchResult{}
	completed := make(map[string]bool)

	handled := d.collectFromNode("192.168.1.1", nodeResult, result, completed)
	assert.True(t, handled)
	assert.Equal(t, "success", nodeResult.Status)
	assert.Empty(t, nodeResult.Error)
	assert.Empty(t, nodeResult.ErrorFile)
}

func TestCollectFromNodeResultFileDownloadFail(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{WorkDir: "/tmp"}, &config.AppConfig{}, logger.NewLogger(""))
	checkSession := &mockSSHSession{
		closeFunc: func() error { return nil },
		runFunc:   func(cmd string) error { return nil },
	}
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return checkSession, nil },
		},
		sftpClient: &mockSFTPClient{
			openFunc: func(path string) (sftpFile, error) { return nil, fmt.Errorf("open error") },
		},
	}
	d.sshClients["192.168.1.1"] = client

	nodeResult := &config.NodeResult{IP: "192.168.1.1", Status: "running"}
	result := &config.DispatchResult{}
	completed := make(map[string]bool)

	handled := d.collectFromNode("192.168.1.1", nodeResult, result, completed)
	assert.True(t, handled)
	assert.Equal(t, "failed", nodeResult.Status)
}

func TestCollectFromNodeErrorFile(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{WorkDir: "/tmp"}, &config.AppConfig{}, logger.NewLogger(""))
	checkSession := &mockSSHSession{
		closeFunc: func() error { return nil },
		runFunc:   func(cmd string) error { return fmt.Errorf("not found") },
	}
	readSession := &mockSSHSession{
		closeFunc:  func() error { return nil },
		outputFunc: func(cmd string) ([]byte, error) { return []byte("error occurred"), nil },
	}
	sessionIndex := 0
	sessions := []sshSession{checkSession, readSession}
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) {
				s := sessions[sessionIndex]
				sessionIndex++
				return s, nil
			},
		},
	}
	d.sshClients["192.168.1.1"] = client

	nodeResult := &config.NodeResult{IP: "192.168.1.1", Status: "running"}
	result := &config.DispatchResult{}
	completed := make(map[string]bool)

	handled := d.collectFromNode("192.168.1.1", nodeResult, result, completed)
	assert.True(t, handled)
	assert.Equal(t, "failed", nodeResult.Status)
	assert.Equal(t, "error occurred", nodeResult.Error)
}

func TestCollectFromNodeNoFiles(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{WorkDir: "/tmp"}, &config.AppConfig{}, logger.NewLogger(""))
	checkSession := &mockSSHSession{
		closeFunc: func() error { return nil },
		runFunc:   func(cmd string) error { return fmt.Errorf("not found") },
	}
	readSession := &mockSSHSession{
		closeFunc:  func() error { return nil },
		outputFunc: func(cmd string) ([]byte, error) { return nil, fmt.Errorf("not found") },
	}
	sessionIndex := 0
	sessions := []sshSession{checkSession, readSession}
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) {
				s := sessions[sessionIndex]
				sessionIndex++
				return s, nil
			},
		},
	}
	d.sshClients["192.168.1.1"] = client

	nodeResult := &config.NodeResult{IP: "192.168.1.1", Status: "running"}
	result := &config.DispatchResult{}
	completed := make(map[string]bool)

	handled := d.collectFromNode("192.168.1.1", nodeResult, result, completed)
	assert.False(t, handled)
}

func TestCollectResultsWithResultFile(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{WorkDir: "/tmp", Timeout: 1, PollInterval: 1}, &config.AppConfig{}, logger.NewLogger(""))
	checkSession := &mockSSHSession{
		closeFunc: func() error { return nil },
		runFunc:   func(cmd string) error { return nil },
	}
	file := &mockSFTPFile{}
	file.WriteString(`[{"checkType":"kernel","status":"pass"}]`)
	file.closeFunc = func() error { return nil }
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return checkSession, nil },
		},
		sftpClient: &mockSFTPClient{
			openFunc: func(path string) (sftpFile, error) { return file, nil },
		},
	}
	d.sshClients["192.168.1.1"] = client

	nodeStatuses := map[string]*config.NodeResult{
		"192.168.1.1": {IP: "192.168.1.1", Status: "running"},
	}
	result := &config.DispatchResult{}
	d.collectResults(nodeStatuses, result)
	assert.Len(t, result.Nodes, 1)
	assert.Equal(t, "success", result.Nodes[0].Status)
}

func TestCollectResultsWithErrorFile(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{WorkDir: "/tmp", Timeout: 1, PollInterval: 1}, &config.AppConfig{}, logger.NewLogger(""))
	checkSession := &mockSSHSession{
		closeFunc: func() error { return nil },
		runFunc:   func(cmd string) error { return fmt.Errorf("not found") },
	}
	readSession := &mockSSHSession{
		closeFunc:  func() error { return nil },
		outputFunc: func(cmd string) ([]byte, error) { return []byte("node error"), nil },
	}
	sessions := []sshSession{checkSession, readSession}
	sessionIndex := 0
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) {
				s := sessions[sessionIndex]
				sessionIndex++
				return s, nil
			},
		},
	}
	d.sshClients["192.168.1.1"] = client

	nodeStatuses := map[string]*config.NodeResult{
		"192.168.1.1": {IP: "192.168.1.1", Status: "running"},
	}
	result := &config.DispatchResult{}
	d.collectResults(nodeStatuses, result)
	assert.Len(t, result.Nodes, 1)
	assert.Equal(t, "failed", result.Nodes[0].Status)
	assert.Equal(t, "node error", result.Nodes[0].Error)
}

func TestCollectResultsTimeout(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{WorkDir: "/tmp", Timeout: 1, PollInterval: 1}, &config.AppConfig{}, logger.NewLogger(""))
	nodeStatuses := map[string]*config.NodeResult{
		"192.168.1.1": {IP: "192.168.1.1", Status: "running"},
	}
	result := &config.DispatchResult{}
	d.collectResults(nodeStatuses, result)
	assert.Len(t, result.Nodes, 1)
	assert.Equal(t, "failed", result.Nodes[0].Status)
	assert.Contains(t, result.Nodes[0].Error, "SSH client not available")
}

func TestExecuteRemotelySuccess(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{WorkDir: "/tmp"}, &config.AppConfig{}, logger.NewLogger(""))
	session := &mockSSHSession{
		closeFunc:  func() error { return nil },
		outputFunc: func(cmd string) ([]byte, error) { return []byte("1234\n"), nil },
	}
	d.sshClients["192.168.1.1"] = &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return session, nil },
		},
		host: config.Host{Role: []string{"master"}},
	}
	results := d.executeRemotely()
	assert.Len(t, results, 1)
	assert.Equal(t, "running", results["192.168.1.1"].Status)
}

func TestExecuteRemotelySessionError(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{WorkDir: "/tmp"}, &config.AppConfig{}, logger.NewLogger(""))
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return nil, fmt.Errorf("session error") },
		},
		host: config.Host{Role: []string{"master"}},
	}
	d.sshClients["192.168.1.1"] = client
	results := d.executeRemotely()
	assert.Len(t, results, 1)
	assert.Equal(t, "failed", results["192.168.1.1"].Status)
}

func TestDispatchFilesSuccess(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{WorkDir: "/tmp", ConcurrentLimit: 1}, &config.AppConfig{}, logger.NewLogger(""))
	dir := t.TempDir()
	binaryPath := filepath.Join(dir, "envCheck_amd64")
	err := os.WriteFile(binaryPath, []byte("binary"), 0644)
	assert.NoError(t, err)

	oldWd, _ := os.Getwd()
	os.Chdir(dir)
	defer os.Chdir(oldWd)

	compressSession := &mockSSHSession{
		closeFunc: func() error { return nil },
		runFunc:   func(cmd string) error { return nil },
	}
	chmodSession := &mockSSHSession{
		closeFunc: func() error { return nil },
		runFunc:   func(cmd string) error { return nil },
	}
	sessions := []sshSession{compressSession, chmodSession}
	sessionIndex := 0
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) {
				s := sessions[sessionIndex]
				sessionIndex++
				return s, nil
			},
		},
		sftpClient: &mockSFTPClient{
			statFunc: func(path string) (os.FileInfo, error) {
				return nil, fmt.Errorf("not found")
			},
			createFunc: func(path string) (sftpFile, error) {
				return &mockSFTPFile{closeFunc: func() error { return nil }}, nil
			},
			removeFunc: func(path string) error { return nil },
		},
		host: config.Host{Arch: "amd64", IP: "192.168.1.1"},
	}
	d.sshClients["192.168.1.1"] = client

	err = d.transferFiles()
	assert.NoError(t, err)
}

func TestDispatchFilesFailure(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{WorkDir: "/tmp", ConcurrentLimit: 1}, &config.AppConfig{}, logger.NewLogger(""))
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return nil, fmt.Errorf("session error") },
		},
		sftpClient: &mockSFTPClient{
			statFunc: func(path string) (os.FileInfo, error) { return nil, fmt.Errorf("not found") },
		},
		host: config.Host{IP: "192.168.1.1"},
	}
	d.sshClients["192.168.1.1"] = client

	err := d.transferFiles()
	assert.Error(t, err)
}

func TestEstablishConnectionsSuccess(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{ConcurrentLimit: 1}, &config.AppConfig{
		Hosts: []config.Host{{IP: "192.168.1.1"}},
	}, logger.NewLogger(""))
	d.newSSHClient = func(host config.Host) (*SSHClient, error) {
		return &SSHClient{host: host}, nil
	}
	err := d.establishConnections()
	assert.NoError(t, err)
	assert.Len(t, d.sshClients, 1)
}

func TestEstablishConnectionsFailure(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{ConcurrentLimit: 1}, &config.AppConfig{
		Hosts: []config.Host{{IP: "192.168.1.1"}},
	}, logger.NewLogger(""))
	d.newSSHClient = func(host config.Host) (*SSHClient, error) {
		return nil, fmt.Errorf("connection failed")
	}
	err := d.establishConnections()
	assert.Error(t, err)
}

func TestExecuteSuccess(t *testing.T) {
	dir := t.TempDir()
	binaryPath := filepath.Join(dir, "envCheck_amd64")
	err := os.WriteFile(binaryPath, []byte("binary"), 0644)
	assert.NoError(t, err)

	oldWd, _ := os.Getwd()
	os.Chdir(dir)
	defer os.Chdir(oldWd)

	d := NewDispatcher(
		&config.DispatchConfig{WorkDir: "/tmp", ConcurrentLimit: 1, Timeout: 1, PollInterval: 1},
		&config.AppConfig{
			Hosts: []config.Host{{IP: "192.168.1.1", Role: []string{"master"}}},
		},
		logger.NewLogger(""),
	)

	session := &mockSSHSession{
		closeFunc:  func() error { return nil },
		outputFunc: func(cmd string) ([]byte, error) { return []byte("amd64\n"), nil },
		runFunc:    func(cmd string) error { return nil },
	}
	file := &mockSFTPFile{}
	file.WriteString(`[{"checkType":"kernel","status":"pass"}]`)
	file.closeFunc = func() error { return nil }
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return session, nil },
		},
		sftpClient: &mockSFTPClient{
			statFunc:   func(path string) (os.FileInfo, error) { return nil, fmt.Errorf("not found") },
			createFunc: func(path string) (sftpFile, error) { return file, nil },
			removeFunc: func(path string) error { return nil },
			openFunc:   func(path string) (sftpFile, error) { return file, nil },
		},
		host: config.Host{IP: "192.168.1.1", Role: []string{"master"}},
	}
	d.newSSHClient = func(host config.Host) (*SSHClient, error) { return client, nil }

	result, err := d.Execute()
	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, "PASS", result.Summary.Result)
}

func TestUploadFileSuccessFirstAttempt(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{}, &config.AppConfig{}, logger.NewLogger(""))
	dir := t.TempDir()
	localPath := filepath.Join(dir, "testfile")
	err := os.WriteFile(localPath, []byte("data"), 0644)
	assert.NoError(t, err)

	file := &mockSFTPFile{closeFunc: func() error { return nil }}
	client := &SSHClient{
		sftpClient: &mockSFTPClient{
			removeFunc: func(path string) error { return nil },
			createFunc: func(path string) (sftpFile, error) { return file, nil },
			statFunc: func(path string) (os.FileInfo, error) {
				return &mockFileInfo{size: 4}, nil
			},
		},
	}
	err = d.uploadFile(client, localPath, "/remote/testfile")
	assert.NoError(t, err)
}

func TestUploadFileRetrySuccess(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{}, &config.AppConfig{}, logger.NewLogger(""))
	dir := t.TempDir()
	localPath := filepath.Join(dir, "testfile")
	err := os.WriteFile(localPath, []byte("data"), 0644)
	assert.NoError(t, err)

	file := &mockSFTPFile{closeFunc: func() error { return nil }}
	attempt := 0
	client := &SSHClient{
		sftpClient: &mockSFTPClient{
			removeFunc: func(path string) error { return nil },
			createFunc: func(path string) (sftpFile, error) {
				attempt++
				if attempt == 1 {
					return nil, fmt.Errorf("first attempt fail")
				}
				return file, nil
			},
			statFunc: func(path string) (os.FileInfo, error) {
				return &mockFileInfo{size: 4}, nil
			},
		},
	}
	err = d.uploadFile(client, localPath, "/remote/testfile")
	assert.NoError(t, err)
	assert.Equal(t, 2, attempt)
}

func TestUploadFileFinalFailure(t *testing.T) {
	d := NewDispatcher(&config.DispatchConfig{}, &config.AppConfig{}, logger.NewLogger(""))
	dir := t.TempDir()
	localPath := filepath.Join(dir, "testfile")
	err := os.WriteFile(localPath, []byte("data"), 0644)
	assert.NoError(t, err)

	client := &SSHClient{
		sftpClient: &mockSFTPClient{
			removeFunc: func(path string) error { return nil },
			createFunc: func(path string) (sftpFile, error) {
				return nil, fmt.Errorf("always fail")
			},
		},
	}
	err = d.uploadFile(client, localPath, "/remote/testfile")
	assert.Error(t, err)
}

func TestExecuteStepsSuccess(t *testing.T) {
	dir := t.TempDir()
	binaryPath := filepath.Join(dir, "envCheck_amd64")
	err := os.WriteFile(binaryPath, []byte("binary"), 0644)
	assert.NoError(t, err)

	oldWd, _ := os.Getwd()
	os.Chdir(dir)
	defer os.Chdir(oldWd)

	d := NewDispatcher(
		&config.DispatchConfig{WorkDir: "/tmp", ConcurrentLimit: 1, Timeout: 1, PollInterval: 1},
		&config.AppConfig{
			Hosts: []config.Host{{IP: "192.168.1.1", Role: []string{"master"}}},
		},
		logger.NewLogger(""),
	)

	session := &mockSSHSession{
		closeFunc:  func() error { return nil },
		outputFunc: func(cmd string) ([]byte, error) { return []byte("amd64\n"), nil },
		runFunc:    func(cmd string) error { return nil },
	}
	file := &mockSFTPFile{}
	file.WriteString(`[{"checkType":"kernel","status":"pass"}]`)
	file.closeFunc = func() error { return nil }
	client := &SSHClient{
		client: &mockSSHConn{
			newSessionFunc: func() (sshSession, error) { return session, nil },
		},
		sftpClient: &mockSFTPClient{
			statFunc:   func(path string) (os.FileInfo, error) { return nil, fmt.Errorf("not found") },
			createFunc: func(path string) (sftpFile, error) { return file, nil },
			removeFunc: func(path string) error { return nil },
			openFunc:   func(path string) (sftpFile, error) { return file, nil },
		},
		host: config.Host{IP: "192.168.1.1", Role: []string{"master"}},
	}
	d.newSSHClient = func(host config.Host) (*SSHClient, error) { return client, nil }

	result := &config.DispatchResult{}
	err = d.executeSteps(result)
	assert.NoError(t, err)
}
