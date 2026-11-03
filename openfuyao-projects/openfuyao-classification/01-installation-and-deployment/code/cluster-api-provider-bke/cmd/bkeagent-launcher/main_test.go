/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package main

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"

	texttemplate "text/template"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
)

func init() {}

const (
	numZero        = 0
	numOne         = 1
	numTwo         = 2
	numThree       = 3
	numFour        = 4
	numFive        = 5
	numSix         = 6
	numSeven       = 7
	numEight       = 8
	numTwelve      = 12
	numSixteen     = 16
	numThirtySeven = 37
	numFifty       = 50
	numTwoHundred  = 200
	numFourHundred = 400
	numFiveHundred = 500
	numSixHundred  = 600

	testHostname       = "test-node"
	testDebug          = "true"
	testNtpServer      = "ntp.example.com"
	testHealthPort     = "3377"
	testKubeconfigPath = "/etc/bkeagent/launcher/config"
	testServiceContent = "[Unit]\nDescription=BKE Agent"
)

func TestValidateFlagWithEmptyNtpServer(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		os.Getenv,
		"true",
	)
	defer patches.Reset()

	patches.ApplyFunc(
		os.Exit,
		func(code int) {
		},
	)

	ntpServer = ""
	healthPort = testHealthPort
	kubeconfig = "/path/to/kubeconfig"

	validateFlag()
}

func TestValidateFlagWithEmptyHealthPort(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		os.Getenv,
		"true",
	)
	defer patches.Reset()

	exitCalled := false
	patches.ApplyFunc(
		os.Exit,
		func(code int) {
			exitCalled = true
		},
	)

	ntpServer = testNtpServer
	healthPort = ""
	kubeconfig = "/path/to/kubeconfig"

	validateFlag()

	assert.True(t, exitCalled)
}

func TestValidateFlagWithEmptyKubeconfig(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		os.Getenv,
		"true",
	)
	defer patches.Reset()

	exitCalled := false
	patches.ApplyFunc(
		os.Exit,
		func(code int) {
			exitCalled = true
		},
	)

	ntpServer = testNtpServer
	healthPort = testHealthPort
	kubeconfig = ""

	validateFlag()

	assert.True(t, exitCalled)
}

func TestValidateFlagWithAllValid(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		os.Getenv,
		"true",
	)
	defer patches.Reset()

	ntpServer = testNtpServer
	healthPort = testHealthPort
	kubeconfig = "/path/to/kubeconfig"

	validateFlag()
}

func TestGetHostnameSuccess(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		executeCommand,
		testHostname,
		nil,
	)
	defer patches.Reset()

	hostname, err := getHostname()

	assert.NoError(t, err)
	assert.Equal(t, testHostname, hostname)
}

func TestGetHostnameError(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		executeCommand,
		"",
		errors.New("command error"),
	)
	defer patches.Reset()

	patches.ApplyFunc(
		os.Exit,
		func(code int) {
		},
	)

	getHostname()
}

func TestCopyFileSuccess(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		executeCommand,
		"",
		nil,
	)
	defer patches.Reset()

	err := copyFile("/src", "/dst")

	assert.NoError(t, err)
}

func TestCopyFileError(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		executeCommand,
		"",
		errors.New("copy error"),
	)
	defer patches.Reset()

	err := copyFile("/src", "/dst")

	assert.Error(t, err)
}

func TestWriteFileSuccess(t *testing.T) {
	tmpDir := t.TempDir()
	testFile := tmpDir + "/testfile"

	err := writeFile(testFile, "test content")

	assert.NoError(t, err)

	content, err := os.ReadFile(testFile)
	assert.NoError(t, err)
	assert.Equal(t, "test content", string(content))
}

func TestWriteFilePermission(t *testing.T) {
	testFile := filepath.Join(os.TempDir(), "bkeagent_test_node")

	err := writeFile(testFile, "test")

	assert.NoError(t, err)

	os.Remove(testFile)
}

func TestPrepareBkeagentBinarySuccess(t *testing.T) {
	var mockCmd exec.Cmd
	execPatches := gomonkey.ApplyFunc(
		exec.Command,
		func(name string, args ...string) *exec.Cmd {
			return &mockCmd
		},
	)
	defer execPatches.Reset()

	execPatches.ApplyMethod(
		&mockCmd,
		"CombinedOutput",
		func(cmd *exec.Cmd) ([]byte, error) {
			return []byte(""), nil
		},
	)

	copyFilePatches := gomonkey.ApplyFunc(
		copyFile,
		func(src, dst string) error {
			return nil
		},
	)
	defer copyFilePatches.Reset()

	err := prepareBkeagentBinary()

	assert.NoError(t, err)
}

func TestPrepareBkeagentBinaryCopyError(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		executeCommand,
		"",
		errors.New("copy error"),
	)
	defer patches.Reset()

	err := prepareBkeagentBinary()

	assert.Error(t, err)
}

func TestPrepareBkeagentServiceTemplateParseError(t *testing.T) {
	tmpDir := t.TempDir()
	originalSrc := bkeagentServiceSrc
	bkeagentServiceSrc = tmpDir + "/bkeagent.service"

	tmpl := texttemplate.New("test")
	patches := gomonkey.ApplyFunc(
		texttemplate.New,
		func(name string) *texttemplate.Template {
			return tmpl
		},
	)
	defer patches.Reset()

	patches.ApplyFunc(
		(*texttemplate.Template).Parse,
		func(tmpl *texttemplate.Template, text string) (*texttemplate.Template, error) {
			return nil, errors.New("parse error")
		},
	)

	debug = testDebug
	ntpServer = testNtpServer
	healthPort = testHealthPort

	err := prepareBkeagentService()

	assert.Error(t, err)

	bkeagentServiceSrc = originalSrc
}

func TestPrepareKubeconfigCopyFileError(t *testing.T) {
	tmpDir := t.TempDir()
	originalKubeconfigSrc := kubeconfigSrc
	originalKubeconfigDst := kubeconfigDst
	kubeconfigSrc = tmpDir + "/config"
	kubeconfigDst = tmpDir + "/config"

	patches := gomonkey.ApplyFuncReturn(
		clientcmd.LoadFromFile,
		&clientcmdapi.Config{
			APIVersion: "v1",
			Kind:       "Config",
			Clusters:   map[string]*clientcmdapi.Cluster{},
			AuthInfos:  map[string]*clientcmdapi.AuthInfo{},
			Contexts:   map[string]*clientcmdapi.Context{},
		},
		nil,
	)
	defer patches.Reset()

	patches.ApplyFuncReturn(
		clientcmd.WriteToFile,
		nil,
	)

	patches.ApplyFuncReturn(
		copyFile,
		errors.New("copy error"),
	)

	originalKubeconfig := kubeconfig
	kubeconfig = tmpDir + "/kubeconfig"

	err := prepareKubeconfig()

	assert.Error(t, err)

	kubeconfigSrc = originalKubeconfigSrc
	kubeconfigDst = originalKubeconfigDst
	kubeconfig = originalKubeconfig
}

func TestPrepareNodeFileCopyError(t *testing.T) {
	tmpDir := t.TempDir()
	originalNodeSrc := nodeSrc
	originalNodeDst := nodeDst
	nodeSrc = tmpDir + "/node"
	nodeDst = tmpDir + "/node"

	patches := gomonkey.ApplyFuncReturn(
		copyFile,
		errors.New("copy error"),
	)
	defer patches.Reset()

	err := prepareNodeFile(testHostname)

	assert.Error(t, err)

	nodeSrc = originalNodeSrc
	nodeDst = originalNodeDst
}

func TestStartPrePrepareBkeagentServiceError(t *testing.T) {
	callCount := 0
	patches := gomonkey.ApplyFunc(
		executeCommand,
		func(cmdStr string) (string, error) {
			callCount++
			if callCount <= numTwo {
				return "", nil
			}
			return "", errors.New("service error")
		},
	)
	defer patches.Reset()

	patches.ApplyFunc(
		prepareBkeagentBinary,
		func() error {
			return nil
		},
	)

	patches.ApplyFunc(
		prepareBkeagentService,
		func() error {
			return errors.New("service error")
		},
	)

	err := startPre()

	assert.Error(t, err)
}

func TestStartPrePrepareKubeconfigError(t *testing.T) {
	callCount := 0
	patches := gomonkey.ApplyFunc(
		executeCommand,
		func(cmdStr string) (string, error) {
			callCount++
			if callCount <= numTwo {
				return "", nil
			}
			return "", errors.New("kubeconfig error")
		},
	)
	defer patches.Reset()

	patches.ApplyFunc(
		prepareBkeagentBinary,
		func() error {
			return nil
		},
	)

	patches.ApplyFunc(
		prepareBkeagentService,
		func() error {
			return nil
		},
	)

	patches.ApplyFunc(
		prepareKubeconfig,
		func() error {
			return errors.New("kubeconfig error")
		},
	)

	err := startPre()

	assert.Error(t, err)
}

func TestPrepareBkeagentServiceSuccess(t *testing.T) {
	tmpDir := t.TempDir()
	originalSrc := bkeagentServiceSrc
	bkeagentServiceSrc = tmpDir + "/bkeagent.service"

	patches := gomonkey.ApplyFunc(
		copyFile,
		func(src, dst string) error {
			return nil
		},
	)
	defer patches.Reset()

	debug = testDebug
	ntpServer = testNtpServer
	healthPort = testHealthPort
	logLevel = ""

	err := prepareBkeagentService()

	assert.NoError(t, err)

	bkeagentServiceSrc = originalSrc
}

func TestPrepareBkeagentServiceWithLogLevel(t *testing.T) {
	tmpDir := t.TempDir()
	originalSrc := bkeagentServiceSrc
	defer func() {
		bkeagentServiceSrc = originalSrc
		logLevel = ""
	}()
	servicePath := filepath.Join(tmpDir, "bkeagent.service")
	bkeagentServiceSrc = servicePath

	patches := gomonkey.ApplyFunc(
		copyFile,
		func(src, dst string) error {
			return nil
		},
	)
	defer patches.Reset()

	debug = testDebug
	ntpServer = testNtpServer
	healthPort = testHealthPort
	logLevel = "DEBUG"

	require.NoError(t, prepareBkeagentService())

	data, err := os.ReadFile(servicePath)
	require.NoError(t, err)
	content := string(data)
	assert.Contains(t, content, "--log-level=DEBUG")
	assert.Contains(t, content, "--ntpserver="+testNtpServer)
	assert.Contains(t, content, "--health-port="+testHealthPort)
}

func TestPrepareBkeagentServiceOpenError(t *testing.T) {
	originalSrc := bkeagentServiceSrc
	defer func() {
		bkeagentServiceSrc = originalSrc
	}()
	notDir := filepath.Join(t.TempDir(), "not-dir")
	assert.NoError(t, os.WriteFile(notDir, []byte("file"), 0644))
	bkeagentServiceSrc = filepath.Join(notDir, "bkeagent.service")

	err := prepareBkeagentService()

	assert.Error(t, err)
}

func TestPrepareKubeconfigSuccess(t *testing.T) {
	tmpDir := t.TempDir()
	originalKubeconfigSrc := kubeconfigSrc
	originalKubeconfigDst := kubeconfigDst
	kubeconfigSrc = tmpDir + "/config"
	kubeconfigDst = tmpDir + "/config"

	patches := gomonkey.ApplyFunc(
		clientcmd.LoadFromFile,
		func(filename string) (*clientcmdapi.Config, error) {
			return &clientcmdapi.Config{
				APIVersion: "v1",
				Kind:       "Config",
				Clusters:   map[string]*clientcmdapi.Cluster{},
				AuthInfos:  map[string]*clientcmdapi.AuthInfo{},
				Contexts:   map[string]*clientcmdapi.Context{},
			}, nil
		},
	)
	defer patches.Reset()

	patches.ApplyFunc(
		clientcmd.WriteToFile,
		func(config clientcmdapi.Config, filename string) error {
			// Write a real file so prepareKubeconfig can chmod it (mode 0600).
			return os.WriteFile(filename, []byte("apiVersion: v1\nkind: Config\n"), 0o644)
		},
	)

	patches.ApplyFunc(
		copyFile,
		func(src, dst string) error {
			return nil
		},
	)

	originalKubeconfig := kubeconfig
	kubeconfig = tmpDir + "/kubeconfig"

	err := prepareKubeconfig()

	assert.NoError(t, err)
	info, statErr := os.Stat(kubeconfigSrc)
	assert.NoError(t, statErr)
	if runtime.GOOS != "windows" {
		assert.Equal(t, os.FileMode(utils.KubeConfigFilePerm), info.Mode().Perm())
	}

	kubeconfigSrc = originalKubeconfigSrc
	kubeconfigDst = originalKubeconfigDst
	kubeconfig = originalKubeconfig
}

func TestPrepareKubeconfigLoadError(t *testing.T) {
	originalKubeconfig := kubeconfig
	kubeconfig = "/invalid/kubeconfig"

	patches := gomonkey.ApplyFuncReturn(
		clientcmd.LoadFromFile,
		nil,
		errors.New("load error"),
	)
	defer patches.Reset()

	err := prepareKubeconfig()

	assert.Error(t, err)

	kubeconfig = originalKubeconfig
}

func TestPrepareKubeconfigWriteError(t *testing.T) {
	tmpDir := t.TempDir()
	originalKubeconfigSrc := kubeconfigSrc
	originalKubeconfigDst := kubeconfigDst
	kubeconfigSrc = tmpDir + "/config"
	kubeconfigDst = tmpDir + "/config"

	patches := gomonkey.ApplyFuncReturn(
		clientcmd.LoadFromFile,
		&clientcmdapi.Config{
			APIVersion: "v1",
			Kind:       "Config",
			Clusters:   map[string]*clientcmdapi.Cluster{},
			AuthInfos:  map[string]*clientcmdapi.AuthInfo{},
			Contexts:   map[string]*clientcmdapi.Context{},
		},
		nil,
	)
	defer patches.Reset()

	patches.ApplyFuncReturn(
		clientcmd.LoadFromFile,
		&clientcmdapi.Config{
			APIVersion: "v1",
			Kind:       "Config",
			Clusters:   map[string]*clientcmdapi.Cluster{},
			AuthInfos:  map[string]*clientcmdapi.AuthInfo{},
			Contexts:   map[string]*clientcmdapi.Context{},
		},
		nil,
	)
	defer patches.Reset()

	originalKubeconfig := kubeconfig
	kubeconfig = tmpDir + "/kubeconfig"

	err := prepareKubeconfig()

	assert.Error(t, err)

	kubeconfigSrc = originalKubeconfigSrc
	kubeconfigDst = originalKubeconfigDst
	kubeconfig = originalKubeconfig
}

func TestPrepareNodeFileSuccess(t *testing.T) {
	tmpDir := t.TempDir()
	originalNodeSrc := nodeSrc
	originalNodeDst := nodeDst
	nodeSrc = tmpDir + "/node"
	nodeDst = tmpDir + "/node"

	patches := gomonkey.ApplyFunc(
		copyFile,
		func(src, dst string) error {
			return nil
		},
	)
	defer patches.Reset()

	err := prepareNodeFile(testHostname)

	assert.NoError(t, err)

	nodeSrc = originalNodeSrc
	nodeDst = originalNodeDst
}

func TestPrepareNodeFileWriteError(t *testing.T) {
	originalNodeSrc := nodeSrc
	defer func() {
		nodeSrc = originalNodeSrc
	}()
	notDir := filepath.Join(t.TempDir(), "not-dir")
	assert.NoError(t, os.WriteFile(notDir, []byte("file"), 0644))
	nodeSrc = filepath.Join(notDir, "node")

	err := prepareNodeFile(testHostname)

	assert.Error(t, err)
}

func TestStartPreStopBkeagentError(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		executeCommand,
		"",
		errors.New("stop error"),
	)
	defer patches.Reset()

	err := startPre()

	assert.Error(t, err)
}

func TestStartPreResolveNodeNameError(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		executeCommand,
		"",
		nil,
	)
	defer patches.Reset()

	patches.ApplyFunc(
		resolveNodeName,
		func() (string, error) {
			return "", errors.New("node name error")
		},
	)

	err := startPre()
	assert.Error(t, err)
}

func TestStartPrePrepareBkeagentBinaryError(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		executeCommand,
		"",
		nil,
	)
	defer patches.Reset()

	patches.ApplyFunc(
		resolveNodeName,
		func() (string, error) {
			return testHostname, nil
		},
	)

	patches.ApplyFunc(
		prepareBkeagentBinary,
		func() error {
			return errors.New("binary error")
		},
	)

	err := startPre()

	assert.Error(t, err)
}

func TestStartPreSuccess(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		executeCommand,
		"",
		nil,
	)
	defer patches.Reset()

	patches.ApplyFunc(
		resolveNodeName,
		func() (string, error) {
			return testHostname, nil
		},
	)

	patches.ApplyFunc(
		prepareBkeagentBinary,
		func() error {
			return nil
		},
	)

	patches.ApplyFunc(
		prepareBkeagentService,
		func() error {
			return nil
		},
	)

	patches.ApplyFunc(
		prepareKubeconfig,
		func() error {
			return nil
		},
	)

	patches.ApplyFunc(
		prepareNodeFile,
		func(nodeName string) error {
			return nil
		},
	)

	err := startPre()

	assert.NoError(t, err)
}

func TestPingBKEAgentSuccess(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		executeCommand,
		"active",
		nil,
	)
	defer patches.Reset()

	err := pingBKEAgent()

	assert.NoError(t, err)
}

func TestPingBKEAgentNotActive(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		executeCommand,
		"inactive",
		nil,
	)
	defer patches.Reset()

	err := pingBKEAgent()

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not ready")
}

func TestPingBKEAgentCommandError(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		executeCommand,
		"",
		errors.New("command error"),
	)
	defer patches.Reset()

	err := pingBKEAgent()

	assert.Error(t, err)
}

func TestStartSuccess(t *testing.T) {
	callCount := 0
	patches := gomonkey.ApplyFunc(
		executeCommand,
		func(cmdStr string) (string, error) {
			callCount++
			return "", nil
		},
	)
	defer patches.Reset()

	err := start()

	assert.NoError(t, err)
	assert.Equal(t, numThree, callCount)
}

func TestStartDaemonReloadError(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		executeCommand,
		"",
		errors.New("daemon-reload error"),
	)
	defer patches.Reset()

	err := start()

	assert.Error(t, err)
}

func TestStartStartBkeagentError(t *testing.T) {
	callCount := 0
	patches := gomonkey.ApplyFunc(
		executeCommand,
		func(cmdStr string) (string, error) {
			callCount++
			if callCount == numTwo {
				return "", errors.New("start error")
			}
			return "", nil
		},
	)
	defer patches.Reset()

	err := start()

	assert.Error(t, err)
}

func TestStartEnableBkeagentError(t *testing.T) {
	callCount := 0
	patches := gomonkey.ApplyFunc(
		executeCommand,
		func(cmdStr string) (string, error) {
			callCount++
			if callCount == numThree {
				return "", errors.New("enable error")
			}
			return "", nil
		},
	)
	defer patches.Reset()

	err := start()

	assert.Error(t, err)
}

func TestExecuteCommandSuccess(t *testing.T) {
	var mockCmd exec.Cmd
	execPatches := gomonkey.ApplyFunc(
		exec.Command,
		func(name string, args ...string) *exec.Cmd {
			return &mockCmd
		},
	)
	defer execPatches.Reset()

	execPatches.ApplyMethod(
		&mockCmd,
		"CombinedOutput",
		func(cmd *exec.Cmd) ([]byte, error) {
			return []byte("test output"), nil
		},
	)

	output, err := executeCommand("echo test")

	assert.NoError(t, err)
	assert.Equal(t, "test output", output)
}

func TestExecuteCommandError(t *testing.T) {
	var mockCmd exec.Cmd
	execPatches := gomonkey.ApplyFunc(
		exec.Command,
		func(name string, args ...string) *exec.Cmd {
			return &mockCmd
		},
	)
	defer execPatches.Reset()

	execPatches.ApplyMethod(
		&mockCmd,
		"CombinedOutput",
		func(cmd *exec.Cmd) ([]byte, error) {
			return []byte("error output"), errors.New("command error")
		},
	)

	output, err := executeCommand("invalid_command_xyz")

	assert.Error(t, err)
	assert.Equal(t, "error output", output)
}

func TestConstants(t *testing.T) {
	assert.Equal(t, "/etc/openFuyao/bkeagent/launcher", launcherDir)
	assert.Equal(t, "/etc/openFuyao/bkeagent", workDir)
	assert.Equal(t, "./bkeagent", localBKEAgentBinarySrc)
}

func TestPathConstants(t *testing.T) {
	assert.Equal(t, filepath.Join("/", "etc", "openFuyao", "bkeagent", "launcher", "bkeagent"), bkeagentBinarySrc)
	assert.Equal(t, filepath.Join("/", "etc", "openFuyao", "bkeagent", "launcher", "bkeagent.service"), bkeagentServiceSrc)
	assert.Equal(t, filepath.Join("/", "etc", "openFuyao", "bkeagent", "launcher", "config"), kubeconfigSrc)
	assert.Equal(t, filepath.Join("/", "etc", "openFuyao", "bkeagent", "launcher", "node"), nodeSrc)
	assert.Equal(t, "/usr/local/bin/bkeagent", bkeagentBinaryDst)
	assert.Equal(t, "/etc/systemd/system/bkeagent.service", bkeagentServiceDst)
	assert.Equal(t, filepath.Join("/", "etc", "openFuyao", "bkeagent", "config"), kubeconfigDst)
	assert.Equal(t, filepath.Join("/", "etc", "openFuyao", "bkeagent", "node"), nodeDst)
}

func TestInitFlag(t *testing.T) {
	initFlag()
}

func TestValidateFlagAllEmpty(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		os.Getenv,
		"true",
	)
	defer patches.Reset()

	exitCalled := false
	patches.ApplyFunc(
		os.Exit,
		func(code int) {
			exitCalled = true
		},
	)

	ntpServer = ""
	healthPort = ""
	kubeconfig = ""

	validateFlag()

	assert.True(t, exitCalled)
}
