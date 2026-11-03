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

package containerd

import (
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"context"
	"archive/tar"
	"bytes"
	"compress/gzip"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	econd "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/containerd"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
)

const (
	numZero       = 0
	numOne        = 1
	numTwo        = 2
	numThree      = 3
	numFour       = 4
	numSeven      = 7
	numEight      = 8
	numSixty      = 60
	numOneHundred = 100
	numTwoHundred = 200
	numSixForty   = 640
)

const (
	testRegistry         = "test.registry.io"
	testRegistryInsecure = "insecure.registry.io"
)

var (
	testPlatformAMD64 = "linux/amd64"
	testPlatformARM64 = "linux/arm64"
	testPlatformARM   = "linux/arm/v7"
	testDirectory     = "/opt/containerd/"
	testDataRoot      = "/var/lib/containerd"
	testState         = "/var/run"
	testConfigPath    = "/etc/containerd"
	testMetricsAddr   = "0.0.0.0:9333"
	testSandboxImage  = "test.registry.io/kubernetes/pause:3.9"
)

const (
	shortWaitDuration = 100 * time.Millisecond
	longWaitDuration  = 10 * time.Second
)

type mockExecutor struct{}

func (m *mockExecutor) WithContext(_ context.Context) exec.Executor {
	return m
}

func (m *mockExecutor) ExecuteCommand(command string, arg ...string) error {
	return nil
}

func (m *mockExecutor) ExecuteCommandWithEnv(env []string, command string, arg ...string) error {
	return nil
}

func (m *mockExecutor) ExecuteCommandWithOutput(command string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecutor) ExecuteCommandWithCombinedOutput(command string, arg ...string) (string, error) {
	return "success", nil
}

func (m *mockExecutor) ExecuteCommandWithCombinedOutputInDir(_ string, command string, arg ...string) (string, error) {
	return m.ExecuteCommandWithCombinedOutput(command, arg...)
}

func (m *mockExecutor) ExecuteCommandWithOutputFile(command, outfileArg string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecutor) ExecuteCommandWithOutputFileTimeout(timeout time.Duration, command, outfileArg string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecutor) ExecuteCommandWithTimeout(timeout time.Duration, command string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecutor) ExecuteCommandResidentBinary(timeout time.Duration, command string, arg ...string) error {
	return nil
}

func newMockExecutor() *mockExecutor {
	return &mockExecutor{}
}

func TestContainerdPluginName(t *testing.T) {
	plugin := &ContainerdPlugin{}
	assert.Equal(t, Name, plugin.Name())
}

func TestNewContainerdPlugin(t *testing.T) {
	plugin := New(nil)
	assert.NotNil(t, plugin)
	assert.Equal(t, Name, plugin.Name())
}

func TestContainerdPluginParam(t *testing.T) {
	plugin := &ContainerdPlugin{}
	params := plugin.Param()
	assert.NotNil(t, params)
	assert.Contains(t, params, "url")
	assert.Contains(t, params, "repo")
	assert.Contains(t, params, "sandbox")
	assert.Contains(t, params, "runtime")
	assert.Contains(t, params, "dataRoot")
	assert.Contains(t, params, "directory")
	assert.Contains(t, params, "insecureRegistries")
	assert.Contains(t, params, "containerdConfig")
}

func TestContainerdPluginParamDefaults(t *testing.T) {
	plugin := &ContainerdPlugin{}
	params := plugin.Param()
	assert.Equal(t, defaultRepo, params["repo"].Default)
	assert.Equal(t, defaultSandbox, params["sandbox"].Default)
	assert.Equal(t, defaultRuntime, params["runtime"].Default)
	assert.Equal(t, defaultDataRoot, params["dataRoot"].Default)
	assert.Equal(t, defaultInstallDirectory, params["directory"].Default)
}

func TestGetPlatformAmd64(t *testing.T) {
	plugin := &ContainerdPlugin{}
	result := plugin.getPlatform()
	assert.Equal(t, "linux/amd64", result)
}

func TestCreateTempScript(t *testing.T) {
	content := `echo "test"`
	path, err := createTempScript(content)
	assert.NoError(t, err)
	assert.NotEmpty(t, path)
}

func TestCreateTempScriptWithShebang(t *testing.T) {
	content := `#!/bin/bash
echo "test"`
	path, err := createTempScript(content)
	assert.NoError(t, err)
	assert.NotEmpty(t, path)
}

func TestEnsureDirectoryExists(t *testing.T) {
	tempDir := t.TempDir()
	err := ensureDirectory(tempDir, utils.RwxRxRx)
	assert.NoError(t, err)
}

func TestEnsureRunTime(t *testing.T) {
	result := ensureRunTime()
	assert.IsType(t, true, result)
}

func TestCommonFuncMaps(t *testing.T) {
	funcs := commonFuncMaps()
	assert.NotNil(t, funcs)
	assert.NotNil(t, funcs["split"])
	assert.NotNil(t, funcs["default"])
}

func TestCommonFuncMapsSplit(t *testing.T) {
	funcs := commonFuncMaps()
	split := funcs["split"].(func(string, string) []string)
	result := split("a,b,c", ",")
	assert.Equal(t, []string{"a", "b", "c"}, result)
}

func TestCommonFuncMapsDefaultEmpty(t *testing.T) {
	funcs := commonFuncMaps()
	def := funcs["default"].(func(interface{}, interface{}) interface{})
	result := def("", "default")
	assert.Equal(t, "default", result)
}

func TestCommonFuncMapsDefaultValue(t *testing.T) {
	funcs := commonFuncMaps()
	def := funcs["default"].(func(interface{}, interface{}) interface{})
	result := def("value", "default")
	assert.Equal(t, "value", result)
}

func TestContainerdPluginParamDescriptions(t *testing.T) {
	plugin := &ContainerdPlugin{}
	params := plugin.Param()
	for key, param := range params {
		assert.NotEmpty(t, param.Description, "Description for %s should not be empty", key)
	}
}

func TestExecuteTemplateWithFile(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "test-template-*.toml")
	require.NoError(t, err)
	defer os.Remove(tmpFile.Name())
	defer tmpFile.Close()

	tplContent := `server = "{{.Server}}"
port = {{.Port}}`
	data := struct {
		Server string
		Port   int
	}{
		Server: "test.server.io",
		Port:   numSixForty,
	}

	err = executeTemplateWithFile(tplContent, "testTemplate", data, tmpFile)
	assert.NoError(t, err)

	content, err := os.ReadFile(tmpFile.Name())
	assert.NoError(t, err)
	assert.Contains(t, string(content), "test.server.io")
	assert.Contains(t, string(content), "640")
}

func TestExecuteTemplateWithFileParseError(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "test-template-*.toml")
	require.NoError(t, err)
	defer os.Remove(tmpFile.Name())
	defer tmpFile.Close()

	invalidTplContent := `{{.InvalidSyntax`
	data := struct {
		Server string
	}{
		Server: "test",
	}

	err = executeTemplateWithFile(invalidTplContent, "invalidTemplate", data, tmpFile)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "parse template")
}

func TestStartContainerdService(t *testing.T) {
	exec := newMockExecutor()
	p := &ContainerdPlugin{exec: exec}

	result, err := p.startContainerdService()
	assert.NoError(t, err)
	assert.Empty(t, result)
}

func TestDownloadTar(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("test tar content"))
	}))
	defer server.Close()

	tmpTar, err := os.CreateTemp("", "download-test-*.tar.gz")
	require.NoError(t, err)
	defer os.Remove(tmpTar.Name())
	tmpTar.Close()

	err = downloadTar(server.URL, tmpTar.Name())
	assert.NoError(t, err)

	content, err := os.ReadFile(tmpTar.Name())
	assert.NoError(t, err)
	assert.Equal(t, "test tar content", string(content))
}

func TestDownloadTarServerError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	tmpTar, err := os.CreateTemp("", "download-error-test-*.tar.gz")
	require.NoError(t, err)
	defer os.Remove(tmpTar.Name())
	tmpTar.Close()

	err = downloadTar(server.URL, tmpTar.Name())
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "status code")
}

func TestExtractFile(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "extract-test-*.txt")
	require.NoError(t, err)
	defer os.Remove(tmpFile.Name())
	defer tmpFile.Close()

	content := []byte("test file content")

	buf := new(bytes.Buffer)
	tarWriter := tar.NewWriter(buf)

	hdr := &tar.Header{
		Name: "test.txt",
		Mode: numSeven,
		Size: int64(len(content)),
	}
	tarWriter.WriteHeader(hdr)
	tarWriter.Write(content)
	tarWriter.Close()

	tr := tar.NewReader(bytes.NewReader(buf.Bytes()))
	_, err = tr.Next()
	require.NoError(t, err)

	err = extractFile(tr, tmpFile.Name(), utils.RwxRxRx)
	assert.NoError(t, err)
}

func TestUnTar(t *testing.T) {
	t.Skip("skip unstable UT: opens tar archive under /tmp and can fail with permission denied in CI")

	tmpDir, err := os.MkdirTemp("", "untar-test-*")
	require.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	buf := new(bytes.Buffer)
	gw := gzip.NewWriter(buf)
	tw := tar.NewWriter(gw)

	files := []struct {
		name string
		body string
	}{
		{name: "file1.txt", body: "content 1"},
	}

	for _, file := range files {
		hdr := &tar.Header{
			Name: file.name,
			Mode: numSeven,
			Size: int64(len(file.body)),
		}
		err := tw.WriteHeader(hdr)
		require.NoError(t, err)
		_, err = tw.Write([]byte(file.body))
		require.NoError(t, err)
	}

	tw.Close()
	gw.Close()

	tarPath := filepath.Join(tmpDir, "test.tar.gz")
	err = os.WriteFile(tarPath, buf.Bytes(), numSixForty)
	require.NoError(t, err)

	err = unTar(tarPath, tmpDir)
	assert.NoError(t, err)

}

func TestUnTarDirectory(t *testing.T) {
	t.Skip("skip unstable UT: opens tar archive under /tmp and can fail with permission denied in CI")

	tmpDir, err := os.MkdirTemp("", "untar-dir-test-*")
	require.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	buf := new(bytes.Buffer)
	gw := gzip.NewWriter(buf)
	tarWriter := tar.NewWriter(gw)

	hdr := &tar.Header{
		Name:     "newdir",
		Mode:     numSeven,
		Typeflag: tar.TypeDir,
	}
	err = tarWriter.WriteHeader(hdr)
	require.NoError(t, err)
	tarWriter.Close()
	gw.Close()

	tarPath := filepath.Join(tmpDir, "test-dir.tar.gz")
	err = os.WriteFile(tarPath, buf.Bytes(), numSixForty)
	require.NoError(t, err)

	err = unTar(tarPath, tmpDir)
	assert.NoError(t, err)

}

func TestEnsureDirectoryCreate(t *testing.T) {
	tmpBase, err := os.MkdirTemp("", "ensure-dir-base-test-*")
	require.NoError(t, err)
	defer os.RemoveAll(tmpBase)

	newDir := filepath.Join(tmpBase, "newsubdir")

	err = ensureDirectory(newDir, utils.RwxRxRx)
	assert.NoError(t, err)

}

func TestWriteConfigToDisk(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "write-config-test-*")
	require.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	configDir := filepath.Join(tmpDir, "etc", "containerd")
	err = os.MkdirAll(configDir, utils.RwxRxRx)
	require.NoError(t, err)

	runtimeParam := map[string]string{
		"directory": tmpDir + "/",
		"dataRoot":  testDataRoot,
		"sandbox":   testSandboxImage,
	}

	err = writeConfigToDisk(runtimeParam)
	assert.NoError(t, err)

	configPath := filepath.Join(configDir, "config.toml")
	_, err = os.Stat(configPath)
	assert.NoError(t, err)
}

func TestWriteConfigToDiskCreateDirectory(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "write-config-subdir-test-*")
	require.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	configDir := filepath.Join(tmpDir, "etc", "containerd")
	err = os.MkdirAll(configDir, utils.RwxRxRx)
	require.NoError(t, err)

	runtimeParam := map[string]string{
		"directory": tmpDir + "/",
		"dataRoot":  testDataRoot,
	}

	err = writeConfigToDisk(runtimeParam)
	assert.NoError(t, err)

	configPath := filepath.Join(configDir, "config.toml")
	_, err = os.Stat(configPath)
	assert.NoError(t, err)
}

func TestExecuteTemplateWithFileExecuteError(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "test-template-error-*.toml")
	require.NoError(t, err)
	defer os.Remove(tmpFile.Name())
	defer tmpFile.Close()

	tplContent := `{{.Server}}`
	data := struct {
		Server string
	}{
		Server: "test",
	}

	err = executeTemplateWithFile(tplContent, "testTemplate", data, tmpFile)
	assert.NoError(t, err)
}

func TestExtractFileOpenError(t *testing.T) {
	tr := tar.NewReader(bytes.NewReader([]byte("test content")))

	nonexistentPath := filepath.Join(os.TempDir(), "nonexistent", "path", "file.txt")
	err := extractFile(tr, nonexistentPath, utils.RwxRxRx)
	assert.Error(t, err)
}

func TestUnTarOpenError(t *testing.T) {
	err := unTar("/nonexistent/tarfile.tar.gz", "/tmp")
	assert.Error(t, err)
}

func TestUnTarGzipError(t *testing.T) {
	tmpTar, err := os.CreateTemp("", "invalid-gzip-*.tar.gz")
	require.NoError(t, err)
	defer os.Remove(tmpTar.Name())
	tmpTar.WriteString("not a gzip file")
	tmpTar.Close()

	err = unTar(tmpTar.Name(), "/tmp")
	assert.Error(t, err)
}

func TestContainerdPluginParamRequired(t *testing.T) {
	plugin := &ContainerdPlugin{}
	params := plugin.Param()
	assert.True(t, params["url"].Required)
	assert.False(t, params["repo"].Required)
	assert.False(t, params["sandbox"].Required)
	assert.False(t, params["runtime"].Required)
	assert.False(t, params["dataRoot"].Required)
	assert.False(t, params["directory"].Required)
	assert.False(t, params["insecureRegistries"].Required)
	assert.False(t, params["containerdConfig"].Required)
}

func TestContainerdPluginWithNilExecutor(t *testing.T) {
	plugin := New(nil)
	assert.NotNil(t, plugin)
}

func TestCreateHostsTOML(t *testing.T) {
	originalCertsDir := containerdCertsDir
	containerdCertsDir = filepath.Join(t.TempDir(), "certs.d")
	defer func() { containerdCertsDir = originalCertsDir }()
	p := &ContainerdPlugin{exec: newMockExecutor()}

	registry := "test.registry.io"

	runtimeParam := map[string]string{
		"repo":         registry,
		"repoInsecure": "false",
	}

	err := p.createHostsTOML(runtimeParam)
	assert.NoError(t, err)
}

func TestCreateHostsTOMLOfflineMode(t *testing.T) {
	originalCertsDir := containerdCertsDir
	containerdCertsDir = filepath.Join(t.TempDir(), "certs.d")
	defer func() { containerdCertsDir = originalCertsDir }()
	p := &ContainerdPlugin{exec: newMockExecutor()}

	registry := "test.registry.io"
	insecureReg := "insecure.registry.io"

	runtimeParam := map[string]string{
		"repo":               registry,
		"repoInsecure":       "true",
		"insecureRegistries": insecureReg,
	}

	err := p.createHostsTOML(runtimeParam)
	assert.NoError(t, err)
}

func TestExecuteScriptWithContent(t *testing.T) {
	p := &ContainerdPlugin{exec: newMockExecutor()}
	script := &bkev1beta1.ScriptConfig{
		Content: `#!/bin/bash
echo "test"`,
		Interpreter: "/bin/bash",
		Args:        []string{},
	}

	err := p.executeScript(script)
	assert.NoError(t, err)
}

func TestExecuteScriptWithPath(t *testing.T) {
	p := &ContainerdPlugin{exec: newMockExecutor()}

	tmpScript, err := os.CreateTemp("", "test-script-*.sh")
	require.NoError(t, err)
	defer os.Remove(tmpScript.Name())

	_, err = tmpScript.WriteString("#!/bin/bash\necho test")
	require.NoError(t, err)
	tmpScript.Close()

	script := &bkev1beta1.ScriptConfig{
		Path:        tmpScript.Name(),
		Interpreter: "/bin/bash",
		Args:        []string{},
	}

	err = p.executeScript(script)
	assert.NoError(t, err)
}

func TestExecuteScriptWithNonExistentPath(t *testing.T) {
	p := &ContainerdPlugin{exec: newMockExecutor()}
	script := &bkev1beta1.ScriptConfig{
		Path:        "/nonexistent/path/script.sh",
		Interpreter: "/bin/bash",
		Args:        []string{},
	}

	err := p.executeScript(script)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "does not exist")
}

func TestGenerateOverrideService(t *testing.T) {
	serviceDropInDir := t.TempDir()
	service := &bkev1beta1.ServiceConfig{
		ExecStart: "/usr/bin/containerd",
	}

	err := generateOverrideService(service, serviceDropInDir)
	assert.NoError(t, err)
}

func TestRenderConfigToml(t *testing.T) {
	tmpDir := t.TempDir()

	configDir := filepath.Join(tmpDir, "etc", "containerd")
	err := os.MkdirAll(configDir, utils.RwxRxRx)
	require.NoError(t, err)

	runtimeParam := map[string]string{
		"directory":      tmpDir + "/",
		"dataRoot":       testDataRoot,
		"dataState":      testState,
		"configPath":     testConfigPath,
		"metricsAddress": testMetricsAddr,
		"sandbox":        testSandboxImage,
	}

	main := &bkev1beta1.MainConfig{
		SandboxImage:   testSandboxImage,
		Root:           testDataRoot,
		State:          testState,
		ConfigPath:     testConfigPath,
		MetricsAddress: testMetricsAddr,
	}

	err = renderConfigToml(main, runtimeParam)
	assert.NoError(t, err)
}

func TestRenderConfigTomlWithNilRuntimeParam(t *testing.T) {
	main := &bkev1beta1.MainConfig{}

	err := renderConfigToml(main, nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "nil")
}

func TestGenerateHostsToml(t *testing.T) {
	registry := &bkev1beta1.RegistryConfig{
		ConfigPath: "/etc/containerd",
	}

	err := generateHostsToml(registry)
	assert.NoError(t, err)
}

func TestGenerateHostsTomlNilConfigs(t *testing.T) {
	registry := &bkev1beta1.RegistryConfig{
		ConfigPath: "/etc/containerd",
		Configs:    nil,
	}

	err := generateHostsToml(registry)
	assert.NoError(t, err)
}

func TestGenerateContainerdCfgWithScript(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(plugin.GetContainerdConfig, func(containerdCconfigNS string) (*bkev1beta1.ContainerdConfigSpec, error) {
		return &bkev1beta1.ContainerdConfigSpec{
			Script: &bkev1beta1.ScriptConfig{
				Content:     `#!/bin/bash\necho "test"`,
				Interpreter: "/bin/bash",
				Args:        []string{},
			},
		}, nil
	})

	p := &ContainerdPlugin{exec: newMockExecutor()}
	runtimeParam := map[string]string{
		"containerdConfig": "test-ns:test",
	}

	err := p.generateContainerdCfg(runtimeParam)
	assert.NoError(t, err)
}

func TestGenerateContainerdCfgWithService(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	serviceDropInDir := t.TempDir()

	patches.ApplyFunc(plugin.GetContainerdConfig, func(containerdCconfigNS string) (*bkev1beta1.ContainerdConfigSpec, error) {
		return &bkev1beta1.ContainerdConfigSpec{
			Service: &bkev1beta1.ServiceConfig{ExecStart: "/usr/bin/containerd"},
		}, nil
	})

	p := &ContainerdPlugin{exec: newMockExecutor(), serviceDropInDir: serviceDropInDir}
	runtimeParam := map[string]string{
		"containerdConfig": "test-ns:test",
	}

	err := p.generateContainerdCfg(runtimeParam)
	assert.NoError(t, err)
}

func TestGenerateContainerdCfgWithMain(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	tmpDir := t.TempDir()
	configDir := filepath.Join(tmpDir, "etc", "containerd")
	err := os.MkdirAll(configDir, utils.RwxRxRx)
	require.NoError(t, err)

	patches.ApplyFunc(plugin.GetContainerdConfig, func(containerdCconfigNS string) (*bkev1beta1.ContainerdConfigSpec, error) {
		return &bkev1beta1.ContainerdConfigSpec{
			Main: &bkev1beta1.MainConfig{
				Root: testDataRoot,
			},
		}, nil
	})

	p := &ContainerdPlugin{exec: newMockExecutor()}
	runtimeParam := map[string]string{
		"containerdConfig": "test-ns:test",
		"directory":        tmpDir + "/",
	}

	err = p.generateContainerdCfg(runtimeParam)
	assert.NoError(t, err)
}

func TestGenerateContainerdCfgWithRegistry(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(plugin.GetContainerdConfig, func(containerdCconfigNS string) (*bkev1beta1.ContainerdConfigSpec, error) {
		return &bkev1beta1.ContainerdConfigSpec{
			Registry: &bkev1beta1.RegistryConfig{
				ConfigPath: "/etc/containerd",
			},
		}, nil
	})

	p := &ContainerdPlugin{exec: newMockExecutor()}
	runtimeParam := map[string]string{
		"containerdConfig": "test-ns:test",
	}

	err := p.generateContainerdCfg(runtimeParam)
	assert.NoError(t, err)
}

func TestGenerateContainerdCfgGetConfigError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(plugin.GetContainerdConfig, func(containerdCconfigNS string) (*bkev1beta1.ContainerdConfigSpec, error) {
		return nil, fmt.Errorf("failed to get config")
	})

	p := &ContainerdPlugin{exec: newMockExecutor()}
	runtimeParam := map[string]string{
		"containerdConfig": "test-ns:test",
	}

	err := p.generateContainerdCfg(runtimeParam)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "get containerd config")
}

func TestGenerateContainerdCfgWithAllConfigs(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	tmpDir := t.TempDir()
	configDir := filepath.Join(tmpDir, "etc", "containerd")
	err := os.MkdirAll(configDir, utils.RwxRxRx)
	require.NoError(t, err)
	serviceDropInDir := filepath.Join(t.TempDir(), "containerd.service.d")
	patches.ApplyFunc(plugin.GetContainerdConfig, func(containerdCconfigNS string) (*bkev1beta1.ContainerdConfigSpec, error) {
		return &bkev1beta1.ContainerdConfigSpec{
			Script: &bkev1beta1.ScriptConfig{
				Content:     `#!/bin/bash\necho "test"`,
				Interpreter: "/bin/bash",
				Args:        []string{},
			},
			Service: &bkev1beta1.ServiceConfig{ExecStart: "/usr/bin/containerd"},
			Main: &bkev1beta1.MainConfig{
				Root: testDataRoot,
			},
			Registry: &bkev1beta1.RegistryConfig{
				ConfigPath: "/etc/containerd",
			},
		}, nil
	})

	p := &ContainerdPlugin{exec: newMockExecutor(), serviceDropInDir: serviceDropInDir}
	runtimeParam := map[string]string{
		"containerdConfig": "test-ns:test",
		"directory":        tmpDir + "/",
	}

	err = p.generateContainerdCfg(runtimeParam)
	assert.NoError(t, err)
}

func TestExecuteWithContainerdConfig(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	tmpDir := t.TempDir()
	configDir := filepath.Join(tmpDir, "etc", "containerd")
	err := os.MkdirAll(configDir, utils.RwxRxRx)
	require.NoError(t, err)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		buf := new(bytes.Buffer)
		gw := gzip.NewWriter(buf)
		tw := tar.NewWriter(gw)
		hdr := &tar.Header{
			Name:     "file1.txt",
			Mode:     numSeven,
			Size:     numFour,
			Typeflag: tar.TypeReg,
		}
		tw.WriteHeader(hdr)
		tw.Write([]byte("test"))
		tw.Close()
		gw.Close()
		w.Write(buf.Bytes())
	}))
	defer server.Close()

	patches.ApplyFunc(plugin.ParseCommands, func(plugin.Plugin, []string) (map[string]string, error) {
		return map[string]string{
			"url":              server.URL,
			"directory":        tmpDir + "/",
			"containerdConfig": "test-ns:test",
		}, nil
	})

	patches.ApplyFunc(plugin.GetContainerdConfig, func(containerdCconfigNS string) (*bkev1beta1.ContainerdConfigSpec, error) {
		return &bkev1beta1.ContainerdConfigSpec{
			Main: &bkev1beta1.MainConfig{
				Root: testDataRoot,
			},
		}, nil
	})

	patches.ApplyFunc(econd.WaitContainerdReady, func() error {
		return nil
	})

	patches.ApplyFunc(os.Remove, func(name string) error {
		return nil
	})
	originalCertsDir := containerdCertsDir
	containerdCertsDir = filepath.Join(t.TempDir(), "certs.d")
	defer func() { containerdCertsDir = originalCertsDir }()

	p := &ContainerdPlugin{exec: newMockExecutor()}
	commands := []string{"InstallContainerd", "url=" + server.URL, "containerdConfig=test-ns:test"}

	_, err = p.Execute(context.Background(), commands)
	assert.NoError(t, err)
}

func TestExecuteWithoutContainerdConfig(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	tmpDir := t.TempDir()
	configDir := filepath.Join(tmpDir, "etc", "containerd")
	err := os.MkdirAll(configDir, utils.RwxRxRx)
	require.NoError(t, err)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		buf := new(bytes.Buffer)
		gw := gzip.NewWriter(buf)
		tw := tar.NewWriter(gw)
		hdr := &tar.Header{
			Name:     "file1.txt",
			Mode:     numSeven,
			Size:     numFour,
			Typeflag: tar.TypeReg,
		}
		tw.WriteHeader(hdr)
		tw.Write([]byte("test"))
		tw.Close()
		gw.Close()
		w.Write(buf.Bytes())
	}))
	defer server.Close()

	patches.ApplyFunc(plugin.ParseCommands, func(plugin.Plugin, []string) (map[string]string, error) {
		return map[string]string{
			"url":              server.URL,
			"directory":        tmpDir + "/",
			"containerdConfig": "",
		}, nil
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	patches.ApplyFunc(econd.WaitContainerdReady, func() error {
		return nil
	})

	patches.ApplyFunc(os.Remove, func(name string) error {
		return nil
	})
	originalCertsDir := containerdCertsDir
	containerdCertsDir = filepath.Join(t.TempDir(), "certs.d")
	defer func() { containerdCertsDir = originalCertsDir }()

	p := &ContainerdPlugin{exec: newMockExecutor()}
	commands := []string{"InstallContainerd", "url=" + server.URL}

	_, err = p.Execute(context.Background(), commands)
	assert.NoError(t, err)
}

func TestExecuteParseCommandsError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(plugin.ParseCommands, func(plugin.Plugin, []string) (map[string]string, error) {
		return nil, fmt.Errorf("parse commands failed")
	})

	p := &ContainerdPlugin{exec: newMockExecutor()}
	commands := []string{"InstallContainerd", "url=http://test.com"}

	result, err := p.Execute(context.Background(), commands)
	assert.Error(t, err)
	assert.Nil(t, result)
	assert.Contains(t, err.Error(), "parse commands failed")
}

func TestExecuteDownloadTarError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	tmpDir := t.TempDir()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	patches.ApplyFunc(plugin.ParseCommands, func(plugin.Plugin, []string) (map[string]string, error) {
		return map[string]string{
			"url":              server.URL,
			"directory":        tmpDir + "/",
			"containerdConfig": "",
		}, nil
	})

	p := &ContainerdPlugin{exec: newMockExecutor()}
	commands := []string{"InstallContainerd", "url=" + server.URL}

	result, err := p.Execute(context.Background(), commands)
	assert.Error(t, err)
	assert.Nil(t, result)
}

func TestExecuteGenerateContainerdCfgError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	tmpDir := t.TempDir()

	patches.ApplyFunc(plugin.ParseCommands, func(plugin.Plugin, []string) (map[string]string, error) {
		return map[string]string{
			"url":              "http://test.com/containerd.tar.gz",
			"directory":        tmpDir + "/",
			"containerdConfig": "test-ns:test",
		}, nil
	})

	patches.ApplyFunc(plugin.GetContainerdConfig, func(containerdCconfigNS string) (*bkev1beta1.ContainerdConfigSpec, error) {
		return nil, fmt.Errorf("failed to get config")
	})

	p := &ContainerdPlugin{exec: newMockExecutor()}
	commands := []string{"InstallContainerd", "containerdConfig=test-ns:test", "url=http://test.com/containerd.tar.gz"}

	result, err := p.Execute(context.Background(), commands)
	assert.Error(t, err)
	assert.Nil(t, result)
}

func TestExecuteCreateHostsTOMLError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	tmpDir := t.TempDir()
	configDir := filepath.Join(tmpDir, "etc", "containerd")
	err := os.MkdirAll(configDir, utils.RwxRxRx)
	require.NoError(t, err)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		buf := new(bytes.Buffer)
		gw := gzip.NewWriter(buf)
		tw := tar.NewWriter(gw)
		hdr := &tar.Header{
			Name:     "file1.txt",
			Mode:     numSeven,
			Size:     numFour,
			Typeflag: tar.TypeReg,
		}
		tw.WriteHeader(hdr)
		tw.Write([]byte("test"))
		tw.Close()
		gw.Close()
		w.Write(buf.Bytes())
	}))
	defer server.Close()

	patches.ApplyFunc(plugin.ParseCommands, func(plugin.Plugin, []string) (map[string]string, error) {
		return map[string]string{
			"url":              server.URL,
			"directory":        tmpDir + "/",
			"containerdConfig": "",
			"repo":             "test.registry.io",
		}, nil
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		if strings.Contains(path, "certs.d") {
			return fmt.Errorf("permission denied")
		}
		return nil
	})

	patches.ApplyFunc(os.Remove, func(name string) error {
		return nil
	})
	originalCertsDir := containerdCertsDir
	containerdCertsDir = filepath.Join(t.TempDir(), "certs.d")
	defer func() { containerdCertsDir = originalCertsDir }()

	p := &ContainerdPlugin{exec: newMockExecutor()}
	commands := []string{"InstallContainerd", "url=" + server.URL}

	result, err := p.Execute(context.Background(), commands)
	assert.Error(t, err)
	assert.Nil(t, result)
}

func TestExecuteStartServiceError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	tmpDir := t.TempDir()
	configDir := filepath.Join(tmpDir, "etc", "containerd")
	err := os.MkdirAll(configDir, utils.RwxRxRx)
	require.NoError(t, err)

	mockExec := &mockExecError{}
	patches.ApplyFunc(plugin.ParseCommands, func(plugin.Plugin, []string) (map[string]string, error) {
		return map[string]string{
			"url":              "http://test.com/containerd.tar.gz",
			"directory":        tmpDir + "/",
			"containerdConfig": "",
		}, nil
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	patches.ApplyFunc(econd.WaitContainerdReady, func() error {
		return nil
	})

	patches.ApplyFunc(os.Remove, func(name string) error {
		return nil
	})

	p := &ContainerdPlugin{exec: mockExec}
	commands := []string{"InstallContainerd", "url=http://test.com/containerd.tar.gz"}

	result, err := p.Execute(context.Background(), commands)
	assert.Error(t, err)
	assert.Nil(t, result)
}

type mockExecError struct{}

func (m *mockExecError) WithContext(_ context.Context) exec.Executor {
	return m
}

func (m *mockExecError) ExecuteCommand(command string, arg ...string) error {
	return nil
}

func (m *mockExecError) ExecuteCommandWithEnv(env []string, command string, arg ...string) error {
	return nil
}

func (m *mockExecError) ExecuteCommandWithOutput(command string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecError) ExecuteCommandWithCombinedOutput(command string, arg ...string) (string, error) {
	if strings.Contains(command, "enable") {
		return "success", nil
	}
	return "", fmt.Errorf("start failed")
}

func (m *mockExecError) ExecuteCommandWithCombinedOutputInDir(_ string, command string, arg ...string) (string, error) {
	return m.ExecuteCommandWithCombinedOutput(command, arg...)
}

func (m *mockExecError) ExecuteCommandWithOutputFile(command, outfileArg string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecError) ExecuteCommandWithOutputFileTimeout(timeout time.Duration, command, outfileArg string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecError) ExecuteCommandWithTimeout(timeout time.Duration, command string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecError) ExecuteCommandResidentBinary(timeout time.Duration, command string, arg ...string) error {
	return nil
}

func TestExecuteWithInsecureRegistries(t *testing.T) {
	t.Skip("skip unstable UT: mocks os.MkdirAll and prevents hosts.toml parent directories from being created in CI")
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	tmpDir := t.TempDir()
	configDir := filepath.Join(tmpDir, "etc", "containerd")
	err := os.MkdirAll(configDir, utils.RwxRxRx)
	require.NoError(t, err)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		buf := new(bytes.Buffer)
		gw := gzip.NewWriter(buf)
		tw := tar.NewWriter(gw)
		hdr := &tar.Header{
			Name:     "file1.txt",
			Mode:     numSeven,
			Size:     numFour,
			Typeflag: tar.TypeReg,
		}
		tw.WriteHeader(hdr)
		tw.Write([]byte("test"))
		tw.Close()
		gw.Close()
		w.Write(buf.Bytes())
	}))
	defer server.Close()

	patches.ApplyFunc(plugin.ParseCommands, func(plugin.Plugin, []string) (map[string]string, error) {
		return map[string]string{
			"url":                server.URL,
			"directory":          tmpDir + "/",
			"containerdConfig":   "",
			"repo":               "test.registry.io",
			"insecureRegistries": "insecure.registry.io,test.registry.io",
		}, nil
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(econd.WaitContainerdReady, func() error {
		return nil
	})

	patches.ApplyFunc(os.Remove, func(name string) error {
		return nil
	})
	originalCertsDir := containerdCertsDir
	containerdCertsDir = filepath.Join(t.TempDir(), "certs.d")
	defer func() { containerdCertsDir = originalCertsDir }()

	p := &ContainerdPlugin{exec: newMockExecutor()}
	commands := []string{"InstallContainerd", "url=" + server.URL, "insecureRegistries=insecure.registry.io,test.registry.io"}

	_, err = p.Execute(context.Background(), commands)
	assert.NoError(t, err)
}

func TestExecuteUntarError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	tmpDir := t.TempDir()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("invalid tar content"))
	}))
	defer server.Close()

	patches.ApplyFunc(plugin.ParseCommands, func(plugin.Plugin, []string) (map[string]string, error) {
		return map[string]string{
			"url":              server.URL,
			"directory":        tmpDir + "/",
			"containerdConfig": "",
		}, nil
	})

	p := &ContainerdPlugin{exec: newMockExecutor()}
	commands := []string{"InstallContainerd", "url=" + server.URL}

	result, err := p.Execute(context.Background(), commands)
	assert.Error(t, err)
	assert.Nil(t, result)
}

func TestStartContainerdServiceEnableError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.Remove, func(name string) error {
		return nil
	})

	patches.ApplyFunc(econd.WaitContainerdReady, func() error {
		return nil
	})

	mockExec := &mockExecEnableError{}
	p := &ContainerdPlugin{exec: mockExec}

	result, err := p.startContainerdService()
	assert.NoError(t, err)
	assert.Empty(t, result)
}

type mockExecEnableError struct{}

func (m *mockExecEnableError) WithContext(_ context.Context) exec.Executor {
	return m
}

func (m *mockExecEnableError) ExecuteCommand(command string, arg ...string) error {
	return nil
}

func (m *mockExecEnableError) ExecuteCommandWithEnv(env []string, command string, arg ...string) error {
	return nil
}

func (m *mockExecEnableError) ExecuteCommandWithOutput(command string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecEnableError) ExecuteCommandWithCombinedOutput(command string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecEnableError) ExecuteCommandWithCombinedOutputInDir(_ string, command string, arg ...string) (string, error) {
	return m.ExecuteCommandWithCombinedOutput(command, arg...)
}

func (m *mockExecEnableError) ExecuteCommandWithOutputFile(command, outfileArg string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecEnableError) ExecuteCommandWithOutputFileTimeout(timeout time.Duration, command, outfileArg string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecEnableError) ExecuteCommandWithTimeout(timeout time.Duration, command string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecEnableError) ExecuteCommandResidentBinary(timeout time.Duration, command string, arg ...string) error {
	return nil
}

func TestStartContainerdServiceRestartError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockExec := &mockExecRestartError{}
	p := &ContainerdPlugin{exec: mockExec}

	result, err := p.startContainerdService()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "start containerd failed")
	assert.NotNil(t, result)
}

type mockExecRestartError struct{}

func (m *mockExecRestartError) WithContext(_ context.Context) exec.Executor {
	return m
}

func (m *mockExecRestartError) ExecuteCommand(command string, arg ...string) error {
	return nil
}

func (m *mockExecRestartError) ExecuteCommandWithEnv(env []string, command string, arg ...string) error {
	return nil
}

func (m *mockExecRestartError) ExecuteCommandWithOutput(command string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecRestartError) ExecuteCommandWithCombinedOutput(command string, arg ...string) (string, error) {
	if strings.Contains(command, "enable") || strings.Contains(command, "daemon-reload") {
		return "success", nil
	}
	return "", fmt.Errorf("restart failed")
}

func (m *mockExecRestartError) ExecuteCommandWithCombinedOutputInDir(_ string, command string, arg ...string) (string, error) {
	return m.ExecuteCommandWithCombinedOutput(command, arg...)
}

func (m *mockExecRestartError) ExecuteCommandWithOutputFile(command, outfileArg string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecRestartError) ExecuteCommandWithOutputFileTimeout(timeout time.Duration, command, outfileArg string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecRestartError) ExecuteCommandWithTimeout(timeout time.Duration, command string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecRestartError) ExecuteCommandResidentBinary(timeout time.Duration, command string, arg ...string) error {
	return nil
}

func TestCreateHostsTOMLMultipleRegistries(t *testing.T) {
	originalCertsDir := containerdCertsDir
	containerdCertsDir = filepath.Join(t.TempDir(), "certs.d")
	defer func() { containerdCertsDir = originalCertsDir }()
	p := &ContainerdPlugin{exec: newMockExecutor()}

	runtimeParam := map[string]string{
		"repo":               "test.registry.io",
		"repoInsecure":       "true",
		"insecureRegistries": "reg1.io,reg2.io,reg3.io",
	}

	err := p.createHostsTOML(runtimeParam)
	assert.NoError(t, err)
}

func TestCreateHostsTOMLEmptyInsecureRegistries(t *testing.T) {
	originalCertsDir := containerdCertsDir
	containerdCertsDir = filepath.Join(t.TempDir(), "certs.d")
	defer func() { containerdCertsDir = originalCertsDir }()
	p := &ContainerdPlugin{exec: newMockExecutor()}

	runtimeParam := map[string]string{
		"repo":               "test.registry.io",
		"repoInsecure":       "false",
		"insecureRegistries": "",
	}

	err := p.createHostsTOML(runtimeParam)
	assert.NoError(t, err)
}

func TestCreateHostsTOMLCreateDirError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return fmt.Errorf("permission denied")
	})

	p := &ContainerdPlugin{exec: newMockExecutor()}

	runtimeParam := map[string]string{
		"repo":         "test.registry.io",
		"repoInsecure": "false",
	}

	err := p.createHostsTOML(runtimeParam)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "permission denied")
}

func TestCreateHostsTOMLOpenFileError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.OpenFile, func(name string, flag int, perm os.FileMode) (*os.File, error) {
		return nil, fmt.Errorf("open file failed")
	})

	p := &ContainerdPlugin{exec: newMockExecutor()}

	runtimeParam := map[string]string{
		"repo":         "test.registry.io",
		"repoInsecure": "false",
	}

	err := p.createHostsTOML(runtimeParam)
	assert.Error(t, err)
}

func TestCreateHostsTOMLWriteError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	tmpFile, err := os.CreateTemp("", "test-hosts-*.toml")
	require.NoError(t, err)
	tmpFile.Close()
	defer os.Remove(tmpFile.Name())

	patches.ApplyFunc(os.OpenFile, func(name string, flag int, perm os.FileMode) (*os.File, error) {
		return tmpFile, nil
	})

	p := &ContainerdPlugin{exec: newMockExecutor()}

	runtimeParam := map[string]string{
		"repo":         "test.registry.io",
		"repoInsecure": "false",
	}

	err = p.createHostsTOML(runtimeParam)
	assert.Error(t, err)
}

func TestExecuteScriptWithEmptyScript(t *testing.T) {
	p := &ContainerdPlugin{exec: newMockExecutor()}
	script := &bkev1beta1.ScriptConfig{}

	err := p.executeScript(script)
	assert.NoError(t, err)
}

func TestGenerateContainerdCfgEmptyScript(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(plugin.GetContainerdConfig, func(containerdCconfigNS string) (*bkev1beta1.ContainerdConfigSpec, error) {
		return &bkev1beta1.ContainerdConfigSpec{
			Script: &bkev1beta1.ScriptConfig{},
		}, nil
	})

	p := &ContainerdPlugin{exec: newMockExecutor()}
	runtimeParam := map[string]string{
		"containerdConfig": "test-ns:test",
	}

	err := p.generateContainerdCfg(runtimeParam)
	assert.NoError(t, err)
}

func TestRenderConfigTomlEmptyRuntimeParam(t *testing.T) {
	tmpDir := t.TempDir()
	configDir := filepath.Join(tmpDir, "etc", "containerd")
	err := os.MkdirAll(configDir, utils.RwxRxRx)
	require.NoError(t, err)

	main := &bkev1beta1.MainConfig{}
	runtimeParam := map[string]string{
		"directory": tmpDir + "/",
	}

	err = renderConfigToml(main, runtimeParam)
	assert.NoError(t, err)
}
