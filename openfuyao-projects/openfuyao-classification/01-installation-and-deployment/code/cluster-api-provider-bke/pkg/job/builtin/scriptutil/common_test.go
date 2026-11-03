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

package scriptutil

import (
	"os"
	"strings"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/pkg/errors"
	"github.com/stretchr/testify/assert"

	bkenet "gopkg.openfuyao.cn/cluster-api-provider-bke/common/utils/net"
)

const (
	numZero    = 0
	numOne     = 1
	numTwo     = 2
	numTen     = 10
	numHundred = 100
)

func TestSanitizeFileName(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"test:file", "test_file"},
		{"path/to/script", "path_to_script"},
		{"script\\name", "script_name"},
		{"file with spaces", "file_with_spaces"},
		{"normal_name", "normal_name"},
		{":/\\ ", "____"},
	}

	for _, tt := range tests {
		result := SanitizeFileName(tt.input)
		assert.Equal(t, tt.expected, result)
	}
}

func TestSanitizeFileNameEmpty(t *testing.T) {
	result := SanitizeFileName("")
	assert.Equal(t, "", result)
}

func TestPreviewScript(t *testing.T) {
	script := "echo hello"

	result := PreviewScript(script, numTen)
	assert.Equal(t, script, result)
}

func TestPreviewScriptTruncated(t *testing.T) {
	script := "echo hello world"
	maxLen := numTen

	result := PreviewScript(script, maxLen)
	assert.Equal(t, "echo hello ... (truncated)", result)
}

func TestPreviewScriptNegativeMax(t *testing.T) {
	script := "echo hello"

	result := PreviewScript(script, -1)
	assert.Equal(t, "", result)
}

func TestPreviewScriptZeroMax(t *testing.T) {
	script := "echo hello"

	result := PreviewScript(script, numZero)
	assert.Equal(t, "", result)
}

func TestPreviewScriptExactLength(t *testing.T) {
	script := "hello"
	maxLen := 5

	result := PreviewScript(script, maxLen)
	assert.Equal(t, "hello", result)
}

func TestRenderScriptWithParams(t *testing.T) {
	scriptContent := `echo ${name} && echo ${value}`
	params := map[string]string{
		"name":  "test",
		"value": "123",
	}

	result := RenderScriptWithParams(scriptContent, params)
	assert.Equal(t, "echo test && echo 123", result)
}

func TestRenderScriptWithParamsMissing(t *testing.T) {
	scriptContent := `echo ${name} && echo ${missing}`
	params := map[string]string{
		"name": "test",
	}

	result := RenderScriptWithParams(scriptContent, params)
	assert.Equal(t, "echo test && echo ${missing}", result)
}

func TestRenderScriptWithParamsEmptyParams(t *testing.T) {
	scriptContent := `echo ${name}`

	result := RenderScriptWithParams(scriptContent, nil)
	assert.Equal(t, "echo ${name}", result)
}

func TestRenderScriptWithParamsNoParams(t *testing.T) {
	scriptContent := "echo hello world"

	result := RenderScriptWithParams(scriptContent, map[string]string{})
	assert.Equal(t, scriptContent, result)
}

func TestRenderScriptWithParamsMultipleSameParam(t *testing.T) {
	scriptContent := `echo ${var} ${var} ${var}`
	params := map[string]string{
		"var": "test",
	}

	result := RenderScriptWithParams(scriptContent, params)
	assert.Equal(t, "echo test test test", result)
}

func TestRenderScriptWithParamsUnderscoreParam(t *testing.T) {
	scriptContent := `echo ${_private}`
	params := map[string]string{
		"_private": "value",
	}

	result := RenderScriptWithParams(scriptContent, params)
	assert.Equal(t, "echo value", result)
}

func TestRenderScriptWithParamsNumericParam(t *testing.T) {
	scriptContent := `echo ${var123}`
	params := map[string]string{
		"var123": "value",
	}

	result := RenderScriptWithParams(scriptContent, params)
	assert.Equal(t, "echo value", result)
}

func TestWriteRenderedScriptToDisk(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	tmpDir, err := os.MkdirTemp("", "scriptutil-test")
	assert.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	scriptName := "test.sh"
	nodeIP := "192.168.1.100"
	renderedScript := "#!/bin/bash\necho hello"

	patches.ApplyFunc(bkenet.GetAllInterfaceIP, func() ([]string, error) {
		return []string{nodeIP}, nil
	})

	scriptPath, err := WriteRenderedScriptToDisk(tmpDir, scriptName, nodeIP, renderedScript)

	assert.NoError(t, err)
	assert.NotEmpty(t, scriptPath)

	content, err := os.ReadFile(scriptPath)
	assert.NoError(t, err)
	assert.Equal(t, renderedScript, string(content))
}

func TestWriteRenderedScriptToDiskMkdirError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	scriptName := "test.sh"
	nodeIP := "192.168.1.100"
	renderedScript := "#!/bin/bash\necho hello"

	patches.ApplyFunc(os.MkdirAll, func(string, os.FileMode) error {
		return errors.New("mkdir error")
	})

	scriptPath, err := WriteRenderedScriptToDisk("/invalid/path", scriptName, nodeIP, renderedScript)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "mkdir error")
	assert.Empty(t, scriptPath)
}

func TestWriteRenderedScriptToDiskWriteFileError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	tmpDir, err := os.MkdirTemp("", "scriptutil-test")
	assert.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	scriptName := "test.sh"
	nodeIP := "192.168.1.100"
	renderedScript := "#!/bin/bash\necho hello"

	patches.ApplyFunc(os.WriteFile, func(string, []byte, os.FileMode) error {
		return errors.New("write error")
	})

	scriptPath, err := WriteRenderedScriptToDisk(tmpDir, scriptName, nodeIP, renderedScript)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "write error")
	assert.Empty(t, scriptPath)
}

func TestWriteRenderedScriptToDiskWithExtension(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "scriptutil-test")
	assert.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	scriptName := "test.py"
	nodeIP := "192.168.1.100"
	renderedScript := "print('hello')"

	scriptPath, err := WriteRenderedScriptToDisk(tmpDir, scriptName, nodeIP, renderedScript)

	assert.NoError(t, err)
	assert.True(t, strings.HasSuffix(scriptPath, ".py"))
}

func TestWriteRenderedScriptToDiskSanitizeIP(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "scriptutil-test")
	assert.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	scriptName := "test.sh"
	nodeIP := "192.168.1.100:8080"
	renderedScript := "#!/bin/bash"

	scriptPath, err := WriteRenderedScriptToDisk(tmpDir, scriptName, nodeIP, renderedScript)

	assert.NoError(t, err)
	assert.Contains(t, scriptPath, "192.168.1.100_8080")
}

func TestGetCurrentNodeIPError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(bkenet.GetAllInterfaceIP, func() ([]string, error) {
		return nil, errors.New("get IP error")
	})

	ip, err := GetCurrentNodeIP()

	assert.Error(t, err)
	assert.Empty(t, ip)
}

func TestGetCurrentNodeIPNoValidIP(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(bkenet.GetAllInterfaceIP, func() ([]string, error) {
		return []string{"127.0.0.1"}, nil
	})

	ip, err := GetCurrentNodeIP()

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no valid node IP")
	assert.Empty(t, ip)
}

func TestGetCurrentNodeIPWithInvalidCIDR(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(bkenet.GetAllInterfaceIP, func() ([]string, error) {
		return []string{"192.168.1.100/24", "10.0.0.1"}, nil
	})

	ip, err := GetCurrentNodeIP()

	assert.NoError(t, err)
	assert.Equal(t, "192.168.1.100", ip)
}

func TestScriptConfig(t *testing.T) {
	config := ScriptConfig{
		ScriptName: "test.sh",
		Order:      1,
		Params: map[string]string{
			"key": "value",
		},
	}

	assert.Equal(t, "test.sh", config.ScriptName)
	assert.Equal(t, 1, config.Order)
	assert.Equal(t, "value", config.Params["key"])
}
