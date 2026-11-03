/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package containerd

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
)

func TestNewServiceDropInGenerator(t *testing.T) {
	tests := []struct {
		name       string
		configPath string
		expected   string
	}{
		{
			name:       "with custom path",
			configPath: "/custom/path",
			expected:   "/custom/path",
		},
		{
			name:       "with empty path",
			configPath: "",
			expected:   ServiceDropInDir,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			generator := NewServiceDropInGenerator(tt.configPath)
			assert.Equal(t, tt.expected, generator.ConfigPath)
		})
	}
}

func TestGenerateServiceDropInConfigNil(t *testing.T) {
	tempDir := t.TempDir()
	generator := NewServiceDropInGenerator(tempDir)

	err := generator.GenerateServiceDropIn(nil)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "service config is nil")
}

func TestGenerateServiceDropInBasicConfig(t *testing.T) {
	tempDir := t.TempDir()
	generator := NewServiceDropInGenerator(tempDir)

	config := &bkev1beta1.ServiceConfig{
		ExecStart: "/usr/bin/containerd --config /etc/containerd/config.toml",
		Slice:     "system.slice",
	}

	err := generator.GenerateServiceDropIn(config)
	require.NoError(t, err)

	// 验证文件是否创建
	expectedPath := filepath.Join(tempDir, DropInFileName)
	_, err = os.Stat(expectedPath)
	assert.NoError(t, err)

	// 验证文件内容
	content, err := os.ReadFile(expectedPath)
	require.NoError(t, err)

	contentStr := string(content)
	assert.Contains(t, contentStr, "ExecStart=/usr/bin/containerd --config /etc/containerd/config.toml")
	assert.Contains(t, contentStr, "Slice=system.slice")
}

func TestGenerateServiceDropInFullConfig(t *testing.T) {
	tempDir := t.TempDir()
	generator := NewServiceDropInGenerator(tempDir)

	config := &bkev1beta1.ServiceConfig{
		ExecStart:          "/usr/bin/containerd --config /etc/containerd/config.toml",
		Slice:              "system.slice",
		KillMode:           "process",
		Restart:            "always",
		RestartSec:         "5s",
		TimeoutStopSec:     "90s",
		StartLimitInterval: "10s",
		StartLimitBurst:    5,
		Logging: &bkev1beta1.ServiceLogging{
			StandardOutput:   "journal",
			StandardError:    "journal",
			SyslogIdentifier: "containerd",
			LogLevelMax:      "info",
		},
		CustomExtra: map[string]string{
			"LimitNOFILE": "infinity",
			"TasksMax":    "infinity",
		},
	}

	err := generator.GenerateServiceDropIn(config)
	require.NoError(t, err)

	content, err := os.ReadFile(filepath.Join(tempDir, DropInFileName))
	require.NoError(t, err)

	contentStr := string(content)

	// 验证所有配置项
	assert.Contains(t, contentStr, "ExecStart=/usr/bin/containerd --config /etc/containerd/config.toml")
	assert.Contains(t, contentStr, "Slice=system.slice")
	assert.Contains(t, contentStr, "KillMode=process")
	assert.Contains(t, contentStr, "Restart=always")
	assert.Contains(t, contentStr, "RestartSec=5s")
	assert.Contains(t, contentStr, "TimeoutStopSec=90s")
	assert.Contains(t, contentStr, "StartLimitBurst=5")
	assert.Contains(t, contentStr, "StandardOutput=journal")
	assert.Contains(t, contentStr, "StandardError=journal")
	assert.Contains(t, contentStr, "SyslogIdentifier=containerd")
	assert.Contains(t, contentStr, "LogLevelMax=info")
	assert.Contains(t, contentStr, "LimitNOFILE=infinity")
	assert.Contains(t, contentStr, "TasksMax=infinity")
}

func TestGenerateServiceDropInOverwriteExisting(t *testing.T) {
	tempDir := t.TempDir()
	generator := NewServiceDropInGenerator(tempDir)

	// 第一次生成
	config1 := &bkev1beta1.ServiceConfig{
		ExecStart: "/old/path/containerd",
		Slice:     "old.slice",
	}

	err := generator.GenerateServiceDropIn(config1)
	require.NoError(t, err)

	// 第二次生成，覆盖原有文件
	config2 := &bkev1beta1.ServiceConfig{
		ExecStart: "/new/path/containerd",
		Slice:     "new.slice",
	}

	err = generator.GenerateServiceDropIn(config2)
	require.NoError(t, err)

	// 验证内容已被更新
	content, err := os.ReadFile(filepath.Join(tempDir, DropInFileName))
	require.NoError(t, err)

	contentStr := string(content)
	assert.Contains(t, contentStr, "ExecStart=/new/path/containerd")
	assert.Contains(t, contentStr, "Slice=new.slice")
	assert.NotContains(t, contentStr, "/old/path/containerd")
}

func TestRemoveServiceDropInFileExists(t *testing.T) {
	tempDir := t.TempDir()
	generator := NewServiceDropInGenerator(tempDir)

	// 先创建文件
	config := &bkev1beta1.ServiceConfig{
		ExecStart: "/usr/bin/containerd",
	}
	err := generator.GenerateServiceDropIn(config)
	require.NoError(t, err)

	// 验证文件存在
	filePath := filepath.Join(tempDir, DropInFileName)
	_, err = os.Stat(filePath)
	assert.NoError(t, err)

	// 删除文件
	err = generator.RemoveServiceDropIn()
	require.NoError(t, err)

	// 验证文件已被删除
	_, err = os.Stat(filePath)
	assert.Error(t, err)
	assert.True(t, os.IsNotExist(err))
}
