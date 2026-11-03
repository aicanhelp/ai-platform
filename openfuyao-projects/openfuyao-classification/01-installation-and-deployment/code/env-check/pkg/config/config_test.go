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

package config

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLoadConfigFromFileFileNotFound(t *testing.T) {
	var config AppConfig
	err := LoadConfigFromFile("/nonexistent/file.json", &config)
	assert.NotNil(t, err)
}

func TestLoadConfigFromFileInvalidJSON(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "bad-config-*.json")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())
	defer tmpFile.Close()

	// 写入无效JSON
	_, err = tmpFile.WriteString("{invalid json")
	if err != nil {
		return
	}

	var config AppConfig
	err = LoadConfigFromFile(tmpFile.Name(), &config)
	assert.NotNil(t, err)
}

func TestValidateConfig(t *testing.T) {
	tests := []struct {
		name    string
		config  AppConfig
		wantErr bool
	}{
		{
			name: "valid query config",
			config: AppConfig{
				Mode:         ModeQuery,
				Paths:        []string{"/tmp"},
				OutputFormat: "text",
			},
			wantErr: false,
		},
		{
			name: "invalid mode",
			config: AppConfig{
				Mode:  OperationMode("invalid"),
				Paths: []string{"/tmp"},
			},
			wantErr: true,
		},
		{
			name: "no paths for query mode",
			config: AppConfig{
				Mode:  ModeQuery,
				Paths: []string{},
			},
			wantErr: true,
		},
		{
			name: "no program for programCheck mode",
			config: AppConfig{
				Mode:        ModeProgramCheck,
				ProgramList: []ProgramConfig{},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateConfig(&tt.config)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}
