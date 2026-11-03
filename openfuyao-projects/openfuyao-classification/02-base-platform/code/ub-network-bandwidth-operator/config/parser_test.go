/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParseConfig(t *testing.T) {
	tmpDir := t.TempDir()

	validConfig := `bandwidths:
  - priority: 0
    bandwidth: 30
  - priority: 1
    bandwidth: 20
`
	validPath := filepath.Join(tmpDir, "valid.yaml")
	err := os.WriteFile(validPath, []byte(validConfig), 0644)
	assert.NoError(t, err)

	emptyPath := filepath.Join(tmpDir, "empty.yaml")
	err = os.WriteFile(emptyPath, []byte(""), 0644)
	assert.NoError(t, err)

	notExistPath := filepath.Join(tmpDir, "notexist.yaml")

	invalidPath := filepath.Join(tmpDir, "invalid.yaml")
	err = os.WriteFile(invalidPath, []byte("{{invalid yaml}}"), 0644)
	assert.NoError(t, err)

	fractionalBandwidthPath := filepath.Join(tmpDir, "fractional-bandwidth.yaml")
	err = os.WriteFile(fractionalBandwidthPath, []byte(`bandwidths:
  - priority: 0
    bandwidth: 25.5
`), 0644)
	assert.NoError(t, err)

	fractionalPriorityPath := filepath.Join(tmpDir, "fractional-priority.yaml")
	err = os.WriteFile(fractionalPriorityPath, []byte(`bandwidths:
  - priority: 0.5
    bandwidth: 25
`), 0644)
	assert.NoError(t, err)

	stringBandwidthPath := filepath.Join(tmpDir, "string-bandwidth.yaml")
	err = os.WriteFile(stringBandwidthPath, []byte(`bandwidths:
  - priority: 0
    bandwidth: "25"
`), 0644)
	assert.NoError(t, err)

	stringPriorityPath := filepath.Join(tmpDir, "string-priority.yaml")
	err = os.WriteFile(stringPriorityPath, []byte(`bandwidths:
  - priority: "0"
    bandwidth: 25
`), 0644)
	assert.NoError(t, err)

	tests := []struct {
		name    string
		path    string
		want    *BandwidthConfig
		wantErr bool
		wantNil bool
	}{
		{
			name: "valid config",
			path: validPath,
			want: &BandwidthConfig{
				Bandwidths: []BandwidthItem{
					{Priority: 0, Bandwidth: 30},
					{Priority: 1, Bandwidth: 20},
				},
			},
		},
		{
			name:    "empty file returns nil",
			path:    emptyPath,
			wantNil: true,
		},
		{
			name:    "non-existent file returns nil",
			path:    notExistPath,
			wantNil: true,
		},
		{
			name:    "invalid yaml returns error",
			path:    invalidPath,
			wantErr: true,
		},
		{
			name:    "fractional bandwidth returns error",
			path:    fractionalBandwidthPath,
			wantErr: true,
		},
		{
			name:    "fractional priority returns error",
			path:    fractionalPriorityPath,
			wantErr: true,
		},
		{
			name:    "string bandwidth returns error",
			path:    stringBandwidthPath,
			wantErr: true,
		},
		{
			name:    "string priority returns error",
			path:    stringPriorityPath,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseConfig(tt.path)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			if tt.wantNil {
				assert.Nil(t, got)
				return
			}
			assert.Equal(t, tt.want, got)
		})
	}
}
