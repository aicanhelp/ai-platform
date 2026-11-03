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

package phaseutil

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
)

func TestUpgradeStrategy(t *testing.T) {
	assert.Equal(t, UpgradeStrategy("rolling"), UpgradePolicyRolling)
	assert.Equal(t, UpgradeStrategy("proportion"), UpgradePolicyDrain)
}

func TestDrainTime(t *testing.T) {
	assert.Equal(t, 20, DrainTime)
}

func TestWriter_Write(t *testing.T) {
	called := false
	w := writer{
		logFunc: func(reason, msg string, args ...interface{}) {
			called = true
			assert.Equal(t, constant.DrainNodeReason, reason)
		},
	}
	n, err := w.Write([]byte("test message\n"))
	assert.NoError(t, err)
	assert.Equal(t, 13, n)
	assert.True(t, called)
}

func TestGetPatchConfig(t *testing.T) {
	tests := []struct {
		name    string
		data    string
		wantErr bool
	}{
		{
			name: "valid yaml",
			data: `registry:
  imageAddress: "test.io"
openfuyaoVersion: "v1.0.0"`,
			wantErr: false,
		},
		{
			name:    "invalid yaml",
			data:    "invalid: [yaml",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg, err := GetPatchConfig(tt.data)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, cfg)
			}
		})
	}
}
