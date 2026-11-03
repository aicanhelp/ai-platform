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
	"testing"

	"github.com/stretchr/testify/assert"
)

func validBandwidthConfig(items ...BandwidthItem) *BandwidthConfig {
	return &BandwidthConfig{Bandwidths: items}
}

func TestValidate(t *testing.T) {
	tests := []struct {
		name    string
		cfg     *BandwidthConfig
		wantErr bool
	}{
		{name: "nil_config", cfg: nil, wantErr: true},
		{name: "empty_bandwidths", cfg: validBandwidthConfig(), wantErr: true},
		{
			name:    "three_items",
			cfg:     validBandwidthConfig(BandwidthItem{Priority: 0, Bandwidth: 10}, BandwidthItem{Priority: 1, Bandwidth: 10}, BandwidthItem{Priority: 0, Bandwidth: 10}),
			wantErr: true,
		},
		{
			name:    "two_items_ok_10_20",
			cfg:     validBandwidthConfig(BandwidthItem{Priority: 0, Bandwidth: 10}, BandwidthItem{Priority: 1, Bandwidth: 20}),
			wantErr: false,
		},
		{
			name:    "two_items_ok_cross_priority",
			cfg:     validBandwidthConfig(BandwidthItem{Priority: 1, Bandwidth: 25}, BandwidthItem{Priority: 0, Bandwidth: 50}),
			wantErr: false,
		},
		{
			name:    "two_items_ok_25_50",
			cfg:     validBandwidthConfig(BandwidthItem{Priority: 0, Bandwidth: 25}, BandwidthItem{Priority: 1, Bandwidth: 50}),
			wantErr: false,
		},
		{
			name:    "two_items_duplicated_priority",
			cfg:     validBandwidthConfig(BandwidthItem{Priority: 0, Bandwidth: 10}, BandwidthItem{Priority: 0, Bandwidth: 20}),
			wantErr: true,
		},
		{
			name:    "two_items_both_gt_25",
			cfg:     validBandwidthConfig(BandwidthItem{Priority: 0, Bandwidth: 30}, BandwidthItem{Priority: 1, Bandwidth: 40}),
			wantErr: true,
		},
		{
			name:    "two_items_one_exceeds_50",
			cfg:     validBandwidthConfig(BandwidthItem{Priority: 0, Bandwidth: 10}, BandwidthItem{Priority: 1, Bandwidth: 51}),
			wantErr: true,
		},
		{
			name:    "single_prio0_25",
			cfg:     validBandwidthConfig(BandwidthItem{Priority: 0, Bandwidth: 25}),
			wantErr: false,
		},
		{
			name:    "single_prio1_1",
			cfg:     validBandwidthConfig(BandwidthItem{Priority: 1, Bandwidth: 1}),
			wantErr: false,
		},
		{
			name:    "single_prio2_invalid",
			cfg:     validBandwidthConfig(BandwidthItem{Priority: 2, Bandwidth: 25}),
			wantErr: true,
		},
		{
			name:    "single_prio1_26",
			cfg:     validBandwidthConfig(BandwidthItem{Priority: 1, Bandwidth: 26}),
			wantErr: true,
		},
		{
			name:    "single_bandwidth_zero",
			cfg:     validBandwidthConfig(BandwidthItem{Priority: 0, Bandwidth: 0}),
			wantErr: true,
		},
		{
			name:    "two_items_one_bw_zero",
			cfg:     validBandwidthConfig(BandwidthItem{Priority: 0, Bandwidth: 0}, BandwidthItem{Priority: 1, Bandwidth: 30}),
			wantErr: true,
		},
		{
			name:    "two_items_invalid_priority",
			cfg:     validBandwidthConfig(BandwidthItem{Priority: 3, Bandwidth: 10}, BandwidthItem{Priority: 0, Bandwidth: 20}),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := Validate(tt.cfg)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}
