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

package common

import (
	"testing"
)

func TestXPUDevice_GetUsedVidCount(t *testing.T) {
	tests := []struct {
		name     string
		usedVids uint
		want     uint
	}{
		{
			name:     "no vids used",
			usedVids: 0b0000,
			want:     0,
		},
		{
			name:     "one vid used",
			usedVids: 0b0001,
			want:     1,
		},
		{
			name:     "multiple consecutive vids used",
			usedVids: 0b0111,
			want:     3,
		},
		{
			name:     "non-consecutive vids used",
			usedVids: 0b1010,
			want:     2,
		},
		{
			name:     "all bits used in byte",
			usedVids: 0b11111111,
			want:     8,
		},
		{
			name:     "sparse vids used",
			usedVids: 0b10001001,
			want:     3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := &XPUDevice{
				UsedVids: tt.usedVids,
			}
			if got := x.GetUsedVidCount(); got != tt.want {
				t.Errorf("GetUsedVidCount() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestXPUDevice_AllocateVid(t *testing.T) {
	tests := []struct {
		name            string
		initialUsedVids uint
		wantVid         uint
		wantUsedVids    uint
	}{
		{
			name:            "allocate first vid when none used",
			initialUsedVids: 0b0000,
			wantVid:         0,
			wantUsedVids:    0b0001,
		},
		{
			name:            "allocate second vid when first used",
			initialUsedVids: 0b0001,
			wantVid:         1,
			wantUsedVids:    0b0011,
		},
		{
			name:            "allocate vid in gap",
			initialUsedVids: 0b0101,
			wantVid:         1,
			wantUsedVids:    0b0111,
		},
		{
			name:            "allocate after consecutive vids",
			initialUsedVids: 0b0111,
			wantVid:         3,
			wantUsedVids:    0b1111,
		},
		{
			name:            "allocate first available in sparse pattern",
			initialUsedVids: 0b1001,
			wantVid:         1,
			wantUsedVids:    0b1011,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := &XPUDevice{
				UsedVids: tt.initialUsedVids,
			}
			gotVid := x.AllocateVid()
			if gotVid != tt.wantVid {
				t.Errorf("AllocateVid() vid = %v, want %v", gotVid, tt.wantVid)
			}
			if x.UsedVids != tt.wantUsedVids {
				t.Errorf("AllocateVid() UsedVids = %b, want %b", x.UsedVids, tt.wantUsedVids)
			}
		})
	}
}

func TestXPUDevice_AllocateVid_Sequential(t *testing.T) {
	x := &XPUDevice{
		UsedVids: 0,
	}

	expectedVids := []uint{0, 1, 2, 3, 4}
	for i, expected := range expectedVids {
		vid := x.AllocateVid()
		if vid != expected {
			t.Errorf("Allocation %d: got vid %v, want %v", i, vid, expected)
		}
	}

	expectedUsedVids := uint(0b11111)
	if x.UsedVids != expectedUsedVids {
		t.Errorf("Final UsedVids = %b, want %b", x.UsedVids, expectedUsedVids)
	}
}

func TestXPUDevice_MarkVidUsed(t *testing.T) {
	tests := []struct {
		name            string
		initialUsedVids uint
		vidToMark       uint
		wantOk          bool
		wantUsedVids    uint
	}{
		{
			name:            "mark unused vid as used",
			initialUsedVids: 0b0000,
			vidToMark:       0,
			wantOk:          false,
			wantUsedVids:    0b0001,
		},
		{
			name:            "mark already used vid",
			initialUsedVids: 0b0001,
			vidToMark:       0,
			wantOk:          true,
			wantUsedVids:    0b0001,
		},
		{
			name:            "mark middle vid as used",
			initialUsedVids: 0b0101,
			vidToMark:       1,
			wantOk:          false,
			wantUsedVids:    0b0111,
		},
		{
			name:            "mark high bit vid",
			initialUsedVids: 0b0001,
			vidToMark:       3,
			wantOk:          false,
			wantUsedVids:    0b1001,
		},
		{
			name:            "mark vid in sparse pattern",
			initialUsedVids: 0b1010,
			vidToMark:       2,
			wantOk:          false,
			wantUsedVids:    0b1110,
		},
		{
			name:            "re-mark already used vid in middle",
			initialUsedVids: 0b0110,
			vidToMark:       1,
			wantOk:          true,
			wantUsedVids:    0b0110,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := &XPUDevice{
				UsedVids: tt.initialUsedVids,
			}
			gotOk := x.MarkVidUsed(tt.vidToMark)
			if gotOk != tt.wantOk {
				t.Errorf("MarkVidUsed() ok = %v, want %v", gotOk, tt.wantOk)
			}
			if x.UsedVids != tt.wantUsedVids {
				t.Errorf("MarkVidUsed() UsedVids = %b, want %b", x.UsedVids, tt.wantUsedVids)
			}
		})
	}
}
