/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package score

import "testing"

func TestComputePrefixCacheScore(t *testing.T) {
	tests := []struct {
		name        string
		matchBlocks int
		totalBlocks int
		want        float64
	}{
		{name: "partial hit", matchBlocks: 3, totalBlocks: 4, want: 0.75},
		{name: "full hit", matchBlocks: 4, totalBlocks: 4, want: 1},
		{name: "no hit", matchBlocks: 0, totalBlocks: 4, want: 0},
		{name: "negative match clamps to zero", matchBlocks: -1, totalBlocks: 4, want: 0},
		{name: "match above total clamps to one", matchBlocks: 5, totalBlocks: 4, want: 1},
		{name: "zero total returns zero", matchBlocks: 3, totalBlocks: 0, want: 0},
		{name: "negative total returns zero", matchBlocks: 3, totalBlocks: -1, want: 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ComputePrefixCacheScore(tt.matchBlocks, tt.totalBlocks)
			if got != tt.want {
				t.Fatalf("expected %v, got %v", tt.want, got)
			}
		})
	}
}
