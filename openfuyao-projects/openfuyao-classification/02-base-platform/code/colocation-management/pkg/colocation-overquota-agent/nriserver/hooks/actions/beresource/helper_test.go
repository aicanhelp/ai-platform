/*
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
* Author: YuXiang Guo
* Date: 2025-08-18
 */

package beresource

import "testing"

// TestMilliCPUToShares 测试MilliCPUToShares函数
func TestMilliCPUToShares(t *testing.T) {
	tests := []struct {
		name     string
		milliCPU int64
		expected int64
	}{
		{
			name:     "zero milliCPU",
			milliCPU: 0,
			expected: MinShares,
		},
		{
			name:     "minimum milliCPU",
			milliCPU: 1,
			expected: MinShares,
		},
		{
			name:     "below threshold milliCPU",
			milliCPU: 1,
			expected: MinShares,
		},
		{
			name:     "exactly one CPU",
			milliCPU: 1000,
			expected: SharesPerCPU,
		},
		{
			name:     "half CPU",
			milliCPU: 500,
			expected: 512,
		},
		{
			name:     "two CPUs",
			milliCPU: 2000,
			expected: 2 * SharesPerCPU,
		},
		{
			name:     "quarter CPU",
			milliCPU: 250,
			expected: 256,
		},
		{
			name:     "very small milliCPU",
			milliCPU: 2,
			expected: MinShares,
		},
		{
			name:     "negative milliCPU",
			milliCPU: -100,
			expected: MinShares,
		},
		{
			name:     "maximum shares boundary",
			milliCPU: MaxShares * MilliCPUToCPU / SharesPerCPU,
			expected: MaxShares,
		},
		{
			name:     "exceed maximum shares",
			milliCPU: (MaxShares + 100) * MilliCPUToCPU / SharesPerCPU,
			expected: MaxShares,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := MilliCPUToShares(tt.milliCPU)
			if result != tt.expected {
				t.Errorf("MilliCPUToShares(%d) = %d, expected %d ",
					tt.milliCPU, result, tt.expected)
			}
		})
	}
}

// TestMilliCPUToQuota 测试MilliCPUToQuota函数
func TestMilliCPUToQuota(t *testing.T) {
	tests := []struct {
		name     string
		milliCPU int64
		expected int64
	}{
		{
			name:     "zero milliCPU",
			milliCPU: 0,
			expected: 0,
		},
		{
			name:     "minimum positive milliCPU",
			milliCPU: 1,
			expected: MinQuotaPeriod,
		},
		{
			name:     "exactly one CPU",
			milliCPU: 1000,
			expected: QuotaPeriod,
		},
		{
			name:     "half CPU",
			milliCPU: 500,
			expected: QuotaPeriod / 2,
		},
		{
			name:     "two CPUs",
			milliCPU: 2000,
			expected: 2 * QuotaPeriod,
		},
		{
			name:     "quarter CPU",
			milliCPU: 250,
			expected: QuotaPeriod / 4,
		},
		{
			name:     "very small milliCPU",
			milliCPU: 10,
			expected: MinQuotaPeriod,
		},
		{
			name:     "negative milliCPU",
			milliCPU: -100,
			expected: -1,
		},
		{
			name:     "below min quota",
			milliCPU: 5,
			expected: MinQuotaPeriod,
		},
		{
			name:     "large milliCPU",
			milliCPU: 100000,
			expected: 100 * QuotaPeriod,
		},
		{
			name:     "fractional milliCPU calculation",
			milliCPU: 333,
			expected: 33300,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := MilliCPUToQuota(tt.milliCPU)
			if result != tt.expected {
				t.Errorf("MilliCPUToQuota(%d) = %d, expected %d",
					tt.milliCPU, result, tt.expected)
			}
		})
	}
}

// TestEdgeCases 测试边界情况
func TestEdgeCases(t *testing.T) {
	t.Run("MilliCPUToShares edge values", func(t *testing.T) {

		minMilliCPU := (MinShares*MilliCPUToCPU + SharesPerCPU - 1) / SharesPerCPU

		result := MilliCPUToShares(int64(minMilliCPU))
		if result < MinShares {
			t.Errorf("Expected at least MinShares (%d), got %d for milliCPU %d",
				MinShares, result, minMilliCPU)
		}
	})

	t.Run("MilliCPUToQuota edge values", func(t *testing.T) {
		minMilliCPU := (MinQuotaPeriod*MilliCPUToCPU + QuotaPeriod - 1) / QuotaPeriod

		result := MilliCPUToQuota(int64(minMilliCPU))
		if result < MinQuotaPeriod && result != -1 {
			t.Errorf("Expected at least MinQuotaPeriod (%d) or -1, got %d for milliCPU %d",
				MinQuotaPeriod, result, minMilliCPU)
		}
	})
}
