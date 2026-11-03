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

// ComputePrefixCacheScore normalizes the matched prefix-cache blocks into [0, 1].
func ComputePrefixCacheScore(matchBlocks, totalBlocks int) float64 {
	if totalBlocks <= 0 || matchBlocks <= 0 {
		return 0
	}
	if matchBlocks >= totalBlocks {
		return 1
	}
	return float64(matchBlocks) / float64(totalBlocks)
}
