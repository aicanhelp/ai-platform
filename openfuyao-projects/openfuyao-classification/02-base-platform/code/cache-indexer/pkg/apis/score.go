/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package apis

// RequestKeyView is the per-request view produced by blockkey.Builder and
// consumed by the index plane MatchPrefix calls.
type RequestKeyView struct {
	ModelContext ModelContext
	BlockKeys    []BlockHash
	TotalBlocks  int
}

// MatchResult is the per-(server, plane) longest-continuous-prefix match.
type MatchResult struct {
	MatchedBlocks int
	MatchedTokens int
	Keys          []BlockHash
}

// ServerScore is the per-server score returned by score.Service.Compute.
// Order MUST follow the input serverIPs order.
type ServerScore struct {
	ServerIP        string
	L1HitRatio      float64
	L3HitRatio      float64
	L1MatchedBlocks int
	L3MatchedBlocks int
	TotalBlocks     int
}
