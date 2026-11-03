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

package prefixcache

type hitRateRequest struct {
	ServerIP []string           `json:"server_ip"`
	Body     hitRateRequestBody `json:"body"`
}

type hitRateRequestBody struct {
	TokenIDs  []uint32 `json:"token_ids"`
	CacheSalt string   `json:"cache_salt"`
	BlockSize int      `json:"block_size"`
}

type hitRateResponse struct {
	ServerScoreList []serverScore `json:"server_score_list"`
	Message         string        `json:"message"`
	Status          int           `json:"status"`
}

type serverScore struct {
	ServerIP   string  `json:"server_ip"`
	L1HitRatio float64 `json:"l1_hit_ratio"`
	L3HitRatio float64 `json:"l3_hit_ratio"`
}
