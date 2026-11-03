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

package predictiondata

const (
	// PodTypeAggregate is an exported constant used by Hermes Router.
	PodTypeAggregate = "aggregate"
	// PodTypePrefill is an exported constant used by Hermes Router.
	PodTypePrefill = "prefill"
	// PodTypeDecode is an exported constant used by Hermes Router.
	PodTypeDecode = "decode"

	// RoleLeader is an exported constant used by Hermes Router.
	RoleLeader = "leader"
	// RolePrefill is an exported constant used by Hermes Router.
	RolePrefill = "prefill"
	// RoleDecode is an exported constant used by Hermes Router.
	RoleDecode = "decode"
)

// Input is the normalized prediction feature row shared by online prediction,
// lifecycle persistence, and offline training.
type Input struct {
	// InputID stores the InputID value.
	InputID int `json:"inputId"`
	// PodType stores the PodType value.
	PodType string `json:"podType"`
	// InputTokenLength stores the InputTokenLength value.
	InputTokenLength int `json:"inputTokenLength"`
	// PrefixCacheScore stores the PrefixCacheScore value.
	PrefixCacheScore float64 `json:"prefixCacheScore"`
	// NumRequestWaiting stores the NumRequestWaiting value.
	NumRequestWaiting int `json:"numRequestWaiting"`
	// NumRequestRunning stores the NumRequestRunning value.
	NumRequestRunning int `json:"numRequestRunning"`
	// KVCacheUsage stores the KVCacheUsage value.
	KVCacheUsage float64 `json:"kvCacheUsage"`
	// InflightTokens stores the InflightTokens value.
	InflightTokens int `json:"inflightTokens"`
	// HBMUtilization stores the HBMUtilization value.
	HBMUtilization float64 `json:"hbmUtilization"`
	// AICoreUtilization stores the AICoreUtilization value.
	AICoreUtilization float64 `json:"aiCoreUtilization"`
	// BandwidthUtilization stores the BandwidthUtilization value.
	BandwidthUtilization float64 `json:"bandwidthUtilization"`
	// ServerIP stores the ServerIP value.
	ServerIP string `json:"serverIP"`
}

// Output is the sidecar prediction row correlated back to Input by InputID.
type Output struct {
	// InputID stores the InputID value.
	InputID int `json:"inputId"`
	// ServerIP stores the ServerIP value.
	ServerIP string `json:"serverIP"`
	// PredictedTTFTMillis stores the PredictedTTFTMillis value.
	PredictedTTFTMillis float64 `json:"predictedTTFTMillis"`
	// PredictedTPOTMillis stores the PredictedTPOTMillis value.
	PredictedTPOTMillis float64 `json:"predictedTPOTMillis"`
}
