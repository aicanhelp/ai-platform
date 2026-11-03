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

// Package predictiondata provides shared prediction request and response types.
package predictiondata

import pb "hermes-router/api/prediction/v1"

// InputToProto provides the exported InputToProto operation.
func InputToProto(input Input) *pb.PredictInput {
	return &pb.PredictInput{
		InputId:              int32(input.InputID),
		PodType:              input.PodType,
		InputTokenLength:     int32(input.InputTokenLength),
		PrefixCacheScore:     input.PrefixCacheScore,
		NumRequestWaiting:    int32(input.NumRequestWaiting),
		NumRequestRunning:    int32(input.NumRequestRunning),
		KvCacheUsage:         input.KVCacheUsage,
		InflightTokens:       int32(input.InflightTokens),
		HbmUtilization:       input.HBMUtilization,
		AiCoreUtilization:    input.AICoreUtilization,
		BandwidthUtilization: input.BandwidthUtilization,
		ServerIp:             input.ServerIP,
	}
}

// InputFromProto provides the exported InputFromProto operation.
func InputFromProto(input *pb.PredictInput) Input {
	if input == nil {
		return Input{}
	}
	return Input{
		InputID:              int(input.GetInputId()),
		PodType:              input.GetPodType(),
		InputTokenLength:     int(input.GetInputTokenLength()),
		PrefixCacheScore:     input.GetPrefixCacheScore(),
		NumRequestWaiting:    int(input.GetNumRequestWaiting()),
		NumRequestRunning:    int(input.GetNumRequestRunning()),
		KVCacheUsage:         input.GetKvCacheUsage(),
		InflightTokens:       int(input.GetInflightTokens()),
		HBMUtilization:       input.GetHbmUtilization(),
		AICoreUtilization:    input.GetAiCoreUtilization(),
		BandwidthUtilization: input.GetBandwidthUtilization(),
		ServerIP:             input.GetServerIp(),
	}
}

// OutputToProto provides the exported OutputToProto operation.
func OutputToProto(output Output) *pb.PredictOutput {
	return &pb.PredictOutput{
		InputId:             int32(output.InputID),
		ServerIp:            output.ServerIP,
		PredictedTtftMillis: output.PredictedTTFTMillis,
		PredictedTpotMillis: output.PredictedTPOTMillis,
	}
}

// OutputFromProto provides the exported OutputFromProto operation.
func OutputFromProto(output *pb.PredictOutput) Output {
	if output == nil {
		return Output{}
	}
	return Output{
		InputID:             int(output.GetInputId()),
		ServerIP:            output.GetServerIp(),
		PredictedTTFTMillis: output.GetPredictedTtftMillis(),
		PredictedTPOTMillis: output.GetPredictedTpotMillis(),
	}
}
