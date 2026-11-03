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

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

func TestInputJSONRetainsZeroValues(t *testing.T) {
	input := Input{
		InputID:          7,
		PodType:          PodTypeAggregate,
		InputTokenLength: 64,
		ServerIP:         "10.0.0.2",
	}

	raw, err := json.Marshal(input)
	if err != nil {
		t.Fatalf("marshal input: %v", err)
	}
	text := string(raw)
	for _, want := range []string{
		`"inputId":7`,
		`"prefixCacheScore":0`,
		`"numRequestWaiting":0`,
		`"kvCacheUsage":0`,
		`"serverIP":"10.0.0.2"`,
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("expected JSON to contain %s, got %s", want, text)
		}
	}
}

func TestCanonicalPodTypeValidation(t *testing.T) {
	for _, value := range []string{PodTypeAggregate, PodTypePrefill, PodTypeDecode} {
		if err := ValidatePodType(value); err != nil {
			t.Fatalf("expected %q to be valid: %v", value, err)
		}
	}
	if err := ValidatePodType("custom-prefill"); err == nil {
		t.Fatal("expected custom pod type to be invalid")
	}
}

func TestOutputValidation(t *testing.T) {
	if err := ValidateOutput(Output{InputID: 0, PredictedTTFTMillis: 1, PredictedTPOTMillis: 2}); err != nil {
		t.Fatalf("expected output to be valid: %v", err)
	}
	for _, output := range []Output{
		{InputID: -1, PredictedTTFTMillis: 1, PredictedTPOTMillis: 2},
		{InputID: 0, PredictedTTFTMillis: math.NaN(), PredictedTPOTMillis: 2},
		{InputID: 0, PredictedTTFTMillis: 1, PredictedTPOTMillis: math.Inf(1)},
		{InputID: 0, PredictedTTFTMillis: -1, PredictedTPOTMillis: 2},
	} {
		if err := ValidateOutput(output); err == nil {
			t.Fatalf("expected output %+v to be invalid", output)
		}
	}
}

func TestProtoRoundTrip(t *testing.T) {
	input := Input{
		InputID:              3,
		PodType:              PodTypePrefill,
		InputTokenLength:     12,
		PrefixCacheScore:     0.25,
		NumRequestWaiting:    1,
		NumRequestRunning:    2,
		KVCacheUsage:         0.5,
		InflightTokens:       99,
		HBMUtilization:       0.7,
		AICoreUtilization:    0.8,
		BandwidthUtilization: 0.9,
		ServerIP:             "10.0.0.3",
	}
	if got := InputFromProto(InputToProto(input)); got != input {
		t.Fatalf("input round trip mismatch: got %+v want %+v", got, input)
	}

	output := Output{InputID: 3, ServerIP: "10.0.0.3", PredictedTTFTMillis: 11, PredictedTPOTMillis: 0.6}
	if got := OutputFromProto(OutputToProto(output)); got != output {
		t.Fatalf("output round trip mismatch: got %+v want %+v", got, output)
	}
}
