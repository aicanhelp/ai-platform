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
	"fmt"
	"math"
)

// IsCanonicalPodType provides the exported IsCanonicalPodType operation.
func IsCanonicalPodType(value string) bool {
	switch value {
	case PodTypeAggregate, PodTypePrefill, PodTypeDecode:
		return true
	default:
		return false
	}
}

// ValidatePodType provides the exported ValidatePodType operation.
func ValidatePodType(value string) error {
	if IsCanonicalPodType(value) {
		return nil
	}
	return fmt.Errorf("podType must be one of %q, %q, or %q", PodTypeAggregate, PodTypePrefill, PodTypeDecode)
}

// ValidateOutput provides the exported ValidateOutput operation.
func ValidateOutput(output Output) error {
	if output.InputID < 0 {
		return fmt.Errorf("inputId must be >= 0")
	}
	if !finiteNonNegative(output.PredictedTTFTMillis) {
		return fmt.Errorf("predictedTTFTMillis must be finite and >= 0")
	}
	if !finiteNonNegative(output.PredictedTPOTMillis) {
		return fmt.Errorf("predictedTPOTMillis must be finite and >= 0")
	}
	return nil
}

func finiteNonNegative(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0) && value >= 0
}
