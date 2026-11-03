/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package controller

import (
	"fmt"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

func int32OrDefault(v *int32, def int32) int32 {
	if v == nil || *v <= 0 {
		return def
	}
	return *v
}

// EngineWorkloadGroupSize returns dataParallelSize/dataParallelSizeLocal (defaults 1/1).
func EngineWorkloadGroupSize(w *infernexv1alpha1.InferenceEngineWorkloadSpec) (int32, error) {
	if w == nil {
		return 1, nil
	}
	dpSize := int32OrDefault(w.DataParallelSize, 1)
	dpLocal := int32OrDefault(w.DataParallelSizeLocal, 1)
	if dpSize%dpLocal != 0 {
		return 0, fmt.Errorf("dataParallelSize (%d) must be divisible by dataParallelSizeLocal (%d)", dpSize, dpLocal)
	}
	return dpSize / dpLocal, nil
}

// ValidateEngineWorkloadDataParallel validates DP sizing fields on a workload block.
func ValidateEngineWorkloadDataParallel(path string, w *infernexv1alpha1.InferenceEngineWorkloadSpec) error {
	if w == nil {
		return nil
	}
	if _, err := EngineWorkloadGroupSize(w); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}
