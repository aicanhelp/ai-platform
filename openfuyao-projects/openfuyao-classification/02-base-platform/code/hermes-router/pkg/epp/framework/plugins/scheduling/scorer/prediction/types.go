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

package prediction

import (
	"context"

	"hermes-router/pkg/epp/internal/predictiondata"
)

// PredictorClient isolates the scorer from the transport implementation.
// Tests inject fakes through this interface so batching, publishing, and
// scoring can be exercised without spinning up a real sidecar.
type PredictorClient interface {
	// Predict stores the Predict value.
	Predict(ctx context.Context, targetModel string, inputs []predictiondata.Input) ([]predictiondata.Output, error)
}

// PredictInput describes PredictInput data used by Hermes Router.
type PredictInput = predictiondata.Input

// PredictOutput describes PredictOutput data used by Hermes Router.
type PredictOutput = predictiondata.Output
