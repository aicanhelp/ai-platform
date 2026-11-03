/******************************************************************
 * Copyright (c) 2026 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

// Package preflight is the SDK facade. A product host resolves its own input to
// a generic OperationPlan/RuleBinding result, then calls Check. The SDK loads
// the bundle, runs the engine and returns a Report.
//
// The facade is host-agnostic: it receives no cobra.Command, reads no env vars,
// writes no stdout/stderr, and never calls os.Exit. A Report and a Go error are
// the only outputs; the host maps Report.Overall to an exit code.
package preflight

import (
	"context"

	"gopkg.openfuyao.cn/preflight/api"
)

// CheckInput is the host-facing generic DTO. Resolved is produced by a product
// adapter before entering SDK core. Bundle is a neutral, in-memory SDK input
// compiled by the host from its external policy. The SDK never reads product
// configuration or policy files itself.
type CheckInput struct {
	Resolved api.ResolveResult
	Bundle   []byte
	Runtime  api.RuntimeOptions
	// Progress receives best-effort execution updates. It is called from the
	// SDK and is never persisted in the returned Report.
	Progress api.ProgressFunc
}

// Dependencies are host-supplied capabilities. Keeping them explicit makes
// Checker construction deterministic and keeps SDK core movable to a separate
// repository.
type Dependencies struct {
	Collector api.Collector
}

// Checker is the SDK facade. Check validates the resolved plan, loads and
// validates the bundle, runs the engine, and returns a Report. A non-nil error
// means an SDK/framework fault (INTERNAL_ERROR); input and bundle problems are
// carried in the Report (INPUT_ERROR / BUNDLE_ERROR).
type Checker interface {
	Check(ctx context.Context, in CheckInput) (api.Report, error)
}
