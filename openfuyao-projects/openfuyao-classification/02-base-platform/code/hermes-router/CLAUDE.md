# Hermes Router Working Notes

## Repository Role

Hermes Router is an out-of-tree plugin repository built on top of the GIE EPP framework.

It is not a fork of GIE and not an independent routing framework. Hermes Router plugins must evolve together with GIE framework changes.

When GIE plugin interfaces, package paths, or runtime contracts change, Hermes Router must update accordingly.

## Source Of Truth

For GIE framework interfaces and plugin development rules, use:

- `docs/superpowers/guides/epp-plugin-development-guide.md`

This path is exposed in Hermes Router through the shared `docs/superpowers` link and is maintained from the GIE source repository.

Do not duplicate GIE framework interface documentation into Hermes Router unless there is a Hermes-specific adaptation note that cannot live in the shared guide.

README remains the public project entry. This file is the internal working guide for agents and developers making code or dependency changes.

## Working Rules

Before changing Hermes Router plugins, read the shared EPP plugin development guide first.

Treat Hermes Router plugin code as GIE-coupled code. Do not change plugin behavior in a way that diverges from current GIE framework contracts without explicitly checking the corresponding GIE interfaces.

All newly added or modified source code files must keep the repository license header at the top of the file when the file type supports source-file comments:

```text
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
```

Do not add a second header to files that already contain the required repository license block. When touching an existing source file that is missing the header, add it as part of the same change unless the file format cannot represent it safely.

Keep the codebase in a clean-code style:

- prefer small focused functions over long multi-responsibility functions
- do not introduce magic numbers, magic strings, or other unexplained hard-coded domain values; extract named constants or configuration when the value carries business meaning or is reused
- do not leave overly long functions in place; split functions before they grow into multi-step procedures with mixed validation, mutation, logging, and branching responsibilities
- document domain-specific constants and scoring thresholds with concise rationale comments when they come from empirical tuning, regression models, or protocol constraints
- remove dead code, unused helpers, and stale TODOs instead of leaving them behind
- avoid redundant comments that merely restate the code; keep only comments that explain non-obvious intent or constraints
- keep source comments in English unless the text must match an external protocol, API, or user-facing payload
- keep package boundaries and responsibilities explicit; prefer small clean splits over piling unrelated behavior into one file or type
- avoid architecture drift by keeping plugin registration, shared helpers, scoring logic, filtering logic, and request preprocessing concerns separated
- deduplicate repeated implementation patterns into shared helpers when multiple plugins use the same initialization, nil-handling, or state-access logic
- keep shared contracts consistent across similar plugins and helpers; when multiple implementations handle the same nil, empty, or invalid-input path, converge on one clear repo-level contract unless there is a deliberate documented reason to diverge
- validate plugin parameters at factory or constructor boundaries before runtime use, especially for network addresses, ports, paths, and timeouts
- reuse shared clients or other reusable resources for repeated outbound calls; honor context cancellation and explicit timeouts for external I/O
- wrap errors with operation context when they cross I/O or abstraction boundaries, but do not add noisy wrapping where no extra context is gained
- keep test-only helpers out of production APIs unless they are genuine runtime utilities
- when a change increases complexity, refactor the touched area enough to keep the result readable and locally coherent

When reviewing or refactoring Hermes Router plugins:

- distinguish correctness, safety, and behavior regressions from clean-code or maintainability issues; do not present them as the same class of problem
- do not treat a plugin implementing multiple GIE interfaces as an automatic design defect; GIE allows one concrete plugin to implement multiple extension interfaces when a single stateful behavior spans those lifecycle hooks
- only flag a multi-interface plugin as a design problem when responsibilities are genuinely tangled, internal boundaries are unclear, or the combined type is becoming hard to test and evolve; prefer internal helper decomposition before forcing a split into multiple exported plugin types
- treat typed `fwkplugin.StateKey` constants as an acceptable baseline; focus review feedback on centralized read/write helpers and call-site consistency rather than on the mere existence of a shared key constant
- treat inconsistent contracts across similar implementations as a maintainability problem even when the current behavior is not yet a user-visible bug; prefer one explicit contract over per-file local conventions
- treat redundant nil-check simplification, interface assertion comments, and comment-language cleanup as clean-code maintenance unless they affect runtime behavior, API clarity, or safety

Do not treat a `go.mod` version bump as a completed GIE upgrade. A GIE upgrade is complete only after imports, interface usage, and minimal compile checks are aligned.

When upgrading GIE:

- update the module version
- run `go mod tidy`
- run at least one minimal compile check
- identify and fix import-path and interface breakpoints together

Do not copy shared `docs/superpowers` content into local duplicated files. Keep the shared guide canonical.

## Current Alignment Status

Target GIE baseline: `v1.5.0`

Current direct dependency in `go.mod`:

- `sigs.k8s.io/gateway-api-inference-extension v1.5.0`

Current repository status:

- dependency target updated to `v1.5.0`
- Hermes Router source is not yet fully API-aligned with GIE `v1.5.0`

Known first-layer upgrade breakpoints:

- removed package path: `pkg/epp/plugins`
- removed package path: `pkg/epp/scheduling/types`
- removed package path: `pkg/epp/scheduling/framework`
- removed package path: `pkg/epp/util/logging`
- old `pkg/epp/backend` import shape no longer matches `v1.5.0`

Expected replacement areas in GIE:

- plugin registration: `pkg/epp/framework/interface/plugin`
- scheduling interfaces and types: `pkg/epp/framework/interface/scheduling`
- request control interfaces: `pkg/epp/framework/interface/requestcontrol`
- logging: `pkg/common/observability/logging`

## Minimum Validation

After changing GIE dependency versions:

- run `go mod tidy`
- run `go build ./cmd/epp`

After changing Hermes Router plugin code:

- run `go build ./cmd/epp`
- if that passes, run `go build ./...`

If `go mod tidy` fails because GIE packages disappeared or moved, treat that as a migration breakpoint, not as a transient dependency error.