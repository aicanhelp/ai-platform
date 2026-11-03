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

// Package api defines the stable public data model and contracts for the
// openFuyao unified preflight check.
//
// The types in this package are intentionally free of BKE, Provider CRD,
// Cobra, HTTP and Kubernetes write-client dependencies. Business types may
// only appear in the openfuyao adapter; the SDK core consumes OperationPlan,
// Bundle, Fact and Report exclusively.
//
// The schema version is v1: backward-compatible optional fields may be
// added without bumping the version; removing, repurposing or tightening a
// default requires a new API version. The runtime-compiled Bundle model
// intentionally dropped the digest and source-lock fields of the earlier
// release-bundle prototype within v1; the SDK ships and is consumed in lock
// step with its bkeadm host, so no external v1 consumers exist.
package api

// SchemaVersion is the version of the preflight data contract (Bundle,
// Report, OperationPlan). It is the value carried in apiVersion
// fields as "preflight.openfuyao.io/v1".
const SchemaVersion = "v1"

// APIGroup is the apiVersion group prefix shared by all preflight objects.
const APIGroup = "preflight.openfuyao.io"

// Version returns the fully-qualified apiVersion string for preflight objects.
func Version() string { return APIGroup + "/" + SchemaVersion }

// PreflightBundleKind is the kind of a bundle object.
const PreflightBundleKind = "PreflightBundle"

// PreflightReportKind is the kind of a report object.
const PreflightReportKind = "PreflightReport"
