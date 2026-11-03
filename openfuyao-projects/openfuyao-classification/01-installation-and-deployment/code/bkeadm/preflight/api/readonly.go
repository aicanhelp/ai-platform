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

package api

import (
	"context"
	"time"
)

// ObjectKey names a Kubernetes object for a read-only lookup.
type ObjectKey struct {
	APIVersion string
	Kind       string
	Namespace  string
	Name       string
}

// ListQuery is a read-only list request.
type ListQuery struct {
	APIVersion string
	Kind       string
	Namespace  string
}

// Discovery is the subset of Kubernetes discovery facts preflight needs.
type Discovery struct {
	ServerVersion string
	APIGroups     []string
	Resources     []DiscoveryResource
}

// DiscoveryResource is a discovered API resource (CRD presence, verbs).
type DiscoveryResource struct {
	Group      string
	Version    string
	Kind       string
	Resource   string
	Namespaced bool
	Verbs      []string
}

// ReadOnlySnapshot is the only Kubernetes surface the adapter and collectors
// may use. It is read by the thin entry once before the adapter runs; the
// adapter and core MUST NOT re-read the same dynamic source.
//
// The contract deliberately exposes only GET, LIST and Discovery. There are
// no Create, Update, Patch or Delete methods at the type level: the read-only
// boundary is enforced by the interface, not by convention. SelfSubjectAccess
// Review and server-side dry-run are separate, explicitly-allowlisted helpers.
type ReadOnlySnapshot interface {
	// Get returns an object by key. Secret material is never returned here:
	// only object identity, resourceVersion and a value digest are kept.
	Get(ctx context.Context, key ObjectKey) (Object, error)
	// List returns objects matching the query.
	List(ctx context.Context, query ListQuery) ([]Object, error)
	// Discovery returns API discovery facts.
	Discovery(ctx context.Context) (Discovery, error)
}

// Object is a desensitized Kubernetes object view. It carries identity and a
// value digest, never Secret data, tokens, private keys or kubeconfig content.
type Object struct {
	APIVersion      string
	Kind            string
	Namespace       string
	Name            string
	ResourceVersion string
	ValueDigest     Digest
	Exists          bool
}

// SSHHostKeyPolicy encodes the host-key trust behavior. Silent TOFU (trust on
// first use) is forbidden: an unknown host key without a trust anchor yields a
// WARN at most, never a silent accept.
type SSHHostKeyPolicy struct {
	// TrustAnchors holds known host-key fingerprints keyed by targetRef. When
	// present, the observed key MUST match; mismatch is a FAIL.
	TrustAnchors map[string]string
	// AllowUnverified, when false and no anchor is present for a target,
	// yields a WARN with the observed fingerprint rather than accepting.
	AllowUnverified bool
}

// SSHReadCommand is a structured, allowlisted read-only command. Collectors
// MUST NOT accept arbitrary shell strings; commands are passed as an argv
// array drawn from a fixed whitelist. Path and endpoint values are
// desensitized when recorded as evidence.
type SSHReadCommand struct {
	Name string   // binary name, must be in the whitelist
	Args []string // structured arguments, no shell interpolation
}

// SSHReadResult contains the bounded output and exit status of one read-only
// SSH command.
type SSHReadResult struct {
	Stdout   string
	Stderr   string
	ExitCode int
}

// ReadOnlyExecutor runs allowlisted read-only commands on a target over SSH.
// It has no write, install, enable, restart or cleanup methods.
type ReadOnlyExecutor interface {
	// RunRead executes a whitelisted read-only command and returns its
	// stdout (size-bounded), stderr and exit status. It MUST NOT accept a
	// raw shell string.
	RunRead(ctx context.Context, target Subject, cmd SSHReadCommand) (SSHReadResult, error)
}

// Collector returns facts for requested keys; it never decides the public
// status and never exposes a write action.
type Collector interface {
	ID() string
	Collect(ctx context.Context, req FactRequest) ([]Fact, error)
}

// FactRequest is a batched read request for facts over a set of subjects.
type FactRequest struct {
	Keys     []string
	Subjects []Subject
	Timeout  time.Duration
}
