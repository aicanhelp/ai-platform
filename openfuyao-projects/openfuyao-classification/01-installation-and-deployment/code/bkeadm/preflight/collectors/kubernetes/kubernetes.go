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

// Package kubernetes collects read-only facts from a management cluster.
//
// It consumes the ReadOnlySnapshot interface (GET/LIST/Discovery only). There
// are NO Create/Update/Patch/Delete methods at the type level: the read-only
// boundary is enforced by the interface. SelfSubjectAccessReview and server-
// side dry-run are explicitly-allowlisted no-side-effect requests.
package kubernetes

import (
	"context"
	"strings"

	"gopkg.openfuyao.cn/preflight/api"
)

const objectSpecParts = 4

// Snapshot is the read-only Kubernetes surface (aliased from api.ReadOnlySnapshot
// so the boundary is enforced by type, not convention).
type Snapshot = api.ReadOnlySnapshot

// Collector is the Kubernetes fact collector.
type Collector struct {
	snap Snapshot
}

// New returns a Collector backed by a read-only snapshot. The snapshot is read
// ONCE by the host before the adapter runs; the collector does not re-read
// the API.
func New(snap Snapshot) *Collector { return &Collector{snap: snap} }

// ID returns the collector identifier.
func (c *Collector) ID() string { return "kubernetes-readonly" }

// Collect returns facts for the requested keys. Facts are about the management
// cluster (CRD presence, API reachability, object existence, authorization).
func (c *Collector) Collect(ctx context.Context, req api.FactRequest) ([]api.Fact, error) {
	if c.snap == nil {
		return errorFacts(req, "UNSUPPORTED", "no kubernetes snapshot configured"), nil
	}
	out := make([]api.Fact, 0, len(req.Keys))
	for _, key := range req.Keys {
		if err := ctx.Err(); err != nil {
			out = append(out, api.Fact{Key: key, Available: false,
				Error: &api.FactError{Code: api.FactErrTimeout, Message: err.Error()}})
			continue
		}
		out = append(out, c.collectKey(ctx, key))
	}
	return out, nil
}

// collectKey maps a fact key to a read-only lookup.
func (c *Collector) collectKey(ctx context.Context, key string) api.Fact {
	switch {
	case key == "kubernetes.api.reachable":
		return c.reachableFact(ctx)
	case key == "kubernetes.discovery":
		return c.discoveryFact(ctx)
	case strings.HasPrefix(key, "crd."):
		return c.crdFact(ctx, strings.TrimPrefix(key, "crd."))
	case strings.HasPrefix(key, "object.exists."):
		// object.exists.<apiVersion>/<kind>.<namespace>/<name>
		spec := strings.TrimPrefix(key, "object.exists.")
		return c.objectExistsFact(ctx, spec)
	default:
		return api.Fact{Key: key, Available: false,
			Error: &api.FactError{Code: api.FactErrUnsupported, Message: "unknown fact key: " + key}}
	}
}

func (c *Collector) reachableFact(ctx context.Context) api.Fact {
	_, err := c.snap.Discovery(ctx)
	if err != nil {
		return discoveryFailure("kubernetes.api.reachable", err)
	}
	return discoverySuccess("kubernetes.api.reachable", true)
}

func (c *Collector) discoveryFact(ctx context.Context) api.Fact {
	d, err := c.snap.Discovery(ctx)
	if err != nil {
		return discoveryFailure("kubernetes.discovery", err)
	}
	return discoverySuccess("kubernetes.discovery", d)
}

func discoveryFailure(key string, err error) api.Fact {
	return api.Fact{Key: key, Available: false,
		Error: &api.FactError{Code: api.FactErrUnreachable, Message: err.Error()}}
}

func discoverySuccess(key string, value any) api.Fact {
	return api.Fact{Key: key, Available: true, Value: value,
		Evidence: []api.Evidence{{Kind: "api", Source: "kubernetes-readonly"}}}
}

// crdFact reports whether a CRD (by plural.resource) is served.
func (c *Collector) crdFact(ctx context.Context, resource string) api.Fact {
	d, err := c.snap.Discovery(ctx)
	if err != nil {
		return api.Fact{Key: "crd." + resource, Available: false,
			Error: &api.FactError{Code: api.FactErrUnreachable, Message: err.Error()}}
	}
	for _, r := range d.Resources {
		if r.Resource == resource {
			return api.Fact{Key: "crd." + resource, Available: true, Value: true,
				Evidence: []api.Evidence{{Kind: "api", Source: "kubernetes-readonly"}}}
		}
	}
	return api.Fact{Key: "crd." + resource, Available: true, Value: false,
		Evidence: []api.Evidence{{Kind: "api", Source: "kubernetes-readonly"}}}
}

// objectExistsFact reports whether an object exists.
//
// Key format: object.exists.<apiVersion>|<kind>|<namespace>|<name>
// where apiVersion keeps its own "/" (e.g. bke.openfuyao.io/v1beta1).
// The "|" separator avoids ambiguity with dots in apiVersion or names.
func (c *Collector) objectExistsFact(ctx context.Context, spec string) api.Fact {
	parts := strings.Split(spec, "|")
	if len(parts) != objectSpecParts {
		return api.Fact{Key: "object.exists." + spec, Available: false,
			Error: &api.FactError{
				Code:    api.FactErrParseError,
				Message: "bad object spec: expected apiVersion|kind|namespace|name",
			}}
	}
	apiVer, kind, ns, name := parts[0], parts[1], parts[2], parts[3]
	if apiVer == "" || kind == "" || name == "" {
		return api.Fact{Key: "object.exists." + spec, Available: false,
			Error: &api.FactError{Code: api.FactErrParseError, Message: "bad object spec: empty required field"}}
	}
	obj, err := c.snap.Get(ctx, api.ObjectKey{
		APIVersion: apiVer, Kind: kind, Namespace: ns, Name: name,
	})
	if err != nil {
		return api.Fact{Key: "object.exists." + spec, Available: false,
			Error: &api.FactError{Code: api.FactErrUnreachable, Message: err.Error()}}
	}
	return api.Fact{Key: "object.exists." + spec, Available: true, Value: obj.Exists,
		Evidence: []api.Evidence{{Kind: "api", Source: "kubernetes-readonly",
			Locator: ns + "/" + name, Observed: boolStr(obj.Exists)}}}
}

func boolStr(b bool) string {
	if b {
		return "true"
	}
	return "false"
}

func errorFacts(req api.FactRequest, code, msg string) []api.Fact {
	out := make([]api.Fact, len(req.Keys))
	for i, k := range req.Keys {
		out[i] = api.Fact{Key: k, Available: false, Error: &api.FactError{Code: code, Message: msg}}
	}
	return out
}
