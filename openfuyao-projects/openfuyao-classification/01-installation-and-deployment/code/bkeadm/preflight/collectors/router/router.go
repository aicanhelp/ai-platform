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

// Package router dispatches a concrete fact request to one read-only
// collector. It keeps product-free SDK collectors composable without giving
// the engine knowledge of local hosts, SSH, Kubernetes or artifacts.
package router

import (
	"context"
	"fmt"

	"gopkg.openfuyao.cn/preflight/api"
)

// Route selects a collector for a fact key and subject.
type Route struct {
	Match     func(api.Subject, string) bool
	Collector api.Collector
}

// Collector dispatches to the first matching route.
type Collector struct{ routes []Route }

// New creates a collector that evaluates routes in declaration order.
func New(routes ...Route) *Collector { return &Collector{routes: append([]Route(nil), routes...)} }

// ID returns the collector identifier.
func (c *Collector) ID() string { return "collector-router" }

// Collect dispatches one subject and fact key to the first matching route.
func (c *Collector) Collect(ctx context.Context, req api.FactRequest) ([]api.Fact, error) {
	if len(req.Keys) != 1 || len(req.Subjects) != 1 {
		return nil, fmt.Errorf("router requires one subject and one fact key")
	}
	for _, route := range c.routes {
		if route.Collector != nil && route.Match != nil && route.Match(req.Subjects[0], req.Keys[0]) {
			return route.Collector.Collect(ctx, req)
		}
	}
	return []api.Fact{{Key: req.Keys[0], Subject: req.Subjects[0].ID, Available: false,
		Error: &api.FactError{Code: api.FactErrUnsupported, Message: "no collector route for fact"}}}, nil
}

var _ api.Collector = (*Collector)(nil)
