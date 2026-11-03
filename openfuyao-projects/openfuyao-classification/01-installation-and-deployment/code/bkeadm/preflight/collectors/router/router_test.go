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

package router

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gopkg.openfuyao.cn/preflight/api"
)

type stubCollector struct {
	facts []api.Fact
	err   error
}

func (*stubCollector) ID() string { return "stub" }

func (c *stubCollector) Collect(context.Context, api.FactRequest) ([]api.Fact, error) {
	return c.facts, c.err
}

func TestCollectorRoutesFirstMatch(t *testing.T) {
	want := []api.Fact{{Key: "platform.os", Subject: "node-1", Available: true, Value: "linux"}}
	collector := New(
		Route{Match: func(api.Subject, string) bool { return true }, Collector: &stubCollector{facts: want}},
		Route{Match: func(api.Subject, string) bool { return true }, Collector: &stubCollector{err: errors.New("late")}},
	)

	facts, err := collector.Collect(context.Background(), api.FactRequest{
		Keys: []string{"platform.os"}, Subjects: []api.Subject{{ID: "node-1"}},
	})
	require.NoError(t, err)
	assert.Equal(t, want, facts)
	assert.Equal(t, "collector-router", collector.ID())
}

func TestCollectorRejectsNonAtomicRequest(t *testing.T) {
	collector := New()
	for _, request := range []api.FactRequest{
		{Keys: []string{"one", "two"}, Subjects: []api.Subject{{ID: "node-1"}}},
		{Keys: []string{"one"}, Subjects: []api.Subject{{ID: "node-1"}, {ID: "node-2"}}},
	} {
		facts, err := collector.Collect(context.Background(), request)
		assert.ErrorContains(t, err, "one subject and one fact key")
		assert.Nil(t, facts)
	}
}

func TestCollectorReturnsUnsupportedWhenNoRouteMatches(t *testing.T) {
	collector := New(
		Route{},
		Route{Match: func(api.Subject, string) bool { return false }, Collector: &stubCollector{}},
	)
	facts, err := collector.Collect(context.Background(), api.FactRequest{
		Keys: []string{"unknown"}, Subjects: []api.Subject{{ID: "node-1"}},
	})
	require.NoError(t, err)
	require.Len(t, facts, 1)
	assert.False(t, facts[0].Available)
	assert.Equal(t, api.FactErrUnsupported, facts[0].Error.Code)
}
