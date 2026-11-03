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

package kubernetes

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/preflight/api"
)

// fakeSnap is a stub ReadOnlySnapshot.
type fakeSnap struct {
	disc    api.Discovery
	discErr error
	obj     api.Object
	objErr  error
}

func (f *fakeSnap) Get(ctx context.Context, key api.ObjectKey) (api.Object, error) {
	return f.obj, f.objErr
}
func (f *fakeSnap) List(ctx context.Context, q api.ListQuery) ([]api.Object, error) {
	return nil, nil
}
func (f *fakeSnap) Discovery(ctx context.Context) (api.Discovery, error) {
	return f.disc, f.discErr
}

func TestCollectReachable(t *testing.T) {
	c := New(&fakeSnap{disc: api.Discovery{ServerVersion: "v1.28.0"}})
	facts, _ := c.Collect(context.Background(), api.FactRequest{Keys: []string{"kubernetes.api.reachable"}})
	assert.True(t, facts[0].Available)
	assert.Equal(t, true, facts[0].Value)
}

func TestCollectUnreachable(t *testing.T) {
	c := New(&fakeSnap{discErr: errors.New("connection refused")})
	facts, _ := c.Collect(context.Background(), api.FactRequest{Keys: []string{"kubernetes.api.reachable"}})
	assert.False(t, facts[0].Available)
	assert.Equal(t, api.FactErrUnreachable, facts[0].Error.Code)
}

func TestCollectCRDPresent(t *testing.T) {
	c := New(&fakeSnap{disc: api.Discovery{Resources: []api.DiscoveryResource{
		{Resource: "bkeclusters"}, {Resource: "bkenodes"},
	}}})
	facts, _ := c.Collect(context.Background(), api.FactRequest{Keys: []string{"crd.bkeclusters", "crd.missing"}})
	byKey := indexFacts(facts)
	assert.True(t, byKey["crd.bkeclusters"].Available)
	assert.Equal(t, true, byKey["crd.bkeclusters"].Value)
	assert.Equal(t, false, byKey["crd.missing"].Value)
}

func TestCollectObjectExists(t *testing.T) {
	c := New(&fakeSnap{obj: api.Object{Exists: true}})
	facts, _ := c.Collect(context.Background(), api.FactRequest{
		Keys: []string{"object.exists.bke.openfuyao.io/v1beta1|BKECluster|cluster-system|my-cluster"}})
	assert.True(t, facts[0].Available)
	assert.Equal(t, true, facts[0].Value)
}

func TestCollectObjectBadSpec(t *testing.T) {
	c := New(&fakeSnap{})
	facts, _ := c.Collect(context.Background(), api.FactRequest{
		Keys: []string{"object.exists.noSeparators"}})
	assert.False(t, facts[0].Available)
	assert.Equal(t, api.FactErrParseError, facts[0].Error.Code)
}

func TestCollectNilSnapshot(t *testing.T) {
	c := New(nil)
	facts, _ := c.Collect(context.Background(), api.FactRequest{Keys: []string{"kubernetes.api.reachable"}})
	assert.False(t, facts[0].Available)
	assert.Equal(t, "UNSUPPORTED", facts[0].Error.Code)
}

func TestCollectUnknownKey(t *testing.T) {
	c := New(&fakeSnap{})
	facts, _ := c.Collect(context.Background(), api.FactRequest{Keys: []string{"bogus.key"}})
	assert.False(t, facts[0].Available)
	assert.Equal(t, api.FactErrUnsupported, facts[0].Error.Code)
}

func TestCollectContextCanceled(t *testing.T) {
	c := New(&fakeSnap{})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	facts, _ := c.Collect(ctx, api.FactRequest{Keys: []string{"kubernetes.api.reachable"}})
	assert.False(t, facts[0].Available)
	assert.Equal(t, api.FactErrTimeout, facts[0].Error.Code)
}

func indexFacts(facts []api.Fact) map[string]api.Fact {
	m := make(map[string]api.Fact, len(facts))
	for _, f := range facts {
		m[f.Key] = f
	}
	return m
}

func TestCollectorIDAndDiscoveryFact(t *testing.T) {
c := New(&fakeSnap{disc: api.Discovery{ServerVersion: "v1.28.0"}})
assert.Equal(t, "kubernetes-readonly", c.ID())
facts, _ := c.Collect(context.Background(), api.FactRequest{Keys: []string{"kubernetes.discovery"}})
assert.True(t, facts[0].Available)
assert.Equal(t, "v1.28.0", facts[0].Value.(api.Discovery).ServerVersion)

failed := New(&fakeSnap{discErr: errors.New("refused")})
facts, _ = failed.Collect(context.Background(), api.FactRequest{Keys: []string{"kubernetes.discovery"}})
assert.False(t, facts[0].Available)
assert.Equal(t, api.FactErrUnreachable, facts[0].Error.Code)
}
