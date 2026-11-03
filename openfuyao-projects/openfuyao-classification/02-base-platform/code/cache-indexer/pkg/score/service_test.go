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

package score

import (
	"context"
	"errors"
	"testing"

	"github.com/go-logr/logr/testr"

	"gitcode.com/openFuyao/cache-indexer/pkg/apis"
)

type fakeBuilder struct {
	keys []apis.BlockHash
	err  error
}

func (f *fakeBuilder) BuildFromRequest(_ apis.ModelContext, _ []int32) ([]apis.BlockHash, error) {
	return f.keys, f.err
}

type fakeL1 struct {
	byIP map[string]*apis.MatchResult
	err  map[string]error
}

func (f *fakeL1) UpsertStoredEvent(*apis.L1StoredEvent) error { return nil }
func (f *fakeL1) RemoveBlocks(*apis.L1RemovedEvent) error     { return nil }
func (f *fakeL1) ClearServer(apis.ModelContext, string) error { return nil }
func (f *fakeL1) MatchPrefix(_ apis.ModelContext, ip string, _ []apis.BlockHash) (*apis.MatchResult, error) {
	if e := f.err[ip]; e != nil {
		return nil, e
	}
	return f.byIP[ip], nil
}

func TestScoreServers_NilDeps_ReturnsZeros(t *testing.T) {
	s := New(testr.New(t), nil, nil, nil, nil)
	out, err := s.Compute(context.Background(), apis.ModelContext{BlockSize: 4},
		[]string{"a", "b"}, []int32{1, 2, 3, 4})
	if err != nil {
		t.Fatalf("err=%v", err)
	}
	if len(out) != 2 || out[0].ServerIP != "a" || out[1].ServerIP != "b" {
		t.Fatalf("ordering broken: %+v", out)
	}
	for _, s := range out {
		if s.L1HitRatio != 0 || s.TotalBlocks != 0 {
			t.Fatalf("expected zero score, got %+v", s)
		}
	}
}

func TestScoreServers_ShortStream_ReturnsZeros(t *testing.T) {
	b := &fakeBuilder{keys: nil}
	l1 := &fakeL1{}
	s := New(testr.New(t), b, l1, nil, nil)
	out, err := s.Compute(context.Background(), apis.ModelContext{BlockSize: 16},
		[]string{"a"}, []int32{1, 2})
	if err != nil {
		t.Fatalf("err=%v", err)
	}
	if out[0].TotalBlocks != 0 || out[0].L1HitRatio != 0 {
		t.Fatalf("got %+v", out[0])
	}
}

func TestScoreServers_RatioComputation(t *testing.T) {
	b := &fakeBuilder{keys: []apis.BlockHash{"k1", "k2", "k3", "k4"}}
	l1 := &fakeL1{byIP: map[string]*apis.MatchResult{
		"a": {MatchedBlocks: 4},
		"b": {MatchedBlocks: 2},
		"c": {MatchedBlocks: 0},
	}}
	s := New(testr.New(t), b, l1, nil, nil)
	out, err := s.Compute(context.Background(), apis.ModelContext{BlockSize: 4},
		[]string{"a", "b", "c"}, []int32{0, 0, 0, 0})
	if err != nil {
		t.Fatalf("err=%v", err)
	}
	if out[0].L1HitRatio != 1.0 || out[0].TotalBlocks != 4 {
		t.Fatalf("a: %+v", out[0])
	}
	if out[1].L1HitRatio != 0.5 {
		t.Fatalf("b: %+v", out[1])
	}
	if out[2].L1HitRatio != 0 {
		t.Fatalf("c: %+v", out[2])
	}
	for _, s := range out {
		if s.L3HitRatio != 0 {
			t.Fatalf("L3 must be 0 in S4, got %+v", s)
		}
	}
}

func TestScoreServers_PerServerErrorIsolated(t *testing.T) {
	b := &fakeBuilder{keys: []apis.BlockHash{"k1", "k2"}}
	l1 := &fakeL1{
		byIP: map[string]*apis.MatchResult{
			"good": {MatchedBlocks: 2},
		},
		err: map[string]error{"bad": errors.New("boom")},
	}
	s := New(testr.New(t), b, l1, nil, nil)
	out, err := s.Compute(context.Background(), apis.ModelContext{BlockSize: 1},
		[]string{"bad", "good"}, []int32{1, 2})
	if err != nil {
		t.Fatalf("aggregate err=%v", err)
	}
	if out[0].ServerIP != "bad" || out[0].L1HitRatio != 0 {
		t.Fatalf("bad not isolated: %+v", out[0])
	}
	if out[1].L1HitRatio != 1.0 {
		t.Fatalf("good poisoned: %+v", out[1])
	}
}

func TestScoreServers_BuilderErrorBubbles(t *testing.T) {
	b := &fakeBuilder{err: errors.New("seed missing")}
	l1 := &fakeL1{}
	s := New(testr.New(t), b, l1, nil, nil)
	_, err := s.Compute(context.Background(), apis.ModelContext{BlockSize: 4},
		[]string{"a"}, []int32{1, 2, 3, 4})
	if err == nil {
		t.Fatal("expected builder error to bubble")
	}
}

// --- L3 path coverage (S5) ---

type fakeL3 struct {
	byEP map[apis.TransportEndpoint]*apis.MatchResult
	err  map[apis.TransportEndpoint]error
}

func (f *fakeL3) ApplyReplicaSnapshot(apis.ModelContext, apis.TransportEndpoint, []apis.L3ReplicaRecord) error {
	return nil
}
func (f *fakeL3) RemoveReplicaKeys(apis.ModelContext, apis.TransportEndpoint, []apis.BlockHash) error {
	return nil
}
func (f *fakeL3) MatchPrefix(_ apis.ModelContext, ep apis.TransportEndpoint, _ []apis.BlockHash) (*apis.MatchResult, error) {
	if e := f.err[ep]; e != nil {
		return nil, e
	}
	return f.byEP[ep], nil
}

type fakeResolver struct {
	byIP map[string]apis.TransportEndpoint
}

func (r *fakeResolver) ResolveTransportEndpoint(ip string) (apis.TransportEndpoint, bool) {
	ep, ok := r.byIP[ip]
	return ep, ok
}

func TestScoreServers_L3PathHappyAndUnresolved(t *testing.T) {
	b := &fakeBuilder{keys: []apis.BlockHash{"k1", "k2"}}
	l1 := &fakeL1{byIP: map[string]*apis.MatchResult{
		"resolved-ip":   {MatchedBlocks: 2},
		"unresolved-ip": {MatchedBlocks: 1},
	}}
	l3 := &fakeL3{byEP: map[apis.TransportEndpoint]*apis.MatchResult{
		"ep-1:8000": {MatchedBlocks: 1},
	}}
	res := &fakeResolver{byIP: map[string]apis.TransportEndpoint{
		"resolved-ip": "ep-1:8000",
	}}
	s := New(testr.New(t), b, l1, l3, res)
	out, err := s.Compute(context.Background(), apis.ModelContext{BlockSize: 1},
		[]string{"resolved-ip", "unresolved-ip"}, []int32{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	if out[0].L1HitRatio != 1.0 || out[0].L3HitRatio != 0.5 {
		t.Errorf("resolved-ip got %+v", out[0])
	}
	if out[1].L1HitRatio != 0.5 || out[1].L3HitRatio != 0 {
		t.Errorf("unresolved-ip should have L3=0, got %+v", out[1])
	}
}

func TestScoreServers_L3ErrorIsolated(t *testing.T) {
	b := &fakeBuilder{keys: []apis.BlockHash{"k1"}}
	l1 := &fakeL1{byIP: map[string]*apis.MatchResult{"ip": {MatchedBlocks: 1}}}
	l3 := &fakeL3{err: map[apis.TransportEndpoint]error{"ep": errors.New("boom")}}
	res := &fakeResolver{byIP: map[string]apis.TransportEndpoint{"ip": "ep"}}
	s := New(testr.New(t), b, l1, l3, res)
	out, err := s.Compute(context.Background(), apis.ModelContext{BlockSize: 1},
		[]string{"ip"}, []int32{1})
	if err != nil {
		t.Fatal(err)
	}
	if out[0].L1HitRatio != 1.0 {
		t.Errorf("L1 poisoned by L3 error: %+v", out[0])
	}
	if out[0].L3HitRatio != 0 {
		t.Errorf("L3 error should yield 0, got %+v", out[0])
	}
}
