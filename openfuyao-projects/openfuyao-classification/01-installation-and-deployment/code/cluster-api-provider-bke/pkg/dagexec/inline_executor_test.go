/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package dagexec

import (
	"context"
	"sync/atomic"
	"testing"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/topology"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
)

type recordingInlineRunner struct {
	calls   atomic.Int32
	handler string
	version string
}

func (r *recordingInlineRunner) Execute(
	_ context.Context,
	_, _ *bkev1beta1.BKECluster,
	handler, version string,
) error {
	r.calls.Add(1)
	r.handler = handler
	r.version = version
	return nil
}

func TestInlineComponentExecutor_ExecuteComponent(t *testing.T) {
	runner := &recordingInlineRunner{}
	exec := &InlineComponentExecutor{Runner: runner}
	if exec.GetComponentType() != ComponentTypeInline {
		t.Fatalf("unexpected type %q", exec.GetComponentType())
	}

	node := &topology.ComponentNode{
		Name: "etcd",
		Inline: &topology.InlineRef{
			Handler: "EnsureEtcdUpgrade",
			Version: "",
		},
	}
	execCtx := NewExecutionContext(NewExecutionContextOptions{})
	if err := exec.ExecuteComponent(context.Background(), node, execCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if runner.calls.Load() != 1 {
		t.Fatalf("expected runner called once")
	}
	if runner.handler != "EnsureEtcdUpgrade" || runner.version != defaultComponentVersion {
		t.Fatalf("handler/version mismatch: %q %q", runner.handler, runner.version)
	}
}

func TestInlineComponentExecutor_SkipsWhenVersionsMatch(t *testing.T) {
	runner := &recordingInlineRunner{}
	exec := &InlineComponentExecutor{Runner: runner}
	vc := upgrade.NewVersionContext()
	vc.SetCurrent("etcd", "v3.5.12")
	vc.SetTarget("etcd", "v3.5.12")
	node := &topology.ComponentNode{
		Name:   "etcd",
		Inline: &topology.InlineRef{Handler: "EnsureEtcdUpgrade", Version: "v3.5.12"},
	}
	execCtx := NewExecutionContext(NewExecutionContextOptions{VersionContext: vc})
	if err := exec.ExecuteComponent(context.Background(), node, execCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if runner.calls.Load() != 0 {
		t.Fatalf("expected skip when versions match, got %d calls", runner.calls.Load())
	}
}

func TestInlineComponentExecutor_RunsWhenNeedsUpgrade(t *testing.T) {
	runner := &recordingInlineRunner{}
	exec := &InlineComponentExecutor{Runner: runner}
	vc := upgrade.NewVersionContext()
	vc.SetCurrent("etcd", "v3.5.10")
	vc.SetTarget("etcd", "v3.5.12")
	node := &topology.ComponentNode{
		Name:   "etcd",
		Inline: &topology.InlineRef{Handler: "EnsureEtcdUpgrade", Version: "v3.5.12"},
	}
	execCtx := NewExecutionContext(NewExecutionContextOptions{VersionContext: vc})
	if err := exec.ExecuteComponent(context.Background(), node, execCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if runner.calls.Load() != 1 {
		t.Fatalf("expected runner called once on upgrade, got %d", runner.calls.Load())
	}
}

func TestInlineComponentExecutor_MissingInline(t *testing.T) {
	exec := &InlineComponentExecutor{Runner: &recordingInlineRunner{}}
	err := exec.ExecuteComponent(context.Background(), &topology.ComponentNode{Name: "x"}, nil)
	if err == nil {
		t.Fatal("expected error for missing inline ref")
	}
}
