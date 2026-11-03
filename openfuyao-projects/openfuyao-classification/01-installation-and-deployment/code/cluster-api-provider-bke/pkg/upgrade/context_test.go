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

package upgrade

import (
	"sync"
	"testing"
)

func TestVersionContext_NeedsUpgrade(t *testing.T) {
	vc := NewVersionContext()
	vc.SetCurrent(ComponentEtcd, "3.5.10")
	vc.SetTarget(ComponentEtcd, "3.5.12")

	if !vc.NeedsUpgrade(ComponentEtcd) {
		t.Fatal("expected etcd upgrade needed")
	}

	vc.SetCurrent(ComponentEtcd, "3.5.12")
	if vc.NeedsUpgrade(ComponentEtcd) {
		t.Fatal("expected no etcd upgrade when versions match")
	}
}

func TestVersionContext_HasTarget(t *testing.T) {
	vc := NewVersionContext()
	if vc.HasTarget(ComponentKubernetesMaster) {
		t.Fatal("expected no target before set")
	}
	vc.SetTarget(ComponentKubernetesMaster, "v1.29.0")
	if !vc.HasTarget(ComponentKubernetesMaster) {
		t.Fatal("expected target after set")
	}
}

func TestVersionContext_HasCurrentAndVersionAccessors(t *testing.T) {
	vc := NewVersionContext()
	if vc.HasCurrent(ComponentEtcd) {
		t.Fatal("expected no current before set")
	}
	if _, ok := vc.CurrentVersion(ComponentEtcd); ok {
		t.Fatal("expected CurrentVersion ok=false")
	}
	if _, ok := vc.TargetVersion(ComponentEtcd); ok {
		t.Fatal("expected TargetVersion ok=false")
	}

	vc.SetCurrent(ComponentEtcd, "3.5.10")
	vc.SetTarget(ComponentEtcd, "3.5.12")
	if !vc.HasCurrent(ComponentEtcd) {
		t.Fatal("expected HasCurrent after set")
	}
	cur, ok := vc.CurrentVersion(ComponentEtcd)
	if !ok || cur != "3.5.10" {
		t.Fatalf("CurrentVersion got %q ok=%v", cur, ok)
	}
	tgt, ok := vc.TargetVersion(ComponentEtcd)
	if !ok || tgt != "3.5.12" {
		t.Fatalf("TargetVersion got %q ok=%v", tgt, ok)
	}
}

func TestVersionContext_Decision(t *testing.T) {
	t.Run("nil VC still executes", func(t *testing.T) {
		var nilVC *VersionContext
		if Decide(nilVC, ComponentEtcd) != DecisionUpgrade {
			t.Fatal("nil VC should not block execution")
		}
		if !NeedsExecution(nil, ComponentEtcd) {
			t.Fatal("nil NeedsExecution should still execute")
		}
	})

	t.Run("missing target skips", func(t *testing.T) {
		vc := NewVersionContext()
		if Decide(vc, ComponentEtcd) != DecisionSkip {
			t.Fatal("missing target should skip")
		}
		if NeedsExecution(vc, ComponentEtcd) {
			t.Fatal("missing target should not execute")
		}
	})

	t.Run("matched versions skip", func(t *testing.T) {
		vc := NewVersionContext()
		vc.SetCurrent(ComponentEtcd, "3.5.12")
		vc.SetTarget(ComponentEtcd, "3.5.12")
		if Decide(vc, ComponentEtcd) != DecisionSkip {
			t.Fatal("expected skip when versions match")
		}
		if NeedsExecution(vc, ComponentEtcd) {
			t.Fatal("expected NeedsExecution false")
		}
	})

	t.Run("no current with target is upgrade", func(t *testing.T) {
		vc := NewVersionContext()
		vc.SetTarget(ComponentEtcd, "3.5.12")
		if Decide(vc, ComponentEtcd) != DecisionUpgrade {
			t.Fatal("expected upgrade when aligning empty current to target")
		}
	})

	t.Run("differing versions are upgrade", func(t *testing.T) {
		vc := NewVersionContext()
		vc.SetCurrent(ComponentEtcd, "3.5.10")
		vc.SetTarget(ComponentEtcd, "3.5.12")
		if Decide(vc, ComponentEtcd) != DecisionUpgrade {
			t.Fatal("expected upgrade")
		}
	})
}

func TestVersionContext_ConcurrentAccess(t *testing.T) {
	vc := NewVersionContext()
	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			name := ComponentEtcd
			vc.SetCurrent(name, "v1")
			vc.SetTarget(name, "v2")
			_ = vc.HasCurrent(name)
			_ = vc.HasTarget(name)
			_, _ = vc.CurrentVersion(name)
			_, _ = vc.TargetVersion(name)
			_ = vc.NeedsUpgrade(name)
			_ = Decide(vc, name)
			_ = NeedsExecution(vc, name)
			_ = vc.AnyTargetNeedsUpgrade()
			_ = vc.TargetNames()
			_ = i
		}(i)
	}
	wg.Wait()
}

func TestVersionContext_NilSafe(t *testing.T) {
	var vc *VersionContext
	vc.SetCurrent(ComponentEtcd, "1")
	if vc.GetCurrent(ComponentEtcd) != "" {
		t.Fatal("nil context should be no-op")
	}
	if vc.NeedsUpgrade(ComponentEtcd) {
		t.Fatal("nil context should not need upgrade")
	}
	if vc.HasCurrent(ComponentEtcd) || vc.HasTarget(ComponentEtcd) {
		t.Fatal("nil context accessors should be false")
	}
	if Decide(vc, ComponentEtcd) != DecisionUpgrade {
		t.Fatal("nil context decision should not block execution")
	}
	if !NeedsExecution(nil, ComponentEtcd) {
		t.Fatal("nil NeedsExecution should still execute")
	}
}
