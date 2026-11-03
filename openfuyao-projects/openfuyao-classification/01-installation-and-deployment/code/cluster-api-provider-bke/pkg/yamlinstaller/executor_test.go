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

package yamlinstaller

import (
	"context"
	"errors"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/dagexec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/manifest"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/topology"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
)

type mapCVStore map[string]*cvv1alpha1.ComponentVersion

func (m mapCVStore) GetComponentVersion(_ context.Context, name, version string) (*cvv1alpha1.ComponentVersion, error) {
	cv, ok := m[name+"@"+version]
	if !ok {
		return nil, errors.New("not found")
	}
	return cv, nil
}

func yamlCV(name, version string) *cvv1alpha1.ComponentVersion {
	return &cvv1alpha1.ComponentVersion{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: cvv1alpha1.ComponentVersionSpec{
			Name:    name,
			Version: version,
			Type:    cvv1alpha1.ComponentTypeYAML,
		},
	}
}

func TestYamlComponentExecutor_GetComponentType(t *testing.T) {
	exec := &YamlComponentExecutor{}
	if got := exec.GetComponentType(); got != dagexec.ComponentTypeYAML {
		t.Fatalf("expected yaml, got %q", got)
	}
}

func TestYamlComponentExecutor_SkipWhenVersionsMatch(t *testing.T) {
	applier := &fakeApplier{}
	exec := &YamlComponentExecutor{
		Installer: NewYamlInstaller(YamlInstallerConfig{
			Store:   fakeStore{pkg: &manifest.ComponentPackage{Name: "coredns"}},
			Applier: applier,
		}),
		CVStore: mapCVStore{
			"coredns@v1.0.0": yamlCV("coredns", "v1.0.0"),
		},
	}
	vc := upgrade.NewVersionContext()
	vc.SetCurrent("coredns", "v1.0.0")
	vc.SetTarget("coredns", "v1.0.0")
	execCtx := dagexec.NewExecutionContext(dagexec.NewExecutionContextOptions{VersionContext: vc})

	if err := exec.ExecuteComponent(context.Background(), &topology.ComponentNode{
		Name: "coredns", Version: "v1.0.0",
	}, execCtx); err != nil {
		t.Fatalf("skip: %v", err)
	}
	if applier.applyCalls != 0 {
		t.Fatalf("Apply must not run on skip, calls=%d", applier.applyCalls)
	}
}

func TestYamlComponentExecutor_ApplySuccess(t *testing.T) {
	applier := &fakeApplier{}
	exec := &YamlComponentExecutor{
		Installer: NewYamlInstaller(YamlInstallerConfig{
			Store:   fakeStore{pkg: &manifest.ComponentPackage{Name: "coredns"}},
			Applier: applier,
		}),
		CVStore: mapCVStore{
			"coredns@v1.1.0": yamlCV("coredns", "v1.1.0"),
		},
	}
	vc := upgrade.NewVersionContext()
	vc.SetCurrent("coredns", "v1.0.0")
	vc.SetTarget("coredns", "v1.1.0")
	execCtx := dagexec.NewExecutionContext(dagexec.NewExecutionContextOptions{VersionContext: vc})

	if err := exec.ExecuteComponent(context.Background(), &topology.ComponentNode{
		Name: "coredns", Version: "v1.1.0",
	}, execCtx); err != nil {
		t.Fatalf("apply: %v", err)
	}
	if applier.applyCalls != 1 {
		t.Fatalf("expected 1 apply call, got %d", applier.applyCalls)
	}
}

func TestYamlComponentExecutor_ApplyFailurePropagates(t *testing.T) {
	applier := &fakeApplier{applyErr: errors.New("ssa conflict")}
	exec := &YamlComponentExecutor{
		Installer: NewYamlInstaller(YamlInstallerConfig{
			Store:   fakeStore{pkg: &manifest.ComponentPackage{Name: "coredns"}},
			Applier: applier,
		}),
		CVStore: mapCVStore{
			"coredns@v1.1.0": yamlCV("coredns", "v1.1.0"),
		},
	}
	vc := upgrade.NewVersionContext()
	vc.SetCurrent("coredns", "v1.0.0")
	vc.SetTarget("coredns", "v1.1.0")
	execCtx := dagexec.NewExecutionContext(dagexec.NewExecutionContextOptions{VersionContext: vc})

	err := exec.ExecuteComponent(context.Background(), &topology.ComponentNode{
		Name: "coredns", Version: "v1.1.0",
	}, execCtx)
	if err == nil || !containsAll(err.Error(), "coredns", "apply", "ssa conflict") {
		t.Fatalf("expected apply failure, got %v", err)
	}
}

func TestYamlComponentExecutor_NilInstaller(t *testing.T) {
	exec := &YamlComponentExecutor{CVStore: mapCVStore{}}
	err := exec.ExecuteComponent(context.Background(), &topology.ComponentNode{Name: "x"}, nil)
	if err == nil || !containsAll(err.Error(), "yaml installer is nil") {
		t.Fatalf("expected nil installer error, got %v", err)
	}
}

func TestYamlComponentExecutor_RejectsNonYAMLType(t *testing.T) {
	cv := yamlCV("coredns", "v1.0.0")
	cv.Spec.Type = cvv1alpha1.ComponentTypeHelm
	exec := &YamlComponentExecutor{
		Installer: NewYamlInstaller(YamlInstallerConfig{
			Store:   fakeStore{pkg: &manifest.ComponentPackage{Name: "coredns"}},
			Applier: &fakeApplier{},
		}),
		CVStore: mapCVStore{"coredns@v1.0.0": cv},
	}
	err := exec.ExecuteComponent(context.Background(), &topology.ComponentNode{
		Name: "coredns", Version: "v1.0.0",
	}, nil)
	if err == nil || !containsAll(err.Error(), "coredns", "not yaml") {
		t.Fatalf("expected type mismatch error, got %v", err)
	}
}

func TestYamlComponentExecutor_UninstallComponent(t *testing.T) {
	applier := &fakePruningApplier{}
	exec := &YamlComponentExecutor{
		Installer: NewYamlInstaller(YamlInstallerConfig{
			Store:   fakeStore{pkg: &manifest.ComponentPackage{Name: "coredns"}},
			Applier: applier,
		}),
		CVStore: mapCVStore{
			"coredns@v1.0.0": yamlCV("coredns", "v1.0.0"),
		},
	}
	if err := exec.UninstallComponent(context.Background(), &topology.ComponentNode{
		Name: "coredns", Version: "v1.0.0",
	}, nil); err != nil {
		t.Fatalf("uninstall: %v", err)
	}
	if applier.deleteCalls != 1 {
		t.Fatalf("expected 1 delete call, got %d", applier.deleteCalls)
	}
}

func TestYamlComponentExecutor_RegisterableWithScheduler(t *testing.T) {
	applier := &fakeApplier{}
	exec := &YamlComponentExecutor{
		Installer: NewYamlInstaller(YamlInstallerConfig{
			Store:   fakeStore{pkg: &manifest.ComponentPackage{Name: "coredns"}},
			Applier: applier,
		}),
		CVStore: mapCVStore{
			"coredns@v1.1.0": yamlCV("coredns", "v1.1.0"),
		},
	}
	sched := dagexec.NewScheduler(dagexec.Config{
		YamlExecutor: exec,
		CVStore: mapCVStore{
			"coredns@v1.1.0": yamlCV("coredns", "v1.1.0"),
		},
	})
	got, ok := sched.Registry.Get(dagexec.ComponentTypeYAML)
	if !ok || got != exec {
		t.Fatalf("expected yaml executor registered, ok=%v", ok)
	}
}
