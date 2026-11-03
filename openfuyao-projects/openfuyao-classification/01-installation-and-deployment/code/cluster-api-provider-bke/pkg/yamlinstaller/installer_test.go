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
	"strings"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"

	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/healthcheck"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/manifest"
)

type fakeStore struct {
	pkg *manifest.ComponentPackage
	err error
}

func (f fakeStore) GetComponentManifests(
	_ context.Context,
	_, _ string,
	_ manifest.TemplateContext,
) (*manifest.ComponentPackage, error) {
	if f.err != nil {
		return nil, f.err
	}
	return f.pkg, nil
}

type fakeApplier struct {
	applyCalls    int
	deleteCalls   int
	applyErr      error
	deleteErr     error
	lastStrategy  string
	lastPkgName   string
	lastManifests [][]byte
}

func (f *fakeApplier) ApplyComponent(_ context.Context, pkg *manifest.ComponentPackage) error {
	f.applyCalls++
	if pkg != nil {
		f.lastStrategy = pkg.ApplyStrategy
		f.lastPkgName = pkg.Name
		f.lastManifests = append([][]byte(nil), pkg.Manifests...)
	}
	return f.applyErr
}

func (f *fakeApplier) DeleteComponent(_ context.Context, pkg *manifest.ComponentPackage) error {
	f.deleteCalls++
	if pkg != nil {
		f.lastPkgName = pkg.Name
	}
	return f.deleteErr
}

func (f *fakeApplier) PruneResources(
	_ context.Context,
	_ map[string]string,
	_ string,
	_ [][]byte,
) error {
	return nil
}

type fakePruningApplier struct {
	fakeApplier
	pruneCalls int
	lastSel    map[string]string
	lastNS     string
	pruneErr   error
}

func (f *fakePruningApplier) PruneResources(
	_ context.Context,
	selector map[string]string,
	namespace string,
	_ [][]byte,
) error {
	f.pruneCalls++
	f.lastSel = selector
	f.lastNS = namespace
	return f.pruneErr
}

func sampleCV(name string, yaml *cvv1alpha1.YAMLSpec) *cvv1alpha1.ComponentVersion {
	return &cvv1alpha1.ComponentVersion{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: cvv1alpha1.ComponentVersionSpec{
			Name:    name,
			Type:    cvv1alpha1.ComponentTypeYAML,
			Version: "v1.0.0",
			YAML:    yaml,
		},
	}
}

func TestNewYamlInstaller(t *testing.T) {
	inst := NewYamlInstaller(YamlInstallerConfig{})
	if inst == nil {
		t.Fatal("expected non-nil installer")
	}
}

func TestApply_NilStore(t *testing.T) {
	inst := NewYamlInstaller(YamlInstallerConfig{Applier: &fakeApplier{}})
	err := inst.Apply(context.Background(), sampleCV("coredns", nil), &ApplyContext{})
	if err == nil || !containsAll(err.Error(), "coredns", "store is nil") {
		t.Fatalf("expected nil store error, got %v", err)
	}
}

func TestApply_NilApplier(t *testing.T) {
	inst := NewYamlInstaller(YamlInstallerConfig{Store: fakeStore{pkg: &manifest.ComponentPackage{Name: "coredns"}}})
	err := inst.Apply(context.Background(), sampleCV("coredns", nil), &ApplyContext{})
	if err == nil || !containsAll(err.Error(), "coredns", "applier is nil") {
		t.Fatalf("expected nil applier error, got %v", err)
	}
}

func TestApply_NilComponentVersion(t *testing.T) {
	inst := NewYamlInstaller(YamlInstallerConfig{
		Store:   fakeStore{pkg: &manifest.ComponentPackage{Name: "x"}},
		Applier: &fakeApplier{},
	})
	err := inst.Apply(context.Background(), nil, &ApplyContext{})
	if err == nil || !containsAll(err.Error(), "component version is nil") {
		t.Fatalf("expected nil cv error, got %v", err)
	}
}

func TestApply_GetManifestsError(t *testing.T) {
	applier := &fakeApplier{}
	inst := NewYamlInstaller(YamlInstallerConfig{
		Store:   fakeStore{err: errors.New("bundle missing")},
		Applier: applier,
	})
	err := inst.Apply(context.Background(), sampleCV("coredns", nil), &ApplyContext{})
	if err == nil || !containsAll(err.Error(), "coredns", "get manifests", "bundle missing") {
		t.Fatalf("expected get manifests error, got %v", err)
	}
	if applier.applyCalls != 0 {
		t.Fatalf("apply should not run after load failure, calls=%d", applier.applyCalls)
	}
}

func TestApply_SuccessWithoutYAML(t *testing.T) {
	applier := &fakeApplier{}
	inst := NewYamlInstaller(YamlInstallerConfig{
		Store: fakeStore{pkg: &manifest.ComponentPackage{
			Name:      "coredns",
			Version:   "v1.0.0",
			Manifests: [][]byte{[]byte("apiVersion: v1\nkind: ConfigMap\n")},
		}},
		Applier: applier,
	})
	if err := inst.Apply(context.Background(), sampleCV("coredns", nil), &ApplyContext{}); err != nil {
		t.Fatalf("apply: %v", err)
	}
	if applier.applyCalls != 1 {
		t.Fatalf("expected 1 apply call, got %d", applier.applyCalls)
	}
}

func TestApply_PruneFalseDoesNotPrune(t *testing.T) {
	applier := &fakePruningApplier{}
	inst := NewYamlInstaller(YamlInstallerConfig{
		Store:   fakeStore{pkg: &manifest.ComponentPackage{Name: "coredns"}},
		Applier: applier,
	})
	cv := sampleCV("coredns", &cvv1alpha1.YAMLSpec{Prune: false})
	if err := inst.Apply(context.Background(), cv, &ApplyContext{}); err != nil {
		t.Fatalf("apply: %v", err)
	}
	if applier.pruneCalls != 0 {
		t.Fatalf("expected no prune when prune=false, got %d", applier.pruneCalls)
	}
}

func TestApply_PruneTrueRequiresSelector(t *testing.T) {
	applier := &fakePruningApplier{}
	inst := NewYamlInstaller(YamlInstallerConfig{
		Store:   fakeStore{pkg: &manifest.ComponentPackage{Name: "coredns"}},
		Applier: applier,
	})
	cv := sampleCV("coredns", &cvv1alpha1.YAMLSpec{Prune: true})
	err := inst.Apply(context.Background(), cv, &ApplyContext{})
	if err == nil || !containsAll(err.Error(), "coredns", "prune", "pruneLabelSelector") {
		t.Fatalf("expected pruneLabelSelector error, got %v", err)
	}
	if applier.pruneCalls != 0 {
		t.Fatalf("prune must not run without selector, calls=%d", applier.pruneCalls)
	}
}

func TestApply_PruneTrueCallsPruner(t *testing.T) {
	applier := &fakePruningApplier{}
	inst := NewYamlInstaller(YamlInstallerConfig{
		Store:   fakeStore{pkg: &manifest.ComponentPackage{Name: "coredns"}},
		Applier: applier,
	})
	sel := map[string]string{"app.kubernetes.io/name": "coredns"}
	cv := sampleCV("coredns", &cvv1alpha1.YAMLSpec{
		Namespace:          "kube-system",
		Prune:              true,
		PruneLabelSelector: sel,
	})
	if err := inst.Apply(context.Background(), cv, &ApplyContext{}); err != nil {
		t.Fatalf("apply: %v", err)
	}
	if applier.pruneCalls != 1 {
		t.Fatalf("expected 1 prune call, got %d", applier.pruneCalls)
	}
	if applier.lastNS != "kube-system" {
		t.Fatalf("expected namespace kube-system, got %q", applier.lastNS)
	}
	if applier.lastSel["app.kubernetes.io/name"] != "coredns" {
		t.Fatalf("unexpected selector %#v", applier.lastSel)
	}
}

func TestApply_PruneErrorPropagates(t *testing.T) {
	applier := &fakePruningApplier{pruneErr: errors.New("delete stale failed")}
	inst := NewYamlInstaller(YamlInstallerConfig{
		Store:   fakeStore{pkg: &manifest.ComponentPackage{Name: "coredns"}},
		Applier: applier,
	})
	cv := sampleCV("coredns", &cvv1alpha1.YAMLSpec{
		Prune:              true,
		PruneLabelSelector: map[string]string{"app.kubernetes.io/name": "coredns"},
	})
	err := inst.Apply(context.Background(), cv, &ApplyContext{})
	if err == nil || !containsAll(err.Error(), "coredns", "prune", "delete stale failed") {
		t.Fatalf("expected prune error to propagate, got %v", err)
	}
	if applier.pruneCalls != 1 {
		t.Fatalf("expected 1 prune call, got %d", applier.pruneCalls)
	}
}

func TestApply_HealthCheckDisabledDoesNotRun(t *testing.T) {
	orig := healthCheckRun
	defer func() { healthCheckRun = orig }()
	calls := 0
	healthCheckRun = func(context.Context, kubernetes.Interface, healthcheck.Spec) error {
		calls++
		return nil
	}

	inst := NewYamlInstaller(YamlInstallerConfig{
		Store:   fakeStore{pkg: &manifest.ComponentPackage{Name: "coredns"}},
		Applier: &fakeApplier{},
	})
	cv := sampleCV("coredns", &cvv1alpha1.YAMLSpec{
		HealthCheck: &cvv1alpha1.HealthCheckSpec{Enabled: false},
	})
	if err := inst.Apply(context.Background(), cv, &ApplyContext{}); err != nil {
		t.Fatalf("apply: %v", err)
	}
	if calls != 0 {
		t.Fatalf("healthcheck.Run must not be called when disabled, calls=%d", calls)
	}
}

func TestApply_HealthCheckEnabledCallsRun(t *testing.T) {
	orig := healthCheckRun
	defer func() { healthCheckRun = orig }()
	var got healthcheck.Spec
	calls := 0
	healthCheckRun = func(_ context.Context, _ kubernetes.Interface, hc healthcheck.Spec) error {
		calls++
		got = hc
		return nil
	}

	timeout := 30 * time.Second
	interval := time.Second

	inst := NewYamlInstaller(YamlInstallerConfig{
		Store:   fakeStore{pkg: &manifest.ComponentPackage{Name: "coredns"}},
		Applier: &fakeApplier{},
	})
	cv := sampleCV("coredns", &cvv1alpha1.YAMLSpec{
		HealthCheck: &cvv1alpha1.HealthCheckSpec{
			Enabled:  true,
			Timeout:  timeout.String(),
			Interval: interval.String(),
			Checks: []cvv1alpha1.HealthCheckItemSpec{{
				Type:   string(healthcheck.CheckTypeCustom),
				Custom: &cvv1alpha1.CustomCheckSpec{Command: "true"},
			}},
		},
	})
	if err := inst.Apply(context.Background(), cv, &ApplyContext{
		TargetClient: fake.NewSimpleClientset(),
	}); err != nil {
		t.Fatalf("apply: %v", err)
	}
	if calls != 1 {
		t.Fatalf("expected 1 healthcheck.Run call, got %d", calls)
	}
	if !got.Enabled || got.Timeout != timeout || got.Interval != interval {
		t.Fatalf("unexpected converted spec: %#v", got)
	}
	if len(got.Checks) != 1 || got.Checks[0].Type != healthcheck.CheckTypeCustom {
		t.Fatalf("unexpected checks: %#v", got.Checks)
	}
	wantCmd := []string{"/bin/sh", "-c", "true"}
	if len(got.Checks[0].Command) != 3 ||
		got.Checks[0].Command[0] != wantCmd[0] ||
		got.Checks[0].Command[1] != wantCmd[1] ||
		got.Checks[0].Command[2] != wantCmd[2] {
		t.Fatalf("expected custom argv %v, got %#v", wantCmd, got.Checks[0].Command)
	}
}

func TestApply_HealthCheckFailureFailsApply(t *testing.T) {
	orig := healthCheckRun
	defer func() { healthCheckRun = orig }()
	healthCheckRun = func(context.Context, kubernetes.Interface, healthcheck.Spec) error {
		return errors.New("pods not ready")
	}

	inst := NewYamlInstaller(YamlInstallerConfig{
		Store:   fakeStore{pkg: &manifest.ComponentPackage{Name: "coredns"}},
		Applier: &fakeApplier{},
	})
	cv := sampleCV("coredns", &cvv1alpha1.YAMLSpec{
		HealthCheck: &cvv1alpha1.HealthCheckSpec{
			Enabled: true,
			Checks: []cvv1alpha1.HealthCheckItemSpec{{
				Type:   string(healthcheck.CheckTypeCustom),
				Custom: &cvv1alpha1.CustomCheckSpec{Command: "true"},
			}},
		},
	})
	err := inst.Apply(context.Background(), cv, &ApplyContext{
		TargetClient: fake.NewSimpleClientset(),
	})
	if err == nil || !containsAll(err.Error(), "coredns", "health check", "pods not ready") {
		t.Fatalf("expected health check failure, got %v", err)
	}
}

func TestApply_HealthCheckRequiresTargetClient(t *testing.T) {
	orig := healthCheckRun
	defer func() { healthCheckRun = orig }()
	calls := 0
	healthCheckRun = func(context.Context, kubernetes.Interface, healthcheck.Spec) error {
		calls++
		return nil
	}

	inst := NewYamlInstaller(YamlInstallerConfig{
		Store:   fakeStore{pkg: &manifest.ComponentPackage{Name: "coredns"}},
		Applier: &fakeApplier{},
	})
	cv := sampleCV("coredns", &cvv1alpha1.YAMLSpec{
		HealthCheck: &cvv1alpha1.HealthCheckSpec{Enabled: true},
	})
	err := inst.Apply(context.Background(), cv, &ApplyContext{})
	if err == nil || !containsAll(err.Error(), "coredns", "health check", "target client") {
		t.Fatalf("expected target client error, got %v", err)
	}
	if calls != 0 {
		t.Fatalf("Run must not be called without target client, calls=%d", calls)
	}
}

func TestApply_PassesApplyStrategy(t *testing.T) {
	applier := &fakeApplier{}
	inst := NewYamlInstaller(YamlInstallerConfig{
		Store:   fakeStore{pkg: &manifest.ComponentPackage{Name: "coredns"}},
		Applier: applier,
	})
	cv := sampleCV("coredns", &cvv1alpha1.YAMLSpec{
		ApplyStrategy: cvv1alpha1.ApplyStrategyReplace,
	})
	if err := inst.Apply(context.Background(), cv, &ApplyContext{}); err != nil {
		t.Fatalf("apply: %v", err)
	}
	if applier.lastStrategy != cvv1alpha1.ApplyStrategyReplace {
		t.Fatalf("expected strategy %q, got %q", cvv1alpha1.ApplyStrategyReplace, applier.lastStrategy)
	}
}

func TestUninstall_SuccessWithoutPrune(t *testing.T) {
	applier := &fakePruningApplier{}
	inst := NewYamlInstaller(YamlInstallerConfig{
		Store:   fakeStore{pkg: &manifest.ComponentPackage{Name: "coredns"}},
		Applier: applier,
	})
	cv := sampleCV("coredns", &cvv1alpha1.YAMLSpec{Prune: false})
	if err := inst.Uninstall(context.Background(), cv, &ApplyContext{}); err != nil {
		t.Fatalf("uninstall: %v", err)
	}
	if applier.deleteCalls != 1 {
		t.Fatalf("expected 1 delete call, got %d", applier.deleteCalls)
	}
	if applier.applyCalls != 0 {
		t.Fatalf("uninstall must not apply, applyCalls=%d", applier.applyCalls)
	}
	if applier.pruneCalls != 0 {
		t.Fatalf("expected no prune when prune=false, got %d", applier.pruneCalls)
	}
}

func TestUninstall_PruneTrueCallsPruner(t *testing.T) {
	applier := &fakePruningApplier{}
	inst := NewYamlInstaller(YamlInstallerConfig{
		Store:   fakeStore{pkg: &manifest.ComponentPackage{Name: "coredns"}},
		Applier: applier,
	})
	sel := map[string]string{"app.kubernetes.io/name": "coredns"}
	cv := sampleCV("coredns", &cvv1alpha1.YAMLSpec{
		Namespace:          "kube-system",
		Prune:              true,
		PruneLabelSelector: sel,
	})
	if err := inst.Uninstall(context.Background(), cv, &ApplyContext{}); err != nil {
		t.Fatalf("uninstall: %v", err)
	}
	if applier.deleteCalls != 1 {
		t.Fatalf("expected 1 delete call, got %d", applier.deleteCalls)
	}
	if applier.pruneCalls != 1 {
		t.Fatalf("expected 1 prune call, got %d", applier.pruneCalls)
	}
}

func TestUninstall_Idempotent(t *testing.T) {
	applier := &fakeApplier{}
	inst := NewYamlInstaller(YamlInstallerConfig{
		Store:   fakeStore{pkg: &manifest.ComponentPackage{Name: "coredns"}},
		Applier: applier,
	})
	cv := sampleCV("coredns", nil)
	if err := inst.Uninstall(context.Background(), cv, &ApplyContext{}); err != nil {
		t.Fatalf("first uninstall: %v", err)
	}
	if err := inst.Uninstall(context.Background(), cv, &ApplyContext{}); err != nil {
		t.Fatalf("second uninstall: %v", err)
	}
	if applier.deleteCalls != 2 {
		t.Fatalf("expected 2 delete calls, got %d", applier.deleteCalls)
	}
}

func TestUninstall_DeleteError(t *testing.T) {
	applier := &fakeApplier{deleteErr: errors.New("delete failed")}
	inst := NewYamlInstaller(YamlInstallerConfig{
		Store:   fakeStore{pkg: &manifest.ComponentPackage{Name: "coredns"}},
		Applier: applier,
	})
	err := inst.Uninstall(context.Background(), sampleCV("coredns", nil), &ApplyContext{})
	if err == nil || !containsAll(err.Error(), "coredns", "delete", "delete failed") {
		t.Fatalf("expected delete error, got %v", err)
	}
}

func containsAll(s string, parts ...string) bool {
	for _, p := range parts {
		if !strings.Contains(s, p) {
			return false
		}
	}
	return true
}
