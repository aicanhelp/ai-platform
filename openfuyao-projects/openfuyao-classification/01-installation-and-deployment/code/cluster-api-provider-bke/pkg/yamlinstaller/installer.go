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
	"fmt"

	"github.com/pkg/errors"
	"k8s.io/client-go/kubernetes"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/healthcheck"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/manifest"
)

// healthCheckRun is the Apply-time health probe entrypoint.
// Tests may override it to assert orchestration without re-testing probe logic.
var healthCheckRun = healthcheck.Run

// ApplyContext is the engine-facing subset of dagexec.ExecutionContext.
// YamlComponentExecutor adapts the full ExecutionContext into this type.
type ApplyContext struct {
	TemplateContext manifest.TemplateContext
	// TargetClient is used by health checks after Apply.
	TargetClient kubernetes.Interface
}

// YamlInstaller applies / uninstalls YAML components via manifest Store + Applier.
type YamlInstaller struct {
	store   manifest.Store
	applier manifest.Applier
	logger  *bkev1beta1.BKELogger
}

// YamlInstallerConfig configures NewYamlInstaller.
// Clientset is not injected here; HealthCheck uses ApplyContext.TargetClient.
type YamlInstallerConfig struct {
	Store   manifest.Store
	Applier manifest.Applier
	Logger  *bkev1beta1.BKELogger
}

// NewYamlInstaller constructs a YamlInstaller. Apply validates Store/Applier at call time.
func NewYamlInstaller(cfg YamlInstallerConfig) *YamlInstaller {
	return &YamlInstaller{
		store:   cfg.Store,
		applier: cfg.Applier,
		logger:  cfg.Logger,
	}
}

// Apply loads manifests, applies them, optionally prunes, then optionally health-checks.
// Health-check execution is gated by YAML.HealthCheck.Enabled.
func (yi *YamlInstaller) Apply(ctx context.Context, cv *cvv1alpha1.ComponentVersion, execCtx *ApplyContext) error {
	name, pkg, err := yi.loadPackage(ctx, cv, execCtx)
	if err != nil {
		return err
	}
	if cv.Spec.YAML != nil {
		pkg.ApplyStrategy = cv.Spec.YAML.ApplyStrategy
	}

	if err := yi.applier.ApplyComponent(ctx, pkg); err != nil {
		return errors.Wrapf(err, "%s: apply", name)
	}

	if err := yi.maybePrune(ctx, cv, pkg); err != nil {
		return errors.Wrapf(err, "%s: prune", name)
	}

	if err := yi.maybeHealthCheck(ctx, cv, execCtx); err != nil {
		return errors.Wrapf(err, "%s: health check", name)
	}
	return nil
}

// Uninstall loads manifests, deletes package resources, then optionally prunes leftovers.
// Delete/Prune treat missing objects as success so repeated Uninstall is idempotent.
// Version-diff auto-uninstall is out of scope; this only exposes the capability.
func (yi *YamlInstaller) Uninstall(ctx context.Context, cv *cvv1alpha1.ComponentVersion, execCtx *ApplyContext) error {
	name, pkg, err := yi.loadPackage(ctx, cv, execCtx)
	if err != nil {
		return err
	}

	if err := yi.applier.DeleteComponent(ctx, pkg); err != nil {
		return errors.Wrapf(err, "%s: delete", name)
	}

	if err := yi.maybePrune(ctx, cv, pkg); err != nil {
		return errors.Wrapf(err, "%s: prune", name)
	}
	return nil
}

func (yi *YamlInstaller) loadPackage(
	ctx context.Context,
	cv *cvv1alpha1.ComponentVersion,
	execCtx *ApplyContext,
) (string, *manifest.ComponentPackage, error) {
	if yi == nil {
		return "", nil, errors.New("yaml installer is nil")
	}
	if cv == nil {
		return "", nil, errors.New("component version is nil")
	}
	name := cv.Spec.Name
	if name == "" {
		name = cv.Name
	}
	if yi.store == nil {
		return name, nil, fmt.Errorf("%s: manifest store is nil", name)
	}
	if yi.applier == nil {
		return name, nil, fmt.Errorf("%s: manifest applier is nil", name)
	}

	tmpl := manifest.TemplateContext{}
	if execCtx != nil {
		tmpl = execCtx.TemplateContext
	}

	pkg, err := yi.store.GetComponentManifests(ctx, name, cv.Spec.Version, tmpl)
	if err != nil {
		return name, nil, errors.Wrapf(err, "%s: get manifests", name)
	}
	if pkg == nil {
		return name, nil, fmt.Errorf("%s: get manifests: nil package", name)
	}
	return name, pkg, nil
}

func (yi *YamlInstaller) maybePrune(ctx context.Context, cv *cvv1alpha1.ComponentVersion, pkg *manifest.ComponentPackage) error {
	yamlSpec := cv.Spec.YAML
	if yamlSpec == nil || !yamlSpec.Prune {
		return nil
	}
	if len(yamlSpec.PruneLabelSelector) == 0 {
		return errors.New("prune=true requires pruneLabelSelector")
	}
	ns := yamlSpec.Namespace
	if err := yi.applier.PruneResources(ctx, yamlSpec.PruneLabelSelector, ns, pkg.Manifests); err != nil {
		return errors.Wrap(err, "prune resources")
	}
	return nil
}

func (yi *YamlInstaller) maybeHealthCheck(ctx context.Context, cv *cvv1alpha1.ComponentVersion, execCtx *ApplyContext) error {
	yamlSpec := cv.Spec.YAML
	if yamlSpec == nil || yamlSpec.HealthCheck == nil || !yamlSpec.HealthCheck.Enabled {
		return nil
	}
	spec, err := healthcheck.FromCRD(yamlSpec.HealthCheck)
	if err != nil {
		return errors.Wrap(err, "convert healthCheck")
	}
	if execCtx == nil || execCtx.TargetClient == nil {
		return errors.New("health check requires target client")
	}
	if err := healthCheckRun(ctx, execCtx.TargetClient, spec); err != nil {
		return errors.Wrap(err, "run health check")
	}
	return nil
}
