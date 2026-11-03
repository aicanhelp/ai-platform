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

package helminstaller

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"strings"
	"text/template"
	"time"

	"github.com/pkg/errors"
	"helm.sh/helm/v3/pkg/action"
	"helm.sh/helm/v3/pkg/chart"
	"helm.sh/helm/v3/pkg/chartutil"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/yaml"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/healthcheck"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/kube"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/manifest"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	defaultNamespace = "default"
	valuesYAMLKey    = "values.yaml"
	releaseNotFound  = "release: not found"
)

// healthCheckRun is separated for consistency with YamlInstaller and to keep
// HelmInstaller focused on orchestration instead of probe implementation.
var healthCheckRun = healthcheck.Run

// ApplyContext is the Helm engine-facing subset of dagexec.ExecutionContext.
type ApplyContext struct {
	TemplateContext  manifest.TemplateContext
	TargetClient     kubernetes.Interface
	TargetRestConfig *rest.Config
}

// HelmInstaller installs or upgrades ComponentVersion(Type=helm) by Helm SDK.
type HelmInstaller struct {
	store        manifest.ComponentFileStore
	localClient  client.Client
	bkeCluster   *bkev1beta1.BKECluster
	targetClient kubernetes.Interface
	restConfig   *rest.Config
	logger       *bkev1beta1.BKELogger
}

// HelmInstallerConfig contains only dependencies needed by release-image Helm execution.
type HelmInstallerConfig struct {
	Store        manifest.ComponentFileStore
	LocalClient  client.Client
	BKECluster   *bkev1beta1.BKECluster
	TargetClient kubernetes.Interface
	RestConfig   *rest.Config
	Logger       *bkev1beta1.BKELogger
}

// NewHelmInstaller constructs a HelmInstaller. Runtime methods validate nil dependencies.
func NewHelmInstaller(cfg HelmInstallerConfig) *HelmInstaller {
	return &HelmInstaller{
		store:        cfg.Store,
		localClient:  cfg.LocalClient,
		bkeCluster:   cfg.BKECluster,
		targetClient: cfg.TargetClient,
		restConfig:   cfg.RestConfig,
		logger:       cfg.Logger,
	}
}

// Apply runs Helm install or upgrade. Empty action means auto: existing release upgrades,
// missing release installs. This supports upgrading from a non-Helm old component.
func (hi *HelmInstaller) Apply(
	ctx context.Context,
	cv *cvv1alpha1.ComponentVersion,
	execCtx *ApplyContext,
	forcedAction Action,
) error {
	name, helmSpec, err := hi.validateComponent(cv)
	if err != nil {
		return err
	}

	namespace := releaseNamespace(helmSpec)
	releaseName := releaseName(name, helmSpec)
	actionConfig, err := hi.initActionConfig(namespace, execCtx)
	if err != nil {
		return errors.Wrapf(err, "%s: init helm action config", name)
	}

	values, err := hi.loadValues(ctx, cv, execCtx)
	if err != nil {
		return errors.Wrapf(err, "%s: load helm values", name)
	}
	chartFile, err := hi.fetchChart(ctx, cv, execCtx)
	if err != nil {
		return errors.Wrapf(err, "%s: fetch helm chart", name)
	}

	helmAction := forcedAction
	if helmAction == "" {
		exists, err := hi.releaseExists(actionConfig, releaseName)
		if err != nil {
			return errors.Wrapf(err, "%s: check helm release", name)
		}
		if exists {
			helmAction = ActionUpgrade
		} else {
			helmAction = ActionInstall
		}
	}

	run := helmRunOptions{
		actionConfig: actionConfig,
		helmSpec:     helmSpec,
		namespace:    namespace,
		releaseName:  releaseName,
		chartFile:    chartFile,
		values:       values,
	}
	switch helmAction {
	case ActionInstall:
		err = hi.install(ctx, run)
	case ActionUpgrade:
		err = hi.upgrade(run)
	default:
		err = fmt.Errorf("unsupported helm action %q", helmAction)
	}
	if err != nil {
		return errors.Wrapf(err, "%s: helm %s", name, helmAction)
	}

	if err := hi.maybeHealthCheck(ctx, helmSpec, execCtx); err != nil {
		return errors.Wrapf(err, "%s: health check", name)
	}
	return nil
}

// Uninstall exposes explicit Helm uninstall capability. Upgrade DAG does not call it.
func (hi *HelmInstaller) Uninstall(ctx context.Context, cv *cvv1alpha1.ComponentVersion, execCtx *ApplyContext) error {
	name, helmSpec, err := hi.validateComponent(cv)
	if err != nil {
		return err
	}
	namespace := releaseNamespace(helmSpec)
	actionConfig, err := hi.initActionConfig(namespace, execCtx)
	if err != nil {
		return errors.Wrapf(err, "%s: init helm action config", name)
	}

	uninstall := action.NewUninstall(actionConfig)
	uninstall.Wait = helmSpec.Strategy.Wait
	uninstall.Timeout, err = parseTimeout(helmSpec.Strategy.Timeout)
	if err != nil {
		return errors.Wrapf(err, "%s: parse helm timeout", name)
	}
	if _, err := uninstall.Run(releaseName(name, helmSpec)); err != nil {
		if strings.Contains(err.Error(), releaseNotFound) {
			return nil
		}
		return errors.Wrapf(err, "%s: helm uninstall", name)
	}
	return nil
}

func (hi *HelmInstaller) validateComponent(
	cv *cvv1alpha1.ComponentVersion,
) (string, *cvv1alpha1.HelmSpec, error) {
	if hi == nil {
		return "", nil, errors.New("helm installer is nil")
	}
	if cv == nil {
		return "", nil, errors.New("component version is nil")
	}
	name := cv.Spec.Name
	if name == "" {
		name = cv.Name
	}
	if cv.Spec.Helm == nil {
		return name, nil, fmt.Errorf("%s: spec.helm is required", name)
	}
	if hi.store == nil {
		return name, nil, fmt.Errorf("%s: component file store is nil", name)
	}
	if hi.localClient == nil {
		return name, nil, fmt.Errorf("%s: management client is nil", name)
	}
	if hi.bkeCluster == nil {
		return name, nil, fmt.Errorf("%s: bke cluster is nil", name)
	}
	return name, cv.Spec.Helm, nil
}

func (hi *HelmInstaller) initActionConfig(namespace string, execCtx *ApplyContext) (*action.Configuration, error) {
	restConfig := hi.restConfig
	if execCtx != nil && execCtx.TargetRestConfig != nil {
		restConfig = execCtx.TargetRestConfig
	}
	if restConfig == nil {
		return nil, errors.New("target rest config is nil")
	}

	actionConfig := new(action.Configuration)
	restClientGetter := kube.NewRESTClientConfig(namespace, restConfig)
	if err := actionConfig.Init(restClientGetter, namespace, "secret", normalLogger(hi.logger).Infof); err != nil {
		return nil, err
	}
	return actionConfig, nil
}

func (hi *HelmInstaller) releaseExists(actionConfig *action.Configuration, releaseName string) (bool, error) {
	_, err := action.NewStatus(actionConfig).Run(releaseName)
	if err == nil {
		return true, nil
	}
	if strings.Contains(err.Error(), releaseNotFound) {
		return false, nil
	}
	return true, err
}

type helmRunOptions struct {
	actionConfig *action.Configuration
	helmSpec     *cvv1alpha1.HelmSpec
	namespace    string
	releaseName  string
	chartFile    *chart.Chart
	values       chartutil.Values
}

func (hi *HelmInstaller) install(ctx context.Context, opts helmRunOptions) error {
	if err := hi.ensureNamespace(ctx, opts.helmSpec, opts.namespace); err != nil {
		return err
	}
	timeout, err := parseTimeout(opts.helmSpec.Strategy.Timeout)
	if err != nil {
		return err
	}
	install := action.NewInstall(opts.actionConfig)
	install.ReleaseName = opts.releaseName
	install.Namespace = opts.namespace
	install.CreateNamespace = opts.helmSpec.Strategy.CreateNamespace
	install.Wait = opts.helmSpec.Strategy.Wait
	install.WaitForJobs = opts.helmSpec.Strategy.WaitForJobs
	install.Timeout = timeout
	install.Atomic = opts.helmSpec.Strategy.Atomic
	_, err = install.Run(opts.chartFile, opts.values)
	return err
}

func (hi *HelmInstaller) upgrade(opts helmRunOptions) error {
	timeout, err := parseTimeout(opts.helmSpec.Strategy.Timeout)
	if err != nil {
		return err
	}
	upgrade := action.NewUpgrade(opts.actionConfig)
	upgrade.Namespace = opts.namespace
	upgrade.Wait = opts.helmSpec.Strategy.Wait
	upgrade.WaitForJobs = opts.helmSpec.Strategy.WaitForJobs
	upgrade.Timeout = timeout
	upgrade.Atomic = opts.helmSpec.Strategy.Atomic
	_, err = upgrade.Run(opts.releaseName, opts.chartFile, opts.values)
	return err
}

func (hi *HelmInstaller) ensureNamespace(ctx context.Context, helmSpec *cvv1alpha1.HelmSpec, namespace string) error {
	if !helmSpec.Strategy.CreateNamespace {
		return nil
	}
	clientset := hi.targetClient
	if clientset == nil {
		return errors.New("target client is nil")
	}
	ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: namespace}}
	if _, err := clientset.CoreV1().Namespaces().Create(ctx, ns, metav1.CreateOptions{}); err != nil &&
		!apierrors.IsAlreadyExists(err) {
		return fmt.Errorf("create namespace %s: %w", namespace, err)
	}
	return nil
}

func (hi *HelmInstaller) fetchChart(
	ctx context.Context,
	cv *cvv1alpha1.ComponentVersion,
	execCtx *ApplyContext,
) (*chart.Chart, error) {
	helmSpec := cv.Spec.Helm
	name, err := renderTemplateString(chartName(cv), templateContext(execCtx))
	if err != nil {
		return nil, fmt.Errorf("render chart.name: %w", err)
	}
	version, err := renderTemplateString(chartVersion(cv), templateContext(execCtx))
	if err != nil {
		return nil, fmt.Errorf("render chart.version: %w", err)
	}
	_ = helmSpec

	chartRepo, authConfig, err := kube.ChartRepositoryConfigFromBKECluster(ctx, hi.localClient, hi.bkeCluster)
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(chartRepo) == "" {
		return nil, errors.New("chart repository is empty")
	}
	return kube.FetchChartUniversal(chartRepo, name, version, authConfig, normalLogger(hi.logger))
}

func (hi *HelmInstaller) loadValues(
	ctx context.Context,
	cv *cvv1alpha1.ComponentVersion,
	execCtx *ApplyContext,
) (chartutil.Values, error) {
	values := chartutil.Values{}
	helmSpec := cv.Spec.Helm
	tmpl := templateContext(execCtx)

	for _, file := range helmSpec.ValuesFiles {
		fileValues, err := hi.loadValueFile(ctx, cv, file, tmpl)
		if err != nil {
			return nil, err
		}
		mergeValues(values, fileValues)
	}
	inlineValues, err := renderValue(helmSpec.Values, tmpl)
	if err != nil {
		return nil, err
	}
	if inlineMap, ok := inlineValues.(map[string]interface{}); ok {
		mergeValues(values, inlineMap)
	}
	return values, nil
}

func (hi *HelmInstaller) loadValueFile(
	ctx context.Context,
	cv *cvv1alpha1.ComponentVersion,
	file string,
	tmpl manifest.TemplateContext,
) (map[string]interface{}, error) {
	name := cv.Spec.Name
	if name == "" {
		name = cv.Name
	}
	data, err := hi.store.GetComponentFile(ctx, name, cv.Spec.Version, file)
	if err != nil {
		if os.IsNotExist(err) {
			warnf(hi.logger, "Helm values file %s/%s/%s is missing, skip it", name, cv.Spec.Version, file)
			return nil, nil
		}
		return nil, err
	}

	cm := &corev1.ConfigMap{}
	if err := yaml.Unmarshal(data, cm); err != nil {
		return nil, fmt.Errorf("parse values ConfigMap %s: %w", file, err)
	}
	raw, ok := cm.Data[valuesYAMLKey]
	if !ok || strings.TrimSpace(raw) == "" {
		warnf(hi.logger, "Helm values file %s/%s/%s has no data[%q], skip it", name, cv.Spec.Version, file, valuesYAMLKey)
		return nil, nil
	}
	rendered, err := renderTemplateString(raw, tmpl)
	if err != nil {
		return nil, fmt.Errorf("render values ConfigMap %s: %w", file, err)
	}
	values, err := chartutil.ReadValues([]byte(rendered))
	if err != nil {
		return nil, fmt.Errorf("parse values.yaml in %s: %w", file, err)
	}
	return values, nil
}

func (hi *HelmInstaller) maybeHealthCheck(ctx context.Context, helmSpec *cvv1alpha1.HelmSpec, execCtx *ApplyContext) error {
	if helmSpec == nil || helmSpec.HealthCheck == nil || !helmSpec.HealthCheck.Enabled {
		return nil
	}
	spec, err := healthcheck.FromCRD(helmSpec.HealthCheck)
	if err != nil {
		return errors.Wrap(err, "convert healthCheck")
	}
	clientset := hi.targetClient
	if execCtx != nil && execCtx.TargetClient != nil {
		clientset = execCtx.TargetClient
	}
	if clientset == nil {
		return errors.New("health check requires target client")
	}
	return healthCheckRun(ctx, clientset, spec)
}

func releaseNamespace(helmSpec *cvv1alpha1.HelmSpec) string {
	if helmSpec != nil && strings.TrimSpace(helmSpec.Namespace) != "" {
		return strings.TrimSpace(helmSpec.Namespace)
	}
	return defaultNamespace
}

func releaseName(componentName string, helmSpec *cvv1alpha1.HelmSpec) string {
	if helmSpec != nil && strings.TrimSpace(helmSpec.ReleaseName) != "" {
		return strings.TrimSpace(helmSpec.ReleaseName)
	}
	return componentName
}

func chartName(cv *cvv1alpha1.ComponentVersion) string {
	if cv.Spec.Helm != nil && strings.TrimSpace(cv.Spec.Helm.Chart.Name) != "" {
		return strings.TrimSpace(cv.Spec.Helm.Chart.Name)
	}
	if strings.TrimSpace(cv.Spec.Name) == "" {
		return cv.Name
	}
	return cv.Spec.Name
}

func chartVersion(cv *cvv1alpha1.ComponentVersion) string {
	if cv.Spec.Helm != nil && strings.TrimSpace(cv.Spec.Helm.Chart.Version) != "" {
		return strings.TrimSpace(cv.Spec.Helm.Chart.Version)
	}
	return cv.Spec.Version
}

func parseTimeout(raw string) (time.Duration, error) {
	if strings.TrimSpace(raw) == "" {
		return 0, nil
	}
	return time.ParseDuration(raw)
}

func templateContext(execCtx *ApplyContext) manifest.TemplateContext {
	if execCtx == nil {
		return manifest.TemplateContext{}
	}
	return execCtx.TemplateContext
}

func renderValue(in interface{}, tmpl manifest.TemplateContext) (interface{}, error) {
	switch v := in.(type) {
	case nil:
		return nil, nil
	case string:
		return renderTemplateString(v, tmpl)
	case map[string]interface{}:
		out := make(map[string]interface{}, len(v))
		for key, value := range v {
			renderedKey, err := renderTemplateString(key, tmpl)
			if err != nil {
				return nil, err
			}
			renderedValue, err := renderValue(value, tmpl)
			if err != nil {
				return nil, err
			}
			out[renderedKey] = renderedValue
		}
		return out, nil
	case []interface{}:
		out := make([]interface{}, 0, len(v))
		for _, item := range v {
			rendered, err := renderValue(item, tmpl)
			if err != nil {
				return nil, err
			}
			out = append(out, rendered)
		}
		return out, nil
	default:
		return in, nil
	}
}

func renderTemplateString(raw string, tmpl manifest.TemplateContext) (string, error) {
	if !strings.Contains(raw, "{{") {
		return raw, nil
	}
	t, err := template.New("helm-value").Option("missingkey=error").Parse(raw)
	if err != nil {
		return "", err
	}
	var buf bytes.Buffer
	if err := t.Execute(&buf, tmpl); err != nil {
		return "", err
	}
	return buf.String(), nil
}

func mergeValues(dst map[string]interface{}, src map[string]interface{}) {
	if dst == nil || src == nil {
		return
	}
	for key, srcValue := range src {
		if srcMap, ok := srcValue.(map[string]interface{}); ok {
			if dstMap, ok := dst[key].(map[string]interface{}); ok {
				mergeValues(dstMap, srcMap)
				continue
			}
		}
		dst[key] = srcValue
	}
}

func normalLogger(logger *bkev1beta1.BKELogger) *log.Logger {
	if logger != nil && logger.NormalLogger != nil {
		return logger.NormalLogger
	}
	return log.With("module", "helminstaller")
}

func warnf(logger *bkev1beta1.BKELogger, msg string, args ...interface{}) {
	if logger != nil {
		logger.Warn("helm installer", msg, args...)
		return
	}
	log.Warnf(msg, args...)
}
