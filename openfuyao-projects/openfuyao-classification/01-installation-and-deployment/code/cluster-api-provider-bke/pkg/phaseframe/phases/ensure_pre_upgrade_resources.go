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

package phases

import (
	"fmt"
	"sort"
	"strings"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/yaml"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
)

const EnsurePreUpgradeResourcesName confv1beta1.BKEClusterPhase = "EnsurePreUpgradeResources"

// EnsurePreUpgradeResources provisions upgrade prerequisite resources declared in ComponentVersion.spec.resources.
type EnsurePreUpgradeResources struct {
	phaseframe.BasePhase
	componentVersion *cvv1alpha1.ComponentVersion
}

func NewEnsurePreUpgradeResourcesPhase(ctx *phaseframe.PhaseContext) *EnsurePreUpgradeResources {
	return &EnsurePreUpgradeResources{BasePhase: phaseframe.NewBasePhase(ctx, EnsurePreUpgradeResourcesName)}
}

func NewEnsurePreUpgradeResources(ctx *phaseframe.PhaseContext) phaseframe.Phase {
	return NewEnsurePreUpgradeResourcesPhase(ctx)
}

func NewEnsurePreUpgradeResourcesWithComponentVersion(ctx *phaseframe.PhaseContext, cv cvv1alpha1.ComponentVersion) phaseframe.Phase {
	phase := NewEnsurePreUpgradeResourcesPhase(ctx)
	phase.SetComponentVersion(cv)
	return phase
}

func (e *EnsurePreUpgradeResources) SetComponentVersion(cv cvv1alpha1.ComponentVersion) {
	e.componentVersion = cv.DeepCopy()
}

func (e *EnsurePreUpgradeResources) GetComponentVersion() *cvv1alpha1.ComponentVersion {
	return e.componentVersion
}

func (e *EnsurePreUpgradeResources) Version() string {
	if e.componentVersion == nil {
		return ""
	}
	return e.componentVersion.Spec.Version
}

func (e *EnsurePreUpgradeResources) NeedExecute(old, new *bkev1beta1.BKECluster) bool {
	if !e.BasePhase.DefaultNeedExecute(old, new) {
		return false
	}
	if decided, need := e.ComponentVersionDecision(upgrade.ComponentPreUpgradeResources); decided {
		if need {
			e.SetStatus(bkev1beta1.PhaseWaiting)
		}
		return need
	}
	return false
}

func (e *EnsurePreUpgradeResources) Execute() (ctrl.Result, error) {
	if e.componentVersion == nil {
		return ctrl.Result{}, fmt.Errorf("component version is not configured for %s", EnsurePreUpgradeResourcesName)
	}
	resources := e.componentVersion.Spec.Resources
	if len(resources) == 0 {
		return ctrl.Result{}, nil
	}
	for _, res := range e.sortResourcesByKind(resources) {
		if err := e.provisionResource(res); err != nil {
			return ctrl.Result{}, fmt.Errorf("provision %s/%s: %w", res.Kind, res.Name, err)
		}
	}
	return ctrl.Result{}, nil
}

func (e *EnsurePreUpgradeResources) provisionResource(spec cvv1alpha1.ResourceSpec) error {
	if strings.TrimSpace(spec.Manifest) != "" && spec.Kind != "ConfigMap" && spec.Kind != "Secret" {
		return e.provisionFromManifest(spec.Manifest)
	}
	switch spec.Kind {
	case "ConfigMap":
		return e.provisionConfigMap(spec)
	case "Secret":
		return e.provisionSecret(spec)
	default:
		if strings.TrimSpace(spec.Manifest) == "" {
			return fmt.Errorf("unsupported pre-upgrade resource kind %q without manifest", spec.Kind)
		}
		return e.provisionFromManifest(spec.Manifest)
	}
}

func (e *EnsurePreUpgradeResources) provisionConfigMap(spec cvv1alpha1.ResourceSpec) error {
	if e.Ctx == nil || e.Ctx.Client == nil {
		return fmt.Errorf("phase client is not configured")
	}
	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      spec.Name,
			Namespace: spec.Namespace,
			Labels:    copyStringMap(spec.Labels),
		},
		Data: copyStringMap(spec.Data),
	}
	if err := e.Ctx.Client.Create(e.Ctx, cm); err != nil && !apierrors.IsAlreadyExists(err) {
		return err
	}
	return nil
}

func (e *EnsurePreUpgradeResources) provisionSecret(spec cvv1alpha1.ResourceSpec) error {
	if e.Ctx == nil || e.Ctx.Client == nil {
		return fmt.Errorf("phase client is not configured")
	}
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      spec.Name,
			Namespace: spec.Namespace,
			Labels:    copyStringMap(spec.Labels),
		},
		StringData: copyStringMap(spec.StringData),
		Data:       make(map[string][]byte, len(spec.StringData)),
	}
	for k, v := range spec.StringData {
		secret.Data[k] = []byte(v)
	}
	if err := e.Ctx.Client.Create(e.Ctx, secret); err != nil && !apierrors.IsAlreadyExists(err) {
		return err
	}
	return nil
}

func (e *EnsurePreUpgradeResources) provisionFromManifest(manifest string) error {
	if e.Ctx == nil || e.Ctx.Client == nil {
		return fmt.Errorf("phase client is not configured")
	}
	obj := &unstructured.Unstructured{}
	if err := yaml.Unmarshal([]byte(manifest), &obj.Object); err != nil {
		return err
	}
	if err := e.Ctx.Client.Create(e.Ctx, obj); err != nil && !apierrors.IsAlreadyExists(err) {
		return err
	}
	return nil
}

func (e *EnsurePreUpgradeResources) sortResourcesByKind(resources []cvv1alpha1.ResourceSpec) []cvv1alpha1.ResourceSpec {
	sorted := append([]cvv1alpha1.ResourceSpec(nil), resources...)
	kindPriority := map[string]int{
		"CustomResourceDefinition": 0,
		"ConfigMap":                1,
		"Secret":                   2,
	}
	sort.SliceStable(sorted, func(i, j int) bool {
		pi, ok := kindPriority[sorted[i].Kind]
		if !ok {
			pi = 100
		}
		pj, ok := kindPriority[sorted[j].Kind]
		if !ok {
			pj = 100
		}
		if pi != pj {
			return pi < pj
		}
		return sorted[i].Name < sorted[j].Name
	})
	return sorted
}

func copyStringMap(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}
