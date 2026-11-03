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

package v1alpha1

import (
	"encoding/json"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

func TestClusterVersionDeepCopy(t *testing.T) {
	now := metav1.Now()
	cv := &ClusterVersion{
		ObjectMeta: metav1.ObjectMeta{Name: "cluster-a", Namespace: "default"},
		Spec: ClusterVersionSpec{
			DesiredVersion: "v26.03",
		},
		Status: ClusterVersionStatus{
			CurrentVersion: "v26.02",
			Phase:          ClusterVersionPhaseUpgrading,
			UpgradeHistory: []ClusterUpgradeRecord{{
				From:      "v26.02",
				To:        "v26.03",
				StartedAt: &now,
				Status:    ClusterUpgradeRecordStatusSucceeded,
			}},
			Conditions: []ClusterVersionCondition{{
				Type:               "Ready",
				Status:             metav1.ConditionTrue,
				LastTransitionTime: now,
			}},
		},
	}

	copied := cv.DeepCopy()
	copied.Status.UpgradeHistory[0].From = "changed"
	copied.Status.Conditions[0].Type = "Changed"

	if cv.Status.UpgradeHistory[0].From != "v26.02" {
		t.Fatalf("upgrade history was not deep copied")
	}
	if cv.Status.Conditions[0].Type != "Ready" {
		t.Fatalf("conditions were not deep copied")
	}
	assertRuntimeObject(t, cv.DeepCopyObject())
}

func TestReleaseImageDeepCopy(t *testing.T) {
	now := metav1.Now()
	ri := &ReleaseImage{
		ObjectMeta: metav1.ObjectMeta{Name: "release-v26.03", Namespace: "default"},
		Spec: ReleaseImageSpec{
			Version: "v26.03",
			Install: &ReleaseImageInstallSpec{
				Components: []ReleaseImageInstallComponent{{Name: "kube-apiserver", Version: "v1.32.0"}},
			},
			Upgrade: &ReleaseImageUpgradeSpec{
				Components: []ReleaseImageUpgradeComponent{{
					Name:    "kubelet",
					Version: "v1.32.0",
					Inline:  &ReleaseImageUpgradeInline{Handler: "restart", Version: "v1"},
				}},
			},
		},
		Status: ReleaseImageStatus{
			Phase:          ReleaseImagePhaseValid,
			ComponentCount: 1,
			Components:     []ComponentStatus{{Name: "kubelet", Version: "v1.32.0", Type: ComponentTypeInline}},
			ValidatedAt:    &now,
		},
	}

	copied := ri.DeepCopy()
	copied.Spec.Install.Components[0].Name = "changed"
	copied.Spec.Upgrade.Components[0].Inline.Handler = "changed"
	copied.Status.Components[0].Name = "changed"

	if ri.Spec.Install.Components[0].Name != "kube-apiserver" {
		t.Fatalf("install components were not deep copied")
	}
	if ri.Spec.Upgrade.Components[0].Inline.Handler != "restart" {
		t.Fatalf("upgrade inline handler was not deep copied")
	}
	if ri.Status.Components[0].Name != "kubelet" {
		t.Fatalf("status components were not deep copied")
	}
	assertRuntimeObject(t, ri.DeepCopyObject())
}

func TestUpgradePathDeepCopy(t *testing.T) {
	now := metav1.Now()
	up := &UpgradePath{
		ObjectMeta: metav1.ObjectMeta{Name: "upgrade-path"},
		Spec: UpgradePathSpec{
			Paths: []UpgradePathRule{{
				From:      "v26.02",
				To:        "v26.03",
				PostCheck: []CheckStep{{Name: "post-health", Required: true}},
			}},
			Versions: []VersionEntry{{Version: "v26.03"}},
		},
		Status: UpgradePathStatus{
			Phase:         UpgradePathPhaseActive,
			LastCheckedAt: &now,
			Conditions:    []metav1.Condition{{Type: "Validated", Status: metav1.ConditionTrue}},
		},
	}

	copied := up.DeepCopy()
	copied.Spec.Paths[0].PostCheck[0].Name = "changed"
	copied.Spec.Versions[0].Version = "changed"
	copied.Status.Conditions[0].Type = "Changed"

	if up.Spec.Paths[0].PostCheck[0].Name != "post-health" {
		t.Fatalf("path checks were not deep copied")
	}
	if up.Spec.Versions[0].Version != "v26.03" {
		t.Fatalf("versions were not deep copied")
	}
	if up.Status.Conditions[0].Type != "Validated" {
		t.Fatalf("conditions were not deep copied")
	}
	assertRuntimeObject(t, up.DeepCopyObject())
}

func TestComponentVersionDeepCopy(t *testing.T) {
	cv := &ComponentVersion{
		ObjectMeta: metav1.ObjectMeta{Name: "kubelet"},
		Spec: ComponentVersionSpec{
			Name:    "kubelet",
			Type:    ComponentTypeInline,
			Version: "v1.32.0",
			Inline:  &InlineSpec{Handler: "restart", Version: "v1"},
			YAML: &YAMLSpec{
				Namespace:     "kube-system",
				ApplyStrategy: ApplyStrategyServerSideApply,
				Prune:         true,
				PruneLabelSelector: map[string]string{
					"app.kubernetes.io/name": "coredns",
				},
				HealthCheck: &HealthCheckSpec{
					Enabled:  true,
					Timeout:  "5m",
					Interval: "2s",
					Checks: []HealthCheckItemSpec{
						{
							Type: "PodReady",
							PodReady: &PodReadyCheckSpec{
								Namespace:     "kube-system",
								LabelSelector: "k8s-app=kube-dns",
								MinReady:      1,
							},
						},
						{
							Type: "Custom",
							Custom: &CustomCheckSpec{
								Command: "true",
							},
						},
					},
				},
			},
			SubComponents: []SubComponent{{Name: "kube-proxy", Version: "v1.32.0"}},
			Compatibility: CompatibilitySpec{Constraints: []Constraint{{Component: "containerd", Rule: ">=1.7"}}},
			Dependencies:  []Dependency{{Name: "containerd", Phase: "Ready"}},
			Resources:     []ResourceSpec{{Kind: "ConfigMap", APIVersion: "v1", Name: "kubelet", Labels: map[string]string{"app": "kubelet"}}},
		},
		Status: ComponentVersionStatus{
			Conditions: []metav1.Condition{{Type: "Ready", Status: metav1.ConditionTrue}},
		},
	}

	copied := cv.DeepCopy()
	copied.Spec.Inline.Handler = "changed"
	copied.Spec.YAML.Namespace = "changed"
	copied.Spec.YAML.PruneLabelSelector["app.kubernetes.io/name"] = "changed"
	copied.Spec.YAML.HealthCheck.Checks[0].PodReady.LabelSelector = "changed"
	copied.Spec.YAML.HealthCheck.Checks[1].Custom.Command = "changed"
	copied.Spec.SubComponents[0].Name = "changed"
	copied.Spec.Compatibility.Constraints[0].Rule = "changed"
	copied.Spec.Dependencies[0].Name = "changed"
	copied.Spec.Resources[0].Labels["app"] = "changed"
	copied.Status.Conditions[0].Type = "Changed"

	if cv.Spec.Inline.Handler != "restart" {
		t.Fatalf("inline spec was not deep copied")
	}
	if cv.Spec.YAML.Namespace != "kube-system" {
		t.Fatalf("yaml namespace was not deep copied")
	}
	if cv.Spec.YAML.PruneLabelSelector["app.kubernetes.io/name"] != "coredns" {
		t.Fatalf("yaml prune labels were not deep copied")
	}
	if cv.Spec.YAML.HealthCheck.Checks[0].PodReady.LabelSelector != "k8s-app=kube-dns" {
		t.Fatalf("yaml health check was not deep copied")
	}
	if cv.Spec.YAML.HealthCheck.Checks[1].Custom.Command != "true" {
		t.Fatalf("yaml health check command was not deep copied")
	}
	if cv.Spec.SubComponents[0].Name != "kube-proxy" {
		t.Fatalf("sub components were not deep copied")
	}
	if cv.Spec.Compatibility.Constraints[0].Rule != ">=1.7" {
		t.Fatalf("compatibility constraints were not deep copied")
	}
	if cv.Spec.Dependencies[0].Name != "containerd" {
		t.Fatalf("dependencies were not deep copied")
	}
	if cv.Spec.Resources[0].Labels["app"] != "kubelet" {
		t.Fatalf("resource labels were not deep copied")
	}
	if cv.Status.Conditions[0].Type != "Ready" {
		t.Fatalf("conditions were not deep copied")
	}
	assertRuntimeObject(t, cv.DeepCopyObject())
}

func TestComponentVersionHelmDeepCopyAndJSON(t *testing.T) {
	raw := []byte(`{
		"apiVersion":"config.openfuyao.com/v1alpha1",
		"kind":"ComponentVersion",
		"metadata":{"name":"coredns"},
		"spec":{
			"name":"coredns",
			"type":"helm",
			"version":"v1.12.3",
			"helm":{
				"namespace":"kube-system",
				"releaseName":"coredns",
				"chart":{"name":"coredns-chart","version":"1.2.3"},
				"valuesFiles":["base.yaml","override.yaml"],
				"values":{"image":{"tag":"v1"},"replicaCount":2},
				"strategy":{"mode":"Upgrade","wait":true,"waitForJobs":true,"timeout":"5m","atomic":true,"createNamespace":true},
				"healthCheck":{"enabled":true,"timeout":"1m","interval":"5s","checks":[
					{"type":"EndpointReady","endpointReady":{"namespace":"kube-system","serviceName":"kube-dns","port":53}},
					{"type":"Custom","custom":{"command":"true"}}
				]}
			}
		}
	}`)
	var cv ComponentVersion
	if err := json.Unmarshal(raw, &cv); err != nil {
		t.Fatalf("unmarshal helm ComponentVersion: %v", err)
	}
	if cv.Spec.Type != ComponentTypeHelm || cv.Spec.Helm == nil {
		t.Fatalf("expected helm spec, got %#v", cv.Spec)
	}
	if cv.Spec.Helm.Strategy.Mode != HelmModeUpgrade || !cv.Spec.Helm.Strategy.CreateNamespace {
		t.Fatalf("unexpected strategy %#v", cv.Spec.Helm.Strategy)
	}

	copied := cv.DeepCopy()
	copied.Spec.Helm.Chart.Name = "changed"
	copied.Spec.Helm.ValuesFiles[0] = "changed.yaml"
	copied.Spec.Helm.Values["image"].(map[string]interface{})["tag"] = "changed"
	copied.Spec.Helm.HealthCheck.Checks[0].EndpointReady.ServiceName = "changed"
	copied.Spec.Helm.HealthCheck.Checks[1].Custom.Command = "changed"

	if cv.Spec.Helm.Chart.Name != "coredns-chart" {
		t.Fatalf("helm chart was not deep copied")
	}
	if cv.Spec.Helm.ValuesFiles[0] != "base.yaml" {
		t.Fatalf("helm valuesFiles were not deep copied")
	}
	if cv.Spec.Helm.Values["image"].(map[string]interface{})["tag"] != "v1" {
		t.Fatalf("helm values were not deep copied")
	}
	if cv.Spec.Helm.HealthCheck.Checks[0].EndpointReady.ServiceName != "kube-dns" {
		t.Fatalf("endpoint health check was not deep copied")
	}
	if cv.Spec.Helm.HealthCheck.Checks[1].Custom.Command != "true" {
		t.Fatalf("custom health check was not deep copied")
	}
}
func TestComponentVersionLegacyWithoutYAMLDeserializes(t *testing.T) {
	// Old ComponentVersion JSON without yaml remains valid; YAML stays nil.
	raw := []byte(`{
		"apiVersion":"config.openfuyao.com/v1alpha1",
		"kind":"ComponentVersion",
		"metadata":{"name":"legacy-yaml"},
		"spec":{
			"name":"coredns",
			"type":"yaml",
			"version":"v1.0.0",
			"resources":[{"kind":"ConfigMap","apiVersion":"v1","name":"coredns"}]
		}
	}`)
	var cv ComponentVersion
	if err := json.Unmarshal(raw, &cv); err != nil {
		t.Fatalf("unmarshal legacy ComponentVersion: %v", err)
	}
	if cv.Spec.YAML != nil {
		t.Fatalf("expected nil YAML on legacy object, got %#v", cv.Spec.YAML)
	}
	copied := cv.DeepCopy()
	if copied.Spec.YAML != nil {
		t.Fatalf("expected nil YAML after deepcopy, got %#v", copied.Spec.YAML)
	}
}

func TestListDeepCopyObjects(t *testing.T) {
	assertRuntimeObject(t, (&ClusterVersionList{Items: []ClusterVersion{{ObjectMeta: metav1.ObjectMeta{Name: "cluster-a"}}}}).DeepCopyObject())
	assertRuntimeObject(t, (&ReleaseImageList{Items: []ReleaseImage{{ObjectMeta: metav1.ObjectMeta{Name: "release-v26.03"}}}}).DeepCopyObject())
	assertRuntimeObject(t, (&UpgradePathList{Items: []UpgradePath{{ObjectMeta: metav1.ObjectMeta{Name: "upgrade-path"}}}}).DeepCopyObject())
	assertRuntimeObject(t, (&ComponentVersionList{Items: []ComponentVersion{{ObjectMeta: metav1.ObjectMeta{Name: "kubelet"}}}}).DeepCopyObject())
}

func assertRuntimeObject(t *testing.T, obj runtime.Object) {
	t.Helper()
	if obj == nil {
		t.Fatalf("expected runtime object")
	}
}
