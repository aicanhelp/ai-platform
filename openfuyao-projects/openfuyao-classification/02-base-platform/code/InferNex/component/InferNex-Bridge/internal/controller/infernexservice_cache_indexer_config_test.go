package controller

import (
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
	corev1 "k8s.io/api/core/v1"
)

func TestInfernexEngineGroupValue(t *testing.T) {
	t.Parallel()

	if got := infernexEngineGroupValue("infernex-bridge-system", "qwen3-8b-pd"); got != "infernex-bridge-system.qwen3-8b-pd" {
		t.Fatalf("unexpected value: %q", got)
	}
	if got := infernexEngineGroupValue("  ns  ", "  name  "); got != "ns.name" {
		t.Fatalf("unexpected trimmed value: %q", got)
	}
}

func TestCacheIndexerConfigYAML(t *testing.T) {
	t.Parallel()

	const (
		ns   = "infernex-bridge-system"
		name = "qwen3-8b-pd"
		val  = "infernex-bridge-system.qwen3-8b-pd"
	)

	t.Run("insvc aggregate uses aggregateGroup and namespace.name value", func(t *testing.T) {
		yaml := cacheIndexerConfigYAML(cacheIndexerDiscoveryInsvcAggregate, ns, name)
		if !strings.Contains(yaml, "engineKey: infernex.io/aggregateGroup") ||
			!strings.Contains(yaml, "engineValue: "+val) ||
			!strings.Contains(yaml, "pdRoleKey: openfuyao.com/pdRole") {
			t.Fatalf("expected aggregateGroup discovery keys, got:\n%s", yaml)
		}
		if !strings.Contains(yaml, "- aggregate") || strings.Contains(yaml, "kserve.io/component") {
			t.Fatalf("expected insvc aggregate pdRole only, got:\n%s", yaml)
		}
	})

	t.Run("insvc pd uses pdEngineGroup prefill only", func(t *testing.T) {
		yaml := cacheIndexerConfigYAML(cacheIndexerDiscoveryInsvcPD, ns, name)
		if !strings.Contains(yaml, "engineKey: infernex.io/pdEngineGroup") ||
			!strings.Contains(yaml, "engineValue: "+val) ||
			!strings.Contains(yaml, "pdRoleKey: openfuyao.com/pdRole") ||
			!strings.Contains(yaml, "- prefill") {
			t.Fatalf("expected pdEngineGroup prefill discovery, got:\n%s", yaml)
		}
		if strings.Contains(yaml, "- aggregate") || strings.Contains(yaml, "- decode") {
			t.Fatalf("insvc pd must not discover aggregate or decode pods, got:\n%s", yaml)
		}
	})

	t.Run("kserve pd uses app name and all prefill component labels", func(t *testing.T) {
		yaml := cacheIndexerConfigYAML(cacheIndexerDiscoveryKServePD, ns, name)
		if !strings.Contains(yaml, "engineKey: app.kubernetes.io/name") ||
			!strings.Contains(yaml, "engineValue: "+name) ||
			!strings.Contains(yaml, "pdRoleKey: app.kubernetes.io/component") ||
			!strings.Contains(yaml, "- "+kserveWorkloadComponentPrefill) ||
			!strings.Contains(yaml, "- "+kserveWorkloadComponentLeaderPrefill) ||
			!strings.Contains(yaml, "- "+kserveWorkloadComponentWorkerPrefill) {
			t.Fatalf("expected kserve name + all prefill component values, got:\n%s", yaml)
		}
		if strings.Contains(yaml, "- decode") {
			t.Fatal("kserve pd cache-indexer must not discover decode")
		}
	})

	t.Run("yaml is parseable for all discovery modes", func(t *testing.T) {
		modes := []cacheIndexerDiscoveryMode{
			cacheIndexerDiscoveryInsvcAggregate,
			cacheIndexerDiscoveryInsvcPD,
			cacheIndexerDiscoveryKServePD,
			cacheIndexerDiscoveryKServeWorkload,
		}
		for _, mode := range modes {
			var doc map[string]any
			cfg := cacheIndexerConfigYAML(mode, ns, name)
			if err := yaml.Unmarshal([]byte(cfg), &doc); err != nil {
				t.Fatalf("mode %d: invalid config yaml: %v\n%s", mode, err, cfg)
			}
		}
	})

	t.Run("kserve workload default profile matches explicit mode", func(t *testing.T) {
		const engineValue = "ex-ag-03-cn-lws"
		got := discoveryProfileForCacheIndexer(cacheIndexerDiscoveryKServeWorkload, engineValue)
		fallback := discoveryProfileForCacheIndexer(cacheIndexerDiscoveryMode(99), engineValue)
		if !reflect.DeepEqual(got, fallback) {
			t.Fatalf("unexpected profile drift:\nexplicit=%#v\nfallback=%#v", got, fallback)
		}
	})

	t.Run("kserve workload uses app name and aggregate component labels", func(t *testing.T) {
		yaml := cacheIndexerConfigYAML(cacheIndexerDiscoveryKServeWorkload, ns, name)
		if !strings.Contains(yaml, "engineKey: app.kubernetes.io/name") ||
			!strings.Contains(yaml, "engineValue: "+name) ||
			!strings.Contains(yaml, "pdRoleKey: app.kubernetes.io/component") ||
			!strings.Contains(yaml, "- "+kserveWorkloadComponentDecode) ||
			!strings.Contains(yaml, "- "+kserveWorkloadComponentLeader) ||
			!strings.Contains(yaml, "- "+kserveWorkloadComponentWorker) {
			t.Fatalf("expected kserve name + deployment/LWS aggregate component values, got:\n%s", yaml)
		}
		if strings.Contains(yaml, "kserve.io/component") ||
			strings.Contains(yaml, "workload-prefill") ||
			strings.Contains(yaml, "openfuyao.com/pdRole") {
			t.Fatal("kserve aggregate must not use kserve.io/component or prefill/insvc labels")
		}
	})
}

func TestApplyCacheIndexerPodTemplate(t *testing.T) {
	t.Parallel()

	tpl := &corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{
				Name: "cache-indexer",
				Ports: []corev1.ContainerPort{{
					Name: "http", ContainerPort: 28080,
				}},
			}},
		},
	}
	applyCacheIndexerPodTemplate(tpl, "demo-cache-indexer-config")

	c := tpl.Spec.Containers[0]
	foundNS := false
	for _, e := range c.Env {
		if e.Name == envPodNamespace && e.ValueFrom != nil && e.ValueFrom.FieldRef != nil &&
			e.ValueFrom.FieldRef.FieldPath == "metadata.namespace" {
			foundNS = true
		}
	}
	if !foundNS {
		t.Fatalf("expected POD_NAMESPACE fieldRef, got %#v", c.Env)
	}
	if c.LivenessProbe == nil || c.ReadinessProbe == nil {
		t.Fatal("expected health probes")
	}
	volFound, mountFound := false, false
	for _, v := range tpl.Spec.Volumes {
		if v.Name == cacheIndexerConfigVolumeName && v.ConfigMap != nil && v.ConfigMap.Name == "demo-cache-indexer-config" {
			volFound = true
		}
	}
	for _, m := range c.VolumeMounts {
		if m.Name == cacheIndexerConfigVolumeName && m.MountPath == cacheIndexerConfigMountPath {
			mountFound = true
		}
	}
	if !volFound || !mountFound {
		t.Fatalf("expected config volume/mount, volumes=%#v mounts=%#v", tpl.Spec.Volumes, c.VolumeMounts)
	}
}

func TestMergeAggregateInferenceWorkloadPodTemplateLabels(t *testing.T) {
	t.Parallel()

	tpl := &corev1.PodTemplateSpec{}
	mergeAggregateInferenceWorkloadPodTemplateLabels(tpl, "infernex-bridge-system", "qwen3-8b-aggregate")
	want := infernexEngineGroupValue("infernex-bridge-system", "qwen3-8b-aggregate")
	if tpl.Labels[labelInfernexAggregateGroup] != want {
		t.Fatalf("expected aggregateGroup %q, got %#v", want, tpl.Labels)
	}
	if tpl.Labels[labelOpenFuyaoPDRole] != openfuyaoPDRoleAggregate {
		t.Fatalf("unexpected aggregate pdRole: %#v", tpl.Labels)
	}
	if _, ok := tpl.Labels[labelInfernexPDEngineGroup]; ok {
		t.Fatal("aggregate must not set pdEngineGroup")
	}
}
