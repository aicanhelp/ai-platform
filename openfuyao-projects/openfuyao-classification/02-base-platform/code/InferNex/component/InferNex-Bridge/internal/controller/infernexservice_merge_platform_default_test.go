/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */


package controller

import (
	"context"
	"strings"
	"testing"

	"k8s.io/utils/ptr"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

func TestNeedsPlatformTemplateFill_Mooncake(t *testing.T) {
	t.Parallel()
	spec := &infernexv1alpha1.InferNexServiceSpec{
		Components: &infernexv1alpha1.InfernexComponentsSpec{
			Mooncake: &infernexv1alpha1.MooncakeComponentSpec{
				Enabled: ptr.To(true),
				Master:  &infernexv1alpha1.TemplateComponentSpec{Replicas: 1},
			},
		},
	}
	if !needsPlatformTemplateFill(spec) {
		t.Fatal("expected true when mooncake enabled without master template")
	}
	spec.Components.Mooncake.Master.Template = &corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "mooncake-master", Image: "img"}}},
	}
	if needsPlatformTemplateFill(spec) {
		t.Fatal("expected false after mooncake master template set")
	}
}

func TestNeedsPlatformTemplateFill_CacheIndexerDoesNotRequirePlatformTemplate(t *testing.T) {
	t.Parallel()
	spec := &infernexv1alpha1.InferNexServiceSpec{
		Components: &infernexv1alpha1.InfernexComponentsSpec{
			CacheIndexer: &infernexv1alpha1.CacheIndexerComponentSpec{
				Enabled: ptr.To(true),
			},
		},
	}
	if needsPlatformTemplateFill(spec) {
		t.Fatal("cache-indexer uses built-in assets; platform template fill not required")
	}
}

func TestFillMissingTemplatesFromPlatformDefault_Mooncake(t *testing.T) {
	t.Parallel()
	dst := &infernexv1alpha1.InferNexServiceSpec{
		Components: &infernexv1alpha1.InfernexComponentsSpec{
			Mooncake: &infernexv1alpha1.MooncakeComponentSpec{
				Enabled: ptr.To(true),
				Master:  &infernexv1alpha1.TemplateComponentSpec{Replicas: 2},
			},
		},
	}
	src := &infernexv1alpha1.InferNexServiceSpec{
		Components: &infernexv1alpha1.InfernexComponentsSpec{
			Mooncake: &infernexv1alpha1.MooncakeComponentSpec{
				Enabled: ptr.To(true),
				Master: &infernexv1alpha1.TemplateComponentSpec{
					Replicas: 1,
					Template: &corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{{Name: "mooncake-master", Image: "default/img:v1"}},
						},
					},
				},
			},
		},
	}
	fillMissingTemplatesFromPlatformDefault(dst, src)
	if dst.Components.Mooncake.Master.Template == nil {
		t.Fatal("expected mooncake master template filled from platform default")
	}
	if got := dst.Components.Mooncake.Master.Template.Spec.Containers[0].Image; got != "default/img:v1" {
		t.Fatalf("image: got %q", got)
	}
	if dst.Components.Mooncake.Master.Replicas != 2 {
		t.Fatalf("replicas should stay user value 2, got %d", dst.Components.Mooncake.Master.Replicas)
	}
}

func TestMergePlatformDefaultComponentsFillsOmittedSubtrees(t *testing.T) {
	t.Parallel()
	dst := &infernexv1alpha1.InferNexServiceSpec{
		Components: &infernexv1alpha1.InfernexComponentsSpec{
			CacheIndexer: &infernexv1alpha1.CacheIndexerComponentSpec{
				Enabled:  ptr.To(true),
				Replicas: 1,
			},
		},
	}
	src := &infernexv1alpha1.InferNexServiceSpec{
		Components: &infernexv1alpha1.InfernexComponentsSpec{
			Mooncake: &infernexv1alpha1.MooncakeComponentSpec{
				Enabled: ptr.To(true),
				Master: &infernexv1alpha1.TemplateComponentSpec{
					Template: &corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{{Name: "mooncake-master", Image: "plat/moon:v1"}},
						},
					},
				},
			},
		},
	}
	mergePlatformDefaultComponents(dst, src)
	if dst.Components.Mooncake == nil || dst.Components.Mooncake.Master == nil {
		t.Fatal("expected mooncake.master merged from platform components")
	}
	if dst.Components.CacheIndexer.Replicas != 1 {
		t.Fatal("user cache-indexer replicas must be preserved")
	}
}

func TestMergeCacheIndexer_MergesEnabledAndReplicas(t *testing.T) {
	t.Parallel()
	dst := &infernexv1alpha1.CacheIndexerComponentSpec{Replicas: 2}
	src := &infernexv1alpha1.CacheIndexerComponentSpec{
		Enabled:  ptr.To(true),
		Replicas: 1,
	}
	mergeCacheIndexer(dst, src)
	if dst.Enabled == nil || !*dst.Enabled {
		t.Fatal("expected enabled merged from src")
	}
	if dst.Replicas != 2 {
		t.Fatal("dst replicas should win when non-zero")
	}
}

func TestMergePlatformDefaultTemplates_LinkedAndDirect(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	platform := platformAggregateConfig()
	platform.Spec.InferNexServiceSpec.Components = &infernexv1alpha1.InfernexComponentsSpec{
		Mooncake: &infernexv1alpha1.MooncakeComponentSpec{
			Enabled: ptr.To(true),
			Master: &infernexv1alpha1.TemplateComponentSpec{
				Template: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{Name: "mooncake-master", Image: "plat:v1"}},
					},
				},
			},
		},
	}

	t.Run("linked overlays platform components", func(t *testing.T) {
		t.Parallel()
		insvc := newLinkedInferNexService("ns-a", "demo")
		effective := insvc.Spec.DeepCopy()
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(platform).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s, TemplateNamespace: "tpl-ns"}
		if err := r.mergePlatformDefaultTemplates(ctx, insvc, effective, defaultAggregateTemplateName); err != nil {
			t.Fatalf("mergePlatformDefaultTemplates linked error: %v", err)
		}
		if effective.Components == nil || effective.Components.Mooncake == nil {
			t.Fatal("expected mooncake merged for linked mode")
		}
	})

	t.Run("direct merges full platform spec", func(t *testing.T) {
		t.Parallel()
		insvc := &infernexv1alpha1.InferNexService{
			ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
			Spec: infernexv1alpha1.InferNexServiceSpec{
				Engine: &infernexv1alpha1.InferenceEngineSpec{
					InferenceEngineWorkloadSpec: infernexv1alpha1.InferenceEngineWorkloadSpec{
						Replicas: ptr.To(int32(1)),
						Template: testTemplate("engine:v1", 8000),
					},
				},
			},
		}
		effective := insvc.Spec.DeepCopy()
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(platform).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s, TemplateNamespace: "tpl-ns"}
		if err := r.mergePlatformDefaultTemplates(ctx, insvc, effective, defaultAggregateTemplateName); err != nil {
			t.Fatalf("mergePlatformDefaultTemplates direct error: %v", err)
		}
		if effective.Components == nil || effective.Components.Mooncake == nil {
			t.Fatal("expected mooncake merged for direct mode")
		}
	})

	t.Run("missing platform config errors when fill required", func(t *testing.T) {
		t.Parallel()
		insvc := &infernexv1alpha1.InferNexService{
			ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
			Spec: infernexv1alpha1.InferNexServiceSpec{
				IntelligentGatewayRouting: &infernexv1alpha1.IntelligentGatewayRoutingSpec{
					Router: &infernexv1alpha1.ComponentSpec{Enabled: ptr.To(true)},
				},
			},
		}
		effective := insvc.Spec.DeepCopy()
		cl := fake.NewClientBuilder().WithScheme(s).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s, TemplateNamespace: "tpl-ns"}
		err := r.mergePlatformDefaultTemplates(ctx, insvc, effective, defaultAggregateTemplateName)
		if err == nil || !strings.Contains(err.Error(), "not found") {
			t.Fatalf("expected not found error when router template missing, got %v", err)
		}
	})
}
