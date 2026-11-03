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
	"testing"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

func TestPruneOwnerRefByUID(t *testing.T) {
	t.Parallel()
	ownerUID := types.UID("uid-a")
	otherUID := types.UID("uid-b")
	refs := []metav1.OwnerReference{
		{APIVersion: "infernex.io/v1alpha1", Kind: "InferNexService", Name: "a", UID: ownerUID},
		{APIVersion: "infernex.io/v1alpha1", Kind: "InferNexService", Name: "b", UID: otherUID},
	}

	out, changed := pruneOwnerRefByUID(refs, ownerUID)
	if !changed || len(out) != 1 || out[0].UID != otherUID {
		t.Fatalf("expected one remaining ref, got changed=%v refs=%#v", changed, out)
	}

	out, changed = pruneOwnerRefByUID(refs, types.UID("missing"))
	if changed || len(out) != 2 {
		t.Fatalf("expected no change for missing uid, got changed=%v refs=%#v", changed, out)
	}

	out, changed = pruneOwnerRefByUID(refs, "")
	if changed || len(out) != 2 {
		t.Fatalf("empty uid must not prune, got changed=%v refs=%#v", changed, out)
	}
}

func TestPruneOwnerRefByUIDOrDelete_RemovesLastOwner(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	owner := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a", UID: types.UID("uid-demo")},
	}
	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "redis-service",
			Namespace: "ns-a",
			OwnerReferences: []metav1.OwnerReference{
				{APIVersion: "infernex.io/v1alpha1", Kind: "InferNexService", Name: owner.Name, UID: owner.UID},
			},
		},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(owner, svc).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	if err := r.pruneOwnerRefByUIDOrDelete(ctx, owner, svc); err != nil {
		t.Fatalf("pruneOwnerRefByUIDOrDelete error: %v", err)
	}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "redis-service"}, &corev1.Service{}); !apierrors.IsNotFound(err) {
		t.Fatalf("expected singleton service deleted after last ownerRef pruned, got err=%v", err)
	}
}

func TestPruneOwnerRefByUIDOrDelete_PreservesSharedOwner(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	ownerA := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "isvc-a", Namespace: "ns-a", UID: types.UID("uid-a")},
	}
	ownerB := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "isvc-b", Namespace: "ns-a", UID: types.UID("uid-b")},
	}
	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "redis-service",
			Namespace: "ns-a",
			OwnerReferences: []metav1.OwnerReference{
				{APIVersion: "infernex.io/v1alpha1", Kind: "InferNexService", Name: ownerA.Name, UID: ownerA.UID},
				{APIVersion: "infernex.io/v1alpha1", Kind: "InferNexService", Name: ownerB.Name, UID: ownerB.UID},
			},
		},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(ownerA, ownerB, svc).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	if err := r.pruneOwnerRefByUIDOrDelete(ctx, ownerA, svc); err != nil {
		t.Fatalf("pruneOwnerRefByUIDOrDelete error: %v", err)
	}
	got := &corev1.Service{}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "redis-service"}, got); err != nil {
		t.Fatalf("expected shared service preserved: %v", err)
	}
	assertMissingOwnerRef(t, got.OwnerReferences, ownerA.Name, ownerA.UID)
	assertNonControllerOwnerRef(t, got.OwnerReferences, ownerB.Name, ownerB.UID)
}
