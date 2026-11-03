/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package controller

import (
	"context"
	"fmt"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

func pruneOwnerRefByUID(refs []metav1.OwnerReference, uid types.UID) ([]metav1.OwnerReference, bool) {
	if uid == "" {
		return refs, false
	}
	out := make([]metav1.OwnerReference, 0, len(refs))
	changed := false
	for _, ref := range refs {
		if ref.UID == uid {
			changed = true
			continue
		}
		out = append(out, ref)
	}
	return out, changed
}

func (r *InferNexServiceReconciler) pruneOwnerRefByUIDOrDelete(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
	obj client.Object,
) error {
	if owner == nil || owner.UID == "" {
		return nil
	}
	base, ok := obj.DeepCopyObject().(client.Object)
	if !ok {
		return fmt.Errorf("deep copy %T is not a client.Object", obj)
	}
	refs, changed := pruneOwnerRefByUID(obj.GetOwnerReferences(), owner.UID)
	if !changed {
		return nil
	}
	if len(refs) == 0 {
		if err := r.Delete(ctx, obj); err != nil && !apierrors.IsNotFound(err) {
			return err
		}
		return nil
	}
	obj.SetOwnerReferences(refs)
	return client.IgnoreNotFound(r.Patch(ctx, obj, client.MergeFrom(base)))
}
