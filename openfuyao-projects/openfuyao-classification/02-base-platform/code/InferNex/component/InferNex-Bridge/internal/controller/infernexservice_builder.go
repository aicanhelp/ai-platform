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

// Package controller contains reconciliation logic for InferNexService resources.
package controller

import (
	"context"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

func (r *InferNexServiceReconciler) buildInferNexServiceFromLLMInferenceService(
	ctx context.Context,
	nn types.NamespacedName,
) error {
	llm, err := r.getLLMInferenceService(ctx, nn, "")
	if err != nil {
		if apierrors.IsNotFound(err) {
			if err := r.Delete(ctx, &infernexv1alpha1.InferNexService{
				ObjectMeta: metav1.ObjectMeta{Name: nn.Name, Namespace: nn.Namespace},
			}); err != nil && !apierrors.IsNotFound(err) {
				return err
			}
			return nil
		}
		return err
	}

	if llm.GetLabels()[infernexRuntimeLabel] != infernexRuntimeValue {
		if err := r.Delete(ctx, &infernexv1alpha1.InferNexService{
			ObjectMeta: metav1.ObjectMeta{Name: nn.Name, Namespace: nn.Namespace},
		}); err != nil && !apierrors.IsNotFound(err) {
			return err
		}
		return nil
	}

	infsvc := &infernexv1alpha1.InferNexService{}
	if err := r.Get(ctx, nn, infsvc); err != nil {
		if !apierrors.IsNotFound(err) {
			return err
		}
		infsvc = &infernexv1alpha1.InferNexService{
			ObjectMeta: metav1.ObjectMeta{
				Name:      nn.Name,
				Namespace: nn.Namespace,
				Labels: map[string]string{
					infernexRuntimeLabel: infernexRuntimeValue,
				},
			},
			Spec: infernexv1alpha1.InferNexServiceSpec{
				SourceRef: &infernexv1alpha1.SourceRef{
					APIVersion: llm.GetAPIVersion(),
					Kind:       "LLMInferenceService",
					Name:       nn.Name,
					Namespace:  nn.Namespace,
				},
			},
		}
		if err := r.Create(ctx, infsvc); err != nil && !apierrors.IsAlreadyExists(err) {
			return err
		}
	}

	return nil
}
