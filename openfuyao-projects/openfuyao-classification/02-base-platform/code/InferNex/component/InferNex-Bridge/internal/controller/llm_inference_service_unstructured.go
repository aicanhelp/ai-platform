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

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
)

const (
	llmInferenceServiceGroup            = "serving.kserve.io"
	llmInferenceServiceKind             = "LLMInferenceService"
	llmInferenceServiceAPIVersionV1Alpha1 = "serving.kserve.io/v1alpha1"
	llmInferenceServiceAPIVersionV1Alpha2 = "serving.kserve.io/v1alpha2"
	llmInferenceServiceConfigKind       = "LLMInferenceServiceConfig"
)

// llmInferenceServiceAPIVersions returns API versions to try, preferring explicit ref version then v1alpha2.
func llmInferenceServiceAPIVersions(preferred string) []string {
	out := make([]string, 0, 3)
	appendIfMissing := func(version string) {
		version = strings.TrimSpace(version)
		if version == "" {
			return
		}
		for _, existing := range out {
			if existing == version {
				return
			}
		}
		out = append(out, version)
	}
	appendIfMissing(preferred)
	appendIfMissing(llmInferenceServiceAPIVersionV1Alpha2)
	appendIfMissing(llmInferenceServiceAPIVersionV1Alpha1)
	return out
}

func llmInferenceServiceConfigAPIVersions() []string {
	return []string{
		llmInferenceServiceAPIVersionV1Alpha2,
		llmInferenceServiceAPIVersionV1Alpha1,
	}
}

func (r *InferNexServiceReconciler) getLLMInferenceService(
	ctx context.Context,
	nn types.NamespacedName,
	preferredAPIVersion string,
) (*unstructured.Unstructured, error) {
	for _, apiVersion := range llmInferenceServiceAPIVersions(preferredAPIVersion) {
		llm := &unstructured.Unstructured{}
		llm.SetAPIVersion(apiVersion)
		llm.SetKind(llmInferenceServiceKind)
		if err := r.Get(ctx, nn, llm); err != nil {
			if apierrors.IsNotFound(err) || meta.IsNoMatchError(err) {
				continue
			}
			return nil, err
		}
		return llm, nil
	}
	gr := schema.GroupResource{Group: llmInferenceServiceGroup, Resource: "llminferenceservices"}
	return nil, apierrors.NewNotFound(gr, nn.Name)
}
