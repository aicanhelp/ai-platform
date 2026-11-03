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

	"sigs.k8s.io/controller-runtime/pkg/client"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

// BuildEffectiveSpecForWebhook returns the same effective spec as reconcile-time buildEffectiveSpec
// (baseRefs merge, mode correction, then KServe-like overlay of infernex-default-{aggregate|pd} for direct services).
// so engine/router checks match the controller.
func BuildEffectiveSpecForWebhook(
	ctx context.Context,
	c client.Client,
	templateNamespace string,
	infsvc *infernexv1alpha1.InferNexService,
) (infernexv1alpha1.InferNexServiceSpec, string, error) {
	ns := strings.TrimSpace(templateNamespace)
	if ns == "" {
		ns = strings.TrimSpace(infsvc.Namespace)
	}
	r := &InferNexServiceReconciler{
		Client:            c,
		TemplateNamespace: ns,
	}
	return r.buildEffectiveSpec(ctx, infsvc)
}
