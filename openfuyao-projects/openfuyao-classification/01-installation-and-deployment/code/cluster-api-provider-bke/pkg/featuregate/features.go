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

package featuregate

import (
	"strings"

	"sigs.k8s.io/controller-runtime/pkg/client"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/config"
)

const (
	// UpgradeReadyAnnotationKey is set by ClusterVersionReconciler when an upgrade should run.
	UpgradeReadyAnnotationKey = annotation.CVOUpgradeReadyAnnotationKey
	// HelmComponentAnnotationKey enables yaml/helm ComponentExecutor injection for a single object.
	// When present, its value overrides --helm-component-support (annotation-first).
	HelmComponentAnnotationKey = "cvo.openfuyao.cn/helm-component"
)

// UpgradeReady reports whether ClusterVersionReconciler requested a declarative upgrade.
func UpgradeReady(obj client.Object) (string, bool) {
	if obj == nil {
		return "", false
	}
	v, ok := annotation.HasAnnotation(obj, UpgradeReadyAnnotationKey)
	v = strings.TrimSpace(v)
	return v, ok && v != ""
}

// HelmComponentEnabled reports whether yaml/helm ComponentExecutors may be injected.
// Annotation cvo.openfuyao.cn/helm-component, when present, overrides --helm-component-support.
func HelmComponentEnabled(obj client.Object) bool {
	if obj != nil {
		if v, ok := annotation.HasAnnotation(obj, HelmComponentAnnotationKey); ok {
			return strings.EqualFold(strings.TrimSpace(v), "true")
		}
	}
	return config.HelmComponentSupport
}
