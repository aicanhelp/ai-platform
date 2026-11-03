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

package dagexec

import (
	"context"

	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/manifest"
)

// ComponentVersionStore loads a ComponentVersion for Executor Spec reads and type fallback.
// Implemented by manifest.BundleStore via GetComponentVersion (same Bundle as ManifestStore).
type ComponentVersionStore interface {
	GetComponentVersion(ctx context.Context, name, version string) (*cvv1alpha1.ComponentVersion, error)
}

// Ensure *manifest.BundleStore satisfies ComponentVersionStore.
var _ ComponentVersionStore = (*manifest.BundleStore)(nil)
