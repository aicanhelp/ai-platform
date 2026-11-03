// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package controller

import (
	"errors"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/discovery"
)

// optionalAPIAbsent is only for cleanup. Active components must report missing
// prerequisites. Never hide permission, transport or unrelated discovery errors.
func optionalAPIAbsent(err error, target schema.GroupVersion) bool {
	var discoveryErr *discovery.ErrGroupDiscoveryFailed
	if errors.As(err, &discoveryErr) {
		if discoveryErr == nil || len(discoveryErr.Groups) != 1 {
			return false
		}
		targetErr, exists := discoveryErr.Groups[target]
		return exists && apierrors.IsNotFound(targetErr)
	}
	return meta.IsNoMatchError(err) || apierrors.IsNotFound(err)
}
