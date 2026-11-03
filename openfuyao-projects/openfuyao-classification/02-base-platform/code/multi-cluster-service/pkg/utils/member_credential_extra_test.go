/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package utils

import (
	"errors"
	"testing"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	k8sfake "k8s.io/client-go/kubernetes/fake"
	testcore "k8s.io/client-go/testing"
)

// NOTE: the happy paths of deleteNamespace/isNamespaceDeleted are covered by the
// existing TestIsNamespaceDeleted suite; calling deleteNamespace against a live
// fake namespace object panics inside isNamespaceDeleted (unchecked ns deref and
// Finalizers[0] index access) - recorded as a product defect, not fixed here.
func TestDeleteNamespaceExtra(t *testing.T) {
	t.Run("delete request failed", func(t *testing.T) {
		cs := k8sfake.NewSimpleClientset()
		cs.PrependReactor("delete", "namespaces",
			func(action testcore.Action) (bool, runtime.Object, error) {
				return true, nil, apierrors.NewInternalError(errors.New("boom"))
			})

		if err := deleteNamespace(cs); err == nil {
			t.Error("deleteNamespace() expected error when delete request fails")
		}
	})
}
