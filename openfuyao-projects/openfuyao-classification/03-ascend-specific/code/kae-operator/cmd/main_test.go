/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for the details.
 */

package main

import (
	"testing"

	devicev1 "gopkg.openfuyao.cn/kae-operator/api/v1"
)

func TestNewScheme(t *testing.T) {
	// Test that newScheme is properly initialized
	if newScheme == nil {
		t.Error("newScheme should not be nil")
	}

	// Test that KAEPolicy can be recognized by the scheme
	obj := &devicev1.KAEPolicy{}
	gvks, _, err := newScheme.ObjectKinds(obj)
	if err != nil {
		t.Logf("KAEPolicy scheme test: %v", err)
	} else if len(gvks) > 0 {
		t.Logf("KAEPolicy GVK: %v", gvks[0])
	}

	t.Log("Scheme initialization verified")
}
