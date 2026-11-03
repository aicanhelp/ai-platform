/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package serviceaccount

import (
	"context"
	"testing"
)

const testPk = "test"

func TestFyGetPublicKeysWhenKeysIsNil(t *testing.T) {
	j := fyDynamicPublicKeysGetter{}

	keys := j.GetPublicKeys(context.TODO(), "")
	if keys != nil {
		t.Errorf("expect keys is nil")
	}

	keys = j.GetPublicKeys(context.TODO(), testPk)
	if keys != nil {
		t.Errorf("expect keys is nil when key is %s", testPk)
	}
}

func TestFyGetPublicKeysWhenKeysIsNotNil(t *testing.T) {
	j := fyDynamicPublicKeysGetter{}

	j.publicKeysGetter.Store(&staticPublicKeysGetter{
		allPublicKeys:  []PublicKey{{KeyID: testPk}},
		publicKeysByID: map[string][]PublicKey{testPk: {{KeyID: testPk}}},
	})

	keys := j.GetPublicKeys(context.TODO(), "")
	if keys == nil {
		t.Errorf("expect keys is not nil")
	}

	keys = j.GetPublicKeys(context.TODO(), testPk)
	if keys == nil {
		t.Errorf("expect keys is not nil when key is %s", testPk)
	}
}

func TestFyGetPublicKeysWhenKeysIsNotSlice(t *testing.T) {
	j := fyDynamicPublicKeysGetter{}

	j.publicKeysGetter.Store("")
	keys := j.GetPublicKeys(context.TODO(), "")
	if keys != nil {
		t.Errorf("expect keys is nil")
	}
}
