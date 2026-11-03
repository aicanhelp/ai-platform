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

	"github.com/agiledragon/gomonkey/v2"
)

func TestIsClusterUIDValidExtra(t *testing.T) {
	t.Run("cluster exists", func(t *testing.T) {
		api := newFakeKarmadaAPI(newTestKarmadaCluster("member1", nil, true))
		ok, err := IsClusterUIDValid(api.newClient(t), "member1")
		if err != nil || !ok {
			t.Errorf("IsClusterUIDValid() = %v, %v; want true, nil", ok, err)
		}
	})

	t.Run("cluster missing", func(t *testing.T) {
		api := newFakeKarmadaAPI()
		ok, err := IsClusterUIDValid(api.newClient(t), "ghost")
		if err != nil || ok {
			t.Errorf("IsClusterUIDValid() = %v, %v; want false, nil", ok, err)
		}
	})

	t.Run("list failed", func(t *testing.T) {
		api := newFakeKarmadaAPI()
		api.failList = 500
		if _, err := IsClusterUIDValid(api.newClient(t), "ghost"); err == nil {
			t.Error("IsClusterUIDValid() expected error when list fails")
		}
	})
}

func TestIsClusterNameRepeatedExtra(t *testing.T) {
	t.Run("name repeated", func(t *testing.T) {
		api := newFakeKarmadaAPI(newTestKarmadaCluster("member1", nil, true))
		repeated, err := IsClusterNameRepeated("member1", api.newClient(t))
		if err != nil || !repeated {
			t.Errorf("IsClusterNameRepeated() = %v, %v; want true, nil", repeated, err)
		}
	})

	t.Run("name unique", func(t *testing.T) {
		api := newFakeKarmadaAPI()
		repeated, err := IsClusterNameRepeated("member1", api.newClient(t))
		if err != nil || repeated {
			t.Errorf("IsClusterNameRepeated() = %v, %v; want false, nil", repeated, err)
		}
	})

	t.Run("list failed", func(t *testing.T) {
		api := newFakeKarmadaAPI()
		api.failList = 500
		if _, err := IsClusterNameRepeated("member1", api.newClient(t)); err == nil {
			t.Error("IsClusterNameRepeated() expected error when list fails")
		}
	})
}

func TestIsClusterDeletedExtra(t *testing.T) {
	t.Run("already deleted", func(t *testing.T) {
		api := newFakeKarmadaAPI()
		if err := IsClusterDeleted(api.newClient(t), "member1"); err != nil {
			t.Errorf("IsClusterDeleted() error = %v, want nil", err)
		}
	})
}

func TestIsKarmadaServiceReadyExtra(t *testing.T) {
	t.Run("service ready", func(t *testing.T) {
		patches := gomonkey.ApplyFunc(isKarmadaAPIServiceReady, func(string) error {
			return nil
		})
		defer patches.Reset()

		api := newFakeKarmadaAPI(newTestKarmadaCluster("host", nil, true))
		if err := IsKarmadaServiceReady(api.newClient(t)); err != nil {
			t.Errorf("IsKarmadaServiceReady() error = %v, want nil", err)
		}
	})

	t.Run("apiserver not reachable", func(t *testing.T) {
		patches := gomonkey.ApplyFunc(isKarmadaAPIServiceReady, func(string) error {
			return errors.New("connection refused")
		})
		defer patches.Reset()

		api := newFakeKarmadaAPI()
		if err := IsKarmadaServiceReady(api.newClient(t)); err == nil {
			t.Error("IsKarmadaServiceReady() expected error when apiserver unreachable")
		}
	})
}
