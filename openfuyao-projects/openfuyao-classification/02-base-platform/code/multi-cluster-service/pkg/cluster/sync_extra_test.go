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

package cluster

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestRetrieveClusterStatus(t *testing.T) {
	t.Run("success with resource summary", func(t *testing.T) {
		api := newFakeKarmadaAPI(newTestKarmadaCluster("member1", map[string]string{"a": "b"}, true))
		usage, conditions, err := RetrieveClusterStatus("member1", api.newClient(t))
		if err != nil {
			t.Fatalf("RetrieveClusterStatus() error = %v", err)
		}
		if usage == nil {
			t.Fatal("RetrieveClusterStatus() usage = nil")
		}
		if usage.AllocatableCPU != "4.0" {
			t.Errorf("AllocatableCPU = %q, want 4.0", usage.AllocatableCPU)
		}
		if usage.AllocatableMemory == "" || usage.AllocatedCPU != "1.0" || usage.AllocatedMemory == "" {
			t.Errorf("unexpected usage: %+v", usage)
		}
		if len(conditions) != 0 {
			// cluster object has no conditions; the copied slice must stay empty
			t.Errorf("conditions = %v, want empty", conditions)
		}
	})

	t.Run("cluster object is not ready when cluster missing", func(t *testing.T) {
		api := newFakeKarmadaAPI()
		usage, _, err := RetrieveClusterStatus("ghost", api.newClient(t))
		if err == nil {
			t.Fatal("RetrieveClusterStatus() expected error for missing cluster")
		}
		if err.Error() != "cluster object is not ready" {
			t.Errorf("error = %v, want cluster object is not ready", err)
		}
		if usage != nil {
			t.Errorf("usage = %v, want nil", usage)
		}
	})

	t.Run("cluster object with conditions", func(t *testing.T) {
		c := newTestKarmadaCluster("member2", nil, true)
		c.Status.Conditions = []metav1.Condition{{Type: "Ready", Status: "True"}}
		api := newFakeKarmadaAPI(c)
		_, conditions, err := RetrieveClusterStatus("member2", api.newClient(t))
		if err != nil {
			t.Fatalf("RetrieveClusterStatus() error = %v", err)
		}
		if len(conditions) != 1 || conditions[0].Type != "Ready" {
			t.Errorf("conditions = %v, want copied Ready condition", conditions)
		}
	})
}
