/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */
package v1beta1

import (
	"testing"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
)

func TestBkeClusterTypes(t *testing.T) {
	t.Run("BKECluster", func(t *testing.T) {
		// Test BKECluster basic structure
		bkeCluster := &BKECluster{
			Status: confv1beta1.BKEClusterStatus{
				Ready: true,
				Phase: "Ready",
			},
		}

		if !bkeCluster.Status.Ready {
			t.Error("expected Ready to be true")
		}

		var nilBkeCluster *BKECluster
		nilBkeCluster.DeepCopy()
		nilBkeCluster.DeepCopyObject()
	})

	t.Run("BKEClusterList", func(t *testing.T) {
		list := &BKEClusterList{
			Items: []BKECluster{
				{Status: confv1beta1.BKEClusterStatus{Ready: true}},
			},
		}

		if len(list.Items) != 1 {
			t.Errorf("expected 1 item, got %d", len(list.Items))
		}

		var nilList *BKEClusterList
		nilList.DeepCopy()
		nilList.DeepCopyObject()
	})
}

// Note: Node state management tests have been moved to bkenode_types_test.go
// as part of BC split. Use BKENodes helper type for node state operations.
