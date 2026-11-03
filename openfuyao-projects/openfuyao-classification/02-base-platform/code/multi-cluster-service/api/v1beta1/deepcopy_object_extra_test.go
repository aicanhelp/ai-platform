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

package v1beta1

import (
	"testing"
)

func TestDeepCopyObjectNilReceiver(t *testing.T) {
	var cluster *Cluster
	if got := cluster.DeepCopyObject(); got != nil {
		t.Errorf("(*Cluster)(nil).DeepCopyObject() = %v, want nil", got)
	}

	var clusterList *ClusterList
	if got := clusterList.DeepCopyObject(); got != nil {
		t.Errorf("(*ClusterList)(nil).DeepCopyObject() = %v, want nil", got)
	}
}

func TestDeepCopyObjectValues(t *testing.T) {
	cluster := &Cluster{}
	cluster.Name = "member1"
	copied := cluster.DeepCopyObject().(*Cluster)
	if copied.Name != "member1" {
		t.Errorf("DeepCopyObject() name = %q, want member1", copied.Name)
	}

	list := &ClusterList{}
	list.Items = []Cluster{*cluster}
	listCopied := list.DeepCopyObject().(*ClusterList)
	if len(listCopied.Items) != 1 || listCopied.Items[0].Name != "member1" {
		t.Errorf("DeepCopyObject() items = %+v", listCopied.Items)
	}
}
