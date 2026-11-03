/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package predicates

import (
	"testing"

	"github.com/stretchr/testify/assert"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	"sigs.k8s.io/controller-runtime/pkg/event"
)

func TestClusterUnPause_Update(t *testing.T) {
	pred := ClusterUnPause()
	cluster := &clusterv1.Cluster{Spec: clusterv1.ClusterSpec{Paused: false}}
	e := event.UpdateEvent{ObjectNew: cluster}
	result := pred.Update(e)
	assert.True(t, result)
}

func TestClusterUnPause_Create(t *testing.T) {
	pred := ClusterUnPause()
	cluster := &clusterv1.Cluster{Spec: clusterv1.ClusterSpec{Paused: false}}
	e := event.CreateEvent{Object: cluster}
	result := pred.Create(e)
	assert.True(t, result)
}
