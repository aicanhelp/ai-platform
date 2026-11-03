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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/event"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
)

func TestBKEAgentReady_Update(t *testing.T) {
	pred := BKEAgentReady()
	cluster := &bkev1beta1.BKECluster{}
	e := event.UpdateEvent{ObjectNew: cluster}
	result := pred.Update(e)
	assert.False(t, result)
}

func TestBKEAgentReady_Create(t *testing.T) {
	pred := BKEAgentReady()
	e := event.CreateEvent{}
	result := pred.Create(e)
	assert.False(t, result)
}

func TestBKEClusterUnPause_Update(t *testing.T) {
	pred := BKEClusterUnPause()
	cluster := &bkev1beta1.BKECluster{Spec: confv1beta1.BKEClusterSpec{Pause: false}}
	e := event.UpdateEvent{ObjectNew: cluster}
	result := pred.Update(e)
	assert.True(t, result)
}

func TestBKEClusterUnPause_Create(t *testing.T) {
	pred := BKEClusterUnPause()
	cluster := &bkev1beta1.BKECluster{Spec: confv1beta1.BKEClusterSpec{Pause: false}}
	e := event.CreateEvent{Object: cluster}
	result := pred.Create(e)
	assert.True(t, result)
}

func TestBKEClusterUnPause_Delete(t *testing.T) {
	pred := BKEClusterUnPause()
	e := event.DeleteEvent{}
	result := pred.Delete(e)
	assert.True(t, result)
}

func TestBKEClusterSpecChange_Create(t *testing.T) {
	pred := BKEClusterSpecChange()
	cluster := &bkev1beta1.BKECluster{}
	e := event.CreateEvent{Object: cluster}
	result := pred.Create(e)
	assert.True(t, result)
}

func TestBKEClusterSpecChange_Update(t *testing.T) {
	pred := BKEClusterSpecChange()
	oldCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Generation: 1}}
	newCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Generation: 2}}
	e := event.UpdateEvent{ObjectOld: oldCluster, ObjectNew: newCluster}
	result := pred.Update(e)
	assert.True(t, result)
}

func TestBKEClusterAnnotationsChange_Update(t *testing.T) {
	pred := BKEClusterAnnotationsChange()
	oldCluster := &bkev1beta1.BKECluster{}
	newCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				annotation.CVOUpgradeReadyAnnotationKey: "v2.5.0",
			},
		},
	}
	e := event.UpdateEvent{ObjectOld: oldCluster, ObjectNew: newCluster}
	result := pred.Update(e)
	assert.True(t, result)
}

func TestBKENodeChange_Create(t *testing.T) {
	pred := BKENodeChange()
	node := &confv1beta1.BKENode{}
	e := event.CreateEvent{Object: node}
	result := pred.Create(e)
	assert.True(t, result)
}

func TestBKENodeChange_Update(t *testing.T) {
	pred := BKENodeChange()
	oldNode := &confv1beta1.BKENode{ObjectMeta: metav1.ObjectMeta{Generation: 1}}
	newNode := &confv1beta1.BKENode{ObjectMeta: metav1.ObjectMeta{Generation: 2}}
	e := event.UpdateEvent{ObjectOld: oldNode, ObjectNew: newNode}
	result := pred.Update(e)
	assert.True(t, result)
}

func TestBKENodeChange_Delete(t *testing.T) {
	pred := BKENodeChange()
	node := &confv1beta1.BKENode{}
	e := event.DeleteEvent{Object: node}
	result := pred.Delete(e)
	assert.True(t, result)
}
