/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package phaseutil

import (
	"testing"

	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
)

func TestRecordControlPlaneConfigUpdateFailure(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c"}}
	old := confv1beta1.ComponentConfigHash{APIServer: "old"}
	cluster.Status.ControlPlaneConfigHash = old
	desired := confv1beta1.ComponentConfigHash{APIServer: "new"}

	assert.False(t, RecordControlPlaneConfigUpdateFailure(cluster, desired, 3))
	assert.Equal(t, old, cluster.Status.ControlPlaneConfigHash)
	assert.Equal(t, "1", cluster.Annotations[annotation.ControlPlaneConfigUpdateFailCountAnnotationKey])

	cluster.Annotations[annotation.ControlPlaneConfigUpdateFailCountAnnotationKey] = "2"
	assert.True(t, RecordControlPlaneConfigUpdateFailure(cluster, desired, 3))
	assert.Equal(t, desired, cluster.Status.ControlPlaneConfigHash)
	assert.Equal(t, "3", cluster.Annotations[annotation.ControlPlaneConfigUpdateFailCountAnnotationKey])
}

func TestRecordControlPlaneConfigUpdateFailureUsesAllowedFailedCount(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c"}}
	old := confv1beta1.ComponentConfigHash{APIServer: "old"}
	cluster.Status.ControlPlaneConfigHash = old
	desired := confv1beta1.ComponentConfigHash{APIServer: "new"}

	// ALLOWED_FAILED_COUNT=4 → 第 4 次失败才写 Status
	assert.False(t, RecordControlPlaneConfigUpdateFailure(cluster, desired, 4))
	assert.Equal(t, old, cluster.Status.ControlPlaneConfigHash)
	cluster.Annotations[annotation.ControlPlaneConfigUpdateFailCountAnnotationKey] = "3"
	assert.True(t, RecordControlPlaneConfigUpdateFailure(cluster, desired, 4))
	assert.Equal(t, desired, cluster.Status.ControlPlaneConfigHash)
	assert.Equal(t, "4", cluster.Annotations[annotation.ControlPlaneConfigUpdateFailCountAnnotationKey])
}

func TestRecordControlPlaneConfigUpdateSuccessClearsFailCount(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{annotation.ControlPlaneConfigUpdateFailCountAnnotationKey: "2"},
		},
	}
	desired := confv1beta1.ComponentConfigHash{APIServer: "ok"}
	RecordControlPlaneConfigUpdateSuccess(cluster, desired)
	assert.Equal(t, desired, cluster.Status.ControlPlaneConfigHash)
	_, ok := cluster.Annotations[annotation.ControlPlaneConfigUpdateFailCountAnnotationKey]
	assert.False(t, ok)
}

func TestClearConfigUpdateFailCounts(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				annotation.ControlPlaneConfigUpdateFailCountAnnotationKey: "1",
				annotation.EtcdConfigUpdateFailCountAnnotationKey:         "2",
			},
		},
	}
	ClearConfigUpdateFailCounts(cluster)
	assert.Empty(t, cluster.Annotations)
}
