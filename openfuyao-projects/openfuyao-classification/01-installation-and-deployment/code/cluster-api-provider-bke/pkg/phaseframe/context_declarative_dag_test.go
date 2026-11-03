/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package phaseframe

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
)

// Test-only constants (RFC 2606 reserved domain / fixture versions; not production config).
const (
	testDeclarativeClusterName = "c"
	testDeclarativeNamespace   = "default"
	testEtcdVersionTarget      = "v3.6.7-of.1"
	testEtcdVersionPrior       = "v3.5.0"
)

func TestFinishDeclarativeDAGForPhaseFlow(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: testDeclarativeClusterName, Namespace: testDeclarativeNamespace},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{
					EtcdVersion: testEtcdVersionTarget,
				},
			},
		},
		Status: confv1beta1.BKEClusterStatus{
			EtcdVersion: testEtcdVersionTarget,
		},
	}
	combinedCM := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testDeclarativeClusterName,
			Namespace: testDeclarativeNamespace,
		},
		Data: map[string]string{
			"nodes":  "[]",
			"status": "[]",
		},
	}
	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster, combinedCM).Build()

	pc := NewReconcilePhaseCtx(t.Context()).
		SetClient(c).
		SetScheme(scheme).
		SetBKECluster(bkeCluster)
	vc := upgrade.NewVersionContext()
	vc.SetCurrent(upgrade.ComponentEtcd, testEtcdVersionPrior)
	vc.SetTarget(upgrade.ComponentEtcd, testEtcdVersionTarget)
	pc.SetVersionContext(vc)

	assert.NoError(t, pc.FinishDeclarativeDAGForPhaseFlow())
	assert.True(t, pc.DeclarativeDAGCompleted)
	assert.Nil(t, pc.VersionContext)
}
