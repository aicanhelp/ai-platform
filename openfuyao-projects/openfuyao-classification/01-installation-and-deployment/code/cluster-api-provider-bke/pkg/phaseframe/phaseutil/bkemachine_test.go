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

package phaseutil

import (
	"context"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
)

func TestCalculateBKEMachineBootNum(t *testing.T) {
	machines := []bkev1beta1.BKEMachine{
		{
			Status: bkev1beta1.BKEMachineStatus{
				Bootstrapped: true,
				Conditions: []clusterv1.Condition{
					{Type: bkev1beta1.BootstrapSucceededCondition, Status: corev1.ConditionTrue},
				},
			},
		},
	}

	failed, success := CalculateBKEMachineBootNum(machines)
	assert.Equal(t, 1, success)
	assert.Equal(t, 0, failed)
}

func TestGenerateProviderID(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test-cluster"},
	}
	node := confv1beta1.Node{IP: "192.168.1.1"}
	result := GenerateProviderID(cluster, node)
	assert.Contains(t, result, "test-cluster")
}

func TestIsControlPlaneBKEMachine(t *testing.T) {
	machine := &bkev1beta1.BKEMachine{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{clusterv1.MachineControlPlaneLabel: ""},
		},
	}
	assert.True(t, IsControlPlaneBKEMachine(machine))

	workerMachine := &bkev1beta1.BKEMachine{ObjectMeta: metav1.ObjectMeta{}}
	assert.False(t, IsControlPlaneBKEMachine(workerMachine))
}

func TestGetControlPlaneBKEMachines(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "test"}}

	patches.ApplyFunc(GetBKEClusterAssociateBKEMachines, func(ctx context.Context, c client.Client, cluster *bkev1beta1.BKECluster) ([]bkev1beta1.BKEMachine, error) {
		return []bkev1beta1.BKEMachine{
			{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{clusterv1.MachineControlPlaneLabel: ""}}},
		}, nil
	})

	machines, err := GetControlPlaneBKEMachines(context.Background(), nil, cluster)
	assert.NoError(t, err)
	assert.Len(t, machines, 1)
}

func TestGetControlPlaneInitBKEMachine(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "test"}}

	patches.ApplyFunc(GetControlPlaneBKEMachines, func(ctx context.Context, c client.Client, cluster *bkev1beta1.BKECluster) ([]*bkev1beta1.BKEMachine, error) {
		return []*bkev1beta1.BKEMachine{
			{ObjectMeta: metav1.ObjectMeta{CreationTimestamp: metav1.Now()}},
		}, nil
	})

	machine, err := GetControlPlaneInitBKEMachine(context.Background(), nil, cluster)
	assert.NoError(t, err)
	assert.NotNil(t, machine)
}
