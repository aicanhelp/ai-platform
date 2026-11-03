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

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	controlplanev1 "sigs.k8s.io/cluster-api/controlplane/kubeadm/api/v1beta1"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestGetClusterAPIKubeadmControlPlane_NilCluster(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = controlplanev1.AddToScheme(scheme)
	c := fake.NewClientBuilder().WithScheme(scheme).Build()

	_, err := GetClusterAPIKubeadmControlPlane(context.Background(), c, nil)
	assert.Error(t, err)
}

func TestGetClusterAPIMachineDeployment_NilCluster(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = clusterv1.AddToScheme(scheme)
	c := fake.NewClientBuilder().WithScheme(scheme).Build()

	_, err := GetClusterAPIMachineDeployment(context.Background(), c, nil)
	assert.Error(t, err)
}

func TestGetClusterAPIMachineDeployment_EmptyList(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = clusterv1.AddToScheme(scheme)
	c := fake.NewClientBuilder().WithScheme(scheme).Build()

	cluster := &clusterv1.Cluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}

	md, err := GetClusterAPIMachineDeployment(context.Background(), c, cluster)
	assert.NoError(t, err)
	assert.Nil(t, md)
}

func TestGetClusterAPIAssociateObjs_NilCluster(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = clusterv1.AddToScheme(scheme)
	c := fake.NewClientBuilder().WithScheme(scheme).Build()

	_, err := GetClusterAPIAssociateObjs(context.Background(), c, nil)
	assert.Error(t, err)
}

func TestGetMachineAssociateBKEMachine_NotFound(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = clusterv1.AddToScheme(scheme)
	c := fake.NewClientBuilder().WithScheme(scheme).Build()

	machine := &clusterv1.Machine{
		Spec: clusterv1.MachineSpec{
			InfrastructureRef: corev1.ObjectReference{
				Namespace: "default",
				Name:      "test-bke",
			},
		},
	}

	_, err := GetMachineAssociateBKEMachine(context.Background(), c, machine)
	assert.Error(t, err)
}
