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
	"fmt"

	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	"sigs.k8s.io/cluster-api/util/conditions"
	"sigs.k8s.io/controller-runtime/pkg/client"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
)

func CalculateBKEMachineBootNum(bkeMachines []bkev1beta1.BKEMachine) (int, int) {
	successBootNodeNum := 0
	failedBootNodeNum := 0
	for _, bkeMachine := range bkeMachines {
		bootCondition := conditions.Get(&bkeMachine, bkev1beta1.BootstrapSucceededCondition)
		if bootCondition == nil {
			continue
		}
		if bootCondition.Status == corev1.ConditionTrue && bkeMachine.Status.Bootstrapped {
			successBootNodeNum++
		}

		if bootCondition.Status == corev1.ConditionFalse && bootCondition.Reason == constant.NodeBootStrapFailedReason {
			failedBootNodeNum++
		}
	}
	return failedBootNodeNum, successBootNodeNum
}

// GenerateProviderID return one unique provider ID
func GenerateProviderID(cluster *bkev1beta1.BKECluster, node confv1beta1.Node) string {
	// example: bke://cluster-name/b64encode(node-ip)
	// ip is unique,when the nodeinfo created by bkeadm
	// cluster api bke also checks the uniqueness of the IP field in nodeinfo created by bkeadm
	return fmt.Sprintf("bke://%s/%s", cluster.Name, utils.B64Encode(node.IP))
}

func IsControlPlaneBKEMachine(machine *bkev1beta1.BKEMachine) bool {
	_, ok := machine.ObjectMeta.Labels[clusterv1.MachineControlPlaneLabel]
	return ok
}

func GetControlPlaneBKEMachines(ctx context.Context, c client.Client, cluster *bkev1beta1.BKECluster) ([]*bkev1beta1.BKEMachine, error) {
	machines, err := GetBKEClusterAssociateBKEMachines(ctx, c, cluster)
	if err != nil {
		return nil, err
	}

	var controlPlaneMachines []*bkev1beta1.BKEMachine

	for _, machine := range machines {
		if IsControlPlaneBKEMachine(&machine) {
			controlPlaneMachines = append(controlPlaneMachines, &machine)
		}
	}

	if len(controlPlaneMachines) == 0 {
		return nil, errors.Errorf("not found any control plane BKEMachine")
	}

	return controlPlaneMachines, nil
}

func GetControlPlaneInitBKEMachine(ctx context.Context, c client.Client, cluster *bkev1beta1.BKECluster) (*bkev1beta1.BKEMachine, error) {
	machines, err := GetControlPlaneBKEMachines(ctx, c, cluster)
	if err != nil {
		return nil, err
	}

	var oldest metav1.Time
	var exceptMachine *bkev1beta1.BKEMachine
	// 最早创建的那个
	for _, machine := range machines {
		if oldest.IsZero() || machine.CreationTimestamp.Before(&oldest) {
			oldest = machine.CreationTimestamp
			exceptMachine = machine
		}
	}
	if exceptMachine == nil {
		return nil, errors.New("not found init BKEMachine")
	}

	return exceptMachine, nil
}
