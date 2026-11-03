/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package precheck

import (
	"context"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"installer/pkg/clients/k8s"
	"installer/pkg/constants"
	"installer/pkg/fuyao/v1beta1/cluster"
	"installer/pkg/zlog"
)

// AddNodesFlightCheck 添加节点检查
type AddNodesFlightCheck struct {
	Nodes cluster.Nodes

	FlightCheck  *FlightCheck
	clusterNodes *corev1.NodeList
}

// NewAddNodesFlightCheck is new AddNodesFlightCheck
func NewAddNodesFlightCheck(nodes cluster.Nodes) *AddNodesFlightCheck {
	return &AddNodesFlightCheck{
		Nodes:       nodes,
		FlightCheck: NewFlightCheck(nodes),
	}
}

func (a *AddNodesFlightCheck) isMasterCPUMemorySatisfy(cpuStandard, memoryStandard int,
	cpuMemInfo map[corev1.ResourceName]map[string]int64) bool {
	isSatisfy := true
	// 已加入集群master节点资源检查
	for resource, info := range cpuMemInfo {
		var stand = 0
		if resource == corev1.ResourceCPU {
			stand = cpuStandard
		} else {
			stand = memoryStandard
		}

		for ip, value := range info {
			if value >= int64(stand) {
				continue
			}
			isSatisfy = false
			zlog.Errorf("%s Node %s resources are insufficient, the number of resources is %d",
				ip, resource, value)
		}
	}
	return isSatisfy
}

func (a *AddNodesFlightCheck) checkMasterCPUMemory() error {
	workerCount := 0
	cpuMemInfo := make(map[corev1.ResourceName]map[string]int64)
	cpuMemInfo[corev1.ResourceCPU] = make(map[string]int64)
	cpuMemInfo[corev1.ResourceMemory] = make(map[string]int64)

	for _, node := range a.clusterNodes.Items {
		if !isMasterNodeByLabel(node.Labels) {
			workerCount++
			continue
		}

		ip := ""
		for _, address := range node.Status.Addresses {
			if address.Type == corev1.NodeInternalIP {
				ip = address.Address
				break
			}
		}

		for key, value := range node.Status.Capacity {
			if key == corev1.ResourceCPU {
				cpuMemInfo[corev1.ResourceCPU][ip] = value.Value()
			}
			if key == corev1.ResourceMemory {
				num := constants.MemoryConversionNum * constants.MemoryConversionNum * constants.MemoryConversionNum
				cpuMemInfo[corev1.ResourceMemory][ip] = value.Value() / int64(num)
			}
		}
	}

	cpuStandard, memoryStandard := computeMasterNodeResource(len(a.Nodes.Workers) + workerCount)

	if !a.isMasterCPUMemorySatisfy(cpuStandard, memoryStandard, cpuMemInfo) {
		return fmt.Errorf("the CPU or memory resources of the master node in the cluster are insufficient." +
			"For details, see logs")
	}

	// 未加入集群节点资源检查
	if err := a.FlightCheck.masterNodesCupCheck(cpuStandard); err != nil {
		return err
	}
	if err := a.FlightCheck.masterNodesMemoryCheck(memoryStandard); err != nil {
		return err
	}
	return nil
}

func (a *AddNodesFlightCheck) canAddMasterCheck() error {
	if len(a.Nodes.Masters) == 1 {
		return nil
	}

	masterCount := 0

	for _, node := range a.clusterNodes.Items {
		for key, value := range node.Labels {
			if key == constants.DefaultMasterLabelName && value == constants.DefaultMasterLabelValue {
				masterCount++
				break
			}
		}
	}

	if masterCount < AddMasterNeedMasterCount {
		return fmt.Errorf("a non-HA cluster cannot be expanded to HA")
	}

	return nil
}

func (a *AddNodesFlightCheck) checkOrSetHostName() error {
	hostnameMap := make(map[string]string)
	for _, node := range a.clusterNodes.Items {
		ip := ""
		for _, address := range node.Status.Addresses {
			if address.Type == corev1.NodeInternalIP {
				ip = address.Address
				break
			}
		}
		hostnameMap[node.Name] = ip
	}

	addNodes := make([]*cluster.Node, 0)
	addNodes = append(addNodes, a.Nodes.Workers...)
	addNodes = append(addNodes, a.Nodes.Masters[1:]...)

	return a.FlightCheck.checkOrSetHostName(addNodes, hostnameMap)
}

func (a *AddNodesFlightCheck) isHasSameIP() error {
	ipMap := make(map[string]bool)

	for _, node := range a.clusterNodes.Items {
		for _, address := range node.Status.Addresses {
			if address.Type == corev1.NodeInternalIP {
				ipMap[address.Address] = true
				break
			}
		}
	}

	addNodes := make([]*cluster.Node, 0)
	addNodes = append(addNodes, a.Nodes.Workers...)
	addNodes = append(addNodes, a.Nodes.Masters[1:]...)

	return a.FlightCheck.isHasSameIP(addNodes, ipMap)
}

// StartCheck 添加节点检查
func (a *AddNodesFlightCheck) StartCheck() error {
	fmt.Println("[preflight] Running pre-flight checks")
	kubeConfigPath, err := k8s.DownLoadKubeConfig(a.Nodes.Masters[0])
	if err != nil {
		return err
	}
	k8sClient, err := k8s.NewKubernetesClient(k8s.NewKubernetesCfg(kubeConfigPath))
	if err != nil {
		return fmt.Errorf("failed to get k8s client: %v", err)
	}
	clusterNodes, err := k8sClient.Kubernetes().CoreV1().Nodes().List(context.Background(), metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("failed to get cluster nodes: %v", err)
	}
	a.clusterNodes = clusterNodes

	for _, node := range a.Nodes.Masters[1:] {
		if node.PrivateKeyPath != "" || len(node.Password) != 0 {
			continue
		}
		node.InputPasswd()
	}

	for _, node := range a.Nodes.Workers {
		if node.PrivateKeyPath != "" || len(node.Password) != 0 {
			continue
		}
		node.InputPasswd()
	}

	checkFuncList := []func() error{a.FlightCheck.allNodePingCheck, a.isHasSameIP, a.checkOrSetHostName,
		a.canAddMasterCheck, a.FlightCheck.checkALLNodesTimeDiff, a.checkMasterCPUMemory, a.FlightCheck.checkAllNodesOsName,
		a.FlightCheck.checkWorkerCPUMemory, a.FlightCheck.allNodeToolCheck}

	for _, checkFunc := range checkFuncList {
		if err := checkFunc(); err != nil {
			return err
		}
	}

	return nil
}

func isMasterNodeByLabel(labels map[string]string) bool {
	isMaster := false
	for key, _ := range labels {
		// 有node-role.kubernetes.io/master或node-role.kubernetes.io/control-plane标签则判定为master节点
		if key == constants.DefaultMasterLabelKey || key == constants.DefaultMasterLabelName {
			isMaster = true
			break
		}
	}

	return isMaster
}
