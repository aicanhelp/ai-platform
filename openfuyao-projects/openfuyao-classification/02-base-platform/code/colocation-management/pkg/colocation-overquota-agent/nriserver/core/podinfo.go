/*
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
* Author: YuXiang Guo
* Date: 2024-11-25
 */

package core

import (
	"github.com/containerd/nri/pkg/api"
	"k8s.io/klog/v2"

	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/runnermanager"
	"openfuyao.com/colocation-management/pkg/utils"
)

// PodSandboxMetadata is the metadata of a pod sandbox.
type PodSandboxMetadata struct {
	Name      string
	UID       string
	Namespace string
}

// PodInfo is the information of a pod.
type PodInfo struct {
	PodRequest   PodRequest
	PodResponese PodResponse
}

// Admit checks whether the pod is admitted.
func (p *PodInfo) Admit() AdmitResult {
	return CheckAnnotationAdmit(p.PodRequest.PodSandboxConfig.Annotations)
}

// Update updates the pod sandbox.
func (p *PodInfo) Update() {
	if p.PodResponese.Resources.CPUShares != nil {
		err := runnermanager.GetRunner().
			SetCpuShares(p.PodRequest.PodSandboxConfig.Linux.CgroupParent, p.PodResponese.Resources)
		if err != nil {
			klog.Errorf("fail setting cpu shares: %v", err)
		}
	}
	if p.PodResponese.Resources.CFSQuota != nil {
		err := runnermanager.GetRunner().
			SetCpuQuota(p.PodRequest.PodSandboxConfig.Linux.CgroupParent, p.PodResponese.Resources)
		if err != nil {
			klog.Errorf("fail setting cpu quota: %v", err)
		}
	}
	if p.PodResponese.Resources.CPUSet != nil {
		err := runnermanager.GetRunner().
			SetCpuSet(p.PodRequest.PodSandboxConfig.Linux.CgroupParent, p.PodResponese.Resources)
		if err != nil {
			klog.Errorf("fail setting cpu set: %v", err)
		}
	}
	if p.PodResponese.Resources.MemoryLimit != nil {
		err := runnermanager.GetRunner().
			SetMemoryLimit(p.PodRequest.PodSandboxConfig.Linux.CgroupParent, p.PodResponese.Resources)
		if err != nil {
			klog.Errorf("fail setting memory limit: %v", err)
		}
	}
}

// UpdateContainer updates the container.
func (p *PodInfo) UpdateContainer() (*api.ContainerAdjustment, *api.ContainerUpdate, error) {
	return nil, nil, nil
}

// Deny denies the pod.
func (p *PodInfo) Deny() AdmitResult {
	// to record reason and status
	return AdmitResult{Admit: false}
}

// PodRequest is the request of a pod.
type PodRequest struct {
	PodMeta                       PodSandboxMetadata
	PodSandboxConfig              utils.PodSandboxConfig
	ExtendedResourceConfiguration *ExtendedResourceConfiguration
}

// PodResponse is the response of a pod.
type PodResponse struct {
	Resources        utils.LinuxContainerResources
	AddPodAnnotation map[string]string
}

// ConvertNRIRawPod2PodInfo converts the NRI raw pod to pod info.
func (p *PodInfo) ConvertNRIRawPod2PodInfo(pod *api.PodSandbox) {
	p.PodRequest.PodMeta = PodSandboxMetadata{
		Name:      pod.GetName(),
		UID:       pod.GetUid(),
		Namespace: pod.GetNamespace(),
	}
	p.PodRequest.PodSandboxConfig = utils.PodSandboxConfig{
		Annotations: pod.GetAnnotations(),
		Labels:      pod.GetLabels(),
		Linux: &utils.LinuxPodSandboxConfig{
			CgroupParent: pod.GetLinux().GetCgroupParent(),
		},
	}
	cfg, err := GetExtendedResourceConfiguration(pod.GetAnnotations())
	if err != nil {
		klog.Errorf("GetExtendedResourceConfiguration error %v", err)
	}
	if cfg != nil {
		p.PodRequest.ExtendedResourceConfiguration = cfg
	}

}
