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

package pod

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"regexp"
	"strings"

	"k8s.io/api/core/v1"
	"k8s.io/klog/v2"

	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/core"
	"openfuyao.com/colocation-management/pkg/common"
)

// PodResourceMutator Pod资源扩Mutator
type PodResourceMutator struct {
}

// ResourceNameList 资源名称列表
type ResourceNameList struct {
	ResourceNameList []v1.ResourceName
}

// NewResourceNameList 根据extender来获取资源名称
func NewResourceNameList(extender bool) *ResourceNameList {
	var resourceNameList []v1.ResourceName
	if extender {
		return &ResourceNameList{
			ResourceNameList: append(resourceNameList, common.ExtenderResourceCPU, common.ExtenderResourceMemory),
		}
	}
	return &ResourceNameList{
		ResourceNameList: append(resourceNameList, v1.ResourceCPU, v1.ResourceMemory),
	}
}

// MutatePod 设置 Pod 的扩展资源配置注解
func (*PodResourceMutator) MutatePod(pod *v1.Pod) (allowed bool, err error) {

	// 离线pod container为空 拒绝
	if len(pod.Spec.Containers) == 0 {
		return false, fmt.Errorf("pod has no containers")
	}

	containerResourceSpec := map[string]core.ExtendedResourceItem{}
	configuration := core.ExtendedResourceConfiguration{}

	// init容器写入注解 逻辑下放
	if len(pod.Spec.InitContainers) != 0 {
		initContainerResourceSpec := map[string]core.ExtendedResourceItem{}
		for _, initContainer := range pod.Spec.InitContainers {
			resources, err := getExtenderResoruces(&initContainer)
			if err != nil {
				return false, err
			}
			initContainerResourceSpec[initContainer.Name] = *resources
		}
		configuration.InitContainers = initContainerResourceSpec
	}

	// 多容器 不允许同时声明普通资源和扩展超分资源
	for _, container := range pod.Spec.Containers {
		resoruces, err := getExtenderResoruces(&container)
		if err != nil {
			return false, err
		}
		containerResourceSpec[container.Name] = *resoruces
	}
	configuration.Containers = containerResourceSpec

	// setAnnotations
	err = setExtenderResourceAnnotationlist(pod, &configuration)
	if err != nil {
		return false, err
	}

	return true, nil
}

func getExtenderResoruces(container *v1.Container) (*core.ExtendedResourceItem, error) {
	resourceNameList := NewResourceNameList(true)

	item := &core.ExtendedResourceItem{
		Requests: v1.ResourceList{},
		Limits:   v1.ResourceList{},
	}
	for _, resourceName := range resourceNameList.ResourceNameList {
		if requset, ok := container.Resources.Requests[resourceName]; ok {
			item.Requests[resourceName] = requset
		}
		if limit, ok := container.Resources.Limits[resourceName]; ok {
			item.Limits[resourceName] = limit
		}
	}

	return checkExtenderResource(container, item)

}

func validateExtenderItem(item *core.ExtendedResourceItem) error {
	if _, ok := item.Requests[common.ExtenderResourceCPU]; ok {
		quantity := item.Requests[common.ExtenderResourceCPU]
		vaild := validateResource(common.ExtenderResourceCPU, quantity.String())
		if vaild == false {
			return fmt.Errorf("container requests ExtenderCPU validate error")
		}
	}
	if _, ok := item.Limits[common.ExtenderResourceCPU]; ok {
		quantity := item.Limits[common.ExtenderResourceCPU]
		vaild := validateResource(common.ExtenderResourceCPU, quantity.String())
		if vaild == false {
			return fmt.Errorf("container limit ExtenderCPU validate error")
		}
	}
	if _, ok := item.Requests[common.ExtenderResourceMemory]; ok {
		quantity := item.Requests[common.ExtenderResourceMemory]
		vaild := validateResource(common.ExtenderResourceMemory, quantity.String())
		if vaild == false {
			return fmt.Errorf("container requests ExtenderMemory validate error")
		}
	}
	if _, ok := item.Limits[common.ExtenderResourceMemory]; ok {
		quantity := item.Limits[common.ExtenderResourceMemory]
		vaild := validateResource(common.ExtenderResourceMemory, quantity.String())
		if vaild == false {
			return fmt.Errorf("container limits ExtenderMemory validate error")
		}
	}
	return nil
}

func validateResource(resourceName v1.ResourceName, value string) bool {
	// CPU 使用整数
	cpuRegex := `^\d+k?$`
	memRegex := `^\d+(Ki|K|Mi|M|Gi|G|Ti|T|Pi|P|Ei|E)?$`

	if resourceName == common.ExtenderResourceCPU {
		matchedCpu, err := regexp.MatchString(cpuRegex, value)
		if err != nil {
			klog.Errorf("regexp.MatchString error: %v", err)
			return false
		}
		return matchedCpu

	}

	// 如果是内存请求
	if resourceName == common.ExtenderResourceMemory {
		matchedMemory, err := regexp.MatchString(memRegex, value)
		if err != nil {
			klog.Errorf("regexp.MatchString error: %v", err)
			return false
		}
		return matchedMemory

	}
	return false

}

func checkExtenderResource(container *v1.Container,
	item *core.ExtendedResourceItem) (*core.ExtendedResourceItem, error) {

	err := validateExtenderItem(item)
	if err != nil {
		return nil, err
	}

	orignResourceList := NewResourceNameList(false)
	// 如果存在原生资源声明 则拒绝
	for _, resourceName := range orignResourceList.ResourceNameList {
		if _, ok := container.Resources.Requests[resourceName]; ok {
			return nil, fmt.Errorf("container requests does not allow to mix the native and extended resources")
		}
		if _, ok := container.Resources.Requests[resourceName]; ok {
			return nil, fmt.Errorf("container limits does not allow to mix the native and extended resources")
		}
	}

	// item check
	// 如果使用了BE声明+离线声明  容器必须声明超分资源
	if len(item.Requests) == 0 {
		if len(item.Limits) == 0 {
			return nil, fmt.Errorf("container has no be-resource  request")
		} else {
			// 容器声明了limits  但是没有requests 按k8s原生资源规则  补齐
			request := v1.ResourceList{}

			for resourceName, resourcevalue := range item.Limits {
				request[resourceName] = resourcevalue
				item.Requests = item.Limits
			}
			item.Requests = request
		}
	}

	return item, nil
}

func setExtenderResourceAnnotationlist(pod *v1.Pod, configuration *core.ExtendedResourceConfiguration) error {
	annotations := pod.GetAnnotations()
	oldConfiguration := &core.ExtendedResourceConfiguration{}
	if len(annotations) == 0 {
		annotations = make(map[string]string)
	}

	spec, ok := annotations[common.ExtenderResourceConfigurationAnnotation]
	if !ok {
		// 注解不存在
		finalAnnotation, err := setExtenderResourceAnnotation(annotations, configuration)
		if err != nil {
			return err
		}
		pod.SetAnnotations(finalAnnotation)
		return nil
	}

	// 注解存在的话 需要对比看是否需要替换
	err := json.Unmarshal([]byte(spec), oldConfiguration)
	if err != nil {
		return err
	}
	if reflect.DeepEqual(oldConfiguration, configuration) {
		// 如果完全一致 继续沿用即可 不需要修改
		return nil
	}

	finalAnnotation, err := setExtenderResourceAnnotation(annotations, configuration)
	if err != nil {
		return err
	}
	pod.SetAnnotations(finalAnnotation)
	return nil
}

func setExtenderResourceAnnotation(annotations map[string]string,
	configuration *core.ExtendedResourceConfiguration) (map[string]string, error) {
	marshal, err := json.Marshal(configuration)
	if err != nil {
		return annotations, err
	}
	if configuration == nil { // 增加判空处理
		return annotations, errors.New("configuration is nil")
	}
	if annotations == nil {
		return annotations, errors.New("annotations is nil")
	}
	annotations[common.ExtenderResourceConfigurationAnnotation] = string(marshal)
	annotations[common.PriorityAnnotationKey] = "true"
	return annotations, nil
}

// NewBEPodResourceMutator BE pod 资源接口
func NewBEPodResourceMutator() PodMutator {
	return &PodResourceMutator{}
}

// VerifyResourceEquality 根据 podtype 验证资源规则
func VerifyResourceEquality(pod *v1.Pod, podType string) (bool, error) {
	// 合并所有容器（包括 InitContainers）
	containers := append([]v1.Container{}, pod.Spec.Containers...)
	if pod.Spec.InitContainers != nil {
		containers = append(containers, pod.Spec.InitContainers...)
	}

	// 验证所有容器
	for _, container := range containers {
		requests := container.Resources.Requests
		limits := container.Resources.Limits

		// 执行三项验证
		cpuEqual, cpuEqualErr := verifyCPUEqual(container.Name, requests, limits)
		if cpuEqualErr != nil {
			klog.Info("CPU equality check fail")
		}

		memEqual, memEqualErr := verifyMemoryEqual(container.Name, requests, limits)
		if memEqualErr != nil {
			klog.Info("Memory equality check fail")
		}

		cpuInteger, cpuIntegerErr := verifyCPUInteger(container.Name, requests)
		if cpuIntegerErr != nil {
			klog.Info("CPU integer check fail")
		}

		// 根据 podType 决定验证策略
		switch podType {
		case "hls":
			// HLS Pod 需要全部条件满足
			if cpuEqualErr != nil || memEqualErr != nil || cpuIntegerErr != nil {
				return false, fmt.Errorf("HLS pod validation failed: %v, %v, %v",
					cpuEqualErr, memEqualErr, cpuIntegerErr)
			}
		case "ls":
			// LS Pod 需要不同时满足三个条件
			if cpuEqual && memEqual && cpuInteger {
				return false, fmt.Errorf("LS pod cannot satisfy all three conditions simultaneously")
			}
		default:
			return false, fmt.Errorf("invalid pod type: %s, must be 'hls' or 'ls'", podType)
		}
	}

	return true, nil
}

// 1. 验证 CPU requests 和 limits 相等
func verifyCPUEqual(containerName string, requests, limits v1.ResourceList) (bool, error) {
	reqCPU, reqExists := requests[v1.ResourceCPU]
	limCPU, limExists := limits[v1.ResourceCPU]

	if !reqExists || !limExists {
		return false, fmt.Errorf("container %s: CPU must have both requests and limits", containerName)
	}

	if !reqCPU.Equal(limCPU) {
		return false, fmt.Errorf("container %s: CPU requests (%s) and limits (%s) must be equal",
			containerName, reqCPU.String(), limCPU.String())
	}

	return true, nil
}

// 2. 验证 Memory requests 和 limits 相等
func verifyMemoryEqual(containerName string, requests, limits v1.ResourceList) (bool, error) {
	reqMem, reqExists := requests[v1.ResourceMemory]
	limMem, limExists := limits[v1.ResourceMemory]

	if !reqExists || !limExists {
		return false, fmt.Errorf("container %s: Memory must have both requests and limits", containerName)
	}

	if !reqMem.Equal(limMem) {
		return false, fmt.Errorf("container %s: Memory requests (%s) and limits (%s) must be equal",
			containerName, reqMem.String(), limMem.String())
	}

	return true, nil
}

// 3. 验证 CPU 是整数值
func verifyCPUInteger(containerName string, requests v1.ResourceList) (bool, error) {
	reqCPU, exists := requests[v1.ResourceCPU]
	if !exists {
		return false, fmt.Errorf("container %s: CPU requests not found", containerName)
	}

	var millicore int64
	millicore = 1000
	if reqCPU.MilliValue()%millicore != 0 {
		return false, fmt.Errorf("container %s: CPU requests (%s) must be an integer",
			containerName, reqCPU.String())
	}

	return true, nil
}

// VerifyPodResources 统一的验证函数，验证Pod资源使用是否符合指定类型要求
// podType:
//
//	"be" - 必须使用extend资源且不能使用raw资源
//	"ls" - 必须使用raw资源且不能使用extend资源
//	"hls" - 必须使用raw资源且不能使用extend资源
func VerifyPodResources(rawPod *v1.Pod, podType string) error {
	// 检查是否使用了 extend 资源
	hasExtend, err := VerifyResource(rawPod, "extend")
	if err != nil {
		return fmt.Errorf("resource verification failed: %v", err)
	}

	// 检查是否使用了 raw 资源
	hasRaw, err := VerifyResource(rawPod, "raw")
	if err != nil {
		return fmt.Errorf("resource verification failed: %v", err)
	}

	switch podType {
	case "be":
		if !hasExtend {
			return fmt.Errorf("BE pod must use extend resources")
		}
		if hasRaw {
			return fmt.Errorf("BE pod cannot use raw resources")
		}

	case "ls", "hls":
		if hasExtend {
			return fmt.Errorf("%s pod cannot use extend resources", strings.ToUpper(podType))
		}
		if !hasRaw {
			return fmt.Errorf("%s pod must use raw resources", strings.ToUpper(podType))
		}

	default:
		// 如果是其他类型就放行
		return nil
	}

	return nil
}

// VerifyResource 验证Pod是否使用了指定类型的资源
// resourceType: "extend" - 验证扩展资源; "raw" - 验证原生资源
func VerifyResource(pod *v1.Pod, resourceType string) (bool, error) {
	// 验证参数合法性
	var isExtend bool
	switch resourceType {
	case "extend":
		isExtend = true
	case "raw":
		isExtend = false
	default:
		return false, errors.New("invalid resource type, must be 'extend' or 'raw'")
	}

	// 获取要检查的资源列表
	resourceNameList := NewResourceNameList(isExtend)

	// 检查所有容器(包括init容器)
	containers := append([]v1.Container{}, pod.Spec.Containers...)
	if pod.Spec.InitContainers != nil {
		containers = append(containers, pod.Spec.InitContainers...)
	}

	for _, container := range containers {
		for _, resourceName := range resourceNameList.ResourceNameList {
			if _, ok := container.Resources.Requests[resourceName]; ok {
				return true, nil
			}
			if _, ok := container.Resources.Limits[resourceName]; ok {
				return true, nil
			}
		}
	}

	return false, nil
}
