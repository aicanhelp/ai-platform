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
* Author: hongchi hu
* Date: 2025-5-25
 */

package pod

import (
	"fmt"
	"reflect"
	"strings"

	"k8s.io/api/core/v1"
	"k8s.io/klog/v2"

	"openfuyao.com/colocation-management/pkg/common"
)

// 基础Mutator结构体
type baseQoSMutator struct{}

// ensurePriorityClass 添加qos level 标签
func (m *baseQoSMutator) ensurePriorityClass(pod *v1.Pod) error {
	annotations := pod.GetAnnotations()
	if annotations == nil {
		return fmt.Errorf("pod annotations are nil")
	}

	qosLevel, ok := annotations[common.QosClassAnnotationKey]
	if !ok {
		klog.Info("qoS level annotation not found in pod")
		return fmt.Errorf("qoS level annotation not found")
	}

	// If PriorityClassName already exists, validate it matches the qosLevel
	if pod.Spec.PriorityClassName != "" {
		// Extract the suffix after "priority-" (e.g., "priority-hls" -> "hls")
		existingLevel := strings.TrimPrefix(pod.Spec.PriorityClassName, "priority-")
		if existingLevel != strings.ToLower(qosLevel) {
			return fmt.Errorf("existing PriorityClassName conflicts with QoS level (current: %s, expected: %s)",
				existingLevel, strings.ToLower(qosLevel))
		}
		// If they match, no action needed
		return nil
	}

	// Clean up legacy priority fields if any
	if err := m.cleanupPriorityField(pod); err != nil {
		return fmt.Errorf("failed to clean up priority fields: %w", err)
	}

	// Set new PriorityClassName
	pod.Spec.PriorityClassName = "priority-" + strings.ToLower(qosLevel)
	klog.Infof("Setting PriorityClassName to %s for pod %s", pod.Spec.PriorityClassName, pod.Name)
	return nil
}

// cleanupPriorityField 清理PodSpec中的priority字段
func (m *baseQoSMutator) cleanupPriorityField(pod *v1.Pod) error {
	// 使用反射检查priority字段是否存在
	priorityField := reflect.ValueOf(&pod.Spec).Elem().FieldByName("Priority")
	if !priorityField.IsValid() {
		return nil // 字段不存在（Kubernetes版本兼容）
	}

	// 如果priority有设置值（非nil且非零值）
	if !priorityField.IsNil() && !priorityField.IsZero() {
		klog.InfoS("Clearing deprecated priority field",
			"pod", klog.KObj(pod),
			"previousPriority", priorityField.Elem().Int())

		// 重置priority字段
		priorityField.Set(reflect.Zero(priorityField.Type()))
	}
	return nil
}

// HLS Mutator实现
type hlsQoSMutator struct {
	baseQoSMutator
}

func (m *hlsQoSMutator) MutatePod(pod *v1.Pod) (bool, error) {
	if err := m.ensurePriorityClass(pod); err != nil {
		klog.Infof("Failed to ensure priority class for HLS pod %s/%s: %v", pod.Namespace, pod.Name, err)
		return false, fmt.Errorf("pod priority class error: %v", err)
	}

	// 添加HLS特有逻辑
	klog.Infof("Finish setting PriorityClassName for hls pod")
	return true, nil
}

// NewHLSPodQoSMutator HLS pod
func NewHLSPodQoSMutator() *hlsQoSMutator {
	return &hlsQoSMutator{}
}

// LS Mutator实现
type lsQoSMutator struct {
	baseQoSMutator
}

func (m *lsQoSMutator) MutatePod(pod *v1.Pod) (bool, error) {
	if err := m.ensurePriorityClass(pod); err != nil {
		klog.Infof("Failed to ensure priority class for LS pod %s/%s: %v", pod.Namespace, pod.Name, err)
		return false, fmt.Errorf("LS pod priority class error: %v", err)
	}

	// 添加LS特有逻辑
	m.addRubikQuotaTurboAnnotation(pod)

	return true, nil
}

// addRubikQuotaTurboAnnotation 为LS pod添加注解
func (m *lsQoSMutator) addRubikQuotaTurboAnnotation(pod *v1.Pod) {
	defer func() {
		if r := recover(); r != nil {
			klog.Warningf("Panic while adding rubik quota-turbo annotation to LS pod %s/%s: %v", pod.Namespace, pod.Name, r)
		}
	}()

	if pod.Annotations == nil {
		pod.Annotations = make(map[string]string)
	}

	pod.Annotations[common.RubikQuotaTurboAnnotation] = "true"
	klog.Infof("Successfully added quota-turbo annotation to LS pod %s/%s", pod.Namespace, pod.Name)
}

// NewLSPodQoSMutator LS pod
func NewLSPodQoSMutator() *lsQoSMutator {
	return &lsQoSMutator{}
}

// BE Mutator实现
type beQoSMutator struct {
	baseQoSMutator
}

func (m *beQoSMutator) MutatePod(pod *v1.Pod) (bool, error) {
	if err := m.ensurePriorityClass(pod); err != nil {
		klog.Infof("Failed to ensure priority class for BE pod %s/%s: %v", pod.Namespace, pod.Name, err)
		return false, fmt.Errorf("LS pod priority class error: %v", err)
	}

	// 添加BE特有逻辑
	return true, nil
}

// NewBEPodQoSMutator be pod
func NewBEPodQoSMutator() *beQoSMutator {
	return &beQoSMutator{}
}
