/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package controller

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	v1 "gopkg.openfuyao.cn/kae-operator/api/v1"
	"gopkg.openfuyao.cn/kae-operator/internal/constants"
	"gopkg.openfuyao.cn/kae-operator/internal/utils"
	"gopkg.openfuyao.cn/kae-operator/internal/zlog"
)

var (
	kaePolicyManager KAEPolicyManager
	securityModes    = []string{"enforce", "audit", "warn"}
	componentPaths   = []string{
		"/opt/kae-operator/component-driver",
		"/opt/kae-operator/component-device-plugin-hpre",
	}
)

// KAEPolicyManager KAEPolicy CR 管理
type KAEPolicyManager struct {
	ctx       context.Context
	namespace string
	client    client.Client
	scheme    *runtime.Scheme

	kaeInstance *v1.KAEPolicy

	componentPaths   []string
	resources        []Resources
	resourceHandlers []resourceHandler
	componentNames   []string
	index            int

	hasKAENodes  bool
	hasNFDLabels bool

	kaeNodes  int
	hpreNodes int
	secNodes  int
	zipNodes  int
}

func (m *KAEPolicyManager) deployComponent() (v1.State, error) {
	result := v1.Ready
	if m.index >= len(m.resourceHandlers) {
		return result, fmt.Errorf("policy manager index: %d has greater than or equal to resourceHandler num: %d",
			m.index, len(m.resourceHandlers))
	}
	for _, handlerFunc := range m.resourceHandlers[m.index] {
		state, err := handlerFunc(*m)
		if err != nil {
			return state, err
		}
		if state != v1.Ready {
			result = state
		}
	}
	m.index++
	return result, nil
}

func (m *KAEPolicyManager) deployComplete() bool {
	return m.index == len(m.resourceHandlers)
}

func (m *KAEPolicyManager) isComponentEnabled(componentName string) bool {
	kaePolicySpec := &m.kaeInstance.Spec

	switch componentName {
	case constants.ComponentDriver:
		return utils.IsDriverEnabled(kaePolicySpec.Driver)
	case constants.ComponentDriverPluginHpre:
		return utils.IsDevicePluginEnabledHPRE(kaePolicySpec.DevicePlugin)
	case constants.ComponentDriverPluginSec:
		return utils.IsDevicePluginEnabledSEC(kaePolicySpec.DevicePlugin)
	case constants.ComponentDriverPluginZip:
		return utils.IsDevicePluginEnabledZIP(kaePolicySpec.DevicePlugin)
	default:
		zlog.Warnf("invalid state file: %s", componentName)
		return false
	}
}

func (m *KAEPolicyManager) init(ctx context.Context, reconciler *KAEPolicyReconciler,
	kaePolicy *v1.KAEPolicy) error {
	m.initManagerParam(ctx, reconciler, kaePolicy)

	if len(m.resourceHandlers) == 0 {
		kaePolicyManager.namespace = os.Getenv("OPERATOR_NAMESPACE")
		if kaePolicyManager.namespace == "" {
			zlog.Warnf("operatorNamespace has empty")
		}

		err := addComponents(m)
		if err != nil {
			return err
		}

	}

	if utils.IsPSAEnabled(kaePolicy.Spec.PSA) {
		zlog.Infof("Pod Security is enabled. Adding labels to KAE Operator %s", m.namespace)
		err := m.setPSALabelForNamespace()
		if err != nil {
			return err
		}
		zlog.Infof("Pod Security Admission labels added to KAE Operator %s", m.namespace)
	}

	return m.labelKAENodes()
}

func (m *KAEPolicyManager) initManagerParam(ctx context.Context, reconciler *KAEPolicyReconciler,
	kaePolicy *v1.KAEPolicy) {
	m.kaeInstance = kaePolicy
	m.ctx = ctx
	m.index = 0
	m.kaeNodes = 0
	m.zipNodes = 0
	m.secNodes = 0
	m.hpreNodes = 0
	m.componentPaths = getComponentPaths("default")
	m.client = reconciler.Client
	m.scheme = reconciler.Scheme
}

// getComponentPaths 根据条件获取部署文件路径，condition 预留参数
func getComponentPaths(condition string) []string {
	return componentPaths
}

func addComponents(n *KAEPolicyManager) error {
	for _, path := range n.componentPaths {
		if err := addComponent(n, path); err != nil {
			return err
		}
	}
	return nil
}

func addComponent(n *KAEPolicyManager, path string) error {
	// 获取资源列表和资源处理函数
	res, handler, err := getResourcesAndHandlers(path)
	if err != nil {
		return err
	}

	n.resources = append(n.resources, res)
	n.resourceHandlers = append(n.resourceHandlers, handler)
	n.componentNames = append(n.componentNames, filepath.Base(path))
	return nil
}
