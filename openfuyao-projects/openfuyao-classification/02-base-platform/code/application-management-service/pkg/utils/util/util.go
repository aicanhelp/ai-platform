/*
 *
 *  * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  * openFuyao is licensed under Mulan PSL v2.
 *  * You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  * You may obtain a copy of Mulan PSL v2 at:
 *  *          http://license.coscl.org.cn/MulanPSL2
 *  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *  * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *  * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *  * See the Mulan PSL v2 for more details.
 *
 */

// Package util include application-management-service level util function
package util

import (
	"io/ioutil"
	"strings"

	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/client-go/kubernetes"

	"application-management-service/pkg/constant"
	"application-management-service/pkg/models/helm"
	"application-management-service/pkg/utils/k8sutil"
	"application-management-service/pkg/zlog"
)

// ClearByte clear byte slice by setting every index to zero
func ClearByte(value []byte) {
	for i := range value {
		value[i] = 0
	}
}

// Contains return if string slice contains string
func Contains(s []string, str string) bool {
	for _, a := range s {
		if a == str {
			return true
		}
	}
	return false
}

// ContainsOne return if leftArr slice contains one of the element in rightArr
func ContainsOne(leftArr, rightArr []string) bool {
	set := make(map[string]struct{})
	for _, s := range leftArr {
		set[s] = struct{}{}
	}

	for _, s := range rightArr {
		if _, ok := set[s]; ok {
			return true
		}
	}
	return false
}

// ContainsAll return if outer string slice contains inner string slice
func ContainsAll(outer, inner []string) bool {
	set := make(map[string]struct{})
	for _, s := range outer {
		set[s] = struct{}{}
	}

	for _, s := range inner {
		if _, ok := set[s]; !ok {
			return false
		}
	}
	return true
}

const (
	marketplaceServiceHost = "marketplace-service-host"
	helmTimeout            = "helm-timeout"
	helmWaitCharts         = "helm-wait-charts"
	configDir              = "/etc/application-management-service/openfuyao-config"
)

// GetApplicationManagementServiceConfig parse the configmap for app-management-service configuration from the cluster
func GetApplicationManagementServiceConfig(c kubernetes.Interface) (*helm.ApplicationManagementServiceConfig, error) {
	configMap, err := k8sutil.GetConfigMap(c, constant.ApplicationManagementServiceConfigmap,
		constant.ApplicationManagementServiceDefaultNamespace)
	if err != nil {
		zlog.Warnf("failed to read config map from k8s cluster  %v", err)
		applicationManagementServiceConfig, err := getConfigFromPod()
		if err != nil {
			zlog.Warnf("failed to read config map from container %v", err)
			return nil, err
		}
		return applicationManagementServiceConfig, err
	}

	applicationManagementServiceConfig := parseConfig(configMap.Data)
	return applicationManagementServiceConfig, nil
}

func parseConfig(applicationManagementConfigMap map[string]string) *helm.ApplicationManagementServiceConfig {
	waitedCharts, err := parseStringSlice(applicationManagementConfigMap[helmWaitCharts])
	if err != nil {
		zlog.Errorf("cannot read %s, err: %v", helmWaitCharts, err)
	}

	zlog.Infof("charts required to be waited when install/upgrade/rollback: %v", waitedCharts)
	return &helm.ApplicationManagementServiceConfig{
		MarketPlaceServiceHost: applicationManagementConfigMap[marketplaceServiceHost],
		HelmTimeout:            applicationManagementConfigMap[helmTimeout],
		HelmWaitCharts:         waitedCharts,
	}
}

func parseStringSlice(content string) ([]string, error) {
	var slices []string
	err := yaml.Unmarshal([]byte(content), &slices)
	if err != nil {
		return nil, err
	}
	return slices, nil
}

func getConfigFromPod() (*helm.ApplicationManagementServiceConfig, error) {
	files, err := ioutil.ReadDir(configDir)
	if err != nil {
		return nil, err
	}
	applicationManagementServiceConfigMap := make(map[string]string)
	for _, file := range files {
		if !file.IsDir() {
			filepath := configDir + "/" + file.Name()
			content, err := ioutil.ReadFile(filepath)
			if err != nil {
				zlog.Warnf("Error reading file %s: %v\n", filepath, err)
				continue
			}
			applicationManagementServiceConfigMap[file.Name()] = string(content)
		}
	}
	return parseConfig(applicationManagementServiceConfigMap), nil
}

// TrimOpenFuyaoRolePrefix trim openfuyao- before return to frontend
func TrimOpenFuyaoRolePrefix(name string) string {
	return strings.TrimPrefix(name, constant.OpenFuyaoRolePrefix)
}
