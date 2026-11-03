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

// Package utils contains a collection of utility functions and helpers that provide
// common functionalities useful across the entire engineering project.
package utils

import (
	"os"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/yaml"

	"openfuyao.com/multi-cluster-service/pkg/zlog"
)

// ConfToUnstructure converts any object (here api.Config structure) to its JSON representation.
func ConfToUnstructure(obj interface{}) ([]byte, error) {
	uncastObj, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
	if err != nil {
		zlog.LogErrorf("Error converting object to unstructure : %v", err)
		return nil, err
	}

	resourceObj := &unstructured.Unstructured{Object: uncastObj}
	kubeconfigJSON, err := resourceObj.MarshalJSON()
	if err != nil {
		zlog.LogErrorf("Error converting unstructure to json : %v", err)
		return nil, err
	}

	return kubeconfigJSON, nil
}

// HostConfToJson converts host config yaml to its json representation.
func HostConfToJson() ([]byte, error) {
	configBytes, err := os.ReadFile("/root/.kube/config")
	if err != nil {
		zlog.LogErrorf("Error retrieving host outcluster kubeconfig : %v", err)
		return nil, err
	}
	configdata, err := yaml.YAMLToJSON(configBytes)
	if err != nil {
		zlog.LogErrorf("Error Generating kubeconfig data : %v", err)
		return nil, err
	}

	return configdata, nil
}
