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

// Package volcano config
package volcano

import (
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	"volcano-config-service/pkg/utils/httputil"
	"volcano-config-service/pkg/zlog"
)

// Operation all operations for volcano
type Operation interface {
	MethodNumaAwareGet() (*httputil.ResponseJson, int)
	MethodNumaAwarePut(status string) (*httputil.ResponseJson, int)
	MethodNumaDistanceGet() (*httputil.ResponseJson, int)
	MethodNumaDistancePut(status string) (*httputil.ResponseJson, int)
	MethodNumaFastGet() (*httputil.ResponseJson, int)
	MethodNumaFastPut(status string) (*httputil.ResponseJson, int)
	MethodBinpackGet() (*httputil.ResponseJson, int)
	MethodBinpackPut(status string) (*httputil.ResponseJson, int)
	MethodBalanceGet() (*httputil.ResponseJson, int)
	MethodBalancePut(status string) (*httputil.ResponseJson, int)
	MethodDrfGet() (*httputil.ResponseJson, int)
	MethodDrfPut(status string) (*httputil.ResponseJson, int)
	MethodGangGet() (*httputil.ResponseJson, int)
	MethodGangPut(status string) (*httputil.ResponseJson, int)
	MethodPriorityGet() (*httputil.ResponseJson, int)
	MethodPriorityPut(status string) (*httputil.ResponseJson, int)
	MethodDefault() (*httputil.ResponseJson, int)
	MethodNodeList() (*httputil.ResponseJson, int)
	MethodNodeListByName(name string) (*httputil.ResponseJson, int)
	MethodCRInfo(name string) (*httputil.ResponseJson, int)
	MethodUuidToName(uuid string) (*httputil.ResponseJson, int)
	MethodSchedulerGet() (*httputil.ResponseJson, int)
}

type volcanoClient struct {
	kubeConfig    *rest.Config
	dynamicClient *dynamic.DynamicClient
	clientset     *kubernetes.Clientset
}

// NewVolcanoOperation  volcano operation requires client for kubernetes resource operation
func NewVolcanoOperation(kubeConfig *rest.Config) (Operation, error) {
	dynamicClient, err := dynamic.NewForConfig(kubeConfig)
	if err != nil {
		zlog.Error("error creating dynamic client, err: %v", err)
		return nil, err
	}

	clientset, err := kubernetes.NewForConfig(kubeConfig)
	if err != nil {
		zlog.Error("error creating client set, err: %v", err)
		return nil, err
	}

	return &volcanoClient{
		kubeConfig:    kubeConfig,
		dynamicClient: dynamicClient,
		clientset:     clientset,
	}, nil
}
