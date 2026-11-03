/*
 *
 *  * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

package ray

import (
	"context"

	rayversioned "github.com/ray-project/kuberay/ray-operator/pkg/client/clientset/versioned"
	"gorm.io/gorm"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	"openfuyao-rayservice/pkg/constant"
	"openfuyao-rayservice/pkg/utils/httputil"
	"openfuyao-rayservice/pkg/zlog"
)

// Operation all operations for ray
type Operation interface {
	MethodInitOperation()
	// Method For Overview
	MethodAllOverview() (*httputil.ResponseJson, int)
	// Method For Crd
	MethodListMetaCrd(crd string) (*httputil.ResponseJson, int)
	MethodCreateCrd(crd, data, createBy string, isStart bool) (*httputil.ResponseJson, int)
	MethodListCrd(crd string, req *CrdListRequest) (*httputil.ResponseJson, int)
	MethodDeleteCrd(crd, ns, name string) (*httputil.ResponseJson, int)
	MethodUpdateCrd(crd, ns, data, name string) (*httputil.ResponseJson, int)
	MethodGetCrd(crd, ns, name string) (*httputil.ResponseJson, int)
	MethodRestartCrd(crd, ns, name string) (*httputil.ResponseJson, int)
	MethodStopCrd(crd, ns, name string) (*httputil.ResponseJson, int)
}

type rayClient struct {
	kubeConfig    *rest.Config
	dynamicClient *dynamic.DynamicClient
	clientset     kubernetes.Interface
	rayClientset  rayversioned.Interface
	dbConnection  *gorm.DB
}

// NewRayOperation  ray operation requires client for kubernetes resource operation
func NewRayOperation(kubeConfig *rest.Config, ctx context.Context) (Operation, error) {
	dynamicClient, err := dynamic.NewForConfig(kubeConfig)

	if err != nil {
		zlog.Errorf("error creating dynamic client, err: %v", err)
		return nil, err
	}

	clientset, err := kubernetes.NewForConfig(kubeConfig)
	if err != nil {
		zlog.Errorf("error creating client set, err: %v", err)
		return nil, err
	}

	rayClientset, err := rayversioned.NewForConfig(kubeConfig)
	if err != nil {
		zlog.Errorf("error creating ray client set, err: %v", err)
		return nil, err
	}

	return &rayClient{
		kubeConfig:    kubeConfig,
		dynamicClient: dynamicClient,
		clientset:     clientset,
		rayClientset:  rayClientset,
		dbConnection:  ctx.Value(constant.ContextDBKey).(*gorm.DB),
	}, nil
}
