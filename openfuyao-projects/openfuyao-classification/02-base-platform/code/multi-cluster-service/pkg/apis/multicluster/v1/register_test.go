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

package v1

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/emicklei/go-restful/v3"
	"github.com/karmada-io/karmada/pkg/generated/clientset/versioned"
	"github.com/stretchr/testify/assert"
	"k8s.io/client-go/kubernetes"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"openfuyao.com/multi-cluster-service/pkg/apis/multicluster/v1/runtime"
	"openfuyao.com/multi-cluster-service/pkg/utils"
)

func TestClusterEdition(t *testing.T) {
	type args struct {
		ws *restful.WebService
		h  *handler
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: "test1",
			args: args{
				ws: runtime.NewWebService(),
				h:  NewHandler(),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clusterEdition(tt.args.ws, tt.args.h)
		})
	}
}

func TestClusterInspectUsage(t *testing.T) {
	type args struct {
		ws *restful.WebService
		h  *handler
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: "test2",
			args: args{
				ws: runtime.NewWebService(),
				h:  NewHandler(),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clusterInspectUsage(tt.args.ws, tt.args.h)
		})
	}
}

func TestClusterRegistration(t *testing.T) {
	type args struct {
		ws *restful.WebService
		h  *handler
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: "test3",
			args: args{
				ws: runtime.NewWebService(),
				h:  NewHandler(),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clusterRegistration(tt.args.ws, tt.args.h)
		})
	}
}

func TestClusterUnregistration(t *testing.T) {
	type args struct {
		ws *restful.WebService
		h  *handler
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: "test4",
			args: args{
				ws: runtime.NewWebService(),
				h:  NewHandler(),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clusterUnregistration(tt.args.ws, tt.args.h)
		})
	}
}

func TestKubeconfigRetrieval(t *testing.T) {
	type args struct {
		ws *restful.WebService
		h  *handler
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: "test5",
			args: args{
				ws: runtime.NewWebService(),
				h:  NewHandler(),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			kubeconfigRetrieval(tt.args.ws, tt.args.h)
		})
	}
}

func TestPolicyRuleUpdation(t *testing.T) {
	type args struct {
		ws *restful.WebService
		h  *handler
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: "test6",
			args: args{
				ws: runtime.NewWebService(),
				h:  NewHandler(),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			policyRuleUpdation(tt.args.ws, tt.args.h)
		})
	}
}

func TestResourceUsageInspection(t *testing.T) {
	type args struct {
		ws *restful.WebService
		h  *handler
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: "test7",
			args: args{
				ws: runtime.NewWebService(),
				h:  NewHandler(),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resourceUsageInspection(tt.args.ws, tt.args.h)
		})
	}
}

func TestSayHello(t *testing.T) {
	type args struct {
		ws *restful.WebService
		h  *handler
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: "test8",
			args: args{
				ws: runtime.NewWebService(),
				h:  NewHandler(),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sayHello(tt.args.ws, tt.args.h)
		})
	}
}

func TestAddToContainer(t *testing.T) {
	type args struct {
		container *restful.Container
		client    client.Client
	}
	tests := []struct {
		name    string
		args    args
		wantErr assert.ErrorAssertionFunc
	}{
		{
			name: "succeed",
			args: args{
				container: restful.NewContainer(),
				client:    nil,
			},
			wantErr: assert.NoError,
		},
		{
			name: "fail",
			args: args{
				container: restful.NewContainer(),
				client:    nil,
			},
			wantErr: assert.NoError,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patch1 := gomonkey.ApplyFunc(utils.CreateKarmadaClusterClient,
				func() (*kubernetes.Clientset, *versioned.Clientset, error) {
					t.Log("utils.CreateKarmadaClusterClient is patched")
					return nil, nil, nil
				})
			patch2 := gomonkey.ApplyMethod(reflect.TypeOf(&restful.Container{}),
				"Add", func(c *restful.Container, _ *restful.WebService) *restful.Container {
					t.Log("container.Add() was patched and skipped.")
					return c
				})
			defer patch1.Reset()
			defer patch2.Reset()
			tt.wantErr(t, AddToContainer(tt.args.container, tt.args.client),
				fmt.Sprintf("AddToContainer(%v, %v)", tt.args.container, tt.args.client))
		})
	}
}
