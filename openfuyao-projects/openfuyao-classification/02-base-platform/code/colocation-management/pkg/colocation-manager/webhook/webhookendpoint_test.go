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
* Date: 2025-09-08
 */

package webhook

import (
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"

	"openfuyao.com/colocation-management/pkg/utils"
)

// 创建模拟的 kubeClient 包装器
type mockKubeClient struct {
	client    *fake.Clientset
	returnErr error
}

// 定义一个可替换的接口，而不是直接替换函数
var (
	getKubeClientFunc = func() (interface{}, error) {
		return utils.InitKubeClient()
	}
)

// TestWaitForSvcRunning 测试主函数
func TestWaitForSvcRunning(t *testing.T) {
	// 保存原始 Endpoint 值
	originalEndpoint := Endpoint
	defer func() {
		Endpoint = originalEndpoint // 恢复原始值
	}()

	var tests []struct {
		name            string
		namespace       string
		serviceName     string
		initialEndpoint string
		mockBehavior    func(*fake.Clientset)
		clientError     error
		expectError     bool
		expectEndpoint  string
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			Endpoint = tt.initialEndpoint

			if tt.initialEndpoint != "" {
				originalGetFunc := getKubeClientFunc
				getKubeClientFunc = func() (interface{}, error) {
					return nil, nil
				}
				defer func() {
					getKubeClientFunc = originalGetFunc
				}()
			} else {
				var fakeClient *fake.Clientset
				if tt.mockBehavior != nil {
					fakeClient = fake.NewSimpleClientset()
					tt.mockBehavior(fakeClient)
				} else {
					fakeClient = fake.NewSimpleClientset()
				}

				originalGetFunc := getKubeClientFunc
				getKubeClientFunc = func() (interface{}, error) {
					if tt.clientError != nil {
						return nil, tt.clientError
					}
					return &mockKubeClient{client: fakeClient}, nil
				}
				defer func() {
					getKubeClientFunc = originalGetFunc
				}()
			}

			interval := 10 * time.Millisecond
			timeout := 100 * time.Millisecond

			err := WaitForSvcRunning(interval, timeout, tt.namespace, tt.serviceName)

			if tt.expectError {
				if err == nil {
					t.Errorf("Expected error but got none")
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				if Endpoint != tt.expectEndpoint {
					t.Errorf("Expected endpoint %s, got %s", tt.expectEndpoint, Endpoint)
				}
			}
		})
	}
}

// TestTimeout 测试超时情况
func TestTimeout(t *testing.T) {
	originalEndpoint := Endpoint
	defer func() {
		Endpoint = originalEndpoint
	}()

	Endpoint = ""

	fakeClient := fake.NewSimpleClientset()
	fakeClient.AddReactor("get", "endpoints", func(action k8stesting.Action) (bool, runtime.Object, error) {
		return true, &v1.Endpoints{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-svc",
				Namespace: "test-ns",
			},
			Subsets: []v1.EndpointSubset{},
		}, nil
	})

	originalGetFunc := getKubeClientFunc
	getKubeClientFunc = func() (interface{}, error) {
		return &mockKubeClient{client: fakeClient}, nil
	}
	defer func() {
		getKubeClientFunc = originalGetFunc
	}()

	err := WaitForSvcRunning(5*time.Millisecond, 10*time.Millisecond, "test-ns", "test-svc")
	if err == nil {
		t.Error("Expected timeout error but got none")
	}
}
