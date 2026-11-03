/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package kube

import (
	"context"
	"errors"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

const (
	testAPINodeName      = "test-node"
	testAPIPodName       = "test-pod"
	testAPIPodNamespace  = "default"
	testAPINodeIPAddress = "192.168.1.10"
)

func TestClientListNodes(t *testing.T) {
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: testAPINodeName},
		Status: corev1.NodeStatus{
			Addresses: []corev1.NodeAddress{
				{Type: corev1.NodeInternalIP, Address: testAPINodeIPAddress},
			},
		},
	}

	tests := []struct {
		name      string
		option    *metav1.ListOptions
		mockError error
		wantErr   bool
	}{
		{
			name:      "list nodes with nil option",
			option:    nil,
			mockError: nil,
			wantErr:   false,
		},
		{
			name:      "list nodes with empty option",
			option:    &metav1.ListOptions{},
			mockError: nil,
			wantErr:   false,
		},
		{
			name:      "list nodes with error",
			option:    &metav1.ListOptions{},
			mockError: errors.New("list failed"),
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockCS := &kubernetes.Clientset{}
			patches := gomonkey.ApplyMethodFunc(mockCS.CoreV1().Nodes(), "List",
				func(ctx context.Context, opts metav1.ListOptions) (*corev1.NodeList, error) {
					if tt.mockError != nil {
						return nil, tt.mockError
					}
					return &corev1.NodeList{Items: []corev1.Node{*node}}, nil
				})
			defer patches.Reset()

			c := &Client{
				ClientSet: mockCS,
				Ctx:       context.Background(),
			}

			list, err := c.ListNodes(tt.option)
			if (err != nil) != tt.wantErr {
				t.Errorf("ListNodes() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && list == nil {
				t.Error("ListNodes() returned nil list")
			}
		})
	}
}

func TestClientGetPod(t *testing.T) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testAPIPodName,
			Namespace: testAPIPodNamespace,
		},
	}

	tests := []struct {
		name      string
		namespace string
		podName   string
		mockError error
		wantErr   bool
	}{
		{
			name:      "get existing pod",
			namespace: testAPIPodNamespace,
			podName:   testAPIPodName,
			mockError: nil,
			wantErr:   false,
		},
		{
			name:      "get pod with error",
			namespace: testAPIPodNamespace,
			podName:   testAPIPodName,
			mockError: errors.New("pod not found"),
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockCS := &kubernetes.Clientset{}
			patches := gomonkey.ApplyMethodFunc(mockCS.CoreV1().Pods(tt.namespace), "Get",
				func(ctx context.Context, name string, opts metav1.GetOptions) (*corev1.Pod, error) {
					if tt.mockError != nil {
						return nil, tt.mockError
					}
					return pod, nil
				})
			defer patches.Reset()

			c := &Client{
				ClientSet: mockCS,
				Ctx:       context.Background(),
			}

			result, err := c.GetPod(tt.namespace, tt.podName)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetPod() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && result == nil {
				t.Error("GetPod() returned nil pod")
			}
		})
	}
}
