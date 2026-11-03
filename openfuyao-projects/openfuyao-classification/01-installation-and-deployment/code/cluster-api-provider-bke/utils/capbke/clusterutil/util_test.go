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

package clusterutil

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common"
	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
)

func TestBuildYumRepoDownloadBaseURLFallbackToIP(t *testing.T) {
	url := BuildYumRepoDownloadBaseURL(bkeinit.BkeConfig{
		Cluster: confv1beta1.Cluster{
			HTTPRepo: confv1beta1.Repo{
				Domain: "bad host",
				Ip:     "10.0.0.1",
				Port:   "40080",
				Prefix: "files",
			},
		},
	})
	assert.Equal(t, "http://10.0.0.1:40080/files", url)
}

func TestAvailableLoadBalancerEndPoint(t *testing.T) {
	tests := []struct {
		name     string
		endPoint confv1beta1.APIEndpoint
		nodes    bkenode.Nodes
		expected bool
	}{
		{
			name:     "invalid endpoint",
			endPoint: confv1beta1.APIEndpoint{Host: "", Port: 0},
			nodes:    bkenode.Nodes{},
			expected: false,
		},
		{
			name:     "valid endpoint not in nodes",
			endPoint: confv1beta1.APIEndpoint{Host: "192.168.1.100", Port: 6443},
			nodes:    bkenode.Nodes{{IP: "192.168.1.1"}},
			expected: true,
		},
		{
			name:     "valid endpoint in nodes",
			endPoint: confv1beta1.APIEndpoint{Host: "192.168.1.1", Port: 6443},
			nodes:    bkenode.Nodes{{IP: "192.168.1.1"}},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := AvailableLoadBalancerEndPoint(tt.endPoint, tt.nodes)
			if result != tt.expected {
				t.Errorf("expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestGetIngressConfig(t *testing.T) {
	tests := []struct {
		name          string
		addons        []confv1beta1.Product
		expectedVIP   string
		expectedNodes []string
	}{
		{
			name:          "no addons",
			addons:        []confv1beta1.Product{},
			expectedVIP:   "",
			expectedNodes: nil,
		},
		{
			name: "no beyondELB addon",
			addons: []confv1beta1.Product{
				{Name: "other", Param: map[string]string{}},
			},
			expectedVIP:   "",
			expectedNodes: nil,
		},
		{
			name: "with beyondELB addon",
			addons: []confv1beta1.Product{
				{
					Name: "beyondELB",
					Param: map[string]string{
						"lbVIP":   "192.168.1.100",
						"lbNodes": "192.168.1.1,192.168.1.2",
					},
				},
			},
			expectedVIP:   "192.168.1.100",
			expectedNodes: []string{},
		},
		{
			name: "with empty lbNodes",
			addons: []confv1beta1.Product{
				{
					Name: "beyondELB",
					Param: map[string]string{
						"lbVIP":   "192.168.1.100",
						"lbNodes": "",
					},
				},
			},
			expectedVIP:   "192.168.1.100",
			expectedNodes: []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			vip, nodes := GetIngressConfig(tt.addons)
			if vip != tt.expectedVIP {
				t.Errorf("expected VIP %s, got %s", tt.expectedVIP, vip)
			}
			if len(nodes) != len(tt.expectedNodes) {
				t.Errorf("expected %d nodes, got %d", len(tt.expectedNodes), len(nodes))
			}
		})
	}
}

func TestBKEConfigCmKey(t *testing.T) {
	key := BKEConfigCmKey()
	if key.Namespace != "cluster-system" {
		t.Errorf("expected namespace cluster-system, got %s", key.Namespace)
	}
	if key.Name != common.BKEClusterConfigFileName {
		t.Errorf("expected name %s, got %s", common.BKEClusterConfigFileName, key.Name)
	}
}

func TestGetBKEConfigCMData(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)

	tests := []struct {
		name      string
		cm        *corev1.ConfigMap
		expectErr bool
	}{
		{
			name: "configmap exists",
			cm: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      common.BKEClusterConfigFileName,
					Namespace: "cluster-system",
				},
				Data: map[string]string{"key": "value"},
			},
			expectErr: false,
		},
		{
			name:      "configmap not exists",
			cm:        nil,
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var objs []client.Object
			if tt.cm != nil {
				objs = append(objs, tt.cm)
			}
			fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(objs...).Build()

			data, err := GetBKEConfigCMData(context.Background(), fakeClient)
			if tt.expectErr {
				if err == nil {
					t.Error("expected error, got nil")
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if data == nil {
					t.Error("expected data, got nil")
				}
			}
		})
	}
}

func TestLoadBalanceType(t *testing.T) {
	assert.Equal(t, "ha", LoadBalanceType(nil))
	assert.Equal(t, "ha", LoadBalanceType(&confv1beta1.BKEConfig{}))
	assert.Equal(t, "ha", LoadBalanceType(&confv1beta1.BKEConfig{Cluster: confv1beta1.Cluster{LoadBalanceType: ""}}))
	assert.Equal(t, "ha", LoadBalanceType(&confv1beta1.BKEConfig{Cluster: confv1beta1.Cluster{LoadBalanceType: "ha"}}))
	assert.Equal(t, "nginx", LoadBalanceType(&confv1beta1.BKEConfig{Cluster: confv1beta1.Cluster{LoadBalanceType: "nginx"}}))
}
