/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
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
	"testing"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestGetCommonParamFromBKECluster_ConfigRefs(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "bke-cluster", Namespace: "bke-cluster"},
		Spec: confv1beta1.BKEClusterSpec{
			KubeletConfigRef: &confv1beta1.KubeletConfigRef{
				Name:      "bke-kubelet",
				Namespace: "bke-kubelet",
			},
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{
					EtcdVersion: "v3.5.21-of.1",
					ContainerdConfigRef: &confv1beta1.ContainerdConfigRef{
						Name:      "bke-containerd",
						Namespace: "bke-containerd",
					},
					ImageRepo: confv1beta1.Repo{
						Domain: "deploy.example.com",
						Ip:     "10.0.0.1",
						Port:   "40443",
						Prefix: "kubernetes",
					},
					HTTPRepo: confv1beta1.Repo{
						Domain: "deploy.example.com",
						Ip:     "10.0.0.1",
						Port:   "40080",
						Prefix: "yum",
					},
					NTPServer:       "ntp.example.com",
					AgentHealthPort: "58080",
					Networking: confv1beta1.Networking{
						ServiceSubnet: "10.96.0.0/12",
					},
				},
				CustomExtra: map[string]string{},
			},
		},
	}

	param, err := getCommonParamFromBKECluster(cluster, bkenode.Nodes{})
	if err != nil {
		t.Fatalf("getCommonParamFromBKECluster returned error: %v", err)
	}

	assertParamEqual(t, param, "etcdVersion", "v3.5.21-of.1")
	assertParamEqual(t, param, "containerdConfigRefName", "bke-containerd")
	assertParamEqual(t, param, "containerdConfigRefNamespace", "bke-containerd")
	assertParamEqual(t, param, "kubeletConfigRefName", "bke-kubelet")
	assertParamEqual(t, param, "kubeletConfigRefNamespace", "bke-kubelet")
}

func TestGetCommonParamFromBKECluster_ConfigRefsOmittedWhenEmpty(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "bke-cluster", Namespace: "bke-cluster"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{
					ImageRepo: confv1beta1.Repo{
						Domain: "deploy.example.com",
						Ip:     "10.0.0.1",
						Port:   "40443",
						Prefix: "kubernetes",
					},
					HTTPRepo: confv1beta1.Repo{
						Domain: "deploy.example.com",
						Ip:     "10.0.0.1",
						Port:   "40080",
						Prefix: "yum",
					},
					Networking: confv1beta1.Networking{
						ServiceSubnet: "10.96.0.0/12",
					},
				},
				CustomExtra: map[string]string{},
			},
		},
	}

	param, err := getCommonParamFromBKECluster(cluster, bkenode.Nodes{})
	if err != nil {
		t.Fatalf("getCommonParamFromBKECluster returned error: %v", err)
	}

	for _, key := range []string{
		"containerdConfigRefName",
		"containerdConfigRefNamespace",
		"kubeletConfigRefName",
		"kubeletConfigRefNamespace",
	} {
		if _, ok := param[key]; ok {
			t.Fatalf("expected %s to be omitted when ref is empty, got %v", key, param[key])
		}
	}
}

func assertParamEqual(t *testing.T, param map[string]interface{}, key, want string) {
	t.Helper()
	got, ok := param[key]
	if !ok {
		t.Fatalf("expected param %q to be set", key)
	}
	gotStr, ok := got.(string)
	if !ok {
		t.Fatalf("expected param %q to be string, got %T", key, got)
	}
	if gotStr != want {
		t.Fatalf("param %q = %q, want %q", key, gotStr, want)
	}
}
