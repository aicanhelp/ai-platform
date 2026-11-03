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

package cluster

import (
	"context"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/karmada-io/karmada/pkg/generated/clientset/versioned"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"openfuyao.com/multi-cluster-service/api/v1beta1"
	"openfuyao.com/multi-cluster-service/pkg/utils"
)

func TestJoinHostBranches(t *testing.T) {
	hostAPIConfig := clientcmdapi.NewConfig()
	hostAPIConfig.Clusters["kubernetes"] = &clientcmdapi.Cluster{Server: "https://127.0.0.1:6443"}
	hostAPIConfig.AuthInfos["kubernetes-admin"] = &clientcmdapi.AuthInfo{}
	hostAPIConfig.Contexts["kubernetes-admin@kubernetes"] = &clientcmdapi.Context{
		Cluster: "kubernetes", AuthInfo: "kubernetes-admin",
	}
	hostAPIConfig.CurrentContext = "kubernetes-admin@kubernetes"

	newHostPatches := func() *gomonkey.Patches {
		patches := gomonkey.ApplyFunc(utils.CreateHostClient,
			func() (*kubernetes.Clientset, *clientcmdapi.Config, error) {
				return &kubernetes.Clientset{}, hostAPIConfig, nil
			})
		patches.ApplyFunc(utils.IsKarmadaServiceReady, func(_ *versioned.Clientset) error {
			return nil
		})
		patches.ApplyFunc(utils.IsClusterUIDUnique,
			func(_ kubernetes.Interface, _ *versioned.Clientset) (string, error) {
				return "host-cluster-uid-0001", nil
			})
		patches.ApplyFunc(utils.GenerateClusterSecret,
			func(_ kubernetes.Interface, tag string) (*corev1.Secret, *corev1.Secret, error) {
				return testSecret(tag), testSecret(tag + "-" + utils.DefaultImpersonatorPrefix), nil
			})
		patches.ApplyFunc(utils.GenerateKarmadaSecret,
			func(_ kubernetes.Interface, clusterSt, impersonatorSt *corev1.Secret, _ string) (*corev1.Secret, *corev1.Secret, error) {
				return clusterSt, impersonatorSt, nil
			})
		patches.ApplyFunc(utils.HostConfToJson, func() ([]byte, error) {
			return []byte(`{"kind":"Config"}`), nil
		})
		return patches
	}

	t.Run("canceled context returns early", func(t *testing.T) {
		c := &OperationClient{}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		c.JoinHost(ctx) // must return without panic
	})

	t.Run("cluster ID already exists is tolerated", func(t *testing.T) {
		// host already registered -> IsClusterNameRepeated hits -> Join error tolerated
		api := newFakeKarmadaAPI(newTestKarmadaCluster("host", nil, true))
		c := &OperationClient{KarmadaVersionedClient: api.newClient(t), MgrClient: newTestMgrClient(t)}

		patches := newHostPatches()
		defer patches.Reset()

		c.JoinHost(context.TODO())
	})

	t.Run("host joined successfully", func(t *testing.T) {
		api := newFakeKarmadaAPI()
		kVersioned := api.newClient(t)
		mgr := newTestMgrClient(t)
		c := &OperationClient{KarmadaVersionedClient: kVersioned, MgrClient: mgr}

		patches := newHostPatches()
		defer patches.Reset()

		c.JoinHost(context.TODO())

		if _, err := kVersioned.ClusterV1alpha1().Clusters().Get(context.TODO(), "host", metav1.GetOptions{}); err != nil {
			t.Fatalf("host cluster not registered: %v", err)
		}
		obj := &v1beta1.Cluster{}
		if err := mgr.Get(context.TODO(), client.ObjectKey{Name: "host"}, obj); err != nil {
			t.Fatalf("v1beta1 host cluster not created: %v", err)
		}
	})
}
