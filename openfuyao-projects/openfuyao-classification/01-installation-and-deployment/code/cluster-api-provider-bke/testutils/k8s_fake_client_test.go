/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */
package testutils

import (
	"context"
	"fmt"
	"testing"

	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
)

var initBkeCluster = &bkev1beta1.BKECluster{TypeMeta: metav1.TypeMeta{
	Kind: "BKECluster", APIVersion: "bke.bocloud.com/v1beta1"},
	ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default", UID: "test"},
	Spec: confv1beta1.BKEClusterSpec{
		ClusterConfig: &confv1beta1.BKEConfig{
			Cluster: confv1beta1.Cluster{
				ControlPlane: confv1beta1.ControlPlane{
					Etcd: &confv1beta1.Etcd{
						ServerCertSANs: []string{"127.0.0.1", "abc.com"},
					},
					APIServer: &confv1beta1.APIServer{
						CertSANs: []string{fmt.Sprintf("%s.%s.%s.%s", "2", "2", "2", "2"), "def.com"},
					},
				},
			},
		},
	},
}
var initApiResource = []*v1.APIResourceList{
	{
		GroupVersion: "bke.bocloud.com/v1beta1", APIResources: []v1.APIResource{
			{
				Name:       "bkeclusters",
				Namespaced: false,
				Kind:       "BKECluster",
				Verbs:      []string{"create", "delete", "get", "list"},
			},
		},
	},
	{
		GroupVersion: "v1", APIResources: []v1.APIResource{
			{
				Name:       "management-admin",
				Namespaced: true,
				Kind:       "Secret",
				Verbs:      []string{"create", "delete", "get", "list"},
			},
		},
	},
}

func TestGetRuntimeFakeClientFun(t *testing.T) {
	t.Run("TestGetRuntimeFakeClient", func(t *testing.T) {
		TestGetRuntimeFakeClient([]func(*runtime.Scheme) error{bkev1beta1.AddToScheme}, initBkeCluster)
		TestGetRuntimeFakeClient([]func(*runtime.Scheme) error{func(scheme *runtime.Scheme) error {
			return errors.New("xxxxxxxxxxx")
		}})
	})
}
func TestGetClientGoFakeClientFun(t *testing.T) {
	t.Run("TestGetClientGoFakeClient", func(t *testing.T) {
		TestGetClientGoFakeClient(nil)
	})
}
func TestGetSimpleDynamicClientWithCustomListKindsFun(t *testing.T) {
	t.Run("TestGetSimpleDynamicClientWithCustomListKinds", func(t *testing.T) {
		kinds := map[schema.GroupVersionResource]string{
			{Group: "bke.bocloud.com", Version: "v1beta1", Resource: "bkeclusters"}: "BKEClusterList",
		}
		TestGetSimpleDynamicClientWithCustomListKinds(bkev1beta1.AddToScheme, nil)
		TestGetSimpleDynamicClientWithCustomListKinds(bkev1beta1.AddToScheme, kinds)

		TestGetSimpleDynamicClientWithCustomListKinds(bkev1beta1.AddToScheme, kinds, &corev1.Secret{})
	})
}
func TestGetManagerClientFun(t *testing.T) {
	t.Run("TestGetManagerClient", func(t *testing.T) {
		client, _ := TestGetManagerClient([]func(*runtime.Scheme) error{bkev1beta1.AddToScheme})
		client, _ = TestGetManagerClient([]func(*runtime.Scheme) error{bkev1beta1.AddToScheme}, &corev1.Secret{})
		client.GetHTTPClient()
		client.GetFieldIndexer()
		client.GetEventRecorderFor("")
		client.GetRESTMapper()
		client.GetAPIReader()
		client.Add(nil)
		client.Elected()
		client.AddMetricsExtraHandler("", nil)
		client.GetLogger()
		client.GetClient()
		client.GetScheme()
		client.GetCache()
		client.Start(context.Background())
		client.GetConfig()
		client.GetControllerOptions()
		client.AddHealthzCheck("", nil)
		client.AddReadyzCheck("", nil)
		client.GetWebhookServer()
		client.GetName()
	})
}
