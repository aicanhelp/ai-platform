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
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/karmada-io/karmada/pkg/apis/cluster/v1alpha1"
	"github.com/karmada-io/karmada/pkg/generated/clientset/versioned"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"openfuyao.com/multi-cluster-service/api/v1beta1"
	"openfuyao.com/multi-cluster-service/pkg/utils"
)

func testSecret(name string) *corev1.Secret {
	s := &corev1.Secret{}
	s.Name = name
	s.Namespace = utils.DefaultKarmadaNamespace
	return s
}

func TestOperationClientJoinMemberCluster(t *testing.T) {
	k8sAPI := newFakeK8sAPI(map[string]*corev1.Namespace{"kube-system": testKubeSystemNS()}, nil)
	kubeconfig := k8sAPI.kubeconfig(t)

	api := newFakeKarmadaAPI()
	kVersioned := api.newClient(t)
	mgr := newTestMgrClient(t)
	c := &OperationClient{KarmadaVersionedClient: kVersioned, MgrClient: mgr}

	patches := gomonkey.ApplyFunc(utils.GenerateClusterSecret,
		func(_ kubernetes.Interface, tag string) (*corev1.Secret, *corev1.Secret, error) {
			return testSecret(tag), testSecret(tag + "-" + utils.DefaultImpersonatorPrefix), nil
		})
	patches.ApplyFunc(utils.GenerateKarmadaSecret,
		func(_ kubernetes.Interface, clusterSt *corev1.Secret, impersonatorSt *corev1.Secret, _ string) (*corev1.Secret, *corev1.Secret, error) {
			return clusterSt, impersonatorSt, nil
		})
	defer patches.Reset()

	opt := JoinOptions{ClusterName: "member1", Kubeconfig: kubeconfig}
	if err := c.Join(context.TODO(), opt); err != nil {
		t.Fatalf("Join() error = %v", err)
	}

	// karmada cluster object registered with the member cluster uid
	registered, err := kVersioned.ClusterV1alpha1().Clusters().Get(context.TODO(), "member1", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("karmada cluster not registered: %v", err)
	}
	if registered.Spec.ID != "test-cluster-uid-0001" {
		t.Errorf("karmada cluster ID = %q, want test-cluster-uid-0001", registered.Spec.ID)
	}
	if registered.Spec.SyncMode != v1alpha1.Push {
		t.Errorf("SyncMode = %v, want Push", registered.Spec.SyncMode)
	}

	// v1beta1 cluster object created with the out-of-band kubeconfig
	obj := &v1beta1.Cluster{}
	if err := mgr.Get(context.TODO(), client.ObjectKey{Name: "member1"}, obj); err != nil {
		t.Fatalf("v1beta1 cluster not created: %v", err)
	}
	if len(obj.Spec.OutClusterConfig.Raw) == 0 {
		t.Error("OutClusterConfig is empty")
	}
}

func TestOperationClientJoinDuplicateName(t *testing.T) {
	k8sAPI := newFakeK8sAPI(map[string]*corev1.Namespace{"kube-system": testKubeSystemNS()}, nil)
	kubeconfig := k8sAPI.kubeconfig(t)

	api := newFakeKarmadaAPI(newTestKarmadaCluster("member1", nil, true))
	c := &OperationClient{KarmadaVersionedClient: api.newClient(t), MgrClient: newTestMgrClient(t)}

	opt := JoinOptions{ClusterName: "member1", Kubeconfig: kubeconfig}
	err := c.Join(context.TODO(), opt)
	if err == nil || err.Error() != "cluster ID already exists" {
		t.Errorf("error = %v, want cluster ID already exists", err)
	}
}

func TestOperationClientJoinKarmadaExists(t *testing.T) {
	k8sAPI := newFakeK8sAPI(map[string]*corev1.Namespace{
		"kube-system":    testKubeSystemNS(),
		"karmada-system": {ObjectMeta: metav1.ObjectMeta{Name: "karmada-system"}},
	}, nil)
	k8sAPI.deployments["karmada-system/karmada-apiserver"] = true
	kubeconfig := k8sAPI.kubeconfig(t)

	api := newFakeKarmadaAPI()
	c := &OperationClient{KarmadaVersionedClient: api.newClient(t), MgrClient: newTestMgrClient(t)}

	opt := JoinOptions{ClusterName: "member1", Kubeconfig: kubeconfig}
	err := c.Join(context.TODO(), opt)
	if err == nil || err.Error() != "target Cluster ID contains karmada system or multicluster service" {
		t.Errorf("error = %v, want karmada service exists error", err)
	}
}

func TestOperationClientJoinInvalidKubeconfig(t *testing.T) {
	api := newFakeKarmadaAPI()
	c := &OperationClient{KarmadaVersionedClient: api.newClient(t), MgrClient: newTestMgrClient(t)}

	opt := JoinOptions{ClusterName: "member1", Kubeconfig: []byte("{invalid-json")}
	if err := c.Join(context.TODO(), opt); err == nil {
		t.Error("Join() expected error for invalid kubeconfig data")
	}
}

func TestOperationClientJoinHostCluster(t *testing.T) {
	api := newFakeKarmadaAPI()
	kVersioned := api.newClient(t)
	mgr := newTestMgrClient(t)
	c := &OperationClient{KarmadaVersionedClient: kVersioned, MgrClient: mgr}

	hostAPIConfig := clientcmdapi.NewConfig()
	hostAPIConfig.Clusters["kubernetes"] = &clientcmdapi.Cluster{Server: "https://127.0.0.1:6443"}
	hostAPIConfig.AuthInfos["kubernetes-admin"] = &clientcmdapi.AuthInfo{}
	hostAPIConfig.Contexts["kubernetes-admin@kubernetes"] = &clientcmdapi.Context{
		Cluster: "kubernetes", AuthInfo: "kubernetes-admin",
	}
	hostAPIConfig.CurrentContext = "kubernetes-admin@kubernetes"

	patches := gomonkey.ApplyFunc(utils.CreateHostClient,
		func() (*kubernetes.Clientset, *clientcmdapi.Config, error) {
			return &kubernetes.Clientset{}, hostAPIConfig, nil
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
		func(_ kubernetes.Interface, clusterSt *corev1.Secret, impersonatorSt *corev1.Secret, _ string) (*corev1.Secret, *corev1.Secret, error) {
			return clusterSt, impersonatorSt, nil
		})
	patches.ApplyFunc(utils.HostConfToJson, func() ([]byte, error) {
		return []byte(`{"kind":"Config"}`), nil
	})
	defer patches.Reset()

	if err := c.Join(context.TODO(), JoinOptions{}); err != nil {
		t.Fatalf("Join() host error = %v", err)
	}

	if _, err := kVersioned.ClusterV1alpha1().Clusters().Get(context.TODO(), "host", metav1.GetOptions{}); err != nil {
		t.Fatalf("host cluster not registered in karmada: %v", err)
	}
	obj := &v1beta1.Cluster{}
	if err := mgr.Get(context.TODO(), client.ObjectKey{Name: "host"}, obj); err != nil {
		t.Fatalf("v1beta1 host cluster not created: %v", err)
	}
}

func TestRegisterHostClusterDirect(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		api := newFakeKarmadaAPI()
		cluster := newTestKarmadaCluster("member9", nil, false)
		if err := registerHostCluster(api.newClient(t), cluster); err != nil {
			t.Errorf("registerHostCluster() error = %v", err)
		}
	})

	t.Run("create failed", func(t *testing.T) {
		broken, err := versioned.NewForConfig(&rest.Config{Host: "127.0.0.1:1"})
		if err != nil {
			t.Fatalf("new broken client: %v", err)
		}
		if err := registerHostCluster(broken, newTestKarmadaCluster("member9", nil, false)); err == nil {
			t.Error("registerHostCluster() expected error on unreachable host")
		}
	})
}

func TestRegisterClusterHTTP(t *testing.T) {
	// KarmadaEndPoints is a compile-time constant pointing at the in-cluster
	// karmada apiserver; outside a real cluster the POST must fail (and must
	// not hang forever).
	done := make(chan error, 1)
	go func() {
		done <- registerCluster(*newTestKarmadaCluster("member9", nil, false))
	}()
	select {
	case err := <-done:
		if err == nil {
			t.Error("registerCluster() expected error posting to unreachable karmada endpoint")
		}
	case <-time.After(30 * time.Second):
		t.Fatal("registerCluster() timed out posting to unreachable karmada endpoint")
	}
}
