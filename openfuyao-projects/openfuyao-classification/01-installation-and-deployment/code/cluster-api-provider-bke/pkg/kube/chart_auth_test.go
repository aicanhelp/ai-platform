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
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
)

func TestChartRepositoryConfigFromBKEClusterEmptyConfig(t *testing.T) {
	repo, auth, err := ChartRepositoryConfigFromBKECluster(context.Background(), nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if repo != "" || auth == nil {
		t.Fatalf("expected empty repo with auth config, got repo=%q auth=%#v", repo, auth)
	}

	repo, auth, err = ChartRepositoryConfigFromBKECluster(context.Background(), nil, &bkev1beta1.BKECluster{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if repo != "" || auth == nil {
		t.Fatalf("expected empty repo with auth config for nil ClusterConfig, got repo=%q auth=%#v", repo, auth)
	}
}

func TestChartRepositoryConfigFromBKEClusterWithSecrets(t *testing.T) {
	server, port := chartAuthServer(t)
	defer server.Close()

	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "bke-system"},
		Spec: confv1beta1.BKEClusterSpec{ClusterConfig: &confv1beta1.BKEConfig{
			Cluster: confv1beta1.Cluster{ChartRepo: confv1beta1.Repo{
				Ip:                    "127.0.0.1",
				Port:                  port,
				Prefix:                "charts",
				InsecureSkipTLSVerify: true,
				AuthSecretRef:         &confv1beta1.AuthSecretRef{Name: "repo-auth"},
				TlsSecretRef:          &confv1beta1.TlsSecretRef{Name: "repo-tls"},
			}},
		}},
	}
	cli := chartAuthFakeClient(t,
		&corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: "repo-auth", Namespace: "bke-system"}, Data: map[string][]byte{
			constant.UsernameKey: []byte("user"),
			constant.PasswordKey: []byte("pass"),
		}},
		&corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: "repo-tls", Namespace: "bke-system"}, Data: map[string][]byte{
			constant.CaKey:   []byte("ca"),
			constant.CertKey: []byte("cert"),
			constant.KeyKey:  []byte("key"),
		}},
	)

	repo, auth, err := ChartRepositoryConfigFromBKECluster(context.Background(), cli, cluster)
	if err != nil {
		t.Fatalf("ChartRepositoryConfigFromBKECluster: %v", err)
	}
	wantRepo := "127.0.0.1:" + port + "/charts"
	if repo != wantRepo {
		t.Fatalf("repo = %q, want %q", repo, wantRepo)
	}
	if string(auth.Username) != "user" || string(auth.Password) != "pass" {
		t.Fatalf("unexpected credentials user=%q password=%q", auth.Username, auth.Password)
	}
	if string(auth.CaFile) != "ca" || string(auth.CertFile) != "cert" || string(auth.KeyFile) != "key" {
		t.Fatalf("unexpected tls data ca=%q cert=%q key=%q", auth.CaFile, auth.CertFile, auth.KeyFile)
	}
	if !auth.InsecureSkipTLSVerify {
		t.Fatal("expected insecure skip tls verify to be propagated")
	}
}

func TestChartRepositoryConfigFromBKEClusterErrors(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "bke-system"},
		Spec: confv1beta1.BKEClusterSpec{ClusterConfig: &confv1beta1.BKEConfig{
			Cluster: confv1beta1.Cluster{ChartRepo: confv1beta1.Repo{
				Port:          "65534",
				Prefix:        "charts",
				AuthSecretRef: &confv1beta1.AuthSecretRef{Name: "missing-auth"},
			}},
		}},
	}
	if _, _, err := ChartRepositoryConfigFromBKECluster(context.Background(), chartAuthFakeClient(t), cluster); err == nil {
		t.Fatal("expected repo resolve error")
	}

	server, port := chartAuthServer(t)
	defer server.Close()
	cluster.Spec.ClusterConfig.Cluster.ChartRepo.Ip = "127.0.0.1"
	cluster.Spec.ClusterConfig.Cluster.ChartRepo.Port = port
	_, auth, err := ChartRepositoryConfigFromBKECluster(context.Background(), chartAuthFakeClient(t), cluster)
	if err != nil {
		t.Fatalf("unexpected missing auth secret error: %v", err)
	}
	if string(auth.Username) != "" || string(auth.Password) != "" {
		t.Fatalf("expected missing auth secret to fall back to empty credentials, got user=%q password=%q", auth.Username, auth.Password)
	}
}

func chartAuthServer(t *testing.T) (*httptest.Server, string) {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	_, port, err := net.SplitHostPort(server.Listener.Addr().String())
	if err != nil {
		server.Close()
		t.Fatalf("split server address: %v", err)
	}
	return server, port
}

func chartAuthFakeClient(t *testing.T, objs ...client.Object) client.Client {
	t.Helper()
	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("add core scheme: %v", err)
	}
	return fake.NewClientBuilder().WithScheme(scheme).WithObjects(objs...).Build()
}
