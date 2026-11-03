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
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/google/go-cmp/cmp"
	"github.com/karmada-io/karmada/pkg/apis/cluster/v1alpha1"
	"github.com/karmada-io/karmada/pkg/generated/clientset/versioned"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	faker "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/klog/v2"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"openfuyao.com/multi-cluster-service/api/v1beta1"
	"openfuyao.com/multi-cluster-service/pkg/utils"
	"openfuyao.com/multi-cluster-service/pkg/zlog"
)

func TestCreateAPICluster(t *testing.T) {
	ctx := context.TODO()
	scheme := runtime.NewScheme()
	v1beta1.AddToScheme(scheme)
	fakeMgrClient := fake.NewClientBuilder().WithScheme(scheme).Build()
	collector := v1beta1.Cluster{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-cluster",
		},
	}
	if err := fakeMgrClient.Create(ctx, &collector); err != nil {
		t.Fatalf("Failed to create collector: %v", err)
	} else {
		klog.Info("Collector resource created successfully")
	}

	type args struct {
		buildOption OptionsOfCluster
		mgrClient   client.Client
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{
			name: "success",
			args: args{
				buildOption: OptionsOfCluster{ClusterName: "test-cluster",
					ClusterLabels: map[string]string{
						"a": "b",
					}},
				mgrClient: fakeMgrClient,
			},
			wantErr: false,
		},
		{
			name: "len(buildOption.ClusterLabels) != 0",
			args: args{
				mgrClient: fakeMgrClient,
				buildOption: OptionsOfCluster{
					ClusterName: "ok",
					ClusterLabels: map[string]string{
						"c": "d",
					},
				},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := createAPICluster(tt.args.mgrClient, tt.args.buildOption); (err != nil) != tt.wantErr {
				t.Errorf("createAPICluster() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestTestClient(t *testing.T) {
	type args struct {
		mgrClient client.Client
		tag       string
		collector v1beta1.Cluster
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: "success",
			args: args{
				mgrClient: MakefakeClient(v1beta1.Cluster{
					ObjectMeta: metav1.ObjectMeta{
						Name: "cluster",
					},
					Spec: v1beta1.ClusterSpec{
						KubeConfig: runtime.RawExtension{
							Raw: []byte(`{
                                "apiVersion": "v1"}`),
						},
					},
				}),
				tag: "cluster",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			testClient(tt.args.mgrClient, tt.args.tag)
		})
	}
}

func MakefakeClient(collector v1beta1.Cluster) client.Client {
	ctx := context.TODO()
	scheme := runtime.NewScheme()
	v1beta1.AddToScheme(scheme)
	fakeMgrClient := fake.NewClientBuilder().WithScheme(scheme).Build()
	if err := fakeMgrClient.Create(ctx, &collector); err != nil {
		zlog.LogError("Failed to create collector: %v", err)
	} else {
		klog.Info("Collector resource created successfully")
	}
	return fakeMgrClient
}

func CreatefakeClient(kubeConfig []byte) client.Client {
	ctx := context.TODO()
	scheme := runtime.NewScheme()
	v1beta1.AddToScheme(scheme)
	fakeMgrClient := fake.NewClientBuilder().WithScheme(scheme).Build()
	collector := v1beta1.Cluster{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-cluster",
		},
		Spec: v1beta1.ClusterSpec{
			KubeConfig: runtime.RawExtension{
				Raw: kubeConfig,
			},
		},
	}
	if err := fakeMgrClient.Create(ctx, &collector); err != nil {
		zlog.LogError("Failed to create collector: %v", err)
	} else {
		klog.Info("Collector resource created successfully")
	}
	return fakeMgrClient
}

func TestCreateApiCluster(t *testing.T) {
	type args struct {
		mgrClient   client.Client
		buildOption OptionsOfCluster
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{
			name: "success",
			args: args{
				mgrClient: CreatefakeClient([]byte(`{
                    "apiVersion": "v1",`)),
				buildOption: OptionsOfCluster{
					ClusterName: "s",
				},
			},
			wantErr: false,
		},
		{
			name: "exit",
			args: args{
				mgrClient: CreatefakeClient([]byte(`{
                    "apiVersion": "v1",`)),
				buildOption: OptionsOfCluster{
					ClusterName: "test-cluster",
				},
			},
			wantErr: false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if err := createAPICluster(tc.args.mgrClient, tc.args.buildOption); (err != nil) != tc.wantErr {
				t.Errorf("createAPICluster() error = %v, wantErr %v", err, tc.wantErr)
			}
		})
	}
}

func TestBuildCluster(t *testing.T) {
	type args struct {
		opt OptionsOfCluster
	}
	tests := []struct {
		name string
		args args
		want *v1alpha1.Cluster
	}{
		{
			name: "success",
			args: args{
				opt: OptionsOfCluster{ClusterName: "a",
					ClusterLabels: map[string]string{"*": "a"}},
			},
			want: &v1alpha1.Cluster{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "a",
					Labels: map[string]string{"*": "a"},
				},
				Spec: v1alpha1.ClusterSpec{
					SyncMode:              v1alpha1.Push,
					SecretRef:             &v1alpha1.LocalSecretReference{},
					ImpersonatorSecretRef: &v1alpha1.LocalSecretReference{},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := buildCluster(tt.args.opt)
			if diff := cmp.Diff(got, tt.want); diff != "" {
				t.Errorf("buildCluster() mismatch (-got +want):\n%s", diff)
			}
		})
	}
}

func TestEstablishClientAndApiConfig(t *testing.T) {
	tests := getTestCases()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.ApplyFunc(utils.CreateHostClient, func() (*kubernetes.Clientset, *api.Config, error) {
				if tt.mockCHCErr != nil {
					return &kubernetes.Clientset{}, api.NewConfig(), tt.mockCHCErr
				}
				return &kubernetes.Clientset{}, api.NewConfig(), nil
			}).ApplyFunc(utils.CreateClient, func(path string) (*kubernetes.Clientset, *api.Config, error) {
				if tt.mockCreateErr != nil {
					return &kubernetes.Clientset{}, api.NewConfig(), tt.mockCreateErr
				}
				return &kubernetes.Clientset{}, api.NewConfig(), nil
			}).ApplyFunc(utils.LoadKubeconfigRawData, func(kubeconfigData json.RawMessage,
				apiKubeconfig *api.Config) error {
				if tt.mockLoadErr != nil {
					return tt.mockLoadErr
				}
				return nil
			}).ApplyFunc(utils.CreateClusterClient, func(config *api.Config) (*kubernetes.Clientset, error) {
				if tt.mockCCCErr != nil {
					return &kubernetes.Clientset{}, tt.mockCCCErr
				}
				return &kubernetes.Clientset{}, nil
			})
			defer patches.Reset()

			got, got1, err := establishClientAndApiConfig(tt.args.opt)
			if (err != nil) != tt.wantErr {
				t.Errorf("establishClientAndApiConfig() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("establishClientAndApiConfig() got = %v, want %v", got, tt.want)
			}
			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("establishClientAndApiConfig() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}

func getTestCases() []struct {
	name          string
	args          struct{ opt JoinOptions }
	want          *kubernetes.Clientset
	want1         *api.Config
	wantErr       bool
	mockCreateErr error
	mockLoadErr   error
	mockCCCErr    error
	mockCHCErr    error
} {
	return []struct {
		name          string
		args          struct{ opt JoinOptions }
		want          *kubernetes.Clientset
		want1         *api.Config
		wantErr       bool
		mockCreateErr error
		mockLoadErr   error
		mockCCCErr    error
		mockCHCErr    error
	}{
		{
			name: "success",
			args: struct{ opt JoinOptions }{
				opt: JoinOptions{},
			},
			want:    &kubernetes.Clientset{},
			want1:   api.NewConfig(),
			wantErr: false,
		},
		{
			name: "no KubeconfigPath ",
			args: struct{ opt JoinOptions }{
				opt: JoinOptions{
					KubeconfigPath: "url",
				},
			},
			want:          nil,
			want1:         nil,
			wantErr:       true,
			mockCreateErr: errors.New("error create"),
		},
		{
			name: "no Kubeconfig",
			args: struct{ opt JoinOptions }{
				opt: JoinOptions{
					Kubeconfig: []byte(`{a}`),
				},
			},
			want:        nil,
			want1:       nil,
			wantErr:     true,
			mockLoadErr: errors.New("error create"),
		},
		{
			name: "no Kubeconfig but another error",
			args: struct{ opt JoinOptions }{
				opt: JoinOptions{
					Kubeconfig: []byte(`{a}`),
				},
			},
			want:       nil,
			want1:      nil,
			wantErr:    true,
			mockCCCErr: errors.New("error ccc"),
		},
		{
			name: "CHC ERROR",
			args: struct{ opt JoinOptions }{
				opt: JoinOptions{},
			},
			want:       nil,
			want1:      nil,
			wantErr:    true,
			mockCHCErr: errors.New("error cHc"),
		},
	}
}
func TestOperationClientCreateCredentials(t *testing.T) {
	tests := getCreateCredentialsTestCases()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &OperationClient{
				MgrClient:              tt.fields.MgrClient,
				KarmadaClient:          tt.fields.KarmadaClient,
				KarmadaVersionedClient: tt.fields.KarmadaVersionedClient,
			}

			patches := applyCreateCredentialsMocks(tt)
			defer patches.Reset()

			got, got1, err := c.createCredentials(tt.args.k, tt.args.t)
			assertCreateCredentialsResults(t, tt, got, got1, err)
		})
	}
}

func getCreateCredentialsTestCases() []struct {
	name   string
	fields struct {
		MgrClient              client.Client
		KarmadaClient          *kubernetes.Clientset
		KarmadaVersionedClient *versioned.Clientset
	}
	args struct {
		k kubernetes.Interface
		t string
	}
	want           *corev1.Secret
	want1          *corev1.Secret
	wantErr        bool
	mockErr        error
	mockKarmadaErr error
} {
	return []struct {
		name   string
		fields struct {
			MgrClient              client.Client
			KarmadaClient          *kubernetes.Clientset
			KarmadaVersionedClient *versioned.Clientset
		}
		args struct {
			k kubernetes.Interface
			t string
		}
		want           *corev1.Secret
		want1          *corev1.Secret
		wantErr        bool
		mockErr        error
		mockKarmadaErr error
	}{
		{
			name: "success",
			fields: struct {
				MgrClient              client.Client
				KarmadaClient          *kubernetes.Clientset
				KarmadaVersionedClient *versioned.Clientset
			}{
				MgrClient: CreatefakeClient([]byte(`{"apiVersion": "v1"}`)),
			},
			args: struct {
				k kubernetes.Interface
				t string
			}{
				k: faker.NewSimpleClientset(),
				t: "true",
			},
			want:    &corev1.Secret{Data: map[string][]byte{"a": []byte("s")}},
			want1:   &corev1.Secret{StringData: map[string]string{"u": "s"}},
			wantErr: false,
		},
		{
			name: "Error Generating cluster Secrets",

			args: struct {
				k kubernetes.Interface
				t string
			}{
				k: faker.NewSimpleClientset(),
				t: "true",
			},
			fields: struct {
				MgrClient              client.Client
				KarmadaClient          *kubernetes.Clientset
				KarmadaVersionedClient *versioned.Clientset
			}{
				MgrClient: CreatefakeClient([]byte(`{"apiVersion": "v1"}`)),
			},
			want:    nil,
			want1:   nil,
			wantErr: true,
			mockErr: errors.New("s"),
		},
		{
			name: "karmada mock",
			fields: struct {
				MgrClient              client.Client
				KarmadaClient          *kubernetes.Clientset
				KarmadaVersionedClient *versioned.Clientset
			}{
				MgrClient: CreatefakeClient([]byte(`{"apiVersion": "v1"}`)),
			},
			want:  nil,
			want1: nil,
			args: struct {
				k kubernetes.Interface
				t string
			}{
				t: "true",
				k: faker.NewSimpleClientset(),
			},
			wantErr:        true,
			mockKarmadaErr: errors.New("s"),
		},
	}
}

func applyCreateCredentialsMocks(tt struct {
	name   string
	fields struct {
		MgrClient              client.Client
		KarmadaClient          *kubernetes.Clientset
		KarmadaVersionedClient *versioned.Clientset
	}
	args struct {
		k kubernetes.Interface
		t string
	}
	want           *corev1.Secret
	want1          *corev1.Secret
	wantErr        bool
	mockErr        error
	mockKarmadaErr error
}) *gomonkey.Patches {
	return gomonkey.ApplyFunc(utils.GenerateClusterSecret,
		func(c kubernetes.Interface, tag string) (*corev1.Secret, *corev1.Secret, error) {
			if tt.mockErr != nil {
				return nil, nil, tt.mockErr
			}
			return &corev1.Secret{}, &corev1.Secret{}, nil
		}).ApplyFunc(utils.GenerateKarmadaSecret, func(c kubernetes.Interface,
		clusterSt *corev1.Secret, impersonatorSt *corev1.Secret, tag string) (*corev1.Secret, *corev1.Secret, error) {
		if tt.mockKarmadaErr != nil {
			return nil, nil, tt.mockKarmadaErr
		}
		return &corev1.Secret{Data: map[string][]byte{"a": []byte("s")}},
			&corev1.Secret{StringData: map[string]string{"u": "s"}}, nil
	})
}

func assertCreateCredentialsResults(t *testing.T, tt struct {
	name   string
	fields struct {
		MgrClient              client.Client
		KarmadaClient          *kubernetes.Clientset
		KarmadaVersionedClient *versioned.Clientset
	}
	args struct {
		k kubernetes.Interface
		t string
	}
	want           *corev1.Secret
	want1          *corev1.Secret
	wantErr        bool
	mockErr        error
	mockKarmadaErr error
}, got *corev1.Secret, got1 *corev1.Secret, err error) {
	if (err != nil) != tt.wantErr {
		t.Errorf("createCredentials() error = %v, wantErr %v", err, tt.wantErr)
		return
	}
	if !reflect.DeepEqual(got, tt.want) {
		t.Errorf("createCredentials() got = %v, want %v", got, tt.want)
	}
	if !reflect.DeepEqual(got1, tt.want1) {
		t.Errorf("createCredentials() got1 = %v, want %v", got1, tt.want1)
	}
}

func TestOperationClientCreateClusterObject(t *testing.T) {
	type fields struct {
		MgrClient              client.Client
		KarmadaClient          *kubernetes.Clientset
		KarmadaVersionedClient *versioned.Clientset
	}
	type args struct {
		buildOption OptionsOfCluster
		registerObj *v1alpha1.Cluster
	}
	tests := []struct {
		name        string
		fields      fields
		args        args
		wantErr     bool
		registerErr error
		apiErr      error
	}{
		{
			name: "fail test",
			fields: fields{
				MgrClient: CreatefakeClient([]byte(`{"api":"v"}`)),
			},
			args:        args{},
			wantErr:     true,
			registerErr: errors.New("test error"),
		},
		{
			name: "fail api",
			fields: fields{
				MgrClient: CreatefakeClient([]byte(`{}`)),
			},
			args:    args{},
			wantErr: true,
			apiErr:  errors.New("api error"),
		},
		{
			name:    "success",
			fields:  fields{},
			args:    args{},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &OperationClient{
				MgrClient:              tt.fields.MgrClient,
				KarmadaClient:          tt.fields.KarmadaClient,
				KarmadaVersionedClient: tt.fields.KarmadaVersionedClient,
			}
			patches := gomonkey.ApplyFunc(registerHostCluster,
				func(kclient *versioned.Clientset, cluster *v1alpha1.Cluster) error {
					if tt.registerErr != nil {
						return tt.registerErr
					}
					return nil
				}).ApplyFunc(createAPICluster, func(mgrClient client.Client, buildOption OptionsOfCluster) error {
				if tt.apiErr != nil {
					return tt.apiErr
				}
				return nil
			})
			defer patches.Reset()
			if err := c.createClusterObject(tt.args.buildOption, tt.args.registerObj); (err != nil) != tt.wantErr {
				t.Errorf("createClusterObject() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestGenerateClusterOptions(t *testing.T) {
	type args struct {
		name      string
		uid       string
		apiConfig *api.Config
		opt       JoinOptions
	}
	tests := []struct {
		name string
		args args
		want OptionsOfCluster
	}{
		{
			name: "generate cluster",
			args: args{
				name: "test",
				uid:  "1",
				apiConfig: &api.Config{
					CurrentContext: "context1",
					Clusters: map[string]*api.Cluster{
						"a": &api.Cluster{Server: "run"},
					},
					Contexts: map[string]*api.Context{
						"context1": &api.Context{
							Cluster: "a",
						},
					},
				},
				opt: JoinOptions{ClusterName: "cluster1", ClusterLabels: map[string]string{"a": "b"}},
			},
			want: OptionsOfCluster{
				ClusterName:        "test",
				ClusterID:          "1",
				ClusterLabels:      map[string]string{"a": "b"},
				ClusterAPIEndpoint: "run",
				ClusterTLSVerify:   false, // default value since it's not set in the test
				ClusterKubeconfig: &api.Config{
					CurrentContext: "context1",
					Contexts: map[string]*api.Context{
						"context1": &api.Context{
							Cluster: "a",
						},
					},
					Clusters: map[string]*api.Cluster{
						"a": &api.Cluster{Server: "run"},
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := generateClusterOptions(tt.args.name, tt.args.uid,
				tt.args.apiConfig, tt.args.opt); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("generateClusterOptions() = %v, want %v", got, tt.want)
			}
		})
	}
}
func TestOperationClientCheckClusterIDAndName(t *testing.T) {
	tests := getCheckClusterIDAndNameTestCases()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &OperationClient{
				MgrClient:              tt.fields.MgrClient,
				KarmadaClient:          tt.fields.KarmadaClient,
				KarmadaVersionedClient: tt.fields.KarmadaVersionedClient,
			}

			patches := applyCheckClusterIDAndNameMocks(tt)
			defer patches.Reset()

			got, err := c.checkClusterIDAndName(tt.args.name, tt.args.client)
			assertCheckClusterIDAndNameResults(t, tt, got, err)
		})
	}
}

func getCheckClusterIDAndNameTestCases() []struct {
	name   string
	fields struct {
		MgrClient              client.Client
		KarmadaClient          *kubernetes.Clientset
		KarmadaVersionedClient *versioned.Clientset
	}
	args struct {
		name   string
		client *kubernetes.Clientset
	}
	want    string
	wantErr bool
	mockErr error
	UIDErr  error
} {
	return []struct {
		name   string
		fields struct {
			MgrClient              client.Client
			KarmadaClient          *kubernetes.Clientset
			KarmadaVersionedClient *versioned.Clientset
		}
		args struct {
			name   string
			client *kubernetes.Clientset
		}
		want    string
		wantErr bool
		mockErr error
		UIDErr  error
	}{
		{
			name: "no repeat but err",
			fields: struct {
				MgrClient              client.Client
				KarmadaClient          *kubernetes.Clientset
				KarmadaVersionedClient *versioned.Clientset
			}{},
			want:    "",
			wantErr: true,
			args: struct {
				name   string
				client *kubernetes.Clientset
			}{
				name: "name",
			},
			mockErr: errors.New("yes"),
		},
		{
			name: "repeat no err",
			fields: struct {
				MgrClient              client.Client
				KarmadaClient          *kubernetes.Clientset
				KarmadaVersionedClient *versioned.Clientset
			}{},
			args: struct {
				name   string
				client *kubernetes.Clientset
			}{},
			wantErr: true,
			want:    "",
		},
		{
			fields: struct {
				MgrClient              client.Client
				KarmadaClient          *kubernetes.Clientset
				KarmadaVersionedClient *versioned.Clientset
			}{},
			name: "IsClusterUIDUnique",
			want: "",
			args: struct {
				name   string
				client *kubernetes.Clientset
			}{
				name: "user",
			},
			wantErr: true,
			UIDErr:  errors.New("news"),
		},
		{
			wantErr: true,
			name:    "IsClusterUIDUnique no err",
			fields: struct {
				MgrClient              client.Client
				KarmadaClient          *kubernetes.Clientset
				KarmadaVersionedClient *versioned.Clientset
			}{},

			want: "",
			args: struct {
				name   string
				client *kubernetes.Clientset
			}{
				name: "user",
			},
		},
	}
}

func applyCheckClusterIDAndNameMocks(tt struct {
	name   string
	fields struct {
		MgrClient              client.Client
		KarmadaClient          *kubernetes.Clientset
		KarmadaVersionedClient *versioned.Clientset
	}
	args struct {
		name   string
		client *kubernetes.Clientset
	}
	want    string
	wantErr bool
	mockErr error
	UIDErr  error
}) *gomonkey.Patches {
	return gomonkey.ApplyFunc(utils.IsClusterNameRepeated, func(name string,
		karmadaclient *versioned.Clientset) (bool, error) {
		if tt.mockErr != nil {
			return false, tt.mockErr
		}
		if tt.args.name == "user" {
			return false, nil
		}
		return true, nil
	}).ApplyFunc(utils.IsClusterUIDUnique, func(clusterclient kubernetes.Interface,
		karmadaclient *versioned.Clientset) (string, error) {
		if tt.UIDErr != nil {
			return "", tt.UIDErr
		}
		if tt.args.name == "success" {
			return "ui", nil
		}
		return "", nil
	})
}

func assertCheckClusterIDAndNameResults(t *testing.T, tt struct {
	name   string
	fields struct {
		MgrClient              client.Client
		KarmadaClient          *kubernetes.Clientset
		KarmadaVersionedClient *versioned.Clientset
	}
	args struct {
		name   string
		client *kubernetes.Clientset
	}
	want    string
	wantErr bool
	mockErr error
	UIDErr  error
}, got string, err error) {
	if (err != nil) != tt.wantErr {
		t.Errorf("checkClusterIDAndName() error = %v, wantErr %v", err, tt.wantErr)
		return
	}
	if got != tt.want {
		t.Errorf("checkClusterIDAndName() got = %v, want %v", got, tt.want)
	}
}

func TestOperationClientJoinHost(t *testing.T) {
	type fields struct {
		MgrClient              client.Client
		KarmadaClient          *kubernetes.Clientset
		KarmadaVersionedClient *versioned.Clientset
	}
	type args struct {
		ctx context.Context
	}
	tests := []struct {
		name       string
		fields     fields
		args       args
		mockErr    error
		karmadaErr error
	}{
		{
			name: "ok",
			fields: fields{
				MgrClient: CreatefakeClient([]byte(`{}`)),
			},
			args: args{
				ctx: context.TODO(),
			},
			mockErr:    errors.New("?"),
			karmadaErr: errors.New("t"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.ApplyFunc(os.Stat, func(path string) (fs.FileInfo, error) {
				if tt.mockErr != nil {
					return nil, tt.mockErr
				}
				return nil, nil
			}).ApplyFunc(utils.IsKarmadaServiceReady, func(karmadaclient *versioned.Clientset) error {
				return tt.karmadaErr
			})
			defer patches.Reset()
			c := &OperationClient{
				MgrClient:              tt.fields.MgrClient,
				KarmadaClient:          tt.fields.KarmadaClient,
				KarmadaVersionedClient: tt.fields.KarmadaVersionedClient,
			}
			c.JoinHost(tt.args.ctx)
		})
	}
}
