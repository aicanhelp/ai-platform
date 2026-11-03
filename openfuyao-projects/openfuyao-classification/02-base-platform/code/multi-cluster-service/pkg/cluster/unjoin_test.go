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
	"errors"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/karmada-io/karmada/pkg/generated/clientset/versioned"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"openfuyao.com/multi-cluster-service/api/v1beta1"
	"openfuyao.com/multi-cluster-service/pkg/utils"
)

func TestOperationClientUnjoin(t *testing.T) {
	tests := createUnjoinTestCases()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := setupUnjoinMocks(tt)
			defer patches.Reset()

			c := createTestOperationClientUnjoin(tt.fields)
			if err := c.Unjoin(tt.args.ctx, tt.args.opt); (err != nil) != tt.wantErr {
				t.Errorf("Unjoin() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestDeleteAPICluster(t *testing.T) {
	tests := createDeleteAPIClusterTestCases()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := deleteAPICluster(tt.args.mgrClient, tt.args.tag); (err != nil) != tt.wantErr {
				t.Errorf("deleteAPICluster() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

// Helper types
type unjoinFields struct {
	MgrClient              client.Client
	KarmadaClient          *kubernetes.Clientset
	KarmadaVersionedClient *versioned.Clientset
}

type unjoinArgs struct {
	ctx context.Context
	opt UnjoinOptions
}

type unjoinTestCase struct {
	name       string
	fields     unjoinFields
	args       unjoinArgs
	wantErr    bool
	mockUID    error
	mockUn     error
	mockDelete error
	mockKR     error
	mockRK     error
	mockDCR    error
	mockApi    error
}

type deleteAPIClusterArgs struct {
	mgrClient client.Client
	tag       string
}

type deleteAPIClusterTestCase struct {
	name    string
	args    deleteAPIClusterArgs
	wantErr bool
}

// Helper functions
func createUnjoinTestCases() []unjoinTestCase {
	return []unjoinTestCase{
		createUnjoinErrorTestCase("exist", errors.New("cuo"), "mockUID"),
		createUnjoinErrorTestCase("Unregister", errors.New("n"), "mockUn"),
		createUnjoinErrorTestCase("delete cluster", errors.New("diu"), "mockDelete"),
		createUnjoinErrorTestCase("delete karmada resource", errors.New("lei"), "mockKR"),
		createUnjoinErrorTestCase("RetriveKubeconfigClient", errors.New("lou"), "mockRK"),
		createUnjoinErrorTestCase("DeleteClusterResources", errors.New("mou"), "mockDCR"),
		createUnjoinErrorTestCase("deleteAPICluster", errors.New("a"), "mockApi"),
		createSuccessUnjoinTestCase(),
	}
}

func createUnjoinErrorTestCase(name string, mockErr error, mockType string) unjoinTestCase {
	tc := unjoinTestCase{
		name:    name,
		wantErr: true,
		fields: unjoinFields{
			KarmadaClient: &kubernetes.Clientset{},
		},
	}

	switch mockType {
	case "mockUID":
		tc.mockUID = mockErr
	case "mockUn":
		tc.mockUn = mockErr
	case "mockDelete":
		tc.mockDelete = mockErr
	case "mockKR":
		tc.mockKR = mockErr
	case "mockRK":
		tc.mockRK = mockErr
	case "mockDCR":
		tc.mockDCR = mockErr
	case "mockApi":
		tc.mockApi = mockErr
	default:
		tc.wantErr = true
	}

	return tc
}

func createSuccessUnjoinTestCase() unjoinTestCase {
	return unjoinTestCase{
		name:    "all",
		wantErr: false,
		fields: unjoinFields{
			KarmadaClient: &kubernetes.Clientset{},
		},
	}
}

func createDeleteAPIClusterTestCases() []deleteAPIClusterTestCase {
	return []deleteAPIClusterTestCase{
		{
			name: "nice",
			args: deleteAPIClusterArgs{
				mgrClient: MakefakeClient(v1beta1.Cluster{
					ObjectMeta: metav1.ObjectMeta{
						Name: "test-cluster",
					},
					Spec: v1beta1.ClusterSpec{
						KubeConfig: runtime.RawExtension{},
					},
				}),
				tag: "test-cluster",
			},
			wantErr: false,
		},
		{
			name: "error",
			args: deleteAPIClusterArgs{
				mgrClient: MakefakeClient(v1beta1.Cluster{
					ObjectMeta: metav1.ObjectMeta{
						Name: "ta",
					},
				}),
				tag: "ter",
			},
			wantErr: true,
		},
	}
}

func createTestOperationClientUnjoin(fields unjoinFields) *OperationClient {
	return &OperationClient{
		MgrClient:              fields.MgrClient,
		KarmadaClient:          fields.KarmadaClient,
		KarmadaVersionedClient: fields.KarmadaVersionedClient,
	}
}

func setupUnjoinMocks(tt unjoinTestCase) *gomonkey.Patches {
	return gomonkey.ApplyFunc(utils.IsClusterUIDValid, func(karmadaclient *versioned.Clientset,
		tag string) (bool, error) {
		if tt.mockUID != nil {
			return true, tt.mockUID
		}
		return true, nil
	}).ApplyFunc(unregsiterHostCluster, func(kclient *versioned.Clientset, tag string) error {
		if tt.mockUn != nil {
			return tt.mockUn
		}
		return nil
	}).ApplyFunc(utils.IsClusterDeleted, func(karmadaclient *versioned.Clientset, tag string) error {
		if tt.mockDelete != nil {
			return tt.mockDelete
		}
		return nil
	}).ApplyFunc(utils.DeleteKarmadaResources, func(c kubernetes.Interface, tag string) error {
		if tt.mockKR != nil {
			return tt.mockKR
		}
		return nil
	}).ApplyFunc(utils.RetriveKubeconfigClient, func(mgrClient client.Client,
		clusterTag string) (*kubernetes.Clientset, error) {
		if tt.mockRK != nil {
			return &kubernetes.Clientset{}, tt.mockRK
		}
		return &kubernetes.Clientset{}, nil
	}).ApplyFunc(utils.DeleteClusterResources, func(c kubernetes.Interface, tag string) error {
		if tt.mockDCR != nil {
			return tt.mockDCR
		}
		return nil
	}).ApplyFunc(deleteAPICluster, func(mgrClient client.Client, tag string) error {
		if tt.mockApi != nil {
			return tt.mockApi
		}
		return nil
	})
}
