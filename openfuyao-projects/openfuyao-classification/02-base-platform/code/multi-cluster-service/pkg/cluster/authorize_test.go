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
	v1 "k8s.io/api/rbac/v1"
	"k8s.io/client-go/kubernetes"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"openfuyao.com/multi-cluster-service/pkg/utils"
)

func TestOperationClientAuthorize(t *testing.T) {
	testCases := createAuthorizeTestCases()
	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			c := createTestOperationClient(tt.fields)
			patches := setupAuthorizeMocks(tt)
			defer patches.Reset()

			if err := c.Authorize(tt.args.ctx, tt.args.opt); (err != nil) != tt.wantErr {
				t.Errorf("Authorize() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

// Helper types
type authorizeFields struct {
	MgrClient              client.Client
	KarmadaClient          *kubernetes.Clientset
	KarmadaVersionedClient *versioned.Clientset
}

type authorizeArgs struct {
	ctx context.Context
	opt AuthorizeOptions
}

type authorizeTestCase struct {
	name            string
	fields          authorizeFields
	args            authorizeArgs
	wantErr         bool
	mockListErr     error
	mockPlatformErr error
	mockClusterErr  error
	mockList        map[string]*v1.ClusterRoleList
}

// Helper functions
func createAuthorizeTestCases() []authorizeTestCase {
	return []authorizeTestCase{
		createSuccessTestCase(),
		createListErrorTestCase(),
		createPlatformErrorTestCase(),
		createClusterErrorTestCase(),
	}
}

func createSuccessTestCase() authorizeTestCase {
	return authorizeTestCase{
		name: "success",
		fields: authorizeFields{
			KarmadaClient: &kubernetes.Clientset{},
		},
		args: authorizeArgs{
			ctx: context.TODO(),
			opt: AuthorizeOptions{
				Users: []*IdentityDescriptor{
					{
						IdentityName:   "admin",
						PlatformAdmin:  true,
						MemberClusters: []string{"cluster1", "cluster2"},
					},
				},
			},
		},
		wantErr:  false,
		mockList: map[string]*v1.ClusterRoleList{},
	}
}

func createListErrorTestCase() authorizeTestCase {
	return authorizeTestCase{
		name: "Error Retrieving clusterrole list",
		fields: authorizeFields{
			KarmadaClient: &kubernetes.Clientset{},
		},
		args: authorizeArgs{
			ctx: context.TODO(),
		},
		wantErr:     true,
		mockListErr: errors.New("list error"),
	}
}

func createPlatformErrorTestCase() authorizeTestCase {
	return authorizeTestCase{
		name: "Error Updating platform",
		fields: authorizeFields{
			KarmadaClient: &kubernetes.Clientset{},
		},
		args: authorizeArgs{
			ctx: context.TODO(),
			opt: AuthorizeOptions{
				Users: []*IdentityDescriptor{
					{
						IdentityName:   "user",
						PlatformAdmin:  true,
						MemberClusters: []string{"cluster3"},
					},
				},
			},
		},
		wantErr:         true,
		mockPlatformErr: errors.New("platform error"),
		mockList:        map[string]*v1.ClusterRoleList{},
	}
}

func createClusterErrorTestCase() authorizeTestCase {
	return authorizeTestCase{
		name: "Error Updating Cluster",
		fields: authorizeFields{
			KarmadaClient: &kubernetes.Clientset{},
		},
		args: authorizeArgs{
			ctx: context.TODO(),
			opt: AuthorizeOptions{
				Users: []*IdentityDescriptor{
					{
						IdentityName:   "hw",
						PlatformAdmin:  false,
						MemberClusters: []string{"cluster"},
					},
				},
			},
		},
		wantErr:        true,
		mockClusterErr: errors.New("cluster error"),
		mockList:       map[string]*v1.ClusterRoleList{},
	}
}

func createTestOperationClient(fields authorizeFields) *OperationClient {
	return &OperationClient{
		MgrClient:              fields.MgrClient,
		KarmadaClient:          fields.KarmadaClient,
		KarmadaVersionedClient: fields.KarmadaVersionedClient,
	}
}

func setupAuthorizeMocks(tt authorizeTestCase) *gomonkey.Patches {
	return gomonkey.ApplyFunc(utils.GetClusterRoleList,
		func(c kubernetes.Interface) (map[string]*v1.ClusterRoleList, error) {
			if tt.mockListErr != nil {
				return nil, tt.mockListErr
			}
			return tt.mockList, nil
		}).ApplyFunc(utils.UpdatePlatformRole, func(c kubernetes.Interface,
		crList *v1.ClusterRoleList, userList []string) error {
		if tt.mockPlatformErr != nil {
			return tt.mockPlatformErr
		}
		return nil
	}).ApplyFunc(utils.UpdateClusterRole, func(c kubernetes.Interface,
		crList *v1.ClusterRoleList, userInfo map[string][]string) error {
		if tt.mockClusterErr != nil {
			return tt.mockClusterErr
		}
		return nil
	})
}
