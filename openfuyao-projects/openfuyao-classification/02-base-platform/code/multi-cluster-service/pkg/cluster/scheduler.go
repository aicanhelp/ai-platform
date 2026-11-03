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

// Package cluster provides a suite og tools adn utilities for managing cluster objects.
package cluster

import (
	"context"
	"encoding/json"
	"fmt"

	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"

	"openfuyao.com/multi-cluster-service/api/v1beta1"
	"openfuyao.com/multi-cluster-service/pkg/utils"
	"openfuyao.com/multi-cluster-service/pkg/zlog"
)

// MemberAuthPolicy configures and applies the platform policies for user list based on the provided cluster info.
func (c *OperationClient) MemberAuthPolicy(data []byte, clusterTag string, method string) error {
	var err error

	if method != "join" && method != "unjoin" {
		return fmt.Errorf("unSupported Operation %s", method)
	}
	if clusterTag == "host" {
		zlog.LogInfof("Host Cluster no need to check authpolicy.")
		return nil
	}

	identityInfo := IdentityCollection{}
	identityInfo.UserInfo = make(map[string]*IdentityDescriptor)
	err = json.Unmarshal(data, &identityInfo)
	if err != nil {
		return fmt.Errorf("invalid user info data")
	}

	// retrieving kubeconfig from cluster object
	clusterKClient, err := utils.RetriveKubeconfigClient(c.MgrClient, clusterTag)
	if err != nil {
		zlog.LogErrorf("Error Creating cluster client from kubeconfig data: %v", err)
		return err
	}

	for name, obj := range identityInfo.UserInfo {
		if name == "admin" {
			continue
		}
		if err = utils.GeneratePlatformAuthPolicy(clusterKClient, obj.IdentityName, method, obj.PlatformAdmin); err != nil {
			zlog.LogErrorf("Error operating user %s clusterrole in cluster %s: %v", name, clusterTag, err)
		}
	}
	return err
}

// CheckUserKubeconfigAccess check the user access for checking cluster-admin kubeconfig.
func (c *OperationClient) CheckUserKubeconfigAccess(uname, cname string) (bool, json.RawMessage, error) {
	clusterObj := &v1beta1.Cluster{}
	if err := c.MgrClient.Get(context.TODO(), types.NamespacedName{Name: cname}, clusterObj); err != nil {
		zlog.LogErrorf("Error Retrieving cluster object from MgrClient : %v", err)
		return false, nil, err
	}
	// retrieving kubeconfig from cluster object
	clusterKClient, err := utils.RetriveKubeconfigClient(c.MgrClient, cname)
	if err != nil {
		zlog.LogErrorf("Error Creating cluster client from kubeconfig data: %v", err)
		return false, nil, err
	}

	pass, err := checkClusterrolebinding(clusterKClient, uname)
	if err != nil {
		return false, nil, err
	}

	if !pass {
		return false, nil, nil
	} else {
		return true, json.RawMessage(clusterObj.Spec.OutClusterConfig.Raw), nil
	}

}

func checkClusterrolebinding(c kubernetes.Interface, uname string) (bool, error) {
	pAdminName := fmt.Sprintf("%s-%s", uname, "platform-admin")
	cAdminName := fmt.Sprintf("%s-%s", uname, "cluster-admin")

	clusterroleList, err := c.RbacV1().ClusterRoleBindings().List(context.TODO(), v1.ListOptions{})
	if err != nil {
		zlog.LogError("Error Retrieving clusterrolebinding")
		return false, err
	}

	for _, item := range clusterroleList.Items {
		if item.Name == pAdminName || item.Name == cAdminName {
			return true, nil
		}
	}

	return false, nil
}
