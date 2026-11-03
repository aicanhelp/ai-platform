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

	"openfuyao.com/multi-cluster-service/pkg/utils"
	"openfuyao.com/multi-cluster-service/pkg/zlog"
)

// Authorize updates the access permissions based on the provided identity information.
func (c *OperationClient) Authorize(ctx context.Context, opt AuthorizeOptions) error {
	crMap, err := utils.GetClusterRoleList(c.KarmadaClient)
	if err != nil {
		zlog.LogErrorf("Error Retrieving clusterrole list : %v", err)
		return err
	}

	var idPlatformList []string
	idClusterList := map[string][]string{}
	for _, obj := range opt.Users {
		if obj.PlatformAdmin {
			idPlatformList = append(idPlatformList, obj.IdentityName)
		} else {
			idClusterList[obj.IdentityName] = obj.MemberClusters
		}
	}

	// check platform role first
	err = utils.UpdatePlatformRole(c.KarmadaClient, crMap["platform"], idPlatformList)
	if err != nil {
		zlog.LogErrorf("Error Updating platform role : %v", err)
		return err
	}

	// check cluster role
	err = utils.UpdateClusterRole(c.KarmadaClient, crMap["cluster"], idClusterList)
	if err != nil {
		zlog.LogErrorf("Error Updating cluster role : %v", err)
		return err
	}
	return nil
}
