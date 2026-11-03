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

	"sigs.k8s.io/controller-runtime/pkg/client"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

// ChartRepositoryConfigFromBKECluster resolves the chart repository URL and
// authentication from BKECluster.spec.clusterConfig.cluster.chartRepo.
func ChartRepositoryConfigFromBKECluster(
	ctx context.Context,
	localClient client.Client,
	bkeCluster *bkev1beta1.BKECluster,
) (string, *AuthConfig, error) {
	auth := NewAuthConfig()
	if bkeCluster == nil || bkeCluster.Spec.ClusterConfig == nil {
		return "", auth, nil
	}

	cfg := bkeinit.BkeConfig(*bkeCluster.Spec.ClusterConfig)
	chartRepo, err := cfg.ResolveReachableChartRepo()
	if err != nil {
		return "", nil, err
	}

	c := &Client{Log: log.With("module", "chart-auth"), Ctx: ctx}
	tlsAuth, err := c.getChartRepoTLSCerts(cfg, bkeCluster.Namespace, localClient)
	if err != nil {
		return "", nil, err
	}
	username, password, err := c.getChartRepoLoginInfo(cfg, bkeCluster.Namespace, localClient)
	if err != nil {
		return "", nil, err
	}
	return chartRepo, tlsAuth.SetUsername(username).SetPassword(password).
		SetInsecureSkipTLSVerify(cfg.Cluster.ChartRepo.InsecureSkipTLSVerify), nil
}
