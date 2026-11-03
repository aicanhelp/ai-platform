/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package clusterutil

import (
	"context"
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common"
	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/validation"
	bkesource "gopkg.openfuyao.cn/cluster-api-provider-bke/common/source"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
)

func AvailableLoadBalancerEndPoint(endPoint confv1beta1.APIEndpoint, nodes bkenode.Nodes) bool {
	if endPoint.IsValid() {
		host := endPoint.Host
		if nodes.Filter(bkenode.FilterOptions{"IP": host}).Length() == 0 {
			return true
		}
	}
	return false
}

// LoadBalanceType returns the cluster load balance mode, empty value defaults to "ha"
func LoadBalanceType(cfg *confv1beta1.BKEConfig) string {
	if cfg == nil || cfg.Cluster.LoadBalanceType == "" {
		return "ha"
	}
	return cfg.Cluster.LoadBalanceType
}

// GetIngressConfig 获取bkecluster ingress(ELB) addon配置
func GetIngressConfig(addons []confv1beta1.Product) (string, []string) {
	var lbVIP string
	var lbNodes []string
	var elbAddon *confv1beta1.Product
	for _, addon := range addons {
		if addon.Name == "beyondELB" {
			elbAddon = &addon
			break
		}
	}
	if elbAddon != nil {
		lbVIP = elbAddon.Param["lbVIP"]
		lbNodes = strings.Split(elbAddon.Param["lbNodes"], ",")
		// 去除空字符串
		lbNodes = utils.TrimSpaceSlice(lbNodes)
	}
	return lbVIP, lbNodes
}

func BKEConfigCmKey() client.ObjectKey {
	return client.ObjectKey{
		Namespace: "cluster-system",
		Name:      common.BKEClusterConfigFileName,
	}
}

func GetBKEConfigCMData(ctx context.Context, c client.Client) (map[string]string, error) {
	config := &corev1.ConfigMap{}
	err := c.Get(ctx, BKEConfigCmKey(), config)
	if err != nil {
		return nil, err
	}
	return config.Data, nil
}

// BuildYumRepoDownloadBaseURL builds download base URL by HTTP repo settings.
// When domain is not reachable, fallback to HTTPRepo.Ip (offline env).
// When HTTPRepo.Prefix is configured, keep prefix in the final URL directly.
// Otherwise, fallback to legacy GetCustomDownloadPath behavior.
func BuildYumRepoDownloadBaseURL(cfg bkeinit.BkeConfig) string {
	baseURL := validation.ResolveReachableHTTPRepoBaseURL(cfg.Cluster.HTTPRepo)
	if cfg.Cluster.HTTPRepo.Prefix != "" {
		return fmt.Sprintf("%s/%s", baseURL, cfg.Cluster.HTTPRepo.Prefix)
	}
	return bkesource.GetCustomDownloadPath(baseURL)
}
