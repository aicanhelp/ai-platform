/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package utils

import (
	"fmt"
	"reflect"
	"time"
	_ "time/tzdata" // 嵌入时区数据

	"gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// IsChanClosed 安全判断通道是否关闭
// 修复点：移除 SelectRecv 类型 case 的 Send 字段赋值
func IsChanClosed(channel interface{}) bool {
	// 1. 校验入参为通道类型（保留原有panic逻辑）
	chanType := reflect.TypeOf(channel)
	if chanType.Kind() != reflect.Chan {
		panic("only channels error!")
	}

	caseRecv := reflect.SelectCase{
		Dir:  reflect.SelectRecv, // 接收操作（等同于 reflect.RecvDir）
		Chan: reflect.ValueOf(channel),
	}
	caseDefault := reflect.SelectCase{Dir: reflect.SelectDefault} // 非阻塞默认分支

	index, _, ok := reflect.Select([]reflect.SelectCase{caseRecv, caseDefault})

	return index == 0 && !ok
}

// ConvertToChinaTime 转换时间
func ConvertToChinaTime(createTimestamp metav1.Time) (string, error) {
	chinaTimeZone, err := time.LoadLocation("Asia/Shanghai")
	if err != nil {
		return "", fmt.Errorf("error loading location: %w", err)
	}

	chinaTime := createTimestamp.In(chinaTimeZone)
	return chinaTime.Format("2006-01-02 15:04:05"), nil
}

const (
	clusterAPIAddonName                = "cluster-api"
	clusterAPIManagerAppliedAnnotation = "bke.bocloud.com/cluster-api-manager-applied"
)

// ExtractAddons extract addons info from cluster
func ExtractAddons(cluster v1beta1.BKECluster) ([]string, error) {
	if cluster.Spec.ClusterConfig == nil {
		return nil, fmt.Errorf("cluster config is nil")
	}
	originAddons := cluster.Spec.ClusterConfig.Addons
	addons := make([]string, len(originAddons))
	for i, addon := range originAddons {
		addons[i] = addon.Name
	}

	return addons, nil
}

// IsAdminCluster reports whether the cluster should be treated as a management cluster.
// Self-managed CRs created by 004-manage-bkecluster.yaml omit addons but set
// cluster-api-manager-applied=true, so both signals are checked.
func IsAdminCluster(cluster v1beta1.BKECluster, addons []string) bool {
	for _, name := range addons {
		if name == clusterAPIAddonName {
			return true
		}
	}
	if cluster.Annotations != nil && cluster.Annotations[clusterAPIManagerAppliedAnnotation] == "true" {
		return true
	}
	return false
}

// FindExtraNodes 找多出的ip
func FindExtraNodes(bcNode []v1beta1.Node, realNode []v1.Node) []v1beta1.Node {
	extraNodes := make([]v1beta1.Node, 0)

	// 将 nodes 转换为 map 方便快速查找
	nodeMap := make(map[string]bool)
	for _, node := range realNode {
		var ip string
		for _, addr := range node.Status.Addresses {
			if addr.Type == v1.NodeInternalIP {
				ip = addr.Address
				break
			}
		}
		nodeMap[ip] = true // 节点有 ip 字段作为唯一标识
	}

	for _, node := range bcNode {
		if !nodeMap[node.IP] {
			extraNodes = append(extraNodes, node)
		}
	}
	return extraNodes
}
