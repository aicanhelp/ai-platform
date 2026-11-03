/******************************************************************
 * Copyright (c) 2024 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package node

import (
	"gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/utils"
)

type NodeOperate string

const (
	CreateNode  NodeOperate = "create"
	UpdateNode  NodeOperate = "update"
	RemoveNode  NodeOperate = "delete"
	UpgradeNode NodeOperate = "upgrade"
)

type NodeTransfer struct {
	Node    *v1beta1.Node
	Operate NodeOperate
}

// CompareBKEConfigNode compare nodes in []v1beta1.Node
func CompareBKEConfigNode(old []v1beta1.Node, dst []v1beta1.Node) ([]*NodeTransfer, bool) {
	var transferNode []*NodeTransfer
	for _, p := range dst {
		t := p.DeepCopy()
		if pp, ok := IsExistInNodeList(p, old); !ok {
			transferNode = append(transferNode, &NodeTransfer{Node: t, Operate: CreateNode})
		} else {
			if IsNodeUpdate(&p, pp) {
				transferNode = append(transferNode, &NodeTransfer{Node: t, Operate: UpdateNode})
				continue
			}
			if !IsNodeEqual(&p, pp) {
				transferNode = append(transferNode, &NodeTransfer{Node: t, Operate: UpdateNode})
			}
		}
	}
	for _, p := range old {
		t := p.DeepCopy()
		if _, ok := IsExistInNodeList(p, dst); !ok {
			transferNode = append(transferNode, &NodeTransfer{Node: t, Operate: RemoveNode})
		}
	}
	return transferNode, len(transferNode) > 0
}

func IsExistInNodeList(p v1beta1.Node, dst []v1beta1.Node) (*v1beta1.Node, bool) {
	for i, d := range dst {
		if d.IP == p.IP {
			return &dst[i], true
		}
	}
	return nil, false
}

func IsNodeEqual(p, d *v1beta1.Node) bool {
	if p.IP != d.IP {
		return false
	}
	if p.Hostname != d.Hostname {
		return false
	}
	if !utils.SliceEqualString(p.Role, d.Role) {
		return false
	}
	// todo add more compare
	return true
}

func IsNodeUpdate(p, d *v1beta1.Node) bool {
	if p.Hostname != d.Hostname {
		return true
	}
	if p.Port != d.Port {
		return true
	}
	if p.Username != d.Username {
		return true
	}
	if p.Password != d.Password {
		return true
	}

	if !utils.SliceEqualString(p.Role, d.Role) {
		return true
	}
	// todo compare k8s component
	return false
}
