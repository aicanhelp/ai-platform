/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package mock

import fuyaov1beta1 "installer/pkg/fuyao/v1beta1/cluster"

// GetAllNodes mock function to get all nodes
func GetAllNodes() fuyaov1beta1.Nodes {
	master := &fuyaov1beta1.Node{
		Hostname: "master",
		User:     "fuyao",
		Password: []byte("000"),
		IP:       IP0,
		Port:     22,
	}

	worker := &fuyaov1beta1.Node{
		Hostname: "worker",
		User:     "fuyao",
		Password: []byte("000"),
		IP:       IP1,
		Port:     22,
	}

	registry := &fuyaov1beta1.Node{
		Hostname: "registry",
		User:     "fuyao",
		Password: []byte("000"),
		IP:       IP2,
		Port:     22,
	}

	nodes := fuyaov1beta1.Nodes{Masters: []*fuyaov1beta1.Node{master}, Workers: []*fuyaov1beta1.Node{worker},
		Registry: []*fuyaov1beta1.Node{registry}}
	return nodes
}

// GetOneMasterNode mock function to get one master node
func GetOneMasterNode() fuyaov1beta1.Nodes {
	master := &fuyaov1beta1.Node{
		Hostname: "master",
		User:     "fuyao",
		Password: []byte("000"),
		IP:       IP0,
		Port:     22,
	}

	nodes := fuyaov1beta1.Nodes{Masters: []*fuyaov1beta1.Node{master}}
	return nodes
}
