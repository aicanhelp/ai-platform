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

package validation

import (
	"github.com/pkg/errors"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
)

// NoMasterNodeError returns an error indicating that no master node was found.
func NoMasterNodeError() error {
	return errors.Errorf("nodes at least one node contain %q or %q role", node.MasterNodeRole, node.MasterWorkerNodeRole)
}

// MasterNodeOddError returns an error indicating that the master node count must be odd.
func MasterNodeOddError() error {
	return errors.Errorf("nodes %q and %q role node count must be odd", node.MasterNodeRole, node.MasterWorkerNodeRole)
}

// NoEtcdNodeError returns an error indicating that no etcd node was found.
func NoEtcdNodeError() error {
	return errors.Errorf(
		"nodes at least one %q or %q role node contain %q role",
		node.MasterNodeRole, node.MasterWorkerNodeRole, node.EtcdNodeRole)
}

// NoWorkerNodeError returns an error indicating that no worker node was found.
func NoWorkerNodeError() error {
	return errors.Errorf("nodes at least one %q or %q role node", node.WorkerNodeRole, node.MasterWorkerNodeRole)
}
