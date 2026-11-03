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

package label

// label key
const (
	// WorkerNodeHost  worker node marked label name,value is the node host
	// Used to tell the BKEMachine that this node has been bound
	WorkerNodeHost = "bke.bocloud.com/worker-node"

	// MasterNodeHost master node marked label name,value is the node host
	// same as WorkerNodeHost
	MasterNodeHost = "bke.bocloud.com/master-node"

	// NodeRoleMasterLabel master node role label name
	NodeRoleMasterLabel = "node-role.kubernetes.io/master"
	// NodeRoleNodeLabel node role label name
	NodeRoleNodeLabel = "node-role.kubernetes.io/node"

	// NodeRoleControlPlaneLabelKey control plane role label key
	NodeRoleControlPlaneLabelKey = "node-role.kubernetes.io/control-plane"
	// NodeRoleControlPlaneLabelValue control plane role label value
	NodeRoleControlPlaneLabelValue = "control-plane"
	// NodeRoleWorkerLabelKey worker role label name
	NodeRoleWorkerLabelKey = "node-role.kubernetes.io/worker"
	// NodeRoleWorkerLabelValue worker role label value
	NodeRoleWorkerLabelValue = "worker"

	// AlertLabelKey alert label key
	AlertLabelKey = "kubernetes.io/alert"
	// AlertLabelValue alert label value
	AlertLabelValue = "elastalert"

	// BareMetalLabelKey baremetal label key
	BareMetalLabelKey = "kubernetes.customized/bocloud_custom_bare_metal"

	// BeyondELBLabelKey beyond elb label key
	BeyondELBLabelKey   = "nodetype"
	BeyondELBLabelValue = "loadbalance"

	// BocVersionLabelKey bocloud version label key
	BocVersionLabelKey = "kubernetes.customized/bocloud_custom_version"

	// ScriptsLabelKey scripts label key
	ScriptsLabelKey = "bke.bocloud.com/scripts"
)
