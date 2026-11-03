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

package join

import (
	"fmt"
	"reflect"
	"runtime"
	"sync"

	"installer/pkg/constants"
	"installer/pkg/fuyao/v1beta1/cluster"
	"installer/pkg/utils"
	"installer/pkg/zlog"
)

// NodeJoinExecutor is node join cluster executor
type NodeJoinExecutor struct {
	nodes   cluster.Nodes
	operate *NodeJoinClusterOperate

	RoutineMaximum *utils.GoRoutineLimit
}

// NewNodeJoinExecutor is new NodeJoinExecutor
func NewNodeJoinExecutor(nodes cluster.Nodes, operate *NodeJoinClusterOperate) *NodeJoinExecutor {
	return &NodeJoinExecutor{
		nodes:          nodes,
		operate:        operate,
		RoutineMaximum: utils.NewGoRoutineLimits(),
	}
}

type function func(node *cluster.Node) error

// executor 执行operate
func (n *NodeJoinExecutor) executor(node *cluster.Node, funcSlice []function, wg *sync.WaitGroup) {
	defer wg.Done()
	defer n.RoutineMaximum.Done()

	for _, def := range funcSlice {
		zlog.Debugf("exec %s on %s", runtime.FuncForPC(reflect.ValueOf(def).Pointer()).Name(), node.IP)
		err := def(node)
		if err != nil {
			zlog.Fatalf("error:", err)
		}
	}
}

// masterJoinCluster master节点加入集群
func (n *NodeJoinExecutor) masterJoinCluster(execOperate string) error {
	if len(n.nodes.Masters) == masterCountMin && execOperate == constants.AddNodes {
		return nil
	}

	if len(n.nodes.Masters) == masterCountMin && execOperate == constants.InstallCluster {
		if err := n.operate.removeTaint(n.nodes.Masters[0]); err != nil {
			return fmt.Errorf("failed to remove taint %v", err)
		}
		if err := n.operate.masterNodeAddLabels(n.nodes.Masters[0]); err != nil {
			return fmt.Errorf("failed to add labels: %v", err)
		}
		return nil
	}
	if err := n.operate.downloadCertFromRemote(n.nodes.Masters[0]); err != nil {
		return fmt.Errorf("failed download cert %v", err)
	}

	// 一个线程内顺序执行的操作
	operateSlice := []function{
		n.operate.copyCertToRemote,
		n.operate.masterJoinCluster,
		n.operate.generateKubeConfig,
		n.operate.removeTaint,
		n.operate.masterNodeAddLabels,
	}

	for _, master := range n.nodes.Masters[1:] {
		for _, def := range operateSlice {
			zlog.Debugf("exec %s on %s", runtime.FuncForPC(reflect.ValueOf(def).Pointer()).Name(), master.IP)
			err := def(master)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func (n *NodeJoinExecutor) workerJoinCluster(wg *sync.WaitGroup) {
	operateSlice := []function{
		n.operate.workerJoinCluster,
		n.operate.copyKubeConfigToWorker,
		n.operate.workerNodeAddLabels,
	}

	for _, worker := range n.nodes.Workers {
		wg.Add(1)
		n.RoutineMaximum.Add()
		go n.executor(worker, operateSlice, wg)
	}
}
