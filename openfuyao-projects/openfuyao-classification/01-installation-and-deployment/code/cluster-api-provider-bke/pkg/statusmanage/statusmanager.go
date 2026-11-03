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

package statusmanage

import (
	"os"
	"strconv"
	"strings"
	"sync"

	ctrl "sigs.k8s.io/controller-runtime"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/condition"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const DefaultAllowedFailedCount = 10

var (
	ReconcileAllowedFailedCount int
	bkeClusterStatusManager     = NewStatusManager()
	statusLogger                = log.With("name", "statusManager")
)

func DefaultStatusManager() *StatusManager {
	return bkeClusterStatusManager
}

func SetDefaultStatusManager(manager *StatusManager) {
	bkeClusterStatusManager = manager
}

// StatusManager is used to record the status
// 用来控制 BKECluster 的失败状态，使用单例模式运行，在BKECluster更新的末端调用
type StatusManager struct {
	// cmux sync.RWMutex for cluster
	cmux sync.RWMutex
	// cmux sync.Mutex for nodes
	nmux sync.RWMutex

	BKEClusterStatusMap map[string]*StatusRecord
	BKENodesStatusMap   map[string]map[string]*StatusRecord
}

func init() {
	env, b := os.LookupEnv("ALLOWED_FAILED_COUNT")
	if b {
		envAllowed, err := strconv.Atoi(env)
		if err != nil {
			ReconcileAllowedFailedCount = DefaultAllowedFailedCount
		} else {
			ReconcileAllowedFailedCount = envAllowed
		}
	} else {
		ReconcileAllowedFailedCount = DefaultAllowedFailedCount
	}
}

// LogInitConfig prints resolved startup configuration after process init.
// Call this from main() instead of init() to avoid triggering logger side
// effects during package initialization.
func LogInitConfig() {
	statusLogger.Infof("ReconcileAllowedFailedCount: %d", ReconcileAllowedFailedCount)
}

func NewStatusManager() *StatusManager {
	return &StatusManager{
		BKEClusterStatusMap: map[string]*StatusRecord{},
		BKENodesStatusMap:   map[string]map[string]*StatusRecord{},
	}
}

// SetStatus is used to set the status of BKECluster
func (b *StatusManager) SetStatus(bkeCluster *bkev1beta1.BKECluster, bkeNodes bkev1beta1.BKENodes) {
	b.recordBKEClusterStatus(bkeCluster)
	b.recordBKENodesStatus(bkeCluster, bkeNodes)
}

func (b *StatusManager) RemoveClusterStatusManagerCache(bkeCluster *bkev1beta1.BKECluster) {
	b.RemoveBKEClusterStatusCache(bkeCluster)
	b.RemoveNodesStatusCache(bkeCluster)
}

func (b *StatusManager) GetCtrlResult(bkeCluster *bkev1beta1.BKECluster) ctrl.Result {

	if bkeCluster.Status.ClusterStatus == bkev1beta1.ClusterPaused {
		return ctrl.Result{}
	}

	b.cmux.RLock()
	defer b.cmux.RUnlock()

	key := utils.ClientObjNS(bkeCluster)
	sr := b.BKEClusterStatusMap[key]

	if sr == nil {
		return ctrl.Result{}
	}

	return ctrl.Result{Requeue: sr.NeedRequeue}
}

func (b *StatusManager) GetNodesResult(bkeCluster *bkev1beta1.BKECluster, nodeIP string) bool {
	b.nmux.RLock()
	defer b.nmux.RUnlock()

	key := utils.ClientObjNS(bkeCluster)

	if sr, ok := b.BKENodesStatusMap[key]; ok {
		if sr[nodeIP] == nil {
			return true
		}
		return sr[nodeIP].NeedRequeue
	}

	return true

}

func (b *StatusManager) recordBKEClusterStatus(bkeCluster *bkev1beta1.BKECluster) {
	if _, ok := annotation.HasAnnotation(bkeCluster, annotation.StatusRecordAnnotationKey); !ok {
		return
	}
	defer annotation.RemoveAnnotation(bkeCluster, annotation.StatusRecordAnnotationKey)

	log := statusLogger.With("bkeCluster", utils.ClientObjNS(bkeCluster))

	state := string(bkeCluster.Status.ClusterStatus)
	if state == "" {
		return
	}
	key := utils.ClientObjNS(bkeCluster)

	b.cmux.Lock()
	defer b.cmux.Unlock()

	// 首先查询是否记录过
	sr := b.BKEClusterStatusMap[key]

	// debug
	defer func() {
		if sr.LatestFailedState != "" {
			log.Debugf("(cluster) Latest FailedState %s count: %d, Latest NormalState %s", sr.LatestFailedState, sr.StatusCount, sr.LatestNormalState)
		}
	}()

	// unhealth状态不再此处控制它不算正常也不算失败状态，就是需要一直重新入队的状态，phase返回的requeue优先级高于此处

	// 首次记录
	if sr == nil {
		sr = &StatusRecord{}
		b.BKEClusterStatusMap[key] = sr
	}

	// 不记录暂停状态
	if state == string(bkev1beta1.ClusterPaused) {
		log.Debugf("(cluster) ClusterPaused, skip record status")
		sr.NeedRequeue = false
		return
	}

	sr.SetCurrentClusterState(bkeCluster.Status.ClusterStatus)

	failedState := strings.HasSuffix(state, "Failed")

	// 正常的状态
	if !failedState {
		sr.SetLatestNormalState(state)
		sr.NeedRequeue = false
		// 重置失败计数：前一个 phase 已成功，下一个 phase 应获得全新的重试预算
		sr.Reset()
		return
	}

	// 失败的状态
	// 对比上次的失败状态 处理计数器
	if sr.Equal(state) {
		// 与上次一致 计数器+1
		sr.Inc()
		log.Debugf("(cluster) Equal latest FailedState %s, count inc to %d", state, sr.StatusCount)
	} else {
		// 与上次不一致 记录新的失败状态
		sr.Reset()
		sr.SetLatestFailedState(state)
		sr.Inc()
		log.Infof("(cluster) Refresh latest FailedState %s", state)
	}

	// 如果没有超过允许失败的次数，修改bkeCluster状态为上一次的正常状态
	// bkeCluster的状态使用phaseframe的钩子函数自动设置
	// 通常sr的latestNormalState为进入phase的初始状态
	// sr的latestFailedState为出来phase的最终状态
	// 在phaseframe中对于拥有失败状态的bkeCluster会再次入队处理
	// 在一定失败次数内对其状态进行修正，表现出的效果为，实际执行失败但显示正常，控制器重试一定次数后停止重试
	// 并暂停对该bkeCluster的调谐，直至spec被修改
	if sr.AllowFailed() {
		bkeCluster.Status.ClusterStatus = confv1beta1.ClusterStatus(sr.LatestNormalState)
		phaseutil.SyncStatusFields(bkeCluster)
		sr.NeedRequeue = true
		return
	} else {
		log.Infof("(cluster) The failedStatus %s occur more than %d times, not allow to retry", sr.LatestFailedState, ReconcileAllowedFailedCount)

		if sr.CurrentClusterState != bkev1beta1.ClusterUnhealthy && sr.CurrentClusterState != bkev1beta1.ClusterStatusReady {
			// CurrentClusterState 已经是 Failed 状态（由 post-hook SetClusterStatus 设置）
			// ClusterStatus 和 SyncStatusFields 已由 post-hook 完成，此处仅需设置 condition msg
			msg := string(sr.CurrentClusterState)

			v, ok := condition.HasCondition(bkev1beta1.ClusterHealthyStateCondition, bkeCluster)
			if ok && v != nil {
				condition.ConditionMark(bkeCluster, v.Type, confv1beta1.ConditionFalse, v.Reason, msg)
			}
		}
		// 超过限制次数后清空计数器
		sr.Reset()
		sr.NeedRequeue = false
		return
	}

}

func (b *StatusManager) RemoveBKEClusterStatusCache(bkeCluster *bkev1beta1.BKECluster) {
	b.cmux.Lock()
	defer b.cmux.Unlock()
	log := statusLogger.With("bkeCluster", utils.ClientObjNS(bkeCluster))
	key := utils.ClientObjNS(bkeCluster)
	delete(b.BKEClusterStatusMap, key)
	log.Infof("cluster %s status aready removed from status manager cache", key)
}

func (b *StatusManager) recordBKENodesStatus(bkeCluster *bkev1beta1.BKECluster, bkeNodes bkev1beta1.BKENodes) {
	if bkeNodes == nil || len(bkeNodes) == 0 {
		return
	}
	log := statusLogger.With("bkeCluster", utils.ClientObjNS(bkeCluster))

	key := utils.ClientObjNS(bkeCluster)

	b.nmux.Lock()
	defer b.nmux.Unlock()

	nodesStatusMap := b.BKENodesStatusMap[key]

	// 初始化
	if nodesStatusMap == nil {
		nodesStatusMap = map[string]*StatusRecord{}
		b.BKENodesStatusMap[key] = nodesStatusMap
	}

	// 不是第一次记录
	for i := range bkeNodes {
		b.recordSingleNodeState(&bkeNodes[i], nodesStatusMap, bkeNodes, log)
	}

}

func (b *StatusManager) RemoveNodesStatusCache(bkeCluster *bkev1beta1.BKECluster) {
	b.nmux.Lock()
	defer b.nmux.Unlock()
	log := statusLogger.With("bkeCluster", utils.ClientObjNS(bkeCluster))
	key := utils.ClientObjNS(bkeCluster)
	delete(b.BKENodesStatusMap, key)
	log.Infof("cluster %s nodes status aready removed from status manager cache", key)
}

func (b *StatusManager) recordSingleNodeState(
	bkeNode *confv1beta1.BKENode, nodesStatusMap map[string]*StatusRecord,
	bkeNodes bkev1beta1.BKENodes, log *log.Logger,
) {
	nodeIP := bkeNode.Spec.IP
	if !bkeNodes.GetNodeStateFlag(nodeIP, bkev1beta1.NodeStateNeedRecord) {
		return
	}
	defer bkeNodes.UnmarkNodeStateFlag(nodeIP, bkev1beta1.NodeStateNeedRecord)

	state := string(bkeNode.Status.State)
	if state == "" {
		return
	}

	if nodesStatusMap == nil {
		return
	}

	failedState := strings.HasSuffix(state, "Failed")
	sr := nodesStatusMap[nodeIP]

	// debug
	defer func() {
		if sr.LatestFailedState != "" {
			log.Debugf("(node %s) Latest FailedState %s count: %d, Latest NormalState %s", phaseutil.NodeInfo(bkeNode.ToNode()), sr.LatestFailedState, sr.StatusCount, sr.LatestNormalState)
		}
	}()

	if sr == nil {
		sr = &StatusRecord{}
		nodesStatusMap[nodeIP] = sr
	}

	// 正常的状态
	if !failedState {
		sr.SetLatestNormalState(state)
		return
	}

	// 失败的状态
	// 对比上次的失败状态 处理计数器
	if sr.Equal(state) {
		// 与上次一致 计数器+1
		sr.Inc()
		log.Debugf("(node %s) Equal latest FailedState %s, count inc to %d", phaseutil.NodeInfo(bkeNode.ToNode()), state, sr.StatusCount)
	} else {
		// 与上次不一致 记录新的失败状态
		sr.Reset()
		sr.SetLatestFailedState(state)
		sr.Inc()
		log.Infof("(node %s) Refresh latest FailedState %s", phaseutil.NodeInfo(bkeNode.ToNode()), state)
	}

	// 如果没有超过允许失败的次数，修改bkeCluster状态为上一次的正常状态
	// bkeCluster的状态使用phaseframe的钩子函数自动设置
	// 通常sr的latestNormalState为进入phase的初始状态
	// sr的latestFailedState为出来phase的最终状态
	// 在phaseframe中对于拥有失败状态的bkeCluster会再次入队处理
	// 在一定失败次数内对其状态进行修正，表现出的效果为，实际执行失败但显示正常，控制器重试一定次数后停止重试
	// 并暂停对该bkeCluster的调谐，直至spec被修改
	if sr.AllowFailed() {
		bkeNodes.SetNodeState(nodeIP, confv1beta1.NodeState(sr.LatestNormalState))
		sr.NeedRequeue = true
		return
	} else {
		log.Infof("(node %s) The failedStatus %s occur more than %d times, not allow to retry", phaseutil.NodeInfo(bkeNode.ToNode()), sr.LatestFailedState, ReconcileAllowedFailedCount)
		// 超过限制次数后清空计数器
		sr.Reset()
		sr.NeedRequeue = false
		// 标记失败，这将会让后续所有调谐跳过该节点
		bkeNodes.SetNodeState(nodeIP, confv1beta1.NodeState(state))
		bkeNodes.MarkNodeStateFlag(nodeIP, bkev1beta1.NodeFailedFlag)
	}
}

func (b *StatusManager) RemoveSingleNodeStatusCache(bkeCluster *bkev1beta1.BKECluster, nodeIP string) {
	b.nmux.Lock()
	defer b.nmux.Unlock()

	log := statusLogger.With("bkeCluster", utils.ClientObjNS(bkeCluster))
	key := utils.ClientObjNS(bkeCluster)
	nodesStatusMap := b.BKENodesStatusMap[key]
	if nodesStatusMap == nil {
		return
	}
	delete(nodesStatusMap, nodeIP)
	log.Infof("node %s status aready removed from status manager cache", nodeIP)
}
