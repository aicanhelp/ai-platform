/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package command

import (
	"context"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

// DefaultTTLJanitorInterval is the default polling interval of CommandTTLJanitor.
const DefaultTTLJanitorInterval = 30 * time.Second

// CommandTTLJanitor 周期兜底清理已完成且超过 TTL 的 Command，并摘除
// terminating Command 的 finalizer，保证 spec.TTLSecondsAfterFinished 真正生效。
//
// 为什么需要它：
//  1. TTL 的既有消费方在 agent 侧（ttlSecondAfterFinished 轮询），依赖 agent
//     内存里的任务表，agent 重启、任务丢失或删除失败后对象就会漏掉；
//  2. agent 删除 Command 时不摘 finalizer（节点 agent 不允许持有 commands
//     patch/update 权限，摘除职责在 bke-controller），裸 Delete 只会打出
//     deletionTimestamp，对象永久 terminating。
//
// 管理端持有 commands 的 patch 权限（bke-manager-role），因此在这里统一
// 走 PurgeCommand 兜底。经 mgr.Add 启动，仅在 leader 上运行。
type CommandTTLJanitor struct {
	Client   client.Client
	Interval time.Duration
}

// Start implements manager.Runnable.
func (j *CommandTTLJanitor) Start(ctx context.Context) error {
	interval := j.Interval
	if interval <= 0 {
		interval = DefaultTTLJanitorInterval
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return nil
		case <-ticker.C:
			j.purgeOnce(ctx)
		}
	}
}

// purgeOnce 执行一轮清理：terminating 的直接摘 finalizer，已完成且 TTL
// 到期的通过 PurgeCommand 删除。单个命令失败不影响其余命令。
func (j *CommandTTLJanitor) purgeOnce(ctx context.Context) {
	commandList := &agentv1beta1.CommandList{}
	if err := j.Client.List(ctx, commandList); err != nil {
		log.Warnf("command ttl janitor failed to list commands: %v", err)
		return
	}
	for i := range commandList.Items {
		cmd := &commandList.Items[i]
		if !cmd.DeletionTimestamp.IsZero() || ttlExpired(cmd, time.Now()) {
			if err := PurgeCommand(ctx, j.Client, cmd); err != nil {
				log.Warnf("command ttl janitor failed to purge command %s: %v", cmd.GetName(), err)
			}
		}
	}
}

// ttlExpired 判断命令是否已成功完成且超过 TTL。
// 成功语义：所有节点 Status==ConditionTrue（含 Skip 完成态，agent 侧 Skip 会写 True）。
// 失败或仍在运行的命令不在此清理——失败对象回收依赖 Unique 覆盖清理与人工标注，
// 不设独立 Failed TTL（oFEP-0068 / P5 决议）。未设置 TTL（<=0）不删除。
func ttlExpired(command *agentv1beta1.Command, now time.Time) bool {
	if command.Spec.TTLSecondsAfterFinished <= 0 {
		return false
	}
	if len(command.Status) == 0 {
		return false
	}
	var finishedAt time.Time
	for _, nodeStatus := range command.Status {
		if nodeStatus == nil || nodeStatus.Status != metav1.ConditionTrue {
			return false
		}
		if nodeStatus.CompletionTime != nil && nodeStatus.CompletionTime.After(finishedAt) {
			finishedAt = nodeStatus.CompletionTime.Time
		}
	}
	if finishedAt.IsZero() {
		return false
	}
	return !now.Before(finishedAt.Add(time.Duration(command.Spec.TTLSecondsAfterFinished) * time.Second))
}
