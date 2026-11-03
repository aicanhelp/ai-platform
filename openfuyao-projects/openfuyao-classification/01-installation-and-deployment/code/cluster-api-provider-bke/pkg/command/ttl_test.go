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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
)

// makeTTLTestCommand 构造带节点状态的测试 Command。
// nodeStatuses: 节点名 -> 节点级聚合状态；completedAgo: completionTime 距今时长。
func makeTTLTestCommand(name string, ttl int, nodeStatuses map[string]metav1.ConditionStatus,
	completedAgo time.Duration, withFinalizer, terminating bool) *agentv1beta1.Command {
	cmd := &agentv1beta1.Command{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: testNS,
		},
		Spec: agentv1beta1.CommandSpec{
			NodeSelector:            &metav1.LabelSelector{},
			TTLSecondsAfterFinished: ttl,
		},
	}
	status := map[string]*agentv1beta1.CommandStatus{}
	for node, st := range nodeStatuses {
		ns := &agentv1beta1.CommandStatus{
			Status: st,
			Phase:  agentv1beta1.CommandRunning,
		}
		if completedAgo > 0 {
			ns.CompletionTime = &metav1.Time{Time: time.Now().Add(-completedAgo)}
			ns.Phase = agentv1beta1.CommandComplete
		}
		status[node] = ns
	}
	cmd.Status = status
	if withFinalizer {
		cmd.Finalizers = []string{agentv1beta1.CommandFinalizer}
	}
	if terminating {
		cmd.DeletionTimestamp = &metav1.Time{Time: time.Now()}
	}
	return cmd
}

func TestTTLExpired(t *testing.T) {
	allSucceeded := map[string]metav1.ConditionStatus{"node-1": metav1.ConditionTrue}
	tests := []struct {
		name    string
		command *agentv1beta1.Command
		want    bool
	}{
		{"ttl not set keeps command", makeTTLTestCommand("c1", 0, allSucceeded, 48*time.Hour, true, false), false},
		{"no node status keeps command", makeTTLTestCommand("c2", 600, nil, 0, true, false), false},
		{
			"running node keeps command",
			makeTTLTestCommand("c3", 600, map[string]metav1.ConditionStatus{"node-1": ""}, 48*time.Hour, true, false),
			false,
		},
		{
			"failed node keeps command",
			makeTTLTestCommand("c4", 600, map[string]metav1.ConditionStatus{"node-1": metav1.ConditionFalse}, 48*time.Hour, true, false),
			false,
		},
		{
			"one failed node among succeeded keeps command",
			makeTTLTestCommand("c5", 600, map[string]metav1.ConditionStatus{
				"node-1": metav1.ConditionTrue, "node-2": metav1.ConditionFalse,
			}, 48*time.Hour, true, false),
			false,
		},
		{
			"succeeded without completion time keeps command",
			makeTTLTestCommand("c6", 600, allSucceeded, 0, true, false),
			false,
		},
		{
			"succeeded within ttl keeps command",
			makeTTLTestCommand("c7", 600, allSucceeded, 5*time.Minute, true, false),
			false,
		},
		{
			"succeeded past ttl expires",
			makeTTLTestCommand("c8", 600, allSucceeded, 11*time.Minute, true, false),
			true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, ttlExpired(tt.command, time.Now()))
		})
	}
}

func TestCommandTTLJanitorPurgeOnce(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = agentv1beta1.AddToScheme(scheme)

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(
		// terminating 僵尸：无论 TTL 都应摘掉 finalizer 让对象消失
		makeTTLTestCommand("zombie-terminating", 0,
			map[string]metav1.ConditionStatus{"node-1": metav1.ConditionTrue}, 0, true, true),
		// 成功完成且超过 TTL：应被清掉
		makeTTLTestCommand("expired-succeeded", 600,
			map[string]metav1.ConditionStatus{"node-1": metav1.ConditionTrue}, 11*time.Minute, true, false),
		// 成功完成但未到期：保留
		makeTTLTestCommand("fresh-succeeded", 600,
			map[string]metav1.ConditionStatus{"node-1": metav1.ConditionTrue}, 5*time.Minute, true, false),
		// 失败命令：本轮范围外，保留
		makeTTLTestCommand("failed-command", 600,
			map[string]metav1.ConditionStatus{"node-1": metav1.ConditionFalse}, 48*time.Hour, true, false),
		// 运行中命令：保留
		makeTTLTestCommand("running-command", 600,
			map[string]metav1.ConditionStatus{"node-1": ""}, 0, true, false),
	).Build()

	janitor := &CommandTTLJanitor{Client: fakeClient}
	janitor.purgeOnce(context.Background())

	assertGone := func(name string) {
		t.Helper()
		err := fakeClient.Get(context.Background(),
			client.ObjectKey{Namespace: testNS, Name: name}, &agentv1beta1.Command{})
		assert.True(t, apierrors.IsNotFound(err), "command %s should be purged", name)
	}
	assertKept := func(name string) {
		t.Helper()
		assert.NoError(t, fakeClient.Get(context.Background(),
			client.ObjectKey{Namespace: testNS, Name: name}, &agentv1beta1.Command{}))
	}

	assertGone("zombie-terminating")
	assertGone("expired-succeeded")
	assertKept("fresh-succeeded")
	assertKept("failed-command")
	assertKept("running-command")

	// 僵尸被摘除 finalizer 后消失，其余保留对象仍带 finalizer
	kept := &agentv1beta1.Command{}
	assert.NoError(t, fakeClient.Get(context.Background(),
		client.ObjectKey{Namespace: testNS, Name: "fresh-succeeded"}, kept))
	assert.True(t, controllerutil.ContainsFinalizer(kept, agentv1beta1.CommandFinalizer))
}
