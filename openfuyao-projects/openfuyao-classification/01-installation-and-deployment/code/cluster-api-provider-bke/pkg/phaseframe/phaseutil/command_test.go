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

package phaseutil

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
)

func TestLogCommandFailed_EmptyNodes(t *testing.T) {
	cmd := agentv1beta1.Command{}
	recorder := record.NewFakeRecorder(100)
	log := bkev1beta1.NewBKELogger(nil, recorder, nil)

	result, err := LogCommandFailed(cmd, []string{}, log, "test")
	assert.NoError(t, err)
	assert.Nil(t, result)
}

func TestGetMasterInitCommand_NotFound(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = agentv1beta1.AddToScheme(scheme)
	_ = bkev1beta1.AddToScheme(scheme)

	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	c := fake.NewClientBuilder().WithScheme(scheme).Build()

	_, err := GetMasterInitCommand(context.Background(), c, cluster)
	assert.Error(t, err)
}

func TestGetMasterJoinCommand_NotFound(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = agentv1beta1.AddToScheme(scheme)
	_ = bkev1beta1.AddToScheme(scheme)

	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	c := fake.NewClientBuilder().WithScheme(scheme).Build()

	_, err := GetMasterJoinCommand(context.Background(), c, cluster)
	assert.Error(t, err)
}

func TestGetWorkerJoinCommand_NotFound(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = agentv1beta1.AddToScheme(scheme)
	_ = bkev1beta1.AddToScheme(scheme)

	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	c := fake.NewClientBuilder().WithScheme(scheme).Build()

	_, err := GetWorkerJoinCommand(context.Background(), c, cluster)
	assert.Error(t, err)
}

func TestGetNodeIPFromCommandWaitResult(t *testing.T) {
	assert.Equal(t, "192.168.1.1", GetNodeIPFromCommandWaitResult("192.168.1.1"))
	assert.Equal(t, "192.168.1.2", GetNodeIPFromCommandWaitResult("node1/192.168.1.2"))
}

func TestGetNotSkipFailedNodeWithBKENodes(t *testing.T) {
	nodes := bkev1beta1.BKENodes{
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{NeedSkip: false}},
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2"}, Status: confv1beta1.BKENodeStatus{NeedSkip: true}},
	}
	failed := []string{"192.168.1.1", "192.168.1.2"}
	count := GetNotSkipFailedNodeWithBKENodes(nodes, failed)
	assert.Equal(t, 1, count)
}

func TestGetNotSkipFailedNode(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{}
	failed := []string{"192.168.1.1"}
	count := GetNotSkipFailedNode(cluster, failed)
	assert.Equal(t, 0, count)
}

func TestProcessNodeConditionsWithParams(t *testing.T) {
	cmd := &agentv1beta1.Command{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	recorder := record.NewFakeRecorder(100)
	log := bkev1beta1.NewBKELogger(nil, recorder, nil)
	conditions := []*agentv1beta1.Condition{
		{ID: "1", Status: metav1.ConditionFalse, StdErr: []string{"error1"}},
	}
	params := ProcessNodeConditionsParams{
		Conditions: conditions,
		Node:       "192.168.1.1",
		Cmd:        cmd,
		Log:        log,
		Reason:     "test",
	}
	errs, err := processNodeConditionsWithParams(params)
	assert.Error(t, err)
	assert.Len(t, errs, 1)
}

func TestLogCommandInfo(t *testing.T) {
	cmd := agentv1beta1.Command{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Status: map[string]*agentv1beta1.CommandStatus{
			"192.168.1.1": {
				Conditions: []*agentv1beta1.Condition{
					{ID: "1", Status: metav1.ConditionTrue, StdOut: []string{"output"}},
				},
			},
		},
	}
	recorder := record.NewFakeRecorder(100)
	log := bkev1beta1.NewBKELogger(nil, recorder, nil)
	LogCommandInfo(cmd, log, "test")
}

func TestMarkNodeStatusByCommandErrs_EmptyErrs(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	c := fake.NewClientBuilder().WithScheme(scheme).Build()
	cluster := &bkev1beta1.BKECluster{}
	MarkNodeStatusByCommandErrs(context.Background(), c, cluster, nil)
}

func TestLogCommandFailed_WithFailedNodes(t *testing.T) {
	cmd := agentv1beta1.Command{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Status: map[string]*agentv1beta1.CommandStatus{
			"192.168.1.1": {
				Conditions: []*agentv1beta1.Condition{
					{ID: "1", Status: metav1.ConditionFalse, StdErr: []string{"error"}},
				},
			},
		},
	}
	recorder := record.NewFakeRecorder(100)
	log := bkev1beta1.NewBKELogger(nil, recorder, nil)
	result, err := LogCommandFailed(cmd, []string{"192.168.1.1"}, log, "test")
	assert.Error(t, err)
	assert.NotNil(t, result)
	assert.Len(t, result["192.168.1.1"], 1)
}
