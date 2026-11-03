/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
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
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
)

type fakeNodeFlagReader struct {
	mu sync.Mutex
	// polls counts how many times each ip was queried
	polls map[string]int
	// readyAfter makes ip return true after N polls (N<=0 means immediately true)
	readyAfter map[string]int
	// errForFirst makes ip return error for first N polls
	errForFirst map[string]int
}

func (f *fakeNodeFlagReader) GetNodeStateFlagForCluster(_ context.Context, _ *bkev1beta1.BKECluster, ip string, _ int) (bool, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	if f.polls == nil {
		f.polls = map[string]int{}
	}
	f.polls[ip]++

	if n := f.errForFirst[ip]; n > 0 && f.polls[ip] <= n {
		return false, context.DeadlineExceeded // any non-nil error is treated as transient
	}

	after, ok := f.readyAfter[ip]
	if !ok {
		return false, nil
	}
	if after <= 0 {
		return true, nil
	}
	return f.polls[ip] >= after, nil
}

func TestWaitNodesStateFlagVisible_Success(t *testing.T) {
	r := &fakeNodeFlagReader{
		readyAfter: map[string]int{
			"10.0.0.1": 2,
			"10.0.0.2": 3,
		},
	}
	cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "c"}}
	nodes := bkenode.Nodes{
		{IP: "10.0.0.1"},
		{IP: "10.0.0.2"},
	}

	err := WaitNodesStateFlagVisible(WaitNodesStateFlagVisibleInput{
		Context: context.Background(), Reader: r, BKECluster: cluster, Nodes: nodes, Flag: 123,
		Options: WaitNodesStateFlagVisibleOptions{Timeout: 200 * time.Millisecond, Interval: 5 * time.Millisecond},
	})
	require.NoError(t, err)
}

func TestWaitNodesStateFlagVisible_TransientErrorsEventuallySuccess(t *testing.T) {
	r := &fakeNodeFlagReader{
		readyAfter: map[string]int{
			"10.0.0.1": 4,
		},
		errForFirst: map[string]int{
			"10.0.0.1": 2,
		},
	}
	cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "c"}}
	nodes := bkenode.Nodes{
		{IP: "10.0.0.1"},
	}

	err := WaitNodesStateFlagVisible(WaitNodesStateFlagVisibleInput{
		Context: context.Background(), Reader: r, BKECluster: cluster, Nodes: nodes, Flag: 123,
		Options: WaitNodesStateFlagVisibleOptions{Timeout: 200 * time.Millisecond, Interval: 5 * time.Millisecond},
	})
	require.NoError(t, err)
}

func TestWaitNodesStateFlagVisible_Timeout(t *testing.T) {
	r := &fakeNodeFlagReader{
		readyAfter: map[string]int{
			// never queried ip will default to false
		},
	}
	cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "c"}}
	nodes := bkenode.Nodes{
		{IP: "10.0.0.9"},
	}

	err := WaitNodesStateFlagVisible(WaitNodesStateFlagVisibleInput{
		Context: context.Background(), Reader: r, BKECluster: cluster, Nodes: nodes, Flag: 123,
		Options: WaitNodesStateFlagVisibleOptions{Timeout: 40 * time.Millisecond, Interval: 5 * time.Millisecond},
	})
	require.Error(t, err)
	require.Contains(t, err.Error(), "timed out waiting")
}

func TestWaitNodesStateFlagVisible_ContextCancelled(t *testing.T) {
	r := &fakeNodeFlagReader{}
	cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "c"}}
	nodes := bkenode.Nodes{
		{IP: "10.0.0.9"},
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := WaitNodesStateFlagVisible(WaitNodesStateFlagVisibleInput{
		Context: ctx, Reader: r, BKECluster: cluster, Nodes: nodes, Flag: 123,
		Options: WaitNodesStateFlagVisibleOptions{Timeout: 200 * time.Millisecond, Interval: 5 * time.Millisecond},
	})
	require.Error(t, err)
	require.Contains(t, err.Error(), "cancelled waiting")
}
