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

package phases

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/command"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/kube"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/mfutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	k8sfake "k8s.io/client-go/kubernetes/fake"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	ctrlfake "sigs.k8s.io/controller-runtime/pkg/client/fake"
	fakeclient "sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestEnsureEtcdUpgradeConstants(t *testing.T) {
	assert.Equal(t, "EnsureEtcdUpgrade", string(EnsureEtcdUpgradeName))
}

func TestNewEnsureEtcdUpgrade(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()
	ctx := &phaseframe.PhaseContext{
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	phase := NewEnsureEtcdUpgrade(ctx)
	assert.NotNil(t, phase)
}

// ---- createUpgradeCommand ----

func TestEnsureEtcdUpgradeCreateUpgradeCommand(t *testing.T) {
	mkParams := func(e *EnsureEtcdUpgrade, node, backup confv1beta1.Node, needBackup bool) UpgradeCommandParams {
		return UpgradeCommandParams{
			Ctx:        context.Background(),
			Client:     e.Ctx.Client,
			BKECluster: e.Ctx.BKECluster,
			Scheme:     e.Ctx.Scheme,
			Node:       node,
			NeedBackup: needBackup,
			BackupNode: backup,
		}
	}

	t.Run("sets BackUpEtcd when need backup and node matches backup", func(t *testing.T) {
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.6"))
		node := etcdNode("10.0.0.1", "node1")
		cmd := e.createUpgradeCommand(mkParams(e, node, node, true))
		assert.True(t, cmd.BackUpEtcd)
		assert.Equal(t, "v3.5.6", cmd.EtcdVersion)
		assert.Equal(t, "c1", cmd.BKEConfig)
		assert.NotNil(t, cmd.Node)
	})

	t.Run("no backup when NeedBackup false", func(t *testing.T) {
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.6"))
		node := etcdNode("10.0.0.1", "node1")
		cmd := e.createUpgradeCommand(mkParams(e, node, node, false))
		assert.False(t, cmd.BackUpEtcd)
	})

	t.Run("no backup when node IP differs from backup IP", func(t *testing.T) {
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.6"))
		node := etcdNode("10.0.0.1", "node1")
		backup := etcdNode("10.0.0.2", "node2")
		cmd := e.createUpgradeCommand(mkParams(e, node, backup, true))
		assert.False(t, cmd.BackUpEtcd)
	})

	t.Run("empty etcd version when cluster config missing", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
		e := newEtcdUpgradePhaseCov(t, cluster)
		node := etcdNode("10.0.0.1", "node1")
		cmd := e.createUpgradeCommand(mkParams(e, node, node, true))
		assert.Empty(t, cmd.EtcdVersion)
	})
}

// ---- executeUpgradeCommand ----

func TestEnsureEtcdUpgradeExecuteUpgradeCommand(t *testing.T) {
	t.Run("New success returns nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.6"))
		patches.ApplyMethod(&command.Upgrade{}, "New", func(_ *command.Upgrade) error { return nil })
		upgrade := &command.Upgrade{}
		require.NoError(t, e.executeUpgradeCommand(upgrade, etcdNode("10.0.0.1", "node1"), e.Ctx.Log))
	})

	t.Run("New error returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.6"))
		patches.ApplyMethod(&command.Upgrade{}, "New", func(_ *command.Upgrade) error {
			return errors.New("new failed")
		})
		upgrade := &command.Upgrade{}
		err := e.executeUpgradeCommand(upgrade, etcdNode("10.0.0.1", "node1"), e.Ctx.Log)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "create upgrade command")
	})
}

// ---- waitForUpgradeComplete ----

func TestEnsureEtcdUpgradeWaitForUpgradeComplete(t *testing.T) {
	mkWaitParams := func(e *EnsureEtcdUpgrade) WaitUpgradeParams {
		return WaitUpgradeParams{
			Upgrade:    &command.Upgrade{BaseCommand: command.BaseCommand{Command: &agentv1beta1.Command{}}},
			BKECluster: e.Ctx.BKECluster,
			Node:       etcdNode("10.0.0.1", "node1"),
			Log:        e.Ctx.Log,
		}
	}

	t.Run("wait error returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.6"))
		patches.ApplyMethod(&command.Upgrade{}, "Wait", func(_ *command.Upgrade) (error, []string, []string) {
			return errors.New("wait failed"), nil, nil
		})
		err := e.waitForUpgradeComplete(mkWaitParams(e))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "wait upgrade command complete failed")
	})

	t.Run("failed nodes returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.6"))
		patches.ApplyMethod(&command.Upgrade{}, "Wait", func(_ *command.Upgrade) (error, []string, []string) {
			return nil, []string{}, []string{"10.0.0.1"}
		})
		patches.ApplyFunc(phaseutil.LogCommandFailed,
			func(_ agentv1beta1.Command, _ []string, _ *bkev1beta1.BKELogger, _ string) (map[string][]string, error) {
				return map[string][]string{"10.0.0.1": {"err"}}, nil
			})
		patches.ApplyFunc(phaseutil.MarkNodeStatusByCommandErrs,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster, _ map[string][]string) {})
		err := e.waitForUpgradeComplete(mkWaitParams(e))
		require.Error(t, err)
	})

	t.Run("success returns nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.6"))
		patches.ApplyMethod(&command.Upgrade{}, "Wait", func(_ *command.Upgrade) (error, []string, []string) {
			return nil, []string{"10.0.0.1"}, nil
		})
		require.NoError(t, e.waitForUpgradeComplete(mkWaitParams(e)))
	})
}

// ---- waitForEtcdHealthCheck ----

func TestEnsureEtcdUpgradeWaitForEtcdHealthCheck(t *testing.T) {
	t.Run("version match returns nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.6"))
		patches.ApplyPrivateMethod(e, "getEtcdImageVersion",
			func(_ *EnsureEtcdUpgrade, _ confv1beta1.Node) (string, error) {
				return "3.5.6", nil
			})
		params := HealthCheckParams{
			Ctx:     context.Background(),
			Node:    etcdNode("10.0.0.1", "node1"),
			Version: "v3.5.6",
			Log:     e.Ctx.Log,
		}
		require.NoError(t, e.waitForEtcdHealthCheck(params))
	})

	t.Run("version mismatch times out returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.6"))
		patches.ApplyPrivateMethod(e, "getEtcdImageVersion",
			func(_ *EnsureEtcdUpgrade, _ confv1beta1.Node) (string, error) {
				return "3.4.0", nil
			})
		patches.ApplyFunc(wait.PollWithContext,
			func(_ context.Context, _ time.Duration, _ time.Duration, _ wait.ConditionWithContextFunc) error {
				return wait.ErrWaitTimeout
			})
		params := HealthCheckParams{
			Ctx:     context.Background(),
			Node:    etcdNode("10.0.0.1", "node1"),
			Version: "v3.5.6",
			Log:     e.Ctx.Log,
		}
		err := e.waitForEtcdHealthCheck(params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "pass healthy check failed")
	})
}

// ---- getEtcdImageVersion ----

func TestEnsureEtcdUpgradeGetEtcdImageVersion(t *testing.T) {
	t.Run("GetTargetClusterClient error returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.6"))
		patches.ApplyFunc(kube.GetTargetClusterClient,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) (*kubernetes.Clientset, dynamic.Interface, error) {
				return nil, nil, errors.New("no remote client")
			})
		_, err := e.getEtcdImageVersion(etcdNode("10.0.0.1", "node1"))
		require.Error(t, err)
	})
}

func newEtcdUpgradePhaseCov(t *testing.T, bkeCluster *bkev1beta1.BKECluster) *EnsureEtcdUpgrade {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	c := ctrlfake.NewClientBuilder().WithScheme(scheme).Build()
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	return &EnsureEtcdUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}
}

func etcdUpgradeCluster(etcdVer string) *bkev1beta1.BKECluster {
	return &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{Cluster: confv1beta1.Cluster{EtcdVersion: etcdVer}},
		},
	}
}

func etcdNode(ip, host string) confv1beta1.Node {
	return confv1beta1.Node{IP: ip, Hostname: host, Role: []string{"etcd"}}
}

// patchSyncStatus patches mergecluster.SyncStatusUntilComplete to return the given error.
func patchSyncStatus(patches *gomonkey.Patches, err error) {
	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error {
		return err
	})
}

// ---- extractVersionFromImage (pure) ----

func TestExtractVersionFromImage(t *testing.T) {
	tests := []struct {
		name  string
		image string
		want  string
	}{
		{"image with tag", "registry.io/etcd:v3.5.0", "v3.5.0"},
		{"image with path and tag", "registry.io/etcd/etcd:v3.5.0", "v3.5.0"},
		{"no tag returns unknown", "registry.io/etcd", "unknown"},
		{"empty returns unknown", "", "unknown"},
		{"port and tag", "registry.io:5000/etcd:v3.5.1-0", "v3.5.1-0"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, extractVersionFromImage(tt.image))
		})
	}
}

// ---- Version ----

func TestEnsureEtcdUpgradeVersionCov(t *testing.T) {
	t.Run("returns etcd version", func(t *testing.T) {
		cluster := etcdUpgradeCluster("v3.5.0")
		cluster.Status.EtcdVersion = "v3.4.0"
		e := newEtcdUpgradePhaseCov(t, cluster)
		assert.Equal(t, "v3.4.0", e.Version())
	})
	t.Run("nil ctx", func(t *testing.T) {
		assert.Equal(t, "", (&EnsureEtcdUpgrade{}).Version())
	})
}

// ---- etcdVersionsMatch ----
// shouldSkipNode was removed upstream (commit ba4039e) and its version-match
// logic was extracted into the pure helper etcdVersionsMatch, used by
// waitForEtcdHealthCheck. The coverage is preserved by testing that helper.

func TestEnsureEtcdUpgradeEtcdVersionsMatch(t *testing.T) {
	tests := []struct {
		name    string
		want    string
		current string
		expect  bool
	}{
		{name: "match when versions equal", want: "3.5.0", current: "3.5.0", expect: true},
		{name: "match ignores v prefix on want", want: "v3.5.0", current: "3.5.0", expect: true},
		{name: "match ignores v prefix on current", want: "3.5.0", current: "v3.5.0", expect: true},
		{name: "match trims surrounding whitespace", want: " 3.5.0 ", current: " 3.5.0 ", expect: true},
		{name: "no match when versions differ", want: "3.5.0", current: "3.4.0", expect: false},
		{name: "no match when want empty -> do not skip", want: "", current: "3.5.0", expect: false},
		{name: "no match when current empty", want: "3.5.0", current: "", expect: false},
		{name: "no match when current unknown", want: "3.5.0", current: "unknown", expect: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expect, etcdVersionsMatch(tt.want, tt.current))
		})
	}
}

// ---- determineBackupNode ----

func TestEnsureEtcdUpgradeDetermineBackupNode(t *testing.T) {
	t.Run("returns first etcd node", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.0"))
		patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
			return &nodeutil.FetchResult{Nodes: bkenode.Nodes{etcdNode("192.168.1.1", "node1"), etcdNode("192.168.1.2", "node2")}}, nil
		})
		need, node := e.determineBackupNode(e.Ctx.BKECluster, e.Ctx.Log)
		assert.True(t, need)
		assert.Equal(t, "node1", node.Hostname)
	})

	t.Run("no etcd nodes returns false", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.0"))
		patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
			return &nodeutil.FetchResult{Nodes: bkenode.Nodes{}}, nil
		})
		need, node := e.determineBackupNode(e.Ctx.BKECluster, e.Ctx.Log)
		assert.False(t, need)
		assert.Equal(t, confv1beta1.Node{}, node)
	})

	t.Run("fetch error returns false", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.0"))
		patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
			return nil, errors.New("fetch failed")
		})
		need, node := e.determineBackupNode(e.Ctx.BKECluster, e.Ctx.Log)
		assert.False(t, need)
		assert.Equal(t, confv1beta1.Node{}, node)
	})
}

// ---- filterUpgradeableNodes ----

func TestEnsureEtcdUpgradeFilterUpgradeableNodes(t *testing.T) {
	t.Run("filters to ready nodes", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.0"))
		patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
			return &nodeutil.FetchResult{Nodes: bkenode.Nodes{etcdNode("192.168.1.1", "node1"), etcdNode("192.168.1.2", "node2")}}, nil
		})
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetNodeStateFlag, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _, _ string, _ int) (bool, error) {
			return true, nil
		})
		nodes, err := e.filterUpgradeableNodes(e.Ctx.BKECluster, e.Ctx.Log)
		require.NoError(t, err)
		assert.Len(t, nodes, 2)
	})

	t.Run("fetch error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.0"))
		patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
			return nil, errors.New("fetch failed")
		})
		_, err := e.filterUpgradeableNodes(e.Ctx.BKECluster, e.Ctx.Log)
		require.Error(t, err)
	})

	t.Run("no etcd nodes", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.0"))
		patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
			return &nodeutil.FetchResult{Nodes: bkenode.Nodes{}}, nil
		})
		_, err := e.filterUpgradeableNodes(e.Ctx.BKECluster, e.Ctx.Log)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "no etcd role nodes")
	})

	t.Run("all agents not ready", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.0"))
		patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
			return &nodeutil.FetchResult{Nodes: bkenode.Nodes{etcdNode("192.168.1.1", "node1")}}, nil
		})
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetNodeStateFlag, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _, _ string, _ int) (bool, error) {
			return false, nil
		})
		_, err := e.filterUpgradeableNodes(e.Ctx.BKECluster, e.Ctx.Log)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "not ready")
	})
}

// ---- markNodeUpgrading / markNodeUpgradeSuccess ----

func TestEnsureEtcdUpgradeMarkNodeStatus(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.0"))
	patches.ApplyFunc((*nodeutil.NodeFetcher).SetNodeStateWithMessageForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster, _ string, _ confv1beta1.NodeState, _ string) error {
		return nil
	})
	params := NodeStatusParams{Client: e.Ctx.Client, BKECluster: e.Ctx.BKECluster, Node: etcdNode("192.168.1.1", "node1")}

	t.Run("mark upgrading success", func(t *testing.T) {
		patchSyncStatus(patches, nil)
		require.NoError(t, e.markNodeUpgrading(params))
	})
	t.Run("mark upgrading sync error", func(t *testing.T) {
		patchSyncStatus(patches, errors.New("sync failed"))
		err := e.markNodeUpgrading(params)
		require.Error(t, err)
	})
	t.Run("mark success", func(t *testing.T) {
		patchSyncStatus(patches, nil)
		require.NoError(t, e.markNodeUpgradeSuccess(params))
	})
	t.Run("mark success sync error", func(t *testing.T) {
		patchSyncStatus(patches, errors.New("sync failed"))
		require.Error(t, e.markNodeUpgradeSuccess(params))
	})
}

// ---- handleUpgradeFailure ----

func TestEnsureEtcdUpgradeHandleUpgradeFailure(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.0"))
	patches.ApplyFunc((*nodeutil.NodeFetcher).SetNodeStateWithMessageForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster, _ string, _ confv1beta1.NodeState, _ string) error {
		return nil
	})
	params := UpgradeFailureParams{
		Client:     e.Ctx.Client,
		BKECluster: e.Ctx.BKECluster,
		Node:       etcdNode("192.168.1.1", "node1"),
		Error:      errors.New("upgrade boom"),
		Log:        e.Ctx.Log,
	}

	t.Run("returns original error on sync success", func(t *testing.T) {
		patchSyncStatus(patches, nil)
		err := e.handleUpgradeFailure(params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "upgrade boom")
	})

	t.Run("returns sync error when sync fails", func(t *testing.T) {
		patchSyncStatus(patches, errors.New("sync boom"))
		err := e.handleUpgradeFailure(params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "sync boom")
	})
}

// ---- finalizeUpgrade ----

func TestEnsureEtcdUpgradeFinalizeUpgrade(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.0"))

	t.Run("success sets etcd version", func(t *testing.T) {
		patchSyncStatus(patches, nil)
		_, err := e.finalizeUpgrade(e.Ctx.Client, e.Ctx.BKECluster, e.Ctx.Log)
		require.NoError(t, err)
		assert.Equal(t, "v3.5.0", e.Ctx.BKECluster.Status.EtcdVersion)
	})

	t.Run("sync error", func(t *testing.T) {
		patchSyncStatus(patches, errors.New("sync failed"))
		_, err := e.finalizeUpgrade(e.Ctx.Client, e.Ctx.BKECluster, e.Ctx.Log)
		require.Error(t, err)
	})
}

// ---- upgradeNodes ----

func TestEnsureEtcdUpgradeUpgradeNodes(t *testing.T) {
	t.Run("empty nodes succeeds", func(t *testing.T) {
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.0"))
		params := NodeUpgradeParams{Client: e.Ctx.Client, BKECluster: e.Ctx.BKECluster, Nodes: bkenode.Nodes{}, Log: e.Ctx.Log}
		require.NoError(t, e.upgradeNodes(params))
	})

	t.Run("single node success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.0"))
		patches.ApplyPrivateMethod(e, "upgradeSingleNode", func(_ *EnsureEtcdUpgrade, _ SingleNodeUpgradeParams) error {
			return nil
		})
		params := NodeUpgradeParams{
			Client: e.Ctx.Client, BKECluster: e.Ctx.BKECluster,
			Nodes: bkenode.Nodes{etcdNode("192.168.1.1", "node1")}, Log: e.Ctx.Log,
		}
		require.NoError(t, e.upgradeNodes(params))
	})

	t.Run("single node error propagates", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.0"))
		patches.ApplyPrivateMethod(e, "upgradeSingleNode", func(_ *EnsureEtcdUpgrade, _ SingleNodeUpgradeParams) error {
			return errors.New("node failed")
		})
		params := NodeUpgradeParams{
			Client: e.Ctx.Client, BKECluster: e.Ctx.BKECluster,
			Nodes: bkenode.Nodes{etcdNode("192.168.1.1", "node1")}, Log: e.Ctx.Log,
		}
		err := e.upgradeNodes(params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "node failed")
	})
}

// ---- upgradeSingleNode ----

func TestEnsureEtcdUpgradeUpgradeSingleNode(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.0"))
	params := SingleNodeUpgradeParams{
		Client: e.Ctx.Client, BKECluster: e.Ctx.BKECluster,
		Node: etcdNode("192.168.1.1", "node1"), Log: e.Ctx.Log,
	}
	patches.ApplyFunc((*nodeutil.NodeFetcher).SetNodeStateWithMessageForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster, _ string, _ confv1beta1.NodeState, _ string) error {
		return nil
	})

	t.Run("success path", func(t *testing.T) {
		patches.ApplyPrivateMethod(e, "markNodeUpgrading", func(_ *EnsureEtcdUpgrade, _ NodeStatusParams) error { return nil })
		patches.ApplyPrivateMethod(e, "upgradeEtcd", func(_ *EnsureEtcdUpgrade, _ EtcdUpgradeParams) error { return nil })
		patches.ApplyPrivateMethod(e, "markNodeUpgradeSuccess", func(_ *EnsureEtcdUpgrade, _ NodeStatusParams) error { return nil })
		require.NoError(t, e.upgradeSingleNode(params))
	})

	t.Run("mark upgrading error", func(t *testing.T) {
		patches.ApplyPrivateMethod(e, "markNodeUpgrading", func(_ *EnsureEtcdUpgrade, _ NodeStatusParams) error { return errors.New("mark failed") })
		err := e.upgradeSingleNode(params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "mark failed")
	})

	t.Run("upgrade etcd error delegates to handleUpgradeFailure", func(t *testing.T) {
		patches.ApplyPrivateMethod(e, "markNodeUpgrading", func(_ *EnsureEtcdUpgrade, _ NodeStatusParams) error { return nil })
		patches.ApplyPrivateMethod(e, "upgradeEtcd", func(_ *EnsureEtcdUpgrade, _ EtcdUpgradeParams) error { return errors.New("etcd boom") })
		patches.ApplyPrivateMethod(e, "handleUpgradeFailure", func(_ *EnsureEtcdUpgrade, _ UpgradeFailureParams) error { return errors.New("handled") })
		err := e.upgradeSingleNode(params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "handled")
	})
}

// ---- rolloutUpgrade ----

func TestEnsureEtcdUpgradeRolloutUpgrade(t *testing.T) {
	t.Run("filter error requeues", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.0"))
		patches.ApplyPrivateMethod(e, "filterUpgradeableNodes", func(_ *EnsureEtcdUpgrade, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKELogger) (bkenode.Nodes, error) {
			return nil, errors.New("filter failed")
		})
		result, err := e.rolloutUpgrade()
		require.Error(t, err)
		assert.True(t, result.Requeue)
	})

	t.Run("happy path", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.0"))
		patches.ApplyPrivateMethod(e, "filterUpgradeableNodes", func(_ *EnsureEtcdUpgrade, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKELogger) (bkenode.Nodes, error) {
			return bkenode.Nodes{etcdNode("192.168.1.1", "node1")}, nil
		})
		patches.ApplyPrivateMethod(e, "determineBackupNode", func(_ *EnsureEtcdUpgrade, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKELogger) (bool, confv1beta1.Node) {
			return true, etcdNode("192.168.1.1", "node1")
		})
		patches.ApplyPrivateMethod(e, "upgradeNodes", func(_ *EnsureEtcdUpgrade, _ NodeUpgradeParams) error { return nil })
		patches.ApplyPrivateMethod(e, "finalizeUpgrade", func(_ *EnsureEtcdUpgrade, _ client.Client, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKELogger) (ctrl.Result, error) {
			return ctrl.Result{}, nil
		})
		_, err := e.rolloutUpgrade()
		require.NoError(t, err)
	})

	t.Run("upgrade nodes error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.0"))
		patches.ApplyPrivateMethod(e, "filterUpgradeableNodes", func(_ *EnsureEtcdUpgrade, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKELogger) (bkenode.Nodes, error) {
			return bkenode.Nodes{etcdNode("192.168.1.1", "node1")}, nil
		})
		patches.ApplyPrivateMethod(e, "determineBackupNode", func(_ *EnsureEtcdUpgrade, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKELogger) (bool, confv1beta1.Node) {
			return false, confv1beta1.Node{}
		})
		patches.ApplyPrivateMethod(e, "upgradeNodes", func(_ *EnsureEtcdUpgrade, _ NodeUpgradeParams) error { return errors.New("nodes failed") })
		_, err := e.rolloutUpgrade()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "nodes failed")
	})
}

// ---- Execute / reconcileEtcdUpgrade ----

func TestEnsureEtcdUpgradeExecute(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.0"))
		patches.ApplyPrivateMethod(e, "rolloutUpgrade", func(_ *EnsureEtcdUpgrade) (ctrl.Result, error) {
			return ctrl.Result{}, nil
		})
		_, err := e.Execute()
		require.NoError(t, err)
	})

	t.Run("rollout error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.0"))
		patches.ApplyPrivateMethod(e, "rolloutUpgrade", func(_ *EnsureEtcdUpgrade) (ctrl.Result, error) {
			return ctrl.Result{Requeue: true}, errors.New("rollout failed")
		})
		result, err := e.Execute()
		require.Error(t, err)
		assert.True(t, result.Requeue)
	})
}

// etcdUpgradeGapsPod builds a static etcd pod for the given node hostname with
// the supplied image. Used to drive the mockClient seam in getEtcdImageVersion.
func etcdUpgradeGapsPod(hostname, image string, phase corev1.PodPhase, ready bool) *corev1.Pod {
	var conds []corev1.PodCondition
	if ready {
		conds = append(conds, corev1.PodCondition{Type: corev1.PodReady, Status: corev1.ConditionTrue})
	}
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      kube.StaticPodName(mfutil.Etcd, hostname),
			Namespace: metav1.NamespaceSystem,
		},
		Status: corev1.PodStatus{Phase: phase, Conditions: conds},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: "etcd", Image: image}},
		},
	}
}

// ---- upgradeEtcd (orchestration, was 0%) ----

func TestEnsureEtcdUpgradeUpgradeEtcdGaps(t *testing.T) {
	// happy path: all sub-steps succeed, createUpgradeCommand runs for real.
	t.Run("happy path returns nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.6"))
		patches.ApplyMethod(&phaseframe.PhaseContext{}, "SyncUpgradeTargetsToClusterSpec",
			func(_ *phaseframe.PhaseContext) error { return nil })
		patches.ApplyPrivateMethod(e, "executeUpgradeCommand",
			func(_ *EnsureEtcdUpgrade, _ *command.Upgrade, _ confv1beta1.Node, _ *bkev1beta1.BKELogger) error {
				return nil
			})
		patches.ApplyPrivateMethod(e, "waitForUpgradeComplete",
			func(_ *EnsureEtcdUpgrade, _ WaitUpgradeParams) error { return nil })
		patches.ApplyPrivateMethod(e, "waitForEtcdHealthCheck",
			func(_ *EnsureEtcdUpgrade, _ HealthCheckParams) error { return nil })
		params := EtcdUpgradeParams{
			NeedBackup: false,
			Node:       etcdNode("10.0.0.1", "node1"),
			Version:    "v3.5.6",
		}
		require.NoError(t, e.upgradeEtcd(params))
	})

	t.Run("sync targets error returns wrapped error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.6"))
		patches.ApplyMethod(&phaseframe.PhaseContext{}, "SyncUpgradeTargetsToClusterSpec",
			func(_ *phaseframe.PhaseContext) error { return errors.New("sync boom") })
		params := EtcdUpgradeParams{Node: etcdNode("10.0.0.1", "node1"), Version: "v3.5.6"}
		err := e.upgradeEtcd(params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "sync etcd upgrade target to cluster spec")
	})

	t.Run("executeUpgradeCommand error returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.6"))
		patches.ApplyMethod(&phaseframe.PhaseContext{}, "SyncUpgradeTargetsToClusterSpec",
			func(_ *phaseframe.PhaseContext) error { return nil })
		patches.ApplyPrivateMethod(e, "executeUpgradeCommand",
			func(_ *EnsureEtcdUpgrade, _ *command.Upgrade, _ confv1beta1.Node, _ *bkev1beta1.BKELogger) error {
				return errors.New("exec boom")
			})
		params := EtcdUpgradeParams{Node: etcdNode("10.0.0.1", "node1"), Version: "v3.5.6"}
		err := e.upgradeEtcd(params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "exec boom")
	})

	t.Run("waitForUpgradeComplete error returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.6"))
		patches.ApplyMethod(&phaseframe.PhaseContext{}, "SyncUpgradeTargetsToClusterSpec",
			func(_ *phaseframe.PhaseContext) error { return nil })
		patches.ApplyPrivateMethod(e, "executeUpgradeCommand",
			func(_ *EnsureEtcdUpgrade, _ *command.Upgrade, _ confv1beta1.Node, _ *bkev1beta1.BKELogger) error {
				return nil
			})
		patches.ApplyPrivateMethod(e, "waitForUpgradeComplete",
			func(_ *EnsureEtcdUpgrade, _ WaitUpgradeParams) error { return errors.New("wait boom") })
		params := EtcdUpgradeParams{Node: etcdNode("10.0.0.1", "node1"), Version: "v3.5.6"}
		err := e.upgradeEtcd(params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "wait boom")
	})

	t.Run("waitForEtcdHealthCheck error returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.6"))
		patches.ApplyMethod(&phaseframe.PhaseContext{}, "SyncUpgradeTargetsToClusterSpec",
			func(_ *phaseframe.PhaseContext) error { return nil })
		patches.ApplyPrivateMethod(e, "executeUpgradeCommand",
			func(_ *EnsureEtcdUpgrade, _ *command.Upgrade, _ confv1beta1.Node, _ *bkev1beta1.BKELogger) error {
				return nil
			})
		patches.ApplyPrivateMethod(e, "waitForUpgradeComplete",
			func(_ *EnsureEtcdUpgrade, _ WaitUpgradeParams) error { return nil })
		patches.ApplyPrivateMethod(e, "waitForEtcdHealthCheck",
			func(_ *EnsureEtcdUpgrade, _ HealthCheckParams) error { return errors.New("health boom") })
		params := EtcdUpgradeParams{Node: etcdNode("10.0.0.1", "node1"), Version: "v3.5.6"}
		err := e.upgradeEtcd(params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "health boom")
	})
}

// ---- getEtcdImageVersion (was 14.8%, only error path covered) ----

func TestEnsureEtcdUpgradeGetEtcdImageVersionGaps(t *testing.T) {
	t.Run("success returns image version", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.6"))
		e.mockClient = k8sfake.NewSimpleClientset(
			etcdUpgradeGapsPod("node1", "registry.io/etcd:v3.5.6", corev1.PodRunning, true))
		ver, err := e.getEtcdImageVersion(etcdNode("10.0.0.1", "node1"))
		require.NoError(t, err)
		assert.Equal(t, "v3.5.6", ver)
	})

	t.Run("etcd container not found returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.6"))
		pod := etcdUpgradeGapsPod("node1", "registry.io/etcd:v3.5.6", corev1.PodRunning, true)
		pod.Spec.Containers = []corev1.Container{{Name: "other", Image: "registry.io/other:v1.0.0"}}
		e.mockClient = k8sfake.NewSimpleClientset(pod)
		_, err := e.getEtcdImageVersion(etcdNode("10.0.0.1", "node1"))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "etcd container not found")
	})

	t.Run("poll error returns wrapped error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.6"))
		e.mockClient = k8sfake.NewSimpleClientset(
			etcdUpgradeGapsPod("node1", "registry.io/etcd:v3.5.6", corev1.PodRunning, true))
		patches.ApplyFunc(wait.PollImmediate,
			func(_ time.Duration, _ time.Duration, _ wait.ConditionFunc) error {
				return errors.New("poll boom")
			})
		_, err := e.getEtcdImageVersion(etcdNode("10.0.0.1", "node1"))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed waiting for etcd pod")
	})

	t.Run("pod not running then times out", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.6"))
		e.mockClient = k8sfake.NewSimpleClientset(
			etcdUpgradeGapsPod("node1", "registry.io/etcd:v3.5.6", corev1.PodPending, false))
		patches.ApplyFunc(wait.PollImmediate,
			func(_ time.Duration, _ time.Duration, cond wait.ConditionFunc) error {
				cond() // pod phase != Running -> (false, nil)
				return wait.ErrWaitTimeout
			})
		_, err := e.getEtcdImageVersion(etcdNode("10.0.0.1", "node1"))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed waiting for etcd pod")
	})

	t.Run("pod not found then times out", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.6"))
		e.mockClient = k8sfake.NewSimpleClientset() // no pod -> IsNotFound
		patches.ApplyFunc(wait.PollImmediate,
			func(_ time.Duration, _ time.Duration, cond wait.ConditionFunc) error {
				cond() // IsNotFound -> (false, nil)
				return wait.ErrWaitTimeout
			})
		_, err := e.getEtcdImageVersion(etcdNode("10.0.0.1", "node1"))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed waiting for etcd pod")
	})

	t.Run("pod ready condition false then times out", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.6"))
		pod := etcdUpgradeGapsPod("node1", "registry.io/etcd:v3.5.6", corev1.PodRunning, false)
		pod.Status.Conditions = []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionFalse}}
		e.mockClient = k8sfake.NewSimpleClientset(pod)
		patches.ApplyFunc(wait.PollImmediate,
			func(_ time.Duration, _ time.Duration, cond wait.ConditionFunc) error {
				cond() // running but not ready -> (false, nil)
				return wait.ErrWaitTimeout
			})
		_, err := e.getEtcdImageVersion(etcdNode("10.0.0.1", "node1"))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed waiting for etcd pod")
	})
}

// ---- waitForEtcdHealthCheck (was 78.6%, condition body branches missing) ----

func TestEnsureEtcdUpgradeWaitForEtcdHealthCheckGaps(t *testing.T) {
	// PollWithContext stub that runs the real condition once then signals timeout.
	runCondOnceThenTimeout := func(patches *gomonkey.Patches) {
		patches.ApplyFunc(wait.PollWithContext,
			func(ctx context.Context, _ time.Duration, _ time.Duration, cond wait.ConditionWithContextFunc) error {
				_, _ = cond(ctx)
				return wait.ErrWaitTimeout
			})
	}

	t.Run("getEtcdImageVersion error inside poll returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.6"))
		patches.ApplyPrivateMethod(e, "getEtcdImageVersion",
			func(_ *EnsureEtcdUpgrade, _ confv1beta1.Node) (string, error) {
				return "", errors.New("probe failed")
			})
		runCondOnceThenTimeout(patches)
		params := HealthCheckParams{
			Ctx:     context.Background(),
			Node:    etcdNode("10.0.0.1", "node1"),
			Version: "v3.5.6",
			Log:     e.Ctx.Log,
		}
		err := e.waitForEtcdHealthCheck(params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "pass healthy check failed")
	})

	t.Run("version mismatch inside poll returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.6"))
		patches.ApplyPrivateMethod(e, "getEtcdImageVersion",
			func(_ *EnsureEtcdUpgrade, _ confv1beta1.Node) (string, error) {
				return "3.4.0", nil
			})
		runCondOnceThenTimeout(patches)
		params := HealthCheckParams{
			Ctx:     context.Background(),
			Node:    etcdNode("10.0.0.1", "node1"),
			Version: "v3.5.6",
			Log:     e.Ctx.Log,
		}
		err := e.waitForEtcdHealthCheck(params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "pass healthy check failed")
	})

	t.Run("cancelled context inside poll returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.6"))
		patches.ApplyPrivateMethod(e, "getEtcdImageVersion",
			func(_ *EnsureEtcdUpgrade, _ confv1beta1.Node) (string, error) {
				return "3.5.6", nil // not reached due to CtxDone
			})
		runCondOnceThenTimeout(patches)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		params := HealthCheckParams{
			Ctx:     ctx,
			Node:    etcdNode("10.0.0.1", "node1"),
			Version: "v3.5.6",
			Log:     e.Ctx.Log,
		}
		err := e.waitForEtcdHealthCheck(params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "pass healthy check failed")
	})
}

// ---- resolveEtcdUpgradeVersion (was 87.5%, VersionContext path missing) ----

func TestEnsureEtcdUpgradeResolveEtcdUpgradeVersionGaps(t *testing.T) {
	t.Run("returns version context target when set", func(t *testing.T) {
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.0"))
		vc := upgrade.NewVersionContext()
		vc.SetTarget(upgrade.ComponentEtcd, "v3.5.10")
		e.Ctx.VersionContext = vc
		assert.Equal(t, "v3.5.10", e.resolveEtcdUpgradeVersion())
	})

	t.Run("version context set but target empty falls back to spec", func(t *testing.T) {
		e := newEtcdUpgradePhaseCov(t, etcdUpgradeCluster("v3.5.0"))
		vc := upgrade.NewVersionContext()
		e.Ctx.VersionContext = vc
		assert.Equal(t, "v3.5.0", e.resolveEtcdUpgradeVersion())
	})

	t.Run("nil ctx returns empty", func(t *testing.T) {
		e := &EnsureEtcdUpgrade{}
		assert.Equal(t, "", e.resolveEtcdUpgradeVersion())
	})

	t.Run("nil cluster config returns empty", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
		e := newEtcdUpgradePhaseCov(t, cluster)
		assert.Equal(t, "", e.resolveEtcdUpgradeVersion())
	})
}
