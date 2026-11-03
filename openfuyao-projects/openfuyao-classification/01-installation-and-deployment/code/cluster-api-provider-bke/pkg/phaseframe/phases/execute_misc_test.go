package phases

import (
	"context"
	"errors"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/kube"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// ---- EnsureDryRun.Execute ----

func TestEnsureDryRunExecuteCov(t *testing.T) {
	mkPhase := func(t *testing.T) *EnsureDryRun {
		cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
		ctx := newAdditionalPhaseContext(t, cluster)
		return &EnsureDryRun{BasePhase: phaseframe.NewBasePhase(ctx, EnsureDryRunName)}
	}

	t.Run("reconcile success returns nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := mkPhase(t)
		patches.ApplyPrivateMethod(e, "reconcileDryRun", func(_ *EnsureDryRun) error { return nil })
		_, err := e.Execute()
		require.NoError(t, err)
	})

	t.Run("reconcile error returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := mkPhase(t)
		patches.ApplyPrivateMethod(e, "reconcileDryRun", func(_ *EnsureDryRun) error {
			return errors.New("dry run failed")
		})
		_, err := e.Execute()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "dry run failed")
	})
}

// ---- EnsureNodesPostProcess.Execute ----

func TestEnsureNodesPostProcessExecuteCov(t *testing.T) {
	mkPhase := func(t *testing.T) *EnsureNodesPostProcess {
		cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
		ctx := newAdditionalPhaseContext(t, cluster)
		return &EnsureNodesPostProcess{BasePhase: phaseframe.NewBasePhase(ctx, EnsureNodesPostProcessName)}
	}

	t.Run("success returns nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := mkPhase(t)
		patches.ApplyMethod(e, "CheckOrRunPostProcess", func(_ *EnsureNodesPostProcess) (ctrl.Result, error) {
			return ctrl.Result{}, nil
		})
		_, err := e.Execute()
		require.NoError(t, err)
	})

	t.Run("error returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := mkPhase(t)
		patches.ApplyMethod(e, "CheckOrRunPostProcess", func(_ *EnsureNodesPostProcess) (ctrl.Result, error) {
			return ctrl.Result{}, errors.New("post process failed")
		})
		_, err := e.Execute()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "post process failed")
	})
}

// ---- GetTargetClusterNodes ----

func TestGetTargetClusterNodesCov(t *testing.T) {
	t.Run("GetTargetClusterClient error returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
		ctx := newAdditionalPhaseContext(t, cluster)
		patches.ApplyFunc(kube.GetTargetClusterClient,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) (*kubernetes.Clientset, dynamic.Interface, error) {
				return nil, nil, errors.New("no remote client")
			})
		_, err := GetTargetClusterNodes(context.Background(), ctx.Client, cluster)
		require.Error(t, err)
	})
}
