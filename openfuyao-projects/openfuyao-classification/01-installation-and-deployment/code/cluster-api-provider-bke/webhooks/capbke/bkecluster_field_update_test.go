package capbke

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
)

func TestIsPathNotAllowedAndExtractPaths(t *testing.T) {
	notAllowed := [][]string{{"spec", "clusterConfig", "cluster", "networking", "*"}}
	assert.True(t, isPathNotAllowed(notAllowed, []string{"spec", "clusterConfig", "cluster", "networking", "podSubnet"}))
	assert.False(t, isPathNotAllowed(notAllowed, []string{"spec", "dryRun"}))

	paths := extractPaths([]string{}, map[string]interface{}{
		"spec": map[string]interface{}{
			"dryRun": true,
			"clusterConfig": map[string]interface{}{
				"cluster": map[string]interface{}{
					"networking": map[string]interface{}{
						"podSubnet": "10.0.0.0/16",
					},
				},
			},
		},
	})
	require.Len(t, paths, 2)
}

func TestValidateDryRunRejectsNonBKECluster(t *testing.T) {
	webhook := &BKECluster{}
	old := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c", Namespace: "ns"},
		Spec:       confv1beta1.BKEClusterSpec{DryRun: true},
	}
	newCluster := old.DeepCopy()
	newCluster.Annotations = map[string]string{
		common.BKEClusterFromAnnotationKey: common.BKEClusterFromAnnotationValueOther,
	}
	assert.Error(t, webhook.validateDryRun(newCluster, old))
}

func TestValidateFieldUpdateRejectsProtectedPaths(t *testing.T) {
	webhook := &BKECluster{}
	old := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c", Namespace: "ns"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{
					Networking: confv1beta1.Networking{PodSubnet: "10.0.0.0/16"},
				},
			},
		},
	}
	newCluster := old.DeepCopy()
	newCluster.Spec.ClusterConfig.Cluster.Networking.PodSubnet = "10.1.0.0/16"
	assert.Error(t, webhook.validateFieldUpdate(newCluster, old, notAllowedPaths))
}

func TestValidateDryRunAllowsBKECluster(t *testing.T) {
	webhook := &BKECluster{}
	old := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c", Namespace: "ns"},
		Spec:       confv1beta1.BKEClusterSpec{DryRun: true},
	}
	require.NoError(t, webhook.validateDryRun(old, old))
}

func TestValidateCommonUpgradeabilityRequiresHealthyCluster(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	webhook := &BKECluster{NodeFetcher: nodeutil.NewNodeFetcher(fake.NewClientBuilder().WithScheme(scheme).Build())}
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c", Namespace: "ns"},
		Status:     confv1beta1.BKEClusterStatus{ClusterStatus: bkev1beta1.ClusterUnhealthy},
	}
	assert.Error(t, webhook.validateCommonUpgradeability(context.Background(), cluster, cluster))

	cluster.Status.ClusterStatus = bkev1beta1.ClusterStatusReady
	require.NoError(t, webhook.validateCommonUpgradeability(context.Background(), cluster, cluster))
}

func TestValidateNodeAgentStatus(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))

	cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c", Namespace: "ns"}}
	readyNode := &confv1beta1.BKENode{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-ready", Namespace: "ns",
			Labels: map[string]string{nodeutil.ClusterNameLabel: "c"},
		},
		Spec:   confv1beta1.BKENodeSpec{IP: "10.0.0.1"},
		Status: confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeAgentReadyFlag},
	}
	notReadyNode := &confv1beta1.BKENode{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-bad", Namespace: "ns",
			Labels: map[string]string{nodeutil.ClusterNameLabel: "c"},
		},
		Spec:   confv1beta1.BKENodeSpec{IP: "10.0.0.2"},
		Status: confv1beta1.BKENodeStatus{StateCode: 0},
	}
	failedNode := &confv1beta1.BKENode{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-failed", Namespace: "ns",
			Labels: map[string]string{nodeutil.ClusterNameLabel: "c"},
		},
		Spec:   confv1beta1.BKENodeSpec{IP: "10.0.0.3"},
		Status: confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeFailedFlag},
	}
	client := fake.NewClientBuilder().WithScheme(scheme).WithObjects(cluster, readyNode, failedNode).Build()
	webhook := &BKECluster{NodeFetcher: nodeutil.NewNodeFetcher(client)}
	require.NoError(t, webhook.validateNodeAgentStatus(context.Background(), cluster))

	notReadyClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(cluster, readyNode, notReadyNode).Build()
	webhook.NodeFetcher = nodeutil.NewNodeFetcher(notReadyClient)
	assert.Error(t, webhook.validateNodeAgentStatus(context.Background(), cluster))
}
