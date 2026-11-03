/*
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
 */

package recommend

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"openfuyao.com/colocation-management/cmd/colocation-manager/apps"
	"openfuyao.com/colocation-management/pkg/colocation-manager/aggregate"
	"openfuyao.com/colocation-management/pkg/common"
)

func mustScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	s := runtime.NewScheme()
	if err := clientgoscheme.AddToScheme(s); err != nil {
		t.Fatalf("add clientgo scheme: %v", err)
	}
	return s
}

func newTestNode(name string, allocatable corev1.ResourceList) *corev1.Node {
	return &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Status: corev1.NodeStatus{
			Allocatable: allocatable,
			Capacity:    allocatable,
		},
	}
}

func newNodeState(node *corev1.Node) *aggregate.NodeState {
	return aggregate.NewNodeState(
		context.Background(),
		node,
		&apps.Configuration{},
		func(aggregate.AggregateStateKey, *aggregate.AggregateContainerState) {},
	)
}

// ---- pure helpers ----

func TestShouldSkipUpdate(t *testing.T) {
	r := &ResourceRecommender{}
	tests := []struct {
		name            string
		newRes          string
		oldRes          string
		threshold       string
		wantSkip        bool
		wantDiffNonZero bool
	}{
		{name: "equal", newRes: "5", oldRes: "5", threshold: "1", wantSkip: true, wantDiffNonZero: false},
		{name: "new>old within threshold", newRes: "6", oldRes: "5", threshold: "10", wantSkip: true, wantDiffNonZero: true},
		{name: "new>old beyond threshold", newRes: "50", oldRes: "5", threshold: "10", wantSkip: false, wantDiffNonZero: true},
		{name: "old>new within threshold", newRes: "5", oldRes: "6", threshold: "10", wantSkip: true, wantDiffNonZero: true},
		{name: "old>new beyond threshold", newRes: "5", oldRes: "50", threshold: "10", wantSkip: false, wantDiffNonZero: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			skip, diff := r.shouldSkipUpdate(
				resource.MustParse(tt.newRes),
				resource.MustParse(tt.oldRes),
				resource.MustParse(tt.threshold),
			)
			assert.Equal(t, tt.wantSkip, skip)
			assert.Equal(t, tt.wantDiffNonZero, diff.Sign() != 0)
		})
	}
}

func TestGetResourceConfig(t *testing.T) {
	r := &ResourceRecommender{config: &apps.Configuration{
		BERecommendMinCPUMillicores: 100,
		BERecommendMinMemoryMb:      500,
	}}
	ext, threshold := r.getResourceConfig(corev1.ResourceCPU)
	assert.Equal(t, corev1.ResourceName(common.ExtenderResourceCPU), ext)
	assert.Equal(t, int64(100), threshold.MilliValue())

	ext, threshold = r.getResourceConfig(corev1.ResourceMemory)
	assert.Equal(t, corev1.ResourceName(common.ExtenderResourceMemory), ext)
	assert.Equal(t, int64(500), threshold.Value())

	ext, threshold = r.getResourceConfig(corev1.ResourceName("unknown"))
	assert.Equal(t, corev1.ResourceName(""), ext)
	assert.Equal(t, int64(0), threshold.Value())
}

func TestFilterControlledResources(t *testing.T) {
	estimation := aggregate.Resources{
		aggregate.ResourceCPU:    1000,
		aggregate.ResourceMemory: 2048,
		aggregate.ResourceName("gpu"): 5,
	}
	got := FilterControlledResources(estimation, []aggregate.ResourceName{
		aggregate.ResourceCPU, aggregate.ResourceName("missing"),
	})
	assert.Equal(t, 1, len(got))
	assert.Equal(t, aggregate.ResourceAmount(1000), got[aggregate.ResourceCPU])
}

// ---- HLS pod usage ----

func TestCalculateHlsPodsUsageEmpty(t *testing.T) {
	cpu, mem, err := CalculateHlsPodsUsage(nil)
	assert.NoError(t, err)
	assert.Equal(t, int64(0), cpu.Value())
	assert.Equal(t, int64(0), mem.Value())
}

func TestCalculateHlsPodsUsageWithContainers(t *testing.T) {
	pods := []corev1.Pod{{
		Spec: corev1.PodSpec{
			NodeName: "n1",
			Containers: []corev1.Container{
				{Name: "c1", Resources: corev1.ResourceRequirements{
					Requests: corev1.ResourceList{
						corev1.ResourceCPU:    resource.MustParse("1"),
						corev1.ResourceMemory: resource.MustParse("1Gi"),
					}}},
			},
			InitContainers: []corev1.Container{
				{Name: "init", Resources: corev1.ResourceRequirements{
					Requests: corev1.ResourceList{
						corev1.ResourceCPU:    resource.MustParse("2"),
						corev1.ResourceMemory: resource.MustParse("512Mi"),
					}}},
			},
		},
	}}
	cpu, mem, err := CalculateHlsPodsUsage(pods)
	assert.NoError(t, err)
	// regular containers add (1 core); init uses max → CPU max(1,2)=2
	assert.Equal(t, int64(2), cpu.Value())
	assert.True(t, mem.Value() >= int64(1024*1024*1024))
}

func TestAddResourceIfExists(t *testing.T) {
	total := resource.NewQuantity(5, resource.DecimalSI)
	addResourceIfExists(total, corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("3")}, corev1.ResourceCPU)
	assert.Equal(t, int64(8), total.Value())
	addResourceIfExists(total, corev1.ResourceList{}, corev1.ResourceCPU) // missing key → no change
	assert.Equal(t, int64(8), total.Value())
}

func TestMaxResourceIfGreater(t *testing.T) {
	total := resource.NewQuantity(5, resource.DecimalSI)
	maxResourceIfGreater(total, corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("2")}, corev1.ResourceCPU)
	assert.Equal(t, int64(5), total.Value()) // 2 not greater → unchanged
	maxResourceIfGreater(total, corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("9")}, corev1.ResourceCPU)
	assert.Equal(t, int64(9), total.Value()) // 9 greater → replace
	maxResourceIfGreater(total, corev1.ResourceList{}, corev1.ResourceCPU) // missing key → unchanged
	assert.Equal(t, int64(9), total.Value())
}

func TestScaleQuantityCpuAndMemory(t *testing.T) {
	cpu := scaleQuantityCpu(resource.MustParse("1000m"), 0.9)
	assert.Equal(t, int64(900), cpu.MilliValue())
	mem := scaleQuantityMeomry(resource.MustParse("1000"), 0.5)
	assert.Equal(t, int64(500), mem.Value())
}

// ---- OverHeadNodeAllocate (big branchy function) ----

func TestOverHeadNodeAllocateErrors(t *testing.T) {
	cpu := resource.NewQuantity(1, resource.DecimalSI)
	mem := resource.NewQuantity(1, resource.BinarySI)

	assert.Error(t, OverHeadNodeAllocate(nil, cpu, mem)) // nil state

	ns := newNodeState(nil)
	assert.Error(t, OverHeadNodeAllocate(ns, cpu, mem)) // nil node

	ns = newNodeState(newTestNode("n", corev1.ResourceList{}))
	assert.Error(t, OverHeadNodeAllocate(ns, nil, nil)) // nil req totals

	ns = newNodeState(&corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "n"}})
	assert.Error(t, OverHeadNodeAllocate(ns, cpu, mem)) // nil allocatable

	ns = newNodeState(newTestNode("n", corev1.ResourceList{corev1.ResourceMemory: resource.MustParse("1Gi")}))
	assert.Error(t, OverHeadNodeAllocate(ns, cpu, mem)) // no cpu in allocatable
}

func TestOverHeadNodeAllocateCPU(t *testing.T) {
	cpuReq := resource.MustParse("4") // 4 cores
	memReq := resource.MustParse("1Gi")

	t.Run("allocatable greater than req", func(t *testing.T) {
		node := newTestNode("n", corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("32"),
			corev1.ResourceMemory: resource.MustParse("64Gi"),
		})
		ns := newNodeState(node)
		assert.NoError(t, OverHeadNodeAllocate(ns, &cpuReq, &memReq))
		gotCPU := node.Status.Allocatable[corev1.ResourceCPU]
		assert.True(t, gotCPU.MilliValue() > 0)
	})

	t.Run("allocatable equal to req", func(t *testing.T) {
		node := newTestNode("n", corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("4"),
			corev1.ResourceMemory: resource.MustParse("1Gi"),
		})
		ns := newNodeState(node)
		assert.NoError(t, OverHeadNodeAllocate(ns, &cpuReq, &memReq))
		gotCPUZero := node.Status.Allocatable[corev1.ResourceCPU]
		assert.Equal(t, int64(0), gotCPUZero.Value())
	})

	t.Run("allocatable less than req (cpu)", func(t *testing.T) {
		node := newTestNode("n", corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("2"),
			corev1.ResourceMemory: resource.MustParse("1Gi"),
		})
		ns := newNodeState(node)
		assert.Error(t, OverHeadNodeAllocate(ns, &cpuReq, &memReq))
	})
}

func TestOverHeadNodeAllocateMemoryBranches(t *testing.T) {
	cpuReq := resource.MustParse("2")
	memReqLarge := resource.MustParse("4Gi")
	memReqSmall := resource.MustParse("1Gi")
	t.Run("mem allocatable greater than req", func(t *testing.T) {
		node := newTestNode("n", corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("8"),
			corev1.ResourceMemory: resource.MustParse("16Gi"),
		})
		ns := newNodeState(node)
		assert.NoError(t, OverHeadNodeAllocate(ns, &cpuReq, &memReqLarge))
		gotMem := node.Status.Allocatable[corev1.ResourceMemory]
		assert.True(t, gotMem.Value() > 0)
	})
	t.Run("mem allocatable less than req", func(t *testing.T) {
		node := newTestNode("n", corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("8"),
			corev1.ResourceMemory: resource.MustParse("1Gi"),
		})
		ns := newNodeState(node)
		assert.Error(t, OverHeadNodeAllocate(ns, &cpuReq, &memReqLarge))
	})
	t.Run("no memory in allocatable", func(t *testing.T) {
		node := newTestNode("n", corev1.ResourceList{
			corev1.ResourceCPU: resource.MustParse("8"),
		})
		ns := newNodeState(node)
		assert.Error(t, OverHeadNodeAllocate(ns, &cpuReq, &memReqSmall))
	})
}

// ---- node recommender ----

func TestGetNodeResourceEstimationNil(t *testing.T) {
	r := NewNodeResourceRecommender(&apps.Configuration{})
	assert.Nil(t, r.GetNodeResourceEstimation(nil))
}

func TestEstimateNodeRes(t *testing.T) {
	r := &ResourceRecommender{
		config:          &apps.Configuration{},
		nodeRecommender: NewNodeResourceRecommender(&apps.Configuration{}),
	}
	node := newTestNode("n", corev1.ResourceList{
		corev1.ResourceCPU:    resource.MustParse("32"),
		corev1.ResourceMemory: resource.MustParse("64Gi"),
	})
	ns := newNodeState(node)
	got := r.estimateNodeRes(ns)
	assert.Contains(t, got, corev1.ResourceCPU)
	assert.Contains(t, got, corev1.ResourceMemory)
}

// ---- handleResUpdateJitter ----

func TestHandleResUpdateJitter(t *testing.T) {
	r := &ResourceRecommender{config: &apps.Configuration{
		BERecommendMinCPUMillicores: 100,
		BERecommendMinMemoryMb:      500,
	}}
	t.Run("no extended resource present -> not skip", func(t *testing.T) {
		node := newTestNode("n", corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("32")})
		assert.False(t, r.handleResUpdateJitter(node, corev1.ResourceCPU, resource.MustParse("10")))
	})
	t.Run("cpu extended resource present, big diff -> not skip", func(t *testing.T) {
		node := newTestNode("n", corev1.ResourceList{
			corev1.ResourceCPU:             resource.MustParse("32"),
			common.ExtenderResourceCPU:    resource.MustParse("10"),
			common.ExtenderResourceMemory: resource.MustParse("1Gi"),
		})
		assert.False(t, r.handleResUpdateJitter(node, corev1.ResourceCPU, resource.MustParse("5000")))
	})
	t.Run("memory extended resource present, equal -> skip", func(t *testing.T) {
		node := newTestNode("n", corev1.ResourceList{
			common.ExtenderResourceMemory: resource.MustParse("5Gi"),
		})
		assert.True(t, r.handleResUpdateJitter(node, corev1.ResourceMemory, resource.MustParse("5Gi")))
	})
}

// ---- listHlsPodOnNode (fake client) ----

func TestListHlsPodOnNode(t *testing.T) {
	hlsPod := func(name, node string) *corev1.Pod {
		return &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: name, Namespace: "ns",
				Labels: map[string]string{string(common.QosClassAnnotationKey): string(common.QosHLS)},
			},
			Spec:   corev1.PodSpec{NodeName: node},
			Status: corev1.PodStatus{Phase: corev1.PodRunning},
		}
	}
	cl := fake.NewClientBuilder().WithScheme(mustScheme(t)).
		WithObjects(
			hlsPod("hls-a", "n1"),
			hlsPod("hls-b", "n2"),
			&corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "be-a", Namespace: "ns"}, Spec: corev1.PodSpec{NodeName: "n1"}},
		).Build()

	got, err := listHlsPodOnNode(context.Background(), cl, "n1")
	assert.NoError(t, err)
	for _, p := range got {
		assert.Equal(t, "n1", p.Spec.NodeName)
	}
}

// ---- updateNodeStatus (fake client) ----

func TestUpdateNodeStatusSkipBoth(t *testing.T) {
	r := &ResourceRecommender{}
	assert.NoError(t, r.updateNodeStatus(&corev1.Node{}, corev1.ResourceList{}, true, true))
}

func TestUpdateNodeStatusUpdate(t *testing.T) {
	node := newTestNode("n", corev1.ResourceList{
		corev1.ResourceCPU:    resource.MustParse("32"),
		corev1.ResourceMemory: resource.MustParse("64Gi"),
	})
	cl := fake.NewClientBuilder().WithScheme(mustScheme(t)).
		WithObjects(node).
		WithStatusSubresource(&corev1.Node{}).
		Build()
	r := &ResourceRecommender{Client: cl, ctx: context.Background()}
	recommended := corev1.ResourceList{
		corev1.ResourceCPU:    resource.MustParse("10"),
		corev1.ResourceMemory: resource.MustParse("16Gi"),
	}
	assert.NoError(t, r.updateNodeStatus(node, recommended, false, false))
}

// ---- doUpdateNodeRecommendation (empty state early return) ----

func TestDoUpdateNodeRecommendationEmpty(t *testing.T) {
	cs := aggregate.NewClusterState(context.Background(), &apps.Configuration{})
	r := &ResourceRecommender{clusterState: cs}
	r.doUpdateNodeRecommendation()
}
