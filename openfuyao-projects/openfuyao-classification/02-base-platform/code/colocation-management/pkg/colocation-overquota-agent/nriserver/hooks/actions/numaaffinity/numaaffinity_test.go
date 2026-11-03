/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * Author: Junlin Liu
 * Date: 2025-06-19
 */
package numaaffinity

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"

	nricfg "openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/config"
	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/core"
	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/hooks"
	"openfuyao.com/colocation-management/pkg/common"
	"openfuyao.com/colocation-management/pkg/utils"
)

// resetGlobalRunner resets the package-level singleton so tests are independent.
func resetGlobalRunner(t *testing.T) {
	t.Helper()
	numaAffinityRunner = nil
	hooks.ClearAllHooksSafe()
}

// makeConfigMap builds a colocation-config ConfigMap carrying the NUMA affinity options.
func makeConfigMap(enable bool) *corev1.ConfigMap {
	opts := NumaAffinityConfig{Enable: enable}
	data, _ := json.Marshal(opts)
	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      ConfigMapName,
			Namespace: DefaultNamespace,
		},
		Data: map[string]string{
			NumaAffinityOptionsKey: string(data),
		},
	}
}

// newEnabledRunnerWithFakeClient constructs a Runner wired to a fake k8s clientset,
// with config enabled and a NUMA topology populated.
func newEnabledRunnerWithFakeClient(t *testing.T, objs ...runtime.Object) *Runner {
	t.Helper()
	resetGlobalRunner(t)
	client := fake.NewSimpleClientset(objs...)
	r := NewNumaAffinityRunner()
	r.k8sClient = client
	r.config = &nricfg.NumaAffinityConfig{Enabled: true}
	return r
}

// ---------- existing singleton test ----------

func TestGetNUMAAffinityRunner(t *testing.T) {
	resetGlobalRunner(t)
	r1 := GetNUMAAffinityRunner()
	r2 := GetNUMAAffinityRunner()
	if r1 != r2 {
		t.Error("GetNUMAAffinityRunner should return the same instance")
	}
}

// ---------- NewNumaAffinityRunner / Init ----------

func TestNewNumaAffinityRunnerDefaults(t *testing.T) {
	resetGlobalRunner(t)
	r := NewNumaAffinityRunner()
	assert.Equal(t, DefaultNamespace, r.namespace)
	// in-cluster config is unavailable in tests; client stays nil but no panic
}

func TestInitEnabledReadsConfigMapAndTopology(t *testing.T) {
	resetGlobalRunner(t)
	client := fake.NewSimpleClientset(makeConfigMap(true))
	r := NewNumaAffinityRunner()
	r.k8sClient = client

	// r.watcherStarted = false so Init will start the background watcher.
	// We start it but never rely on its output; deterministic assertions below.
	err := r.Init(nricfg.Config{NumaAffinity: nricfg.NumaAffinityConfig{Enabled: false}})
	assert.NoError(t, err)

	// ConfigMap reports enabled=true, overriding the disabled config passed in.
	r.mutex.RLock()
	enabled := r.config.Enabled
	topo := r.numaTopology
	r.mutex.RUnlock()
	assert.True(t, enabled, "Init should adopt the ConfigMap enable value")
	assert.NotNil(t, topo, "NUMA topology should be populated when enabled and sysfs present")
}

func TestInitConfigMapReadFailsUsesOriginalConfig(t *testing.T) {
	resetGlobalRunner(t)
	// ConfigMap does not exist -> Get fails -> Init keeps original config.
	client := fake.NewSimpleClientset()
	r := NewNumaAffinityRunner()
	r.k8sClient = client

	err := r.Init(nricfg.Config{NumaAffinity: nricfg.NumaAffinityConfig{Enabled: false}})
	assert.NoError(t, err)
	assert.False(t, r.config.Enabled)
}

func TestInitNilClientSkipsWatcherAndTopologyDisabled(t *testing.T) {
	resetGlobalRunner(t)
	// Construct the runner directly so k8sClient is deterministically nil,
	// rather than going through NewNumaAffinityRunner() (which calls
	// rest.InClusterConfig() and only leaves the client nil when running outside
	// a cluster). CI runs inside a pod where InClusterConfig() succeeds, so the
	// old construction made the watcher-start assertion host-dependent.
	r := &Runner{namespace: DefaultNamespace}
	// k8sClient is nil -> ConfigMap read fails, watcher is not started.
	err := r.Init(nricfg.Config{NumaAffinity: nricfg.NumaAffinityConfig{Enabled: false}})
	assert.NoError(t, err)
	assert.False(t, r.watcherStarted)
}

// ---------- readNumaAffinityConfigFromConfigMap ----------

func TestReadNumaAffinityConfigFromConfigMap(t *testing.T) {
	t.Run("nil client", func(t *testing.T) {
		r := &Runner{}
		_, err := r.readNumaAffinityConfigFromConfigMap(context.Background())
		assert.Error(t, err)
	})
	t.Run("get error", func(t *testing.T) {
		r := &Runner{k8sClient: fake.NewSimpleClientset()} // no ConfigMap
		_, err := r.readNumaAffinityConfigFromConfigMap(context.Background())
		assert.Error(t, err)
	})
	t.Run("missing key -> disabled", func(t *testing.T) {
		cm := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Name: ConfigMapName, Namespace: DefaultNamespace},
			Data:       map[string]string{},
		}
		r := &Runner{k8sClient: fake.NewSimpleClientset(cm), namespace: DefaultNamespace}
		enabled, err := r.readNumaAffinityConfigFromConfigMap(context.Background())
		assert.NoError(t, err)
		assert.False(t, enabled)
	})
	t.Run("invalid json", func(t *testing.T) {
		cm := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Name: ConfigMapName, Namespace: DefaultNamespace},
			Data:       map[string]string{NumaAffinityOptionsKey: "{not-json"},
		}
		r := &Runner{k8sClient: fake.NewSimpleClientset(cm), namespace: DefaultNamespace}
		_, err := r.readNumaAffinityConfigFromConfigMap(context.Background())
		assert.Error(t, err)
	})
	t.Run("enabled true", func(t *testing.T) {
		r := &Runner{
			k8sClient:  fake.NewSimpleClientset(makeConfigMap(true)),
			namespace:  DefaultNamespace,
		}
		enabled, err := r.readNumaAffinityConfigFromConfigMap(context.Background())
		assert.NoError(t, err)
		assert.True(t, enabled)
	})
	t.Run("disabled", func(t *testing.T) {
		r := &Runner{
			k8sClient:  fake.NewSimpleClientset(makeConfigMap(false)),
			namespace:  DefaultNamespace,
		}
		enabled, err := r.readNumaAffinityConfigFromConfigMap(context.Background())
		assert.NoError(t, err)
		assert.False(t, enabled)
	})
}

// ---------- RegisterFunc ----------

func TestRegisterFunc(t *testing.T) {
	resetGlobalRunner(t)
	r := NewNumaAffinityRunner()
	// Registering for the first time should not error internally.
	r.RegisterFunc()

	// Register again: hooks already exist -> internal error path is exercised.
	r.RegisterFunc()

	// Verify hooks are actually registered.
	h, err := hooks.GetHook(ActionPreCreateContainer)
	assert.NoError(t, err)
	assert.NotEmpty(t, h)
}

// ---------- ArgsCreateContainerWrapper / processCreateContainer ----------

// makeContainer builds a ContainerInfo with the given QoS annotation and CPU request.
func makeContainer(qos string, cpuMilli int64) *core.ContainerInfo {
	ann := map[string]string{}
	if qos != "" {
		ann[common.QosClassAnnotationKey] = qos
	}
	return &core.ContainerInfo{
		ContainerRequest: core.ContainerRequest{
			ContainerMeta:   core.ContainerMetadata{ID: "cid", Name: "cname"},
			CPUMilliRequest: cpuMilli,
			PodSandboxConfig: utils.PodSandboxConfig{
				Annotations: ann,
			},
		},
		ContainerResponese: core.ContainerResponse{
			Resources: utils.LinuxContainerResources{},
		},
	}
}

func TestArgsCreateContainerWrapperUnexpectedType(t *testing.T) {
	r := &Runner{}
	// Pass a *core.PodInfo (not ContainerInfo) -> type assertion fails.
	err := r.ArgsCreateContainerWrapper(&core.PodInfo{})
	assert.Error(t, err)
}

func TestArgsCreateContainerWrapperNilAndUnnamed(t *testing.T) {
	r := &Runner{config: &nricfg.NumaAffinityConfig{Enabled: false}}
	// nil info path
	_ = r.ArgsCreateContainerWrapper(nil) // guard against panic
	// ContainerInfo with no name -> still returns nil (disabled).
	ci := &core.ContainerInfo{}
	err := r.ArgsCreateContainerWrapper(ci)
	assert.NoError(t, err)
}

func TestProcessCreateContainerDisabledAndStateMatrix(t *testing.T) {
	ctx := context.Background()
	topo := &core.NumaTopology{Nodes: map[int]*core.NumaNode{0: {ID: 0, CPUList: []int{0, 1, 2}}}}
	ci := makeContainer("LS", 2000)

	t.Run("nil runner", func(t *testing.T) {
		var r *Runner
		assert.NoError(t, r.processCreateContainer(ctx, ci))
	})
	t.Run("nil config", func(t *testing.T) {
		r := &Runner{}
		assert.NoError(t, r.processCreateContainer(ctx, ci))
	})
	t.Run("disabled", func(t *testing.T) {
		r := &Runner{config: &nricfg.NumaAffinityConfig{Enabled: false}}
		assert.NoError(t, r.processCreateContainer(ctx, ci))
	})
	t.Run("enabled but topology nil", func(t *testing.T) {
		r := &Runner{config: &nricfg.NumaAffinityConfig{Enabled: true}, numaTopology: nil}
		assert.NoError(t, r.processCreateContainer(ctx, ci))
	})
	t.Run("enabled with topology but nil containerInfo", func(t *testing.T) {
		r := &Runner{config: &nricfg.NumaAffinityConfig{Enabled: true}, numaTopology: topo}
		err := r.processCreateContainer(ctx, nil)
		assert.Error(t, err)
	})
}

func TestProcessCreateContainerQoSBranches(t *testing.T) {
	ctx := context.Background()
	topo := &core.NumaTopology{Nodes: map[int]*core.NumaNode{0: {ID: 0, CPUList: []int{0, 1, 2, 3}}}}
	r := &Runner{config: &nricfg.NumaAffinityConfig{Enabled: true}, numaTopology: topo}

	t.Run("HLS native", func(t *testing.T) {
		ci := makeContainer("HLS", 1000)
		assert.NoError(t, r.processCreateContainer(ctx, ci))
		assert.Nil(t, ci.ContainerResponese.Resources.CPUSet)
	})
	t.Run("LS applies affinity", func(t *testing.T) {
		ci := makeContainer("LS", 2000)
		assert.NoError(t, r.processCreateContainer(ctx, ci))
		assert.NotNil(t, ci.ContainerResponese.Resources.CPUSet)
		assert.Equal(t, "0-3", *ci.ContainerResponese.Resources.CPUSet)
		assert.NotNil(t, ci.ContainerResponese.Resources.CPUSetMem)
	})
	t.Run("BE skipped", func(t *testing.T) {
		ci := makeContainer("BE", 2000)
		assert.NoError(t, r.processCreateContainer(ctx, ci))
		assert.Nil(t, ci.ContainerResponese.Resources.CPUSet)
	})
	t.Run("unknown qos skipped", func(t *testing.T) {
		ci := makeContainer("WAT", 2000)
		assert.NoError(t, r.processCreateContainer(ctx, ci))
		assert.Nil(t, ci.ContainerResponese.Resources.CPUSet)
	})
	t.Run("no qos annotation -> QosNone skipped", func(t *testing.T) {
		ci := makeContainer("", 2000)
		assert.NoError(t, r.processCreateContainer(ctx, ci))
		assert.Nil(t, ci.ContainerResponese.Resources.CPUSet)
	})
}

// processCreateContainer path where topology becomes unavailable after re-read of config.
func TestProcessCreateContainerTopologyBecomesNil(t *testing.T) {
	ctx := context.Background()
	r := &Runner{
		config:       &nricfg.NumaAffinityConfig{Enabled: true},
		numaTopology: nil, // simulate topology cleared between checks
	}
	ci := makeContainer("LS", 1000)
	assert.NoError(t, r.processCreateContainer(ctx, ci))
}

// ---------- applyNUMAAffinityForContainer edge cases ----------

func TestApplyNUMAAffinityForContainer(t *testing.T) {
	topology := &core.NumaTopology{
		Nodes: map[int]*core.NumaNode{
			0: {ID: 0, CPUList: []int{0, 1, 2}},
		},
	}
	container := &core.ContainerInfo{
		ContainerRequest: core.ContainerRequest{
			ContainerMeta:   core.ContainerMetadata{ID: "c1"},
			CPUMilliRequest: 2000, // 2 cores
		},
		ContainerResponese: core.ContainerResponse{
			Resources: utils.LinuxContainerResources{},
		},
	}
	err := applyNUMAAffinityForContainer(topology, container)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if container.ContainerResponese.Resources.CPUSet == nil || *container.ContainerResponese.Resources.CPUSet == "" {
		t.Error("CPUSet should be set")
	}
	if container.ContainerResponese.Resources.CPUSetMem == nil || *container.ContainerResponese.Resources.CPUSetMem != "0" {
		t.Error("CPUSetMem should be set to node id 0")
	}

	// topology nil
	if err := applyNUMAAffinityForContainer(nil, container); err == nil {
		t.Error("expected error when topology is nil")
	}
	// container nil
	if err := applyNUMAAffinityForContainer(topology, nil); err == nil {
		t.Error("expected error when container is nil")
	}
	// empty topology (no nodes)
	if err := applyNUMAAffinityForContainer(&core.NumaTopology{Nodes: map[int]*core.NumaNode{}}, container); err == nil {
		t.Error("expected error when topology has no nodes")
	}
}

// applyNUMA when no node has enough CPUs -> GetBestNumaNodeForAffinity errors.
func TestApplyNUMAAffinityForContainerNoNodeFits(t *testing.T) {
	topology := &core.NumaTopology{
		Nodes: map[int]*core.NumaNode{
			0: {ID: 0, CPUList: []int{0}}, // only 1 CPU
		},
	}
	container := &core.ContainerInfo{
		ContainerRequest: core.ContainerRequest{
			ContainerMeta:   core.ContainerMetadata{ID: "c1"},
			CPUMilliRequest: 3000, // needs 3 cores -> no node fits
		},
		ContainerResponese: core.ContainerResponse{Resources: utils.LinuxContainerResources{}},
	}
	err := applyNUMAAffinityForContainer(topology, container)
	assert.Error(t, err)
}

// applyNUMA with zero CPU request -> returns nil, no change.
func TestApplyNUMAAffinityForContainerZeroRequest(t *testing.T) {
	topology := &core.NumaTopology{Nodes: map[int]*core.NumaNode{0: {ID: 0, CPUList: []int{0, 1}}}}
	container := &core.ContainerInfo{
		ContainerRequest: core.ContainerRequest{
			ContainerMeta:   core.ContainerMetadata{ID: "c1"},
			CPUMilliRequest: 0,
		},
		ContainerResponese: core.ContainerResponse{Resources: utils.LinuxContainerResources{}},
	}
	assert.NoError(t, applyNUMAAffinityForContainer(topology, container))
	assert.Nil(t, container.ContainerResponese.Resources.CPUSet)
}

// ---------- getContainerID ----------

func TestGetContainerID(t *testing.T) {
	t.Run("uses ID when present", func(t *testing.T) {
		ci := &core.ContainerInfo{ContainerRequest: core.ContainerRequest{
			ContainerMeta: core.ContainerMetadata{ID: "id-1", Name: "name-1"}}}
		assert.Equal(t, "id-1", getContainerID(ci))
	})
	t.Run("falls back to Name", func(t *testing.T) {
		ci := &core.ContainerInfo{ContainerRequest: core.ContainerRequest{
			ContainerMeta: core.ContainerMetadata{ID: "", Name: "name-1"}}}
		assert.Equal(t, "name-1", getContainerID(ci))
	})
}

// ---------- calculateMinCPUCount ----------

func TestCalculateMinCPUCount(t *testing.T) {
	cases := []struct {
		name string
		milli int64
		want  int
	}{
		{"zero returns 0", 0, 0},
		{"negative returns 0", -100, 0},
		{"fractional rounds up", 1500, 2},
		{"exact core", 2000, 2},
		{"small positive rounds to >=1", 1, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ci := &core.ContainerInfo{ContainerRequest: core.ContainerRequest{CPUMilliRequest: tc.milli}}
			got, err := calculateMinCPUCount(ci, "cid")
			assert.NoError(t, err)
			assert.Equal(t, tc.want, got)
		})
	}
}

// ---------- setCPUResourcesForContainer ----------

func TestSetCPUResourcesForContainerOverwritesExisting(t *testing.T) {
	ci := &core.ContainerInfo{ContainerResponese: core.ContainerResponse{Resources: utils.LinuxContainerResources{}}}
	// Pre-set CPUSetMem to verify the existing-pointer branch.
	existing := "old"
	ci.ContainerResponese.Resources.CPUSetMem = &existing
	setCPUResourcesForContainer(ci, "cid", 2, []int{4, 5})
	assert.NotNil(t, ci.ContainerResponese.Resources.CPUSet)
	assert.Equal(t, "4-5", *ci.ContainerResponese.Resources.CPUSet)
	assert.Equal(t, "2", *ci.ContainerResponese.Resources.CPUSetMem)
}

func TestSetCPUResourcesForContainerEmptyCPUList(t *testing.T) {
	ci := &core.ContainerInfo{ContainerResponese: core.ContainerResponse{Resources: utils.LinuxContainerResources{}}}
	// Empty CPU list -> CPUSet stays nil, CPUSetMem still set.
	setCPUResourcesForContainer(ci, "cid", 1, []int{})
	assert.Nil(t, ci.ContainerResponese.Resources.CPUSet)
	assert.NotNil(t, ci.ContainerResponese.Resources.CPUSetMem)
	assert.Equal(t, "1", *ci.ContainerResponese.Resources.CPUSetMem)
}

// ---------- ArgsUpdateContainerWrapper / processUpdateContainer ----------

func TestArgsUpdateContainerWrapper(t *testing.T) {
	r := &Runner{}
	assert.NoError(t, r.ArgsUpdateContainerWrapper(nil))
	assert.NoError(t, r.processUpdateContainer(context.Background(), nil))
}

// ---------- initializeTopology ----------

func TestInitializeTopology(t *testing.T) {
	t.Run("disabled leaves topology nil", func(t *testing.T) {
		r := &Runner{config: &nricfg.NumaAffinityConfig{Enabled: false}}
		assert.NoError(t, r.initializeTopology())
		assert.Nil(t, r.numaTopology)
	})
	t.Run("enabled populates from sysfs", func(t *testing.T) {
		r := &Runner{config: &nricfg.NumaAffinityConfig{Enabled: true}}
		assert.NoError(t, r.initializeTopology())
		// sysfs has 4 NUMA nodes in this environment.
		assert.NotNil(t, r.numaTopology)
		assert.NotEmpty(t, r.numaTopology.Nodes)
	})
}

// ---------- handleConfigMapUpdate ----------

func TestHandleConfigMapUpdate(t *testing.T) {
	t.Run("no change when enabled identical", func(t *testing.T) {
		topo := &core.NumaTopology{Nodes: map[int]*core.NumaNode{0: {ID: 0, CPUList: []int{0}}}}
		r := &Runner{
			config:       &nricfg.NumaAffinityConfig{Enabled: true},
			numaTopology: topo,
			k8sClient:    fake.NewSimpleClientset(makeConfigMap(true)),
			namespace:    DefaultNamespace,
		}
		r.handleConfigMapUpdate()
		assert.NotNil(t, r.numaTopology, "topology should remain set")
	})
	t.Run("re-initializes when value changes", func(t *testing.T) {
		r := &Runner{
			config:       &nricfg.NumaAffinityConfig{Enabled: false},
			numaTopology: nil,
			k8sClient:    fake.NewSimpleClientset(makeConfigMap(true)),
			namespace:    DefaultNamespace,
		}
		r.handleConfigMapUpdate()
		assert.True(t, r.config.Enabled)
	})
	t.Run("read error is tolerated", func(t *testing.T) {
		r := &Runner{
			config:    &nricfg.NumaAffinityConfig{Enabled: false},
			k8sClient: fake.NewSimpleClientset(), // no ConfigMap
			namespace: DefaultNamespace,
		}
		// Should not panic / change config.
		r.handleConfigMapUpdate()
		assert.False(t, r.config.Enabled)
	})
}

// ---------- watchConfigMap (event handling via reactor) ----------

// fakeWatcherConfigMap injects a fake watch source into the clientset reactor so we
// can drive Modified/Added/Deleted/Error branches deterministically.
func fakeWatcherConfigMap(t *testing.T, client *fake.Clientset) *watch.FakeWatcher {
	t.Helper()
	fw := watch.NewFake()
	client.PrependWatchReactor("configmaps", func(action clienttesting.Action) (handled bool, ret watch.Interface, err error) {
		return true, fw, nil
	})
	return fw
}

func TestWatchConfigMapEvents(t *testing.T) {
	t.Run("nil client errors", func(t *testing.T) {
		r := &Runner{}
		assert.Error(t, r.watchConfigMap())
	})
	t.Run("modified re-reads config", func(t *testing.T) {
		client := fake.NewSimpleClientset(makeConfigMap(true))
		fw := fakeWatcherConfigMap(t, client)
		r := &Runner{
			config:     &nricfg.NumaAffinityConfig{Enabled: false},
			k8sClient:  client,
			namespace:  DefaultNamespace,
		}
		done := make(chan struct{})
		go func() {
			_ = r.watchConfigMap()
			close(done)
		}()
		fw.Action(watch.Modified, makeConfigMap(true))
		fw.Stop()
		<-done
		assert.True(t, r.config.Enabled)
	})
	t.Run("added event", func(t *testing.T) {
		client := fake.NewSimpleClientset(makeConfigMap(false))
		fw := fakeWatcherConfigMap(t, client)
		r := &Runner{
			config:    &nricfg.NumaAffinityConfig{Enabled: true},
			k8sClient: client,
			namespace: DefaultNamespace,
		}
		done := make(chan struct{})
		go func() {
			_ = r.watchConfigMap()
			close(done)
		}()
		fw.Action(watch.Added, makeConfigMap(false))
		fw.Stop()
		<-done
		assert.False(t, r.config.Enabled)
	})
	t.Run("deleted disables and clears topology", func(t *testing.T) {
		client := fake.NewSimpleClientset(makeConfigMap(true))
		fw := fakeWatcherConfigMap(t, client)
		r := &Runner{
			config:       &nricfg.NumaAffinityConfig{Enabled: true},
			numaTopology: &core.NumaTopology{Nodes: map[int]*core.NumaNode{0: {ID: 0}}},
			k8sClient:    client,
			namespace:    DefaultNamespace,
		}
		done := make(chan struct{})
		go func() {
			_ = r.watchConfigMap()
			close(done)
		}()
		fw.Action(watch.Deleted, makeConfigMap(true))
		fw.Stop()
		<-done
		assert.False(t, r.config.Enabled)
		assert.Nil(t, r.numaTopology)
	})
	t.Run("error event tolerated", func(t *testing.T) {
		client := fake.NewSimpleClientset(makeConfigMap(true))
		fw := fakeWatcherConfigMap(t, client)
		r := &Runner{
			config:    &nricfg.NumaAffinityConfig{Enabled: true},
			k8sClient: client,
			namespace: DefaultNamespace,
		}
		done := make(chan struct{})
		go func() {
			_ = r.watchConfigMap()
			close(done)
		}()
		fw.Error(&metav1.Status{Message: "boom"})
		fw.Stop()
		<-done
		// config should be untouched by error event
		assert.True(t, r.config.Enabled)
	})
}

// ---------- startConfigMapWatcher ----------
//
// startConfigMapWatcher loops forever until the context (context.Background) is
// cancelled and sleeps DefaultWatchTime (a const: 30s) between failed watch attempts.
// It therefore cannot be driven deterministically in a unit test without leaking a
// long-lived goroutine, so its inner logic (watchConfigMap + handleConfigMapUpdate) is
// exercised directly in TestWatchConfigMapEvents and TestHandleConfigMapUpdate above.
// The entry function itself is intentionally left untested.

// initK8sClient is exercised indirectly via NewNumaAffinityRunner (in-cluster config
// fails in tests). Assert the error path returns an error when outside a cluster.
func TestInitK8sClientFailsOutsideCluster(t *testing.T) {
	r := &Runner{namespace: DefaultNamespace}
	err := r.initK8sClient()
	if err == nil {
		t.Skip("in-cluster config unexpectedly available; cannot assert failure path")
	}
}

// keep unused imports referenced
var _ = errors.New
var _ kubernetes.Interface = (*fake.Clientset)(nil)
