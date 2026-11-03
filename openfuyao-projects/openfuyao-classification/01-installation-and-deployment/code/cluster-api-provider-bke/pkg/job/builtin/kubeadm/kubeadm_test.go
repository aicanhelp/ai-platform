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

package kubeadm

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/pkg/errors"
	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	"sigs.k8s.io/controller-runtime/pkg/client"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	bkenetutil "gopkg.openfuyao.cn/cluster-api-provider-bke/common/utils/net"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/clientutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/mfutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/pkiutil"
)

const (
	numOne        = 1
	numTwo        = 2
	numThree      = 3
	numFive       = 5
	numTen        = 10
	numSixty      = 60
	numOneHundred = 100

	testHostName    = "test-node"
	testHostIP      = "192.168.1.10"
	testClusterName = "test-cluster"
	testNamespace   = "test-namespace"
	testVersion     = "v1.26.0"
	testEtcdVersion = "3.6.4"
)

type mockExecutor struct {
	exec.Executor
}

func (m *mockExecutor) WithContext(_ context.Context) exec.Executor {
	return m
}

func (m *mockExecutor) ExecuteCommand(_ string, _ ...string) error {
	return nil
}

func (m *mockExecutor) ExecuteCommandWithEnv(_ []string, _ string, _ ...string) error {
	return nil
}

func (m *mockExecutor) ExecuteCommandWithOutput(_ string, _ ...string) (string, error) {
	return "", nil
}

func (m *mockExecutor) ExecuteCommandWithCombinedOutput(_ string, _ ...string) (string, error) {
	return "", nil
}

func (m *mockExecutor) ExecuteCommandWithCombinedOutputInDir(_ string, command string, arg ...string) (string, error) {
	return m.ExecuteCommandWithCombinedOutput(command, arg...)
}

func (m *mockExecutor) ExecuteCommandWithOutputFile(_ string, _ string, _ ...string) (string, error) {
	return "", nil
}

func (m *mockExecutor) ExecuteCommandWithOutputFileTimeout(_ time.Duration, _ string, _ string, _ ...string) (string, error) {
	return "", nil
}

func (m *mockExecutor) ExecuteCommandWithTimeout(_ time.Duration, _ string, _ ...string) (string, error) {
	return "", nil
}

func (m *mockExecutor) ExecuteCommandResidentBinary(_ time.Duration, _ string, _ ...string) error {
	return nil
}

type fakeClient struct {
	client.Client
	shouldUpdate bool
	createCount  int
}

func (m *fakeClient) Get(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
	return nil
}

func (m *fakeClient) List(ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
	return nil
}

func (m *fakeClient) Create(ctx context.Context, obj client.Object, opts ...client.CreateOption) error {
	m.createCount++
	if m.createCount > 1 {
		return errors.New("already exists")
	}
	return nil
}

func (m *fakeClient) Update(ctx context.Context, obj client.Object, opts ...client.UpdateOption) error {
	if m.shouldUpdate {
		return nil
	}
	return errors.New("update error")
}

func (m *fakeClient) Delete(ctx context.Context, obj client.Object, opts ...client.DeleteOption) error {
	return nil
}

func (m *fakeClient) Patch(ctx context.Context, obj client.Object, patch client.Patch, opts ...client.PatchOption) error {
	return nil
}

func (m *fakeClient) DeleteAllOf(ctx context.Context, obj client.Object, opts ...client.DeleteAllOfOption) error {
	return nil
}

func (m *fakeClient) Status() client.StatusWriter {
	return &mockStatusWriter{}
}

func (m *fakeClient) Scheme() *runtime.Scheme {
	return nil
}

type mockStatusWriter struct{}

func (m *mockStatusWriter) Create(ctx context.Context, obj client.Object, sub client.Object, opts ...client.SubResourceCreateOption) error {
	return nil
}

func (m *mockStatusWriter) Update(ctx context.Context, obj client.Object, opts ...client.SubResourceUpdateOption) error {
	return nil
}

func (m *mockStatusWriter) Patch(ctx context.Context, obj client.Object, patch client.Patch, opts ...client.SubResourcePatchOption) error {
	return nil
}

func TestName(t *testing.T) {
	kp := &KubeadmPlugin{}
	assert.Equal(t, Name, kp.Name())
}

func TestNew(t *testing.T) {
	mockExec := &mockExecutor{}
	p := New(mockExec, nil)

	kp, ok := p.(*KubeadmPlugin)
	assert.True(t, ok)
	assert.NotNil(t, kp)
	assert.Equal(t, mockExec, kp.exec)
	assert.Nil(t, kp.k8sClient)
	assert.NotNil(t, kp.boot)
	assert.Equal(t, "", kp.controlPlaneEndpoint)
}

func TestNewWithClient(t *testing.T) {
	mockExec := &mockExecutor{}
	mockClient := &fakeClient{}

	p := New(mockExec, mockClient)

	kp := p.(*KubeadmPlugin)
	assert.Equal(t, mockExec, kp.exec)
	assert.Equal(t, mockClient, kp.k8sClient)
}

func TestParam(t *testing.T) {
	kp := &KubeadmPlugin{}
	params := kp.Param()

	assert.Contains(t, params, "phase")
	assert.Contains(t, params, "bkeConfig")
	assert.Contains(t, params, "backUpEtcd")
	assert.Contains(t, params, "clusterType")
	assert.Contains(t, params, "etcdVersion")
}

func TestParamRequiredFields(t *testing.T) {
	kp := &KubeadmPlugin{}
	params := kp.Param()

	assert.True(t, params["phase"].Required)
	assert.False(t, params["bkeConfig"].Required)
	assert.False(t, params["backUpEtcd"].Required)
	assert.False(t, params["clusterType"].Required)
	assert.False(t, params["etcdVersion"].Required)
}

func TestParamDefaultValues(t *testing.T) {
	kp := &KubeadmPlugin{}
	params := kp.Param()

	assert.Equal(t, "initControlPlane", params["phase"].Default)
	assert.Equal(t, "false", params["backUpEtcd"].Default)
	assert.Equal(t, "bke", params["clusterType"].Default)
	assert.Equal(t, "", params["etcdVersion"].Default)
}

func TestExecuteWithParseCommandsError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockExec := &mockExecutor{}
	kp := &KubeadmPlugin{
		exec: mockExec,
	}

	patches.ApplyFunc(plugin.ParseCommands,
		func(plugin.Plugin, []string) (map[string]string, error) {
			return nil, errors.New("parse error")
		})

	_, err := kp.Execute(context.Background(), []string{Name})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "parse error")
}

func TestExecuteWithUnknownPhase(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockExec := &mockExecutor{}
	kp := &KubeadmPlugin{
		exec: mockExec,
	}

	patches.ApplyFunc(plugin.ParseCommands,
		func(plugin.Plugin, []string) (map[string]string, error) {
			return map[string]string{
				"phase": "unknownPhase",
			}, nil
		})

	_, err := kp.Execute(context.Background(), []string{Name})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unknown command")
}

func TestExecuteInitControlPlane(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockExec := &mockExecutor{}
	kp := &KubeadmPlugin{
		exec:      mockExec,
		boot:      &mfutil.BootScope{},
		isManager: false,
	}

	patches.ApplyMethod(kp, "Execute", func(_ *KubeadmPlugin, _ context.Context, _ []string) ([]string, error) {
		return nil, nil
	})

	result, err := kp.Execute(context.Background(), []string{Name})

	assert.NoError(t, err)
	assert.Nil(t, result)
}

func TestExecuteInitControlPlaneWithManager(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockExec := &mockExecutor{}
	kp := &KubeadmPlugin{
		exec:           mockExec,
		boot:           &mfutil.BootScope{},
		isManager:      true,
		GableNameSpace: testNamespace,
		clusterName:    testClusterName,
	}

	patches.ApplyMethod(kp, "Execute", func(_ *KubeadmPlugin, _ context.Context, _ []string) ([]string, error) {
		return nil, nil
	})

	result, err := kp.Execute(context.Background(), []string{Name})

	assert.NoError(t, err)
	assert.Nil(t, result)
}

func TestExecuteJoinControlPlane(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockExec := &mockExecutor{}
	kp := &KubeadmPlugin{
		exec:      mockExec,
		boot:      &mfutil.BootScope{},
		isManager: false,
	}

	patches.ApplyMethod(kp, "Execute", func(_ *KubeadmPlugin, _ context.Context, _ []string) ([]string, error) {
		return nil, nil
	})

	result, err := kp.Execute(context.Background(), []string{Name})

	assert.NoError(t, err)
	assert.Nil(t, result)
}

func TestExecuteJoinWorker(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockExec := &mockExecutor{}
	kp := &KubeadmPlugin{
		exec:      mockExec,
		boot:      &mfutil.BootScope{},
		isManager: false,
	}

	patches.ApplyMethod(kp, "Execute", func(_ *KubeadmPlugin, _ context.Context, _ []string) ([]string, error) {
		return nil, nil
	})

	result, err := kp.Execute(context.Background(), []string{Name})

	assert.NoError(t, err)
	assert.Nil(t, result)
}

func TestExecuteUpgradeControlPlane(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockExec := &mockExecutor{}
	kp := &KubeadmPlugin{
		exec:      mockExec,
		boot:      &mfutil.BootScope{},
		isManager: false,
	}

	patches.ApplyMethod(kp, "Execute", func(_ *KubeadmPlugin, _ context.Context, _ []string) ([]string, error) {
		return nil, nil
	})

	result, err := kp.Execute(context.Background(), []string{Name})

	assert.NoError(t, err)
	assert.Nil(t, result)
}

func TestKubeadmPluginFields(t *testing.T) {
	mockExec := &mockExecutor{}
	kp := &KubeadmPlugin{
		k8sClient:            nil,
		localK8sClient:       nil,
		exec:                 mockExec,
		boot:                 &mfutil.BootScope{},
		isManager:            true,
		clusterName:          testClusterName,
		controlPlaneEndpoint: "192.168.1.100:6443",
		GableNameSpace:       testNamespace,
	}

	assert.Equal(t, mockExec, kp.exec)
	assert.True(t, kp.isManager)
	assert.Equal(t, testClusterName, kp.clusterName)
	assert.Equal(t, "192.168.1.100:6443", kp.controlPlaneEndpoint)
	assert.Equal(t, testNamespace, kp.GableNameSpace)
}

func TestConstants(t *testing.T) {
	assert.Equal(t, "Kubeadm", Name)
	assert.Equal(t, 500*time.Millisecond, PollImmeInternal)
	assert.Equal(t, 1*time.Minute, PollImmeTimeout)
}

func TestExecuteUpgradeControlPlaneWithBocloud(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockExec := &mockExecutor{}

	kp := &KubeadmPlugin{
		exec: mockExec,
		boot: createTestBootScope(),
	}

	patches.ApplyMethod(kp, "Execute", func(_ *KubeadmPlugin, _ context.Context, _ []string) ([]string, error) {
		return nil, nil
	})

	result, err := kp.Execute(context.Background(), []string{Name})

	assert.NoError(t, err)
	assert.Nil(t, result)
}

func TestGetBKEConfigWithControlPlaneEndpoint(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockExec := &mockExecutor{}
	kp := &KubeadmPlugin{
		exec:      mockExec,
		boot:      &mfutil.BootScope{},
		isManager: false,
	}

	testConfig := &bkev1beta1.BKEConfig{
		Cluster: bkev1beta1.Cluster{
			KubernetesVersion: testVersion,
		},
	}

	patches.ApplyFunc(plugin.ParseCommands,
		func(plugin.Plugin, []string) (map[string]string, error) {
			return map[string]string{
				"phase":     utils.InitControlPlane,
				"bkeConfig": testNamespace + ":" + testClusterName,
			}, nil
		})

	patches.ApplyFunc(plugin.GetBKECluster,
		func(string) (*bkev1beta1.BKECluster, error) {
			return &bkev1beta1.BKECluster{
				ObjectMeta: metav1.ObjectMeta{
					Name:      testClusterName,
					Namespace: testNamespace,
				},
				Spec: bkev1beta1.BKEClusterSpec{
					ClusterConfig: testConfig,
					ControlPlaneEndpoint: bkev1beta1.APIEndpoint{
						Host: "192.168.1.100",
						Port: 6443,
					},
				},
			}, nil
		})

	patches.ApplyFunc(plugin.GetBkeConfigFromBkeCluster,
		func(*bkev1beta1.BKECluster) (*bkev1beta1.BKEConfig, error) {
			return testConfig, nil
		})

	patches.ApplyFunc(plugin.GetClusterData,
		func(string) (*plugin.ClusterData, error) {
			return &plugin.ClusterData{
				Nodes: bkenode.Nodes{{IP: testHostIP, Role: []string{"master"}}},
			}, nil
		})

	patches.ApplyFunc(bkenetutil.GetAllInterfaceIP, func() ([]string, error) {
		return []string{testHostIP}, nil
	})

	patches.ApplyFunc(utils.HostName, func() string {
		return testHostName
	})

	err := kp.getBKEConfig(testNamespace + ":" + testClusterName)

	assert.NoError(t, err)
	assert.Equal(t, "192.168.1.100", kp.controlPlaneEndpoint)
}

func TestGetBKEConfigWithoutManagerAddon(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockExec := &mockExecutor{}
	kp := &KubeadmPlugin{
		exec:      mockExec,
		boot:      &mfutil.BootScope{},
		isManager: false,
	}

	testConfig := &bkev1beta1.BKEConfig{
		Cluster: bkev1beta1.Cluster{
			KubernetesVersion: testVersion,
		},
	}

	patches.ApplyFunc(plugin.GetBKECluster,
		func(string) (*bkev1beta1.BKECluster, error) {
			return &bkev1beta1.BKECluster{
				ObjectMeta: metav1.ObjectMeta{
					Name:      testClusterName,
					Namespace: testNamespace,
				},
				Spec: bkev1beta1.BKEClusterSpec{
					ClusterConfig: testConfig,
				},
			}, nil
		})

	patches.ApplyFunc(plugin.GetBkeConfigFromBkeCluster,
		func(*bkev1beta1.BKECluster) (*bkev1beta1.BKEConfig, error) {
			return testConfig, nil
		})

	patches.ApplyFunc(plugin.GetClusterData,
		func(string) (*plugin.ClusterData, error) {
			return &plugin.ClusterData{
				Nodes: bkenode.Nodes{{IP: testHostIP, Role: []string{"master"}}},
			}, nil
		})

	patches.ApplyFunc(bkenetutil.GetAllInterfaceIP, func() ([]string, error) {
		return []string{testHostIP}, nil
	})

	patches.ApplyFunc(utils.HostName, func() string {
		return testHostName
	})

	err := kp.getBKEConfig(testNamespace + ":" + testClusterName)

	assert.NoError(t, err)
	assert.False(t, kp.isManager)
}

func TestExecuteWithBkeConfigError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockExec := &mockExecutor{}
	kp := &KubeadmPlugin{
		exec: mockExec,
	}

	patches.ApplyFunc(plugin.ParseCommands,
		func(plugin.Plugin, []string) (map[string]string, error) {
			return map[string]string{
				"phase":     utils.InitControlPlane,
				"bkeConfig": "invalid:config",
			}, nil
		})

	patches.ApplyFunc(plugin.GetBKECluster,
		func(string) (*bkev1beta1.BKECluster, error) {
			return nil, errors.New("failed to get BKECluster")
		})

	_, err := kp.Execute(context.Background(), []string{Name})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to get BKECluster")
}

func TestExecuteWithGetBkeConfigFromBkeClusterError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockExec := &mockExecutor{}
	kp := &KubeadmPlugin{
		exec: mockExec,
	}

	patches.ApplyFunc(plugin.ParseCommands,
		func(plugin.Plugin, []string) (map[string]string, error) {
			return map[string]string{
				"phase":     utils.InitControlPlane,
				"bkeConfig": testNamespace + ":" + testClusterName,
			}, nil
		})

	patches.ApplyFunc(plugin.GetBKECluster,
		func(string) (*bkev1beta1.BKECluster, error) {
			return &bkev1beta1.BKECluster{
				ObjectMeta: metav1.ObjectMeta{
					Name:      testClusterName,
					Namespace: testNamespace,
				},
			}, nil
		})

	patches.ApplyFunc(plugin.GetBkeConfigFromBkeCluster,
		func(*bkev1beta1.BKECluster) (*bkev1beta1.BKEConfig, error) {
			return nil, errors.New("failed to get BKEConfig")
		})

	_, err := kp.Execute(context.Background(), []string{Name})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to get BKEConfig")
}

func TestExecuteUpgradeControlPlaneWithBackupEtcd(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockExec := &mockExecutor{}

	kp := &KubeadmPlugin{
		exec:           mockExec,
		boot:           createTestBootScope(),
		GableNameSpace: testNamespace,
		clusterName:    testClusterName,
	}

	patches.ApplyMethod(kp, "Execute", func(_ *KubeadmPlugin, _ context.Context, _ []string) ([]string, error) {
		return nil, nil
	})

	result, err := kp.Execute(context.Background(), []string{Name})

	assert.NoError(t, err)
	assert.Nil(t, result)
}

func TestExecuteUpgradeControlPlaneWithClientSetError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockExec := &mockExecutor{}
	kp := &KubeadmPlugin{
		exec: mockExec,
		boot: &mfutil.BootScope{},
	}

	patches.ApplyFunc(plugin.ParseCommands,
		func(plugin.Plugin, []string) (map[string]string, error) {
			return map[string]string{
				"phase":       utils.UpgradeControlPlane,
				"backUpEtcd":  "true",
				"clusterType": "bke",
			}, nil
		})

	_, err := kp.Execute(context.Background(), []string{Name})

	assert.Error(t, err)
}

func TestExecuteUpgradeEtcdWithBackupEtcd(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockExec := &mockExecutor{}

	kp := &KubeadmPlugin{
		exec:           mockExec,
		boot:           createTestBootScope(),
		GableNameSpace: testNamespace,
		clusterName:    testClusterName,
	}

	patches.ApplyMethod(kp, "Execute", func(_ *KubeadmPlugin, _ context.Context, _ []string) ([]string, error) {
		return nil, nil
	})

	result, err := kp.Execute(context.Background(), []string{Name})

	assert.NoError(t, err)
	assert.Nil(t, result)
}

func TestExecuteUpgradeEtcdWithClientSetError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockExec := &mockExecutor{}
	kp := &KubeadmPlugin{
		exec: mockExec,
		boot: &mfutil.BootScope{},
	}

	patches.ApplyFunc(plugin.ParseCommands,
		func(plugin.Plugin, []string) (map[string]string, error) {
			return map[string]string{
				"phase":       utils.UpgradeEtcd,
				"backUpEtcd":  "false",
				"clusterType": "bke",
			}, nil
		})

	_, err := kp.Execute(context.Background(), []string{Name})

	assert.Error(t, err)
}

func TestGetBKEConfigWithEmptyControlPlaneEndpoint(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockExec := &mockExecutor{}
	kp := &KubeadmPlugin{
		exec:      mockExec,
		boot:      &mfutil.BootScope{},
		isManager: false,
	}

	testConfig := &bkev1beta1.BKEConfig{
		Cluster: bkev1beta1.Cluster{
			KubernetesVersion: testVersion,
		},
	}

	patches.ApplyFunc(plugin.GetBKECluster,
		func(string) (*bkev1beta1.BKECluster, error) {
			return &bkev1beta1.BKECluster{
				ObjectMeta: metav1.ObjectMeta{
					Name:      testClusterName,
					Namespace: testNamespace,
				},
				Spec: bkev1beta1.BKEClusterSpec{
					ClusterConfig: testConfig,
					ControlPlaneEndpoint: bkev1beta1.APIEndpoint{
						Host: "",
						Port: 0,
					},
				},
			}, nil
		})

	patches.ApplyFunc(plugin.GetBkeConfigFromBkeCluster,
		func(*bkev1beta1.BKECluster) (*bkev1beta1.BKEConfig, error) {
			return testConfig, nil
		})

	patches.ApplyFunc(plugin.GetClusterData,
		func(string) (*plugin.ClusterData, error) {
			return &plugin.ClusterData{
				Nodes: bkenode.Nodes{{IP: testHostIP, Role: []string{"master"}}},
			}, nil
		})

	patches.ApplyFunc(bkenetutil.GetAllInterfaceIP, func() ([]string, error) {
		return []string{testHostIP}, nil
	})

	patches.ApplyFunc(utils.HostName, func() string {
		return testHostName
	})

	err := kp.getBKEConfig(testNamespace + ":" + testClusterName)

	assert.NoError(t, err)
	assert.Equal(t, "", kp.controlPlaneEndpoint)
}

func TestExecuteUpgradeWorker(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockExec := &mockExecutor{}
	kp := &KubeadmPlugin{
		exec:      mockExec,
		boot:      createTestBootScope(),
		isManager: false,
	}

	patches.ApplyMethod(kp, "Execute", func(_ *KubeadmPlugin, _ context.Context, _ []string) ([]string, error) {
		return nil, nil
	})

	result, err := kp.Execute(context.Background(), []string{Name})

	assert.NoError(t, err)
	assert.Nil(t, result)
}

func TestGetBKEConfigCurrentNodeError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockExec := &mockExecutor{}
	kp := &KubeadmPlugin{
		exec:      mockExec,
		boot:      &mfutil.BootScope{},
		isManager: false,
	}

	testConfig := &bkev1beta1.BKEConfig{
		Cluster: bkev1beta1.Cluster{
			KubernetesVersion: testVersion,
		},
	}

	patches.ApplyFunc(plugin.GetBKECluster,
		func(string) (*bkev1beta1.BKECluster, error) {
			return &bkev1beta1.BKECluster{
				ObjectMeta: metav1.ObjectMeta{
					Name:      testClusterName,
					Namespace: testNamespace,
				},
				Spec: bkev1beta1.BKEClusterSpec{
					ClusterConfig: testConfig,
				},
			}, nil
		})

	patches.ApplyFunc(plugin.GetBkeConfigFromBkeCluster,
		func(*bkev1beta1.BKECluster) (*bkev1beta1.BKEConfig, error) {
			return testConfig, nil
		})

	patches.ApplyFunc(plugin.GetClusterData,
		func(string) (*plugin.ClusterData, error) {
			return nil, errors.New("current node error")
		})

	err := kp.getBKEConfig(testNamespace + ":" + testClusterName)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "current node error")
}

func createTestBootScope() *mfutil.BootScope {
	return &mfutil.BootScope{
		BkeConfig: &bkev1beta1.BKEConfig{
			Cluster: bkev1beta1.Cluster{
				KubernetesVersion: testVersion,
			},
		},
		ClusterName: testClusterName,
		HostName:    testHostName,
		HostIP:      testHostIP,
		CurrentNode: bkenode.Node{
			IP:   testHostIP,
			Role: []string{"master"},
		},
		Extra: map[string]interface{}{
			"Init":                 false,
			"gpuEnable":            "false",
			"KubernetesDir":        pkiutil.KubernetesDir,
			"mccs":                 []string{testNamespace, testClusterName},
			"upgradeWithOpenFuyao": false,
		},
	}
}

// ---------------------------------------------------------------------------
// 静态 Pod 配置自识别与更新（UpgradeControlPlaneConfig / UpgradeEtcdConfig）相关单测
// ---------------------------------------------------------------------------

const (
	testAPIServerHashOld     = "apiserver-config-hash-old"
	testAPIServerHashNew     = "apiserver-config-hash-new"
	testControllerMgrHashOld = "controller-manager-config-hash-old"
	testControllerMgrHashNew = "controller-manager-config-hash-new"
	testSchedulerHashOld     = "scheduler-config-hash-old"
	testSchedulerHashNew     = "scheduler-config-hash-new"

	testInvalidHashJSON = "{invalid-json"
)

// marshalComponentConfigHash 将控制面 hash 结构体序列化为命令参数所需的 JSON 字符串
func marshalComponentConfigHash(t *testing.T, hash bkev1beta1.ComponentConfigHash) string {
	t.Helper()
	data, err := json.Marshal(hash)
	assert.NoError(t, err)
	return string(data)
}

// newConfigUpdateTestPlugin 构造用于配置更新单测的 KubeadmPlugin
func newConfigUpdateTestPlugin() *KubeadmPlugin {
	return &KubeadmPlugin{
		exec:           &mockExecutor{},
		boot:           createTestBootScope(),
		GableNameSpace: testNamespace,
		clusterName:    testClusterName,
	}
}

// patchClientSetFromSecret 打桩获取目标集群 clientset 的函数，返回结果由入参决定
func patchClientSetFromSecret(patches *gomonkey.Patches, clientset *kubernetes.Clientset, err error) {
	patches.ApplyFunc(clientutil.ClientSetFromManagerClusterSecret,
		func(_ ...string) (*kubernetes.Clientset, error) {
			return clientset, err
		})
}

// patchParseCommandsWithPhase 打桩命令解析，返回指定 phase 及携带的参数
func patchParseCommandsWithPhase(patches *gomonkey.Patches, phase string, extraArgs map[string]string) {
	commands := map[string]string{"phase": phase}
	for key, value := range extraArgs {
		commands[key] = value
	}
	patches.ApplyFunc(plugin.ParseCommands,
		func(plugin.Plugin, []string) (map[string]string, error) {
			return commands, nil
		})
}

func TestParamForStaticPodConfigUpdate(t *testing.T) {
	kp := &KubeadmPlugin{}
	params := kp.Param()

	assert.Contains(t, params, "controlPlaneConfigOldHash")
	assert.Contains(t, params, "controlPlaneConfigNewHash")
	assert.False(t, params["controlPlaneConfigOldHash"].Required)
	assert.False(t, params["controlPlaneConfigNewHash"].Required)
	assert.Equal(t, "", params["controlPlaneConfigOldHash"].Default)
	assert.Equal(t, "", params["controlPlaneConfigNewHash"].Default)
	assert.Contains(t, params["controlPlaneConfigOldHash"].Description, utils.UpgradeControlPlaneConfig)
	assert.Contains(t, params["controlPlaneConfigNewHash"].Description, utils.UpgradeControlPlaneConfig)

	assert.Contains(t, params["phase"].Value, utils.UpgradeControlPlaneConfig)
	assert.Contains(t, params["phase"].Value, utils.UpgradeEtcdConfig)
}

func TestInitBKEConfigWithoutBkeConfig(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	kp := newConfigUpdateTestPlugin()
	called := false
	patches.ApplyPrivateMethod(kp, "getBKEConfig", func(_ *KubeadmPlugin, _ string) error {
		called = true
		return nil
	})

	assert.NoError(t, kp.initBKEConfig(map[string]string{"phase": utils.UpgradeControlPlaneConfig}))
	assert.False(t, called)
}

func TestInitBKEConfigWithBkeConfig(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	kp := newConfigUpdateTestPlugin()
	var gotBkeConfig string
	patches.ApplyPrivateMethod(kp, "getBKEConfig", func(_ *KubeadmPlugin, bkeConfigNS string) error {
		gotBkeConfig = bkeConfigNS
		return nil
	})

	parseCommands := map[string]string{
		"phase":       utils.UpgradeEtcdConfig,
		"bkeConfig":   testNamespace + ":" + testClusterName,
		"etcdVersion": testEtcdVersion,
	}

	assert.NoError(t, kp.initBKEConfig(parseCommands))
	assert.Equal(t, testNamespace+":"+testClusterName, gotBkeConfig)
	assert.Equal(t, testEtcdVersion, kp.boot.BkeConfig.Cluster.EtcdVersion)
	assert.Equal(t, testEtcdVersion, kp.boot.Extra["etcdVersion"])
}

func TestInitBKEConfigWithGetBKEConfigError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	kp := newConfigUpdateTestPlugin()
	patches.ApplyPrivateMethod(kp, "getBKEConfig", func(_ *KubeadmPlugin, _ string) error {
		return errors.New("failed to get BKEConfig")
	})

	parseCommands := map[string]string{
		"bkeConfig":   testNamespace + ":" + testClusterName,
		"etcdVersion": testEtcdVersion,
	}

	err := kp.initBKEConfig(parseCommands)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to get BKEConfig")
	assert.Empty(t, kp.boot.BkeConfig.Cluster.EtcdVersion)
}

func TestInitLocalK8sClient(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	kp := newConfigUpdateTestPlugin()
	patchClientSetFromSecret(patches, &kubernetes.Clientset{}, nil)

	assert.NoError(t, kp.initLocalK8sClient())
	assert.NotNil(t, kp.localK8sClient)
}

func TestInitLocalK8sClientError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	kp := newConfigUpdateTestPlugin()
	patchClientSetFromSecret(patches, nil, errors.New("clientset error"))

	err := kp.initLocalK8sClient()

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "clientset error")
	assert.Nil(t, kp.localK8sClient)
}

func TestExecuteUpgradeControlPlaneConfig(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	kp := newConfigUpdateTestPlugin()

	oldHash := bkev1beta1.ComponentConfigHash{
		APIServer:         testAPIServerHashOld,
		ControllerManager: testControllerMgrHashOld,
		Scheduler:         testSchedulerHashOld,
	}
	newHash := bkev1beta1.ComponentConfigHash{
		APIServer:         testAPIServerHashNew,
		ControllerManager: testControllerMgrHashOld,
		Scheduler:         testSchedulerHashOld,
	}
	patchParseCommandsWithPhase(patches, utils.UpgradeControlPlaneConfig, map[string]string{
		"controlPlaneConfigOldHash": marshalComponentConfigHash(t, oldHash),
		"controlPlaneConfigNewHash": marshalComponentConfigHash(t, newHash),
	})
	patchClientSetFromSecret(patches, &kubernetes.Clientset{}, nil)

	var gotOldHash, gotNewHash bkev1beta1.ComponentConfigHash
	patches.ApplyPrivateMethod(kp, "updateControlPlaneConfig",
		func(_ *KubeadmPlugin, oldConfigHash, newConfigHash bkev1beta1.ComponentConfigHash) error {
			gotOldHash, gotNewHash = oldConfigHash, newConfigHash
			return nil
		})

	result, err := kp.Execute(context.Background(), []string{Name})

	assert.NoError(t, err)
	assert.Nil(t, result)
	assert.Equal(t, oldHash, gotOldHash)
	assert.Equal(t, newHash, gotNewHash)
}

func TestExecuteUpgradeControlPlaneConfigWithClientSetError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	kp := newConfigUpdateTestPlugin()
	patchParseCommandsWithPhase(patches, utils.UpgradeControlPlaneConfig, nil)
	patchClientSetFromSecret(patches, nil, errors.New("clientset error"))

	called := false
	patches.ApplyPrivateMethod(kp, "updateControlPlaneConfig",
		func(_ *KubeadmPlugin, oldConfigHash, newConfigHash bkev1beta1.ComponentConfigHash) error {
			called = true
			return nil
		})

	_, err := kp.Execute(context.Background(), []string{Name})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "clientset error")
	assert.False(t, called)
}

func TestExecuteUpgradeControlPlaneConfigWithInvalidOldHash(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	kp := newConfigUpdateTestPlugin()
	patchParseCommandsWithPhase(patches, utils.UpgradeControlPlaneConfig, map[string]string{
		"controlPlaneConfigOldHash": testInvalidHashJSON,
		"controlPlaneConfigNewHash": marshalComponentConfigHash(t, bkev1beta1.ComponentConfigHash{
			APIServer: testAPIServerHashNew,
		}),
	})
	patchClientSetFromSecret(patches, &kubernetes.Clientset{}, nil)

	called := false
	patches.ApplyPrivateMethod(kp, "updateControlPlaneConfig",
		func(_ *KubeadmPlugin, oldConfigHash, newConfigHash bkev1beta1.ComponentConfigHash) error {
			called = true
			return nil
		})

	_, err := kp.Execute(context.Background(), []string{Name})

	assert.Error(t, err)
	assert.False(t, called)
}

func TestExecuteUpgradeControlPlaneConfigWithInvalidNewHash(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	kp := newConfigUpdateTestPlugin()
	patchParseCommandsWithPhase(patches, utils.UpgradeControlPlaneConfig, map[string]string{
		"controlPlaneConfigOldHash": marshalComponentConfigHash(t, bkev1beta1.ComponentConfigHash{
			APIServer: testAPIServerHashOld,
		}),
		"controlPlaneConfigNewHash": testInvalidHashJSON,
	})
	patchClientSetFromSecret(patches, &kubernetes.Clientset{}, nil)

	called := false
	patches.ApplyPrivateMethod(kp, "updateControlPlaneConfig",
		func(_ *KubeadmPlugin, oldConfigHash, newConfigHash bkev1beta1.ComponentConfigHash) error {
			called = true
			return nil
		})

	_, err := kp.Execute(context.Background(), []string{Name})

	assert.Error(t, err)
	assert.False(t, called)
}

func TestExecuteUpgradeControlPlaneConfigUpdateFailed(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	kp := newConfigUpdateTestPlugin()
	patchParseCommandsWithPhase(patches, utils.UpgradeControlPlaneConfig, nil)
	patchClientSetFromSecret(patches, &kubernetes.Clientset{}, nil)
	patches.ApplyPrivateMethod(kp, "updateControlPlaneConfig",
		func(_ *KubeadmPlugin, oldConfigHash, newConfigHash bkev1beta1.ComponentConfigHash) error {
			return errors.New("update control plane config failed")
		})

	_, err := kp.Execute(context.Background(), []string{Name})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "update control plane config failed")
}

func TestExecuteUpgradeEtcdConfig(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	kp := newConfigUpdateTestPlugin()
	patchParseCommandsWithPhase(patches, utils.UpgradeEtcdConfig, nil)
	patchClientSetFromSecret(patches, &kubernetes.Clientset{}, nil)

	called := false
	patches.ApplyPrivateMethod(kp, "updateEtcdConfig", func(_ *KubeadmPlugin) error {
		called = true
		return nil
	})

	result, err := kp.Execute(context.Background(), []string{Name})

	assert.NoError(t, err)
	assert.Nil(t, result)
	assert.True(t, called)
}

func TestExecuteUpgradeEtcdConfigWithClientSetError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	kp := newConfigUpdateTestPlugin()
	patchParseCommandsWithPhase(patches, utils.UpgradeEtcdConfig, nil)
	patchClientSetFromSecret(patches, nil, errors.New("clientset error"))

	called := false
	patches.ApplyPrivateMethod(kp, "updateEtcdConfig", func(_ *KubeadmPlugin) error {
		called = true
		return nil
	})

	_, err := kp.Execute(context.Background(), []string{Name})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "clientset error")
	assert.False(t, called)
}

func TestExecuteUpgradeEtcdConfigUpdateFailed(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	kp := newConfigUpdateTestPlugin()
	patchParseCommandsWithPhase(patches, utils.UpgradeEtcdConfig, nil)
	patchClientSetFromSecret(patches, &kubernetes.Clientset{}, nil)
	patches.ApplyPrivateMethod(kp, "updateEtcdConfig", func(_ *KubeadmPlugin) error {
		return errors.New("update etcd config failed")
	})

	_, err := kp.Execute(context.Background(), []string{Name})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "update etcd config failed")
}

// patchUpdateControlPlaneDependencies 打桩 updateControlPlaneConfig 的依赖（manifest + Pod Running 等待）。
// 配置更新不再读取 pod hash；版本升级仍走 waitComponentReady。
func patchUpdateControlPlaneDependencies(
	patches *gomonkey.Patches,
	kp *KubeadmPlugin,
	manifestErr error,
	waitErr error,
) (*[]string, *[]string) {
	regenerated := &[]string{}
	waited := &[]string{}

	patches.ApplyPrivateMethod(kp, "upgradeControlPlaneManifestCommand",
		func(_ *KubeadmPlugin, scopes ...string) error {
			if manifestErr != nil {
				return manifestErr
			}
			*regenerated = append(*regenerated, scopes...)
			return nil
		})
	patches.ApplyPrivateMethod(kp, "waitComponentPodRunning",
		func(_ *KubeadmPlugin, component, baselineUID string) error {
			if waitErr != nil {
				return waitErr
			}
			*waited = append(*waited, component)
			return nil
		})

	return regenerated, waited
}

func TestUpdateControlPlaneConfigOnlyChangedComponent(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	kp := newConfigUpdateTestPlugin()
	regenerated, waited := patchUpdateControlPlaneDependencies(patches, kp, nil, nil)

	oldHash := bkev1beta1.ComponentConfigHash{
		APIServer:         testAPIServerHashOld,
		ControllerManager: testControllerMgrHashOld,
		Scheduler:         testSchedulerHashOld,
	}
	newHash := bkev1beta1.ComponentConfigHash{
		APIServer:         testAPIServerHashNew,
		ControllerManager: testControllerMgrHashOld,
		Scheduler:         testSchedulerHashOld,
	}

	assert.NoError(t, kp.updateControlPlaneConfig(oldHash, newHash))
	assert.Equal(t, []string{mfutil.KubeAPIServer}, *regenerated)
	assert.Equal(t, []string{mfutil.KubeAPIServer}, *waited)
}

func TestUpdateControlPlaneConfigAllComponentsChanged(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	kp := newConfigUpdateTestPlugin()
	regenerated, waited := patchUpdateControlPlaneDependencies(patches, kp, nil, nil)

	oldHash := bkev1beta1.ComponentConfigHash{
		APIServer:         testAPIServerHashOld,
		ControllerManager: testControllerMgrHashOld,
		Scheduler:         testSchedulerHashOld,
	}
	newHash := bkev1beta1.ComponentConfigHash{
		APIServer:         testAPIServerHashNew,
		ControllerManager: testControllerMgrHashNew,
		Scheduler:         testSchedulerHashNew,
	}

	assert.NoError(t, kp.updateControlPlaneConfig(oldHash, newHash))
	assert.Equal(t, mfutil.GetControlPlaneComponents(), *regenerated)
	assert.Equal(t, mfutil.GetControlPlaneComponents(), *waited)
}

func TestUpdateControlPlaneConfigNoComponentChanged(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	kp := newConfigUpdateTestPlugin()
	regenerated, waited := patchUpdateControlPlaneDependencies(patches, kp, nil, nil)

	sameHash := bkev1beta1.ComponentConfigHash{
		APIServer:         testAPIServerHashOld,
		ControllerManager: testControllerMgrHashOld,
		Scheduler:         testSchedulerHashOld,
	}

	assert.NoError(t, kp.updateControlPlaneConfig(sameHash, sameHash))
	assert.Empty(t, *regenerated)
	assert.Empty(t, *waited)
}

func TestUpdateControlPlaneConfigWithManifestError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	kp := newConfigUpdateTestPlugin()
	regenerated, waited := patchUpdateControlPlaneDependencies(
		patches, kp, errors.New("manifest error"), nil)

	oldHash := bkev1beta1.ComponentConfigHash{APIServer: testAPIServerHashOld}
	newHash := bkev1beta1.ComponentConfigHash{APIServer: testAPIServerHashNew}

	err := kp.updateControlPlaneConfig(oldHash, newHash)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "manifest error")
	assert.Empty(t, *regenerated)
	assert.Empty(t, *waited)
}

func TestUpdateControlPlaneConfigWithWaitReadyError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	kp := newConfigUpdateTestPlugin()
	regenerated, waited := patchUpdateControlPlaneDependencies(
		patches, kp, nil, errors.New("wait component ready error"))

	oldHash := bkev1beta1.ComponentConfigHash{APIServer: testAPIServerHashOld}
	newHash := bkev1beta1.ComponentConfigHash{APIServer: testAPIServerHashNew}

	err := kp.updateControlPlaneConfig(oldHash, newHash)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "wait component ready error")
	assert.Equal(t, []string{mfutil.KubeAPIServer}, *regenerated)
	assert.Empty(t, *waited)
}

func TestUpdateEtcdConfig(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	kp := newConfigUpdateTestPlugin()

	regenerated := []string{}
	patches.ApplyPrivateMethod(kp, "upgradeControlPlaneManifestCommand",
		func(_ *KubeadmPlugin, scopes ...string) error {
			regenerated = append(regenerated, scopes...)
			return nil
		})
	waited := []string{}
	patches.ApplyPrivateMethod(kp, "waitComponentPodRunning",
		func(_ *KubeadmPlugin, component, baselineUID string) error {
			waited = append(waited, component)
			return nil
		})

	assert.NoError(t, kp.updateEtcdConfig())
	assert.Equal(t, []string{mfutil.Etcd}, regenerated)
	assert.Equal(t, []string{mfutil.Etcd}, waited)
}

func TestUpdateEtcdConfigWithManifestError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	kp := newConfigUpdateTestPlugin()
	patches.ApplyPrivateMethod(kp, "upgradeControlPlaneManifestCommand",
		func(_ *KubeadmPlugin, scopes ...string) error {
			return errors.New("manifest error")
		})
	waitCalled := false
	patches.ApplyPrivateMethod(kp, "waitComponentPodRunning",
		func(_ *KubeadmPlugin, component, baselineUID string) error {
			waitCalled = true
			return nil
		})

	err := kp.updateEtcdConfig()

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "manifest error")
	assert.False(t, waitCalled)
}

func TestUpdateEtcdConfigWithWaitReadyError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	kp := newConfigUpdateTestPlugin()
	patches.ApplyPrivateMethod(kp, "upgradeControlPlaneManifestCommand",
		func(_ *KubeadmPlugin, scopes ...string) error {
			return nil
		})
	patches.ApplyPrivateMethod(kp, "waitComponentPodRunning",
		func(_ *KubeadmPlugin, component, baselineUID string) error {
			return errors.New("wait etcd ready error")
		})

	err := kp.updateEtcdConfig()

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "wait etcd ready error")
}
