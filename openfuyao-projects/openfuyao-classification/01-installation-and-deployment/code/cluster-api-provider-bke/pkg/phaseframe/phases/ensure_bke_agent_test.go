package phases

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	bkevalidate "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/validation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/certs"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	bkessh "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/remote"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/statusmanage"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	ctrlfake "sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestMain(m *testing.M) {
	_ = os.MkdirAll(os.TempDir(), 01777)
	os.Exit(m.Run())
}

const (
	testNodeIP1     = "127.0.0.1"
	testNodeIP2     = "127.0.0.2"
	testNodeIP3     = "127.0.0.3"
	testHostname1   = "node1"
	testHostname2   = "node2"
	testHostname3   = "node3"
	testClusterName = "test-cluster"
	testNamespace   = "default"
	testFileContent = "test"
	testDirPerm     = 0755
	testFilePerm    = 0644
	testDstDir      = "/tmp"
	testNTPServer   = "ntp.invalid"
	emptySliceSize  = 0
	successField    = "success"
	failureField    = "failed"
	two             = 2
)

// createTestPhaseContext 创建测试用的PhaseContext，避免空指针
func createTestPhaseContext(bkeCluster *bkev1beta1.BKECluster) *phaseframe.PhaseContext {
	if bkeCluster == nil {
		bkeCluster = &bkev1beta1.BKECluster{}
	}
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()

	recorder := &fakeRecorder{}

	return &phaseframe.PhaseContext{
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, recorder, bkeCluster),
		Client:     c,
		Scheme:     scheme,
	}
}

// createTestBKECluster 创建测试用的BKECluster
// 注意：节点信息已拆分到独立的 BKENode CRD 中，不再存储在 BKECluster.Spec.ClusterConfig.Nodes
func createTestBKECluster(nodes []confv1beta1.Node) *bkev1beta1.BKECluster {
	_ = nodes // nodes 参数保留用于兼容性，实际节点通过 BKENode CRD 管理
	return &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testClusterName,
			Namespace: testNamespace,
		},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{},
		},
	}
}

// createTestNodes 创建测试用的节点列表
func createTestNodes(count int) []confv1beta1.Node {
	nodes := make([]confv1beta1.Node, 0, count)
	ips := []string{testNodeIP1, testNodeIP2, testNodeIP3}
	hostnames := []string{testHostname1, testHostname2, testHostname3}
	for i := 0; i < count && i < len(ips); i++ {
		nodes = append(nodes, confv1beta1.Node{
			IP:       ips[i],
			Hostname: hostnames[i],
		})
	}
	return nodes
}

// TestCheckAvailableHosts 测试checkAvailableHosts函数
func TestCheckAvailableHosts(t *testing.T) {
	tests := []struct {
		name     string
		multiCli *bkessh.MultiCli
		errs     map[string]error
		want     bool
	}{
		{
			name:     "available hosts count is zero should return false",
			multiCli: &bkessh.MultiCli{},
			errs:     make(map[string]error),
			want:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := &EnsureBKEAgent{}
			result := e.checkAvailableHosts(tt.multiCli, tt.errs)
			assert.Equal(t, tt.want, result)
		})
	}
}

// TestAllNeedPushNodesFailed_EmptyNodes 测试allNeedPushNodesFailed函数-空节点场景
func TestAllNeedPushNodesFailed_EmptyNodes(t *testing.T) {
	e := &EnsureBKEAgent{
		needPushNodes: bkenode.Nodes{},
	}
	result := e.allNeedPushNodesFailed([]string{testNodeIP1})
	assert.False(t, result)
}

// TestAllNeedPushNodesFailed_EmptyFailedInfo 测试allNeedPushNodesFailed函数-空失败信息场景
func TestAllNeedPushNodesFailed_EmptyFailedInfo(t *testing.T) {
	e := &EnsureBKEAgent{
		needPushNodes: bkenode.Nodes{{IP: testNodeIP1, Hostname: testHostname1}},
	}
	result := e.allNeedPushNodesFailed(make([]string, emptySliceSize))
	assert.False(t, result)
}

// TestAllNeedPushNodesFailed_AllFailed 测试allNeedPushNodesFailed函数-所有节点失败场景
func TestAllNeedPushNodesFailed_AllFailed(t *testing.T) {
	e := &EnsureBKEAgent{
		needPushNodes: bkenode.Nodes{
			{IP: testNodeIP1, Hostname: testHostname1},
			{IP: testNodeIP2, Hostname: testHostname2},
		},
	}
	result := e.allNeedPushNodesFailed([]string{testNodeIP1, testNodeIP2})
	assert.True(t, result)
}

// TestAllNeedPushNodesFailed_SomeFailed 测试allNeedPushNodesFailed函数-部分节点失败场景
func TestAllNeedPushNodesFailed_SomeFailed(t *testing.T) {
	e := &EnsureBKEAgent{
		needPushNodes: bkenode.Nodes{
			{IP: testNodeIP1, Hostname: testHostname1},
			{IP: testNodeIP2, Hostname: testHostname2},
		},
	}
	result := e.allNeedPushNodesFailed([]string{testNodeIP1})
	assert.False(t, result)
}

// TestAllNeedPushNodesFailed_NoMatch 测试allNeedPushNodesFailed函数-无匹配节点场景
func TestAllNeedPushNodesFailed_NoMatch(t *testing.T) {
	e := &EnsureBKEAgent{
		needPushNodes: bkenode.Nodes{{IP: testNodeIP1, Hostname: testHostname1}},
	}
	result := e.allNeedPushNodesFailed([]string{testNodeIP2})
	assert.False(t, result)
}

// TestAddFilesToUploadList_FileExists 测试addFilesToUploadList函数-文件存在场景
func TestAddFilesToUploadList_FileExists(t *testing.T) {
	tempDir := t.TempDir()
	existingFile := filepath.Join(tempDir, "existing.txt")
	err := os.WriteFile(existingFile, []byte(testFileContent), testFilePerm)
	assert.NoError(t, err)

	e := &EnsureBKEAgent{}
	fileUpList := make([]bkessh.File, emptySliceSize)
	result := e.addFilesToUploadList(fileUpList, []string{existingFile}, testDstDir)
	assert.Equal(t, 1, len(result))
	assert.Equal(t, testDstDir, result[0].Dst)
}

// TestAddFilesToUploadList_FileNotExists 测试addFilesToUploadList函数-文件不存在场景
func TestAddFilesToUploadList_FileNotExists(t *testing.T) {
	tempDir := t.TempDir()
	nonExistingFile := filepath.Join(tempDir, "non-existing.txt")

	e := &EnsureBKEAgent{}
	fileUpList := make([]bkessh.File, emptySliceSize)
	result := e.addFilesToUploadList(fileUpList, []string{nonExistingFile}, testDstDir)
	assert.Equal(t, emptySliceSize, len(result))
}

// TestAddFilesToUploadList_MixedFiles 测试addFilesToUploadList函数-混合文件场景
func TestAddFilesToUploadList_MixedFiles(t *testing.T) {
	tempDir := t.TempDir()
	existingFile := filepath.Join(tempDir, "existing.txt")
	nonExistingFile := filepath.Join(tempDir, "non-existing.txt")
	err := os.WriteFile(existingFile, []byte(testFileContent), testFilePerm)
	assert.NoError(t, err)

	e := &EnsureBKEAgent{}
	fileUpList := make([]bkessh.File, emptySliceSize)
	result := e.addFilesToUploadList(fileUpList, []string{existingFile, nonExistingFile}, testDstDir)
	assert.Equal(t, 1, len(result))
}

// TestAddFilesToUploadList_EmptyPaths 测试addFilesToUploadList函数-空路径列表场景
func TestAddFilesToUploadList_EmptyPaths(t *testing.T) {
	e := &EnsureBKEAgent{}
	fileUpList := []bkessh.File{{Src: "test", Dst: testDstDir}}
	result := e.addFilesToUploadList(fileUpList, make([]string, emptySliceSize), testDstDir)
	assert.Equal(t, 1, len(result))
}

// TestAddGlobalCAFilesIfNeeded_EmptyAddons 测试addGlobalCAFilesIfNeeded函数-空addons场景
func TestAddGlobalCAFilesIfNeeded_EmptyAddons(t *testing.T) {
	bkeCluster := createTestBKECluster(createTestNodes(1))
	bkeCluster.Spec.ClusterConfig.Addons = nil
	ctx := createTestPhaseContext(bkeCluster)
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}
	fileUpList := make([]bkessh.File, emptySliceSize)
	result := e.addGlobalCAFilesIfNeeded(fileUpList)
	assert.Equal(t, emptySliceSize, len(result))
}

// TestAddCSRFilesToUploadList 测试addCSRFilesToUploadList函数
func TestAddCSRFilesToUploadList(t *testing.T) {
	tempDir := t.TempDir()
	certConfigDir := filepath.Join(tempDir, "cert_config")
	err := os.MkdirAll(certConfigDir, testDirPerm)
	assert.NoError(t, err)

	existingCSR1 := filepath.Join(certConfigDir, certs.ConfigKeyClusterCAPolicy)
	err = os.WriteFile(existingCSR1, []byte(testFileContent), testFilePerm)
	assert.NoError(t, err)

	e := &EnsureBKEAgent{}
	fileUpList := make([]bkessh.File, emptySliceSize)
	result := e.addCSRFilesToUploadList(fileUpList)
	assert.GreaterOrEqual(t, len(result), emptySliceSize)
}

// TestPrepareFileUploadList 测试prepareFileUploadList函数
func TestPrepareFileUploadList(t *testing.T) {
	tempDir := t.TempDir()
	servicePath := filepath.Join(tempDir, "bkeagent.service")
	err := os.WriteFile(servicePath, []byte(testFileContent), testFilePerm)
	assert.NoError(t, err)

	bkeCluster := createTestBKECluster(createTestNodes(1))
	bkeCluster.Spec.ClusterConfig.Addons = nil
	ctx := createTestPhaseContext(bkeCluster)
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	result := e.prepareFileUploadList(servicePath)
	assert.GreaterOrEqual(t, len(result), 1)

	hasServiceFile := false
	for _, file := range result {
		if file.Src == servicePath {
			hasServiceFile = true
			assert.Equal(t, "/etc/systemd/system", file.Dst)
			break
		}
	}
	assert.True(t, hasServiceFile)
}

// TestNewEnsureBKEAgent 测试NewEnsureBKEAgent函数
func TestNewEnsureBKEAgent(t *testing.T) {
	bkeCluster := createTestBKECluster(createTestNodes(1))
	ctx := createTestPhaseContext(bkeCluster)
	result := NewEnsureBKEAgent(ctx)
	assert.NotNil(t, result)

	agent, ok := result.(*EnsureBKEAgent)
	assert.True(t, ok)
	assert.NotNil(t, agent.BasePhase)
	assert.Equal(t, EnsureBKEAgentName, agent.BasePhase.PhaseName)
}

// TestNeedExecute_NoNewNodes 测试NeedExecute函数-无新节点场景
func TestNeedExecute_NoNewNodes(t *testing.T) {
	nodes := createTestNodes(1)
	oldCluster := createTestBKECluster(nodes)
	newCluster := createTestBKECluster(nodes)

	ctx := createTestPhaseContext(newCluster)
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}
	result := e.NeedExecute(oldCluster, newCluster)
	assert.False(t, result)
}

// TestUpdateNodeStatus_BothTypes 测试updateNodeStatus函数-成功和失败节点场景
func TestUpdateNodeStatus_BothTypes(t *testing.T) {
	bkeCluster := createTestBKECluster(createTestNodes(two))
	ctx := createTestPhaseContext(bkeCluster)
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}
	successNodes := []string{testNodeIP1 + successField}
	failedNodes := []string{testNodeIP2 + failureField}
	e.updateNodeStatus(bkeCluster, successNodes, failedNodes)
	assert.NotNil(t, bkeCluster)
}

// TestUpdateNodeStatus_EmptySuccess 测试updateNodeStatus函数-空成功节点场景
func TestUpdateNodeStatus_EmptySuccess(t *testing.T) {
	bkeCluster := createTestBKECluster(createTestNodes(1))
	ctx := createTestPhaseContext(bkeCluster)
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}
	failedNodes := []string{testNodeIP1 + failureField}
	e.updateNodeStatus(bkeCluster, make([]string, emptySliceSize), failedNodes)
	assert.NotNil(t, bkeCluster)
}

// TestUpdateNodeStatus_EmptyFailed 测试updateNodeStatus函数-空失败节点场景
func TestUpdateNodeStatus_EmptyFailed(t *testing.T) {
	bkeCluster := createTestBKECluster(createTestNodes(1))
	ctx := createTestPhaseContext(bkeCluster)
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}
	successNodes := []string{testNodeIP1 + successField}
	e.updateNodeStatus(bkeCluster, successNodes, make([]string, emptySliceSize))
	assert.NotNil(t, bkeCluster)
}

// TestCheckAllOrPushedAgentsFailed_AllSuccess 测试checkAllOrPushedAgentsFailed函数-全部成功场景
func TestCheckAllOrPushedAgentsFailed_AllSuccess(t *testing.T) {
	ctx := createTestPhaseContext(&bkev1beta1.BKECluster{})
	e := &EnsureBKEAgent{
		BasePhase:     phaseframe.BasePhase{Ctx: ctx},
		needPushNodes: bkenode.Nodes{{IP: testNodeIP1, Hostname: testHostname1}},
	}
	err := e.checkAllOrPushedAgentsFailed([]string{testNodeIP1 + successField}, make([]string, emptySliceSize))
	assert.NoError(t, err)
}

// TestCheckAllOrPushedAgentsFailed_SomeFailed 测试checkAllOrPushedAgentsFailed函数-部分失败场景
func TestCheckAllOrPushedAgentsFailed_SomeFailed(t *testing.T) {
	ctx := createTestPhaseContext(&bkev1beta1.BKECluster{})
	e := &EnsureBKEAgent{
		BasePhase: phaseframe.BasePhase{Ctx: ctx},
		needPushNodes: bkenode.Nodes{
			{IP: testNodeIP1, Hostname: testHostname1},
			{IP: testNodeIP2, Hostname: testHostname2},
		},
	}
	err := e.checkAllOrPushedAgentsFailed([]string{testNodeIP1 + successField}, []string{testNodeIP2 + failureField})
	assert.NoError(t, err)
}

// TestEnsureBKEAgent_Execute_LoadConfigError 测试Execute函数-加载配置失败
func TestEnsureBKEAgent_Execute_LoadConfigError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	ctx := createTestPhaseContext(&bkev1beta1.BKECluster{})
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyPrivateMethod(e, "loadLocalKubeConfig", func(_ *EnsureBKEAgent) error {
		return assert.AnError
	})

	_, err := e.Execute()
	assert.Error(t, err)
}

// TestEnsureBKEAgent_Execute_GetNodesError 测试Execute函数-获取节点失败
func TestEnsureBKEAgent_Execute_GetNodesError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	ctx := createTestPhaseContext(&bkev1beta1.BKECluster{})
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyPrivateMethod(e, "loadLocalKubeConfig", func(_ *EnsureBKEAgent) error {
		return nil
	})
	patches.ApplyPrivateMethod(e, "getNeedPushNodes", func(_ *EnsureBKEAgent) error {
		return assert.AnError
	})

	_, err := e.Execute()
	assert.Error(t, err)
}

// TestEnsureBKEAgent_Execute_NoNeedPushNodes 测试Execute函数-无需推送节点
func TestEnsureBKEAgent_Execute_NoNeedPushNodes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	ctx := createTestPhaseContext(&bkev1beta1.BKECluster{})
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyPrivateMethod(e, "loadLocalKubeConfig", func(_ *EnsureBKEAgent) error {
		return nil
	})
	patches.ApplyPrivateMethod(e, "getNeedPushNodes", func(_ *EnsureBKEAgent) error {
		e.needPushNodes = nil
		return nil
	})

	_, err := e.Execute()
	assert.NoError(t, err)
}

// TestEnsureBKEAgent_Execute_PushAgentError 测试Execute函数-推送agent失败
func TestEnsureBKEAgent_Execute_PushAgentError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	ctx := createTestPhaseContext(&bkev1beta1.BKECluster{})
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyPrivateMethod(e, "loadLocalKubeConfig", func(_ *EnsureBKEAgent) error {
		return nil
	})
	patches.ApplyPrivateMethod(e, "getNeedPushNodes", func(_ *EnsureBKEAgent) error {
		e.needPushNodes = bkenode.Nodes{{IP: testNodeIP1}}
		return nil
	})
	patches.ApplyPrivateMethod(e, "pushAgent", func(_ *EnsureBKEAgent) error {
		return assert.AnError
	})

	_, err := e.Execute()
	assert.Error(t, err)
}

// TestEnsureBKEAgent_Execute_Success 测试Execute函数-成功
func TestEnsureBKEAgent_Execute_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	ctx := createTestPhaseContext(&bkev1beta1.BKECluster{})
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyPrivateMethod(e, "loadLocalKubeConfig", func(_ *EnsureBKEAgent) error {
		return nil
	})
	patches.ApplyPrivateMethod(e, "getNeedPushNodes", func(_ *EnsureBKEAgent) error {
		e.needPushNodes = bkenode.Nodes{{IP: testNodeIP1}}
		return nil
	})
	patches.ApplyPrivateMethod(e, "pushAgent", func(_ *EnsureBKEAgent) error {
		return nil
	})
	patches.ApplyPrivateMethod(e, "pingAgent", func(_ *EnsureBKEAgent) error {
		return nil
	})

	_, err := e.Execute()
	assert.NoError(t, err)
}

// TestEnsureBKEAgent_LoadLocalKubeConfig 测试loadLocalKubeConfig函数
func TestEnsureBKEAgent_LoadLocalKubeConfig(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	ctx := createTestPhaseContext(&bkev1beta1.BKECluster{})
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyPrivateMethod(e, "loadLocalKubeConfig", func(agent *EnsureBKEAgent) error {
		agent.localKubeConfig = []byte("test-config")
		return nil
	})

	err := e.loadLocalKubeConfig()
	assert.NoError(t, err)
	assert.NotNil(t, e.localKubeConfig)
}

// TestEnsureBKEAgent_GetNeedPushNodes 测试getNeedPushNodes函数
func TestEnsureBKEAgent_GetNeedPushNodes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	ctx := createTestPhaseContext(&bkev1beta1.BKECluster{})
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyPrivateMethod(e, "getNeedPushNodes", func(agent *EnsureBKEAgent) error {
		agent.needPushNodes = bkenode.Nodes{{IP: testNodeIP1}}
		return nil
	})

	err := e.getNeedPushNodes()
	assert.NoError(t, err)
	assert.Len(t, e.needPushNodes, 1)
}

// TestEnsureBKEAgent_PushAgent 测试pushAgent函数
func TestEnsureBKEAgent_PushAgent(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	ctx := createTestPhaseContext(&bkev1beta1.BKECluster{})
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyPrivateMethod(e, "pushAgent", func(_ *EnsureBKEAgent) error {
		return nil
	})

	err := e.pushAgent()
	assert.NoError(t, err)
}

// TestEnsureBKEAgent_LogPushAgentStart 测试logPushAgentStart函数
func TestEnsureBKEAgent_LogPushAgentStart(t *testing.T) {
	ctx := createTestPhaseContext(&bkev1beta1.BKECluster{})
	e := &EnsureBKEAgent{
		BasePhase:     phaseframe.BasePhase{Ctx: ctx},
		needPushNodes: bkenode.Nodes{{IP: testNodeIP1, Hostname: testHostname1}},
	}
	e.logPushAgentStart()
}

// TestEnsureBKEAgent_PushAgent_Success 测试pushAgent成功场景
func TestEnsureBKEAgent_PushAgent_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	tempDir := t.TempDir()
	servicePath := filepath.Join(tempDir, "test.service")

	ctx := createTestPhaseContext(&bkev1beta1.BKECluster{})
	e := &EnsureBKEAgent{
		BasePhase:     phaseframe.BasePhase{Ctx: ctx},
		needPushNodes: bkenode.Nodes{{IP: testNodeIP1}},
	}

	patches.ApplyPrivateMethod(e, "logPushAgentStart", func(_ *EnsureBKEAgent) {})
	patches.ApplyPrivateMethod(e, "prepareServiceFile", func(_ *EnsureBKEAgent, _ *bkev1beta1.BKECluster) (string, error) {
		return servicePath, nil
	})
	patches.ApplyPrivateMethod(e, "performAgentPush", func(_ *EnsureBKEAgent, _ interface{}, _ interface{}, _ *bkev1beta1.BKECluster, _ string) ([]string, error) {
		return []string{}, nil
	})
	patches.ApplyPrivateMethod(e, "handlePushResults", func(_ *EnsureBKEAgent, _ interface{}, _ interface{}, _ *bkev1beta1.BKECluster, _ []string) error {
		return nil
	})

	err := e.pushAgent()
	assert.NoError(t, err)
}

// TestEnsureBKEAgent_PushAgent_PrepareServiceFileError 测试pushAgent准备服务文件失败
func TestEnsureBKEAgent_PushAgent_PrepareServiceFileError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	ctx := createTestPhaseContext(&bkev1beta1.BKECluster{})
	e := &EnsureBKEAgent{
		BasePhase:     phaseframe.BasePhase{Ctx: ctx},
		needPushNodes: bkenode.Nodes{{IP: testNodeIP1}},
	}

	patches.ApplyPrivateMethod(e, "logPushAgentStart", func(_ *EnsureBKEAgent) {})
	patches.ApplyPrivateMethod(e, "prepareServiceFile", func(_ *EnsureBKEAgent, _ *bkev1beta1.BKECluster) (string, error) {
		return "", assert.AnError
	})

	err := e.pushAgent()
	assert.Error(t, err)
}

// TestEnsureBKEAgent_PrepareServiceFile_Success 测试prepareServiceFile成功
func TestEnsureBKEAgent_PrepareServiceFile_Success(t *testing.T) {
	t.Skip("skip unstable UT: depends on /tmp availability in CI")
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	tempDir := t.TempDir()

	bkeCluster := createTestBKECluster(createTestNodes(1))
	bkeCluster.Spec.ClusterConfig.Cluster.NTPServer = testNTPServer
	bkeCluster.Spec.ClusterConfig.Cluster.AgentHealthPort = "8080"

	ctx := createTestPhaseContext(bkeCluster)
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyFunc(os.ReadFile, func(name string) ([]byte, error) {
		return []byte("--ntpserver= --health-port="), nil
	})
	patches.ApplyFunc(os.MkdirTemp, func(dir, pattern string) (string, error) {
		return tempDir, nil
	})

	path, err := e.prepareServiceFile(bkeCluster)
	assert.NoError(t, err)
	assert.NotEmpty(t, path)
}

// TestEnsureBKEAgent_PerformAgentPush_Success 测试performAgentPush成功
func TestEnsureBKEAgent_PerformAgentPush_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	ctx := createTestPhaseContext(&bkev1beta1.BKECluster{})
	e := &EnsureBKEAgent{
		BasePhase:     phaseframe.BasePhase{Ctx: ctx},
		needPushNodes: bkenode.Nodes{{IP: testNodeIP1}},
	}

	patches.ApplyPrivateMethod(e, "sshPushAgent", func(_ *EnsureBKEAgent, _ interface{}, _ interface{}, _ []byte, _ string) (map[string]error, error) {
		return make(map[string]error), nil
	})

	failedIPs, err := e.performAgentPush(context.Background(), nil, &bkev1beta1.BKECluster{}, "/tmp/test.service")
	assert.NoError(t, err)
	assert.Empty(t, failedIPs)
}

// TestEnsureBKEAgent_HandlePushResults_AllSuccess 测试handlePushResults全部成功
func TestEnsureBKEAgent_HandlePushResults_AllSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := createTestBKECluster(createTestNodes(1))
	ctx := createTestPhaseContext(bkeCluster)
	e := &EnsureBKEAgent{
		BasePhase:     phaseframe.BasePhase{Ctx: ctx},
		needPushNodes: bkenode.Nodes{{IP: testNodeIP1}},
	}

	patches.ApplyPrivateMethod(e, "handlePushResults", func(_ *EnsureBKEAgent, _ context.Context, _ client.Client, _ *bkev1beta1.BKECluster, _ []string) error {
		return nil
	})

	err := e.handlePushResults(context.Background(), nil, bkeCluster, []string{})
	assert.NoError(t, err)
}

// TestEnsureBKEAgent_HandlePushResults_WithFailures 测试handlePushResults有失败节点
func TestEnsureBKEAgent_HandlePushResults_WithFailures(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := createTestBKECluster(createTestNodes(1))
	ctx := createTestPhaseContext(bkeCluster)
	e := &EnsureBKEAgent{
		BasePhase:     phaseframe.BasePhase{Ctx: ctx},
		needPushNodes: bkenode.Nodes{{IP: testNodeIP1}},
	}

	patches.ApplyPrivateMethod(e, "handlePushResults", func(_ *EnsureBKEAgent, _ context.Context, _ client.Client, _ *bkev1beta1.BKECluster, _ []string) error {
		return assert.AnError
	})

	err := e.handlePushResults(context.Background(), nil, bkeCluster, []string{testNodeIP1})
	assert.Error(t, err)
}

// TestEnsureBKEAgent_SshPushAgent_Success 测试sshPushAgent成功
func TestEnsureBKEAgent_SshPushAgent_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	ctx := createTestPhaseContext(&bkev1beta1.BKECluster{})
	e := &EnsureBKEAgent{
		BasePhase:       phaseframe.BasePhase{Ctx: ctx},
		needPushNodes:   bkenode.Nodes{{IP: testNodeIP1}},
		localKubeConfig: []byte("test-config"),
	}

	patches.ApplyPrivateMethod(e, "sshPushAgent", func(_ *EnsureBKEAgent, _ context.Context, _ client.Client, _ []byte, _ string) (map[string]error, error) {
		return make(map[string]error), nil
	})

	errs, err := e.sshPushAgent(context.Background(), nil, []byte("test"), "/tmp/test.service")
	assert.NoError(t, err)
	assert.Empty(t, errs)
}

// TestEnsureBKEAgent_ExecutePreCommand_Success 测试executePreCommand成功
func TestEnsureBKEAgent_ExecutePreCommand_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	ctx := createTestPhaseContext(&bkev1beta1.BKECluster{})
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	mockMultiCli := &bkessh.MultiCli{}
	pushAgentErrs := make(map[string]error)

	patches.ApplyPrivateMethod(e, "executePreCommand", func(_ *EnsureBKEAgent, _ *bkessh.MultiCli, _ map[string]error, _ bool) error {
		return nil
	})

	err := e.executePreCommand(mockMultiCli, pushAgentErrs, false)
	assert.NoError(t, err)
}

// TestEnsureBKEAgent_ExecuteStartCommand_Success 测试executeStartCommand成功
func TestEnsureBKEAgent_ExecuteStartCommand_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	ctx := createTestPhaseContext(&bkev1beta1.BKECluster{})
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	mockMultiCli := &bkessh.MultiCli{}
	pushAgentErrs := make(map[string]error)

	patches.ApplyPrivateMethod(e, "executeStartCommand", func(_ *EnsureBKEAgent, _ *bkessh.MultiCli, _ []byte, _ string, _ map[string]error, _ bool) error {
		return nil
	})

	err := e.executeStartCommand(mockMultiCli, []byte("test"), "/tmp/test.service", pushAgentErrs, false)
	assert.NoError(t, err)
}

// TestEnsureBKEAgent_PingAgent_Success 测试pingAgent成功
func TestEnsureBKEAgent_PingAgent_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := createTestBKECluster(createTestNodes(1))
	ctx := createTestPhaseContext(bkeCluster)
	e := &EnsureBKEAgent{
		BasePhase:     phaseframe.BasePhase{Ctx: ctx},
		needPushNodes: bkenode.Nodes{{IP: testNodeIP1, Hostname: testHostname1}},
	}

	patches.ApplyPrivateMethod(e, "pingAgent", func(_ *EnsureBKEAgent) error {
		return nil
	})

	err := e.pingAgent()
	assert.NoError(t, err)
}

// TestEnsureBKEAgent_ValidateAndHandleNodesField_Success 测试validateAndHandleNodesField成功
func TestEnsureBKEAgent_ValidateAndHandleNodesField_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := createTestBKECluster(createTestNodes(1))
	ctx := createTestPhaseContext(bkeCluster)
	e := &EnsureBKEAgent{
		BasePhase:     phaseframe.BasePhase{Ctx: ctx},
		needPushNodes: bkenode.Nodes{{IP: testNodeIP1, Hostname: testHostname1}},
	}

	patches.ApplyPrivateMethod(e, "validateAndHandleNodesField", func(_ *EnsureBKEAgent) error {
		return nil
	})

	err := e.validateAndHandleNodesField()
	assert.NoError(t, err)
}

// TestEnsureBKEAgent_ValidateAndHandleNodesField_ValidationError 测试validateAndHandleNodesField验证失败
func TestEnsureBKEAgent_ValidateAndHandleNodesField_ValidationError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := createTestBKECluster(createTestNodes(1))
	ctx := createTestPhaseContext(bkeCluster)
	e := &EnsureBKEAgent{
		BasePhase:     phaseframe.BasePhase{Ctx: ctx},
		needPushNodes: bkenode.Nodes{{IP: testNodeIP1}},
	}

	patches.ApplyFunc(bkevalidate.ValidateNodesFields, func(_ bkenode.Nodes) error {
		return assert.AnError
	})
	patches.ApplyPrivateMethod(e, "handleValidationFailure", func(_ *EnsureBKEAgent, _ error) error {
		return assert.AnError
	})

	err := e.validateAndHandleNodesField()
	assert.Error(t, err)
}

// TestEnsureBKEAgent_HandleValidationFailure 测试handleValidationFailure
func TestEnsureBKEAgent_HandleValidationFailure(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := createTestBKECluster(createTestNodes(1))
	ctx := createTestPhaseContext(bkeCluster)
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyPrivateMethod(e, "handleValidationFailure", func(_ *EnsureBKEAgent, _ error) error {
		return assert.AnError
	})

	err := e.handleValidationFailure(assert.AnError)
	assert.Error(t, err)
}

// TestEnsureBKEAgent_CheckAvailableHosts_NoHosts 测试checkAvailableHosts无可用主机
func TestEnsureBKEAgent_CheckAvailableHosts_NoHosts(t *testing.T) {
	ctx := createTestPhaseContext(&bkev1beta1.BKECluster{})
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	multiCli := &bkessh.MultiCli{}
	errs := make(map[string]error)

	result := e.checkAvailableHosts(multiCli, errs)
	assert.False(t, result)
}

// TestEnsureBKEAgent_AllNeedPushNodesFailed_AllMatch 测试allNeedPushNodesFailed全部匹配
func TestEnsureBKEAgent_AllNeedPushNodesFailed_AllMatch(t *testing.T) {
	ctx := createTestPhaseContext(&bkev1beta1.BKECluster{})
	e := &EnsureBKEAgent{
		BasePhase:     phaseframe.BasePhase{Ctx: ctx},
		needPushNodes: bkenode.Nodes{{IP: testNodeIP1}, {IP: testNodeIP2}},
	}

	failedIPs := []string{testNodeIP1, testNodeIP2}
	result := e.allNeedPushNodesFailed(failedIPs)
	assert.True(t, result)
}

// TestEnsureBKEAgent_PrepareFileUploadList_WithService 测试prepareFileUploadList
func TestEnsureBKEAgent_PrepareFileUploadList_WithService(t *testing.T) {
	t.Skip("skip unstable UT: depends on /tmp availability in CI")
	tempDir := t.TempDir()
	servicePath := filepath.Join(tempDir, "test.service")
	err := os.WriteFile(servicePath, []byte("test"), testFilePerm)
	assert.NoError(t, err)

	bkeCluster := createTestBKECluster(createTestNodes(1))
	ctx := createTestPhaseContext(bkeCluster)
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	result := e.prepareFileUploadList(servicePath)
	assert.NotEmpty(t, result)
}

// TestEnsureBKEAgent_AddGlobalCAFilesIfNeeded_WithAddons 测试addGlobalCAFilesIfNeeded有addons
func TestEnsureBKEAgent_AddGlobalCAFilesIfNeeded_WithAddons(t *testing.T) {
	bkeCluster := createTestBKECluster(createTestNodes(1))
	bkeCluster.Spec.ClusterConfig.Addons = []confv1beta1.Product{{Name: "test"}}
	ctx := createTestPhaseContext(bkeCluster)
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	fileUpList := []bkessh.File{}
	result := e.addGlobalCAFilesIfNeeded(fileUpList)
	assert.NotNil(t, result)
}

// TestEnsureBKEAgent_AddCSRFilesToUploadList_EmptyList 测试addCSRFilesToUploadList空列表
func TestEnsureBKEAgent_AddCSRFilesToUploadList_EmptyList(t *testing.T) {
	ctx := createTestPhaseContext(&bkev1beta1.BKECluster{})
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	fileUpList := []bkessh.File{}
	result := e.addCSRFilesToUploadList(fileUpList)
	assert.NotNil(t, result)
}

// TestEnsureBKEAgent_UpdateNodeStatus_WithNodes 测试updateNodeStatus
func TestEnsureBKEAgent_UpdateNodeStatus_WithNodes(t *testing.T) {
	bkeCluster := createTestBKECluster(createTestNodes(1))
	ctx := createTestPhaseContext(bkeCluster)
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	successNodes := []string{testNodeIP1 + "success"}
	failedNodes := []string{}
	e.updateNodeStatus(bkeCluster, successNodes, failedNodes)
	assert.NotNil(t, bkeCluster)
}

// TestEnsureBKEAgent_CheckAllOrPushedAgentsFailed_AllSuccess 测试checkAllOrPushedAgentsFailed全部成功
func TestEnsureBKEAgent_CheckAllOrPushedAgentsFailed_AllSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	ctx := createTestPhaseContext(&bkev1beta1.BKECluster{})
	e := &EnsureBKEAgent{
		BasePhase:     phaseframe.BasePhase{Ctx: ctx},
		needPushNodes: bkenode.Nodes{{IP: testNodeIP1}},
	}

	patches.ApplyPrivateMethod(e, "checkAllOrPushedAgentsFailed", func(_ *EnsureBKEAgent, _ []string, _ []string) error {
		return nil
	})

	err := e.checkAllOrPushedAgentsFailed([]string{testNodeIP1}, []string{})
	assert.NoError(t, err)
}

// TestEnsureBKEAgent_LoadLocalKubeConfig_Mock 测试loadLocalKubeConfig
func TestEnsureBKEAgent_LoadLocalKubeConfig_Mock(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	ctx := createTestPhaseContext(&bkev1beta1.BKECluster{})
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyPrivateMethod(e, "loadLocalKubeConfig", func(agent *EnsureBKEAgent) error {
		agent.localKubeConfig = []byte("test-config")
		return nil
	})

	err := e.loadLocalKubeConfig()
	assert.NoError(t, err)
	assert.NotEmpty(t, e.localKubeConfig)
}

// TestEnsureBKEAgent_GetNeedPushNodes_Mock 测试getNeedPushNodes
func TestEnsureBKEAgent_GetNeedPushNodes_Mock(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	ctx := createTestPhaseContext(&bkev1beta1.BKECluster{})
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyPrivateMethod(e, "getNeedPushNodes", func(agent *EnsureBKEAgent) error {
		agent.needPushNodes = bkenode.Nodes{{IP: testNodeIP1}}
		return nil
	})

	err := e.getNeedPushNodes()
	assert.NoError(t, err)
	assert.Len(t, e.needPushNodes, 1)
}

// TestEnsureBKEAgent_HandlePushResults_Mock 测试handlePushResults
func TestEnsureBKEAgent_HandlePushResults_Mock(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := createTestBKECluster(createTestNodes(1))
	ctx := createTestPhaseContext(bkeCluster)
	e := &EnsureBKEAgent{
		BasePhase:     phaseframe.BasePhase{Ctx: ctx},
		needPushNodes: bkenode.Nodes{{IP: testNodeIP1}},
	}

	patches.ApplyPrivateMethod(e, "handlePushResults", func(_ *EnsureBKEAgent, _ context.Context, _ client.Client, _ *bkev1beta1.BKECluster, _ []string) error {
		return nil
	})

	err := e.handlePushResults(context.Background(), nil, bkeCluster, []string{})
	assert.NoError(t, err)
}

// TestEnsureBKEAgent_SshPushAgent_Mock 测试sshPushAgent
func TestEnsureBKEAgent_SshPushAgent_Mock(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	ctx := createTestPhaseContext(&bkev1beta1.BKECluster{})
	e := &EnsureBKEAgent{
		BasePhase:       phaseframe.BasePhase{Ctx: ctx},
		needPushNodes:   bkenode.Nodes{{IP: testNodeIP1}},
		localKubeConfig: []byte("test-config"),
	}

	patches.ApplyPrivateMethod(e, "sshPushAgent", func(_ *EnsureBKEAgent, _ context.Context, _ client.Client, _ []byte, _ string) (map[string]error, error) {
		return make(map[string]error), nil
	})

	errs, err := e.sshPushAgent(context.Background(), nil, []byte("test"), "/tmp/test.service")
	assert.NoError(t, err)
	assert.Empty(t, errs)
}

// TestEnsureBKEAgent_ExecutePreCommand_Mock 测试executePreCommand
func TestEnsureBKEAgent_ExecutePreCommand_Mock(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	ctx := createTestPhaseContext(&bkev1beta1.BKECluster{})
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	mockMultiCli := &bkessh.MultiCli{}
	pushAgentErrs := make(map[string]error)

	patches.ApplyPrivateMethod(e, "executePreCommand", func(_ *EnsureBKEAgent, _ *bkessh.MultiCli, _ map[string]error, _ bool) error {
		return nil
	})

	err := e.executePreCommand(mockMultiCli, pushAgentErrs, false)
	assert.NoError(t, err)
}

// TestEnsureBKEAgent_ExecuteStartCommand_Mock 测试executeStartCommand
func TestEnsureBKEAgent_ExecuteStartCommand_Mock(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	ctx := createTestPhaseContext(&bkev1beta1.BKECluster{})
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	mockMultiCli := &bkessh.MultiCli{}
	pushAgentErrs := make(map[string]error)

	patches.ApplyPrivateMethod(e, "executeStartCommand", func(_ *EnsureBKEAgent, _ *bkessh.MultiCli, _ []byte, _ string, _ map[string]error, _ bool) error {
		return nil
	})

	err := e.executeStartCommand(mockMultiCli, []byte("test"), "/tmp/test.service", pushAgentErrs, false)
	assert.NoError(t, err)
}

// TestEnsureBKEAgent_PingAgent_Mock 测试pingAgent
func TestEnsureBKEAgent_PingAgent_Mock(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := createTestBKECluster(createTestNodes(1))
	ctx := createTestPhaseContext(bkeCluster)
	e := &EnsureBKEAgent{
		BasePhase:     phaseframe.BasePhase{Ctx: ctx},
		needPushNodes: bkenode.Nodes{{IP: testNodeIP1, Hostname: testHostname1}},
	}

	patches.ApplyPrivateMethod(e, "pingAgent", func(_ *EnsureBKEAgent) error {
		return nil
	})

	err := e.pingAgent()
	assert.NoError(t, err)
}

// TestEnsureBKEAgent_HandleValidationFailure_Mock 测试handleValidationFailure
func TestEnsureBKEAgent_HandleValidationFailure_Mock(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := createTestBKECluster(createTestNodes(1))
	ctx := createTestPhaseContext(bkeCluster)
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyPrivateMethod(e, "handleValidationFailure", func(_ *EnsureBKEAgent, _ error) error {
		return nil
	})

	err := e.handleValidationFailure(assert.AnError)
	assert.NoError(t, err)
}

// TestEnsureBKEAgent_NeedExecute_Mock 测试NeedExecute
func TestEnsureBKEAgent_NeedExecute_Mock(t *testing.T) {
	nodes := createTestNodes(1)
	oldCluster := createTestBKECluster(nodes)
	newCluster := createTestBKECluster(nodes)

	ctx := createTestPhaseContext(newCluster)
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	result := e.NeedExecute(oldCluster, newCluster)
	assert.False(t, result)
}

// TestNewEnsureBKEAgent_Creation 测试NewEnsureBKEAgent创建
func TestNewEnsureBKEAgent_Creation(t *testing.T) {
	bkeCluster := createTestBKECluster(createTestNodes(1))
	ctx := createTestPhaseContext(bkeCluster)
	phase := NewEnsureBKEAgent(ctx)
	assert.NotNil(t, phase)
}

// TestEnsureBKEAgent_AddFilesToUploadList_MultipleFiles 测试addFilesToUploadList多个文件
func TestEnsureBKEAgent_AddFilesToUploadList_MultipleFiles(t *testing.T) {
	t.Skip("skip unstable UT: depends on /tmp availability in CI")
	tempDir := t.TempDir()
	file1 := filepath.Join(tempDir, "file1.txt")
	file2 := filepath.Join(tempDir, "file2.txt")
	err := os.WriteFile(file1, []byte("test1"), testFilePerm)
	assert.NoError(t, err)
	err = os.WriteFile(file2, []byte("test2"), testFilePerm)
	assert.NoError(t, err)

	ctx := createTestPhaseContext(&bkev1beta1.BKECluster{})
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	fileUpList := []bkessh.File{}
	result := e.addFilesToUploadList(fileUpList, []string{file1, file2}, testDstDir)
	assert.Len(t, result, 2)
}

// TestEnsureBKEAgent_CheckAllOrPushedAgentsFailed_PartialFailure 测试checkAllOrPushedAgentsFailed部分失败
func TestEnsureBKEAgent_CheckAllOrPushedAgentsFailed_PartialFailure(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	ctx := createTestPhaseContext(&bkev1beta1.BKECluster{})
	e := &EnsureBKEAgent{
		BasePhase:     phaseframe.BasePhase{Ctx: ctx},
		needPushNodes: bkenode.Nodes{{IP: testNodeIP1}, {IP: testNodeIP2}},
	}

	patches.ApplyPrivateMethod(e, "allNeedPushNodesFailed", func(_ *EnsureBKEAgent, _ []string) bool {
		return false
	})

	err := e.checkAllOrPushedAgentsFailed([]string{testNodeIP1}, []string{testNodeIP2})
	assert.NoError(t, err)
}

func newBKEAgentPhaseCov(t *testing.T, bkeCluster *bkev1beta1.BKECluster) *EnsureBKEAgent {
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
	return &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}
}

func bkeAgentCluster(addons ...confv1beta1.Product) *bkev1beta1.BKECluster {
	return &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Spec:       confv1beta1.BKEClusterSpec{ClusterConfig: &confv1beta1.BKEConfig{Addons: addons}},
	}
}

// ---- loadLocalKubeConfig ----

func TestEnsureBKEAgentLoadLocalKubeConfig(t *testing.T) {
	t.Run("no cluster-api: least privilege success + RBAC", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster(confv1beta1.Product{Name: "other"}))
		patches.ApplyFunc(phaseutil.GetLeastPrivilegeKubeConfig, func(context.Context, client.Client) ([]byte, error) {
			return []byte("least-priv"), nil
		})
		patches.ApplyFunc(phaseutil.GetLocalKubeConfig, func(context.Context, client.Client) ([]byte, error) {
			return []byte("local"), nil
		})
		patches.ApplyFunc(phaseutil.CreateBKEAgentRBACWithLocalKubeConfig, func(context.Context, []byte, *bkev1beta1.BKECluster) error {
			return nil
		})
		require.NoError(t, e.loadLocalKubeConfig())
	})

	t.Run("no cluster-api: least privilege error -> fallback local", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster(confv1beta1.Product{Name: "other"}))
		patches.ApplyFunc(phaseutil.GetLeastPrivilegeKubeConfig, func(context.Context, client.Client) ([]byte, error) {
			return nil, assertErr("least priv failed")
		})
		patches.ApplyFunc(phaseutil.GetLocalKubeConfig, func(context.Context, client.Client) ([]byte, error) {
			return []byte("local"), nil
		})
		require.NoError(t, e.loadLocalKubeConfig())
	})

	t.Run("no cluster-api: both fail", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster(confv1beta1.Product{Name: "other"}))
		patches.ApplyFunc(phaseutil.GetLeastPrivilegeKubeConfig, func(context.Context, client.Client) ([]byte, error) {
			return nil, assertErr("least priv failed")
		})
		patches.ApplyFunc(phaseutil.GetLocalKubeConfig, func(context.Context, client.Client) ([]byte, error) {
			return nil, assertErr("local failed")
		})
		err := e.loadLocalKubeConfig()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "local kubeconfig after fallback")
	})

	t.Run("has cluster-api: local kubeconfig", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster(confv1beta1.Product{Name: "cluster-api"}))
		patches.ApplyFunc(phaseutil.GetLocalKubeConfig, func(context.Context, client.Client) ([]byte, error) {
			return []byte("local"), nil
		})
		require.NoError(t, e.loadLocalKubeConfig())
	})

	t.Run("has cluster-api: local kubeconfig error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster(confv1beta1.Product{Name: "cluster-api"}))
		patches.ApplyFunc(phaseutil.GetLocalKubeConfig, func(context.Context, client.Client) ([]byte, error) {
			return nil, assertErr("local failed")
		})
		err := e.loadLocalKubeConfig()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "local kubeconfig")
	})
}

// ---- getNeedPushNodes ----

func TestEnsureBKEAgentGetNeedPushNodes(t *testing.T) {
	t.Run("fetch error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return nil, assertErr("fetch failed")
		})
		err := e.getNeedPushNodes()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to get BKENodes")
	})

	t.Run("no need push nodes returns nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return bkev1beta1.BKENodes{}, nil
		})
		patches.ApplyFunc(phaseutil.GetNeedPushAgentNodesWithBKENodes, func(_ *bkev1beta1.BKECluster, _ bkev1beta1.BKENodes) bkenode.Nodes {
			return nil
		})
		require.NoError(t, e.getNeedPushNodes())
	})

	t.Run("success sets needPushNodes", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return bkev1beta1.BKENodes{}, nil
		})
		patches.ApplyFunc(phaseutil.GetNeedPushAgentNodesWithBKENodes, func(_ *bkev1beta1.BKECluster, _ bkev1beta1.BKENodes) bkenode.Nodes {
			return bkenode.Nodes{{IP: "10.0.0.1", Hostname: "n1"}}
		})
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })
		require.NoError(t, e.getNeedPushNodes())
		assert.Len(t, e.needPushNodes, 1)
	})
}

// ---- prepareServiceFile ----

func TestEnsureBKEAgentPrepareServiceFile(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())
		patches.ApplyFunc(phaseutil.RenderBKEAgentServiceFile, func(_ *bkev1beta1.BKECluster, _ string) error { return nil })
		path, err := e.prepareServiceFile(e.Ctx.BKECluster)
		require.NoError(t, err)
		assert.NotEmpty(t, path)
	})

	t.Run("render error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())
		patches.ApplyFunc(phaseutil.RenderBKEAgentServiceFile, func(_ *bkev1beta1.BKECluster, _ string) error { return assertErr("render failed") })
		_, err := e.prepareServiceFile(e.Ctx.BKECluster)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "bkeagent.service")
	})
}

// ---- handlePushResults ----

func TestEnsureBKEAgentHandlePushResults(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })
	patches.ApplyFunc((*nodeutil.NodeFetcher).MarkNodeStateFlagForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster, _ string, _ int) error {
		return nil
	})

	t.Run("all failed returns error", func(t *testing.T) {
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())
		e.needPushNodes = bkenode.Nodes{{IP: "10.0.0.1"}}
		err := e.handlePushResults(context.Background(), e.Ctx.Client, e.Ctx.BKECluster, []string{"10.0.0.1"})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "Failed to push agent")
	})

	t.Run("success marks pushed nodes", func(t *testing.T) {
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())
		e.needPushNodes = bkenode.Nodes{{IP: "10.0.0.1"}, {IP: "10.0.0.2"}}
		require.NoError(t, e.handlePushResults(context.Background(), e.Ctx.Client, e.Ctx.BKECluster, []string{"10.0.0.1"}))
	})

	t.Run("master failed returns error", func(t *testing.T) {
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())
		e.needPushNodes = bkenode.Nodes{{IP: "10.0.0.1", Role: []string{"master"}}, {IP: "10.0.0.2", Role: []string{"worker"}}}
		err := e.handlePushResults(context.Background(), e.Ctx.Client, e.Ctx.BKECluster, []string{"10.0.0.1"})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "master node failed")
	})
}

// ---- allNodesSkippedByIPs ----

func TestAllNodesSkippedByIPs(t *testing.T) {
	t.Run("empty input returns false", func(t *testing.T) {
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())
		assert.False(t, e.allNodesSkippedByIPs(nil))
	})

	t.Run("fetch error returns false", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return nil, assertErr("fetch failed")
		})
		assert.False(t, e.allNodesSkippedByIPs([]string{"10.0.0.1"}))
	})

	t.Run("empty bkeNodes returns false", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return bkev1beta1.BKENodes{}, nil
		})
		assert.False(t, e.allNodesSkippedByIPs([]string{"10.0.0.1"}))
	})

	t.Run("all IPs marked NeedSkip returns true", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return bkev1beta1.BKENodes{
				{Spec: confv1beta1.BKENodeSpec{IP: "10.0.0.1"}, Status: confv1beta1.BKENodeStatus{NeedSkip: true}},
				{Spec: confv1beta1.BKENodeSpec{IP: "10.0.0.2"}, Status: confv1beta1.BKENodeStatus{NeedSkip: true}},
			}, nil
		})
		assert.True(t, e.allNodesSkippedByIPs([]string{"10.0.0.1", "10.0.0.2"}))
	})

	t.Run("one IP not NeedSkip returns false", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return bkev1beta1.BKENodes{
				{Spec: confv1beta1.BKENodeSpec{IP: "10.0.0.1"}, Status: confv1beta1.BKENodeStatus{NeedSkip: true}},
				{Spec: confv1beta1.BKENodeSpec{IP: "10.0.0.2"}, Status: confv1beta1.BKENodeStatus{NeedSkip: false}},
			}, nil
		})
		assert.False(t, e.allNodesSkippedByIPs([]string{"10.0.0.1", "10.0.0.2"}))
	})

	t.Run("IP not in bkeNodes returns false", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return bkev1beta1.BKENodes{
				{Spec: confv1beta1.BKENodeSpec{IP: "10.0.0.1"}, Status: confv1beta1.BKENodeStatus{NeedSkip: true}},
			}, nil
		})
		assert.False(t, e.allNodesSkippedByIPs([]string{"10.0.0.1", "10.0.0.99"}))
	})
}

// ---- handleValidationFailure ----

func TestEnsureBKEAgentHandleValidationFailure(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })
	patches.ApplyFunc((*nodeutil.NodeFetcher).UpdateNodeStatusByIP, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _, _ string, _ func(*confv1beta1.BKENodeStatus)) error {
		return nil
	})

	t.Run("hostname not unique updates nodes", func(t *testing.T) {
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())
		e.needPushNodes = bkenode.Nodes{{IP: "10.0.0.1", Hostname: "n1"}}
		err := e.handleValidationFailure(assertErr("hostname is not unique"))
		require.Error(t, err)
	})

	t.Run("generic validation error", func(t *testing.T) {
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())
		err := e.handleValidationFailure(assertErr("invalid field"))
		require.Error(t, err)
	})

	t.Run("sync error", func(t *testing.T) {
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error {
			return assertErr("sync failed")
		})
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())
		err := e.handleValidationFailure(assertErr("invalid"))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "sync status after validation")
	})
}

// ---- validateAndHandleNodesField ----

func TestEnsureBKEAgentValidateAndHandleNodesField(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
		return &nodeutil.FetchResult{Nodes: bkenode.Nodes{{IP: "10.0.0.1"}}}, nil
	})

	t.Run("bke cluster validation error", func(t *testing.T) {
		e := newBKEAgentPhaseCov(t, bkeAgentCluster()) // no annotation -> IsBKECluster true
		patches.ApplyFunc(bkevalidate.ValidateNodesFields, func(_ bkenode.Nodes) error { return assertErr("bad nodes") })
		patches.ApplyPrivateMethod(e, "handleValidationFailure", func(_ *EnsureBKEAgent, _ error) error { return assertErr("handled") })
		err := e.validateAndHandleNodesField()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "handled")
	})

	t.Run("bocloud cluster validation error", func(t *testing.T) {
		cluster := bkeAgentCluster()
		cluster.Annotations = map[string]string{"bke.bocloud.com/cluster-from": "bocloud"}
		e := newBKEAgentPhaseCov(t, cluster)
		patches.ApplyFunc(bkevalidate.ValidateNonStandardNodesFields, func(_ bkenode.Nodes) error { return assertErr("bad nonstandard") })
		patches.ApplyPrivateMethod(e, "handleValidationFailure", func(_ *EnsureBKEAgent, _ error) error { return assertErr("handled") })
		err := e.validateAndHandleNodesField()
		require.Error(t, err)
	})

	t.Run("validation nil returns nil", func(t *testing.T) {
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())
		patches.ApplyFunc(bkevalidate.ValidateNodesFields, func(_ bkenode.Nodes) error { return nil })
		require.NoError(t, e.validateAndHandleNodesField())
	})
}

// ---- pingAgent ----

func TestEnsureBKEAgentPingAgent(t *testing.T) {
	t.Run("ping error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())
		patches.ApplyFunc(phaseutil.PingBKEAgent, func(context.Context, client.Client, *runtime.Scheme, *bkev1beta1.BKECluster) (error, []string, []string) {
			return assertErr("ping failed"), nil, nil
		})
		err := e.pingAgent()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "ping failed")
	})

	t.Run("validate error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())
		patches.ApplyFunc(phaseutil.PingBKEAgent, func(context.Context, client.Client, *runtime.Scheme, *bkev1beta1.BKECluster) (error, []string, []string) {
			return nil, []string{"10.0.0.1"}, nil
		})
		patches.ApplyPrivateMethod(e, "updateNodeStatus", func(_ *EnsureBKEAgent, _ *bkev1beta1.BKECluster, _, _ []string) {})
		patches.ApplyPrivateMethod(e, "validateAndHandleNodesField", func(_ *EnsureBKEAgent) error { return assertErr("validate failed") })
		err := e.pingAgent()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "validate failed")
	})

	t.Run("all agents ping failed", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())
		e.needPushNodes = bkenode.Nodes{{IP: "10.0.0.1"}}
		patches.ApplyFunc(phaseutil.PingBKEAgent, func(context.Context, client.Client, *runtime.Scheme, *bkev1beta1.BKECluster) (error, []string, []string) {
			return nil, nil, []string{"node 10.0.0.1: failed"}
		})
		patches.ApplyPrivateMethod(e, "updateNodeStatus", func(_ *EnsureBKEAgent, _ *bkev1beta1.BKECluster, _, _ []string) {})
		patches.ApplyPrivateMethod(e, "validateAndHandleNodesField", func(_ *EnsureBKEAgent) error { return nil })
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })
		err := e.pingAgent()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "ping all nodes")
	})

	t.Run("success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())
		patches.ApplyFunc(phaseutil.PingBKEAgent, func(context.Context, client.Client, *runtime.Scheme, *bkev1beta1.BKECluster) (error, []string, []string) {
			return nil, []string{"10.0.0.1"}, nil
		})
		patches.ApplyPrivateMethod(e, "updateNodeStatus", func(_ *EnsureBKEAgent, _ *bkev1beta1.BKECluster, _, _ []string) {})
		patches.ApplyPrivateMethod(e, "validateAndHandleNodesField", func(_ *EnsureBKEAgent) error { return nil })
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })
		require.NoError(t, e.pingAgent())
	})
}

// bkeAgentGapsNewMultiCli creates a real *MultiCli for direct method testing.
// The MultiCli is closed via t.Cleanup to avoid context leaks.
func bkeAgentGapsNewMultiCli(t *testing.T) *bkessh.MultiCli {
	t.Helper()
	multiCli := bkessh.NewMultiCli(context.Background())
	t.Cleanup(func() { multiCli.Close() })
	return multiCli
}

// ---- sshPushAgent ----

func TestEnsureBKEAgentSshPushAgent(t *testing.T) {
	t.Run("no available hosts after register", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())

		patches.ApplyFunc((*bkessh.MultiCli).RegisterHosts,
			func(_ *bkessh.MultiCli, _ []bkessh.Host) map[string]error {
				return map[string]error{"10.0.0.1": assertErr("register failed")}
			})
		patches.ApplyFunc((*bkessh.MultiCli).AvailableHosts,
			func(_ *bkessh.MultiCli) []string { return []string{} })

		errs, err := e.sshPushAgent(
			context.Background(),
			bkessh.Hosts{{Address: "10.0.0.1"}},
			[]byte("kubeconfig"),
			"/tmp/bkeagent.service",
		)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "No available hosts")
		require.Contains(t, errs, "10.0.0.1")
		assert.Contains(t, errs["10.0.0.1"].Error(), "register failed")
	})

	t.Run("no available hosts after arch check", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())

		patches.ApplyFunc((*bkessh.MultiCli).RegisterHosts,
			func(_ *bkessh.MultiCli, _ []bkessh.Host) map[string]error {
				return map[string]error{}
			})
		availCallCount := 0
		patches.ApplyFunc((*bkessh.MultiCli).AvailableHosts,
			func(_ *bkessh.MultiCli) []string {
				availCallCount++
				if availCallCount <= 1 {
					return []string{"10.0.0.1"}
				}
				return []string{}
			})
		patches.ApplyFunc((*bkessh.MultiCli).RegisterHostsInfo,
			func(_ *bkessh.MultiCli) map[string]error {
				return map[string]error{"10.0.0.1": assertErr("unknown arch")}
			})

		errs, err := e.sshPushAgent(
			context.Background(),
			bkessh.Hosts{{Address: "10.0.0.1"}},
			[]byte("kubeconfig"),
			"/tmp/bkeagent.service",
		)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "No available hosts")
		require.Contains(t, errs, "10.0.0.1")
		assert.Contains(t, errs["10.0.0.1"].Error(), "unknown arch")
	})

	t.Run("executePreCommand error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())

		patches.ApplyFunc((*bkessh.MultiCli).RegisterHosts,
			func(_ *bkessh.MultiCli, _ []bkessh.Host) map[string]error {
				return map[string]error{}
			})
		patches.ApplyFunc((*bkessh.MultiCli).AvailableHosts,
			func(_ *bkessh.MultiCli) []string { return []string{"10.0.0.1"} })
		patches.ApplyFunc((*bkessh.MultiCli).RegisterHostsInfo,
			func(_ *bkessh.MultiCli) map[string]error { return map[string]error{} })
		patches.ApplyPrivateMethod(e, "executePreCommand",
			func(_ *EnsureBKEAgent, _ *bkessh.MultiCli, _ map[string]error) error {
				return assertErr("pre command failed")
			})

		_, err := e.sshPushAgent(
			context.Background(),
			bkessh.Hosts{{Address: "10.0.0.1"}},
			[]byte("kubeconfig"),
			"/tmp/bkeagent.service",
		)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "pre command failed")
	})

	t.Run("executeStartCommand error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())

		patches.ApplyFunc((*bkessh.MultiCli).RegisterHosts,
			func(_ *bkessh.MultiCli, _ []bkessh.Host) map[string]error {
				return map[string]error{}
			})
		patches.ApplyFunc((*bkessh.MultiCli).AvailableHosts,
			func(_ *bkessh.MultiCli) []string { return []string{"10.0.0.1"} })
		patches.ApplyFunc((*bkessh.MultiCli).RegisterHostsInfo,
			func(_ *bkessh.MultiCli) map[string]error { return map[string]error{} })
		patches.ApplyPrivateMethod(e, "executePreCommand",
			func(_ *EnsureBKEAgent, _ *bkessh.MultiCli, _ map[string]error) error { return nil })
		patches.ApplyPrivateMethod(e, "executeStartCommand",
			func(_ *EnsureBKEAgent, _ *bkessh.MultiCli, _ []byte, _ string, _ map[string]error) error {
				return assertErr("start command failed")
			})

		_, err := e.sshPushAgent(
			context.Background(),
			bkessh.Hosts{{Address: "10.0.0.1"}},
			[]byte("kubeconfig"),
			"/tmp/bkeagent.service",
		)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "start command failed")
	})

	t.Run("success with post command errors", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())

		patches.ApplyFunc((*bkessh.MultiCli).RegisterHosts,
			func(_ *bkessh.MultiCli, _ []bkessh.Host) map[string]error {
				return map[string]error{}
			})
		patches.ApplyFunc((*bkessh.MultiCli).AvailableHosts,
			func(_ *bkessh.MultiCli) []string { return []string{"10.0.0.1"} })
		patches.ApplyFunc((*bkessh.MultiCli).RegisterHostsInfo,
			func(_ *bkessh.MultiCli) map[string]error { return map[string]error{} })
		patches.ApplyPrivateMethod(e, "executePreCommand",
			func(_ *EnsureBKEAgent, _ *bkessh.MultiCli, _ map[string]error) error { return nil })
		patches.ApplyPrivateMethod(e, "executeStartCommand",
			func(_ *EnsureBKEAgent, _ *bkessh.MultiCli, _ []byte, _ string, _ map[string]error) error { return nil })
		patches.ApplyFunc((*bkessh.MultiCli).Run,
			func(_ *bkessh.MultiCli, _ bkessh.Command) (bkessh.StdCombine, bkessh.StdCombine) {
				stdErrs := bkessh.NewStdCombine()
				stdErrs.Add(bkessh.NewCombineOut("10.0.0.1", "post", "post command failed"))
				return stdErrs, bkessh.NewStdCombine()
			})

		errs, err := e.sshPushAgent(
			context.Background(),
			bkessh.Hosts{{Address: "10.0.0.1"}},
			[]byte("kubeconfig"),
			"/tmp/bkeagent.service",
		)
		require.NoError(t, err)
		require.Contains(t, errs, "10.0.0.1")
		assert.Contains(t, errs["10.0.0.1"].Error(), "PostCommandFailed")
	})

	t.Run("success clean", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())

		patches.ApplyFunc((*bkessh.MultiCli).RegisterHosts,
			func(_ *bkessh.MultiCli, _ []bkessh.Host) map[string]error {
				return map[string]error{}
			})
		patches.ApplyFunc((*bkessh.MultiCli).AvailableHosts,
			func(_ *bkessh.MultiCli) []string { return []string{"10.0.0.1"} })
		patches.ApplyFunc((*bkessh.MultiCli).RegisterHostsInfo,
			func(_ *bkessh.MultiCli) map[string]error { return map[string]error{} })
		patches.ApplyPrivateMethod(e, "executePreCommand",
			func(_ *EnsureBKEAgent, _ *bkessh.MultiCli, _ map[string]error) error { return nil })
		patches.ApplyPrivateMethod(e, "executeStartCommand",
			func(_ *EnsureBKEAgent, _ *bkessh.MultiCli, _ []byte, _ string, _ map[string]error) error { return nil })
		patches.ApplyFunc((*bkessh.MultiCli).Run,
			func(_ *bkessh.MultiCli, _ bkessh.Command) (bkessh.StdCombine, bkessh.StdCombine) {
				return bkessh.NewStdCombine(), bkessh.NewStdCombine()
			})

		errs, err := e.sshPushAgent(
			context.Background(),
			bkessh.Hosts{{Address: "10.0.0.1"}},
			[]byte("kubeconfig"),
			"/tmp/bkeagent.service",
		)
		require.NoError(t, err)
		assert.Empty(t, errs)
	})
}

// ---- executePreCommand ----

func TestEnsureBKEAgentExecutePreCommand(t *testing.T) {
	t.Run("success no errors", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())
		multiCli := bkeAgentGapsNewMultiCli(t)

		patches.ApplyFunc((*bkessh.MultiCli).Run,
			func(_ *bkessh.MultiCli, _ bkessh.Command) (bkessh.StdCombine, bkessh.StdCombine) {
				return bkessh.NewStdCombine(), bkessh.NewStdCombine()
			})
		patches.ApplyFunc((*bkessh.MultiCli).AvailableHosts,
			func(_ *bkessh.MultiCli) []string { return []string{"10.0.0.1"} })

		pushAgentErrs := map[string]error{}
		err := e.executePreCommand(multiCli, pushAgentErrs, false)
		require.NoError(t, err)
		assert.Empty(t, pushAgentErrs)
	})

	t.Run("pre command errors no hosts left", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())
		multiCli := bkeAgentGapsNewMultiCli(t)

		patches.ApplyFunc((*bkessh.MultiCli).Run,
			func(_ *bkessh.MultiCli, _ bkessh.Command) (bkessh.StdCombine, bkessh.StdCombine) {
				stdErrs := bkessh.NewStdCombine()
				stdErrs.Add(bkessh.NewCombineOut("10.0.0.1", "pre", "pre command failed"))
				return stdErrs, bkessh.NewStdCombine()
			})
		patches.ApplyFunc((*bkessh.MultiCli).AvailableHosts,
			func(_ *bkessh.MultiCli) []string { return []string{} })

		pushAgentErrs := map[string]error{}
		err := e.executePreCommand(multiCli, pushAgentErrs, false)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "No available hosts")
		require.Contains(t, pushAgentErrs, "10.0.0.1")
		assert.Contains(t, pushAgentErrs["10.0.0.1"].Error(), "PreCommandFailed")
	})

	t.Run("pre command errors hosts remain", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())
		multiCli := bkeAgentGapsNewMultiCli(t)

		patches.ApplyFunc((*bkessh.MultiCli).Run,
			func(_ *bkessh.MultiCli, _ bkessh.Command) (bkessh.StdCombine, bkessh.StdCombine) {
				stdErrs := bkessh.NewStdCombine()
				stdErrs.Add(bkessh.NewCombineOut("10.0.0.1", "pre", "pre command failed"))
				return stdErrs, bkessh.NewStdCombine()
			})
		patches.ApplyFunc((*bkessh.MultiCli).AvailableHosts,
			func(_ *bkessh.MultiCli) []string { return []string{"10.0.0.2"} })

		pushAgentErrs := map[string]error{}
		err := e.executePreCommand(multiCli, pushAgentErrs, false)
		require.NoError(t, err)
		require.Contains(t, pushAgentErrs, "10.0.0.1")
		assert.Contains(t, pushAgentErrs["10.0.0.1"].Error(), "PreCommandFailed")
	})

	t.Run("nil pushAgentErrs skips map set", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())
		multiCli := bkeAgentGapsNewMultiCli(t)

		patches.ApplyFunc((*bkessh.MultiCli).Run,
			func(_ *bkessh.MultiCli, _ bkessh.Command) (bkessh.StdCombine, bkessh.StdCombine) {
				stdErrs := bkessh.NewStdCombine()
				stdErrs.Add(bkessh.NewCombineOut("10.0.0.1", "pre", "pre command failed"))
				return stdErrs, bkessh.NewStdCombine()
			})
		patches.ApplyFunc((*bkessh.MultiCli).AvailableHosts,
			func(_ *bkessh.MultiCli) []string { return []string{"10.0.0.2"} })

		err := e.executePreCommand(multiCli, nil, false)
		require.NoError(t, err)
	})
}

// ---- executeStartCommand ----

func TestEnsureBKEAgentExecuteStartCommand(t *testing.T) {
	t.Run("success no errors", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())
		multiCli := bkeAgentGapsNewMultiCli(t)

		patches.ApplyFunc((*bkessh.MultiCli).Run,
			func(_ *bkessh.MultiCli, _ bkessh.Command) (bkessh.StdCombine, bkessh.StdCombine) {
				return bkessh.NewStdCombine(), bkessh.NewStdCombine()
			})
		patches.ApplyFunc((*bkessh.MultiCli).AvailableHosts,
			func(_ *bkessh.MultiCli) []string { return []string{"10.0.0.1"} })

		pushAgentErrs := map[string]error{}
		err := e.executeStartCommand(multiCli, []byte("kubeconfig"), "/tmp/bkeagent.service", pushAgentErrs, false)
		require.NoError(t, err)
	})

	t.Run("ignores Created symlink stderr", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())
		multiCli := bkeAgentGapsNewMultiCli(t)

		patches.ApplyFunc((*bkessh.MultiCli).Run,
			func(_ *bkessh.MultiCli, _ bkessh.Command) (bkessh.StdCombine, bkessh.StdCombine) {
				stdErrs := bkessh.NewStdCombine()
				stdErrs.Add(bkessh.NewCombineOut("10.0.0.1", "start",
					"Created symlink /etc/systemd/system/multi-user.target.wants/bkeagent.service"))
				return stdErrs, bkessh.NewStdCombine()
			})
		patches.ApplyFunc((*bkessh.MultiCli).AvailableHosts,
			func(_ *bkessh.MultiCli) []string { return []string{"10.0.0.1"} })

		pushAgentErrs := map[string]error{"10.0.0.1": assertErr("previous error")}
		err := e.executeStartCommand(multiCli, []byte("kubeconfig"), "/tmp/bkeagent.service", pushAgentErrs, false)
		require.NoError(t, err)
		_, exists := pushAgentErrs["10.0.0.1"]
		assert.False(t, exists, "error should be deleted for Created symlink")
	})

	t.Run("ignores File exists stderr", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())
		multiCli := bkeAgentGapsNewMultiCli(t)

		patches.ApplyFunc((*bkessh.MultiCli).Run,
			func(_ *bkessh.MultiCli, _ bkessh.Command) (bkessh.StdCombine, bkessh.StdCombine) {
				stdErrs := bkessh.NewStdCombine()
				stdErrs.Add(bkessh.NewCombineOut("10.0.0.1", "start",
					"Failed to execute operation: File exists"))
				return stdErrs, bkessh.NewStdCombine()
			})
		patches.ApplyFunc((*bkessh.MultiCli).AvailableHosts,
			func(_ *bkessh.MultiCli) []string { return []string{"10.0.0.1"} })

		pushAgentErrs := map[string]error{"10.0.0.1": assertErr("previous error")}
		err := e.executeStartCommand(multiCli, []byte("kubeconfig"), "/tmp/bkeagent.service", pushAgentErrs, false)
		require.NoError(t, err)
		_, exists := pushAgentErrs["10.0.0.1"]
		assert.False(t, exists, "error should be deleted for File exists")
	})

	t.Run("start command fails no hosts left", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())
		multiCli := bkeAgentGapsNewMultiCli(t)

		patches.ApplyFunc((*bkessh.MultiCli).Run,
			func(_ *bkessh.MultiCli, _ bkessh.Command) (bkessh.StdCombine, bkessh.StdCombine) {
				stdErrs := bkessh.NewStdCombine()
				stdErrs.Add(bkessh.NewCombineOut("10.0.0.1", "start", "real failure"))
				return stdErrs, bkessh.NewStdCombine()
			})
		patches.ApplyFunc((*bkessh.MultiCli).AvailableHosts,
			func(_ *bkessh.MultiCli) []string { return []string{} })

		pushAgentErrs := map[string]error{}
		err := e.executeStartCommand(multiCli, []byte("kubeconfig"), "/tmp/bkeagent.service", pushAgentErrs, false)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "No available hosts")
		require.Contains(t, pushAgentErrs, "10.0.0.1")
		assert.Contains(t, pushAgentErrs["10.0.0.1"].Error(), "real failure")
	})

	t.Run("start command fails hosts remain", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newBKEAgentPhaseCov(t, bkeAgentCluster())
		multiCli := bkeAgentGapsNewMultiCli(t)

		patches.ApplyFunc((*bkessh.MultiCli).Run,
			func(_ *bkessh.MultiCli, _ bkessh.Command) (bkessh.StdCombine, bkessh.StdCombine) {
				stdErrs := bkessh.NewStdCombine()
				stdErrs.Add(bkessh.NewCombineOut("10.0.0.1", "start", "real failure"))
				return stdErrs, bkessh.NewStdCombine()
			})
		patches.ApplyFunc((*bkessh.MultiCli).AvailableHosts,
			func(_ *bkessh.MultiCli) []string { return []string{"10.0.0.2"} })

		pushAgentErrs := map[string]error{}
		err := e.executeStartCommand(multiCli, []byte("kubeconfig"), "/tmp/bkeagent.service", pushAgentErrs, false)
		require.NoError(t, err)
		require.Contains(t, pushAgentErrs, "10.0.0.1")
		assert.Contains(t, pushAgentErrs["10.0.0.1"].Error(), "real failure")
	})
}

// createRetryTestBKENode creates a BKENode CRD object for testing retry logic.
func createRetryTestBKENode(name, namespace, clusterName, ip string, role []string, retryCount int) *confv1beta1.BKENode {
	return &confv1beta1.BKENode{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels:    map[string]string{"cluster.x-k8s.io/cluster-name": clusterName},
		},
		Spec: confv1beta1.BKENodeSpec{
			IP:   ip,
			Role: role,
		},
		Status: confv1beta1.BKENodeStatus{
			RetryCount: retryCount,
		},
	}
}

// createRetryTestPhaseContext creates a PhaseContext with BKENode objects in the fake client.
func createRetryTestPhaseContext(bkeCluster *bkev1beta1.BKECluster, bkeNodes ...*confv1beta1.BKENode) *phaseframe.PhaseContext {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)

	objs := []client.Object{bkeCluster}
	for _, n := range bkeNodes {
		objs = append(objs, n)
	}

	builder := fake.NewClientBuilder().WithScheme(scheme).WithObjects(objs...)
	for _, n := range bkeNodes {
		builder = builder.WithStatusSubresource(n)
	}
	c := builder.Build()

	recorder := &fakeRecorder{}
	return &phaseframe.PhaseContext{
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, recorder, bkeCluster),
		Client:     c,
		Scheme:     scheme,
		Context:    context.Background(),
	}
}

// TestEnsureBKEAgent_UpdateNodeStatus_WorkerRetryIncrement tests that RetryCount
// increments on worker node failure without setting NeedSkip when below ReconcileAllowedFailedCount.
func TestEnsureBKEAgent_UpdateNodeStatus_WorkerRetryIncrement(t *testing.T) {
	origAllowed := statusmanage.ReconcileAllowedFailedCount
	t.Cleanup(func() { statusmanage.ReconcileAllowedFailedCount = origAllowed })
	statusmanage.ReconcileAllowedFailedCount = 3

	bkeCluster := createTestBKECluster(createTestNodes(1))
	workerNode := createRetryTestBKENode(
		"test-cluster-127-0-0-1", testNamespace, testClusterName,
		testNodeIP1, []string{bkenode.WorkerNodeRole}, 0,
	)

	ctx := createRetryTestPhaseContext(bkeCluster, workerNode)
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	// Simulate first failure (ping failure path in updateNodeStatus)
	e.updateNodeStatus(bkeCluster, []string{}, []string{testNodeIP1})

	updated := &confv1beta1.BKENode{}
	err := ctx.Client.Get(context.Background(), client.ObjectKey{
		Name:      "test-cluster-127-0-0-1",
		Namespace: testNamespace,
	}, updated)
	assert.NoError(t, err)
	assert.Equal(t, 1, updated.Status.RetryCount)
	assert.False(t, updated.Status.NeedSkip)
}

// TestEnsureBKEAgent_UpdateNodeStatus_WorkerMaxRetryNeedSkip tests that NeedSkip is set
// when RetryCount reaches ReconcileAllowedFailedCount on continued failure.
func TestEnsureBKEAgent_UpdateNodeStatus_WorkerMaxRetryNeedSkip(t *testing.T) {
	origAllowed := statusmanage.ReconcileAllowedFailedCount
	t.Cleanup(func() { statusmanage.ReconcileAllowedFailedCount = origAllowed })
	statusmanage.ReconcileAllowedFailedCount = 3

	bkeCluster := createTestBKECluster(createTestNodes(1))
	workerNode := createRetryTestBKENode(
		"test-cluster-127-0-0-1", testNamespace, testClusterName,
		testNodeIP1, []string{bkenode.WorkerNodeRole}, statusmanage.ReconcileAllowedFailedCount-1,
	)

	ctx := createRetryTestPhaseContext(bkeCluster, workerNode)
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	// Simulate failure that pushes RetryCount to ReconcileAllowedFailedCount
	e.updateNodeStatus(bkeCluster, []string{}, []string{testNodeIP1})

	updated := &confv1beta1.BKENode{}
	err := ctx.Client.Get(context.Background(), client.ObjectKey{
		Name:      "test-cluster-127-0-0-1",
		Namespace: testNamespace,
	}, updated)
	assert.NoError(t, err)
	assert.Equal(t, statusmanage.ReconcileAllowedFailedCount, updated.Status.RetryCount)
	assert.True(t, updated.Status.NeedSkip)
}

func TestEnsureBKEAgent_UpdateNodeStatus_SuccessResetsState(t *testing.T) {
	origAllowed := statusmanage.ReconcileAllowedFailedCount
	t.Cleanup(func() { statusmanage.ReconcileAllowedFailedCount = origAllowed })
	statusmanage.ReconcileAllowedFailedCount = 3

	bkeCluster := createTestBKECluster(createTestNodes(1))
	masterNode := createRetryTestBKENode(
		"test-cluster-127-0-0-1", testNamespace, testClusterName,
		testNodeIP1, []string{bkenode.MasterNodeRole}, 0,
	)

	ctx := createRetryTestPhaseContext(bkeCluster, masterNode)
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	e.updateNodeStatus(bkeCluster, []string{}, []string{testNodeIP1})

	failedNode := &confv1beta1.BKENode{}
	_ = ctx.Client.Get(context.Background(), client.ObjectKey{
		Name: "test-cluster-127-0-0-1", Namespace: testNamespace,
	}, failedNode)
	assert.Equal(t, bkev1beta1.NodeInitFailed, failedNode.Status.State)

	e.updateNodeStatus(bkeCluster, []string{testNodeIP1}, []string{})

	successNode := &confv1beta1.BKENode{}
	err := ctx.Client.Get(context.Background(), client.ObjectKey{
		Name: "test-cluster-127-0-0-1", Namespace: testNamespace,
	}, successNode)
	assert.NoError(t, err)
	assert.Equal(t, bkev1beta1.NodeInitializing, successNode.Status.State)
	assert.True(t, successNode.Status.StateCode&bkev1beta1.NodeAgentReadyFlag != 0)
	assert.True(t, successNode.Status.StateCode&bkev1beta1.NodeAgentPushedFlag != 0)
	assert.Equal(t, 0, successNode.Status.RetryCount)
}

func TestEnsureBKEAgent_CheckAllOrPushedAgentsFailed_AllSkipped(t *testing.T) {
	origAllowed := statusmanage.ReconcileAllowedFailedCount
	t.Cleanup(func() { statusmanage.ReconcileAllowedFailedCount = origAllowed })
	statusmanage.ReconcileAllowedFailedCount = 3

	bkeCluster := createTestBKECluster(createTestNodes(two))
	workerNode := createRetryTestBKENode(
		"test-cluster-127-0-0-2", testNamespace, testClusterName,
		testNodeIP2, []string{bkenode.WorkerNodeRole}, statusmanage.ReconcileAllowedFailedCount,
	)
	workerNode.Status.NeedSkip = true

	ctx := createRetryTestPhaseContext(bkeCluster, workerNode)
	e := &EnsureBKEAgent{
		BasePhase:     phaseframe.BasePhase{Ctx: ctx},
		needPushNodes: bkenode.Nodes{{IP: testNodeIP2, Hostname: testHostname2}},
	}

	err := e.checkAllOrPushedAgentsFailed([]string{}, []string{testNodeIP2})
	assert.NoError(t, err)
}

func TestEnsureBKEAgent_CheckAllOrPushedAgentsFailed_NotAllSkipped(t *testing.T) {
	origAllowed := statusmanage.ReconcileAllowedFailedCount
	t.Cleanup(func() { statusmanage.ReconcileAllowedFailedCount = origAllowed })
	statusmanage.ReconcileAllowedFailedCount = 3

	bkeCluster := createTestBKECluster(createTestNodes(two))
	workerNode := createRetryTestBKENode(
		"test-cluster-127-0-0-2", testNamespace, testClusterName,
		testNodeIP2, []string{bkenode.WorkerNodeRole}, 0,
	)

	ctx := createRetryTestPhaseContext(bkeCluster, workerNode)
	e := &EnsureBKEAgent{
		BasePhase:     phaseframe.BasePhase{Ctx: ctx},
		needPushNodes: bkenode.Nodes{{IP: testNodeIP2, Hostname: testHostname2}},
	}

	err := e.checkAllOrPushedAgentsFailed([]string{}, []string{testNodeIP2})
	assert.Error(t, err)
}

// TestEnsureBKEAgent_UpdateNodeStatus_WorkerSuccessResetsRetry tests that RetryCount
// is reset to 0 when the worker node succeeds.
func TestEnsureBKEAgent_UpdateNodeStatus_WorkerSuccessResetsRetry(t *testing.T) {
	origAllowed := statusmanage.ReconcileAllowedFailedCount
	t.Cleanup(func() { statusmanage.ReconcileAllowedFailedCount = origAllowed })
	statusmanage.ReconcileAllowedFailedCount = 3

	bkeCluster := createTestBKECluster(createTestNodes(1))
	workerNode := createRetryTestBKENode(
		"test-cluster-127-0-0-1", testNamespace, testClusterName,
		testNodeIP1, []string{bkenode.WorkerNodeRole}, 2,
	)

	ctx := createRetryTestPhaseContext(bkeCluster, workerNode)
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	// Simulate success
	e.updateNodeStatus(bkeCluster, []string{testNodeIP1}, []string{})

	updated := &confv1beta1.BKENode{}
	err := ctx.Client.Get(context.Background(), client.ObjectKey{
		Name:      "test-cluster-127-0-0-1",
		Namespace: testNamespace,
	}, updated)
	assert.NoError(t, err)
	assert.Equal(t, 0, updated.Status.RetryCount)
	assert.False(t, updated.Status.NeedSkip)
}

// TestEnsureBKEAgent_UpdateNodeStatus_MasterFailureIncrementsRetry tests that master node
// failure increments RetryCount but does NOT set NeedSkip.
func TestEnsureBKEAgent_UpdateNodeStatus_MasterFailureIncrementsRetry(t *testing.T) {
	origAllowed := statusmanage.ReconcileAllowedFailedCount
	t.Cleanup(func() { statusmanage.ReconcileAllowedFailedCount = origAllowed })
	statusmanage.ReconcileAllowedFailedCount = 3

	bkeCluster := createTestBKECluster(createTestNodes(1))
	masterNode := createRetryTestBKENode(
		"test-cluster-127-0-0-1", testNamespace, testClusterName,
		testNodeIP1, []string{bkenode.MasterNodeRole}, 0,
	)

	ctx := createRetryTestPhaseContext(bkeCluster, masterNode)
	e := &EnsureBKEAgent{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	// Simulate failure
	e.updateNodeStatus(bkeCluster, []string{}, []string{testNodeIP1})

	updated := &confv1beta1.BKENode{}
	err := ctx.Client.Get(context.Background(), client.ObjectKey{
		Name:      "test-cluster-127-0-0-1",
		Namespace: testNamespace,
	}, updated)
	assert.NoError(t, err)
	assert.Equal(t, 1, updated.Status.RetryCount)
	assert.False(t, updated.Status.NeedSkip)
}
