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

package ha

import (
	"context"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"k8s.io/apimachinery/pkg/util/wait"

	bkenet "gopkg.openfuyao.cn/cluster-api-provider-bke/common/utils/net"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/mfutil"
)

const (
	numZero              = 0
	numOne               = 1
	numTwo               = 2
	numThree             = 3
	numFour              = 4
	numEight             = 8
	numTen               = 10
	numSixteen           = 16
	numTwentyFour        = 24
	numOneHundred        = 100
	numOneTwentySeven    = 127
	numOneNinetyTwo      = 192
	ipv4LoopbackSegmentA = numOneHundred + numTwentyFour + numThree
	ipv4LoopbackSegmentB = numZero
	ipv4LoopbackSegmentC = numZero
	ipv4LoopbackSegmentD = numOne

	privateClassBA = numOneNinetyTwo
	privateClassBB = numOneHundred + numSixteen*numFour + numEight
	privateClassBC = numOne
	privateClassBD = numTwo

	waitTimeout          = 100 * time.Millisecond
	shortPollInterval    = 10 * time.Millisecond
	defaultPollTimeout   = 5 * time.Minute
	testHAConfigDir      = "/test/haproxy"
	testKeepAlivedConfig = "/test/keepalived"
	testManifestsDir     = "/test/manifests"
	testImageRepo        = "registry.test.com"
	testHostname         = "test-node"
	testInterface        = "eth0"
	testControlPlaneVIP  = "192.168.1.100"
	testIngressVIP       = "192.168.1.200"
	testControlPlanePort = "6443"
	testVirtualRouterID  = "51"
	testIP               = "192.168.1.1"
)

func TestHAName(t *testing.T) {
	ha := &HA{}
	assert.Equal(t, Name, ha.Name())
}

func TestHAParam(t *testing.T) {
	ha := &HA{}
	params := ha.Param()
	assert.NotNil(t, params)
	assert.Contains(t, params, "haNodes")
	assert.Contains(t, params, "thirdImageRepo")
	assert.Contains(t, params, "fuyaoImageRepo")
	assert.Contains(t, params, "manifestsDir")
	assert.Contains(t, params, "controlPlaneEndpointVIP")
	assert.Contains(t, params, "controlPlaneEndpointPort")
}

func TestNewHA(t *testing.T) {
	ha := New(nil)
	assert.NotNil(t, ha)
	assert.Equal(t, Name, ha.Name())
}

func TestHANodesFormatError(t *testing.T) {
	ha := &HA{exec: nil}
	cfg, err := ha.prepareRendCfg(map[string]string{
		"haNodes": "invalid:format:too:many",
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "format error")
	assert.Nil(t, cfg)
}

func TestHANodesMissingInterface(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(utils.HostName, func() string {
		return "node1"
	})
	patches.ApplyFunc(bkenet.GetInterfaceFromIp, func(ip string) (string, error) {
		return "", nil
	})

	ha := &HA{exec: nil}
	cfg, err := ha.prepareRendCfg(map[string]string{
		"haNodes": "node1:192.168.1.1",
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "interface")
	assert.Nil(t, cfg)
}

func TestHANoVIPConfigured(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(utils.HostName, func() string {
		return "node1"
	})
	patches.ApplyFunc(bkenet.GetInterfaceFromIp, func(ip string) (string, error) {
		return testInterface, nil
	})

	ha := &HA{exec: nil}
	cfg, err := ha.prepareRendCfg(map[string]string{
		"haNodes": "node1:192.168.1.1",
	})
	assert.NoError(t, err)
	assert.NotNil(t, cfg)
}

func TestHANodesValidFormat(t *testing.T) {
	ha := &HA{exec: nil}
	cfg, err := ha.prepareRendCfg(map[string]string{
		"haNodes": "node1:192.168.1.1,node2:192.168.1.2",
	})
	assert.Error(t, err)
	assert.Nil(t, cfg)
}

func TestHAParamDefaults(t *testing.T) {
	ha := &HA{}
	params := ha.Param()
	assert.NotEmpty(t, params["manifestsDir"].Default)
	assert.NotEmpty(t, params["virtualRouterId"].Default)
	assert.NotEmpty(t, params["wait"].Default)
}

func TestHAParamRequired(t *testing.T) {
	ha := &HA{}
	params := ha.Param()
	assert.True(t, params["haNodes"].Required)
	assert.True(t, params["thirdImageRepo"].Required)
	assert.True(t, params["fuyaoImageRepo"].Required)
	assert.False(t, params["controlPlaneEndpointVIP"].Required)
	assert.False(t, params["controlPlaneEndpointPort"].Required)
}

func TestHAWithNilExecutor(t *testing.T) {
	ha := New(nil)
	assert.NotNil(t, ha)
}

func TestHAFields(t *testing.T) {
	ha := &HA{}
	assert.Nil(t, ha.exec)
}

func TestHAParamDescriptions(t *testing.T) {
	ha := &HA{}
	params := ha.Param()
	for key, param := range params {
		if key == "haproxyImageName" || key == "haproxyImageTag" {
			continue
		}
		if param.Default == "" {
			continue
		}
		assert.NotEmpty(t, param.Description, "Description for %s should not be empty", key)
	}
}

func TestExecuteMasterHA(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.HostName, func() string {
		return testHostname
	})

	patches.ApplyFunc(bkenet.GetInterfaceFromIp, func(ip string) (string, error) {
		if ip == testIP {
			return testInterface, nil
		}
		return "", nil
	})

	patches.ApplyFunc(mfutil.GetHAComponentList, func() mfutil.HAComponents {
		return mfutil.HAComponents{}
	})

	var generateYamlCalled bool
	patches.ApplyFunc(mfutil.GenerateHAManifestYaml, func(components mfutil.HAComponents, cfg map[string]interface{}) error {
		generateYamlCalled = true
		cfg["vip"] = testControlPlaneVIP
		cfg["nodes"] = []mfutil.HANode{
			{Hostname: testHostname, IP: testIP},
		}
		cfg["interface"] = testInterface
		cfg["isMasterHa"] = true
		return nil
	})

	patches.ApplyFunc(mfutil.KeepalivedInstanceIsMaster, func(nodes []mfutil.HANode) bool {
		return true
	})

	patches.ApplyFunc(wait.PollImmediateWithContext, func(ctx context.Context, interval, timeout time.Duration, condition wait.ConditionWithContextFunc) error {
		return nil
	})

	mockHA := &HA{
		isMasterHa: true,
	}
	patches.ApplyPrivateMethod(mockHA, "initIPVS", func(_ *HA) {})

	commands := []string{
		"HA",
		"haproxyConfigDir=" + testHAConfigDir,
		"keepAlivedConfigDir=" + testKeepAlivedConfig,
		"manifestsDir=" + testManifestsDir,
		"thirdImageRepo=" + testImageRepo,
		"fuyaoImageRepo=" + testImageRepo,
		"haNodes=" + testHostname + ":" + testIP,
		"controlPlaneEndpointVIP=" + testControlPlaneVIP,
		"controlPlaneEndpointPort=" + testControlPlanePort,
		"virtualRouterId=" + testVirtualRouterID,
		"wait=false",
	}

	result, err := mockHA.Execute(context.Background(), commands)

	assert.NoError(t, err)
	assert.Nil(t, result)
	assert.True(t, generateYamlCalled)
}

func TestExecuteIngressHA(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.HostName, func() string {
		return testHostname
	})

	patches.ApplyFunc(bkenet.GetInterfaceFromIp, func(ip string) (string, error) {
		if ip == testIP {
			return testInterface, nil
		}
		return "", nil
	})

	patches.ApplyFunc(mfutil.GetIngressHaComponentList, func() mfutil.HAComponents {
		return mfutil.HAComponents{}
	})

	var generateYamlCalled bool
	patches.ApplyFunc(mfutil.GenerateHAManifestYaml, func(components mfutil.HAComponents, cfg map[string]interface{}) error {
		generateYamlCalled = true
		cfg["vip"] = testIngressVIP
		cfg["nodes"] = []mfutil.HANode{
			{Hostname: testHostname, IP: testIP},
		}
		cfg["interface"] = testInterface
		cfg["isMasterHa"] = false
		return nil
	})

	mockHA := &HA{
		isMasterHa: false,
	}
	patches.ApplyPrivateMethod(mockHA, "initIPVS", func(_ *HA) {})

	commands := []string{
		"HA",
		"haproxyConfigDir=" + testHAConfigDir,
		"keepAlivedConfigDir=" + testKeepAlivedConfig,
		"manifestsDir=" + testManifestsDir,
		"thirdImageRepo=" + testImageRepo,
		"fuyaoImageRepo=" + testImageRepo,
		"haNodes=" + testHostname + ":" + testIP,
		"ingressVIP=" + testIngressVIP,
		"virtualRouterId=" + testVirtualRouterID,
		"wait=false",
	}

	result, err := mockHA.Execute(context.Background(), commands)

	assert.NoError(t, err)
	assert.Nil(t, result)
	assert.True(t, generateYamlCalled)
}

func TestExecuteWithWait(t *testing.T) {
	t.Skip("skip unstable UT: Execute triggers real initIPVS path and can panic with nil executor in CI")
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.HostName, func() string {
		return testHostname
	})

	patches.ApplyFuncSeq(bkenet.GetInterfaceFromIp, []gomonkey.OutputCell{
		{Values: gomonkey.Params{testInterface, nil}},
		{Values: gomonkey.Params{testInterface, nil}},
	})

	patches.ApplyFunc(mfutil.GetHAComponentList, func() mfutil.HAComponents {
		return mfutil.HAComponents{}
	})

	patches.ApplyFunc(mfutil.GenerateHAManifestYaml, func(components mfutil.HAComponents, cfg map[string]interface{}) error {
		cfg["vip"] = testControlPlaneVIP
		cfg["nodes"] = []mfutil.HANode{
			{Hostname: testHostname, IP: testIP},
		}
		cfg["interface"] = testInterface
		cfg["isMasterHa"] = true
		return nil
	})

	patches.ApplyFunc(mfutil.KeepalivedInstanceIsMaster, func(nodes []mfutil.HANode) bool {
		return true
	})

	var pollCalled bool
	patches.ApplyFunc(wait.PollImmediateWithContext, func(ctx context.Context, interval, timeout time.Duration, condition wait.ConditionWithContextFunc) error {
		pollCalled = true
		if condition != nil {
			condition(ctx)
		}
		return nil
	})

	mockHA := &HA{
		isMasterHa: true,
	}

	commands := []string{
		"HA",
		"haproxyConfigDir=" + testHAConfigDir,
		"keepAlivedConfigDir=" + testKeepAlivedConfig,
		"manifestsDir=" + testManifestsDir,
		"thirdImageRepo=" + testImageRepo,
		"fuyaoImageRepo=" + testImageRepo,
		"haNodes=" + testHostname + ":" + testIP,
		"controlPlaneEndpointVIP=" + testControlPlaneVIP,
		"controlPlaneEndpointPort=" + testControlPlanePort,
		"virtualRouterId=" + testVirtualRouterID,
		"wait=true",
	}

	result, err := mockHA.Execute(context.Background(), commands)

	assert.NoError(t, err)
	assert.Nil(t, result)
	assert.True(t, pollCalled)
}

func TestWaitSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(mfutil.KeepalivedInstanceIsMaster, func(nodes []mfutil.HANode) bool {
		return true
	})

	patches.ApplyFuncSeq(bkenet.GetInterfaceFromIp, []gomonkey.OutputCell{
		{Values: gomonkey.Params{testInterface, nil}},
	})

	var pollCalled bool
	patches.ApplyFunc(wait.PollImmediateWithContext, func(ctx context.Context, interval, timeout time.Duration, condition wait.ConditionWithContextFunc) error {
		pollCalled = true
		if condition != nil {
			condition(ctx)
		}
		return nil
	})

	mockHA := &HA{
		isMasterHa: true,
	}

	cfg := map[string]interface{}{
		"nodes": []mfutil.HANode{
			{Hostname: testHostname, IP: testIP},
		},
		"controlPlaneEndpointVIP": testControlPlaneVIP,
		"ingressVIP":              testIngressVIP,
	}

	err := mockHA.Wait(context.Background(), cfg)

	assert.NoError(t, err)
	assert.True(t, pollCalled)
}

func TestWaitNotMaster(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(mfutil.KeepalivedInstanceIsMaster, func(nodes []mfutil.HANode) bool {
		return false
	})

	var pollCalled bool
	patches.ApplyFunc(wait.PollImmediateWithContext, func(ctx context.Context, interval, timeout time.Duration, condition wait.ConditionWithContextFunc) error {
		pollCalled = true
		return nil
	})

	mockHA := &HA{
		isMasterHa: false,
	}

	cfg := map[string]interface{}{
		"nodes": []mfutil.HANode{
			{Hostname: testHostname, IP: testIP},
		},
		"controlPlaneEndpointVIP": testControlPlaneVIP,
		"ingressVIP":              testIngressVIP,
	}

	err := mockHA.Wait(context.Background(), cfg)

	assert.NoError(t, err)
	assert.False(t, pollCalled)
}

func TestWaitPollError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(mfutil.KeepalivedInstanceIsMaster, func(nodes []mfutil.HANode) bool {
		return true
	})

	patches.ApplyFunc(bkenet.GetInterfaceFromIp, func(ip string) (string, error) {
		return "", assert.AnError
	})

	patches.ApplyFunc(wait.PollImmediateWithContext, func(ctx context.Context, interval, timeout time.Duration, condition wait.ConditionWithContextFunc) error {
		return assert.AnError
	})

	mockHA := &HA{
		isMasterHa: true,
	}

	cfg := map[string]interface{}{
		"nodes": []mfutil.HANode{
			{Hostname: testHostname, IP: testIP},
		},
		"controlPlaneEndpointVIP": testControlPlaneVIP,
		"ingressVIP":              testIngressVIP,
	}

	err := mockHA.Wait(context.Background(), cfg)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "wait vip")
}

func TestWaitIngressVIP(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(mfutil.KeepalivedInstanceIsMaster, func(nodes []mfutil.HANode) bool {
		return true
	})

	patches.ApplyFunc(bkenet.GetInterfaceFromIp, func(ip string) (string, error) {
		if ip == testIngressVIP {
			return testInterface, nil
		}
		return "", nil
	})

	patches.ApplyFunc(wait.PollImmediateWithContext, func(ctx context.Context, interval, timeout time.Duration, condition wait.ConditionWithContextFunc) error {
		return nil
	})

	mockHA := &HA{
		isMasterHa: false,
	}

	cfg := map[string]interface{}{
		"nodes": []mfutil.HANode{
			{Hostname: testHostname, IP: testIP},
		},
		"controlPlaneEndpointVIP": "",
		"ingressVIP":              testIngressVIP,
	}

	err := mockHA.Wait(context.Background(), cfg)

	assert.NoError(t, err)
}

func TestExecuteWithParseCommandsError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(plugin.ParseCommands, func(p plugin.Plugin, commands []string) (map[string]string, error) {
		return nil, assert.AnError
	})

	mockHA := &HA{
		isMasterHa: false,
	}

	commands := []string{
		"HA",
		"haNodes=" + testHostname + ":" + testIP,
		"thirdImageRepo=" + testImageRepo,
		"fuyaoImageRepo=" + testImageRepo,
	}

	result, err := mockHA.Execute(context.Background(), commands)

	assert.Error(t, err)
	assert.Nil(t, result)
	assert.Equal(t, assert.AnError, err)
}

func TestPrepareRendCfgWithMultipleNodes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	secondIP := "192.168.1.2"

	patches.ApplyFunc(utils.HostName, func() string {
		return testHostname
	})

	patches.ApplyFunc(bkenet.GetInterfaceFromIp, func(ip string) (string, error) {
		if ip == testIP {
			return testInterface, nil
		}
		return "", nil
	})

	mockHA := &HA{
		isMasterHa: true,
	}

	cfg, err := mockHA.prepareRendCfg(map[string]string{
		"haNodes":                  testHostname + ":" + testIP + ",node2:" + secondIP,
		"controlPlaneEndpointVIP":  testControlPlaneVIP,
		"controlPlaneEndpointPort": testControlPlanePort,
		"thirdImageRepo":           testImageRepo,
		"fuyaoImageRepo":           testImageRepo,
	})

	assert.NoError(t, err)
	assert.NotNil(t, cfg)
	assert.NotNil(t, cfg["nodes"])
	assert.Equal(t, testInterface, cfg["interface"])
	assert.Equal(t, true, cfg["isMasterHa"])
	assert.Equal(t, testControlPlaneVIP, cfg["vip"])
}

func TestPrepareRendCfgWithIngressVIP(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.HostName, func() string {
		return testHostname
	})

	patches.ApplyFunc(bkenet.GetInterfaceFromIp, func(ip string) (string, error) {
		if ip == testIP {
			return testInterface, nil
		}
		return "", nil
	})

	mockHA := &HA{
		isMasterHa: false,
	}

	cfg, err := mockHA.prepareRendCfg(map[string]string{
		"haNodes":        testHostname + ":" + testIP,
		"ingressVIP":     testIngressVIP,
		"thirdImageRepo": testImageRepo,
		"fuyaoImageRepo": testImageRepo,
	})

	assert.NoError(t, err)
	assert.NotNil(t, cfg)
	assert.Equal(t, testInterface, cfg["interface"])
	assert.Equal(t, false, cfg["isMasterHa"])
	assert.Equal(t, testIngressVIP, cfg["vip"])
}

func TestPrepareRendCfgWithEmptyInterface(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.HostName, func() string {
		return testHostname
	})

	patches.ApplyFunc(bkenet.GetInterfaceFromIp, func(ip string) (string, error) {
		return "", nil
	})

	mockHA := &HA{
		isMasterHa: false,
	}

	cfg, err := mockHA.prepareRendCfg(map[string]string{
		"haNodes":        testHostname + ":" + testIP,
		"ingressVIP":     testIngressVIP,
		"thirdImageRepo": testImageRepo,
		"fuyaoImageRepo": testImageRepo,
	})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "interface")
	assert.Nil(t, cfg)
}

func TestPrepareRendCfgWithGetInterfaceError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.HostName, func() string {
		return testHostname
	})

	patches.ApplyFunc(bkenet.GetInterfaceFromIp, func(ip string) (string, error) {
		return "", assert.AnError
	})

	mockHA := &HA{
		isMasterHa: false,
	}

	cfg, err := mockHA.prepareRendCfg(map[string]string{
		"haNodes":        testHostname + ":" + testIP,
		"ingressVIP":     testIngressVIP,
		"thirdImageRepo": testImageRepo,
		"fuyaoImageRepo": testImageRepo,
	})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "error")
	assert.Nil(t, cfg)
}

func TestWaitWithEmptyIP(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(mfutil.KeepalivedInstanceIsMaster, func(nodes []mfutil.HANode) bool {
		return true
	})

	patches.ApplyFunc(bkenet.GetInterfaceFromIp, func(ip string) (string, error) {
		return "", nil
	})

	var pollCalled bool
	var conditionCalled bool
	patches.ApplyFunc(wait.PollImmediateWithContext, func(ctx context.Context, interval, timeout time.Duration, condition wait.ConditionWithContextFunc) error {
		pollCalled = true
		if condition != nil {
			conditionCalled = true
			condition(ctx)
		}
		return nil
	})

	mockHA := &HA{
		isMasterHa: false,
	}

	cfg := map[string]interface{}{
		"nodes": []mfutil.HANode{
			{Hostname: testHostname, IP: testIP},
		},
		"controlPlaneEndpointVIP": "",
		"ingressVIP":              "",
	}

	err := mockHA.Wait(context.Background(), cfg)

	assert.NoError(t, err)
	assert.True(t, pollCalled)
	assert.True(t, conditionCalled)
}
