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

package plugin

import (
	"context"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/pkg/errors"
	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	v1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/clientutil"
)

const (
	numOne        = 1
	numTwo        = 2
	numThree      = 3
	numFive       = 5
	numTen        = 10
	numSixty      = 60
	numOneHundred = 100

	testNamespace = "test-namespace"
	testName      = "test-name"
)

type mockPlugin struct {
	nameValue   string
	paramsValue map[string]PluginParam
}

func (m *mockPlugin) Name() string {
	return m.nameValue
}

func (m *mockPlugin) Param() map[string]PluginParam {
	return m.paramsValue
}

func (m *mockPlugin) Execute(ctx context.Context, commands []string) ([]string, error) {
	return nil, nil
}

func TestParseCommands(t *testing.T) {
	mockP := &mockPlugin{
		nameValue: "TestPlugin",
		paramsValue: map[string]PluginParam{
			"phase": {
				Key:         "phase",
				Value:       "init,join",
				Required:    true,
				Default:     "init",
				Description: "phase to execute",
			},
			"optional": {
				Key:         "optional",
				Value:       "true,false",
				Required:    false,
				Default:     "false",
				Description: "optional param",
			},
		},
	}

	result, err := ParseCommands(mockP, []string{"TestPlugin", "phase=join"})

	assert.NoError(t, err)
	assert.Equal(t, "join", result["phase"])
	assert.Equal(t, "false", result["optional"])
}

func TestParseCommandsMissingRequired(t *testing.T) {
	mockP := &mockPlugin{
		nameValue: "TestPlugin",
		paramsValue: map[string]PluginParam{
			"requiredParam": {
				Key:         "requiredParam",
				Value:       "",
				Required:    true,
				Default:     "",
				Description: "required param",
			},
		},
	}

	result, err := ParseCommands(mockP, []string{"TestPlugin"})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "Missing required parameters")
	assert.Empty(t, result)
}

func TestParseCommandsWithDefaultValue(t *testing.T) {
	mockP := &mockPlugin{
		nameValue: "TestPlugin",
		paramsValue: map[string]PluginParam{
			"optional": {
				Key:         "optional",
				Value:       "a,b,c",
				Required:    false,
				Default:     "b",
				Description: "optional param",
			},
		},
	}

	result, err := ParseCommands(mockP, []string{"TestPlugin"})

	assert.NoError(t, err)
	assert.Equal(t, "b", result["optional"])
}

func TestParseCommandsInvalidFormat(t *testing.T) {
	mockP := &mockPlugin{
		nameValue:   "TestPlugin",
		paramsValue: map[string]PluginParam{},
	}

	result, err := ParseCommands(mockP, []string{"TestPlugin", "invalid-format"})

	assert.NoError(t, err)
	assert.Empty(t, result)
}

func TestParseCommandsEmptyCommands(t *testing.T) {
	mockP := &mockPlugin{
		nameValue:   "TestPlugin",
		paramsValue: map[string]PluginParam{},
	}

	result, err := ParseCommands(mockP, []string{"TestPlugin"})

	assert.NoError(t, err)
	assert.Empty(t, result)
}

func TestParseCommandsMultipleParams(t *testing.T) {
	mockP := &mockPlugin{
		nameValue: "TestPlugin",
		paramsValue: map[string]PluginParam{
			"param1": {Key: "param1", Value: "", Required: true, Default: "", Description: "param1"},
			"param2": {Key: "param2", Value: "", Required: false, Default: "default2", Description: "param2"},
			"param3": {Key: "param3", Value: "", Required: false, Default: "default3", Description: "param3"},
		},
	}

	result, err := ParseCommands(mockP, []string{"TestPlugin", "param1=value1", "param2=value2"})

	assert.NoError(t, err)
	assert.Equal(t, "value1", result["param1"])
	assert.Equal(t, "value2", result["param2"])
	assert.Equal(t, "default3", result["param3"])
}

func TestParseCommandsExternalParamOverride(t *testing.T) {
	mockP := &mockPlugin{
		nameValue: "TestPlugin",
		paramsValue: map[string]PluginParam{
			"param": {
				Key:         "param",
				Value:       "a,b",
				Required:    false,
				Default:     "default",
				Description: "param",
			},
		},
	}

	result, err := ParseCommands(mockP, []string{"TestPlugin", "param=override"})

	assert.NoError(t, err)
	assert.Equal(t, "override", result["param"])
}

func TestParseCommandsOnlyRequiredProvided(t *testing.T) {
	mockP := &mockPlugin{
		nameValue: "TestPlugin",
		paramsValue: map[string]PluginParam{
			"required1": {Key: "required1", Value: "", Required: true, Default: "", Description: "required1"},
			"required2": {Key: "required2", Value: "", Required: true, Default: "", Description: "required2"},
			"optional1": {Key: "optional1", Value: "", Required: false, Default: "opt1", Description: "optional1"},
		},
	}

	result, err := ParseCommands(mockP, []string{"TestPlugin", "required1=val1", "required2=val2"})

	assert.NoError(t, err)
	assert.Equal(t, "val1", result["required1"])
	assert.Equal(t, "val2", result["required2"])
	assert.Equal(t, "opt1", result["optional1"])
}

func TestLogDebugInfo(t *testing.T) {
	parseCommand := map[string]string{
		"key1": "value1",
		"key2": "value2",
	}

	LogDebugInfo(parseCommand, "TestPlugin")
}

func TestPluginParamStruct(t *testing.T) {
	param := PluginParam{
		Key:         "test-key",
		Value:       "test-value",
		Required:    true,
		Default:     "default",
		Description: "test description",
	}

	assert.Equal(t, "test-key", param.Key)
	assert.Equal(t, "test-value", param.Value)
	assert.True(t, param.Required)
	assert.Equal(t, "default", param.Default)
	assert.Equal(t, "test description", param.Description)
}

func TestGetBKECluster(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := &v1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testName,
			Namespace: testNamespace,
		},
		Spec: v1beta1.BKEClusterSpec{
			ClusterConfig: &v1beta1.BKEConfig{
				Cluster: v1beta1.Cluster{
					KubernetesVersion: "v1.26.0",
					Networking: v1beta1.Networking{
						PodSubnet:     "10.244.0.0/16",
						ServiceSubnet: "10.96.0.0/12",
					},
				},
			},
		},
	}

	patches.ApplyFunc(GetBKECluster,
		func(string) (*v1beta1.BKECluster, error) {
			return bkeCluster, nil
		})

	result, err := GetBKECluster(testNamespace + ":" + testName)

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, testName, result.GetName())
	assert.Equal(t, testNamespace, result.GetNamespace())
}

func TestGetBKEClusterWithNodes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := &v1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testName,
			Namespace: testNamespace,
		},
		Spec: v1beta1.BKEClusterSpec{
			ClusterConfig: &v1beta1.BKEConfig{
				Cluster: v1beta1.Cluster{
					KubernetesVersion: "v1.26.0",
					Networking: v1beta1.Networking{
						PodSubnet:     "10.244.0.0/16",
						ServiceSubnet: "10.96.0.0/12",
					},
				},
			},
		},
	}

	patches.ApplyFunc(GetBKECluster,
		func(string) (*v1beta1.BKECluster, error) {
			return bkeCluster, nil
		})

	result, err := GetBKECluster(testNamespace + ":" + testName)

	assert.NoError(t, err)
	assert.NotNil(t, result)
}

func TestGetBKEClusterNotFound(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(GetBKECluster,
		func(string) (*v1beta1.BKECluster, error) {
			return nil, errors.New("not found")
		})

	result, err := GetBKECluster(testNamespace + ":" + testName)

	assert.Error(t, err)
	assert.Nil(t, result)
}

func TestGetBKEClusterInvalidNamespaceFormat(t *testing.T) {
	result, err := GetBKECluster("invalid-format")

	assert.Error(t, err)
	assert.Nil(t, result)
}

func TestGetContainerdConfig(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	containerdConfig := &v1beta1.ContainerdConfigSpec{}

	patches.ApplyFunc(GetContainerdConfig,
		func(string) (*v1beta1.ContainerdConfigSpec, error) {
			return containerdConfig, nil
		})

	result, err := GetContainerdConfig(testNamespace + ":" + testName)

	assert.NoError(t, err)
	assert.NotNil(t, result)
}

func TestGetContainerdConfigNotFound(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(GetContainerdConfig,
		func(string) (*v1beta1.ContainerdConfigSpec, error) {
			return nil, errors.New("not found")
		})

	result, err := GetContainerdConfig(testNamespace + ":" + testName)

	assert.Error(t, err)
	assert.Nil(t, result)
}

func TestGetContainerdConfigInvalidNamespaceFormat(t *testing.T) {
	result, err := GetContainerdConfig("invalid-format")

	assert.Error(t, err)
	assert.Nil(t, result)
}

func TestGetBkeConfigFromBkeCluster(t *testing.T) {
	bkeCluster := &v1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testName,
			Namespace: testNamespace,
		},
		Spec: v1beta1.BKEClusterSpec{
			ClusterConfig: &v1beta1.BKEConfig{
				Cluster: v1beta1.Cluster{
					KubernetesVersion: "v1.26.0",
					Networking: v1beta1.Networking{
						PodSubnet:     "10.244.0.0/16",
						ServiceSubnet: "10.96.0.0/12",
					},
				},
			},
		},
	}

	result, err := GetBkeConfigFromBkeCluster(bkeCluster)

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, "v1.26.0", result.Cluster.KubernetesVersion)
}

func TestGetBkeConfigFromBkeClusterWithBocloudAnnotation(t *testing.T) {
	bkeCluster := &v1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testName,
			Namespace: testNamespace,
			Annotations: map[string]string{
				common.BKEClusterFromAnnotationKey: common.BKEClusterFromAnnotationValueBocloud,
			},
		},
		Spec: v1beta1.BKEClusterSpec{
			ClusterConfig: &v1beta1.BKEConfig{
				Cluster: v1beta1.Cluster{
					Networking: v1beta1.Networking{
						PodSubnet:     "10.244.0.0/16",
						ServiceSubnet: "10.96.0.0/12",
						DNSDomain:     "cluster.local",
					},
					HTTPRepo: v1beta1.Repo{
						Domain: "repo.example.com",
						Port:   "443",
					},
					ImageRepo: v1beta1.Repo{
						Domain: "registry.example.com",
						Port:   "443",
					},
				},
			},
		},
	}

	result, err := GetBkeConfigFromBkeCluster(bkeCluster)

	assert.NoError(t, err)
	assert.NotNil(t, result)
}

func TestGetBkeConfigFromBkeClusterInvalidBKEConfig(t *testing.T) {
	bkeCluster := &v1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testName,
			Namespace: testNamespace,
			Annotations: map[string]string{
				common.BKEClusterFromAnnotationKey: common.BKEClusterFromAnnotationValueBKE,
			},
		},
		Spec: v1beta1.BKEClusterSpec{
			ClusterConfig: &v1beta1.BKEConfig{
				Cluster: v1beta1.Cluster{
					Networking: v1beta1.Networking{
						PodSubnet:     "",
						ServiceSubnet: "",
						DNSDomain:     "",
					},
				},
			},
		},
	}

	_, err := GetBkeConfigFromBkeCluster(bkeCluster)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "is invalid")
}

func TestGetBkeConfigFromBkeClusterWithBocloudAnnotationInvalid(t *testing.T) {
	bkeCluster := &v1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testName,
			Namespace: testNamespace,
			Annotations: map[string]string{
				common.BKEClusterFromAnnotationKey: common.BKEClusterFromAnnotationValueBocloud,
			},
		},
		Spec: v1beta1.BKEClusterSpec{
			ClusterConfig: &v1beta1.BKEConfig{
				Cluster: v1beta1.Cluster{
					Networking: v1beta1.Networking{
						PodSubnet:     "",
						ServiceSubnet: "",
						DNSDomain:     "",
					},
				},
			},
		},
	}

	_, err := GetBkeConfigFromBkeCluster(bkeCluster)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "is invalid")
}

func TestGetBkeConfigFromBkeClusterWithEmptyAnnotation(t *testing.T) {
	bkeCluster := &v1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testName,
			Namespace: testNamespace,
			Annotations: map[string]string{
				common.BKEClusterFromAnnotationKey: "",
			},
		},
		Spec: v1beta1.BKEClusterSpec{
			ClusterConfig: &v1beta1.BKEConfig{
				Cluster: v1beta1.Cluster{
					KubernetesVersion: "v1.26.0",
					Networking: v1beta1.Networking{
						PodSubnet:     "10.244.0.0/16",
						ServiceSubnet: "10.96.0.0/12",
						DNSDomain:     "cluster.local",
					},
				},
			},
		},
	}

	result, err := GetBkeConfigFromBkeCluster(bkeCluster)

	assert.NoError(t, err)
	assert.NotNil(t, result)
}

func TestGetBkeConfigFromBkeClusterNilAnnotations(t *testing.T) {
	bkeCluster := &v1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:        testName,
			Namespace:   testNamespace,
			Annotations: nil,
		},
		Spec: v1beta1.BKEClusterSpec{
			ClusterConfig: &v1beta1.BKEConfig{
				Cluster: v1beta1.Cluster{
					KubernetesVersion: "v1.26.0",
					Networking: v1beta1.Networking{
						PodSubnet:     "10.244.0.0/16",
						ServiceSubnet: "10.96.0.0/12",
					},
				},
			},
		},
	}

	result, err := GetBkeConfigFromBkeCluster(bkeCluster)

	assert.NoError(t, err)
	assert.NotNil(t, result)
}

func TestGetBkeConfig(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := &v1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testName,
			Namespace: testNamespace,
		},
		Spec: v1beta1.BKEClusterSpec{
			ClusterConfig: &v1beta1.BKEConfig{
				Cluster: v1beta1.Cluster{
					KubernetesVersion: "v1.26.0",
					Networking: v1beta1.Networking{
						PodSubnet:     "10.244.0.0/16",
						ServiceSubnet: "10.96.0.0/12",
					},
				},
			},
		},
	}

	patches.ApplyFunc(GetBKECluster,
		func(string) (*v1beta1.BKECluster, error) {
			return bkeCluster, nil
		})

	result, err := GetBkeConfig(testNamespace + ":" + testName)

	assert.NoError(t, err)
	assert.NotNil(t, result)
}

func TestGetBkeConfigError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(GetBKECluster,
		func(string) (*v1beta1.BKECluster, error) {
			return nil, errors.New("get cluster error")
		})

	result, err := GetBkeConfig(testNamespace + ":" + testName)

	assert.Error(t, err)
	assert.Nil(t, result)
}

func TestParseCommandsWithOnlyEqualsSign(t *testing.T) {
	mockP := &mockPlugin{
		nameValue: "TestPlugin",
		paramsValue: map[string]PluginParam{
			"param": {
				Key:         "param",
				Value:       "",
				Required:    false,
				Default:     "default",
				Description: "param",
			},
		},
	}

	result, err := ParseCommands(mockP, []string{"TestPlugin", "param="})

	assert.NoError(t, err)
	assert.Equal(t, "", result["param"])
}

func TestParseCommandsDuplicateParams(t *testing.T) {
	mockP := &mockPlugin{
		nameValue: "TestPlugin",
		paramsValue: map[string]PluginParam{
			"param": {
				Key:         "param",
				Value:       "",
				Required:    false,
				Default:     "default",
				Description: "param",
			},
		},
	}

	result, err := ParseCommands(mockP, []string{"TestPlugin", "param=first", "param=second"})

	assert.NoError(t, err)
	assert.Equal(t, "second", result["param"])
}

func TestGetNodesDataFromNs(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(GetNodesDataFromNs,
		func(string, string) (bkenode.Nodes, error) {
			return bkenode.Nodes{
				{IP: "192.168.1.10", Role: []string{"master"}},
			}, nil
		})

	result, err := GetNodesDataFromNs(testNamespace, testName)

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Len(t, result, 1)
}

func TestGetNodesDataFromNsInvalidNamespace(t *testing.T) {
	result, err := GetNodesDataFromNs("", testName)

	assert.Error(t, err)
	assert.Nil(t, result)
}

func TestGetNodesDataFromNsInvalidName(t *testing.T) {
	result, err := GetNodesDataFromNs(testNamespace, "")

	assert.Error(t, err)
	assert.Nil(t, result)
}

func TestGetNodesData(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(GetClusterData,
		func(string) (*ClusterData, error) {
			return &ClusterData{
				Nodes: bkenode.Nodes{
					{IP: "192.168.1.10", Role: []string{"master"}},
					{IP: "192.168.1.11", Role: []string{"worker"}},
				},
			}, nil
		})

	result, err := GetNodesData(testNamespace + ":" + testName)

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Len(t, result, 2)
}

func TestGetNodesDataError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(GetClusterData,
		func(string) (*ClusterData, error) {
			return nil, errors.New("get cluster data error")
		})

	result, err := GetNodesData(testNamespace + ":" + testName)

	assert.Error(t, err)
	assert.Nil(t, result)
}

func TestGetClusterData(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(GetClusterData,
		func(string) (*ClusterData, error) {
			return &ClusterData{
				Cluster: &v1beta1.BKECluster{
					ObjectMeta: metav1.ObjectMeta{
						Name:      testName,
						Namespace: testNamespace,
					},
				},
				Nodes: bkenode.Nodes{
					{IP: "192.168.1.10", Role: []string{"master"}},
				},
			}, nil
		})

	result, err := GetClusterData(testNamespace + ":" + testName)

	assert.NoError(t, err)
	assert.NotNil(t, result)
}

func TestGetClusterDataNewClientError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(clientutil.NewKubernetesClient,
		func(string) (*clientutil.Client, error) {
			return nil, errors.New("new client error")
		})

	result, err := GetClusterData(testNamespace + ":" + testName)

	assert.Error(t, err)
	assert.Nil(t, result)
}

func TestGetClusterDataGetBKEClusterError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(GetBKEClusterFromClient,
		func(*clientutil.Client, string) (*v1beta1.BKECluster, error) {
			return nil, errors.New("get cluster error")
		})

	patches.ApplyFunc(clientutil.NewKubernetesClient,
		func(string) (*clientutil.Client, error) {
			return &clientutil.Client{}, nil
		})

	result, err := GetClusterData(testNamespace + ":" + testName)

	assert.Error(t, err)
	assert.Nil(t, result)
}

func TestGetBKEClusterFromClientInvalidNamespace(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	client := &clientutil.Client{}

	patches.ApplyFunc(utils.SplitNameSpaceName,
		func(string) (string, string, error) {
			return "", "", errors.New("invalid namespace")
		})

	result, err := GetBKEClusterFromClient(client, "invalid")

	assert.Error(t, err)
	assert.Nil(t, result)
}
