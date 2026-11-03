/******************************************************************
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package config

import (
	"os"
	"path/filepath"

	"installer/pkg/common"
	"installer/pkg/constants"
	"installer/pkg/fuyao/v1beta1/base"
	fuyaov1beta1 "installer/pkg/fuyao/v1beta1/cluster"
	"installer/pkg/fuyao/v1beta1/manifests"
	"installer/pkg/fuyao/v1beta1/values"
	"installer/pkg/zlog"
)

// ClusterConfigRuntime is generate cluster cfg runtime
type ClusterConfigRuntime struct {
	AllParam      bool
	Ha            bool
	AdvancedParam bool
	SavePath      string
}

const (
	// ExampleCluster is metadata name
	ExampleCluster   = "example-cluster"
	exampleHaCluster = "example-ha-cluster"
	// YamlName is file name
	YamlName = "example-cluster.yaml"
)

// ReadComponentManifests 读取组件清单
func ReadComponentManifests() (*manifests.ComponentManifests, error) {
	absPath, err := common.GetAssetAbsPath()
	if err != nil {
		return nil, err
	}
	componentManifests := &manifests.ComponentManifests{}
	manifestPath := filepath.Join(absPath, constants.ComponentManifest)
	if err := common.UnmarshalYaml[*manifests.ComponentManifests](componentManifests, manifestPath); err != nil {
		return nil, err
	}
	return componentManifests, nil
}

// getAdvancedParams 获取高级参数
func (c *ClusterConfigRuntime) getAdvancedParams() (*values.Values, error) {
	pkgPath, pkgName, err := common.UnCompressOnePackage()
	if err != nil {
		return nil, err
	}
	pkgPath = filepath.Join(pkgPath, pkgName)

	advancedParam := &values.Values{}
	yamPath := filepath.Join(pkgPath, constants.Config, constants.AdvancedParamsFile)
	if err := common.UnmarshalYaml[*values.Values](advancedParam, yamPath); err != nil {
		return nil, err
	}
	return advancedParam, nil
}

// getAllComponentsAndValues 获取所有组件与组件参数
func (c *ClusterConfigRuntime) getAllComponentsAndValues() ([]fuyaov1beta1.Component, error) {
	// 获取组件清单
	componentsManifests, err := ReadComponentManifests()
	if err != nil {
		return nil, err
	}

	pkgPath, pkgName, err := common.UnCompressOnePackage()
	if err != nil {
		return nil, err
	}
	pkgPath = filepath.Join(pkgPath, pkgName)

	allComponents := make([]fuyaov1beta1.Component, 0)

	// 组件分组
	for group, groupComponents := range componentsManifests.Spec.Components {

		groupPath := filepath.Join(pkgPath, group)
		// 分组内的具体组件
		for _, componentInfo := range groupComponents {
			defaultValues, err := c.getComponentsAndValues(groupPath, componentInfo)
			if err != nil {
				return nil, err
			}

			component := fuyaov1beta1.Component{
				Name:  componentInfo.Name,
				Param: defaultValues.Spec.Params,
			}
			allComponents = append(allComponents, component)
		}

	}

	return allComponents, nil
}

func (c *ClusterConfigRuntime) getComponentsAndValues(groupPath string,
	component manifests.Component) (*values.Values, error) {
	defaultValues := &values.Values{}
	componentPath := filepath.Join(groupPath, component.Name)
	yamPath := filepath.Join(componentPath, constants.DefaultValues)
	_, err := os.Stat(yamPath)
	if err == nil {
		if err := common.UnmarshalYaml[*values.Values](defaultValues, yamPath); err != nil {
			return nil, err
		}
	} else if os.IsNotExist(err) {
		zlog.Info("not find default-values.yaml, the component is ", component.Name)
	} else {
		return nil, err
	}
	return defaultValues, nil
}

func (c *ClusterConfigRuntime) getHaComponentsAndParams() ([]fuyaov1beta1.Component, error) {
	// 获取组件清单
	componentsManifests, err := ReadComponentManifests()
	if err != nil {
		return nil, err
	}

	pkgPath, pkgName, err := common.UnCompressOnePackage()
	if err != nil {
		return nil, err
	}
	pkgPath = filepath.Join(pkgPath, pkgName)

	HaComponents := make([]fuyaov1beta1.Component, 0)

	// 组件分组
	for group, groupComponents := range componentsManifests.Spec.Components {

		if group != constants.Ha {
			continue
		}

		groupPath := filepath.Join(pkgPath, group)
		// 分组内的具体组件
		for _, componentInfo := range groupComponents {
			componentPath := filepath.Join(groupPath, componentInfo.Name)

			defaultValues := &values.Values{}
			yamPath := filepath.Join(componentPath, constants.DefaultValues)
			if err := common.UnmarshalYaml[*values.Values](defaultValues, yamPath); err != nil {
				return nil, err
			}

			component := fuyaov1beta1.Component{
				Name:  componentInfo.Name,
				Param: defaultValues.Spec.Params,
			}
			HaComponents = append(HaComponents, component)
		}

	}

	return HaComponents, nil
}

// CreateAllParamCfgFile 获取全量参数配置文件
// 此配置文件包含显示安装的全量组件、组件参数、高级参数
func (c *ClusterConfigRuntime) CreateAllParamCfgFile() error {
	allComponents, err := c.getAllComponentsAndValues()
	if err != nil {
		return err
	}

	advancedParams, err := c.getAdvancedParams()
	if err != nil {
		return err
	}

	allParamYaml := fuyaov1beta1.NewCluster()
	allParamYaml.Metadata = base.Metadata{Name: exampleHaCluster}
	allParamYaml.Spec = fuyaov1beta1.Spec{
		Nodes: fuyaov1beta1.Nodes{
			Masters: getMasterNodes(),
			Workers: getWorkerNode(),
		},
		Components:     allComponents,
		AdvancedParams: advancedParams.Spec.Params,
	}

	return common.MarshalYaml[fuyaov1beta1.Cluster](filepath.Join(c.SavePath, YamlName), allParamYaml)
}

func (c *ClusterConfigRuntime) CreateHaCfgFile() error {
	components, err := c.getHaComponentsAndParams()
	if err != nil {
		return err
	}

	haYaml := fuyaov1beta1.NewCluster()
	haYaml.Metadata = base.Metadata{Name: exampleHaCluster}
	haYaml.Spec = fuyaov1beta1.Spec{
		Nodes: fuyaov1beta1.Nodes{
			Masters: getMasterNodes(),
			Workers: getWorkerNode(),
		},
		Components: components,
	}

	return common.MarshalYaml[fuyaov1beta1.Cluster](filepath.Join(c.SavePath, YamlName), haYaml)
}

func (c *ClusterConfigRuntime) CreateHaCfgFileWithAdvanced() error {
	advancedParams, err := c.getAdvancedParams()
	if err != nil {
		return err
	}

	haComponents, err := c.getHaComponentsAndParams()
	if err != nil {
		return err
	}

	haWithAdvancedYaml := fuyaov1beta1.NewCluster()
	haWithAdvancedYaml.Metadata = base.Metadata{Name: exampleHaCluster}
	haWithAdvancedYaml.Spec = fuyaov1beta1.Spec{
		Nodes: fuyaov1beta1.Nodes{
			Masters: getMasterNodes(),
			Workers: getWorkerNode(),
		},
		Components:     haComponents,
		AdvancedParams: advancedParams.Spec.Params,
	}

	return common.MarshalYaml[fuyaov1beta1.Cluster](filepath.Join(c.SavePath, YamlName), haWithAdvancedYaml)
}

func (c *ClusterConfigRuntime) CreateSimplestCfgFileWithAdvanced() error {
	advancedParams, err := c.getAdvancedParams()
	if err != nil {
		return err
	}

	simpleWithAdvancedYaml := fuyaov1beta1.NewCluster()
	simpleWithAdvancedYaml.Metadata = base.Metadata{Name: ExampleCluster}
	simpleWithAdvancedYaml.Spec = fuyaov1beta1.Spec{
		Nodes: fuyaov1beta1.Nodes{
			Masters: getMasterNode(),
			Workers: getWorkerNode(),
		},
		AdvancedParams: advancedParams.Spec.Params,
	}

	return common.MarshalYaml[fuyaov1beta1.Cluster](filepath.Join(c.SavePath, YamlName), simpleWithAdvancedYaml)
}

// CreateSimplestCfgFile 创建最简集群配置文件
func (c *ClusterConfigRuntime) CreateSimplestCfgFile() error {
	simpleYaml := fuyaov1beta1.NewCluster()
	simpleYaml.Metadata = base.Metadata{Name: ExampleCluster}
	simpleYaml.Spec = fuyaov1beta1.Spec{
		Nodes: fuyaov1beta1.Nodes{
			Masters: getMasterNode(),
			Workers: getWorkerNode(),
		},
	}

	return common.MarshalYaml[fuyaov1beta1.Cluster](filepath.Join(c.SavePath, YamlName), simpleYaml)
}

func getLabel() []fuyaov1beta1.Labels {
	label := fuyaov1beta1.Labels{
		Name:  "label-name",
		Value: "label-value",
	}

	labels := make([]fuyaov1beta1.Labels, 0)
	labels = append(labels, label)
	return labels
}

func masterNode() *fuyaov1beta1.Node {
	node := &fuyaov1beta1.Node{
		Hostname:       "fuyao-master",
		User:           "root",
		IP:             "o.x.y.z",
		Port:           22,
		PrivateKeyPath: "/root/.ssh/id_rsa",
		RemoveTaint:    true,
		Labels:         getLabel(),
	}
	return node
}

func getMasterNode() []*fuyaov1beta1.Node {
	masters := make([]*fuyaov1beta1.Node, 0)
	masters = append(masters, masterNode())
	return masters
}

func getMasterNodes() []*fuyaov1beta1.Node {
	masters := getMasterNode()
	masters = append(masters, masterNode())
	masters = append(masters, masterNode())
	return masters
}

func getWorkerNode() []*fuyaov1beta1.Node {
	workerNode := &fuyaov1beta1.Node{
		Hostname:       "fuyao-worker",
		User:           "root",
		IP:             "o.x.y.z",
		Port:           22,
		PrivateKeyPath: "/root/.ssh/id_rsa",
		Labels:         getLabel(),
	}

	workers := make([]*fuyaov1beta1.Node, 0)
	workers = append(workers, workerNode)

	return workers
}
