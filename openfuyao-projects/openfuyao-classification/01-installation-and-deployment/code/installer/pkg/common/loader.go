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

package common

import (
	"fmt"
	"os"
	"os/user"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"

	"installer/pkg/constants"
	"installer/pkg/fuyao/v1beta1/base"
	fuyaov1beta1 "installer/pkg/fuyao/v1beta1/cluster"
	"installer/pkg/fuyao/v1beta1/manifests"
	"installer/pkg/fuyao/v1beta1/metadata"
	"installer/pkg/fuyao/v1beta1/values"
	"installer/pkg/utils"
	"installer/pkg/zlog"
)

const (
	// 添加ha组件时master节点数量最小值
	masterNodeCount = 2
)

// Loader is load config file
type Loader interface {
	Load() (*fuyaov1beta1.Cluster, error)
}

// NewLoader is new Loader
func NewLoader(arg Argument) Loader {
	// 镜像仓
	if len(arg.RegistryPasswd) != 0 {
		return NewRegistryLoader(arg)
	}

	// 集群
	var loaderType string
	if arg.FilePath != "" {
		loaderType = File
	} else {
		loaderType = AllInOne
	}

	switch loaderType {
	case File:
		return NewFileLoader(arg)
	case AllInOne:
		return NewDefaultLoader(arg)
	default:
		return NewDefaultLoader(arg)
	}
}

// FileLoader 从文件获取集群配置信息
type FileLoader struct {
	arg      Argument
	FilePath string
}

// NewFileLoader is new FileLoader
func NewFileLoader(arg Argument) *FileLoader {
	return &FileLoader{
		arg:      arg,
		FilePath: arg.FilePath,
	}
}

// Load is load the config file
func (f *FileLoader) Load() (*fuyaov1beta1.Cluster, error) {
	zlog.Info("start load multinode config file")
	fp, err := filepath.Abs(f.FilePath)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to look up current directory")
	}
	clusterCfg := &fuyaov1beta1.Cluster{}
	if err := UnmarshalYaml[*fuyaov1beta1.Cluster](clusterCfg, fp); err != nil {
		return nil, errors.Wrap(err, "Failed to read YAML file")
	}
	if len(clusterCfg.Spec.Nodes.Masters) == 0 {
		return nil, fmt.Errorf("the number of master nodes must be greater than or equal to zero")
	}

	if f.arg.FuyaoZone == CN {
		clusterCfg.Spec.AdvancedParams = append(clusterCfg.Spec.AdvancedParams, base.Param{
			Name:  constants.FuyaoZone,
			Value: CN,
		})
	}

	if f.arg.DestroyCluster && f.arg.EnableOffline {
		clusterCfg, err = f.addHubNode(clusterCfg)
		if err != nil {
			return nil, err
		}
		clusterCfg, err = addHubComponentWithHarbor(clusterCfg)
		if err != nil {
			return nil, err
		}
	}

	if f.arg.AddNodes {
		clusterCfg, err = f.addNodesComponents(clusterCfg)
	} else {
		clusterCfg, err = f.allComponents(clusterCfg)
	}
	if err != nil {
		return nil, err
	}

	return handlerComponentsParams(f.arg, clusterCfg)
}

// addHubNode 配置文件中添加安装hub的节点
func (f *FileLoader) addHubNode(clusterCfg *fuyaov1beta1.Cluster) (*fuyaov1beta1.Cluster, error) {
	if len(clusterCfg.Spec.Nodes.Registry) != 0 {
		return clusterCfg, nil
	}

	localIp, err := utils.GetLocalIP()
	if err != nil {
		return nil, err
	}

	// 未提供ip，默认为本机ip
	if f.arg.RegistryNodeIP == "" {
		f.arg.RegistryNodeIP = localIp
	}

	// 未提供密码，判断安装harbor节点是否为集群内节点，如果是则获取集群内节点的密码与登录密钥
	privateKeyPath := ""
	if len(f.arg.RegistryNodePasswd) == 0 {
		var nodes []*fuyaov1beta1.Node
		nodes = append(nodes, clusterCfg.Spec.Nodes.Masters...)
		nodes = append(nodes, clusterCfg.Spec.Nodes.Workers...)
		for _, node := range nodes {
			if node.IP == f.arg.RegistryNodeIP {
				f.arg.RegistryNodePasswd = node.Password
				privateKeyPath = node.PrivateKeyPath
				break
			}
		}
	}

	registry := &fuyaov1beta1.Node{
		IP:             f.arg.RegistryNodeIP,
		Password:       f.arg.RegistryNodePasswd,
		User:           f.arg.RegistryNodeUsername,
		Port:           fuyaov1beta1.DefaultSSHPort,
		PrivateKeyPath: privateKeyPath,
	}

	clusterCfg.Spec.Nodes.Registry = append(clusterCfg.Spec.Nodes.Registry, registry)
	return clusterCfg, nil
}

func (f *FileLoader) addNodesComponents(addNodesCfg *fuyaov1beta1.Cluster) (*fuyaov1beta1.Cluster, error) {
	componentManifests, err := GetComponentManifests()
	if err != nil {
		return nil, err
	}

	groupMap := make(map[string]bool)
	groupMap[constants.Cni] = true
	groupMap[constants.PkgManager] = true
	groupMap[constants.Core] = true
	groupMap[constants.Cri] = true
	groupMap[constants.Oci] = true

	deploy, err := IsDeployComponentByNs(addNodesCfg.Spec.Nodes.Masters[0], constants.FuyaoSystem)
	if err != nil {
		return nil, err
	}
	if deploy {
		groupMap[constants.Tools] = true
	}

	isHa := false
	if len(addNodesCfg.Spec.Nodes.Masters) >= masterNodeCount {
		isHa = true
		groupMap[constants.Ha] = true
	}

	for group, components := range componentManifests.Spec.Components {
		_, isInstall := groupMap[group]
		if !isInstall {
			// 集群配置文件中包含无需安装的组件则报错
			isCover, name := f.isClusterCfgCoverGroupComponents(addNodesCfg.Spec.Components, components)
			if isCover {
				return nil, fmt.Errorf("the cluster expansion configuration file should not include the %s component", name)
			}
			continue
		}
		addNodesCfg, err = f.addComponents(addNodesCfg, components, group, isHa)
		if err != nil {
			return nil, err
		}
	}
	return addNodesCfg, nil
}

// isClusterCfgCoverGroupComponents 判断集群配置文件中是否包含无需安装的组件
func (f *FileLoader) isClusterCfgCoverGroupComponents(cfgComponents []fuyaov1beta1.Component,
	manifestsComponents []manifests.Component) (bool, string) {
	if len(cfgComponents) == 0 {
		return false, ""
	}

	for _, cfgComponent := range cfgComponents {
		for _, manifestsComponent := range manifestsComponents {
			if cfgComponent.Name == manifestsComponent.Name {
				return true, cfgComponent.Name
			}
		}
	}
	return false, ""
}

// components 配置文件中加载组件信息
func (f *FileLoader) allComponents(clusterCfg *fuyaov1beta1.Cluster) (*fuyaov1beta1.Cluster, error) {
	componentManifests, err := GetComponentManifests()
	if err != nil {
		return nil, err
	}

	isHa := false
	if len(clusterCfg.Spec.Nodes.Masters) >= 3 {
		isHa = true
	}

	for group, components := range componentManifests.Spec.Components {
		if group == constants.Ha && !isHa {
			// 集群配置文件中包含无需安装的组件则报错
			isCover, name := f.isClusterCfgCoverGroupComponents(clusterCfg.Spec.Components, components)
			if isCover {
				return nil, fmt.Errorf("non-ha deployments cannot contain the %s component", name)
			}
			continue
		}
		// 判断是否包含有ai组件，如果包含，则安装ai，如果不包含，则不安装
		if group == constants.AI && !f.arg.EnableAI {
			continue
		}

		clusterCfg, err = f.addComponents(clusterCfg, components, group, isHa)
		if err != nil {
			return nil, err
		}
	}
	return clusterCfg, nil
}

func (f *FileLoader) addComponents(clusterCfg *fuyaov1beta1.Cluster, components []manifests.Component,
	group string, isHa bool) (*fuyaov1beta1.Cluster, error) {
	for _, component := range components {
		// 用户提供的配置文件中存在的组件不在写入
		if f.isExistsComponent(component.Name, clusterCfg.Spec.Components) {
			// 非高可用部署，配置文件中不能包含ha组件
			if group == constants.Ha && !isHa {
				return nil, fmt.Errorf("non-ha deployments cannot contain ha component")
			}
			clusterCfg.Spec.Components = f.addComponentInfo(component, clusterCfg.Spec.Components, group)
			continue
		}
		clusterCfg.Spec.Components = append(clusterCfg.Spec.Components, fuyaov1beta1.Component{
			Name:    component.Name,
			Version: component.Version,
			Group:   group,
		})
	}
	return clusterCfg, nil
}

func (f *FileLoader) addComponentInfo(targetComponent manifests.Component, components []fuyaov1beta1.Component,
	group string) []fuyaov1beta1.Component {
	newComponents := make([]fuyaov1beta1.Component, 0)
	for _, component := range components {
		if component.Name == targetComponent.Name {
			component.Group = group
			component.Version = targetComponent.Version
		}
		newComponents = append(newComponents, component)
	}
	return newComponents
}

func (f *FileLoader) isExistsComponent(name string, components []fuyaov1beta1.Component) bool {
	for _, component := range components {
		if component.Name == name {
			return true
		}
	}
	return false
}

// DefaultLoader 自动获取本机配置信息
type DefaultLoader struct {
	arg Argument
}

// NewDefaultLoader is new DefaultLoader
func NewDefaultLoader(arg Argument) *DefaultLoader {
	return &DefaultLoader{
		arg: arg,
	}
}

// Load is automatically build config file
func (d *DefaultLoader) Load() (*fuyaov1beta1.Cluster, error) {
	zlog.Info("start load all in one config file")
	if err := isHasPrivateKey(); err != nil {
		return nil, err
	}

	hostname, err := os.Hostname()
	if err != nil {
		return nil, fmt.Errorf("failed to get hostname: %v", err)
	}

	u, err := user.Current()
	if err != nil {
		return nil, fmt.Errorf("failed to get user: %v", err)
	}

	ip, err := utils.GetLocalIP()
	if err != nil {
		return nil, err
	}

	allInOne := &fuyaov1beta1.Cluster{}
	allInOne.Spec.Nodes.Masters = append(allInOne.Spec.Nodes.Masters, &fuyaov1beta1.Node{
		Hostname:       strings.ToLower(hostname),
		User:           u.Name,
		Password:       []byte{},
		IP:             ip,
		Port:           fuyaov1beta1.DefaultSSHPort,
		PrivateKeyPath: fmt.Sprintf("%s/.ssh/id_rsa", u.HomeDir),
		RemoveTaint:    true,
	})

	if d.arg.FuyaoZone == CN {
		allInOne.Spec.AdvancedParams = append(allInOne.Spec.AdvancedParams, base.Param{
			Name:  constants.FuyaoZone,
			Value: CN,
		})
	}
	if d.arg.DestroyCluster && d.arg.EnableOffline {
		allInOne, err = d.addHubNode(allInOne, ip, u.HomeDir)
		if err != nil {
			return nil, err
		}
		allInOne, err = addHubComponentWithHarbor(allInOne)
		if err != nil {
			return nil, err
		}
	}

	return d.components(allInOne)
}

func (d *DefaultLoader) addHubNode(allInOne *fuyaov1beta1.Cluster, localIp,
	homeDir string) (*fuyaov1beta1.Cluster, error) {
	if d.arg.RegistryNodeIP == "" {
		d.arg.RegistryNodeIP = localIp
	}

	generateSSHKey := false
	if len(d.arg.RegistryNodePasswd) == 0 && d.arg.RegistryNodeIP == localIp {
		generateSSHKey = true
	}

	registry := &fuyaov1beta1.Node{
		User:     d.arg.RegistryNodeUsername,
		IP:       d.arg.RegistryNodeIP,
		Port:     fuyaov1beta1.DefaultSSHPort,
		Password: d.arg.RegistryNodePasswd,
		PrivateKeyPath: func() string {
			if generateSSHKey {
				return fmt.Sprintf("%s/.ssh/id_rsa", homeDir)
			}
			return ""
		}(),
	}

	allInOne.Spec.Nodes.Registry = append(allInOne.Spec.Nodes.Registry, registry)
	return allInOne, nil
}

// components 配置文件中加载组件信息
func (d *DefaultLoader) components(allInOne *fuyaov1beta1.Cluster) (*fuyaov1beta1.Cluster, error) {
	zlog.Info("start load components info")
	componentManifests, err := GetComponentManifests()
	if err != nil {
		return nil, err
	}

	// all in one 全量安装
	if d.arg.AllComponent {
		return handlerComponentsParams(d.arg, d.loadAllComponents(componentManifests, allInOne))
	}

	groupMap := make(map[string]bool)
	groupMap[constants.Cni] = true
	groupMap[constants.PkgManager] = true
	groupMap[constants.Core] = true
	groupMap[constants.Cri] = true
	groupMap[constants.Oci] = true
	if d.arg.EnableAI {
		groupMap[constants.AI] = true
	}

	for group, components := range componentManifests.Spec.Components {
		_, isInstall := groupMap[group]
		if !isInstall {
			continue
		}
		d.addComponents(group, components, allInOne)
	}
	return handlerComponentsParams(d.arg, allInOne)
}

func (d *DefaultLoader) loadAllComponents(componentManifests *manifests.ComponentManifests,
	allInOne *fuyaov1beta1.Cluster) *fuyaov1beta1.Cluster {
	for group, components := range componentManifests.Spec.Components {
		if group == constants.Ha || (group == constants.AI && !d.arg.EnableAI) {
			continue
		}
		d.addComponents(group, components, allInOne)
	}
	return allInOne
}

func (d *DefaultLoader) addComponents(group string, components []manifests.Component, allInOne *fuyaov1beta1.Cluster) {
	for _, component := range components {
		allInOne.Spec.Components = append(allInOne.Spec.Components, fuyaov1beta1.Component{
			Name:    component.Name,
			Version: component.Version,
			Group:   group,
		})
	}
}

func isExistsComponent(name string, components []fuyaov1beta1.Component) bool {
	for _, component := range components {
		if component.Name == name {
			return true
		}
	}

	return false
}

func addHubComponentWithHarbor(clusterCfg *fuyaov1beta1.Cluster) (*fuyaov1beta1.Cluster, error) {
	cfg, err := addHubComponent(clusterCfg)
	if err != nil {
		return nil, err
	}

	component := fuyaov1beta1.Component{
		Name:  constants.Harbor,
		Group: constants.Hub,
	}
	cfg.Spec.Components = append(cfg.Spec.Components, component)

	return cfg, nil
}

// addHubComponent 配置文件中添加安装时hub所需组件
func addHubComponent(clusterCfg *fuyaov1beta1.Cluster) (*fuyaov1beta1.Cluster, error) {
	// 随机解压一个部署包
	pkgPath := filepath.Join(constants.OptPath, constants.OpenFuyao, constants.Tmp)
	pkgName, err := UnCompressOnePackageRandom(pkgPath)
	if err != nil {
		return nil, err
	}

	hubComponentPath := filepath.Join(pkgPath, pkgName, constants.Hub, constants.Harbor)
	hubMetaData := &metadata.MetaData{}
	metadataPath := filepath.Join(hubComponentPath, constants.MetaDataFile)
	if err := UnmarshalJSON[*metadata.MetaData](hubMetaData, metadataPath); err != nil {
		return nil, err
	}

	componentManifests, err := GetComponentManifestsWithHubTmp("", filepath.Join(pkgPath, pkgName))
	if err != nil {
		return nil, err
	}

	// 根据hub的依赖组件和组件清单动态获取安装hub时所用的组件
	for _, depend := range hubMetaData.Dependencies {
		if isExistsComponent(depend.Name, clusterCfg.Spec.Components) {
			continue
		}
		clusterCfg = execAddHubComponent(componentManifests, clusterCfg, depend.Name)
	}
	return clusterCfg, nil
}

func execAddHubComponent(componentManifests *manifests.ComponentManifests, clusterCfg *fuyaov1beta1.Cluster,
	dependComponentName string) *fuyaov1beta1.Cluster {
	for group, components := range componentManifests.Spec.Components {
		for _, component := range components {
			if component.Name == dependComponentName {
				clusterCfg.Spec.Components = append(clusterCfg.Spec.Components, fuyaov1beta1.Component{
					Name:    dependComponentName,
					Version: component.Version,
					Group:   group,
				})
				break
			}
		}
	}
	return clusterCfg
}

func handlerComponentsParams(arg Argument, clusterCfg *fuyaov1beta1.Cluster) (*fuyaov1beta1.Cluster, error) {
	if !arg.EnableOffline {
		return clusterCfg, nil
	}

	zlog.Info("start handle components params")
	registryCfg, err := GetRegistryConfigYaml()
	if err != nil {
		return nil, err
	} else if err == nil && registryCfg == nil {
		// 离线使用用户自己镜像仓场景
		return clusterCfg, nil
	}

	tmpPath := filepath.Join(constants.OptPath, constants.OpenFuyao, constants.Tmp)
	if err := utils.MkdirAll(tmpPath, constants.Rwx); err != nil {
		return nil, err
	}
	pkgName, err := UnCompressOnePackageRandom(tmpPath)
	if err != nil {
		return nil, err
	}

	// 不填写域名则ip为域名
	registry := registryCfg.Spec.Domain
	if registryCfg.Spec.Domain == "" {
		registry = registryCfg.Spec.IP
	}

	components := make([]fuyaov1beta1.Component, 0)
	for _, component := range clusterCfg.Spec.Components {
		zlog.Debugf("set component %s registry", component.Name)

		defaultValues := &values.Values{}
		valuesYamlPath := filepath.Join(tmpPath, pkgName, component.Group, component.Name, constants.DefaultValues)
		if err := UnmarshalYaml[*values.Values](defaultValues, valuesYamlPath); err != nil {
			return nil, err
		}

		component.Param = getNewComponentParam(registry, defaultValues.Spec.Params, component.Param)
		components = append(components, component)
	}
	clusterCfg.Spec.Components = components
	return clusterCfg, nil
}

func getNewComponentParam(registry string, defaultParams []base.Param, componentParams []base.Param) []base.Param {
	var newParam []base.Param
	// 离线部署，优先使用用户提供的镜像仓，若用户未提供，则使用自己创建的
	for _, param := range defaultParams {
		if param.Name != constants.Registry {
			continue
		}

		setRegistry := true
		for _, item := range componentParams {
			if item.Name == constants.Registry && item.Value == "" {
				item.Value = registry
			}
			if item.Name == constants.Registry {
				setRegistry = false
			}
			newParam = append(newParam, item)
		}

		// default-values.yaml中有registry默认参数，但配置文件未配置registry参数，则配置registry
		if setRegistry {
			newParam = append(newParam, base.Param{
				Name:  constants.Registry,
				Value: registry,
			})
		}
	}

	return newParam
}

// RegistryLoader 镜像仓
type RegistryLoader struct {
	arg                    Argument
	registryNodeUser       string
	registryNodeIP         string
	registryNodePasswd     []byte
	registryDomain         string
	registryPassword       []byte
	registryDBPassword     []byte
	registryCertExpiryDays string
}

// NewRegistryLoader is new RegistryLoader
func NewRegistryLoader(arg Argument) *RegistryLoader {
	return &RegistryLoader{
		arg: arg,

		registryDomain:         arg.RegistryDomain,
		registryPassword:       arg.RegistryPasswd,
		registryDBPassword:     arg.RegistryDBPasswd,
		registryNodeIP:         arg.RegistryNodeIP,
		registryNodePasswd:     arg.RegistryNodePasswd,
		registryNodeUser:       arg.RegistryNodeUsername,
		registryCertExpiryDays: arg.RegistryCertExpiryDays,
	}
}

// Load is load registry config file
func (r *RegistryLoader) Load() (*fuyaov1beta1.Cluster, error) {
	zlog.Info("start load registry config file")
	if err := isHasPrivateKey(); err != nil {
		return nil, err
	}

	ip, err := utils.GetLocalIP()
	if err != nil {
		return nil, err
	}
	// 确定镜像仓节点ip
	onlyOneNode := false
	if r.registryNodeIP == "" {
		r.registryNodeIP = ip
		onlyOneNode = true
	}

	u, err := user.Current()
	if err != nil {
		return nil, fmt.Errorf("failed to get user: %v", err)
	}

	registry := &fuyaov1beta1.Cluster{}
	registry.Spec.Nodes.Registry = append(registry.Spec.Nodes.Registry, &fuyaov1beta1.Node{
		User:           r.registryNodeUser,
		Password:       r.registryNodePasswd,
		IP:             r.registryNodeIP,
		Port:           fuyaov1beta1.DefaultSSHPort,
		PrivateKeyPath: fmt.Sprintf("%s/.ssh/id_rsa", u.HomeDir),
	})

	if !onlyOneNode {
		registry.Spec.Nodes.Workers = append(registry.Spec.Nodes.Workers, &fuyaov1beta1.Node{
			User:           u.Name,
			IP:             ip,
			Port:           fuyaov1beta1.DefaultSSHPort,
			PrivateKeyPath: fmt.Sprintf("%s/.ssh/id_rsa", u.HomeDir),
		})
	}

	return r.components(registry)
}

// components 增加部署镜像仓所需的组件
func (r *RegistryLoader) components(registry *fuyaov1beta1.Cluster) (*fuyaov1beta1.Cluster, error) {
	registryCfg, err := addHubComponent(registry)
	if err != nil {
		return nil, err
	}

	harbor := r.generateHarborCfg(registry.Spec.Nodes.Registry[0])
	registryCfg.Spec.Components = append(registryCfg.Spec.Components, harbor)
	return registryCfg, nil
}

// handleHarborComponent 生成harbor组件与参数的结构体
func (r *RegistryLoader) generateHarborCfg(node *fuyaov1beta1.Node) fuyaov1beta1.Component {
	var HarborParams []base.Param
	HarborParams = append(HarborParams, base.Param{
		Name:  RegistryPassword,
		Value: string(r.registryPassword),
	})
	HarborParams = append(HarborParams, base.Param{
		Name:  RegistryDBPassword,
		Value: string(r.registryDBPassword),
	})

	if r.registryDomain == "" {
		r.registryDomain = node.IP
	}
	HarborParams = append(HarborParams, base.Param{
		Name:  RegistryDomain,
		Value: r.registryDomain,
	})
	HarborParams = append(HarborParams, base.Param{
		Name:  RegistryCertExpiryDays,
		Value: r.registryCertExpiryDays,
	})

	HarborParams = append(HarborParams, base.Param{
		Name:  IP,
		Value: node.IP,
	})

	HarborParams = append(HarborParams, base.Param{
		Name:  RegistryHttpsPort,
		Value: constants.DefaultRegistryHttpsPort,
	})

	HarborParams = append(HarborParams, base.Param{
		Name:  RegistryHttpPort,
		Value: constants.DefaultRegistryHttpPort,
	})

	return fuyaov1beta1.Component{
		Name:    constants.Harbor,
		Group:   constants.Hub,
		Version: "",
		Param:   HarborParams,
	}
}

func isHasPrivateKey() error {
	cmd := `if [ -f "$HOME/.ssh/authorized_keys" ] && [ -f "$HOME/.ssh/id_rsa" ]; then
                echo "true"; 
            else
                echo "false";
            fi;`
	output, err := utils.ExecLocalCmd(cmd)
	if err != nil {
		return fmt.Errorf("failed to determine if the file exists: %v\n%s", err, output)
	}
	if strings.TrimRight(output, "\r\n") == "false" {
		return fmt.Errorf("not found authorized_keys or id_rsa file in the $HOME/.ssh, please check the environment")
	}
	return nil
}
