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

package pipelines

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"installer/pkg/common"
	"installer/pkg/constants"
	"installer/pkg/fuyao/v1beta1/manifests"
	"installer/pkg/fuyao/v1beta1/metadata"
	"installer/pkg/utils"
	"installer/pkg/zlog"
)

// ComponentPackage 组件代码包
type ComponentPackage struct {
	AbsPath      string
	AssetPath    string
	ManifestFile string

	ComponentManifests *manifests.ComponentManifests
	manifestsMetaData  *metadata.Version
}

// NewComponentPackage is new ComponentPackage
func NewComponentPackage() *ComponentPackage {
	return &ComponentPackage{
		ComponentManifests: &manifests.ComponentManifests{},
	}
}

// copyComponentScripts 拷贝组件脚本和配置文件
func (c *ComponentPackage) copyComponentScripts(savePath, assetOsPath string) error {
	for group, components := range c.ComponentManifests.Spec.Components {
		// 创建组件分组目录
		groupPath := filepath.Join(savePath, group)

		// 拷贝具体组件
		for _, component := range components {
			// 目标组件脚本目录
			dstComponentPath := filepath.Join(groupPath, component.Name)
			// 原始组件脚本目录
			srcComponentPath := filepath.Join(assetOsPath, group, component.Name, component.Version)
			err := common.CopyLocalDir(srcComponentPath, dstComponentPath)
			if err != nil {
				return fmt.Errorf("failed to copy shell scripts: %v", err)
			}
		}
	}
	return nil
}

// copyConfigFile 拷贝配置文件->高级参数等配置文件
func (c *ComponentPackage) copyConfigFile(savePath string) error {
	FuyaoMetaData := &metadata.Version{}
	metadataPath := filepath.Join(c.AbsPath, constants.Manifests, constants.MetaDataFile)
	if err := common.UnmarshalJSON[*metadata.Version](FuyaoMetaData, metadataPath); err != nil {
		return err
	}

	dstConfigPath := filepath.Join(savePath, constants.Config)
	srcConfigPath := filepath.Join(c.AbsPath, constants.Manifests, constants.Config, FuyaoMetaData.Installer)
	err := common.CopyLocalDir(srcConfigPath, dstConfigPath)
	if err != nil {
		return fmt.Errorf("failed to copy config file: %v", err)
	}
	return nil
}

// copyCommonDir 拷贝通用路径
func (c *ComponentPackage) copyCommonDir(dirMap map[string]string) error {
	for src, dst := range dirMap {
		err := common.CopyLocalDir(src, dst)
		if err != nil {
			return fmt.Errorf("failed to copy common file: %v", err)
		}
	}
	return nil
}

// copyCommonFile 拷贝通用文件
func (c *ComponentPackage) copyCommonFile(pathMap map[string]string) error {
	for src, dst := range pathMap {
		err := common.CopyLocalFile(src, dst)
		if err != nil {
			return fmt.Errorf("failed to copy common file: %v", err)
		}
	}
	return nil
}

// createTmpSavePath 创建临时存储安装包路径
func (c *ComponentPackage) createTmpSavePath(osInfo manifests.OsInfo) (string, error) {
	tmpPath := filepath.Join(c.AbsPath, constants.Manifests, osInfo.CpuArch, osInfo.OsName)
	if _, err := os.Stat(tmpPath); os.IsNotExist(err) {
		c.removeTmpPath(filepath.Join(c.AbsPath, constants.Tmp))
		return "", fmt.Errorf("this operating system or cpu architecture is not supported: %s, %s",
			osInfo.CpuArch, osInfo.OsName)
	}

	sliceTmp := []string{constants.Fuyao, osInfo.OsName, osInfo.Version, osInfo.CpuArch}
	packageName := strings.Join(sliceTmp, constants.Minus)
	tmpSavePath := filepath.Join(c.AbsPath, constants.Tmp, packageName)
	err := os.MkdirAll(tmpSavePath, constants.RwxRxRx)
	if err != nil {
		return "", fmt.Errorf("failed to create folder: %v", err)
	}
	return tmpSavePath, nil
}

func (c *ComponentPackage) removeTmpPath(tmpPath string) {
	if err := os.RemoveAll(tmpPath); err != nil {
		zlog.Error("failed to delete directory", err)
	}
}

// getCurrentOsPath 获取当前操作系统存放shell脚本路径
func (c *ComponentPackage) getCurrentOsPath(osInfo manifests.OsInfo) (string, error) {
	// 当前遇到未支持的操作系统版本号也会执行打包
	currentOsPath := filepath.Join(c.AbsPath, constants.Manifests, osInfo.CpuArch, osInfo.OsName, osInfo.Version)
	if _, err := os.Stat(currentOsPath); os.IsNotExist(err) {
		currentOsPath = filepath.Join(c.AbsPath, constants.Manifests, osInfo.CpuArch, osInfo.OsName,
			constants.CommonScripts)

		versionMetaData := &manifests.OsVersionMetaData{}
		metadataPath := filepath.Join(c.AbsPath, constants.Manifests, osInfo.CpuArch, osInfo.OsName,
			constants.CommonScripts, constants.MetaDataFile)
		err := common.UnmarshalJSON[*manifests.OsVersionMetaData](versionMetaData, metadataPath)
		if err != nil {
			return "", err
		}

		isCover := false
		for _, version := range versionMetaData.OsVersion {
			if version == osInfo.Version {
				isCover = true
				break
			}
		}

		if !isCover {
			fmt.Printf("warning: Deployment on the %s %s %s may fail\n",
				osInfo.OsName, osInfo.Version, osInfo.CpuArch)
		}
	}

	return currentOsPath, nil
}

// CreateUnifyComponentPackage 创建统一代码包
// 根据组件清单打包组件资产，打出代码包为大包，完成打包后无需解压可直接执行
// 自动创建targets目录，shell脚本和配置文件打包到targets/asset目录下，fuyao可执行文件打包到targets目录下
func (c *ComponentPackage) CreateUnifyComponentPackage() error {
	for _, osInfo := range c.ComponentManifests.Spec.Ossa {
		zlog.Info("current os name ", osInfo.OsName)
		// 创建临时存储路径
		tmpSavePath, err := c.createTmpSavePath(osInfo)
		if err != nil {
			return err
		}

		currentOsPath, err := c.getCurrentOsPath(osInfo)
		if err != nil {
			return err
		}

		// 拷贝安装脚本
		if err := c.copyComponentScripts(tmpSavePath, currentOsPath); err != nil {
			return err
		}
		// 拷贝config
		if err := c.copyConfigFile(tmpSavePath); err != nil {
			return err
		}

		// 拷贝通用文件
		dirMap := make(map[string]string)
		dirMap[filepath.Join(c.AbsPath, constants.Manifests, constants.Utils)] = filepath.Join(tmpSavePath,
			constants.Utils)
		dirMap[filepath.Join(currentOsPath, constants.Hub)] = filepath.Join(tmpSavePath, constants.Hub)
		if err := c.copyCommonDir(dirMap); err != nil {
			return err
		}
		filePathMap := make(map[string]string)
		filePathMap[filepath.Join(c.AbsPath, constants.Manifests, constants.MetaDataFile)] = filepath.Join(tmpSavePath,
			constants.MetaDataFile)
		if err := c.copyCommonFile(filePathMap); err != nil {
			return err
		}

		// 生成压缩包文件
		sliceTmp := []string{constants.Fuyao, osInfo.OsName, osInfo.Version, osInfo.CpuArch}
		err = common.CompressTarGZ(tmpSavePath, filepath.Join(c.AssetPath,
			strings.Join(sliceTmp, constants.Minus))+constants.TarGz)
		if err != nil {
			return err
		}
	}

	// 拷贝清单文件
	filePathMap := make(map[string]string)
	filePathMap[c.ManifestFile] = filepath.Join(c.AssetPath, constants.ComponentManifest)
	filePathMap[filepath.Join(c.AbsPath, constants.Manifests, constants.MetaDataFile)] = filepath.Join(c.AssetPath,
		constants.MetaDataFile)
	serviceName := common.GetServiceName()
	filePathMap[filepath.Join(c.AbsPath, serviceName)] = filepath.Join(c.AbsPath, constants.Targets, serviceName)
	if err := c.copyCommonFile(filePathMap); err != nil {
		return err
	}

	c.removeTmpPath(filepath.Join(c.AbsPath, constants.Tmp))
	fmt.Println("Successfully completed component code packaging.")
	return nil
}

func (c *ComponentPackage) execCopy(manifestsSavePath, currentManifestPath, pkgAssetPath, pkgPath string) error {
	// 拷贝安装脚本
	if err := c.copyComponentScripts(manifestsSavePath, currentManifestPath); err != nil {
		return err
	}
	// 拷贝config
	if err := c.copyConfigFile(manifestsSavePath); err != nil {
		return err
	}

	// 拷贝通用文件
	// 拷贝hub文件
	dirMap := make(map[string]string)
	dirMap[filepath.Join(c.AbsPath, constants.Manifests, constants.Utils)] = filepath.Join(manifestsSavePath,
		constants.Utils)
	dirMap[filepath.Join(currentManifestPath, constants.Hub)] = filepath.Join(manifestsSavePath, constants.Hub)
	if err := c.copyCommonDir(dirMap); err != nil {
		return err
	}

	// 拷贝清单文件、元数据文件、二进制
	filePathMap := make(map[string]string)
	filePathMap[filepath.Join(c.AbsPath, constants.Manifests, constants.MetaDataFile)] = filepath.Join(manifestsSavePath,
		constants.MetaDataFile)
	filePathMap[filepath.Join(c.AbsPath, constants.Manifests, constants.MetaDataFile)] = filepath.Join(pkgAssetPath,
		constants.MetaDataFile)
	filePathMap[c.ManifestFile] = filepath.Join(pkgAssetPath, constants.ComponentManifest)
	serviceName := common.GetServiceName()
	filePathMap[filepath.Join(c.AbsPath, serviceName)] = filepath.Join(pkgPath, serviceName)
	if err := c.copyCommonFile(filePathMap); err != nil {
		return err
	}
	return nil
}

// CreateSubComponentTarPackage 创建每个操作系统的单独包
// 构建好后的代码包在targets目录下
func (c *ComponentPackage) CreateSubComponentTarPackage() error {
	packageNameMap := make(map[string]bool)

	// 创建临时存储路径 targets/tmp
	targetsPath := filepath.Join(c.AbsPath, constants.Targets)
	c.removeTmpPath(targetsPath)
	tmpPath := filepath.Join(targetsPath, constants.Tmp)
	if err := utils.MkdirAll(tmpPath, constants.RwxRxRx); err != nil {
		return err
	}

	for _, osInfo := range c.ComponentManifests.Spec.Ossa {
		if runtime.GOARCH != osInfo.CpuArch {
			zlog.Infof("different CPU arch: local %s、 config %s", runtime.GOARCH, osInfo.CpuArch)
			continue
		}
		// 创建当前临时代码包路径 tmp/fuyao-v0.5.1-openEuler-23.03-arm64
		pkgName := c.getPkgName(osInfo)
		_, ok := packageNameMap[pkgName]
		if ok {
			zlog.Info("same os info")
			continue
		}

		pkgPath := filepath.Join(tmpPath, pkgName)
		pkgAssetPath := filepath.Join(pkgPath, constants.Asset)
		sliceTmp := []string{constants.Fuyao, osInfo.OsName, osInfo.Version, osInfo.CpuArch}
		manifestPkgName := strings.Join(sliceTmp, constants.Minus)
		// tmp/fuyao-v0.5.1-openEuler-23.03-arm64/asset/package-path
		manifestsSavePath := filepath.Join(pkgAssetPath, manifestPkgName)
		if err := utils.MkdirAll(manifestsSavePath, constants.RwxRxRx); err != nil {
			return err
		}

		// 获取这个版本OS存放脚本路径路径
		currentManifestPath, err := c.getCurrentOsPath(osInfo)
		if err != nil {
			return err
		}

		if err := c.execCopy(manifestsSavePath, currentManifestPath, pkgAssetPath, pkgPath); err != nil {
			return err
		}

		// 生成清单的压缩包
		err = common.CompressTarGZ(manifestsSavePath, filepath.Join(pkgAssetPath, manifestPkgName)+constants.TarGz)
		if err != nil {
			return err
		}
		c.removeTmpPath(manifestsSavePath)
		err = os.RemoveAll(manifestsSavePath)

		// 生成统一压缩包
		err = common.CompressTarGZ(pkgPath, filepath.Join(targetsPath, pkgName)+constants.TarGz)
		if err != nil {
			return err
		}
	}
	c.removeTmpPath(tmpPath)
	// 删除临时目录
	fmt.Println("Successfully completed component code packaging.")
	return nil
}

func (c *ComponentPackage) getPkgName(osInfo manifests.OsInfo) string {
	pkgName := fmt.Sprintf("%s-v%s-%s-%s-%s", constants.Fuyao, c.manifestsMetaData.FuyaoSystem, osInfo.OsName,
		osInfo.Version, osInfo.CpuArch)
	// latest版本的包名不带版本号
	if c.manifestsMetaData.FuyaoSystem == constants.LatestVersion {
		pkgName = fmt.Sprintf("%s-%s-%s-%s-%s", constants.Fuyao, c.manifestsMetaData.FuyaoSystem, osInfo.OsName,
			osInfo.Version, osInfo.CpuArch)
	}
	return pkgName
}

// NewCreateComponentPackagePipeline is new create pkg pipline
func NewCreateComponentPackagePipeline(args common.Argument, manifestFilePath string) error {
	componentManifests := &manifests.ComponentManifests{}
	if err := common.UnmarshalYaml[*manifests.ComponentManifests](componentManifests, manifestFilePath); err != nil {
		return err
	}
	absPath, err := common.GetAbsPath()
	if err != nil {
		return err
	}
	// 获取当前版本信息 取自meta-data.json
	manifestPath := filepath.Join(absPath, constants.Manifests)
	ManifestMetaDataPath := filepath.Join(manifestPath, constants.MetaDataFile)
	ManifestMetaData := &manifests.ManifestMetaData{}
	if err := common.UnmarshalJSON[*manifests.ManifestMetaData](ManifestMetaData, ManifestMetaDataPath); err != nil {
		return err
	}

	manifestsMetaData := &metadata.Version{}
	metadataPath := filepath.Join(absPath, constants.Manifests, constants.MetaDataFile)
	if err := common.UnmarshalJSON[*metadata.Version](manifestsMetaData, metadataPath); err != nil {
		return err
	}

	componentPkg := NewComponentPackage()
	componentPkg.manifestsMetaData = manifestsMetaData

	componentPkg.AbsPath = absPath
	componentPkg.ManifestFile = manifestFilePath
	componentPkg.ComponentManifests = componentManifests
	// 分包不去创建公共的asset文件夹
	if !args.EnableSubpackageTar {
		assetPath, err := common.CreateAssetDirectory()
		if err != nil {
			return err
		}
		componentPkg.AssetPath = assetPath
		return componentPkg.CreateUnifyComponentPackage()
	}
	return componentPkg.CreateSubComponentTarPackage()
}

// CreateComponentPackage 创建组件代码包
func CreateComponentPackage(args common.Argument) error {
	manifestFilePath := filepath.Clean(args.FilePath)
	if !strings.HasSuffix(manifestFilePath, constants.SuffixYaml) {
		return fmt.Errorf("please provide the correct component config file")
	}
	_, err := os.Stat(manifestFilePath)
	if os.IsNotExist(err) || err != nil {
		return fmt.Errorf("component config file is not exist or failed to get file state")
	}
	// fuyao-v0.5.1-openEuler-23.03-arm64.tar.gz
	return NewCreateComponentPackagePipeline(args, manifestFilePath)
}
