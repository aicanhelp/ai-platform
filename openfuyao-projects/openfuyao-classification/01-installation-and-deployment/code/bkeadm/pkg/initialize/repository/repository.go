/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package repository

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	bkecommon "gopkg.openfuyao.cn/cluster-api-provider-bke/common"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/validation"

	"gopkg.openfuyao.cn/bkeadm/pkg/build"
	"gopkg.openfuyao.cn/bkeadm/pkg/common"
	econd "gopkg.openfuyao.cn/bkeadm/pkg/executor/containerd"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/server"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

var (
	imageFile           = fmt.Sprintf("%s/%s-%s", global.Workspace, utils.ImageFile, runtime.GOARCH)
	imageDataFile       = fmt.Sprintf("%s/%s", global.Workspace, utils.ImageDataFile)
	imageDataDirectory  = fmt.Sprintf("%s/%s", global.Workspace, utils.ImageDataDirectory)
	yumDataFile         = fmt.Sprintf("%s/%s", global.Workspace, utils.SourceDataFile)
	yumDataDirectory    = fmt.Sprintf("%s/%s", global.Workspace, utils.SourceDataDirectory)
	chartDataFile       = fmt.Sprintf("%s/%s", global.Workspace, utils.ChartDataFile)
	chartDataDirectory  = fmt.Sprintf("%s/%s", global.Workspace, utils.ChartDataDirectory)
	nfsDataFile         = fmt.Sprintf("%s/%s", global.Workspace, utils.NFSDataFile)
	nfsDataDirectory    = fmt.Sprintf("%s/%s", global.Workspace, utils.NFSDataDirectory)
	imageLocalDirectory = fmt.Sprintf("%s/%s", global.Workspace, utils.ImageLocalDirectory) // 用于operator/调谐器安装的本地镜像目录
	imageLocalFile      = fmt.Sprintf("%s/%s-%s", imageLocalDirectory, utils.ImageFile, runtime.GOARCH)
)

// decompressConfig holds configuration for decompressing a data package.
type decompressConfig struct {
	dataFile        string
	dataDirectory   string
	name            string
	logMessage      string
	skipMessage     string
	unTarTarget     string // if empty, use dataDirectory + ".tmp"
	createTargetDir bool   // if true, create target directory before untar
	postProcess     func(targetTemp string) error
}

// decompressDataPackage handles the common logic for decompressing data packages.
func decompressDataPackage(cfg decompressConfig) error {
	if utils.Exists(cfg.dataDirectory) && !utils.DirectoryIsEmpty(cfg.dataDirectory) {
		log.Warn(cfg.skipMessage)
		log.Infof("If you want to unzip it again, remove the directory %s", cfg.dataDirectory)
		return nil
	}

	if err := os.Remove(cfg.dataDirectory); err != nil && !os.IsNotExist(err) {
		log.Warnf("Failed to remove %s data directory: %v", cfg.name, err)
	}

	if !utils.Exists(cfg.dataFile) {
		if err := os.Mkdir(cfg.dataDirectory, utils.DefaultDirPermission); err != nil {
			return err
		}
		return verifyDirectoryExists(cfg.dataDirectory)
	}

	log.Infof("%s this may take several minutes", strings.TrimRight(cfg.logMessage, "."))

	start := time.Now()
	targetTemp := cfg.unTarTarget
	if targetTemp == "" {
		targetTemp = cfg.dataDirectory + ".tmp"
	}

	if err := prepareTempDirectory(targetTemp); err != nil {
		return err
	}

	if cfg.createTargetDir {
		if err := os.MkdirAll(targetTemp, utils.DefaultDirPermission); err != nil {
			return err
		}
	}

	count, err := utils.UnTarWithCount(cfg.dataFile, targetTemp)
	if err != nil {
		return err
	}

	if cfg.postProcess != nil {
		if err := cfg.postProcess(targetTemp); err != nil {
			return err
		}
	} else {
		if err := os.Rename(targetTemp, cfg.dataDirectory); err != nil {
			return err
		}
	}

	if err := verifyDirectoryExists(cfg.dataDirectory); err != nil {
		return err
	}
	log.Infof("Decompressed %s package: %d entries, elapsed=%s",
		cfg.name, count, time.Since(start).Round(time.Second))
	return nil
}

// prepareTempDirectory ensures the temporary directory is clean and ready.
func prepareTempDirectory(targetTemp string) error {
	if utils.Exists(targetTemp) {
		if err := os.RemoveAll(targetTemp); err != nil {
			return err
		}
	}
	return nil
}

// verifyDirectoryExists checks if the directory exists and returns an error if not.
func verifyDirectoryExists(dir string) error {
	if !utils.Exists(dir) {
		return fmt.Errorf("%s, not found", dir)
	}
	return nil
}

// PrepareRepositoryDependOn prepares repository dependencies by decompressing
// required data packages including image, chart, and NFS data.
func PrepareRepositoryDependOn(imageFilePath string) error {
	// Prepare image data
	if err := decompressDataPackage(decompressConfig{
		dataFile:      imageDataFile,
		dataDirectory: imageDataDirectory,
		name:          "image",
		logMessage:    "Decompressing the image package...",
		skipMessage:   "If the image file already exists, skip decompressing the volumes/image.tar.gz file. ",
	}); err != nil {
		return err
	}

	// Prepare chart data (special handling - no temp directory)
	if err := prepareChartData(); err != nil {
		return err
	}

	// Prepare NFS data
	if err := decompressDataPackage(decompressConfig{
		dataFile:        nfsDataFile,
		dataDirectory:   nfsDataDirectory,
		name:            "NFS",
		logMessage:      "Decompressing the nfsshare source package...",
		skipMessage:     "If the nfsshare file already exists, skip decompressing the mount/nfsshare.tar.gz file. ",
		unTarTarget:     global.Workspace + "/mount/nfsshare.tmp",
		createTargetDir: true,
		postProcess:     nfsPostProcess,
	}); err != nil {
		return err
	}

	if imageFilePath != "" {
		// Prepare local image
		if err := decompressDataPackage(decompressConfig{
			dataFile:        imageFilePath,
			dataDirectory:   imageLocalDirectory,
			name:            "local image",
			logMessage:      "Decompressing the local image phase1 package...",
			skipMessage:     fmt.Sprintf("If the image file already exists, skip decompressing the %s file. ", imageFilePath),
			createTargetDir: true,
			postProcess:     createImageLocalPostProcess(imageFilePath),
		}); err != nil {
			return err
		}
	}

	return nil
}

// prepareChartData handles chart data preparation with its special logic.
// Chart unpacks directly to mount directory without using a temp directory.
func prepareChartData() error {
	if utils.Exists(chartDataDirectory) && !utils.DirectoryIsEmpty(chartDataDirectory) {
		log.Infof("Skip decompressing charts: directory %s already exists, if you want to unzip it again, remove the directory", chartDataDirectory)
		return nil
	}

	if err := os.Remove(chartDataDirectory); err != nil && !os.IsNotExist(err) {
		log.Warnf("Failed to remove chart data directory: %v", err)
	}

	if utils.Exists(chartDataFile) {
		log.Info("Decompressing the chart source package, this may take several minutes")
		start := time.Now()
		count, err := utils.UnTarWithCount(chartDataFile, global.Workspace+"/mount")
		if err != nil {
			return err
		}
		log.Infof("Decompressed chart source package: %d entries, elapsed=%s",
			count, time.Since(start).Round(time.Second))
	} else {
		if err := os.Mkdir(chartDataDirectory, utils.DefaultDirPermission); err != nil {
			return err
		}
	}

	return verifyDirectoryExists(chartDataDirectory)
}

// nfsPostProcess handles the special post-processing for NFS data.
func nfsPostProcess(targetTemp string) error {
	if utils.Exists(targetTemp + "/nfsshare") {
		if err := os.Rename(targetTemp+"/nfsshare", nfsDataDirectory); err != nil {
			return err
		}
		if err := os.Remove(targetTemp); err != nil && !os.IsNotExist(err) {
			log.Warnf("Failed to remove temp directory: %v", err)
		}
	} else {
		if err := os.Rename(targetTemp, nfsDataDirectory); err != nil {
			return err
		}
	}
	return nil
}

// removeArchiveExtensions removes common archive/compression extensions from filename.
func removeArchiveExtensions(filename string) string {
	extensions := []string{
		".tar.gz",
		".tgz",
	}

	for _, ext := range extensions {
		if strings.HasSuffix(filename, ext) {
			return strings.TrimSuffix(filename, ext)
		}
	}
	return filename
}

// createImageLocalPostProcess creates a post-process function for local image decompression.
func createImageLocalPostProcess(imageFilePath string) func(targetTemp string) error {
	return func(targetTemp string) error {
		// Extract the base name without extension (e.g., "image_amd64" from "image_amd64.tar.gz")
		baseName := removeArchiveExtensions(filepath.Base(imageFilePath))

		expectedSubDir := filepath.Join(targetTemp, baseName) // mount/local_image.tmp/image_amd64
		if utils.Exists(expectedSubDir) {
			if err := os.Rename(expectedSubDir, imageLocalDirectory); err != nil {
				return err
			}
			if err := os.RemoveAll(targetTemp); err != nil && !os.IsNotExist(err) {
				log.Warnf("Failed to remove temp directory: %v", err)
			}
		} else {
			if err := os.Rename(targetTemp, imageLocalDirectory); err != nil {
				return err
			}
		}
		return nil
	}
}

// LoadLocalRepository loads images from the default image file into the local container runtime
func LoadLocalRepository() error {
	return common.LoadLocalRepositoryFromFile(imageFile)
}

// LoadLocalImage loads images from the default local image file into the local container runtime
func LoadLocalImage() error {
	return common.LoadLocalRepositoryFromFile(imageLocalFile)
}

func ContainerServer(localImage, imageRegistryPort, otherRepo, onlineImage string) error {
	var image string

	// 优先级：localImage > otherRepo > (onlineImage为空时使用本地) > 默认值
	if localImage != "" {
		// localImage 不为空，使用本地镜像
		image = utils.DefaultLocalImageRegistry
	} else if otherRepo != "" {
		// otherRepo 不为空，使用 otherRepo + DefaultLocalImageRegistry
		image = fmt.Sprintf("%s%s", otherRepo, utils.DefaultLocalImageRegistry)
	} else if onlineImage == "" {
		// onlineImage 为空（离线场景），使用本地镜像
		image = utils.DefaultLocalImageRegistry
	} else {
		// 默认使用第三方镜像
		image = fmt.Sprintf("%s/%s", utils.DefaultThirdMirror, utils.DefaultLocalImageRegistry)
	}

	err := server.StartImageRegistry(utils.LocalImageRegistryName, image, imageRegistryPort, imageDataDirectory)
	if err != nil {
		if rmErr := server.RemoveImageRegistry(utils.LocalImageRegistryName); rmErr != nil {
			log.Warnf("Remove image registry failed before retry: %v", rmErr)
		}
		for i := 0; i < 3; i++ {
			log.Infof("Retrying to start image registry (%d/3)...", i+1)
			time.Sleep(utils.DefaultSleepSeconds * time.Second)
			err = server.StartImageRegistry(utils.LocalImageRegistryName, image, imageRegistryPort, imageDataDirectory)
			if err == nil {
				return nil
			}
			log.Debugf("Start image registry retry %d/3 failed: %v", i+1, err)
			if rmErr := server.RemoveImageRegistry(utils.LocalImageRegistryName); rmErr != nil {
				log.Warnf("Remove image registry failed after retry: %v", rmErr)
			}
		}
		return fmt.Errorf("start image registry: %w", err)
	}
	return nil
}

func SyncLocalImage(imageRegistryPort string) error {
	build.SpecificSync(imageLocalDirectory, fmt.Sprintf("127.0.0.1:%s", imageRegistryPort))
	image := fmt.Sprintf("127.0.0.1:%s/%s/%s", imageRegistryPort, bkecommon.ImageRegistryKubernetes, utils.DefaultLocalYumRegistry)
	err := econd.EnsureImageExists(image)
	if err != nil {
		return err
	}
	return nil
}

func YumServer(localImage, imageRegistryPort, yumRegistryPort, otherRepo, onlineImage string) error {
	var image string
	localImagePath := fmt.Sprintf("127.0.0.1:%s/%s/%s", imageRegistryPort, bkecommon.ImageRegistryKubernetes, utils.DefaultLocalYumRegistry)
	if localImage != "" {
		image = localImagePath
	} else if otherRepo != "" {
		image = fmt.Sprintf("%s%s", otherRepo, utils.DefaultLocalYumRegistry)
	} else if onlineImage == "" {
		image = localImagePath
	} else {
		image = fmt.Sprintf("%s/%s", utils.DefaultThirdMirror, utils.DefaultLocalYumRegistry)
	}

	err := server.StartYumRegistry(utils.LocalYumRegistryName, image, yumRegistryPort, yumDataDirectory)
	if err != nil {
		if rmErr := server.RemoveYumRegistry(utils.LocalYumRegistryName); rmErr != nil {
			log.Warnf("Remove yum registry failed before retry: %v", rmErr)
		}
		for i := 0; i < 3; i++ {
			log.Infof("Retrying to start yum registry (%d/3)...", i+1)
			time.Sleep(utils.DefaultSleepSeconds * time.Second)
			err = server.StartYumRegistry(utils.LocalYumRegistryName, image, yumRegistryPort, yumDataDirectory)
			if err == nil {
				return nil
			}
			log.Debugf("Start yum registry retry %d/3 failed: %v", i+1, err)
			if rmErr := server.RemoveYumRegistry(utils.LocalYumRegistryName); rmErr != nil {
				log.Warnf("Remove yum registry failed after retry: %v", rmErr)
			}
		}
		return fmt.Errorf("start yum registry: %w", err)
	}
	return nil
}

func ChartServer(localImage, imageRegistryPort, chartRegistryPort, otherRepo, onlineImage string) error {
	var image string
	localImagePath := fmt.Sprintf("127.0.0.1:%s/%s/%s", imageRegistryPort, bkecommon.ImageRegistryKubernetes, utils.DefaultLocalChartRegistry)
	if localImage != "" {
		image = localImagePath
	} else if otherRepo != "" {
		image = fmt.Sprintf("%s%s", otherRepo, utils.DefaultLocalChartRegistry)
	} else if onlineImage == "" {
		image = localImagePath
	} else {
		image = fmt.Sprintf("%s/%s", utils.DefaultThirdMirror, utils.DefaultLocalChartRegistry)
	}

	err := server.StartChartRegistry(utils.LocalChartRegistryName, image, chartRegistryPort, chartDataDirectory)
	if err != nil {
		if rmErr := server.RemoveChartRegistry(utils.LocalChartRegistryName); rmErr != nil {
			log.Warnf("Remove chart registry failed before retry: %v", rmErr)
		}
		for i := 0; i < 3; i++ {
			log.Infof("Retrying to start chart registry (%d/3)...", i+1)
			time.Sleep(utils.DefaultSleepSeconds * time.Second)
			err = server.StartChartRegistry(utils.LocalChartRegistryName, image, chartRegistryPort, chartDataDirectory)
			if err == nil {
				return nil
			}
			log.Debugf("Start chart registry retry %d/3 failed: %v", i+1, err)
			if rmErr := server.RemoveChartRegistry(utils.LocalChartRegistryName); rmErr != nil {
				log.Warnf("Remove chart registry failed after retry: %v", rmErr)
			}
		}
		return fmt.Errorf("start chart registry: %w", err)
	}
	return nil
}

func NFSServer(imageRegistryPort, otherRepo, onlineImage string) error {
	var image string
	localImagePath := fmt.Sprintf("127.0.0.1:%s/%s/%s", imageRegistryPort, bkecommon.ImageRegistryKubernetes, utils.DefaultLocalNFSRegistry)
	if otherRepo != "" {
		image = fmt.Sprintf("%s%s", otherRepo, utils.DefaultLocalNFSRegistry)
	} else if onlineImage == "" {
		image = localImagePath
	} else {
		image = fmt.Sprintf("%s/%s", utils.DefaultThirdMirror, utils.DefaultLocalNFSRegistry)
	}

	err := server.StartNFSServer(utils.LocalNFSRegistryName, image, nfsDataDirectory)
	if err != nil {
		if rmErr := server.RemoveNFSServer(utils.LocalNFSRegistryName); rmErr != nil {
			log.Warnf("Remove nfs server failed before retry: %v", rmErr)
		}
		for i := 0; i < 3; i++ {
			log.Infof("Retrying to start nfs server (%d/3)...", i+1)
			time.Sleep(utils.DefaultSleepSeconds * time.Second)
			err = server.StartNFSServer(utils.LocalNFSRegistryName, image, nfsDataDirectory)
			if err == nil {
				return nil
			}
			log.Debugf("Start nfs server retry %d/3 failed: %v", i+1, err)
			if rmErr := server.RemoveNFSServer(utils.LocalNFSRegistryName); rmErr != nil {
				log.Warnf("Remove nfs server failed after retry: %v", rmErr)
			}
		}
		return fmt.Errorf("start nfs server: %w", err)
	}
	return nil
}

// ContainerdFileResult 包含 VerifyContainerdFile 的结果
type ContainerdFileResult struct {
	FilePath       string
	ContainerdList []string
	CniPluginList  []string
}

// VerifyContainerdFile verifies and retrieves containerd and CNI plugin files from the repository
func VerifyContainerdFile(localImage string) (ContainerdFileResult, error) {
	result := ContainerdFileResult{
		FilePath: fmt.Sprintf("%s/%s", yumDataDirectory, "files"),
	}
	if localImage != "" {
		result.FilePath = fmt.Sprintf("%s/%s", imageLocalDirectory, "volumes")
	}

	entries, err := os.ReadDir(result.FilePath)
	if err != nil {
		return result, err
	}
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), utils.CniPluginPrefix) {
			result.CniPluginList = append(result.CniPluginList, entry.Name())
			continue
		}
		if err = validation.ValidateCustomExtra(map[string]string{"containerd": entry.Name()}); err != nil {
			continue
		}
		result.ContainerdList = append(result.ContainerdList, entry.Name())
	}
	if len(result.ContainerdList) == 0 {
		return result, fmt.Errorf("a valid containerd cannot be found")
	}
	if len(result.CniPluginList) == 0 {
		return result, fmt.Errorf("a valid cni plugin cannot be found")
	}
	if len(result.ContainerdList) == 1 {
		global.CustomExtra["containerd"] = result.ContainerdList[0]
	}
	cds := strings.Split(result.ContainerdList[len(result.ContainerdList)-1], "-")
	cds[len(cds)-1] = "{.arch}.tar.gz"
	if len(result.ContainerdList) > 1 {
		global.CustomExtra["containerd"] = strings.Join(cds, "-")
	}
	sort.Sort(sort.Reverse(sort.StringSlice(result.ContainerdList)))
	sort.Sort(sort.Reverse(sort.StringSlice(result.CniPluginList)))
	return result, nil
}

// 解压初始化源文件
func DecompressionSystemSourceFile() error {
	if utils.Exists(yumDataDirectory) && !utils.DirectoryIsEmpty(yumDataDirectory) {
		log.Infof("skip decompressing source package: file %s already exists, If you want to unzip it again, remove the directory", yumDataDirectory)
		return nil
	}

	if err := cleanYumDataDirectory(); err != nil {
		log.Warnf("Failed to remove yum data directory: %v", err)
	}

	if utils.Exists(yumDataFile) {
		if err := decompressYumDataFile(); err != nil {
			return err
		}
	} else {
		if err := os.Mkdir(yumDataDirectory, utils.DefaultDirPermission); err != nil {
			return err
		}
	}

	if !utils.Exists(yumDataDirectory) {
		return fmt.Errorf("%s, not found", yumDataDirectory)
	}
	return nil
}

// cleanYumDataDirectory 清理 yum 数据目录
func cleanYumDataDirectory() error {
	if err := os.Remove(yumDataDirectory); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// decompressYumDataFile 解压 yum 数据文件
func decompressYumDataFile() error {
	log.Info("Decompressing the source package, this may take several minutes")
	start := time.Now()
	targetTemp := yumDataDirectory + ".tmp"

	if utils.Exists(targetTemp) {
		if err := os.RemoveAll(targetTemp); err != nil {
			return err
		}
	}

	count, err := utils.UnTarWithCount(yumDataFile, targetTemp)
	if err != nil {
		return err
	}

	if err := os.Rename(targetTemp, yumDataDirectory); err != nil {
		return err
	}
	log.Infof("Decompressed source package: %d entries, elapsed=%s",
		count, time.Since(start).Round(time.Second))
	return nil
}
