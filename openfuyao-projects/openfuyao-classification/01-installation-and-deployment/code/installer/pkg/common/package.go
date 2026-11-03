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
	"archive/tar"
	"compress/gzip"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"strings"

	"installer/pkg/constants"
	fuyaov1beta1 "installer/pkg/fuyao/v1beta1/cluster"
	"installer/pkg/utils"
	"installer/pkg/zlog"
)

// GetOptPkgPathByNode 根据节点获取安装包路径
func GetOptPkgPathByNode(node *fuyaov1beta1.Node) (string, error) {
	pkgName, err := GetPackageName(node)
	if err != nil {
		return "", err
	}

	pkgPath := filepath.Join(constants.OptPath, constants.OpenFuyao, pkgName)
	return pkgPath, nil
}

// GetPackageName 获取安装包名字，如果没有x86架构则获取arm64架构
func GetPackageName(node *fuyaov1beta1.Node) (string, error) {
	absPath, err := GetAbsPath()
	if err != nil {
		return "", err
	}
	PkgRawSavePath := filepath.Join(absPath, constants.Asset)

	osName, osVersion, cpuArch := node.OsInfo()
	packageName := fmt.Sprintf("%s-%s-%s-%s", constants.Fuyao, osName, osVersion, cpuArch)
	pkgFile := filepath.Join(PkgRawSavePath, packageName+constants.TarGz)
	_, err = os.Stat(pkgFile)
	if err != nil && os.IsNotExist(err) {
		if cpuArch == constants.ARM64 {
			packageName = fmt.Sprintf("%s-%s-%s-%s", constants.Fuyao, osName, osVersion, constants.AMD64)
		} else {
			packageName = fmt.Sprintf("%s-%s-%s-%s", constants.Fuyao, osName, osVersion, constants.ARM64)
		}
	} else if err != nil && !os.IsNotExist(err) {
		return "", fmt.Errorf(fmt.Sprintf("Installation package not found, operating system is %s, operating system version is %s", osName, osVersion))
	} else {
		zlog.Infof("installation package found, os is %s, os version is %s, cpu architecture is %s", osName,
			osVersion, cpuArch)
	}

	pkgFile = filepath.Join(PkgRawSavePath, packageName+constants.TarGz)
	_, err = os.Stat(pkgFile)
	if err != nil && os.IsNotExist(err) {
		return "", fmt.Errorf(fmt.Sprintf("Installation package not found, operating system is %s, operating system version is %s", osName, osVersion))
	} else if err != nil && !os.IsNotExist(err) {
		return "", err
	}

	return packageName, nil
}

// UnCompressPkgToOptPath 解压一个代码包到/opt/openFuyao
func UnCompressPkgToOptPath(node *fuyaov1beta1.Node) error {
	pkgName, err := GetPackageName(node)
	if err != nil {
		return err
	}

	openFuyaoPath := filepath.Join(constants.OptPath, constants.OpenFuyao)
	// 支持非root安装赋予777权限
	if err := utils.MkdirAll(openFuyaoPath, constants.RwxRwxRwx); err != nil {
		return err
	}

	assetPath, err := GetAssetAbsPathReal()
	if err != nil {
		return err
	}
	return UnCompressTarGZ(filepath.Join(assetPath, pkgName+constants.TarGz), openFuyaoPath)
}

// UnCompressOnePackage 解压一个代码包
func UnCompressOnePackage() (string, string, error) {
	pkgPath := filepath.Join(constants.OptPath, constants.OpenFuyao, constants.Tmp)
	if err := utils.MkdirAll(pkgPath, constants.Rwx); err != nil {
		return "", "", err
	}
	pkgName, err := UnCompressOnePackageRandom(pkgPath)
	if err != nil {
		return "", "", err
	}
	return pkgPath, pkgName, err
}

// UnCompressOnePackageRandom 随机解压一个部署包
func UnCompressOnePackageRandom(targetPath string) (string, error) {
	assetPath, err := GetAssetAbsPathReal()
	if err != nil {
		return "", err
	}
	fileNameList, err := ListFiles(assetPath)
	if err != nil {
		return "", err
	}
	for _, fileName := range fileNameList {
		if strings.Contains(fileName, constants.Fuyao) && strings.HasSuffix(fileName, constants.TarGz) {
			err := UnCompressTarGZ(fileName, targetPath)
			return strings.TrimRight(filepath.Base(fileName), constants.TarGz), err
		}
	}
	return "", fmt.Errorf("fuyao installation package not found")
}

// CopyLocalFile 将一个文件拷贝到另一个文件内
// src 源文件
// dst 目标文件
func CopyLocalFile(src, dst string) error {
	dstDir := filepath.Dir(dst)
	err := utils.MkdirAll(dstDir, constants.Rwx)
	if err != nil {
		return err
	}
	return CopyDir(src, dst)
}

// CopyLocalDir 将一个文件夹下的所有文件和目录原封不动地拷贝到另一个文件夹下
// src 源文件夹
// dst 目标文件夹
func CopyLocalDir(src, dst string) error {
	err := utils.MkdirAll(dst, constants.Rwx)
	if err != nil {
		return err
	}
	return CopyDir(src, dst)
}

// CopyDir 将一个文件夹下的所有文件和目录原封不动地拷贝到另一个文件夹下，或将一个文件拷贝到另一个文件内
// src 原文件或文件目录
// dst 目标文件或目录
func CopyDir(src, dst string) error {
	if _, err := os.Stat(src); os.IsNotExist(err) {
		return fmt.Errorf("src path %s is not exist", src)
	}
	zlog.Infof("copy %s to %s", src, dst)

	err := filepath.WalkDir(src, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		relPath, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}

		dstPath := filepath.Join(dst, relPath)

		if d.IsDir() {
			err := utils.MkdirAll(dstPath, constants.RwxRwxRwx)
			if err != nil {
				return err
			}
		} else {
			srcFile, err := os.Open(path)
			if err != nil {
				return err
			}
			defer srcFile.Close()

			dstFile, err := os.Create(dstPath)
			if err != nil {
				return err
			}
			defer dstFile.Close()
			err = dstFile.Chmod(constants.RwxRwxRwx)
			if err != nil {
				return fmt.Errorf("failed to set file permission: %v", err)
			}

			_, err = io.Copy(dstFile, srcFile)
			if err != nil {
				return err
			}
		}

		return nil
	})

	return err
}

// CompressTarGZ 将文件或目录压缩为tar.gz文件
// src 文件名或者目录
// dst 压缩后的路径和文件名
func CompressTarGZ(src, dst string) (err error) {
	// 清理路径字符串
	src = path.Clean(src)

	// 获取文件或目录信息
	fileInfo, err := os.Stat(src)
	if err != nil {
		zlog.Error("failed to get src file stat ", src)
		return fmt.Errorf("failed to get src file stat %v", err)
	}

	// 创建目标文件
	file, err := os.Create(dst)
	if err != nil {
		zlog.Error("failed to create target file ", dst)
		return fmt.Errorf("failed to create target file: %v", err)
	}
	defer file.Close()
	err = file.Chmod(constants.RwxRwxRwx)
	if err != nil {
		return fmt.Errorf("failed to set file permission: %v", err)
	}

	// 执行打包操作
	gzipWriter := gzip.NewWriter(file)
	tarWriter := tar.NewWriter(gzipWriter)
	defer func() {
		// 这里要判断是否关闭成功，如果关闭失败，则 .tar.gz 文件可能不完整
		if twErr := tarWriter.Close(); twErr != nil {
			err = twErr
		}
		if gwErr := gzipWriter.Close(); gwErr != nil {
			err = gwErr
		}
		if fwErr := file.Close(); fwErr != nil {
			err = fwErr
		}
	}()

	// 获取要打包的文件或目录的所在位置和名称
	srcBase, srcRelative := path.Split(src)

	// 开始打包
	if fileInfo.IsDir() {
		return tarDir(srcBase, srcRelative, tarWriter, fileInfo)
	}
	return tarFile(srcBase, srcRelative, tarWriter, fileInfo)
}

// tarDir 将 srcBase 目录下的 srcRelative 子目录写入 tarWriter 中，其中 fileInfo 为 srcRelative 子目录信息
func tarDir(srcBase, srcRelative string, tarWriter *tar.Writer, fileInfo os.FileInfo) (err error) {
	// 获取完整路径
	dirPath := path.Join(srcBase, srcRelative)

	// 获取目录下的文件和子目录列表
	entries, err := os.ReadDir(dirPath)
	if err != nil {
		zlog.Error("failed to read dir ", dirPath)
		return fmt.Errorf("failed to read dir %v", err.Error())
	}

	// 开始遍历
	for _, entry := range entries {
		entryInfo, err := entry.Info()
		if err != nil {
			zlog.Error("failed to get entry info ", dirPath)
			return fmt.Errorf("failed to get entry info: %v", err.Error())
		}
		filename := path.Join(srcRelative, entryInfo.Name())
		if entryInfo.IsDir() {
			if err := tarDir(srcBase, filename, tarWriter, entryInfo); err != nil {
				return err
			}
		} else {
			if err := tarFile(srcBase, filename, tarWriter, entryInfo); err != nil {
				return err
			}
		}
	}

	// 写入目录信息
	if len(srcRelative) > 0 {
		header, err := tar.FileInfoHeader(fileInfo, "")
		if err != nil {
			zlog.Error("failed to create compressed file header info", dirPath)
			return fmt.Errorf("failed to create compressed file header info: %v", err.Error())
		}
		header.Name = srcRelative
		if err = tarWriter.WriteHeader(header); err != nil {
			zlog.Error("failed to write directory header info to the compressed file ", dirPath)
			return fmt.Errorf("failed to write directory header info to the compressed file %v", err.Error())
		}
	}

	return nil
}

// tarFile 将 srcBase 目录下的 srcRelative 文件写入 tarWriter 中，其中 fileInfo 为 srcRelative 文件信息
func tarFile(srcBase, srcRelative string, tarWriter *tar.Writer, fileInfo os.FileInfo) (err error) {
	// 获取完整路径
	filepath := path.Join(srcBase, srcRelative)

	// 写入文件标头
	header, err := tar.FileInfoHeader(fileInfo, "")
	if err != nil {
		zlog.Error("failed to create header info of file ", filepath)
		return fmt.Errorf("failed to create header info %v", err.Error())
	}
	header.Name = srcRelative
	if err = tarWriter.WriteHeader(header); err != nil {
		zlog.Error("failed to write header info to the compressed file ", filepath)
		return fmt.Errorf("failed to write header info to the compressed file: %v", err.Error())
	}

	// 写入文件数据
	file, err := os.Open(filepath)
	if err != nil {
		zlog.Error("failed to open file ", filepath)
		return fmt.Errorf("failed to open file: %v", err.Error())
	}
	defer file.Close()
	if _, err = io.Copy(tarWriter, file); err != nil {
		zlog.Error("failed to write data to the compressed file ", filepath)
		return fmt.Errorf("failed to write data to the compressed file: %v", err.Error())
	}

	return nil
}

// UnCompressTarGZ 解压tar.gz文件
// srcFile tar.gz文件，可以包含目录信息，如：./path/example.tar.gz
// dstDir 指定目录，如：./path/folder folder内不会包含压缩文件名
func UnCompressTarGZ(srcFile, dstDir string) (err error) {
	if !strings.HasSuffix(srcFile, constants.TarGz) {
		return fmt.Errorf("srcFile is not a tar.gz file")
	}

	// 清理路径字符串
	dstDir = path.Clean(dstDir)

	// 打开压缩文件
	file, err := os.Open(srcFile)
	if err != nil {
		zlog.Error("failed to open zip file ", srcFile)
		return fmt.Errorf("failed to open zip file %v", err.Error())
	}
	defer file.Close()

	// 执行解压操作
	gzipReader, err := gzip.NewReader(file)
	if err != nil {
		zlog.Error("failed to uncompress file ", srcFile)
		return fmt.Errorf("failed to uncompress file: %v", err.Error())
	}
	defer gzipReader.Close()
	tarReader := tar.NewReader(gzipReader)

	for {
		header, err := tarReader.Next()
		if err != io.EOF {
			if err != nil {
				return fmt.Errorf("failed to read compress file: %v", err.Error())
			}

			// 获取绝对路径
			dstFullPath := path.Join(dstDir, header.Name)

			if header.Typeflag == tar.TypeDir {
				// 创建目录
				// 支持非root用户安装，赋予777权限
				if err = utils.MkdirAll(dstFullPath, constants.RwxRwxRwx); err != nil {
					return err
				}
				// 设置目录权限
				// 支持非root用户安装，赋予777权限
				if err = os.Chmod(dstFullPath, constants.RwxRwxRwx); err != nil {
					return err
				}
			} else {
				// 创建文件所在的目录
				if err = utils.MkdirAll(path.Dir(dstFullPath), constants.RwxRwxRwx); err != nil {
					return err
				}
				// 将 tarReader 中的数据写入文件中
				if err := unTarFile(dstFullPath, tarReader); err != nil {
					return err
				}
				// 设置文件权限
				// 支持非root用户安装，赋予777权限
				if err = os.Chmod(dstFullPath, constants.RwxRwxRwx); err != nil {
					return err
				}
			}
		} else {
			break
		}
	}
	return nil
}

// unTarFile 从 tarReader 读取解压后的数据并写入 dstFile 文件中
func unTarFile(dstFile string, tarReader *tar.Reader) error {
	file, err := os.Create(dstFile)
	if err != nil {
		zlog.Error("failed to create file ", dstFile)
		return fmt.Errorf("failed to create file, when uncompress file: %v", err.Error())
	}
	defer file.Close()
	if err := file.Chmod(constants.RwxRxRx); err != nil {
		return fmt.Errorf("failed to set file permission: %v", err)
	}

	_, err = io.Copy(file, tarReader)
	if err != nil {
		zlog.Error("failed to uncompress file ", dstFile)
		return fmt.Errorf("failed to uncompress file: %v", err.Error())
	}

	return nil
}
