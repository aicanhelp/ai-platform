/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 * http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package installer

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	yamlutil "k8s.io/apimachinery/pkg/util/yaml"

	"installer-service/pkg/utils/k8sutil"
	"installer-service/pkg/zlog"
)

const (
	// 脚本头部和变量定义部分
	bkeUpgradeScriptHeader = `#!/bin/bash
set -eo pipefail

# Input parameters
PATCH_PATH="%s" # Offline package directory (e.g., /data/patches)
PATCH_NAME="%s" # Offline package filename without extension (e.g., patchV1)

# Combine full filename
FULL_PATCH_FILE="${PATCH_PATH}/${PATCH_NAME}.tar.gz" 
# Switch to the parent directory of the patch file, all subsequent operations will be executed in this directory
cd "${PATCH_PATH}"
PATCH_DIR="${PATCH_NAME}" # Target directory after extraction, now relative to current directory

REGISTRY_TARGET="%s:40443"
REGISTRY_FILES_DIR="/bke/mount/source_registry/files"

echo "[INFO] Upgrade process start"
echo "[INFO] Full Patch File: ${FULL_PATCH_FILE}"
echo "[INFO] Extraction Dir: ${PATCH_DIR}"
`

	// 文件检查和验证部分
	bkeUpgradeScriptFileCheck = `
echo "--- 1. Check offline package file ---"
# Use absolute path to check if file exists
if [ ! -f "${FULL_PATCH_FILE}" ]; then
	echo "Error: Offline package file ${FULL_PATCH_FILE} does not exist." >&2
	exit 1
fi
`

	// 解压离线包部分
	bkeUpgradeScriptExtract = `
echo "--- 2. Extract offline package ---"
# Ensure target directory does not exist, remove if present to avoid leftovers
rm -rf "${PATCH_DIR}"
if [ -d "${PATCH_DIR}" ]; then
    echo "Error: Failed to remove existing directory ${PATCH_DIR}." >&2
    exit 1
fi
mkdir -p "${PATCH_DIR}"

# Extract offline package to PATCH_DIR subdirectory in current directory
tar -xzvf "${FULL_PATCH_FILE}"

if [ $? -ne 0 ]; then
	echo "Error: Failed to extract offline package ${FULL_PATCH_FILE}." >&2
	exit 1
fi
`

	// 镜像同步部分
	bkeUpgradeScriptSyncImages = `
echo "--- 3. Sync images to bootstrap node's image registry service ---"
# Use absolute path reference for extracted directory in case bke command changes working directory
ABSOLUTE_PATCH_DIR="$(pwd)/${PATCH_DIR}"
echo "[DEBUG] Absolute path: ${ABSOLUTE_PATCH_DIR}"
echo "[DEBUG] Check manifests.yaml file:"
ls -la "${ABSOLUTE_PATCH_DIR}/manifests.yaml" || echo "File does not exist"

# Use absolute path
bke registry patch --source "${ABSOLUTE_PATCH_DIR}" --target "${REGISTRY_TARGET}"

if [ $? -ne 0 ]; then
	echo "Error: bke registry patch image synchronization failed." >&2
	exit 1
fi
`

	// 文件复制和完成部分
	bkeUpgradeScriptCopyFiles = `
echo "--- 4. Copy specified binary files to BKE mount directory ---"
if [ -d "${REGISTRY_FILES_DIR}" ]; then
	echo "Finding and copying all files containing 'containerd' or 'kubelet' or 'kubectl'..."
	echo "source dir: ${ABSOLUTE_PATCH_DIR}"
	echo "target dir: ${REGISTRY_FILES_DIR}"
	echo "Start copying containerd/kubelet/kubectl files..."

	COMPONENTS=("containerd" "kubelet" "kubectl")
	for component in "${COMPONENTS[@]}"; do
		echo ""
		echo "Looking for ${component} files..."
		FILES=$(find "${ABSOLUTE_PATCH_DIR}" -type f -name "*${component}*")

		if [ -n "$FILES" ]; then
			COUNT=$(echo "$FILES" | wc -l)
			echo "Found $COUNT files for ${component}:"
			echo "$FILES"
			echo ""

			echo "Copying ${component} files..."
			for file in $FILES; do
				if [ -f "$file" ]; then
					echo "Copy: $file"
					cp -fv "$file" "${REGISTRY_FILES_DIR}/"
					if [ $? -ne 0 ]; then
						echo "Warning: Failed to copy $file" >&2
					fi
				fi
			done
			echo "Successfully copied files containing '${component}'."
		else
			echo "No files found for ${component}."
		fi
	done
	
	echo ""
	echo "File copy operation completed."
else
	echo "Error: Target directory ${REGISTRY_FILES_DIR} does not exist." >&2
	exit 1
fi

echo "--- Deployment preparation process completed ---"
`

	// Validation constants
	emptyStringErrorTemplate      = "%s cannot be empty"
	pathNotExistErrorTemplate     = "path does not exist: %s"
	pathAccessErrorTemplate       = "failed to access path: %s, error: %v"
	pathNotDirectoryErrorTemplate = "path is not a directory: %s"
	fileNotExistErrorTemplate     = "file does not exist: %s"
	fileAccessErrorTemplate       = "failed to access file: %s, error: %v"
	fileNotRegularErrorTemplate   = "file is not a regular file: %s"
	gzipFormatErrorTemplate       = "invalid gzip format in file: %s, error: %v"
	tarFormatErrorTemplate        = "invalid tar format in file: %s, error: %v"
)

// buildUpgradeScript 构建完整的升级脚本
func buildUpgradeScript(patchPath, patchName, bootstrapIp string, online bool) string {
	header := fmt.Sprintf(bkeUpgradeScriptHeader, patchPath, patchName, bootstrapIp)
	ret := header + bkeUpgradeScriptFileCheck + bkeUpgradeScriptExtract
	if !online {
		ret += bkeUpgradeScriptSyncImages // 离线场景才需要镜像同步
	}
	return ret + bkeUpgradeScriptCopyFiles
}

// validateInputParameters 验证输入参数
func validateInputParameters(patchPath, patchName string) error {
	if patchPath == "" {
		return fmt.Errorf(emptyStringErrorTemplate, "patch path")
	}
	if patchName == "" {
		return fmt.Errorf(emptyStringErrorTemplate, "patch name")
	}
	return nil
}

// normalizePath 标准化路径，防止路径遍历攻击
func normalizePath(path string) (string, error) {
	if path == "" {
		return "", fmt.Errorf(emptyStringErrorTemplate, "path")
	}

	// 清理路径，移除多余的路径分隔符和相对路径引用
	cleanPath := filepath.Clean(path)

	// 转换为绝对路径以确保路径标准化
	absPath, err := filepath.Abs(cleanPath)
	if err != nil {
		return "", fmt.Errorf("failed to get absolute path for %s: %v", path, err)
	}

	// 检查是否包含可疑的路径遍历模式
	if strings.Contains(absPath, "..") {
		return "", fmt.Errorf("path contains suspicious pattern: %s", path)
	}

	return absPath, nil
}

// normalizeFileName 标准化文件名，防止命令注入
func normalizeFileName(fileName string) (string, error) {
	if fileName == "" {
		return "", fmt.Errorf(emptyStringErrorTemplate, "file name")
	}

	// 移除路径分隔符，防止目录遍历
	cleanName := filepath.Base(fileName)

	// 检查文件名是否包含可疑字符
	suspiciousChars := []string{"/", "\\", "..", "|", "&", ";", "`", "$", "(", ")", "<", ">"}
	for _, char := range suspiciousChars {
		if strings.Contains(cleanName, char) {
			return "", fmt.Errorf("file name contains suspicious character '%s': %s", char, fileName)
		}
	}

	return cleanName, nil
}

// validateDirectoryExists 验证目录是否存在
func validateDirectoryExists(path string) error {
	fileInfo, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf(pathNotExistErrorTemplate, path)
		}
		return fmt.Errorf(pathAccessErrorTemplate, path, err)
	}
	if !fileInfo.IsDir() {
		return fmt.Errorf(pathNotDirectoryErrorTemplate, path)
	}
	return nil
}

// validateTarGzFormat 验证tar.gz格式
func validateTarGzFormat(filePath string) error {
	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf(fileAccessErrorTemplate, filePath, err)
	}
	defer file.Close()

	gzipReader, err := gzip.NewReader(file)
	if err != nil {
		return fmt.Errorf(gzipFormatErrorTemplate, filePath, err)
	}
	defer gzipReader.Close()

	tarReader := tar.NewReader(gzipReader)
	_, err = tarReader.Next()
	if err != nil && err != io.EOF {
		return fmt.Errorf(tarFormatErrorTemplate, filePath, err)
	}

	return nil
}

// validateTarGzFile 验证tar.gz文件
func validateTarGzFile(filePath string) error {
	fileInfo, err := os.Stat(filePath)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf(fileNotExistErrorTemplate, filePath)
		}
		return fmt.Errorf(fileAccessErrorTemplate, filePath, err)
	}

	if !fileInfo.Mode().IsRegular() {
		return fmt.Errorf(fileNotRegularErrorTemplate, filePath)
	}

	return validateTarGzFormat(filePath)
}

// validatePatchFiles 验证补丁包路径和文件
func validatePatchFiles(patchPath, patchName string) error {
	if err := validateInputParameters(patchPath, patchName); err != nil {
		return err
	}

	normalizedPath, err := normalizePath(patchPath)
	if err != nil {
		return fmt.Errorf("path normalization failed: %v", err)
	}

	normalizedName, err := normalizeFileName(patchName)
	if err != nil {
		return fmt.Errorf("file name normalization failed: %v", err)
	}

	if err := validateDirectoryExists(normalizedPath); err != nil {
		return err
	}

	tarGzPath := filepath.Join(normalizedPath, normalizedName+".tar.gz")
	if err := validateTarGzFile(tarGzPath); err != nil {
		return err
	}

	return nil
}

// executeUpgradeScript 执行升级脚本
func executeUpgradeScript(scriptContent string) error {
	cmd := exec.Command("bash", "-c", scriptContent)
	output, err := cmd.CombinedOutput()

	if err != nil {
		errorMessage := fmt.Sprintf("Auto upgrade preparation failed. Error: %v. Script output: %s",
			err, string(output))
		zlog.Errorf(errorMessage)
		return fmt.Errorf("auto upgrade script failed: %s", string(output))
	}

	zlog.Infof("Auto upgrade script completed successfully. Output: %s", string(output))
	return nil
}

func (c *installerClient) getBootstrapIpFromUnstructured(object string) (string, error) {
	buffer := bytes.NewBuffer([]byte(object))
	decoder := yamlutil.NewYAMLOrJSONDecoder(buffer, decoderBufferSize)

	var rawObj runtime.RawExtension
	if err := decoder.Decode(&rawObj); err != nil {
		zlog.Error("decode raw object failed: %v", err)
		return "", err
	}
	obj, _, err := unstructured.UnstructuredJSONScheme.Decode(rawObj.Raw, nil, nil)
	if err != nil {
		zlog.Error("decode unstructured object failed: %v", err)
		return "", err
	}

	unstructuredObj, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
	if err != nil {
		zlog.Error("convert unstructured object failed: %v", err)
		return "", err
	}
	unstruct := &unstructured.Unstructured{Object: unstructuredObj}
	ip, found, err := unstructured.NestedString(unstruct.Object, "spec", "clusterConfig", "cluster", "imageRepo", "ip")
	if err != nil {
		return "", errors.Wrap(err, "failed to extract bootstrap ip from spec.clusterConfig.cluster.imageRepo.ip")
	}
	if !found {
		return "", errors.New("missing field: spec.clusterConfig.cluster.imageRepo.ip")
	}

	ip = strings.TrimSpace(ip)
	if ip == "" {
		return "", errors.New("spec.clusterConfig.cluster.imageRepo.ip is empty, please provide a valid IP address")
	}

	if net.ParseIP(ip) == nil {
		return "", errors.Errorf("invalid IP address format: %q", ip)
	}

	return ip, nil
}

// getBootstrapIpFromCluster fetches the BKECluster CR in the cluster namespace (clusterName)
// and extracts spec.clusterConfig.cluster.imageRepo.ip as the bootstrap IP.
func (c *installerClient) getBootstrapIpFromCluster(clusterName string) (string, error) {
	if c.dynamicClient == nil {
		return "", fmt.Errorf("dynamic client is nil")
	}

	bc, err := c.dynamicClient.Resource(gvr).Namespace(clusterName).Get(context.TODO(), clusterName, metav1.GetOptions{})
	if err != nil {
		return "", fmt.Errorf("failed to get BKECluster %s/%s: %v", clusterName, clusterName, err)
	}

	ip, found, err := unstructured.NestedString(bc.Object, "spec", "clusterConfig", "cluster", "imageRepo", "ip")
	if err != nil {
		return "", fmt.Errorf("failed to extract bootstrap ip from cluster %s/%s: %v", clusterName, clusterName, err)
	}
	if !found {
		return "", fmt.Errorf("missing field: spec.clusterConfig.cluster.imageRepo.ip in cluster %s", clusterName)
	}

	ip = strings.TrimSpace(ip)
	if ip == "" {
		return "", fmt.Errorf("spec.clusterConfig.cluster.imageRepo.ip is empty for cluster %s", clusterName)
	}

	if net.ParseIP(ip) == nil {
		return "", errors.Errorf("invalid IP address format: %q", ip)
	}

	return ip, nil
}

// AutoUpgradePatchPrepare 同步执行升级准备脚本
func (c *installerClient) AutoUpgradePatchPrepare(clusterName string, req AutoUpgradeRequest) error {
	if req.PatchDir == "" || req.PatchName == "" {
		return fmt.Errorf("patchDir and patchName cannot be empty")
	}

	if err := validatePatchFiles(req.PatchDir, req.PatchName); err != nil {
		return fmt.Errorf("patch file validation failed: %v", err)
	}

	normalizedPath, err := normalizePath(req.PatchDir)
	if err != nil {
		return fmt.Errorf("path normalization failed: %v", err)
	}

	normalizedName, err := normalizeFileName(req.PatchName)
	if err != nil {
		return fmt.Errorf("file name normalization failed: %v", err)
	}

	bootstrapIp, err := c.getBootstrapIpFromCluster(clusterName)
	if err != nil {
		return fmt.Errorf("get bootstrap ip failed: %v", err)
	}

	configMap, err := k8sutil.GetConfigMap(c.clientset, BKEConfigCmKey().Name, BKEConfigCmKey().Namespace)
	if err != nil {
		return fmt.Errorf("get configmap %s/%s failed %v", BKEConfigCmKey().Namespace, BKEConfigCmKey().Name, err)
	}

	online, err := IsOnlineMode(configMap)
	if err != nil {
		return fmt.Errorf("judge is online failed: %v", err)
	}

	scriptContent := buildUpgradeScript(normalizedPath, normalizedName, bootstrapIp, online)
	return executeUpgradeScript(scriptContent)
}
