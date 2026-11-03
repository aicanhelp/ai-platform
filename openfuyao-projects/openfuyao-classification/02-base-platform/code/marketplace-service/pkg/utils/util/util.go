/*
 *
 *  * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  * openFuyao is licensed under Mulan PSL v2.
 *  * You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  * You may obtain a copy of Mulan PSL v2 at:
 *  *          http://license.coscl.org.cn/MulanPSL2
 *  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *  * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *  * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *  * See the Mulan PSL v2 for more details.
 *
 */

/*
Package util include marketplace-service level util function
*/
package util

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"fmt"
	"io"
	"io/ioutil"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"k8s.io/client-go/kubernetes"

	"marketplace-service/pkg/constant"
	"marketplace-service/pkg/models/helm"
	"marketplace-service/pkg/utils/k8sutil"
	"marketplace-service/pkg/zlog"
)

// ClearByte clear byte slice by setting every index to zero
func ClearByte(value []byte) {
	for i := range value {
		value[i] = 0
	}
}

// Contains return if string slice contains string
func Contains(s []string, str string) bool {
	for _, a := range s {
		if a == str {
			return true
		}
	}
	return false
}

// NotContains checks if 'str' is not in the slice 's'
func NotContains(s []string, str string) bool {
	for _, v := range s {
		if v == str {
			return false
		}
	}
	return true
}

// ContainsOne return if leftArr slice contains one of the element in rightArr
func ContainsOne(leftArr, rightArr []string) bool {
	set := make(map[string]struct{})
	for _, s := range leftArr {
		set[s] = struct{}{}
	}

	for _, s := range rightArr {
		if _, ok := set[s]; ok {
			return true
		}
	}
	return false
}

// ContainsAll return if outer string slice contains inner string slice
func ContainsAll(outer, inner []string) bool {
	set := make(map[string]struct{})
	for _, s := range outer {
		set[s] = struct{}{}
	}

	for _, s := range inner {
		if _, ok := set[s]; !ok {
			return false
		}
	}
	return true
}

const (
	marketplaceScenes         = "marketplace-scenes"
	localHarborDisplayName    = "local-harbor-display-name"
	localHarborHost           = "local-harbor-host"
	localHarborProject        = "local-harbor-project"
	officialHarborDisplayName = "official-harbor-display-name"
	officialHarborHost        = "official-harbor-host"
	officialHarborProject     = "official-harbor-project"
	officialHarborTagsURL     = "official-harbor-tags-URL"
	oauthServerHost           = "oauth-server-host"
	marketplaceServiceHost    = "marketplace-service-host"
	consoleWebsiteHost        = "console-website-host"
	alertHost                 = "alert-host"
	monitoringHost            = "monitoring-host"
	insecureSkipVerify        = "insecure-skip-verify"
	serverName                = "server-name"
	chartLimit                = "chart-limit"
	configDir                 = "/etc/marketplace-service/fuyao-config"
)

// GetMarketplaceServiceConfig parse the configmap for marketplace-service configuration from the cluster
func GetMarketplaceServiceConfig(c kubernetes.Interface) (*helm.MarketplaceServiceConfig, error) {
	configMap, err := k8sutil.GetConfigMap(c, constant.MarketplaceServiceConfigmap,
		constant.MarketplaceServiceDefaultNamespace)
	if err != nil {
		zlog.Warnf("failed to read config map from k8s cluster  %v", err)
		MarketplaceServiceConfig, err := getConfigFromPod()
		if err != nil {
			zlog.Warnf("failed to read config map from container %v", err)
			return nil, err
		}
		return MarketplaceServiceConfig, err
	}

	MarketplaceServiceConfig := parseConfig(configMap.Data)
	return MarketplaceServiceConfig, nil
}

func parseConfig(MarketplaceServiceConfigMap map[string]string) *helm.MarketplaceServiceConfig {
	scenes := parseScenesConfig(MarketplaceServiceConfigMap[marketplaceScenes])

	return &helm.MarketplaceServiceConfig{
		LocalHarborDisplayName:    MarketplaceServiceConfigMap[localHarborDisplayName],
		LocalHarborHost:           MarketplaceServiceConfigMap[localHarborHost],
		LocalHarborProject:        MarketplaceServiceConfigMap[localHarborProject],
		OfficialHarborDisplayName: MarketplaceServiceConfigMap[officialHarborDisplayName],
		OfficialHarborHost:        MarketplaceServiceConfigMap[officialHarborHost],
		OfficialHarborProject:     MarketplaceServiceConfigMap[officialHarborProject],
		OfficialHarborTagsURL:     MarketplaceServiceConfigMap[officialHarborTagsURL],
		MarketplaceScenes:         scenes,
		OAuthServerHost:           MarketplaceServiceConfigMap[oauthServerHost],
		ConsoleServerHost:         MarketplaceServiceConfigMap[marketplaceServiceHost],
		ConsoleWebsiteHost:        MarketplaceServiceConfigMap[consoleWebsiteHost],
		AlertHost:                 MarketplaceServiceConfigMap[alertHost],
		MonitoringHost:            MarketplaceServiceConfigMap[monitoringHost],
		InsecureSkipVerify:        MarketplaceServiceConfigMap[insecureSkipVerify],
		ServerName:                MarketplaceServiceConfigMap[serverName],
		ChartLimit:                MarketplaceServiceConfigMap[chartLimit],
	}
}

func parseScenesConfig(rawConfig string) []string {
	lines := strings.Split(rawConfig, "\n")
	var scenes []string
	for _, line := range lines {
		trimmedLine := strings.TrimSpace(line)
		if strings.HasPrefix(trimmedLine, `-`) {
			trimmedLine = strings.TrimPrefix(trimmedLine, `- `)
			scenes = append(scenes, strings.Trim(trimmedLine, `\`))
		}
	}
	return scenes
}

func getConfigFromPod() (*helm.MarketplaceServiceConfig, error) {
	files, err := ioutil.ReadDir(configDir)
	if err != nil {
		return nil, err
	}
	MarketplaceServiceConfigMap := make(map[string]string)
	for _, file := range files {
		if !file.IsDir() {
			filepath := configDir + "/" + file.Name()
			content, err := ioutil.ReadFile(filepath)
			if err != nil {
				zlog.Warnf("Error reading file %s: %v\n", filepath, err)
				continue
			}
			MarketplaceServiceConfigMap[file.Name()] = string(content)
		}
	}
	return parseConfig(MarketplaceServiceConfigMap), nil
}

// IsValidSearchParam is input length valid for search
func IsValidSearchParam(searchParam string) bool {
	if len(searchParam) > constant.SearchParamLengthLimit {
		return false
	}
	return true
}

// IsHTTPClientError check if the status code is 4xx
func IsHTTPClientError(statusCode int) bool {
	return statusCode >= http.StatusBadRequest && statusCode < http.StatusInternalServerError
}

func getFilePartFromMultipart(multipartFile multipart.File, fileHeader *multipart.FileHeader) (io.Reader, error) {
	var buffer bytes.Buffer
	_, err := io.Copy(&buffer, multipartFile)
	if err != nil {
		return nil, fmt.Errorf("failed to copy multipart file content: %v", err)
	}

	reader := multipart.NewReader(&buffer, fileHeader.Header.Get("boundary"))

	for {
		part, err := reader.NextPart()
		if err == io.EOF {
			break // 读取到最后一部分
		}
		if err != nil {
			return nil, fmt.Errorf("failed to read next part: %v", err)
		}

		if part.FormName() == "file" {
			// 返回 part，它实现了 io.Reader，可以直接传递给 gzip.NewReader
			return part, nil
		}
	}

	return nil, fmt.Errorf("file part not found")
}

// ValidateTarGzFile 检查压缩文件，防止压缩包炸弹和路径穿越攻击
func ValidateTarGzFile(file multipart.File, fileHeader *multipart.FileHeader, maxFileSize int64) (bool, error) {
	// 读取文件内容到内存中
	buf := new(bytes.Buffer)
	if _, err := io.Copy(buf, file); err != nil {
		return false, err
	}

	// 创建gzip阅读器
	gzipReader, err := gzip.NewReader(buf)
	if err != nil {
		return false, err
	}
	defer gzipReader.Close()

	// 创建tar阅读器
	tarReader := tar.NewReader(gzipReader)

	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return false, err
		}

		// 检查文件大小
		if header.Size > maxFileSize {
			return false, fmt.Errorf("file %s is too large", header.Name)
		}

		// 检查文件路径，防止路径穿越攻击
		if !isValidPath(header.Name) {
			return false, fmt.Errorf("illegal file path: %s", header.Name)
		}
	}

	return true, nil
}

// isValidPath 检查路径是否有效，防止路径穿越攻击
func isValidPath(filePath string) bool {
	cleanedPath := filepath.Clean(filePath)
	return !strings.HasPrefix(cleanedPath, "..")
}

// CheckFileSize reads the file in chunks and checks if its size exceeds the limit
func CheckFileSize(file multipart.File, bufferSize, maxFileSize int64) (bool, error) {
	var size int64
	buffer := make([]byte, bufferSize)
	for {
		n, err := (file).Read(buffer)
		size += int64(n)
		if err != nil {
			if err == io.EOF {
				break
			}
			return false, err
		}
		// If size exceeds the limit, return false
		if size > maxFileSize {
			return false, nil
		}
	}
	// Final check to ensure size limit was not exceeded at EOF
	if size > maxFileSize {
		return false, nil
	}
	return true, nil
}

func readFile(filePath string) ([]byte, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer func(file *os.File) {
		err := file.Close()
		if err != nil {
			zlog.Errorf("close file error, %v", err)
		}
	}(file)

	content, err := io.ReadAll(file)
	if err != nil {
		return nil, err
	}
	return content, nil
}

var escapeMap = map[string]string{
	"\n": `\n`,
	"\r": `\r`,
	"\t": `\t`,
	"\"": `\"`,
	"\b": `\b`,
	"\f": `\f`,
	"\v": `\v`,
}

// EscapeSpecialChars escape special characters
func EscapeSpecialChars(input string) string {
	var buffer bytes.Buffer

	for _, char := range input {
		str := string(char)
		if escaped, exists := escapeMap[str]; exists {
			buffer.WriteString(escaped)
		} else {
			buffer.WriteString(str)
		}
	}

	return buffer.String()
}

// SanitizeArray sanitize string array with special chars
func SanitizeArray(arr []string) []string {
	for i := range arr {
		arr[i] = EscapeSpecialChars(arr[i])
	}
	return arr
}
