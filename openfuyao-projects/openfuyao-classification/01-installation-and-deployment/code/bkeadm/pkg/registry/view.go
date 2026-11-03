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

package registry

import (
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"text/tabwriter"
	"time"

	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

func (op *Options) View() {
	httpClient, baseURL := initHTTPClient(op.Args[0])
	repos := fetchRepositories(httpClient, fmt.Sprintf("%s/v2/_catalog?n=10000", baseURL))
	if repos == nil {
		return
	} else if repos.Repositories == nil || len(repos.Repositories) == 0 {
		fmt.Println("no repositories found")
		return
	}

	headers := []string{"IMAGE", "TAGS", "ARCHITECTURE", "CREATE TIME", "SIZE"}
	rows, exportList := processRepositories(httpClient, baseURL, repos.Repositories, op.Prefix, op.Tags)
	outputResults(op.Export, exportList, headers, rows)
}

// initHTTPClient 初始化 HTTP 客户端和基础 URL
func initHTTPClient(addr string) (*http.Client, string) {
	httpClient := &http.Client{}
	baseURL := addr
	if !strings.HasPrefix(baseURL, "http") {
		baseURL = "https://" + baseURL
		httpClient.Transport = &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
		}
	}
	return httpClient, baseURL
}

// processRepositories 处理所有仓库并返回表格行和导出列表
func processRepositories(httpClient *http.Client, baseURL string,
	repositories []string, prefix string, maxTags int) ([][]string, map[string]string) {
	var rows [][]string
	exportList := map[string]string{}

	for _, img := range repositories {
		if len(prefix) > 0 && !strings.HasPrefix(img, prefix) {
			continue
		}
		imgRows, imgExports := processImageTags(httpClient, baseURL, img, maxTags)
		rows = append(rows, imgRows...)
		for k, v := range imgExports {
			exportList[k] += v
		}
	}
	return rows, exportList
}

// processImageTags 处理单个镜像的所有标签
func processImageTags(httpClient *http.Client, baseURL, img string, maxTags int) ([][]string, map[string]string) {
	var rows [][]string
	exportList := map[string]string{}

	tagsURL := fmt.Sprintf("%s/v2/%s/tags/list", baseURL, img)
	tags, err := getImageTags(httpClient, tagsURL, img)
	if err != nil {
		return rows, exportList
	}

	count := 0
	for _, tag := range utils.ReverseArray(tags.Tags) {
		if count > maxTags {
			break
		}
		count++
		row, archKey := processSingleTag(httpClient, baseURL, img, tag)
		if row != nil {
			rows = append(rows, row)
			exportList[archKey] += img + ":" + tag + "\n"
		}
	}
	return rows, exportList
}

// processSingleTag 处理单个标签并返回表格行和架构键
func processSingleTag(httpClient *http.Client, baseURL, img, tag string) ([]string, string) {
	manifest, req, err := fetchImageManifest(httpClient, baseURL, img, tag)
	if err != nil {
		return nil, ""
	}
	arch := extractArchitectures(manifest)

	mest, _, err := fetchImageLayers(httpClient, req, img, tag)
	if err != nil {
		return nil, ""
	}

	request := &ImageProcessRequest{HTTPClient: httpClient, BaseURL: baseURL,
		Image: img, Tag: tag, Manifest: manifest, Schema: mest, Arch: arch}
	mest, size, err := processImageLayers(request)
	if err != nil {
		return nil, ""
	}
	request.Schema = mest

	arch, createTime, err := fetchImageMetadata(request)
	if err != nil {
		return nil, ""
	}

	archStr := strings.TrimRight(arch, ",")
	row := []string{img, tag, archStr, createTime, fmt.Sprintf("%dM", size/1024/1024)}
	archKey := strings.ReplaceAll(archStr, ",", "-") + "_image-list.txt"
	return row, archKey
}

func fetchRepositories(httpClient *http.Client, url string) *repo {
	resp, err := httpClient.Get(url)
	if err != nil {
		log.Errorf("View repositories from %s failed: %v", url, err)
		return nil
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Errorf("Read repository catalog from %s failed: %v", url, err)
		return nil
	}
	var repos repo
	err = json.Unmarshal(body, &repos)
	if err != nil {
		log.Errorf("Unmarshal repository catalog from %s failed: %v", url, err)
		log.Error("view failed:", string(body))
		return nil
	}
	return &repos
}

// 抽取获取镜像tags的通用方法
func getImageTags(httpClient *http.Client, tagsURL, img string) (tagResponse, error) {
	resp, err := httpClient.Get(tagsURL)
	if err != nil {
		log.Warnf("%s get tags failed %v", img, err)
		return tagResponse{}, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Warnf("%s get tags failed %v", img, err)
		return tagResponse{}, err
	}
	var tags tagResponse
	err = json.Unmarshal(body, &tags)
	if err != nil {
		log.Warnf("%s unmarshal tags failed %v", img, err)
		return tagResponse{}, err
	}
	return tags, nil
}

// fetchImageManifest 获取镜像的manifest信息，支持v1和v2版本，返回manifest和request对象
func fetchImageManifest(httpClient *http.Client, baseURL, img, tag string) (*DockerV2List, *http.Request, error) {
	manifestURL := fmt.Sprintf("%s/v2/%s/manifests/%s", baseURL, img, tag)
	req, err := http.NewRequest("GET", manifestURL, nil)
	if err != nil {
		log.Warnf("%s:%s get arch failed %v", img, tag, err)
		return nil, nil, err
	}

	req.Header.Set("Accept", DockerV2ListMediaType)
	resp, err := httpClient.Do(req)
	if err != nil {
		log.Warnf("%s:%s get arch failed %v", img, tag, err)
		return nil, nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Warnf("%s:%s get arch failed %v", img, tag, err)
		return nil, nil, err
	}

	var manifest DockerV2List
	err = json.Unmarshal(body, &manifest)
	if err != nil {
		log.Warnf("%s:%s unmarshal arch failed %v", img, tag, err)
		return nil, nil, err
	}

	// 如果没有找到manifests，尝试获取v1 manifest
	if len(manifest.Manifests) == 0 {
		err := fetchV1Manifest(httpClient, req, img, tag, &manifest)
		if err != nil {
			return nil, nil, err
		}
	}

	return &manifest, req, nil
}

// 从http客户端获取V1版本的镜像清单
func fetchV1Manifest(httpClient *http.Client, req *http.Request, img, tag string, manifest *DockerV2List) error {
	// 设置请求头，接受DockerV1ListMediaType类型的响应
	req.Header.Set("Accept", DockerV1ListMediaType)
	// 发送请求，获取响应
	resp, err := httpClient.Do(req)
	if err != nil {
		// 记录警告日志，请求失败
		log.Warnf("%s:%s get arch failed %v", img, tag, err)
		return err
	}
	// 关闭响应体
	defer resp.Body.Close()

	// 读取响应体
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		// 记录警告日志，读取响应体失败
		log.Warnf("%s:%s get arch failed %v", img, tag, err)
		return err
	}
	// 解析响应体为DockerV2List类型的变量
	err = json.Unmarshal(body, manifest)
	if err != nil {
		// 记录警告日志，解析响应体失败
		log.Warnf("%s:%s unmarshal arch failed %v", img, tag, err)
		return err
	}
	return nil
}

// extractArchitectures 从manifest中提取架构信息，返回逗号分隔的架构字符串
func extractArchitectures(manifest *DockerV2List) string {
	var arch string
	for _, m := range manifest.Manifests {
		if m.Platform.Architecture == "" || m.Platform.Architecture == "unknown" {
			continue
		}
		arch += m.Platform.Architecture + ","
	}
	return arch
}

// fetchImageLayers 获取镜像层信息，返回解析后的schema和请求对象
func fetchImageLayers(httpClient *http.Client, req *http.Request, img, tag string) (*DockerV2Schema, *http.Request, error) {
	req.Header.Set("Accept", DockerV2Schema2MediaType)
	resp, err := httpClient.Do(req)
	if err != nil {
		log.Warnf("%s:%s get image layers %v", img, tag, err)
		return nil, nil, err
	}
	defer resp.Body.Close() // 在方法内部关闭，确保资源释放

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Warnf("%s:%s get image layers failed %v", img, tag, err)
		return nil, nil, err
	}

	var mest DockerV2Schema
	err = json.Unmarshal(body, &mest)
	if err != nil {
		log.Warnf("%s:%s get image layers failed %v", img, tag, err)
		return nil, nil, err
	}

	return &mest, req, nil
}

// processImageLayers 处理镜像层信息，包括兼容性处理和大小计算
func processImageLayers(req *ImageProcessRequest) (*DockerV2Schema, int, error) {
	// 如果没有层信息，尝试使用不同的manifest获取
	if len(req.Schema.Layers) == 0 {
		t := req.Tag
		if len(req.Manifest.Manifests) > 0 {
			t = req.Manifest.Manifests[0].Digest
		}
		manifestURL2 := fmt.Sprintf("%s/v2/%s/manifests/%s", req.BaseURL, req.Image, t)
		req2, err := http.NewRequest("GET", manifestURL2, nil)
		if err != nil {
			log.Warnf("%s:%s get arch failed %v", req.Image, req.Tag, err)
			return nil, 0, err
		}
		// 探查 v1 manifest
		req2.Header.Set("Accept", DockerV1Schema2MediaType)
		resp, err := req.HTTPClient.Do(req2)
		if err != nil {
			log.Warnf("%s:%s get image layers %v", req.Image, req.Tag, err)
			return nil, 0, err
		}
		defer resp.Body.Close()

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			log.Warnf("%s:%s get image layers failed %v", req.Image, req.Tag, err)
			return nil, 0, err
		}
		err = json.Unmarshal(body, req.Schema)
		if err != nil {
			log.Warnf("%s:%s get image layers failed %v", req.Image, req.Tag, err)
			return nil, 0, err
		}
	}

	if len(req.Schema.Layers) == 0 {
		log.Warnf("%s:%s image layers is nil", req.Image, req.Tag)
		return nil, 0, fmt.Errorf("no layers found")
	}

	// 获取镜像每一层的大小
	size := 0
	for _, m := range req.Schema.Layers {
		size += m.Size
	}
	if len(req.Arch) > 0 {
		// 多架构镜像需要乘以2，因为需要存储每个架构的镜像层数据
		const mutiSize = 2
		size = size * mutiSize
	}

	return req.Schema, size, nil
}

// fetchImageMetadata 获取镜像的创建时间和架构信息
func fetchImageMetadata(req *ImageProcessRequest) (string, string, error) {
	configDigest := req.Schema.Config.Digest
	configBlobURL := fmt.Sprintf("%s/v2/%s/blobs/%s", req.BaseURL, req.Image, configDigest)
	httpReq, err := http.NewRequest("GET", configBlobURL, nil)
	if err != nil {
		log.Warnf("%s:%s get create time failed %v", req.Image, req.Tag, err)
		return "", "", err
	}
	httpReq.Header.Set("Accept", DockerV2Schema2MediaType)
	resp, err := req.HTTPClient.Do(httpReq)
	if err != nil {
		log.Warnf("%s:%s get create time failed %v", req.Image, req.Tag, err)
		return "", "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Warnf("%s:%s get create time failed %v", req.Image, req.Tag, err)
		return "", "", err
	}
	var blob blobResponse
	err = json.Unmarshal(body, &blob)
	if err != nil {
		log.Warnf("%s:%s get create time failed %v", req.Image, req.Tag, err)
		return "", "", err
	}
	t, err := time.Parse(time.RFC3339Nano, blob.Created)
	if err != nil {
		log.Warnf("%s:%s parse create time failed %v", req.Image, req.Tag, err)
		return "", "", err
	}
	// 此处8 * time.Hour用于转换为北京时间（UTC+8）并格式化为可读格式
	const timeZoneOffset = 8
	createTime := t.Add(timeZoneOffset * time.Hour).Format("2006-01-02 15:04:05")

	// 如果没有架构信息，使用blob中的架构
	finalArch := req.Arch
	if len(req.Arch) == 0 {
		finalArch = blob.Architecture
	}

	return finalArch, createTime, nil
}

// outputResults 输出结果，支持导出到文件或在控制台显示表格
func outputResults(export bool, exportList map[string]string, headers []string, rows [][]string) {
	if export {
		// 导出文件的权限，rw-r--r--
		const filePerm = 0644
		for k, v := range exportList {
			err := os.WriteFile(k, []byte(v), filePerm)
			if err != nil {
				log.Errorf("Export image list to %s failed: %v", k, err)
				return
			}
		}
		fmt.Println("export success")
	} else {
		PrintTable(headers, rows)
	}
}

// PrintTable 使用tabwriter输出表格到标准输出
func PrintTable(headers []string, rows [][]string) {
	// 2是列之间空格数用于tabwriter输出表格
	const padding = 2
	write := tabwriter.NewWriter(os.Stdout, 0, 0, padding, ' ', 0)
	fmt.Fprintln(write, strings.Join(headers, "\t"))
	for _, row := range rows {
		fmt.Fprintln(write, strings.Join(row, "\t"))
	}
	err := write.Flush()
	if err != nil {
		log.Errorf("Flush image tablewriter failed: %v", err)
	}
}

// ImageProcessRequest 镜像处理请求参数结构体
type ImageProcessRequest struct {
	HTTPClient *http.Client
	BaseURL    string
	Image      string
	Tag        string
	Manifest   *DockerV2List
	Schema     *DockerV2Schema
	Arch       string
}

type repo struct {
	Repositories []string `json:"repositories"`
}

type tagResponse struct {
	Tags []string `json:"tags"`
}

type blobResponse struct {
	Created      string `json:"created"`
	Architecture string `json:"architecture"`
}

func ViewRepoImage(address string, images map[string][]string) (map[string][]string, error) {
	log.Debugf("Current request repository : %s", address)
	result := map[string][]string{}
	// 从images初始化result
	for k, v := range images {
		for _, v1 := range v {
			result[k+":"+v1] = []string{address + "/" + k, v1, "unknown", "unknown", "unknown"}
		}
	}
	httpClient, httpPrefix, err := setupHTTPClient(address)
	if err != nil {
		return nil, err
	}
	// loop repos.Repositories
	for img, tgs := range images {
		for _, tag := range tgs {
			arch, createTime, size := "", "", 0
			// 获取镜像manifest信息
			manifest, req, err := fetchImageManifest(httpClient, httpPrefix, img, tag)
			if err != nil {
				continue
			}
			// 单架构这个循环是不执行的,arch=""
			arch = extractArchitectures(manifest)
			// 获取镜像每一层的信息
			mest, _, err := fetchImageLayers(httpClient, req, img, tag)
			if err != nil {
				continue
			}
			// 创建请求结构体
			request := &ImageProcessRequest{HTTPClient: httpClient, BaseURL: httpPrefix, Image: img, Tag: tag, Manifest: manifest, Schema: mest, Arch: arch}
			// 处理镜像层信息和计算大小
			mest, size, err = processImageLayers(request)
			if err != nil {
				continue
			}
			// 更新结构体中的Schema
			request.Schema = mest
			// 获取创建时间和最终架构信息
			arch, createTime, err = fetchImageMetadata(request)
			if err != nil {
				continue
			}
			result[img+":"+tag] = []string{address + "/" + img, tag, strings.TrimRight(arch, ","), createTime, fmt.Sprintf("%dM", size/1024/1024)}
		}
	}
	return result, nil
}

// setupHTTPClient 设置HTTP客户端并测试连接，支持HTTPS到HTTP的回退
func setupHTTPClient(address string) (*http.Client, string, error) {
	httpClient := &http.Client{}
	httpPrefix := "https://" + address
	url := fmt.Sprintf("%s/v2/_catalog?n=1", httpPrefix)
	resp, err := httpClient.Get(url)
	if err != nil {
		log.Debugf("Switch request address : %v", err)
		httpPrefix = "http://" + address
		url = fmt.Sprintf("%s/v2/_catalog?n=1", httpPrefix)
		httpClient.Transport = &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
		}
		resp, err = httpClient.Get(url)
		if err != nil {
			log.Warnf("view failed: %v", err)
			return nil, "", err
		}
	}
	defer resp.Body.Close()

	return httpClient, httpPrefix, nil
}
