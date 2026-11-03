// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package dependencies

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"gopkg.in/yaml.v2"
	"helm.sh/helm/v3/pkg/action"
	"helm.sh/helm/v3/pkg/cli"
)

const destDir = "./downloaded-charts"                              // helm pull 结果存储路径
const dependencyComponentFileName = "dependencyComponentFile.json" // 组件依赖存储文件名
const componentDependencyFileName = "componentDependencyFile.json" // 组件被依赖存储文件名

// 不进行依赖识别的组件（所有版本）
var abandonComponent = []string{"volcano-config-package", "volcano-config-service", "volcano-config-website"}

// HelmIndex 定义完整的 index.yaml 中的 YAML 结构体
type HelmIndex struct {
	APIVersion string                    `yaml:"apiVersion"`
	Entries    map[string][]ChartVersion `yaml:"entries"`
}

// ChartVersion 定义完整的 index.yaml 中的 YAML 结构体
type ChartVersion struct {
	APIVersion   string       `yaml:"apiVersion"`
	AppVersion   string       `yaml:"appVersion"`
	Created      string       `yaml:"created"`
	Dependencies []Dependency `yaml:"dependencies"`
	Description  string       `yaml:"description"`
	Digest       string       `yaml:"digest"`
	Keywords     []string     `yaml:"keywords"`
	Name         string       `yaml:"name"`
	Type         string       `yaml:"type"`
	URLs         []string     `yaml:"urls"`
	Version      string       `yaml:"version"`
}

// Dependency 定义完整的 index.yaml 中的 YAML 结构体
type Dependency struct {
	Name       string `yaml:"name"`
	Repository string `yaml:"repository"`
	Version    string `yaml:"version"`
}

// ChartInfo 存储解析结果的结构
type ChartInfo struct {
	Component  string
	Version    string
	URL        string
	AppVersion string
}

// ChartMetadata 用于解析 Chart.yaml 的结构
type ChartMetadata struct {
	Name    string `yaml:"name"`
	Version string `yaml:"version"`
}

// DependencyMap 封装map<Name, map<Version, []ComponentName>>
type DependencyMap struct {
	Component map[string]map[string][]string `json:"Component"`
}

// NewDependencyMap 构造函数
func NewDependencyMap(charts []ChartInfo) *DependencyMap {
	dependencyMap := make(map[string]map[string][]string)
	for _, chart := range charts {
		if dependencyMap[chart.Component] == nil {
			dependencyMap[chart.Component] = make(map[string][]string)
		}
		dependencyMap[chart.Component][chart.Version] = []string{}
	}
	return &DependencyMap{
		Component: dependencyMap,
	}
}

// Add 安全地添加值 （不去重）
func (dm *DependencyMap) Add(name string, version string, value string) {
	if dm.Component[name] == nil {
		dm.Component[name] = make(map[string][]string)
	}
	if dm.Component[name][version] == nil {
		dm.Component[name][version] = []string{}
	}
	dependencies := dm.Component[name][version]
	dependencies = append(dependencies, value)
	dm.Component[name][version] = dependencies
	log.Printf("向 %s:%s 添加 %s 成功\n", name, version, value)
}

// IdentifyDependenciesChartsReturn 封装identifyDependenciesCharts函数的返回值
type IdentifyDependenciesChartsReturn struct {
	DependencyComponentMap *DependencyMap
	ComponentDependencyMap *DependencyMap
	FailedCharts           []string
	Err                    error
}

// identifyDependenciesCharts 遍历所有chart文件，更新被依赖记录
func identifyDependenciesCharts(charts []ChartInfo, dependencyComponentMap *DependencyMap,
	componentDependencyMap *DependencyMap) *IdentifyDependenciesChartsReturn {
	var failedCharts []string
	var err error
	for _, chart := range charts {
		// 去除已过时的组件，这些组件可能有循环依赖
		if slices.Contains(abandonComponent, chart.Component) {
			continue
		}

		// 开始识别组件被依赖情况
		log.Printf("%s：%s 开始解析", chart.Component, chart.Version)
		dependencyComponentMap, componentDependencyMap, err =
			processChart(chart, dependencyComponentMap, componentDependencyMap, destDir)
		if err != nil {
			failedCharts = append(failedCharts, chart.Component+"-"+chart.Version)
			log.Printf("chart文件解析失败：%v", err)
		}
		log.Printf("%s：%s 解析成功", chart.Component, chart.Version)

		// 清理下载的helm charts
		err = deleteDirectory(destDir)
		if err != nil {
			return &IdentifyDependenciesChartsReturn{
				DependencyComponentMap: dependencyComponentMap,
				ComponentDependencyMap: componentDependencyMap,
				FailedCharts:           failedCharts,
				Err:                    fmt.Errorf("清理下载的helm charts失败：%v", err),
			}
		}
	}
	return &IdentifyDependenciesChartsReturn{
		DependencyComponentMap: dependencyComponentMap,
		ComponentDependencyMap: componentDependencyMap,
		FailedCharts:           failedCharts,
		Err:                    nil,
	}
}

// executeCloseOutWork 执行收尾工作
func executeCloseOutWork(dependencyComponentMap *DependencyMap, componentDependencyMap *DependencyMap,
	failedCharts []string, storageDir string) (string, string, error) {
	// 序列化为JSON
	dependencyComponentFilePath := filepath.Join(storageDir, dependencyComponentFileName)
	err := convertToJson(dependencyComponentMap, dependencyComponentFilePath)
	if err != nil {
		return "", "", fmt.Errorf("依赖文件序列化为JSON文件失败：%v", err)
	}

	componentDependencyFilePath := filepath.Join(storageDir, componentDependencyFileName)
	err = convertToJson(componentDependencyMap, componentDependencyFilePath)
	if err != nil {
		return "", "", fmt.Errorf("被依赖文件序列化为JSON文件失败：%v", err)
	}

	// 最终统计输出
	if len(failedCharts) > 0 {
		log.Printf("识别依赖失败的组件：")
		for _, failedChart := range failedCharts {
			log.Println(failedChart)
		}
	} else {
		log.Printf("所有组件均识别成功！")
	}
	return dependencyComponentFilePath, componentDependencyFilePath, nil
}

// IdentifyDependencies 核心识别被依赖情况逻辑
func IdentifyDependencies(helmURL string, storageDir string) (string, string, error) {
	// 从URL读取并解析Helm索引
	charts, err := fetchAndParseHelmIndex(helmURL)
	if err != nil {
		return "", "", fmt.Errorf("解析index.yaml出现错误: %v", err)
	}
	// 在日志中输出结果
	printChartTable(charts)

	dependencyComponentMap := NewDependencyMap(charts) // 初始化依赖记录
	componentDependencyMap := NewDependencyMap(charts) // 初始化被依赖记录

	// 确保目录存在
	if err := ensureDir(destDir); err != nil {
		return "", "", fmt.Errorf("拉取chart存储目录创建失败: %v", err)
	}

	// 遍历所有chart文件，更新被依赖记录
	var failedCharts []string
	identifyReturn :=
		identifyDependenciesCharts(charts, dependencyComponentMap, componentDependencyMap)
	if identifyReturn.Err != nil {
		return "", "", fmt.Errorf("遍历所有chart文件失败: %v", err)
	}
	failedCharts = identifyReturn.FailedCharts

	// 更新两个map，完成依赖同步
	dependencyComponentMap, componentDependencyMap, err =
		expandMap(identifyReturn.DependencyComponentMap, identifyReturn.ComponentDependencyMap, charts)
	if err != nil {
		return "", "", fmt.Errorf("各组件依赖与被依赖文件同步失败：%v", err)
	}

	// 确保目录存在
	if err := ensureDir(storageDir); err != nil {
		return "", "", fmt.Errorf("最终识别结果存储目录创建失败: %v", err)
	}

	// 执行收尾工作
	var dependencyComponentFilePath, componentDependencyFilePath string
	dependencyComponentFilePath, componentDependencyFilePath, err =
		executeCloseOutWork(dependencyComponentMap, componentDependencyMap, failedCharts, storageDir)
	if err != nil {
		return "", "", fmt.Errorf("收尾工作执行失败: %v", err)
	}
	return dependencyComponentFilePath, componentDependencyFilePath, nil
}

// fetchAndParseHelmIndex 从URL获取并解析Helm索引文件
func fetchAndParseHelmIndex(url string) ([]ChartInfo, error) {
	// 创建HTTP客户端，设置超时
	var timeoutSeconds = 30
	client := &http.Client{
		Timeout: time.Duration(timeoutSeconds) * time.Second,
	}

	// 发送HTTP请求
	resp, err := client.Get(url)
	if err != nil {
		return nil, fmt.Errorf("HTTP请求失败: %w", err)
	}
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			log.Printf("警告: 关闭响应体时出错: %v", closeErr)
		}
	}()

	// 检查响应状态
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("服务器返回错误状态码: %d", resp.StatusCode)
	}

	// 读取响应体
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("读取响应体失败: %w", err)
	}

	// 解析YAML
	return parseHelmIndexYAML(data)
}

// parseHelmIndexYAML 解析Helm索引YAML内容
func parseHelmIndexYAML(data []byte) ([]ChartInfo, error) {
	var index HelmIndex

	// 解析响应数据
	err := yaml.Unmarshal(data, &index)
	if err != nil {
		return nil, fmt.Errorf("YAML解析失败: %w", err)
	}

	// 提取组件信息
	var charts []ChartInfo
	for componentName, chartVersions := range index.Entries {
		for _, chart := range chartVersions {
			if len(chart.URLs) != 1 {
				return nil, fmt.Errorf("%s：%s helm包拉取地址不唯一", componentName, chart.Version)
			}
			chartInfo := ChartInfo{
				Component:  componentName,
				Version:    chart.Version,
				URL:        chart.URLs[0], // 取第一个URL
				AppVersion: chart.AppVersion,
			}
			charts = append(charts, chartInfo)
		}
	}

	return charts, nil
}

// printChartTable 以表格形式打印组件信息
func printChartTable(charts []ChartInfo) {
	if len(charts) == 0 {
		log.Println("未找到任何组件信息")
		return
	}

	log.Printf("从Helm仓库解析结果 (共%d个组件版本):\n", len(charts))
	log.Println("=================================================================================================")
	log.Printf("%-30s %-15s %-15s %s\n", "组件名", "Chart版本", "App版本", "URL")
	log.Printf("%-30s %-15s %-15s %s\n", "------", "----------", "--------", "---")

	for _, chart := range charts {
		log.Printf("%-30s %-15s %-15s %s\n", chart.Component, chart.Version, chart.AppVersion, chart.URL)
	}

	// 统计信息
	componentCount := make(map[string]int)
	for _, chart := range charts {
		componentCount[chart.Component]++
	}

	log.Println("\n组件统计:")
	for component, count := range componentCount {
		log.Printf("  %s: %d个版本\n", component, count)
	}
}

// processChart 处理每一个index.yaml中的chart
func processChart(chart ChartInfo, dependencyComponentMap *DependencyMap, componentDependencyMap *DependencyMap,
	dirPath string) (*DependencyMap, *DependencyMap, error) {
	// helm pull（storage若不存在自动创建）
	err := helmPullUrl(chart.URL)
	if err != nil {
		return dependencyComponentMap, componentDependencyMap, fmt.Errorf("helm pull 执行失败：%v", err)
	}

	// 遍历文件夹
	rootDir := filepath.Join(dirPath, chart.Component)
	dependencyComponentMap, componentDependencyMap, err =
		extractComponentCharts(dependencyComponentMap, componentDependencyMap, rootDir, chart.Component, chart.Version)
	if err != nil {
		return dependencyComponentMap, componentDependencyMap, fmt.Errorf("遍历文件夹获取依赖情况执行失败：%v", err)
	}

	return dependencyComponentMap, componentDependencyMap, nil
}

// 确保存取目录存在
func ensureDir(dirPath string) error {
	// 使用 MkdirAll 创建目录（包括所有父目录），如果目录已存在则不会报错
	var permission = 0755
	if err := os.MkdirAll(dirPath, os.FileMode(permission)); err != nil {
		return fmt.Errorf("创建目录失败 %s: %w", dirPath, err)
	}
	return nil
}

// helmPullUrl 实现helm pull url
func helmPullUrl(url string) error {
	// 1. 初始化 Helm 环境设置
	settings := cli.New()

	// 2. 初始化 action 配置
	actionConfig := new(action.Configuration)
	// 对于 pull 操作，命名空间参数通常可以为空字符串
	if err := actionConfig.Init(settings.RESTClientGetter(), "",
		os.Getenv("HELM_DRIVER"), log.Printf); err != nil {
		return fmt.Errorf("初始化 Helm 配置失败: %v", err)
	}

	// 3. 创建 Pull action 客户端
	client := action.NewPullWithOpts(action.WithConfig(actionConfig)) // 使用 WithConfig 显式关联配置
	client.Settings = settings

	// 配置拉取选项
	client.DestDir = destDir // 指定 chart 下载目录
	client.Untar = true      // 是否解压 tgz 文件
	client.UntarDir = ""     // 解压缩存储路径

	// 4. 执行拉取操作 (OCI 地址)
	chartRef := url
	log.Printf("开始拉取，url：%s\n", chartRef)
	// 定义最大重试次数和初始等待时间
	maxRetries := 3
	baseDelay := time.Second * 1 // 初始等待1秒

	var lastErr error
	for i := 0; i < maxRetries; i++ {
		_, err := client.Run(chartRef)
		if err == nil {
			log.Printf("成功拉取 chart: %s 到目录 %s\n", chartRef, client.DestDir)
			return nil
		}
		// 记录错误
		lastErr = err
		log.Printf("拉取失败 (第 %d 次尝试): %v\n", i+1, err)

		// 如果不是最后一次尝试，则等待后重试
		if i < maxRetries-1 {
			// 计算等待时间（指数退避：1s, 2s, 4s...）
			waitTime := baseDelay * (1 << i) // 2^i 的位移实现
			log.Printf("%v 秒后重试...\n", waitTime)
			time.Sleep(waitTime)
		}
	}
	return fmt.Errorf("拉取 chart 失败（已重试 %d 次）: %v", maxRetries, lastErr)
}

// extractChartInfoFromYAML 从 Chart.yaml 文件中提取信息
func extractChartInfoFromYAML(filePath string) (string, string, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return "", "", fmt.Errorf("从文件路径 %s 读取 chart.yaml 失败: %v", filePath, err)
	}

	var metadata ChartMetadata
	err = yaml.Unmarshal(data, &metadata)
	if err != nil {
		return "", "", fmt.Errorf("从文件路径 %s 提取 chart.yaml 中的关键字段失败: %v", filePath, err)
	}

	if metadata.Name == "" || metadata.Version == "" {
		return "", "", fmt.Errorf("从文件路径 %s 提取 chart.yaml 中的关键字段缺失", filePath)
	}

	return metadata.Name, metadata.Version, nil
}

// containsString 判断当前slice集合是否有target字符串
func containsString(slice []string, target string) bool {
	for _, item := range slice {
		if item == target {
			return true
		}
	}
	return false
}

// ExtractAllChartsParams 封装extractAllChartsParams函数参数
type ExtractAllChartsParams struct {
	DependencyComponentMap *DependencyMap
	ComponentDependencyMap *DependencyMap
	Entries                []os.DirEntry
	ChartsDir              string
	Name                   string
	Version                string
}

// extractAllCharts 递归处理所有子charts
func extractAllCharts(params *ExtractAllChartsParams) (*DependencyMap, *DependencyMap, error) {
	var err error
	for _, entry := range params.Entries {
		if entry.IsDir() {
			subDir := filepath.Join(params.ChartsDir, entry.Name())
			// 递归处理子目录
			params.DependencyComponentMap, params.ComponentDependencyMap, err = extractComponentCharts(
				params.DependencyComponentMap, params.ComponentDependencyMap, subDir, params.Name, params.Version)
			if err != nil {
				return params.DependencyComponentMap, params.ComponentDependencyMap,
					fmt.Errorf("处理子目录 %s 失败: %v", subDir, err)
			}
		}
	}
	return params.DependencyComponentMap, params.ComponentDependencyMap, nil
}

// extractComponentCharts 解析每一个chart的文件夹
func extractComponentCharts(dependencyComponentMap *DependencyMap, componentDependencyMap *DependencyMap,
	rootDir string, rootName string, rootVersion string) (*DependencyMap, *DependencyMap, error) {
	var name, version string

	// 首先处理当前目录的 Chart.yaml
	chartYAMLPath := filepath.Join(rootDir, "Chart.yaml")
	if _, err := os.Stat(chartYAMLPath); err == nil {
		name, version, err = extractChartInfoFromYAML(chartYAMLPath)
		if err != nil {
			return dependencyComponentMap, componentDependencyMap,
				fmt.Errorf("解析 %s 失败: %v", chartYAMLPath, err)
		}
		// 顶层组件无需添加
		var topComponent = rootName + "-" + rootVersion
		var currentComponent = name + "-" + version
		if currentComponent != topComponent {
			if !containsString(dependencyComponentMap.Component[rootName][rootVersion], currentComponent) {
				dependencyComponentMap.Add(rootName, rootVersion, name+"-"+version)
				componentDependencyMap.Add(name, version, rootName+"-"+rootVersion)
				log.Printf("%s：%s 添加 %s：%s 成功！\n", rootName, rootVersion, name, version)
			} else {
				log.Printf("%s：%s 已存在依赖 %s：%s\n", rootName, rootVersion, name, version)
			}
		}
	} else {
		return dependencyComponentMap, componentDependencyMap, fmt.Errorf("查找 %s 失败: %v", chartYAMLPath, err)
	}

	// 处理当前目录的 charts 文件夹
	chartsDir := filepath.Join(rootDir, "charts")
	if entries, err := os.ReadDir(chartsDir); err == nil {
		dependencyComponentMap, componentDependencyMap, err =
			extractAllCharts(&ExtractAllChartsParams{
				DependencyComponentMap: dependencyComponentMap,
				ComponentDependencyMap: componentDependencyMap,
				Entries:                entries,
				ChartsDir:              chartsDir,
				Name:                   name,
				Version:                version,
			})
		if err != nil {
			return dependencyComponentMap, componentDependencyMap, fmt.Errorf("递归处理子目录失败：%v", err)
		}
	}

	return dependencyComponentMap, componentDependencyMap, nil
}

// isDigit 判断是否为数字
func isDigit(num byte) bool {
	return num >= '0' && num <= '9'
}

// splitComponentAndVersionByParts 识别文件名和版本号
func splitComponentAndVersionByParts(input string) (string, string, error) {
	parts := strings.Split(input, "-")

	var expentedPartsLen = 2
	if len(parts) < expentedPartsLen {
		return "", "", fmt.Errorf("字符串格式错误，至少需要2部分: %s", input)
	}

	// 检查从哪一部分开始是版本号
	versionStartIndex := -1
	for i := 0; i < len(parts); i++ {
		// 检查是否以数字开头（简单版本号检测）
		// 1. version: x.x.x(-latest / -release-1.0.x)
		versionType1 := len(parts[i]) > 0 && isDigit(parts[i][0])
		// 2. version: vx.x.x
		versionType2 := len(parts[i]) > 1 && (parts[i][0] == 'v' || parts[i][0] == 'V') && isDigit(parts[i][1])
		if versionType1 || versionType2 {
			versionStartIndex = i
			break
		}
	}

	if versionStartIndex == -1 {
		return "", "", fmt.Errorf("无法识别版本号: %s", input)
	}

	// 构建组件名（版本号之前的部分）
	component := strings.Join(parts[:versionStartIndex], "-")
	// 构建版本号
	version := strings.Join(parts[versionStartIndex:], "-")

	return component, version, nil
}

// RecordFinalResultsParams 封装recordFinalResults函数参数
type RecordFinalResultsParams struct {
	DependencyComponentMap *DependencyMap
	ComponentDependencyMap *DependencyMap
	Name                   string
	Version                string
	DependencyName         string
	DependencyVersion      string
}

// recordFinalResults 记录最终结果
func recordFinalResults(params *RecordFinalResultsParams) (*DependencyMap, *DependencyMap, error) {
	for _, value := range params.DependencyComponentMap.Component[params.DependencyName][params.DependencyVersion] {
		if !containsString(params.DependencyComponentMap.Component[params.Name][params.Version], value) {
			// 跨层存储，即建立该组件和它依赖的依赖之间的连接
			params.DependencyComponentMap.Add(params.Name, params.Version, value)
			valueName, valueVersion, err := splitComponentAndVersionByParts(value)
			if err != nil {
				return params.DependencyComponentMap, params.ComponentDependencyMap,
					fmt.Errorf("在 %s 中识别组件名和版本号失败：%v", value, err)
			}
			params.ComponentDependencyMap.Add(valueName, valueVersion, params.Name+"-"+params.Version)
		}
	}
	return params.DependencyComponentMap, params.ComponentDependencyMap, nil
}

// findAllDependencies 递归同步所有依赖
func findAllDependencies(dependencyComponentMap *DependencyMap, componentDependencyMap *DependencyMap,
	name string, version string) (*DependencyMap, *DependencyMap, error) {
	if dependencyComponentMap.Component[name] == nil ||
		dependencyComponentMap.Component[name][version] == nil ||
		len(dependencyComponentMap.Component[name][version]) == 0 {
		return dependencyComponentMap, componentDependencyMap, nil
	}
	dependencies := dependencyComponentMap.Component[name][version]
	// 将依赖的依赖加入集合
	for _, dependency := range dependencies {
		dependencyName, dependencyVersion, err := splitComponentAndVersionByParts(dependency)
		if err != nil {
			return dependencyComponentMap, componentDependencyMap,
				fmt.Errorf("在 %s 中识别组件名和版本号失败：%v", dependency, err)
		}
		dependencyComponentMap, componentDependencyMap, err =
			findAllDependencies(dependencyComponentMap, componentDependencyMap, dependencyName, dependencyVersion)
		if err != nil {
			return dependencyComponentMap, componentDependencyMap, fmt.Errorf("递归同步所有依赖失败：%v", err)
		}
		dependencyComponentMap, componentDependencyMap, err = recordFinalResults(&RecordFinalResultsParams{
			DependencyComponentMap: dependencyComponentMap,
			ComponentDependencyMap: componentDependencyMap,
			Name:                   name,
			Version:                version,
			DependencyName:         dependencyName,
			DependencyVersion:      dependencyVersion,
		})
		if err != nil {
			return dependencyComponentMap, componentDependencyMap,
				fmt.Errorf("在 %s 中记录最终结果失败：%v", dependency, err)
		}
	}
	return dependencyComponentMap, componentDependencyMap, nil
}

// expandMap 同步map中各层依赖情况
func expandMap(dependencyComponentMap *DependencyMap, componentDependencyMap *DependencyMap,
	charts []ChartInfo) (*DependencyMap, *DependencyMap, error) {
	var err error

	for _, chart := range charts {
		dependencyComponentMap, componentDependencyMap, err =
			findAllDependencies(dependencyComponentMap, componentDependencyMap, chart.Component, chart.Version)
		if err != nil {
			return dependencyComponentMap, componentDependencyMap,
				fmt.Errorf("同步map中各层依赖情况失败：%v", err)
		}
	}

	log.Printf("各组件依赖文件同步成功")
	return dependencyComponentMap, componentDependencyMap, nil
}

// convertToJson 将JSON输出到指定文件路径
func convertToJson(data interface{}, filePath string) error {
	// 创建并打开文件，目录已确保存在
	var permission = 0644
	file, err := os.OpenFile(filePath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, os.FileMode(permission))
	if err != nil {
		return fmt.Errorf("创建文件失败: %w", err)
	}
	defer func(file *os.File) {
		closeErr := file.Close()
		if closeErr != nil && err == nil {
			// 只有主逻辑没有错误时，才用关闭错误覆盖
			err = fmt.Errorf("关闭文件失败: %w", closeErr)
		}
	}(file)

	// 将数据转换为格式化的JSON
	jsonData, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Errorf("JSON编码失败: %w", err)
	}

	// 写入文件
	_, err = file.Write(jsonData)
	if err != nil {
		return fmt.Errorf("写入文件失败: %w", err)
	}

	// 同时在控制台显示保存信息
	log.Printf("JSON数据已成功保存到: %s\n", filePath)
	return nil
}

// deleteDirectory 清理下载的helm charts
func deleteDirectory(dirPath string) error {
	// 检查目录是否存在
	if _, err := os.Stat(dirPath); os.IsNotExist(err) {
		log.Printf("目录 %s 不存在，无需删除\n", dirPath)
		return nil
	}

	// 删除目录及其所有内容
	err := os.RemoveAll(dirPath)
	if err != nil {
		return fmt.Errorf("删除目录 %s 失败: %w", dirPath, err)
	}

	log.Printf("目录 %s 已成功删除\n", dirPath)
	return nil
}
