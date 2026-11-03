/*
Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
*/

// report package provides report functions for oschecktool
package report

import (
	"encoding/csv"
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	"openfuyao.com/oscheck/internal/checker"
	"openfuyao.com/oscheck/internal/utils"
)

var (
	bomForExcel       = []byte{0xEF, 0xBB, 0xBF} // UTF-8 BOM for Excel compatibility
	defaultOutputPath = "./reports"
)

const (
	yamlType          = "yaml"
	csvType           = "csv"
	plainType         = "plain"
	maxHostnameLength = 128
)

// GenerateReport 根据检查结果生成报告
// rslt: 检查结果
// outputPath: 输出路径,如果指定为空，则直接打印到stdout中
// reportType: 报告类型，支持yaml和csv
// 返回生成的报告文件路径和可能的错误
func GenerateReport(rslt checker.Result, outputPath string, reportType string) (string, error) {
	reporter, err := GetReporter(reportType)
	if err != nil {
		return "", fmt.Errorf("failed to get reporter for type %s: %w", reportType, err)
	}
	return reporter.GenerateReport(rslt, outputPath)
}

// GetReporter 根据报告类型获取对应的Reporter实例
func GetReporter(reportType string) (Reporter, error) {
	switch reportType {
	case yamlType:
		return &yamlReporter{}, nil
	case csvType:
		return &csvReporter{}, nil
	case plainType:
		return &plainReporter{}, nil
	default:
		return nil, fmt.Errorf("unsupported report type: %s", reportType)
	}

}

// Reporter 接口定义了生成报告的方法
type Reporter interface {
	// GenerateReport 根据检查结果生成报告
	// rslt: 检查结果
	// outputPath: 输出路径
	// 返回生成的报告文件路径和可能的错误
	GenerateReport(rslt checker.Result, outputPath string) (string, error)
}

func getFileName(hostName, outputPath, suffix string) string {
	if outputPath == "" {
		outputPath = defaultOutputPath
	}
	hostName = escapeHostnameToFilename(hostName)
	currentTime := time.Now().Format("20060102-150405")
	return filepath.Join(outputPath, strings.Join([]string{"CheckReport", hostName, currentTime}, "-")+suffix)
}

// escapeHostnameToFilename 将hostname转义为合法文件名,只保留128位
func escapeHostnameToFilename(hostname string) string {
	// 定义非法字符正则（包含空格、路径分隔符等）
	reg := regexp.MustCompile(`[ \/\\:*?"<>|\s]`)
	safeName := reg.ReplaceAllString(hostname, "_")
	// 处理连续点和开头点
	safeName = strings.ReplaceAll(safeName, "..", "__")
	if strings.HasPrefix(safeName, ".") {
		safeName = "_" + safeName[1:]
	}

	if len(safeName) > maxHostnameLength {
		safeName = safeName[:maxHostnameLength]
	}
	return safeName
}

type yamlReporter struct{}

func (r *yamlReporter) GenerateReport(rslt checker.Result, outputPath string) (string, error) {
	logger := utils.GetLogger("YamlReporter")
	fileName := getFileName(rslt.HostName, outputPath, ".yaml")
	logger.Printf("Start to generate yaml report: %s", fileName)

	// 使用yaml包将rslt转换为yaml格式并写入文件
	file, err := utils.CreateFile(fileName, utils.FileModeOnlyOwnerReadWrite, true)
	if err != nil {
		logger.Printf("Failed to create yaml file: %s, error: %s", fileName, err.Error())
		return "", err
	}

	err = file.Chmod(utils.FileModeOnlyOwnerReadWrite)
	if err != nil {
		logger.Printf("Failed to set file permission: %s, error: %s", fileName, err.Error())
	}
	encoder := yaml.NewEncoder(file)
	defer utils.CloseAll(logger, file, encoder)
	if err := encoder.Encode(rslt); err != nil {
		logger.Printf("Failed to encode check result to yaml: %s, error: %s", fileName, err.Error())
		return "", err
	}
	logger.Printf("Yaml report generated: %s", fileName)
	return fileName, nil
}

type csvReporter struct{}

func (r *csvReporter) GenerateReport(rslt checker.Result, outputPath string) (string, error) {
	logger := utils.GetLogger("CsvReporter")
	fileName := getFileName(rslt.HostName, outputPath, ".csv")
	logger.Printf("Start to generate csv report: %s", fileName)
	// 使用csv包将rslt转换为csv格式并写入文件
	file, err := utils.CreateFile(fileName, utils.FileModeOnlyOwnerReadWrite, true)
	if err != nil {
		logger.Printf("Failed to create csv file: %s, error: %s", fileName, err.Error())
		return "", err
	}
	defer file.Close()
	err = file.Chmod(utils.FileModeOnlyOwnerReadWrite)
	if err != nil {
		logger.Printf("Failed to set file permission: %s, error: %s", fileName, err.Error())
	}
	_, err = file.Write(bomForExcel) // 添加BOM以支持Excel
	if err != nil {
		logger.Printf("Failed to write bom to csv file: %s, error: %s", fileName, err.Error())
		return "", err
	}
	writer := csv.NewWriter(file)
	defer writer.Flush()
	if err := r.writeHeader(writer, rslt); err != nil {
		logger.Printf("Failed to write header to csv file: %s, error: %s", fileName, err.Error())
		return "", err
	}
	// 写入检查结果
	if err := r.writeCheckResult(writer, rslt); err != nil {
		logger.Printf("Failed to write check results to csv file: %s, error: %s", fileName, err.Error())
		return "", err
	}
	logger.Printf("Csv report generated: %s", fileName)
	return fileName, nil
}

// writeHeader 写入CSV文件的头部信息
// rslt: 检查结果
// 返回可能的错误
func (r *csvReporter) writeHeader(writer *csv.Writer, rslt checker.Result) error {
	hostname := rslt.HostName
	content := make([][]string, 0)
	content = append(content, []string{"OS Check Report"})
	content = append(content, []string{"Generated at", time.Now().Format(time.DateTime)})
	content = append(content, []string{"Hostname", hostname})
	content = append(content, []string{"Check Result", getPassString(rslt.Result)})
	return writer.WriteAll(content)
}

func (r *csvReporter) writeCheckResult(writer *csv.Writer, rslt checker.Result) error {
	content := make([][]string, 0)
	content = append(content, []string{"Check Item", "Result", "Check Item Description", "Sub Check Item",
		"Sub Result", "Expect Value", "Actual Value", "Sub Check Item Description"})
	for _, item := range rslt.Items {
		if len(item.SubItems) == 0 {
			content = append(content, []string{item.Key, getPassString(item.Result), item.Doc, "-", "-", "-", "-", "-"})
			continue

		}
		content = append(content, []string{item.Key, getPassString(item.Result), item.Doc, item.SubItems[0].Key,
			getPassString(item.SubItems[0].Result),
			item.SubItems[0].Expect, item.SubItems[0].Real, item.SubItems[0].Doc})
		for _, subItem := range item.SubItems[1:] {
			content = append(content, []string{"", "", "", subItem.Key,
				getPassString(subItem.Result), subItem.Expect, subItem.Real, subItem.Doc})

		}
	}
	if err := writer.WriteAll(content); err != nil {
		return err
	}
	return nil
}

func getPassString(rslt checker.ResultType) string {
	switch rslt {
	case checker.ResultValid:
		return "Pass"
	case checker.ResultInvalid:
		return "Fail"
	case checker.ResultError:
		return "Error"
	default:
		return "Unknown"
	}
}
