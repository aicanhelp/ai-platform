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
	"os"
	"strings"
	"time"

	"github.com/olekukonko/tablewriter"
	"github.com/olekukonko/tablewriter/renderer"
	"github.com/olekukonko/tablewriter/tw"

	"openfuyao.com/oscheck/internal/checker"
	"openfuyao.com/oscheck/internal/utils"
)

const (
	colMaxWidths = 40
)

type plainReporter struct{}

// GenerateReport 生成检查报告
// rslt: 检查结果
// outputPath: 输出路径
// 返回生成的报告文件名和可能的错误
func (r *plainReporter) GenerateReport(rslt checker.Result, outputPath string) (string, error) {
	logger := utils.GetLogger("PlainReporter")
	outputFile := os.Stdout
	fileName := ""
	if strings.TrimSpace(outputPath) != "" {
		logger.Printf("Start to generate plain report: <%s>", outputPath)
		fileName = getFileName(rslt.HostName, outputPath, ".txt")
		// 如果目录不存在，则创建
		var err error
		outputFile, err = utils.CreateFile(fileName, utils.FileModeOnlyOwnerReadWrite, true)
		if err != nil {
			logger.Printf("Failed to create output file: %s, error: %s", fileName, err.Error())
			return "", err
		}
		defer utils.CloseAll(logger, outputFile)
		logger.Printf("Report will be written to: %s", fileName)
	}

	if err := r.writeHeader(outputFile, rslt); err != nil {
		logger.Printf("Failed to write header to csv file, error: %s", err.Error())
		return "", err
	}
	// 写入检查结果
	if err := r.writeCheckResult(outputFile, rslt); err != nil {
		logger.Printf("Failed to write check results to plain file, error: %s", err.Error())
		return "", err
	}
	logger.Printf("Plain report generated: %s", fileName)
	return fileName, nil
}

// writeHeader 写入CSV文件的头部信息
// rslt: 检查结果
// 返回可能的错误
func (r *plainReporter) writeHeader(file *os.File, rslt checker.Result) error {

	// 使用tablewriter写入内容，只有两列，第一列为OS CheckReport， Generated at, Hostname, Check Result, 第二列为具体的值
	table := tablewriter.NewTable(file, tablewriter.WithRenderer(renderer.NewBlueprint(tw.Rendition{
		Borders: tw.BorderNone,
		Symbols: tw.NewSymbols(tw.StyleASCII),
		Settings: tw.Settings{
			Separators: tw.SeparatorsNone,
		},
		Streaming: false,
	})))

	data := [][]string{
		{"Generated at:", time.Now().Format(time.DateTime)},
		{"Hostname:", rslt.HostName},
		{"Check Result:", getPassString(rslt.Result)},
	}
	table.Bulk(data)
	table.Render()
	return nil
}

func (r *plainReporter) writeCheckResult(file *os.File, rslt checker.Result) error {
	table := tablewriter.NewTable(file,
		tablewriter.WithRenderer(renderer.NewBlueprint(tw.Rendition{
			Symbols: tw.NewSymbols(tw.StyleASCII),
			Settings: tw.Settings{
				Separators: tw.Separators{
					ShowHeader:     tw.On,
					ShowFooter:     tw.On,
					BetweenRows:    tw.On,
					BetweenColumns: tw.On,
				},
			},
			Streaming: false,
		})),

		tablewriter.WithConfig(tablewriter.Config{
			Row: tw.CellConfig{
				Formatting:   tw.CellFormatting{AutoWrap: tw.WrapBreak},
				Alignment:    tw.CellAlignment{Global: tw.AlignLeft},
				ColMaxWidths: tw.CellWidth{Global: colMaxWidths}},
		}))
	table.Header([]string{"Check Item",
		"Result",
		"Check Item Description",
		"Sub Check Item",
		"Sub Result",
		"Expect Value",
		"Actual Value",
		"Sub Check Item Description"})
	data := make([][]string, 0, len(rslt.Items))
	for _, item := range rslt.Items {
		if len(item.SubItems) == 0 {
			data = append(data, []string{item.Key, getPassString(item.Result), item.Doc, "-", "-", "-", "-", "-"})
			continue

		}
		data = append(data, []string{
			item.Key, getPassString(item.Result),
			item.Doc, item.SubItems[0].Key,
			getPassString(item.SubItems[0].Result),
			item.SubItems[0].Expect, item.SubItems[0].Real,
			item.SubItems[0].Doc})
		for _, subItem := range item.SubItems[1:] {
			data = append(data, []string{"", "", "",
				subItem.Key, getPassString(subItem.Result), subItem.Expect, subItem.Real, subItem.Doc})
		}
	}
	table.Bulk(data)

	if err := table.Render(); err != nil {
		return err
	}
	return nil
}
