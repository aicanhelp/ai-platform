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

// The main package of the oscheck tool, which checks OS configuration against specified check sets
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"openfuyao.com/oscheck/internal/checker"
	"openfuyao.com/oscheck/internal/conf"
	"openfuyao.com/oscheck/internal/report"
	"openfuyao.com/oscheck/internal/utils"
)

// `./oschecktool [-s chec-set] [-o yaml|csv|plain] [-h]
// -h 打印帮助信息
// -o 指定输出文件格式, 支持yaml,csv, plain三种形式，其中:
//    - yaml: 输出yaml格式的报告
//    - csv: 输出csv格式的报告
//    - plain: 输出纯文本格式的报告
// -s 指定检查集，实际就是指定检查内容
//    - 如果不指定，则默认使用当前目录下的check-sets目录下的所有检查项
//    - 如果指定，则只使用指定的检查集
//    - 如果指定的检查集不存在，则报错, 同时提示可以选择的检查集
// -p 指定参数，格式为key=value1,value2,...
//    - 各参数实际按照检查项定义而指定，多个参数支持，格式为key=value
//    - 如果参数有多个值，则使用","分割
//    - 如果参数值中本身有",",则需要使用\转义
//    - 如果参数值中本身有\，则需要使用\\转义
//    - 如果同名参数多次绑定，则会被合并到一个列表中
//    - 如果指定的参数不存在，则也不会报错，只是会忽略该参数
//  -r 指定report路径，将在该路径下生成报告

// Params 用于存储命令行参数, 复写以支持参数中包含多个值情况
type Params map[string][]string

const (
	numParmsKv = 2
)

// Set 实现flag.Value接口，用于解析命令行参数
func (p *Params) Set(param string) error {
	parts := strings.SplitN(param, "=", numParmsKv)
	if len(parts) != numParmsKv {
		return fmt.Errorf("invalid parameter format: %s, expected key1=value1[,value2,...]", param)
	}
	key := parts[0]
	values := parts[1]

	if _, ok := (*p)[key]; !ok {
		(*p)[key] = []string{}
	}

	// 按照字符便利value,如果遇到了\,且下一个字符是\,则将其替换为一个\
	// 如果下一个字符是, 则截断之，添加到数组中
	start := 0
	for i := 0; i < len(values); i++ {
		if values[i] == '\\' {
			if i+1 < len(values) && values[i+1] == '\\' {
				i++
			} else if i+1 < len(values) && values[i+1] == ',' {
				i++
			}
		} else if values[i] == ',' {
			(*p)[key] = append((*p)[key], values[start:i])
			start = i + 1
		} else if i == len(values)-1 {
			(*p)[key] = append((*p)[key], values[start:])
		}

	}

	return nil
}

// String 实现flag.Value接口，用于打印参数
func (p *Params) String() string {
	return fmt.Sprintf("%v", *p)
}

func main() {
	utils.InitLogFile(filepath.Join(utils.GetStartDir(), "log", "oscheck.log"))
	config, err := conf.LoadConf()
	if err != nil {
		utils.GetDefaultLogger().Printf("Failed to initialize configuration: %s", err.Error())
		utils.PromptMsg("Error: Failed to initialize configuration")
		os.Exit(1)
	}

	if os.Getuid() != 0 {
		utils.PromptMsg("Error: This tool requires root privileges to run. Please run it with sudo.")
		os.Exit(1)
	}
	params, err := getParams(config)
	if err != nil {
		os.Exit(1)
	}
	if params.help {
		return
	}
	// 开始执行检查
	utils.PromptMsg("Start to execute checks...")
	checkResult, ok := executeChecks(params)
	if !ok {
		utils.PromptMsg("Error: Failed to execute checks.")
		os.Exit(1)
	}
	reportPath, err := params.reporter.GenerateReport(checkResult, params.reportPath)
	if err != nil {
		utils.PromptMsg("Error: Failed to generate report: %s", err.Error())
		os.Exit(1)
	}
	if reportPath != "" {
		utils.PromptMsg("Report generated successfully: %s", reportPath)
	}
	if checkResult.Result != checker.ResultValid {
		os.Exit(1)
	}
}

func getParams(config *conf.CheckConfig) (runParams, error) {

	// 解析命令行参数
	var checkSet, outputFormat, reportPath, rootPath string
	var params Params = make(map[string][]string)

	help := false
	flag.CommandLine = flag.NewFlagSet(os.Args[0], flag.ContinueOnError)
	flag.StringVar(&checkSet, "s", "All",
		"Specify the check set to use, if not specified, all check items in the conf/CheckItem directory will be used.")
	flag.StringVar(&outputFormat, "t", "plain", "Specify the output format: yaml, csv, plain,"+
		" if not specified, plain will be used.")
	flag.Var(&params, "p", `Specify the parameters, format: key1=value1[,value2,...],`+
		`This parameter can be bound multiple times.`)
	flag.StringVar(&reportPath, "o", "", "Specify the report output path."+
		" The report file will be generated under this path: CheckReport-{NodeName}-{Time}.format.")

	flag.BoolVar(&help, "h", false, "Print help information")
	flag.Usage = func() {
		utils.PromptMsg("This tool is used to check " +
			"whether the OS configuration is consistent with a specific check set.")
		utils.PromptMsg("Usage: %s [-s check-set] [-o yaml|csv|plain] [-r report-path]"+
			" [-p key1=value1[,value2,...]] [-h]", flag.CommandLine.Name())
		flag.VisitAll(func(f *flag.Flag) {
			printFlag(config, f)
		})
		utils.PromptMsg("\nExample:")
		utils.PromptMsg("  %s -t plain -o ./reports", flag.CommandLine.Name())
	}
	flag.Parse()
	if help {
		flag.Usage()
		return runParams{help: true}, nil
	}

	paramsChecked, ok := checkAndInitParam(config, checkSet, outputFormat, reportPath, rootPath)
	paramsChecked.itemParams = params
	if !ok {
		utils.PromptMsg("Error: Invalid command arguments. Use '-h' for usage instructions.")
		return runParams{}, fmt.Errorf("invalid command arguments")
	}
	return paramsChecked, nil
}

func printFlag(config *conf.CheckConfig, f *flag.Flag) {
	utils.PromptMsg("\t-%s: %s", f.Name, f.Usage)
	switch f.Name {
	case "p":
		hasPromptParams := false
		for _, item := range config.CheckItems {
			if len(item.Params) > 0 {
				if !hasPromptParams {
					hasPromptParams = true
					utils.PromptMsg("\t\t Optional parameters:")
				}
				for _, param := range item.Params {
					utils.PromptMsg("\t\t\t %s: %s", param.Name, param.Desc)
				}
			}
		}
		if !hasPromptParams {
			utils.PromptMsg("\t\t No optional parameters.")
		}
	case "s":
		if len(config.CheckSets) > 0 {
			utils.PromptMsg("\t\t Optional check sets:")
		} else {
			utils.PromptMsg("\t\t No check set available.")
		}
		for _, checkset := range config.CheckSets {
			utils.PromptMsg("\t\t\t %s: %s", checkset.Name, checkset.Desc)
		}
	default:
		return
	}
}

func executeChecks(p runParams) (checker.Result, bool) {
	// 控制并发检查数量
	chanConc := make(chan struct{}, 10)
	defer close(chanConc)
	chanRslt := make(chan checker.ItemRslt, len(p.checkItems))
	var wg sync.WaitGroup
	hasErr := false
	result := checker.Result{
		CheckSet:  p.checkSet,
		Result:    checker.ResultValid,
		StartTime: time.Now(),
		Items:     make([]checker.ItemRslt, 0, len(p.checkItems)),
	}
	checkers, hasErr := getCheckers(p)

	if hasErr {
		return result, false
	}

	for _, checker := range checkers {
		wg.Add(1)
		copiedChecker := checker
		go func() {
			defer wg.Done()
			chanConc <- struct{}{}
			rslt := copiedChecker.Check()
			chanRslt <- rslt
			<-chanConc
		}()
	}
	wg.Wait()
	close(chanRslt)
	for item := range chanRslt {
		if int(result.Result) < int(item.Result) {
			result.Result = item.Result
		}
		result.Items = append(result.Items, item)
	}
	result.HostName = getHostName(p.isInPods, p.rootPath)
	result.EndTime = time.Now()
	return result, true
}

func getCheckers(p runParams) ([]checker.Checker, bool) {
	hasErr := false
	checkers := make([]checker.Checker, 0, len(p.checkItems))
	ctx := checker.Context{
		RootPath: p.rootPath,
	}
	for _, checkItem := range p.checkItems {
		if len(checkItem.Params) > 0 {
			checkItem.FillParamValues(p.itemParams)
		}
		checker, err := checker.GetChecker(checkItem.Kind)
		if err != nil {
			utils.PromptMsg(`Unsupported Check kind "%s"  in file: %s`, checkItem.Kind, checkItem.FilePath)
			hasErr = true
			continue
		}
		checkInitErr := checker.Init(ctx, checkItem)
		if checkInitErr != nil {
			if len(checkInitErr.ErrField) > 0 {
				utils.PromptMsg("Check item %s has configuration error, error fields: %s",
					checkItem.FilePath, strings.Join(checkInitErr.ErrField, ","))
			} else {
				utils.PromptMsg("Check item %s has configuration error.", checkItem.FilePath)
			}
			hasErr = true
			continue
		}
		checkers = append(checkers, checker)
	}
	return checkers, hasErr
}

func getHostName(isInPods bool, rootPath string) string {
	// 尝试从NODE_NAME环境变量中读取hostname
	hostName := ""
	// 容器环境获取hostname
	if isInPods {
		hostName = os.Getenv("NODE_NAME")
		if hostName != "" {
			return hostName
		}
		bytes, err := os.ReadFile(filepath.Join(rootPath, "proc/sys/kernel/hostname"))
		if err == nil {
			hostName = string(bytes)
		} else {
			utils.GetDefaultLogger().Printf("Failed get host name in pods, because: %s", err.Error())
		}
	}

	if hostName == "" {
		var err error
		hostName, err = os.Hostname()
		if err != nil {
			utils.GetDefaultLogger().Printf("Failed get host name in pods, because: %s", err.Error())
		}
	}

	if hostName == "" {
		hostName = "unknown"
	}
	return hostName
}

type runParams struct {
	rootPath   string
	checkSet   string
	reporter   report.Reporter
	reportPath string
	checkItems map[string]conf.CheckItem
	isInPods   bool
	help       bool
	itemParams map[string][]string
}

func checkAndInitParam(config *conf.CheckConfig, checkset, outputFormat,
	reportPath, rootPath string) (runParams, bool) {
	p := runParams{
		checkSet:   "All",
		reportPath: "",
		checkItems: make(map[string]conf.CheckItem),
	}
	ok := true
	ok = ok && checkAndInitRootPath(rootPath, &p)
	ok = ok && checkAndInitOutput(outputFormat, &p)
	ok = ok && checkAndInitReportPath(reportPath, &p)
	ok = ok && checkAndInitCheckSet(config, checkset, &p)
	return p, ok
}

func checkAndInitCheckSet(config *conf.CheckConfig, checkSetName string, p *runParams) bool {
	if checkSetName == "All" {
		p.checkItems = config.CheckItems
	} else {
		checkSet, exist := config.CheckSets[checkSetName]
		if !exist {
			checksetNames := make([]string, 0, len(config.CheckSets))
			for name := range config.CheckSets {
				checksetNames = append(checksetNames, name)
			}
			utils.PromptMsg("CheckSet %s does not exist, please choose from: %v", checkSetName, checksetNames)
			return false
		}
		p.checkItems = checkSet.Items
		p.checkSet = checkSet.Name
	}
	return true
}

func checkAndInitRootPath(rootPath string, p *runParams) bool {
	rootPath = strings.TrimSpace(rootPath)
	if rootPath != "" {
		if fileInfo, err := os.Stat(rootPath); err != nil || fileInfo == nil {
			utils.PromptMsg("Error: Root path %s cannot access, err: %s", rootPath, err.Error())
			return false
		} else if !fileInfo.IsDir() {
			utils.PromptMsg("Error: Root path %s is not a directory", rootPath)
			return false
		} else {
			// 检查rootPath下是否有/proc/sys/kernel/hostname文件，如果没有这个文件，意味着这个root指定的有问题
			hostnamePath := filepath.Join(rootPath, "/proc/sys/kernel/hostname")
			if _, err := os.Stat(hostnamePath); err != nil {
				utils.PromptMsg("Error: Root path %s is not a valid root path, err: %s", rootPath, err.Error())
				return false
			}
		}
		p.rootPath = rootPath
	}
	if utils.IsPodEnv() {
		p.isInPods = true
		if p.rootPath == "" {
			// 提示在容器中运行时必须要指定rootPath
			utils.PromptMsg("Error: Root path must be specified when running in a container")
			return false
		}
	}
	return true
}

func checkAndInitOutput(outputFormat string, p *runParams) bool {
	reporter, err := report.GetReporter(outputFormat)
	if err != nil {
		utils.PromptMsg("Unsupported report type: %s", outputFormat)
		return false
	}
	p.reporter = reporter
	return true
}

func checkAndInitReportPath(reportPath string, p *runParams) bool {

	// 容许空字符串，空字符串表示在默认位置生成报告
	if reportPath == "" {
		return true
	}

	if stat, err := os.Stat(reportPath); os.IsNotExist(err) {
		if err := os.MkdirAll(reportPath, utils.DirModeOwnerReadWriteExec); err != nil {
			utils.PromptMsg("Failed to create output directory: %s, error: %s", reportPath, err.Error())
			return false
		}
	} else if !stat.IsDir() {
		utils.PromptMsg("Output path %s is not a directory", reportPath)
		return false
	}
	p.reportPath = reportPath
	return true
}
