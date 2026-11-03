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

// checker  package provides check functions for oschecktool
package checker

import (
	"encoding/json"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"

	"openfuyao.com/oscheck/internal/conf"
	"openfuyao.com/oscheck/internal/utils"
)

// Plugin 定义插件的基本信息, 整體配結構如下
// plugins:
//   - name: extend-ssh-checker  #扩展插件名称
//     execute: ssh-checker      #扩展插件执行命令
//     input: json|yaml|none     #扩展插件输入类型，可以为json、yaml、none，其中，json|yaml为将spec下的内容从stdin中传入，none则无输入
//     output: json|yaml|none    #扩展插件输出，其中json、yaml均从stdout中获取输入，return_code则用返回码判断，返回0则成功，非0失败
//
// plugins在plugins目录下，在目录下搜索所有plugins.yaml文件, 该yaml文件定义的内容则为插件的定义
// 其中，execute指定可执行文件路径(相对于plugins.yaml)
// input指定输入类型，json|yaml|none，其中，json|yaml为将spec下的内容从stdin中传入，none则无输入
// output指定输出类型，json|yaml|none，其中，json|yaml为从stdout中获取输出，return_code则用返回码判断，返回0则成功，非0失败
type Plugin struct {
	Name     string `json:"name"`
	Execute  string `json:"execute"`
	Input    string `json:"input"`
	Output   string `json:"output"`
	filepath string `json:"-"`
}

// Plugins 定义插件列表
type Plugins struct {
	Plugins []Plugin `json:"plugins"`
}

// loadPlugins 加载插件，并注册到checker中
func loadPlugins() {
	logger := utils.GetLogger("plugin")
	plugins, err := findPlugins(logger)
	if err != nil {
		logger.Printf("find plugins error: %v", err)
		return
	}

	existPlugins := map[string]Plugin{}
	for _, plugin := range plugins {
		plugin.Execute = filepath.Join(filepath.Dir(plugin.filepath), plugin.Execute)
		if plugin.Execute == "" {
			logger.Printf("plugin execute is empty, plugin: %v", plugin)
			continue
		}
		if !filepath.IsAbs(plugin.Execute) {
			plugin.Execute = filepath.Join(utils.GetStartDir(), "plugin", plugin.Execute)
		}
		stat, err := os.Stat(plugin.Execute)
		if err != nil {
			logger.Printf("plugin execute file not exist, plugin: %v, err: %v", plugin, err)
			continue
		}
		if !stat.IsDir() && stat.Mode()&utils.FileModeExecutable == 0 {
			logger.Printf("plugin execute is not executable, plugin: %s, config file: %s, execute: %s",
				plugin.Name,
				plugin.filepath,
				plugin.Execute)
			continue
		}

		if existPlugin, ok := existPlugins[plugin.Name]; ok {
			logger.Printf("plugin name dupllicate, plugin: %s, file:<%s>, <%s>, skip the second", plugin.Name,
				existPlugin.filepath,
				plugin.filepath)
			continue
		}
		RegisterCheckerFactory(plugin.Name, func() Checker {
			return &PluginChecker{conf: plugin}
		})
	}
}

func findPlugins(logger *log.Logger) ([]Plugin, error) {
	plugins := make([]Plugin, 0)
	// 便利./plugins目录下所有的plugins.yaml文件，可能有多个插件及多层级目录，需要记录Plugin的路径
	walkFn := func(path string, info fs.FileInfo, err error) error {
		if err != nil {
			// 记录日志后跳过
			logger.Printf("walk plugins file error, file: <%s>: %v", path, err)
			return err
		}
		if info.IsDir() || !(strings.HasSuffix(info.Name(), ".yaml") || strings.HasSuffix(info.Name(), ".yml")) {
			return nil
		}
		content, err := os.ReadFile(path)
		if err != nil {
			logger.Printf("read plugins file error, file: <%s>: %v", path, err)
			return nil
		}
		pluginSpec := Plugins{}
		if err := yaml.Unmarshal(content, &pluginSpec); err != nil {
			logger.Printf("unmarshal plugins file error, file: <%s>: %v", path, err)
			return nil
		}

		for _, plugin := range pluginSpec.Plugins {
			plugin.filepath = path
			plugins = append(plugins, plugin)
		}
		return nil

	}
	if err := filepath.Walk(filepath.Join(utils.GetStartDir(), "plugin"), walkFn); err != nil {
		return nil, err
	}
	return plugins, nil
}

// PluginChecker 插件检查器,
type PluginChecker struct {
	conf   Plugin
	item   conf.CheckItem
	ctx    Context
	logger *log.Logger
}

// Check 使用扩展插件来进行检查
func (p *PluginChecker) Check() ItemRslt {
	rslt := ItemRslt{
		Result: ResultValid,
	}
	if p.conf.Execute == "" {
		return rslt
	}
	input, err := p.getInput()
	if err != nil {
		p.logger.Printf("get plugin input error, plugin: %s, err: %v", p.conf.Name, err)
		rslt.Result = ResultError
		return rslt
	}
	exec := utils.Command{
		Command:       "sh",
		Args:          []string{"-c", p.conf.Execute},
		RootPath:      p.ctx.RootPath,
		CombineOutput: false,
		Stdin:         input,
		Logger:        p.logger,
	}
	execResult, err := exec.Exec()
	if err != nil {
		p.logger.Printf("exec plugin error, plugin: %s, err: %v", p.conf.Name, err)
		rslt.Result = ResultError
		return rslt
	}
	switch p.conf.Output {
	case "none":
		if execResult.ExitCode != 0 {
			rslt.Result = ResultInvalid
		}
	case "json":
		if err := json.Unmarshal([]byte(execResult.Stdout), &rslt); err != nil {
			p.logger.Printf("unmarshal plugin output error, plugin: %s, err: %v, stdout: %s, stderr:%s", p.conf.Name,
				err, execResult.Stdout, execResult.Stderr)
			rslt.Result = ResultError

		}
	case "yaml":
		if err := yaml.Unmarshal([]byte(execResult.Stdout), &rslt); err != nil {
			p.logger.Printf("unmarshal plugin output error, plugin: %s, err: %v, stdout: %s, stderr:%s", p.conf.Name,
				err, execResult.Stdout, execResult.Stderr)
			rslt.Result = ResultError
		}
	default:
		p.logger.Printf("unknown plugin output type, plugin: %s, type: %s", p.conf.Name, p.conf.Output)
		rslt.Result = ResultError
	}
	rslt.Key = p.item.Name
	rslt.Doc = p.item.Doc
	return rslt
}

func (p *PluginChecker) getInput() (string, error) {
	var inputBytes []byte
	var err error
	switch p.conf.Input {
	case "json":
		inputBytes, err = json.Marshal(p.item)
	case "yaml":
		inputBytes, err = yaml.Marshal(p.item)
	default:
		return "", nil
	}
	if err != nil {
		return "", err
	}
	return string(inputBytes), nil

}

// Init 初始化插件，实际上不做格式检查，只记录信息
func (p *PluginChecker) Init(ctx Context, item conf.CheckItem) *ParamCheckError {
	p.ctx = ctx
	p.item = item
	p.logger = utils.GetLogger("plugin-" + p.conf.Name)
	return nil
}

func init() {
	loadPlugins()
}
