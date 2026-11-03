/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package preprocess

import (
	"context"
	"fmt"
	"regexp"
	"sort"

	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/json"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/scriptutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const Name = "Preprocess"

// PreprocessPlugin 前置处理脚本执行器
type PreprocessPlugin struct {
	exec      exec.Executor
	k8sClient client.Client
}

// PreprocessConfig 前置处理配置
type PreprocessConfig struct {
	NodeIP  string                    `json:"nodeIP,omitempty"`
	BatchId string                    `json:"batchId,omitempty"`
	Scripts []scriptutil.ScriptConfig `json:"scripts"`
}

func New(exec exec.Executor, c client.Client) plugin.Plugin {
	return &PreprocessPlugin{
		k8sClient: c,
		exec:      exec,
	}
}

func (p *PreprocessPlugin) Name() string {
	return Name
}

func (p *PreprocessPlugin) Param() map[string]plugin.PluginParam {
	// nodeIP参数改为可选，如果不提供则自动获取
	return map[string]plugin.PluginParam{
		"nodeIP": {Key: "nodeIP", Value: "", Required: false, Default: "", Description: "node IP, if not provided, will be auto-detected"},
	}
}

// Execute 执行前置处理脚本
// 命令格式：["Preprocess"] 或 ["Preprocess", "nodeIP=192.168.1.10"]
func (p *PreprocessPlugin) Execute(ctx context.Context, commands []string) ([]string, error) {
	pc := *p
	pc.exec = exec.Bound(p.exec, ctx)
	p = &pc

	log.Infof("Preprocess: start, commands=%v", commands)

	// 1. 解析命令参数（nodeIP可选）
	paramMap, err := plugin.ParseCommands(p, commands)
	if err != nil {
		return nil, err
	}

	// 2. 获取节点IP（如果参数中没有提供，则自动获取）
	nodeIP := paramMap["nodeIP"]
	if nodeIP == "" {
		log.Infof("Preprocess: nodeIP not provided, auto-detecting")
		nodeIP, err = scriptutil.GetCurrentNodeIP()
		if err != nil {
			return nil, errors.Wrapf(err, "failed to get current node IP")
		}
	}
	log.Infof("Preprocess: using nodeIP=%s", nodeIP)

	// 3. 读取配置（根据优先级：全局 > 批次 > 节点，三种配置互斥，不合并）
	config, err := p.loadConfig(nodeIP)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to load config for node: %s", nodeIP)
	}
	log.Infof("Preprocess: loaded config, scripts=%d, nodeIP=%s", len(config.Scripts), nodeIP)

	// 4. 从user-system命名空间获取全量脚本列表
	allScripts, err := p.getAllScripts()
	if err != nil {
		return nil, errors.Wrapf(err, "failed to get all scripts")
	}
	log.Infof("Preprocess: total scripts in user-system=%d", len(allScripts))

	// 5. 按Order排序，过滤配置中不存在的脚本
	sort.Slice(config.Scripts, func(i, j int) bool {
		return config.Scripts[i].Order < config.Scripts[j].Order
	})

	// 6. 执行每个脚本
	var results []string
	for _, scriptConfig := range config.Scripts {
		log.Infof("Preprocess: preparing script=%s, order=%d", scriptConfig.ScriptName, scriptConfig.Order)

		// 检查脚本是否存在
		if !p.scriptExists(allScripts, scriptConfig.ScriptName) {
			log.Warnf("Preprocess: script not found in user-system, skip=%s", scriptConfig.ScriptName)
			continue // 如果脚本配置文件中没有该脚本，则不执行
		}

		// 执行脚本
		result, err := p.executeScript(scriptConfig, nodeIP)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to execute script: %s", scriptConfig.ScriptName)
		}
		results = append(results, result)
		log.Infof("Preprocess: script executed successfully, script=%s", scriptConfig.ScriptName)
	}

	log.Infof("Preprocess: finished, executed scripts=%d", len(results))
	return results, nil
}

// loadConfig 加载配置（根据优先级：全局 > 批次 > 节点，三种配置互斥，不合并）
func (p *PreprocessPlugin) loadConfig(nodeIP string) (*PreprocessConfig, error) {
	ctx := context.Background()

	// 1. 优先查找全局配置
	globalConfigCM := &corev1.ConfigMap{}
	globalConfigKey := client.ObjectKey{
		Namespace: "user-system",
		Name:      "preprocess-all-config",
	}
	if err := p.k8sClient.Get(ctx, globalConfigKey, globalConfigCM); err == nil {
		log.Infof("Preprocess: config hit global=%s", globalConfigKey.Name)
		// 全局配置存在，直接使用，不再查找批次和节点配置
		return p.parseConfig(globalConfigCM)
	}
	log.Infof("Preprocess: config miss global=%s", globalConfigKey.Name)

	batchConfig, err := scriptutil.LoadBatchConfig(ctx, p.k8sClient, scriptutil.PreprocessBatchConfig, nodeIP)
	if err != nil {
		log.Warnf("Preprocess: batch config lookup failed for nodeIP=%s, err=%v", nodeIP, err)
	}
	if batchConfig != nil {
		log.Infof("Preprocess: config hit batch=%s for nodeIP=%s", batchConfig.Name, nodeIP)
		return p.parseConfig(batchConfig)
	}

	// 3. 全局和批次配置都不存在，查找节点配置
	nodeConfigCM := &corev1.ConfigMap{}
	nodeConfigKey := client.ObjectKey{
		Namespace: "user-system",
		Name:      fmt.Sprintf("preprocess-config-node-%s", nodeIP),
	}
	if err := p.k8sClient.Get(ctx, nodeConfigKey, nodeConfigCM); err == nil {
		log.Infof("Preprocess: config hit node=%s", nodeConfigKey.Name)
		// 节点配置存在，直接使用
		return p.parseConfig(nodeConfigCM)
	}
	log.Infof("Preprocess: config miss node=%s", nodeConfigKey.Name)

	// 三种配置都不存在
	return nil, errors.New("no config found (global, batch, or node)")
}

// parseConfig 解析配置JSON
func (p *PreprocessPlugin) parseConfig(configCM *corev1.ConfigMap) (*PreprocessConfig, error) {
	configJSON, ok := configCM.Data["config.json"]
	if !ok {
		log.Warnf("Preprocess: config.json not found, configMap=%s", configCM.Name)
		return nil, errors.New("config.json not found in ConfigMap")
	}

	var config PreprocessConfig
	if err := json.Unmarshal([]byte(configJSON), &config); err != nil {
		log.Warnf("Preprocess: parse config.json failed, configMap=%s, err=%v", configCM.Name, err)
		return nil, errors.Wrapf(err, "failed to parse config.json")
	}

	return &config, nil
}

// getAllScripts 获取全量脚本列表
func (p *PreprocessPlugin) getAllScripts() ([]string, error) {
	ctx := context.Background()

	// 列出user-system命名空间下所有标记为脚本的ConfigMap
	scriptList := &corev1.ConfigMapList{}
	if err := p.k8sClient.List(ctx, scriptList, client.InNamespace("user-system"),
		client.MatchingLabels{"bke.preprocess.script": "true"}); err != nil {
		log.Warnf("Preprocess: list scripts failed: %v", err)
		return nil, errors.Wrapf(err, "failed to list scripts")
	}

	var scripts []string
	for _, cm := range scriptList.Items {
		scripts = append(scripts, cm.Name)
	}

	return scripts, nil
}

// scriptExists 检查脚本是否存在
func (p *PreprocessPlugin) scriptExists(allScripts []string, scriptName string) bool {
	for _, s := range allScripts {
		if s == scriptName {
			return true
		}
	}
	return false
}

// executeScript 执行单个脚本
func (p *PreprocessPlugin) executeScript(scriptConfig scriptutil.ScriptConfig, nodeIP string) (string, error) {
	log.Infof("Preprocess: executing script=%s, nodeIP=%s", scriptConfig.ScriptName, nodeIP)

	// 1. 从user-system命名空间读取脚本ConfigMap
	scriptCM := &corev1.ConfigMap{}
	if err := p.k8sClient.Get(context.Background(),
		client.ObjectKey{Namespace: "user-system", Name: scriptConfig.ScriptName}, scriptCM); err != nil {
		log.Warnf("Preprocess: get script configMap failed, script=%s, err=%v", scriptConfig.ScriptName, err)
		return "", errors.Wrapf(err, "failed to get script ConfigMap: %s", scriptConfig.ScriptName)
	}

	// 2. 提取脚本内容（脚本中包含参数模板，如 ${NODE_IP}, ${HTTP_REPO}）
	scriptContent, ok := scriptCM.Data[scriptConfig.ScriptName]
	if !ok {
		log.Warnf("Preprocess: script content not found in configMap, script=%s", scriptConfig.ScriptName)
		return "", errors.Errorf("script content not found in ConfigMap: %s", scriptConfig.ScriptName)
	}

	// 3. 参数校验（防参数注入）- 在参数渲染之前进行校验
	if err := p.validateParams(scriptConfig.Params); err != nil {
		log.Warnf("Preprocess: parameter validation failed, skip script=%s, err=%v", scriptConfig.ScriptName, err)
		return fmt.Sprintf("skipped: script=%s, reason=invalid params, err=%v", scriptConfig.ScriptName, err), nil
	}

	// 4. 参数渲染：将参数值替换到脚本模板中
	renderedScript, err := p.renderScriptWithParams(scriptContent, scriptConfig, nodeIP)
	if err != nil {
		log.Warnf("Preprocess: render script failed, script=%s, err=%v", scriptConfig.ScriptName, err)
		return "", errors.Wrapf(err, "failed to render script with params: %s", scriptConfig.ScriptName)
	}

	// 5. 落盘渲染后的脚本内容
	scriptPath, err := p.writeRenderedScriptToDisk(scriptConfig.ScriptName, nodeIP, renderedScript)
	if err != nil {
		log.Warnf("Preprocess: write rendered script failed, script=%s, err=%v", scriptConfig.ScriptName, err)
		return "", errors.Wrapf(err, "failed to write rendered script to disk: %s", scriptConfig.ScriptName)
	}
	log.Infof("Preprocess: rendered script saved, script=%s, path=%s", scriptConfig.ScriptName, scriptPath)

	// 6. 执行脚本：执行落盘后的脚本
	output, err := p.executeRenderedScript(scriptPath)
	if err != nil {
		log.Warnf("Preprocess: execute rendered script failed, script=%s, nodeIP=%s, path=%s, output=%s, err=%v",
			scriptConfig.ScriptName, nodeIP, scriptPath, output, err)
		log.Warnf("Preprocess: rendered script preview, script=%s, length=%d, preview=%s",
			scriptConfig.ScriptName, len(renderedScript), scriptutil.PreviewScript(renderedScript, 800))
		return "", errors.Wrapf(err, "failed to execute script: %s, output: %s", scriptConfig.ScriptName, output)
	}

	log.Infof("Preprocess: script completed, script=%s", scriptConfig.ScriptName)
	return output, nil
}

// validateParams 参数校验（防参数注入）
// 参数值白名单：只允许 a-z A-Z 0-9 - _ / . 空格 : # =
func (p *PreprocessPlugin) validateParams(params map[string]string) error {
	// 参数值白名单正则：只允许 a-z A-Z 0-9 - _ / . 空格 : # =
	allowedValuePattern := regexp.MustCompile(`^[a-zA-Z0-9\-_/.\s:#=]*$`)

	for key, value := range params {
		// 检查键名
		if !p.isValidParamName(key) {
			log.Warnf("Preprocess: invalid parameter name=%s", key)
			return errors.Errorf("invalid parameter name: %s", key)
		}

		// 检查值：使用白名单模式
		if !allowedValuePattern.MatchString(value) {
			log.Warnf("Preprocess: invalid parameter value, key=%s, value=%s", key, value)
			return errors.Errorf("parameter %s contains invalid characters. Only a-z, A-Z, 0-9, -, _, /, ., space, :, #, = are allowed. Value: %s", key, value)
		}

		// 检查长度（防止过长参数）
		if len(value) > 4096 {
			return errors.Errorf("parameter %s is too long (max 4096 characters)", key)
		}
	}

	return nil
}

// isValidParamName 检查参数名是否有效
func (p *PreprocessPlugin) isValidParamName(name string) bool {
	// 参数名只能包含字母、数字、下划线
	pattern := `^[a-zA-Z_][a-zA-Z0-9_]*$`
	matched, err := regexp.MatchString(pattern, name)

	if err != nil {
		log.Warnf("Regex matching parameter name failed, pattern: %s, paramName: %s, error: %v", pattern, name, err)
		return false
	}

	return matched
}

// renderScriptWithParams 参数渲染：将参数值替换到脚本模板中
func (p *PreprocessPlugin) renderScriptWithParams(scriptContent string, scriptConfig scriptutil.ScriptConfig, nodeIP string) (string, error) {
	// 1. 准备参数映射
	params := make(map[string]string)
	params["NODE_IP"] = nodeIP // 默认参数

	// 添加用户自定义参数
	for key, value := range scriptConfig.Params {
		params[key] = value
	}
	log.Debugf("Preprocess: render params, script=%s, params=%d", scriptConfig.ScriptName, len(params))

	// 2. 在脚本内容中查找参数模板（格式：${PARAM_NAME}）并替换
	renderedScript := scriptutil.RenderScriptWithParams(scriptContent, params)

	return renderedScript, nil
}

// executeRenderedScript 执行渲染后的脚本（直接执行，不通过环境变量）
func (p *PreprocessPlugin) executeRenderedScript(scriptPath string) (string, error) {
	log.Infof("Preprocess: executing script file, path=%s", scriptPath)
	output, err := p.exec.ExecuteCommandWithCombinedOutput("/bin/sh", scriptPath)
	if err != nil {
		return "", errors.Wrapf(err, "failed to execute rendered script, output: %s", output)
	}

	return output, nil
}

func (p *PreprocessPlugin) writeRenderedScriptToDisk(scriptName, nodeIP, renderedScript string) (string, error) {
	scriptPath, err := scriptutil.WriteRenderedScriptToDisk(utils.AgentScripts, scriptName, nodeIP, renderedScript)
	if err != nil {
		return "", err
	}

	log.Infof("Preprocess: script stored on disk, path=%s, size=%d", scriptPath, len(renderedScript))
	return scriptPath, nil
}
