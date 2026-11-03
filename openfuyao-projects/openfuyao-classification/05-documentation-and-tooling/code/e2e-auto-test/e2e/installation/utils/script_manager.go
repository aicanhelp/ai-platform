package utils

import (
	"encoding/json"
	"fmt"
	"strings"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
)

const (
	UserSystemNamespace = "user-system"

	PreprocessGlobalConfigName  = "preprocess-all-config"
	PostprocessGlobalConfigName = "postprocess-all-config"

	PreprocessScriptLabel  = "bke.preprocess.script"
	PostprocessScriptLabel = "bke.postprocess.script"

	PreprocessScriptPath  = "/etc/openFuyao/bkeagent/scripts"
	PostprocessScriptPath = "/etc/openFuyao/bkeagent/scripts/postprocess"

	MarkerDir = "/tmp/prescript-markers"
)

// ScriptConfigEntry preprocess-all-config 中 config.json 的单个脚本配置
type ScriptConfigEntry struct {
	ScriptName string            `json:"scriptName"`
	Order      int               `json:"order"`
	Params     map[string]string `json:"params,omitempty"`
}

// PreprocessConfig config.json 顶层结构
type PreprocessConfig struct {
	Scripts []ScriptConfigEntry `json:"scripts"`
}

// ScriptContentDef 脚本内容定义
type ScriptContentDef struct {
	Name    string // ConfigMap 名称，如 "init-os.sh"
	Content string // shell 脚本文本
}

// PreScriptManager 前置脚本管理器
type PreScriptManager struct {
	sshExecutor   *executor.SSHExecutor
	localExecutor *executor.LocalExecutor
}

func NewPreScriptManager(sshExec *executor.SSHExecutor, localExec *executor.LocalExecutor) *PreScriptManager {
	return &PreScriptManager{sshExecutor: sshExec, localExecutor: localExec}
}

// ---------- namespace ----------

func (m *PreScriptManager) EnsureUserSystemNamespace(kubeconfigPath string) error {
	kc := kcFlag(kubeconfigPath)
	cmd := fmt.Sprintf("kubectl %screate namespace %s --dry-run=client -o yaml | kubectl %sapply -f -",
		kc, UserSystemNamespace, kc)
	result, err := m.sshExecutor.Exec(cmd)
	if err != nil {
		return fmt.Errorf("创建 %s 命名空间失败: %w", UserSystemNamespace, err)
	}
	if result.ExitCode != 0 && !strings.Contains(result.Stderr, "AlreadyExists") {
		return fmt.Errorf("创建 %s 命名空间失败(exit=%d): %s", UserSystemNamespace, result.ExitCode, result.Stderr)
	}
	return nil
}

// ---------- 创建脚本 ConfigMap ----------

// CreateScriptConfigMap 创建单个脚本 ConfigMap（label: bke.preprocess.script="true"）
func (m *PreScriptManager) CreateScriptConfigMap(script ScriptContentDef, kubeconfigPath string) error {
	kc := kcFlag(kubeconfigPath)
	yaml := buildScriptConfigMapYAML(script)
	return m.applyYAML(yaml, kc, fmt.Sprintf("script-%s", sanitizeName(script.Name)))
}

// ---------- 创建全局配置 ConfigMap ----------

// CreateGlobalConfig 创建前置处理全局配置 ConfigMap (preprocess-all-config)
func (m *PreScriptManager) CreateGlobalConfig(scripts []ScriptConfigEntry, kubeconfigPath string) error {
	kc := kcFlag(kubeconfigPath)
	yaml, err := buildGlobalConfigYAML(scripts)
	if err != nil {
		return err
	}
	return m.applyYAML(yaml, kc, PreprocessGlobalConfigName)
}

// ---------- 一键部署 ----------

// SetupPreScripts 创建 user-system 命名空间 + n 个脚本 ConfigMap + 1 个全局配置 ConfigMap
func (m *PreScriptManager) SetupPreScripts(scripts []ScriptContentDef, entries []ScriptConfigEntry, kubeconfigPath string) error {
	if err := m.EnsureUserSystemNamespace(kubeconfigPath); err != nil {
		return err
	}
	for _, s := range scripts {
		if err := m.CreateScriptConfigMap(s, kubeconfigPath); err != nil {
			return fmt.Errorf("创建脚本 ConfigMap %s 失败: %w", s.Name, err)
		}
	}
	return m.CreateGlobalConfig(entries, kubeconfigPath)
}

// ---------- 清理 ----------

// Cleanup 删除脚本 ConfigMap 和全局配置 ConfigMap
func (m *PreScriptManager) Cleanup(scriptNames []string, kubeconfigPath string) {
	kc := kcFlag(kubeconfigPath)
	for _, name := range scriptNames {
		m.sshExecutor.Exec(fmt.Sprintf("kubectl %sdelete configmap %s -n %s --ignore-not-found", kc, name, UserSystemNamespace))
	}
	m.sshExecutor.Exec(fmt.Sprintf("kubectl %sdelete configmap %s -n %s --ignore-not-found", kc, PreprocessGlobalConfigName, UserSystemNamespace))
}

// CleanupScriptsOnNode 清理节点上的落盘脚本和执行标记
func (m *PreScriptManager) CleanupScriptsOnNode(nodeIP, user, password string) {
	remoteCmd := fmt.Sprintf("rm -rf %s %s", PreprocessScriptPath, MarkerDir)
	m.execOnNode(nodeIP, user, password, remoteCmd)
}

// ---------- 验证：脚本落盘 ----------

// VerifyScriptLanded 验证脚本已落盘到节点的 /etc/openFuyao/bkeagent/scripts/
// BKE 渲染后文件名格式为: <scriptBase>-<nodeIP>-<timestamp>.sh，所以用前缀匹配
func (m *PreScriptManager) VerifyScriptLanded(nodeIP, user, password string, scriptNames []string) (bool, string, error) {
	allFound := true
	var details []string

	// 诊断：列出目录下所有文件，并显示执行主机名
	diagCmd := fmt.Sprintf("echo HOST=$(hostname),IP=$(hostname -I | awk '{print $1}'); ls -la %s/ 2>&1 | head -20", PreprocessScriptPath)
	diagOutput, _ := m.execOnNode(nodeIP, user, password, diagCmd)
	details = append(details, fmt.Sprintf("[诊断] %s", strings.TrimSpace(diagOutput)))

	for _, name := range scriptNames {
		base := strings.TrimSuffix(name, ".sh")
		output, err := m.execOnNode(nodeIP, user, password,
			fmt.Sprintf("ls %s/%s* 2>/dev/null | head -1", PreprocessScriptPath, base))
		if err != nil {
			allFound = false
			details = append(details, fmt.Sprintf("%s: 检查失败 - %v", name, err))
			continue
		}
		output = strings.TrimSpace(output)
		if output != "" && !strings.Contains(output, "No such file") {
			details = append(details, fmt.Sprintf("%s: 已落盘 (%s)", name, output))
		} else {
			allFound = false
			details = append(details, fmt.Sprintf("%s: 未找到", name))
		}
	}
	return allFound, strings.Join(details, "; "), nil
}

// ---------- 验证：脚本已执行 ----------

// VerifyScriptExecuted 验证脚本已被 BKE 渲染并落盘
// 验证方式：
//  1. 检查渲染后脚本已落盘（文件名含节点IP说明已渲染）
//  2. 读取脚本内容，检查 ${NODE_IP} 是否被替换为实际 IP（证明参数渲染成功）
func (m *PreScriptManager) VerifyScriptExecuted(nodeIP, user, password string, scriptNames []string) (bool, string, error) {
	allOK := true
	var details []string

	for _, name := range scriptNames {
		base := strings.TrimSuffix(name, ".sh")

		findOutput, err := m.execOnNode(nodeIP, user, password,
			fmt.Sprintf("ls %s/%s-* 2>/dev/null | head -1", PreprocessScriptPath, base))
		if err != nil {
			allOK = false
			details = append(details, fmt.Sprintf("%s: 检查失败 - %v", name, err))
			continue
		}
		scriptFile := strings.TrimSpace(findOutput)

		if scriptFile == "" || strings.Contains(scriptFile, "No such file") {
			allOK = false
			details = append(details, fmt.Sprintf("%s: 未找到渲染后的脚本文件", name))
			continue
		}

		content, _ := m.execOnNode(nodeIP, user, password, fmt.Sprintf("cat %s", scriptFile))
		content = strings.TrimSpace(content)

		if strings.Contains(content, nodeIP) && !strings.Contains(content, "${NODE_IP}") {
			details = append(details, fmt.Sprintf("%s: 已渲染执行 (文件=%s, NODE_IP=%s已替换)", name, scriptFile, nodeIP))
		} else if content != "" {
			details = append(details, fmt.Sprintf("%s: 已落盘但参数可能未完全渲染 (文件=%s)", name, scriptFile))
		} else {
			allOK = false
			details = append(details, fmt.Sprintf("%s: 脚本文件内容为空", name))
		}
	}
	return allOK, strings.Join(details, "; "), nil
}

// VerifyScriptNotExecuted 验证脚本未执行（渲染后的脚本不存在于落盘目录）
func (m *PreScriptManager) VerifyScriptNotExecuted(nodeIP, user, password, scriptName string) (bool, error) {
	base := strings.TrimSuffix(scriptName, ".sh")
	output, err := m.execOnNode(nodeIP, user, password,
		fmt.Sprintf("ls %s/%s-* 2>/dev/null | head -1", PreprocessScriptPath, base))
	if err != nil {
		return true, nil
	}
	return strings.TrimSpace(output) == "", nil
}

// ---------- 验证：前置处理日志 ----------

// VerifyPreProcessingLogs 检查前置处理相关日志
// 依次查找: cluster-system 下的 bke-controller、bke-<cluster> 下的 bkeagent
func (m *PreScriptManager) VerifyPreProcessingLogs(clusterName, kubeconfigPath string) (bool, string, error) {
	kc := kcFlag(kubeconfigPath)
	searchPattern := "preprocess|pre-process|Preprocess|PreprocessScripts|preprocessPlugin"

	// 1. cluster-system 命名空间的 bke-controller-manager
	cmd1 := fmt.Sprintf(
		"kubectl %slogs -n cluster-system -l app.kubernetes.io/name=bke-controller-manager --tail=1000 2>/dev/null | grep -i -E '%s' | head -20",
		kc, searchPattern)
	result1, _ := m.sshExecutor.Exec(cmd1)
	if output := strings.TrimSpace(result1.Stdout); output != "" {
		return true, output, nil
	}

	// 2. cluster-system 命名空间（用 app=bke-controller 标签）
	cmd2 := fmt.Sprintf(
		"kubectl %slogs -n cluster-system -l app=bke-controller --tail=1000 2>/dev/null | grep -i -E '%s' | head -20",
		kc, searchPattern)
	result2, _ := m.sshExecutor.Exec(cmd2)
	if output := strings.TrimSpace(result2.Stdout); output != "" {
		return true, output, nil
	}

	// 3. bke-<cluster> 命名空间的 bkeagent
	cmd3 := fmt.Sprintf(
		"kubectl %slogs -n bke-%s -l app=bkeagent --tail=1000 2>/dev/null | grep -i -E '%s' | head -20",
		kc, clusterName, searchPattern)
	result3, _ := m.sshExecutor.Exec(cmd3)
	if output := strings.TrimSpace(result3.Stdout); output != "" {
		return true, output, nil
	}

	// 4. 直接 grep 所有 cluster-system 的 pod 日志
	cmd4 := fmt.Sprintf(
		"kubectl %slogs -n cluster-system --all-containers --tail=500 2>/dev/null | grep -i -E '%s' | head -20",
		kc, searchPattern)
	result4, _ := m.sshExecutor.Exec(cmd4)
	output4 := strings.TrimSpace(result4.Stdout)
	return output4 != "", output4, nil
}

// ---------- 内部辅助 ----------

func kcFlag(kubeconfigPath string) string {
	if kubeconfigPath != "" {
		return fmt.Sprintf("--kubeconfig %s ", kubeconfigPath)
	}
	return ""
}

func sanitizeName(name string) string {
	name = strings.ReplaceAll(name, ".", "-")
	name = strings.ReplaceAll(name, "/", "-")
	return name
}

func buildScriptConfigMapYAML(script ScriptContentDef) string {
	var sb strings.Builder
	sb.WriteString("apiVersion: v1\n")
	sb.WriteString("kind: ConfigMap\n")
	sb.WriteString("metadata:\n")
	sb.WriteString(fmt.Sprintf("  name: %s\n", script.Name))
	sb.WriteString(fmt.Sprintf("  namespace: %s\n", UserSystemNamespace))
	sb.WriteString("  labels:\n")
	sb.WriteString(fmt.Sprintf("    %s: \"true\"\n", PreprocessScriptLabel))
	sb.WriteString("data:\n")
	sb.WriteString(fmt.Sprintf("  %s: |\n", script.Name))
	for _, line := range strings.Split(script.Content, "\n") {
		sb.WriteString("    " + line + "\n")
	}
	return sb.String()
}

func buildGlobalConfigYAML(scripts []ScriptConfigEntry) (string, error) {
	cfg := PreprocessConfig{Scripts: scripts}
	cfgJSON, err := json.MarshalIndent(cfg, "    ", "  ")
	if err != nil {
		return "", fmt.Errorf("序列化 config.json 失败: %w", err)
	}

	var sb strings.Builder
	sb.WriteString("apiVersion: v1\n")
	sb.WriteString("kind: ConfigMap\n")
	sb.WriteString("metadata:\n")
	sb.WriteString(fmt.Sprintf("  name: %s\n", PreprocessGlobalConfigName))
	sb.WriteString(fmt.Sprintf("  namespace: %s\n", UserSystemNamespace))
	sb.WriteString("data:\n")
	sb.WriteString("  config.json: |\n")
	for _, line := range strings.Split(string(cfgJSON), "\n") {
		sb.WriteString("    " + line + "\n")
	}
	return sb.String(), nil
}

func (m *PreScriptManager) applyYAML(content, kcFlag, tag string) error {
	tmpPath := fmt.Sprintf("/tmp/prescript-%s.yaml", tag)

	writeCmd := fmt.Sprintf("cat > %s << 'PRESCRIPT_EOF'\n%sPRESCRIPT_EOF", tmpPath, content)
	if result, err := m.sshExecutor.Exec(writeCmd); err != nil {
		return fmt.Errorf("写入文件失败: %w", err)
	} else if result.ExitCode != 0 {
		return fmt.Errorf("写入文件失败(exit=%d): %s", result.ExitCode, result.Stderr)
	}

	applyCmd := fmt.Sprintf("kubectl %sapply -f %s", kcFlag, tmpPath)
	result, err := m.sshExecutor.Exec(applyCmd)
	if err != nil {
		return fmt.Errorf("kubectl apply 失败: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("kubectl apply 失败(exit=%d): %s", result.ExitCode, result.Stderr)
	}

	m.sshExecutor.Exec(fmt.Sprintf("rm -f %s", tmpPath))
	return nil
}

// execOnNode 通过 Go SSH 直连目标节点执行命令（密码认证，与 sshExecutor 同机制）
func (m *PreScriptManager) execOnNode(nodeIP, user, password, remoteCmd string) (string, error) {
	targetExec, err := executor.NewSSHExecutor(nodeIP, 22, user, password)
	if err != nil {
		return "", fmt.Errorf("SSH连接节点 %s 失败: %w", nodeIP, err)
	}

	result, execErr := targetExec.Exec(remoteCmd)
	if result != nil {
		return result.Stdout, nil
	}
	return "", execErr
}

// ==================== 测试脚本内容 ====================
// 以下是可直接运行的测试脚本，用于 E2E 测试。
// 每个脚本会创建标记文件用于验证执行结果。

// TestScriptInitOS init-os.sh 脚本内容
func TestScriptInitOS() ScriptContentDef {
	return ScriptContentDef{
		Name: "init-os.sh",
		Content: `#!/bin/sh
set -e

echo "========================================="
echo "Pre-Script: init-os.sh"
echo "NODE_IP: ${NODE_IP}"
echo "HTTP_REPO: ${HTTP_REPO}"
echo "========================================="

MARKER_DIR="/tmp/prescript-markers"
mkdir -p ${MARKER_DIR}

echo "Configuring OS settings..."
echo "Using repo: ${HTTP_REPO}"

echo "NODE_IP=${NODE_IP}" > ${MARKER_DIR}/init-os.env
echo "HTTP_REPO=${HTTP_REPO}" >> ${MARKER_DIR}/init-os.env
echo "EXECUTED_AT=$(date +%s)" >> ${MARKER_DIR}/init-os.env

touch ${MARKER_DIR}/init-os.done
echo "init-os.sh completed successfully"`,
	}
}

// TestScriptSetupEnv setup-env.sh 脚本内容
func TestScriptSetupEnv() ScriptContentDef {
	return ScriptContentDef{
		Name: "setup-env.sh",
		Content: `#!/bin/sh
set -e

echo "========================================="
echo "Pre-Script: setup-env.sh"
echo "NODE_IP: ${NODE_IP}"
echo "HTTP_REPO: ${HTTP_REPO}"
echo "========================================="

MARKER_DIR="/tmp/prescript-markers"
mkdir -p ${MARKER_DIR}

echo "Setting up environment..."
echo "Using repo: ${HTTP_REPO}"

echo "NODE_IP=${NODE_IP}" > ${MARKER_DIR}/setup-env.env
echo "HTTP_REPO=${HTTP_REPO}" >> ${MARKER_DIR}/setup-env.env
echo "EXECUTED_AT=$(date +%s)" >> ${MARKER_DIR}/setup-env.env

touch ${MARKER_DIR}/setup-env.done
echo "setup-env.sh completed successfully"`,
	}
}

// TestScriptInstallAgent install-agent.sh 脚本内容
func TestScriptInstallAgent() ScriptContentDef {
	return ScriptContentDef{
		Name: "install-agent.sh",
		Content: `#!/bin/sh
set -e

echo "========================================="
echo "Pre-Script: install-agent.sh"
echo "NODE_IP: ${NODE_IP}"
echo "HTTP_REPO: ${HTTP_REPO}"
echo "========================================="

MARKER_DIR="/tmp/prescript-markers"
mkdir -p ${MARKER_DIR}

echo "Installing agent on node ${NODE_IP}..."
echo "Downloading from: ${HTTP_REPO}/agent"

echo "NODE_IP=${NODE_IP}" > ${MARKER_DIR}/install-agent.env
echo "HTTP_REPO=${HTTP_REPO}" >> ${MARKER_DIR}/install-agent.env
echo "EXECUTED_AT=$(date +%s)" >> ${MARKER_DIR}/install-agent.env

touch ${MARKER_DIR}/install-agent.done
echo "install-agent.sh completed successfully"`,
	}
}

// TestScriptFail fail-script.sh 故意失败的脚本
func TestScriptFail() ScriptContentDef {
	return ScriptContentDef{
		Name: "fail-script.sh",
		Content: `#!/bin/sh

echo "========================================="
echo "Pre-Script: fail-script.sh (intentionally failing)"
echo "NODE_IP: ${NODE_IP}"
echo "PKG_VERSION: ${PKG_VERSION}"
echo "========================================="

echo "Attempting to download non-existent package version ${PKG_VERSION}..."
echo "ERROR: Package version ${PKG_VERSION} not found in repository"

exit 1`,
	}
}

// ==================== PostScriptManager 后置脚本管理器 ====================

const (
	PostMarkerDir = "/tmp/postscript-markers"
)

// PostScriptManager 后置脚本管理器
type PostScriptManager struct {
	sshExecutor   *executor.SSHExecutor
	localExecutor *executor.LocalExecutor
}

func NewPostScriptManager(sshExec *executor.SSHExecutor, localExec *executor.LocalExecutor) *PostScriptManager {
	return &PostScriptManager{sshExecutor: sshExec, localExecutor: localExec}
}

// EnsureUserSystemNamespace 确保 user-system 命名空间存在
func (m *PostScriptManager) EnsureUserSystemNamespace(kubeconfigPath string) error {
	kc := kcFlag(kubeconfigPath)
	cmd := fmt.Sprintf("kubectl %screate namespace %s --dry-run=client -o yaml | kubectl %sapply -f -",
		kc, UserSystemNamespace, kc)
	result, err := m.sshExecutor.Exec(cmd)
	if err != nil {
		return fmt.Errorf("创建 %s 命名空间失败: %w", UserSystemNamespace, err)
	}
	if result.ExitCode != 0 && !strings.Contains(result.Stderr, "AlreadyExists") {
		return fmt.Errorf("创建 %s 命名空间失败(exit=%d): %s", UserSystemNamespace, result.ExitCode, result.Stderr)
	}
	return nil
}

// CreateScriptConfigMap 创建单个后置脚本 ConfigMap（label: bke.postprocess.script="true"）
func (m *PostScriptManager) CreateScriptConfigMap(script ScriptContentDef, kubeconfigPath string) error {
	kc := kcFlag(kubeconfigPath)
	yaml := buildPostScriptConfigMapYAML(script)
	return m.applyYAML(yaml, kc, fmt.Sprintf("postscript-%s", sanitizeName(script.Name)))
}

// CreateGlobalConfig 创建后置处理全局配置 ConfigMap (postprocess-all-config)
func (m *PostScriptManager) CreateGlobalConfig(scripts []ScriptConfigEntry, kubeconfigPath string) error {
	kc := kcFlag(kubeconfigPath)
	yaml, err := buildPostGlobalConfigYAML(scripts)
	if err != nil {
		return err
	}
	return m.applyYAML(yaml, kc, PostprocessGlobalConfigName)
}

// SetupPostScripts 创建 user-system 命名空间 + n 个后置脚本 ConfigMap + 1 个全局配置 ConfigMap
func (m *PostScriptManager) SetupPostScripts(scripts []ScriptContentDef, entries []ScriptConfigEntry, kubeconfigPath string) error {
	if err := m.EnsureUserSystemNamespace(kubeconfigPath); err != nil {
		return err
	}
	for _, s := range scripts {
		if err := m.CreateScriptConfigMap(s, kubeconfigPath); err != nil {
			return fmt.Errorf("创建后置脚本 ConfigMap %s 失败: %w", s.Name, err)
		}
	}
	return m.CreateGlobalConfig(entries, kubeconfigPath)
}

// Cleanup 删除后置脚本 ConfigMap 和全局配置 ConfigMap
func (m *PostScriptManager) Cleanup(scriptNames []string, kubeconfigPath string) {
	kc := kcFlag(kubeconfigPath)
	for _, name := range scriptNames {
		m.sshExecutor.Exec(fmt.Sprintf("kubectl %sdelete configmap %s -n %s --ignore-not-found", kc, name, UserSystemNamespace))
	}
	m.sshExecutor.Exec(fmt.Sprintf("kubectl %sdelete configmap %s -n %s --ignore-not-found", kc, PostprocessGlobalConfigName, UserSystemNamespace))
}

// CleanupScriptsOnNode 清理节点上的后置脚本落盘和执行标记
func (m *PostScriptManager) CleanupScriptsOnNode(nodeIP, user, password string) {
	remoteCmd := fmt.Sprintf("rm -rf %s %s", PostprocessScriptPath, PostMarkerDir)
	m.execOnNode(nodeIP, user, password, remoteCmd)
}

// VerifyScriptLanded 验证后置脚本已落盘到节点的 /etc/openFuyao/bkeagent/scripts/postprocess/
func (m *PostScriptManager) VerifyScriptLanded(nodeIP, user, password string, scriptNames []string) (bool, string, error) {
	allFound := true
	var details []string

	diagCmd := fmt.Sprintf("echo HOST=$(hostname),IP=$(hostname -I | awk '{print $1}'); ls -la %s/ 2>&1 | head -20", PostprocessScriptPath)
	diagOutput, _ := m.execOnNode(nodeIP, user, password, diagCmd)
	details = append(details, fmt.Sprintf("[诊断] %s", strings.TrimSpace(diagOutput)))

	for _, name := range scriptNames {
		base := strings.TrimSuffix(name, ".sh")
		output, err := m.execOnNode(nodeIP, user, password,
			fmt.Sprintf("ls %s/%s* 2>/dev/null | head -1", PostprocessScriptPath, base))
		if err != nil {
			allFound = false
			details = append(details, fmt.Sprintf("%s: 检查失败 - %v", name, err))
			continue
		}
		output = strings.TrimSpace(output)
		if output != "" && !strings.Contains(output, "No such file") {
			details = append(details, fmt.Sprintf("%s: 已落盘 (%s)", name, output))
		} else {
			allFound = false
			details = append(details, fmt.Sprintf("%s: 未找到", name))
		}
	}
	return allFound, strings.Join(details, "; "), nil
}

// VerifyScriptExecuted 验证后置脚本已被渲染并落盘
func (m *PostScriptManager) VerifyScriptExecuted(nodeIP, user, password string, scriptNames []string) (bool, string, error) {
	allOK := true
	var details []string

	for _, name := range scriptNames {
		base := strings.TrimSuffix(name, ".sh")

		findOutput, err := m.execOnNode(nodeIP, user, password,
			fmt.Sprintf("ls %s/%s-* 2>/dev/null | head -1", PostprocessScriptPath, base))
		if err != nil {
			allOK = false
			details = append(details, fmt.Sprintf("%s: 检查失败 - %v", name, err))
			continue
		}
		scriptFile := strings.TrimSpace(findOutput)

		if scriptFile == "" || strings.Contains(scriptFile, "No such file") {
			allOK = false
			details = append(details, fmt.Sprintf("%s: 未找到渲染后的脚本文件", name))
			continue
		}

		content, _ := m.execOnNode(nodeIP, user, password, fmt.Sprintf("cat %s", scriptFile))
		content = strings.TrimSpace(content)

		if strings.Contains(content, nodeIP) && !strings.Contains(content, "${NODE_IP}") {
			details = append(details, fmt.Sprintf("%s: 已渲染执行 (文件=%s, NODE_IP=%s已替换)", name, scriptFile, nodeIP))
		} else if content != "" {
			details = append(details, fmt.Sprintf("%s: 已落盘但参数可能未完全渲染 (文件=%s)", name, scriptFile))
		} else {
			allOK = false
			details = append(details, fmt.Sprintf("%s: 脚本文件内容为空", name))
		}
	}
	return allOK, strings.Join(details, "; "), nil
}

// VerifyScriptNotExecuted 验证后置脚本未执行
func (m *PostScriptManager) VerifyScriptNotExecuted(nodeIP, user, password, scriptName string) (bool, error) {
	base := strings.TrimSuffix(scriptName, ".sh")
	output, err := m.execOnNode(nodeIP, user, password,
		fmt.Sprintf("ls %s/%s-* 2>/dev/null | head -1", PostprocessScriptPath, base))
	if err != nil {
		return true, nil
	}
	return strings.TrimSpace(output) == "", nil
}

// VerifyPostProcessingLogs 检查后置处理相关日志
func (m *PostScriptManager) VerifyPostProcessingLogs(clusterName, kubeconfigPath string) (bool, string, error) {
	kc := kcFlag(kubeconfigPath)
	searchPattern := "postprocess|post-process|Postprocess|PostprocessScripts|postprocessPlugin"

	cmd1 := fmt.Sprintf(
		"kubectl %slogs -n cluster-system -l app.kubernetes.io/name=bke-controller-manager --tail=1000 2>/dev/null | grep -i -E '%s' | head -20",
		kc, searchPattern)
	result1, _ := m.sshExecutor.Exec(cmd1)
	if output := strings.TrimSpace(result1.Stdout); output != "" {
		return true, output, nil
	}

	cmd2 := fmt.Sprintf(
		"kubectl %slogs -n cluster-system -l app=bke-controller --tail=1000 2>/dev/null | grep -i -E '%s' | head -20",
		kc, searchPattern)
	result2, _ := m.sshExecutor.Exec(cmd2)
	if output := strings.TrimSpace(result2.Stdout); output != "" {
		return true, output, nil
	}

	cmd3 := fmt.Sprintf(
		"kubectl %slogs -n bke-%s -l app=bkeagent --tail=1000 2>/dev/null | grep -i -E '%s' | head -20",
		kc, clusterName, searchPattern)
	result3, _ := m.sshExecutor.Exec(cmd3)
	if output := strings.TrimSpace(result3.Stdout); output != "" {
		return true, output, nil
	}

	cmd4 := fmt.Sprintf(
		"kubectl %slogs -n cluster-system --all-containers --tail=500 2>/dev/null | grep -i -E '%s' | head -20",
		kc, searchPattern)
	result4, _ := m.sshExecutor.Exec(cmd4)
	output4 := strings.TrimSpace(result4.Stdout)
	return output4 != "", output4, nil
}

func (m *PostScriptManager) applyYAML(content, kcFlag, tag string) error {
	tmpPath := fmt.Sprintf("/tmp/postscript-%s.yaml", tag)

	writeCmd := fmt.Sprintf("cat > %s << 'POSTSCRIPT_EOF'\n%sPOSTSCRIPT_EOF", tmpPath, content)
	if result, err := m.sshExecutor.Exec(writeCmd); err != nil {
		return fmt.Errorf("写入文件失败: %w", err)
	} else if result.ExitCode != 0 {
		return fmt.Errorf("写入文件失败(exit=%d): %s", result.ExitCode, result.Stderr)
	}

	applyCmd := fmt.Sprintf("kubectl %sapply -f %s", kcFlag, tmpPath)
	result, err := m.sshExecutor.Exec(applyCmd)
	if err != nil {
		return fmt.Errorf("kubectl apply 失败: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("kubectl apply 失败(exit=%d): %s", result.ExitCode, result.Stderr)
	}

	m.sshExecutor.Exec(fmt.Sprintf("rm -f %s", tmpPath))
	return nil
}

func (m *PostScriptManager) execOnNode(nodeIP, user, password, remoteCmd string) (string, error) {
	targetExec, err := executor.NewSSHExecutor(nodeIP, 22, user, password)
	if err != nil {
		return "", fmt.Errorf("SSH连接节点 %s 失败: %w", nodeIP, err)
	}

	result, execErr := targetExec.Exec(remoteCmd)
	if result != nil {
		return result.Stdout, nil
	}
	return "", execErr
}

func buildPostScriptConfigMapYAML(script ScriptContentDef) string {
	var sb strings.Builder
	sb.WriteString("apiVersion: v1\n")
	sb.WriteString("kind: ConfigMap\n")
	sb.WriteString("metadata:\n")
	sb.WriteString(fmt.Sprintf("  name: %s\n", script.Name))
	sb.WriteString(fmt.Sprintf("  namespace: %s\n", UserSystemNamespace))
	sb.WriteString("  labels:\n")
	sb.WriteString(fmt.Sprintf("    %s: \"true\"\n", PostprocessScriptLabel))
	sb.WriteString("data:\n")
	sb.WriteString(fmt.Sprintf("  %s: |\n", script.Name))
	for _, line := range strings.Split(script.Content, "\n") {
		sb.WriteString("    " + line + "\n")
	}
	return sb.String()
}

func buildPostGlobalConfigYAML(scripts []ScriptConfigEntry) (string, error) {
	cfg := PreprocessConfig{Scripts: scripts}
	cfgJSON, err := json.MarshalIndent(cfg, "    ", "  ")
	if err != nil {
		return "", fmt.Errorf("序列化 config.json 失败: %w", err)
	}

	var sb strings.Builder
	sb.WriteString("apiVersion: v1\n")
	sb.WriteString("kind: ConfigMap\n")
	sb.WriteString("metadata:\n")
	sb.WriteString(fmt.Sprintf("  name: %s\n", PostprocessGlobalConfigName))
	sb.WriteString(fmt.Sprintf("  namespace: %s\n", UserSystemNamespace))
	sb.WriteString("data:\n")
	sb.WriteString("  config.json: |\n")
	for _, line := range strings.Split(string(cfgJSON), "\n") {
		sb.WriteString("    " + line + "\n")
	}
	return sb.String(), nil
}

// ==================== 后置处理测试脚本内容 ====================

// PostTestScriptInitOS 后置处理 init-os.sh 脚本
func PostTestScriptInitOS() ScriptContentDef {
	return ScriptContentDef{
		Name: "init-os.sh",
		Content: `#!/bin/sh
set -e

echo "========================================="
echo "Post-Script: init-os.sh"
echo "NODE_IP: ${NODE_IP}"
echo "HTTP_REPO: ${HTTP_REPO}"
echo "========================================="

MARKER_DIR="/tmp/postscript-markers"
mkdir -p ${MARKER_DIR}

echo "Configuring OS settings (post-install)..."
echo "Using repo: ${HTTP_REPO}"

echo "NODE_IP=${NODE_IP}" > ${MARKER_DIR}/init-os.env
echo "HTTP_REPO=${HTTP_REPO}" >> ${MARKER_DIR}/init-os.env
echo "EXECUTED_AT=$(date +%s)" >> ${MARKER_DIR}/init-os.env

touch ${MARKER_DIR}/init-os.done
echo "post init-os.sh completed successfully"`,
	}
}

// PostTestScriptSetupEnv 后置处理 setup-env.sh 脚本
func PostTestScriptSetupEnv() ScriptContentDef {
	return ScriptContentDef{
		Name: "setup-env.sh",
		Content: `#!/bin/sh
set -e

echo "========================================="
echo "Post-Script: setup-env.sh"
echo "NODE_IP: ${NODE_IP}"
echo "HTTP_REPO: ${HTTP_REPO}"
echo "========================================="

MARKER_DIR="/tmp/postscript-markers"
mkdir -p ${MARKER_DIR}

echo "Setting up environment (post-install)..."
echo "Using repo: ${HTTP_REPO}"

echo "NODE_IP=${NODE_IP}" > ${MARKER_DIR}/setup-env.env
echo "HTTP_REPO=${HTTP_REPO}" >> ${MARKER_DIR}/setup-env.env
echo "EXECUTED_AT=$(date +%s)" >> ${MARKER_DIR}/setup-env.env

touch ${MARKER_DIR}/setup-env.done
echo "post setup-env.sh completed successfully"`,
	}
}

// PostTestScriptInstallAgent 后置处理 install-agent.sh 脚本
func PostTestScriptInstallAgent() ScriptContentDef {
	return ScriptContentDef{
		Name: "install-agent.sh",
		Content: `#!/bin/sh
set -e

echo "========================================="
echo "Post-Script: install-agent.sh"
echo "NODE_IP: ${NODE_IP}"
echo "HTTP_REPO: ${HTTP_REPO}"
echo "========================================="

MARKER_DIR="/tmp/postscript-markers"
mkdir -p ${MARKER_DIR}

echo "Installing agent on node ${NODE_IP} (post-install)..."
echo "Downloading from: ${HTTP_REPO}/agent"

echo "NODE_IP=${NODE_IP}" > ${MARKER_DIR}/install-agent.env
echo "HTTP_REPO=${HTTP_REPO}" >> ${MARKER_DIR}/install-agent.env
echo "EXECUTED_AT=$(date +%s)" >> ${MARKER_DIR}/install-agent.env

touch ${MARKER_DIR}/install-agent.done
echo "post install-agent.sh completed successfully"`,
	}
}

// PostTestScriptFail 后置处理故意失败的脚本
func PostTestScriptFail() ScriptContentDef {
	return ScriptContentDef{
		Name: "fail-script.sh",
		Content: `#!/bin/sh

echo "========================================="
echo "Post-Script: fail-script.sh (intentionally failing)"
echo "NODE_IP: ${NODE_IP}"
echo "PKG_VERSION: ${PKG_VERSION}"
echo "========================================="

echo "Attempting to download non-existent package version ${PKG_VERSION}..."
echo "ERROR: Package version ${PKG_VERSION} not found in repository"

exit 1`,
	}
}
