/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package main

import (
	_ "embed"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"text/template"

	"k8s.io/client-go/tools/clientcmd"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	launcherDir = "/etc/openFuyao/bkeagent/launcher"
	workDir     = "/etc/openFuyao/bkeagent"
)

var (
	localBKEAgentBinarySrc = "./bkeagent"

	bkeagentBinarySrc  = filepath.Join(launcherDir, "bkeagent")
	bkeagentServiceSrc = filepath.Join(launcherDir, "bkeagent.service")
	kubeconfigSrc      = filepath.Join(launcherDir, "config")
	nodeSrc            = filepath.Join(launcherDir, "node")

	bkeagentBinaryDst  = "/usr/local/bin/bkeagent"
	bkeagentServiceDst = "/etc/systemd/system/bkeagent.service"
	kubeconfigDst      = filepath.Join(workDir, "config")
	nodeDst            = filepath.Join(workDir, "node")

	//go:embed bkeagent.service.tmpl
	bkeagentServiceTemplate string
)

var ntpServer string
var healthPort string
var logLevel string
var kubeconfig string
var debug string

var (
	gitCommitId    = "dev"
	architecture   = "unknown"
	buildTime      = "unknown"
	version        = "latest"
	agentVersion   = "unknown"
	agentCommitId  = "unknown"
	agentArch      = "unknown"
	agentBuildTime = "unknown"
)

func initFlag() {
	log.Info("--------------Starting the BKEAgent launcher---------------")
	log.Info(fmt.Sprintf("🤯 Version: %s", version))
	log.Info(fmt.Sprintf("🤔 GitCommitId: %s ", gitCommitId))
	log.Info(fmt.Sprintf("👉 Architecture: %s", architecture))
	log.Info(fmt.Sprintf("⏲ BuildTime: %s", buildTime))
	log.Info("--------------BKEAgent Info--------------------------------")
	log.Info(fmt.Sprintf("🤯 Agent Version: %s", agentVersion))
	log.Info(fmt.Sprintf("🤔 Agent GitCommitId: %s ", agentCommitId))
	log.Info(fmt.Sprintf("👉 Agent Architecture: %s", agentArch))
	log.Info(fmt.Sprintf("⏲ Agent BuildTime: %s", agentBuildTime))
	log.Info("-----------------------------------------------------------")

	flag.StringVar(&ntpServer, "ntp-server", "", "ntp server")
	flag.StringVar(&healthPort, "health-port", "", "bkeagent health port")
	flag.StringVar(&logLevel, "log-level", "", "bkeagent business log level (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL); empty keeps agent default INFO")
	flag.StringVar(&kubeconfig, "kubeconfig", "", "kubeconfig file path")
	flag.StringVar(&debug, "debug", "true", "debug mode")
	flag.Parse()
}

func main() {
	initFlag()
	validateFlag()

	// launcher 被设计来从容器中启动 bkeagent
	// 通过检测 设置环境变量 container = true 来判断是否在容器中运行
	if os.Getenv("container") != "true" {
		log.Errorf("launcher must run in container")
		os.Exit(1)
	}

	if err := startPre(); err != nil {
		log.Errorf("start pre error: %v", err)
		os.Exit(1)
	}
	if err := start(); err != nil {
		log.Errorf("start error: %v", err)
		os.Exit(1)
	}

	if err := startPost(); err != nil {
		log.Errorf("start post error: %v", err)
		os.Exit(1)
	}
}
func validateFlag() {
	if ntpServer == "" {
		log.Warnf("ntp-server is empty")
	} else {
		log.Debugf("ntp-server is: \n%s", ntpServer)
	}

	if healthPort == "" {
		log.Warnf("health-port is empty")
		os.Exit(1)
	} else {
		log.Debugf("health-port is: \n%s", healthPort)
	}

	if kubeconfig == "" {
		log.Errorf("kubeconfig is empty")
		os.Exit(1)
	}
	log.Debugf("kubeconfig is: \n%s", kubeconfig)

	logLevel = strings.TrimSpace(logLevel)
	if logLevel == "" {
		log.Debugf("log-level is empty, bkeagent will use default INFO")
	} else if canonical, ok := log.NormalizeAllowedLevel(logLevel); ok {
		logLevel = canonical
		log.Debugf("log-level is: %s", logLevel)
	} else {
		log.Warnf("invalid log-level %q, ignore and keep bkeagent default INFO", logLevel)
		logLevel = ""
	}
}

// getHostname 获取当前主机名
func getHostname() (string, error) {
	nodeName, err := executeCommand("hostname")
	if err != nil {
		log.Errorf("get hostname error: %v", err)
		os.Exit(1)
	}
	log.Infof("hostname is: %s", nodeName)
	return nodeName, nil
}

// prepareBkeagentBinary 复制 bkeagent 二进制到本地和主机
func prepareBkeagentBinary() error {
	// local copy bkeagent binary
	out, err := exec.Command("cp", localBKEAgentBinarySrc, launcherDir).CombinedOutput()
	if err != nil {
		log.Errorf("copy bkeagent binary error: %v, out: %s", err, out)
		return err
	}
	// 不可变 OS（如 KubeOS）的 /usr/local/bin 为只读，bkeagent 已内置在镜像中，跳过复制
	if isHostPathWritable(bkeagentBinaryDst) {
		return copyFile(bkeagentBinarySrc, bkeagentBinaryDst)
	}
	log.Infof("host path %s is read-only, skip copying bkeagent binary (prebuilt in image)", bkeagentBinaryDst)
	return nil
}

// isHostPathWritable 检测宿主机上指定路径是否可写
func isHostPathWritable(path string) bool {
	testFile := fmt.Sprintf("%s/.bkeagent-writable-test", filepath.Dir(path))
	_, err := executeCommand(fmt.Sprintf("touch %s && rm -f %s", testFile, testFile))
	return err == nil
}

// prepareBkeagentService 渲染并复制 bkeagent.service 文件
func prepareBkeagentService() error {
	const filePermission = 0666
	f, err := os.OpenFile(bkeagentServiceSrc, os.O_RDWR|os.O_CREATE|os.O_TRUNC, filePermission)
	if err != nil {
		log.Warnf("open bkeagent.service file error: %v", err)
		return err
	}
	defer f.Close()

	tmpl, err := template.New("bkeagent.service").Parse(bkeagentServiceTemplate)
	if err != nil {
		return err
	}

	cfg := map[string]string{
		"debug":      debug,
		"ntpServer":  ntpServer,
		"healthPort": healthPort,
		"logLevel":   logLevel,
	}
	if err := tmpl.Execute(f, cfg); err != nil {
		return err
	}

	// host copy bkeagent service
	return copyFile(bkeagentServiceSrc, bkeagentServiceDst)
}

// prepareKubeconfig 验证、保存并复制 kubeconfig 文件
func prepareKubeconfig() error {
	config, err := clientcmd.LoadFromFile(kubeconfig)
	if err != nil {
		return err
	}
	// 放到 /etc/openFuyao/bkeagent/launcher/config
	if err := clientcmd.WriteToFile(*config, kubeconfigSrc); err != nil {
		return err
	}
	if err := os.Chmod(kubeconfigSrc, utils.KubeConfigFilePerm); err != nil {
		return err
	}
	// host copy kubeconfig
	return copyFile(kubeconfigSrc, kubeconfigDst)
}

// prepareNodeFile 写入并复制 node 文件
func prepareNodeFile(nodeName string) error {
	if err := writeFile(nodeSrc, nodeName); err != nil {
		return err
	}
	// host copy node
	return copyFile(nodeSrc, nodeDst)
}

func trimNodeName(content string) string {
	return strings.TrimSpace(content)
}

func readNodeNameFromHost(hostPath string) (string, error) {
	out, err := executeCommand(fmt.Sprintf("cat %s", hostPath))
	if err != nil {
		return "", err
	}
	return trimNodeName(out), nil
}

func resolveNodeName() (string, error) {
	nodeName, err := readNodeNameFromHost(nodeDst)
	if err != nil {
		return "", fmt.Errorf("read host node file %s: %w", nodeDst, err)
	}
	if nodeName != "" {
		log.Infof("use node name from host %s: %s", nodeDst, nodeName)
		return nodeName, nil
	}

	return "", fmt.Errorf("node name is empty, please set %s on the host before launcher starts", nodeDst)
}

// startPre 启动前准备工作
func startPre() error {
	_, err := executeCommand("systemctl stop bkeagent")
	if err != nil {
		return err
	}

	nodeName, err := resolveNodeName()
	if err != nil {
		return err
	}

	if err := prepareBkeagentBinary(); err != nil {
		return err
	}
	if err := prepareBkeagentService(); err != nil {
		return err
	}
	if err := prepareKubeconfig(); err != nil {
		return err
	}
	return prepareNodeFile(nodeName)
}

func startPost() error {
	// start http server
	http.HandleFunc("/readyz", func(w http.ResponseWriter, r *http.Request) {
		err := pingBKEAgent()
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			_, err = w.Write([]byte(err.Error()))
			if err != nil {
				log.Errorf("failed to write HTTP reply: %v", err)
				return
			}
		} else {
			w.WriteHeader(http.StatusOK)
			_, err = w.Write([]byte("ok"))
			if err != nil {
				log.Errorf("failed to write HTTP reply: %v", err)
				return
			}
		}
	})

	if err := http.ListenAndServe(":3377", nil); err != nil {
		log.Errorf("start http server error: %v", err)
		return err
	}
	return nil
}

func start() error {
	_, err := executeCommand("systemctl daemon-reload")
	if err != nil {
		return err
	}

	_, err = executeCommand("systemctl start bkeagent")
	if err != nil {
		return err
	}
	_, err = executeCommand("systemctl enable bkeagent")
	if err != nil {
		return err
	}
	return nil
}

func pingBKEAgent() error {
	cmdStr := fmt.Sprintf("systemctl is-active bkeagent")
	out, err := executeCommand(cmdStr)
	if err != nil {
		return err
	}
	if out != "active" {
		return fmt.Errorf("bkeagent is not ready")
	}
	return nil
}

func copyFile(src, dst string) error {
	cmdStr := fmt.Sprintf("cp %s %s", src, dst)
	_, err := executeCommand(cmdStr)
	return err
}

func writeFile(dst, content string) error {
	return os.WriteFile(dst, []byte(content), utils.RwRR)
}

func executeCommand(cmdStr string) (string, error) {
	finalCmd := fmt.Sprintf("nsenter -t 1 -m -u -i -n -p sh -c '%s'", cmdStr)
	cmd := exec.Command("sh", "-c", finalCmd)
	var response string
	output, err := cmd.CombinedOutput()
	response = string(output)
	response = strings.TrimSuffix(response, "\n")
	if err != nil {
		log.Errorf("run cmd %q error -> \n response：%v \n err-info: %v", cmdStr, response, err)
	}
	log.Infof("run cmd %q success -> \n response：%v", cmdStr, response)
	return response, err
}
