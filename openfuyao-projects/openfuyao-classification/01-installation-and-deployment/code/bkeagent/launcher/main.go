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

	"gopkg.openfuyao.cn/bkeagent/utils/log"
	"k8s.io/client-go/tools/clientcmd"
)

const (
	launcherDir = "/etc/bkeagent/launcher"
	workDir     = "/etc/bkeagent"
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

func init() {
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
	flag.StringVar(&kubeconfig, "kubeconfig", "", "kubeconfig file path")
	flag.StringVar(&debug, "debug", "true", "debug mode")
	flag.Parse()
}

func main() {

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

	if kubeconfig == "" {
		log.Errorf("kubeconfig is empty")
		os.Exit(1)
	}
	log.Debugf("kubeconfig is: \n%s", kubeconfig)
}

func startPre() error {
	_, _ = executeCommand("systemctl stop bkeagent")

	// get host name
	nodeName, err := executeCommand("hostname")
	if err != nil {
		log.Errorf("get hostname error: %v", err)
		os.Exit(1)
	}
	log.Infof("hostname is: %s", nodeName)

	// local copy bkeagent binary
	out, err := exec.Command("cp", localBKEAgentBinarySrc, launcherDir).CombinedOutput()
	if err != nil {
		log.Errorf("copy bkeagent binary error: %v, out: %s", err, out)
		return err
	}

	// host copy bkeagent binary
	if err := copyFile(bkeagentBinarySrc, bkeagentBinaryDst); err != nil {
		return err
	}

	f, err := os.Create(bkeagentServiceSrc)
	if err != nil {
		log.Warnf("open bkeagent.service file error: %v", err)
		return err
	}
	defer f.Close()
	parse, err := template.New("bkeagent.service").Parse(bkeagentServiceTemplate)
	if err != nil {
		return err
	}
	cfg := map[string]string{
		"debug":     debug,
		"ntpServer": ntpServer,
	}
	if err := parse.Execute(f, cfg); err != nil {
		return err
	}

	// host copy bkeagent service
	if err := copyFile(bkeagentServiceSrc, bkeagentServiceDst); err != nil {
		return err
	}

	// validate kubeconfig
	config, err := clientcmd.LoadFromFile(kubeconfig)
	if err != nil {
		return err
	}
	// 放到 /etc/bkeagent/launcher/config
	if err := clientcmd.WriteToFile(*config, kubeconfigSrc); err != nil {
		return err
	}

	// host copy kubeconfig
	if err := copyFile(kubeconfigSrc, kubeconfigDst); err != nil {
		return err
	}

	if err := writeFile(nodeSrc, nodeName); err != nil {
		return err
	}

	// host copy node
	if err := copyFile(nodeSrc, nodeDst); err != nil {
		return err
	}

	return nil
}

func startPost() error {
	// start http server
	http.HandleFunc("/readyz", func(w http.ResponseWriter, r *http.Request) {
		err := pingBKEAgent()
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			w.Write([]byte(err.Error()))
		} else {
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("ok"))
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
	_, _ = executeCommand("systemctl enable bkeagent")
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
	return os.WriteFile(dst, []byte(content), 0644)
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
