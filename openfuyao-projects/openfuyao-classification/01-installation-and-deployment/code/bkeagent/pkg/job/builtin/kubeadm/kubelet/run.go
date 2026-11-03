package kubelet

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"gopkg.openfuyao.cn/bkeagent/utils/clientutil"
	"io"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"
	"text/template"
	"time"

	"github.com/coreos/go-systemd/v22/dbus"
	"github.com/pkg/errors"
	"github.com/shirou/gopsutil/v3/host"
	"gopkg.openfuyao.cn/bkecommon/cluster/imagehelper"
	bkeinit "gopkg.openfuyao.cn/bkecommon/cluster/initialize"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"gopkg.openfuyao.cn/bkeagent/pkg/executor/containerd"
	"gopkg.openfuyao.cn/bkeagent/pkg/executor/docker"
	"gopkg.openfuyao.cn/bkeagent/pkg/executor/exec"
	"gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/bkeagent/utils"
	"gopkg.openfuyao.cn/bkeagent/utils/download"
	"gopkg.openfuyao.cn/bkeagent/utils/httprepo"
	"gopkg.openfuyao.cn/bkeagent/utils/log"
	"gopkg.openfuyao.cn/bkeagent/utils/mfutil"
	"gopkg.openfuyao.cn/bkeagent/utils/pkiutil"
	"gopkg.openfuyao.cn/bkeagent/utils/runtime"

	confv1beta1 "gopkg.openfuyao.cn/bkecommon/cluster/api/v1beta1"
	Runtime "k8s.io/apimachinery/pkg/runtime"
)

var (
	//go:embed tmpl/kubelet.conf.tmpl
	kubeletConf string

	//go:embed tmpl/kubelet.service.tmpl
	kubeletService string

	Name = "RunKubelet"

	defaultContainerName = "kubelet"

	defaultImageRepo = utils.GetDefaultImageRepo()
)

const defaultSleepTime = 3

type kubeletPlugin struct {
	docker     docker.DockerClient
	containerd containerd.ContainerdClient

	exec          exec.Executor
	k8sClient     client.Client
	dynamicClinet dynamic.Interface
	//bkeConfig *bkev1beta1.BKEConfig
}

func New(c client.Client, exec exec.Executor) plugin.Plugin {
	return &kubeletPlugin{
		exec:      exec,
		k8sClient: c,
	}
}

func (kp *kubeletPlugin) Name() string {
	return Name
}

func (kp *kubeletPlugin) Param() map[string]plugin.PluginParam {
	return map[string]plugin.PluginParam{
		"url": {
			Key:         "url",
			Value:       "",
			Required:    true,
			Default:     "",
			Description: "download url",
		},
		"chmod": {
			Key:         "perm",
			Value:       "",
			Required:    false,
			Default:     "0644",
			Description: "file permission",
		},
		"rename": {
			Key:         "rename",
			Value:       "",
			Required:    false,
			Default:     "",
			Description: "rename downloaded file",
		},
		"saveto": {
			Key:         "saveto",
			Value:       "",
			Required:    true,
			Default:     os.TempDir(),
			Description: "save to directory",
		},
		"containerName": {
			Key:         "containerName",
			Value:       "",
			Required:    false,
			Default:     defaultContainerName,
			Description: "kubelet container name",
		},
		"phase": {
			Key:         "phase",
			Value:       "",
			Required:    false,
			Default:     utils.InitControlPlane,
			Description: "run kubelet in which phase",
		},
		"kubeletConfigMap": {
			Key:         "kubeletConfigMap",
			Value:       "",
			Required:    false,
			Default:     "",
			Description: "kubelet configmap ns/name from manager cluster",
		},
		"certificatesDir": {
			Key:         "certificatesDir",
			Value:       "",
			Required:    true,
			Default:     pkiutil.GetDefaultPkiPath(),
			Description: "certificates dir",
		},
		"manifestDir": {
			Key:         "manifestDir",
			Value:       "",
			Required:    false,
			Default:     mfutil.GetDefaultManifestsPath(),
			Description: "manifest dir",
		},
		"imageRepo": {
			Key:         "imageRepo",
			Value:       "",
			Required:    false,
			Default:     defaultImageRepo,
			Description: "image repo",
		},
		"kubernetesVersion": {
			Key:         "kubernetesVersion",
			Value:       "",
			Required:    false,
			Default:     "v1.21.1",
			Description: "kubernetes version",
		},
		"clusterDNSDomain": {
			Key:         "clusterDNSDomain",
			Value:       "",
			Required:    false,
			Default:     bkeinit.DefaultServiceDNSDomain,
			Description: "cluster dns domain",
		},
		"clusterDNSIP": {
			Key:         "clusterDNSIP",
			Value:       "",
			Required:    false,
			Default:     bkeinit.DefaultClusterDNSIP,
			Description: "cluster dns ip",
		},
		"hostIP": {
			Key:         "externalHost",
			Value:       "",
			Required:    false,
			Default:     "",
			Description: "host ip",
		},
		"hostName": {
			Key:         "hostName",
			Value:       "",
			Required:    false,
			Default:     utils.HostName(),
			Description: "host name",
		},
		"extraArgs": {
			Key:         "extraArgs",
			Value:       "",
			Required:    false,
			Default:     "",
			Description: "kubelet extra args,example key=value;key=value,splited by ';'",
		},
		"extraVolumes": {
			Key:         "extraVolumes",
			Value:       "",
			Required:    false,
			Default:     "",
			Description: "kubelet extra volumes,example hostpath:mountpath;hostpath:mountpath;splited by ';' ",
		},
		"generateKubeletConfig": {
			Key:         "generateKubeletConfig",
			Value:       "true",
			Required:    false,
			Default:     "false",
			Description: "generate kubelet config file",
		},
		"kubeconfigPath": {
			Key:         "kubeconfigPath",
			Value:       "",
			Required:    false,
			Default:     pkiutil.GetDefaultKubeConfigPath(),
			Description: "kubeconfig file path",
		},
		"providerID": {
			Key:         "providerID",
			Value:       "",
			Required:    true,
			Default:     "",
			Description: "set providerID to compatible cluster-api",
		},
		"dataRootDir": {
			Key:         "dataRootDir",
			Value:       "",
			Required:    false,
			Default:     bkeinit.DefaultKubeletRootDir,
			Description: "kubelet data root dir",
		},
		"cgroupDriver": {
			Key:         "cgroupDriver",
			Value:       "",
			Required:    false,
			Default:     bkeinit.DefaultCgroupDriver,
			Description: "kubelet cgroup driver",
		},
		// 新增：直接读取 KubeletConfig CR 的参数
		"useDeliveredConfig": {
			Key:         "useDeliveredConfig",                                     //参数名
			Value:       "",                                                       //预留字段
			Required:    false,                                                    //是否必填（true=必须传，false=可缺省）
			Default:     "false",                                                  //缺省值（Required=false 时，没传参数就用这个值）
			Description: "use config from KubeletConfig CR instead of generating", //// 参数描述
		},
		"kubeletConfigName": {
			Key:         "kubeletConfigName",
			Value:       "",
			Required:    false,
			Default:     "",
			Description: "KubeletConfig CR name in manager cluster",
		},
		"kubeletConfigNamespace": {
			Key:         "kubeletConfigNamespace",
			Value:       "",
			Required:    false,
			Default:     "bke-system",
			Description: "KubeletConfig CR namespace in manager cluster",
		},
		"configPath": {
			Key:         "configPath",
			Value:       "",
			Required:    false,
			Default:     utils.GetKubeletConfPath(),
			Description: "local path to store kubelet config",
		},
		"fileBasePath": {
			Key:         "fileBasePath",
			Value:       "",
			Required:    false,
			Default:     "/etc/kubernetes",
			Description: "base path for delivered files",
		},
		"enableVariableSubstitution": {
			Key:      "enableVariableSubstitution",
			Value:    "",
			Required: false,
			Default:  "false",
			Description: "enable variable substitution in kubelet config and service files, " +
				"supports ${VAR_NAME} and ${EXPR|command|END}",
		},
	}
}

// Execute implements the plugin.Plugin interface.
// example:
// ["RunKubelet", "containerName=kubelet"]
func (kp *kubeletPlugin) Execute(commands []string) ([]string, error) {
	config, err := plugin.ParseCommands(kp, commands)
	if err != nil {
		return nil, err
	}
	// 下载并安装 kubelet 二进制文件
	if err := kp.downloadAndInstallKubeletBinary(config); err != nil {
		return nil, err
	}

	// 处理从管理集群直接读取 KubeletConfig CR
	if config["useDeliveredConfig"] == "true" {
		if err := kp.readConfigFromKubeletConfigCR(config); err != nil {
			log.Warnf("failed to read config from kubeletConfig CR:%v, fallback to generate config", err)
			return nil, err
		}
	} else {
		// 原有的kubelet.config配置生成
		if err := kp.generateKubeletConfig(config); err != nil {
			return nil, err
		}
		// 原有的渲染kubelet.service
		if err := kp.renderKubeletService(config); err != nil {
			return nil, err
		}
	}

	if err := kp.ensureImages(config); err != nil {
		return nil, errors.Wrap(err, "failed to ensure images before kubelet start")
	}

	// 启动kubelet,设置为开机自启动并启动
	out, err := kp.exec.ExecuteCommandWithCombinedOutput("sh", "-c",
		"systemctl daemon-reload && systemctl enable kubelet")
	if err != nil {
		log.Warnf("enable kubelet failed, err: %v, out: %s", err, out)
	}

	// Start the kubelet
	out, err = kp.exec.ExecuteCommandWithCombinedOutput("sh", "-c", "systemctl restart kubelet")
	if err != nil {
		errorMsg := fmt.Sprintf("start kubelet failed, err: %v, out: %s", err, out)
		log.Errorf(errorMsg)
		return []string{errorMsg}, fmt.Errorf("start kubelet failed, err: %v, out: %s", err, out)
	}
	// waite for kubelet start
	_, err = isKubeletActive()
	if err != nil {
		return nil, err
	}
	log.Info("kubelet started successfully")
	return nil, nil
}

func (kp *kubeletPlugin) readConfigFromKubeletConfigCR(config map[string]string) error {
	kubeletConfigName := config["kubeletConfigName"]
	kubeletConfigNamespace := config["kubeletConfigNamespace"]
	if kubeletConfigNamespace == "" {
		kubeletConfigNamespace = "bke-system"
	}

	if kubeletConfigName == "" {
		return fmt.Errorf("kubeletConfigName is required when useDeliveredConfig=true")
	}

	// 参照 interface.go 的 GetBKECluster 方法，使用 clientutil.NewKubernetesClient
	c, err := clientutil.NewKubernetesClient(fmt.Sprintf("%s/%s", utils.Workspace, "config"))
	if err != nil {
		return fmt.Errorf("failed to create kubernetes client: %v", err)
	}

	// 定义 KubeletConfig 的 GVR (Group Version Resource)
	gvr := schema.GroupVersionResource{
		Group:    confv1beta1.GVK.Group,
		Version:  confv1beta1.GVK.Version,
		Resource: "kubeletconfigs", // 复数形式
	}

	// 使用 dynamic client 获取 KubeletConfig CR
	unstructuredObj, err := c.DynamicClient.Resource(gvr).Namespace(kubeletConfigNamespace).Get(
		context.Background(), kubeletConfigName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get KubeletConfig %s/%s from manager cluster: %v",
			kubeletConfigNamespace, kubeletConfigName, err)
	}

	// 将 unstructured 对象转换为 KubeletConfig 对象
	kubeletConfig := &confv1beta1.KubeletConfig{}
	if err := Runtime.DefaultUnstructuredConverter.FromUnstructured(unstructuredObj.Object, kubeletConfig); err != nil {
		return fmt.Errorf("failed to convert unstructured object to KubeletConfig: %v", err)
	}

	// 处理 KubeletConfiguration (kubelet.conf)
	if err := kp.processKubeletConfiguration(kubeletConfig.Spec.KubeletConfig, config); err != nil {
		return fmt.Errorf("failed to process kubelet configuration: %v", err)
	}
	log.Infof("process kubelet configuration successfully")
	// 处理 KubeletService (kubelet.service)
	if kubeletConfig.Spec.KubeletService != nil {
		log.Infof("kubelet.service:%v", kubeletConfig.Spec.KubeletService.Service)
		if err := kp.processKubeletService(&kubeletConfig.Spec.KubeletService.Service, config); err != nil {
			return fmt.Errorf("failed to process kubelet service: %v", err)
		}
	}

	return nil
}

func (kp *kubeletPlugin) processKubeletService(kubeletService *confv1beta1.KubeletService,
	config map[string]string) error {
	//处理 kubelet.service
	if kubeletService != nil {
		if err := generateService(kubeletService, config, kp.exec); err != nil {
			return fmt.Errorf("generate kubelet service failed: %v", err)
		}
		return nil
	}
	return fmt.Errorf("kubelet.service is nil")
}

func generateService(service *confv1beta1.KubeletService, config map[string]string, exec exec.Executor) error {
	generate := NewServiceData("")
	if generate == nil {
		return fmt.Errorf("new service in cr failed")
	}
	return generate.GenerateService(service, config, exec)
}

type VariableSubstitutor struct {
	config map[string]string
	exec   exec.Executor
}

func (vs *VariableSubstitutor) Substitute(content string) (string, error) {
	// Step 1: 处理转义（可选）
	content = strings.ReplaceAll(content, "\\$\\{", "${ESCAPED}")
	content = strings.ReplaceAll(content, "\\}", "ESCAPED_BRACE")

	// Step 2: 定义 EXPR 正则：匹配 ${EXPR|...|END} 或 ${expr|...|END}
	reExpr := regexp.MustCompile(`\$\{(?i:expr)\|([\s\S]*?)\|END\}`)

	content = reExpr.ReplaceAllStringFunc(content, func(match string) string {
		// 提取命令部分（第1组）
		command := reExpr.FindStringSubmatch(match)[1]

		// 还原转义（如果需要）
		command = strings.ReplaceAll(command, "ESCAPED_BRACE", "}")
		command = strings.ReplaceAll(command, "${ESCAPED}", "${")

		command = strings.TrimSpace(command)
		if command == "" {
			log.Warnf("empty EXPR command in: %s", match)
			return match
		}

		log.Debugf("Executing EXPR command: %s", command)
		output, err := vs.exec.ExecuteCommandWithOutput("sh", "-c", command)
		if err != nil {
			log.Warnf("EXPR command failed '%s': %v (output: %s)", command, err, output)
			return match // 保留原样
		}

		result := strings.TrimSpace(output)
		log.Debugf("EXPR result: %s", result)
		return result
	})

	// Step 3: 处理普通变量 ${VAR}（不变）
	reNormalVar := regexp.MustCompile(`\$\{([\w]+)\}`)
	content = reNormalVar.ReplaceAllStringFunc(content, func(match string) string {
		varName := strings.Trim(match, "${}")
		if val, ok := vs.config[varName]; ok {
			return val
		}
		if val := os.Getenv(varName); val != "" {
			return val
		}
		log.Warnf("variable not found: %s", varName)
		return match
	})

	// Step 4: 还原转义占位符（如果之前用了）
	content = strings.ReplaceAll(content, "${ESCAPED}", "${")
	return content, nil
}

func (kp *kubeletPlugin) processKubeletConfiguration(kubeletConfiguration map[string]Runtime.RawExtension, config map[string]string) error {
	if kubeletConf, exists := kubeletConfiguration["kubelet.conf"]; exists {
		// 步骤1：解析 kubeletConf.Raw（它是包含 raw 字段的 JSON）
		type RawConfig struct {
			Raw string `json:"raw"` // 匹配 JSON 中的 "raw" 字段
		}
		var rawConfig RawConfig
		if err := json.Unmarshal(kubeletConf.Raw, &rawConfig); err != nil {
			return fmt.Errorf("failed to unmarshal raw config: %v", err)
		}
		// 此时 rawConfig.Raw 就是纯 YAML 字符串（如 "apiVersion: ... kind: ..."）

		// 步骤2：变量替换（复用原有逻辑，对纯 YAML 内容替换）
		content := rawConfig.Raw
		if config["enableVariableSubstitution"] == "true" {
			substitutor := &VariableSubstitutor{
				config: config,
				exec:   kp.exec,
			}
			substitutedContent, err := substitutor.Substitute(content)
			if err != nil {
				return fmt.Errorf("failed to substitute variables: %v", err)
			}
			content = substitutedContent
		}

		// 步骤3：写入纯 YAML 内容到 config.yaml（无 JSON 包装）
		localPath := config["configPath"]
		if localPath == "" {
			localPath = utils.GetKubeletConfPath()
		}
		// 手动打开文件，确保写入后刷新到磁盘
		file, err := os.OpenFile(localPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
		if err != nil {
			return fmt.Errorf("failed to open config file: %v", err)
		}
		if _, err := file.WriteString(content); err != nil {
			return fmt.Errorf("failed to write config content: %v", err)
		}
		if err := file.Sync(); err != nil { // 强制刷盘
			return fmt.Errorf("failed to sync config to disk: %v", err)
		}
		err = file.Close()
		if err != nil {
			return err
		} // 确保关闭

		// 步骤4：检查并追加 providerID
		providerID, providerIDExists := config["providerID"]
		if !providerIDExists || strings.TrimSpace(providerID) == "" {
			log.Warnf("providerID is empty, skip appending to %s", localPath)
		} else {
			// 传入明确的 localPath，避免路径不一致
			if err := kp.appendProviderIDToConfYaml(config); err != nil {
				log.Errorf("failed to append providerID to %s: %v", localPath, err)
				return err // 按需决定是否返回错误（若为必填项则返回，否则 Warn）
			}
		}
		log.Infof("successfully wrote pure YAML config to %s", localPath)
	}
	return nil
}

func (kp *kubeletPlugin) handlerKubeletServiceParam(config map[string]string) map[string]string {
	param := make(map[string]string)
	// 空格用于分割参数
	param["kubeletConfig"] = fmt.Sprintf("%s  ", utils.GetKubeletConfPath())
	param["hostIP"] = config["hostIP"]
	param["hostName"] = utils.HostName()
	// 后续从command中传递
	param["podInfraContainerImage"] = fmt.Sprintf("%s/kubernetes/pause:%s",
		strings.TrimRight(config["imageRepo"], "/"), bkeinit.DefaultPauseImageTag)

	extraArgs := strings.Split(config["extraArgs"], ";")
	extra := ""
	for _, arg := range extraArgs {
		extra += arg + " "
	}
	param["extraArgs"] = extra

	return param
}

func (kp *kubeletPlugin) renderKubeletService(config map[string]string) error {
	if config == nil {
		config = make(map[string]string)
	}

	param := kp.handlerKubeletServiceParam(config)
	switch runtime.DetectRuntime() {
	case runtime.ContainerRuntimeContainerd:
		param["runtimeEndpoint"] = "unix:///run/containerd/containerd.sock"
	case runtime.ContainerRuntimeDocker:
		param["runtimeEndpoint"] = "unix:///var/run/cri-dockerd.sock"
		if err := kp.generateKubeletConfigByHostOS(config); err != nil {
			return err
		}
	default:
		return errors.New("unknown container runtime type when render kubelet service")
	}

	// 渲染kubelet.service文件参数
	t, err := template.New("kubeletService").Parse(kubeletService)
	if err != nil {
		return err
	}
	if !utils.Exists(utils.SystemdDir) {
		if err := os.MkdirAll(utils.SystemdDir, utils.RwxRxRx); err != nil {
			return errors.Errorf("create %q directory failed: %v", utils.SystemdDir, err)
		}
	}
	writer, err := os.OpenFile(utils.GetKubeletServicePath(), os.O_WRONLY|os.O_CREATE|os.O_TRUNC, utils.RwxRxRx)
	defer writer.Close()
	if err != nil {
		return errors.Errorf("open kubelet service file failed: %v", err)
	}
	if err := t.Execute(writer, param); err != nil {
		return errors.Errorf("execute kubelet service template failed: %v", err)
	}

	return nil
}

func (kp *kubeletPlugin) generateKubeletConfigByHostOS(config map[string]string) error {
	if config == nil {
		config = make(map[string]string)
	}
	// 如果是麒麟需要重新生成kubelet config，cgroupDriver需要设置为cgroupfs
	hostOS, _, _, err := host.PlatformInformation()
	if err != nil {
		log.Errorf("get host platform info failed, err: %v", err)
		return errors.Errorf("get host platform info failed, err: %v", err)
	}
	if hostOS == "kylin" {
		if err := httprepo.RepoSearch("docker-ce"); err != nil {
			log.Warnf("[kylin] search docker-ce from repo failed, use docker-engine will set cgroupDriver"+
				" to cgroupfs, err: %v", err)
			config["cgroupDriver"] = "cgroupfs"
			if err = kp.generateKubeletConfig(config); err != nil {
				return err
			}
		}
	}
	return nil
}

// 准备 YAML 格式的 providerID 行
func getProviderIDLine(commandMap map[string]string) (string, error) {
	providerID, exists := commandMap["providerID"]
	if !exists || strings.TrimSpace(providerID) == "" {
		return "", fmt.Errorf("commandMap['providerID'] is empty or missing")
	}
	return fmt.Sprintf("providerID: %s\n", providerID), nil // YAML 需冒号后加空格
}

// 确保配置文件所在目录存在，不存在则创建
func ensureDirExists(filePath string) error {
	dir := filepath.Dir(filePath)
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		if err := os.MkdirAll(dir, 0755); err != nil { // 0755：目录常规权限
			return fmt.Errorf("create dir %s: %w", dir, err)
		}
		log.Infof("created target dir: %s", dir)
	}
	return nil
}

// 打开配置文件（追加模式，不存在则创建）
func openConfFile(filePath string) (*os.File, error) {
	file, err := os.OpenFile(
		filePath,
		os.O_APPEND|os.O_RDWR|os.O_CREATE, // 追加+写+创建
		0644,                              // 0644：配置文件常规权限（所有者读写，其他只读）
	)
	if err != nil {
		return nil, fmt.Errorf("open file %s: %w", filePath, err)
	}
	return file, nil
}

// 确保文件末尾以换行符结尾（避免内容追加到同一行）
func ensureFileEndsWithNewline(file *os.File) error {
	fileInfo, err := file.Stat()
	if err != nil {
		return fmt.Errorf("get file info: %w", err)
	}
	if fileInfo.Size() == 0 { // 空文件无需处理换行
		return nil
	}

	// 移动指针到文件最后1字节，检查是否为换行符
	if _, err := file.Seek(-1, io.SeekEnd); err != nil {
		return fmt.Errorf("seek file end: %w", err)
	}

	buf := make([]byte, 1)
	if _, err := file.Read(buf); err != nil {
		return fmt.Errorf("read file end: %w", err)
	}

	// 无换行符则补充
	if buf[0] != '\n' {
		if _, err := file.WriteString("\n"); err != nil {
			return fmt.Errorf("write newline: %w", err)
		}
	}
	return nil
}

// 将 providerID 追加到 conf.yaml 最后一行
func (kp *kubeletPlugin) appendProviderIDToConfYaml(commandMap map[string]string) error {
	// 步骤1：准备写入内容
	line, err := getProviderIDLine(commandMap)
	localPath := commandMap["configPath"]
	if localPath == "" {
		localPath = utils.GetKubeletConfPath()
	}
	if err != nil {
		log.Errorf("prepare line failed: %v", err)
		return err
	}

	// 步骤2：确保目录存在
	if err := ensureDirExists(localPath); err != nil {
		log.Errorf("ensure dir failed: %v", err)
		return err
	}

	// 步骤3：打开文件（defer 确保关闭）
	file, err := openConfFile(localPath)
	if err != nil {
		log.Errorf("open file failed: %v", err)
		return err
	}
	defer file.Close()

	// 步骤4：处理文件末尾换行
	if err := ensureFileEndsWithNewline(file); err != nil {
		log.Errorf("handle newline failed: %v", err)
		return err
	}

	// 步骤5：追加写入
	if _, err := file.WriteString(line); err != nil {
		log.Errorf("append line failed: %v", err)
		return fmt.Errorf("write line to %s: %w", localPath, err)
	}

	log.Infof("success append providerID to %s: %s", localPath, strings.TrimSpace(line))
	return nil
}

func (kp *kubeletPlugin) downloadAndInstallKubeletBinary(commandMap map[string]string) error {
	url := commandMap["url"]
	rename := commandMap["rename"]
	saveto := commandMap["saveto"]
	chmod := commandMap["chmod"]

	// 防止提前安装kubelet，再次安装时导致不能正常下载kubelet
	out, err := kp.exec.ExecuteCommandWithCombinedOutput("sh", "-c", "systemctl stop kubelet")
	if err != nil {
		errorMsg := fmt.Sprintf("stop kubelet failed, err: %v, out: %s", err, out)
		log.Warnf(errorMsg)
	}
	cmd := fmt.Sprintf("rm -rf %s", filepath.Join(saveto, rename))
	out, err = kp.exec.ExecuteCommandWithCombinedOutput("sh", "-c", cmd)
	if err != nil {
		errorMsg := fmt.Sprintf("rm kubelet failed, err: %v, out: %s", err, out)
		log.Warnf(errorMsg)
	}

	return download.ExecDownload(url, saveto, rename, chmod)
}

func (kp *kubeletPlugin) joinControlPlanePrepare(config map[string]string) error {
	return kp.joinWorkerPrepare(config)
}

// joinWorkerPrepare get kubelet config from manager cluster
func (kp *kubeletPlugin) joinWorkerPrepare(config map[string]string) error {
	if config["generateKubeletConfig"] == "true" {
		return nil
	}

	if kp.k8sClient == nil {
		return errors.New("no manager kubernetes cluster client")
	}
	namespace, name, err := utils.SplitNameSpaceName(config["kubeletConfigMap"])
	if err != nil {
		return err
	}
	kubeletConfigMap := &corev1.ConfigMap{}
	if err := kp.k8sClient.Get(context.Background(), client.ObjectKey{Namespace: namespace, Name: name}, kubeletConfigMap); err != nil {
		if apierrors.IsNotFound(err) {
			if err := kp.generateKubeletConfig(config); err != nil {
				return err
			}
			return nil
		}
		return errors.Wrapf(err, "failed to get kubelet configmap from manager cluster, and generate kubelet config is not enabled")
	}

	return storeKubeletConfig(kubeletConfigMap)
}

func (kp *kubeletPlugin) generateKubeletConfig(config map[string]string) error {
	if !utils.Exists(utils.KubeletConfigPath) {
		if err := os.MkdirAll(utils.KubeletConfigPath, utils.RwxRxRx); err != nil {
			return errors.Errorf("create %q directory failed: %v", utils.KubeletConfigPath, err)
		}
	}

	t, err := template.New("kubeletConfig").Parse(kubeletConf)
	if err != nil {
		return err
	}
	kubeletConfPath := utils.GetKubeletConfPath()
	if err := os.MkdirAll(path.Dir(kubeletConfPath), utils.RwRR); err != nil {
		return err
	}
	writer, err := os.OpenFile(kubeletConfPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, utils.RwRR)
	defer writer.Close()
	if err != nil {
		return err
	}
	if err := t.Execute(writer, config); err != nil {
		return err
	}
	log.Infof("generate kubelet config file in %s", kubeletConfPath)
	return nil
}

func storeKubeletConfig(configMap *corev1.ConfigMap) error {
	if data, ok := configMap.Data["kubelet"]; ok {
		if err := os.MkdirAll(utils.KubeletConfigPath, utils.RwRR); err != nil {
			return err
		}
		configFile := filepath.Join(utils.KubeletConfigPath, utils.KubeletConfigFileName)
		if err := os.WriteFile(configFile, []byte(data), utils.RwRR); err != nil {
			return err
		}
		return nil
	}
	return errors.Errorf("kubelet config data not found at configmap %s/%s", configMap.Namespace, configMap.Name)
}

func mountList() [][]string {
	mountList := [][]string{
		{"/etc/kubernetes", "/etc/kubernetes", ""},
		{"/etc/kubernetes/manifests", "/etc/kubernetes/manifests", ""},
		{"/etc/localtime", "/etc/localtime", ""},
		{"/etc/ssl/certs", "/etc/ssl/certs", ""},
		{"/etc/sysconfig/network-scripts", "/etc/sysconfig/network-scripts", ""},
		{"/etc/resolv.conf", "/etc/resolv.conf", ""},
		{"/etc/cni", "/etc/cni", ""},

		{"/var/lib/kubelet", "/var/lib/kubelet", ""},
		{"/var/lib/etcd", "/var/lib/etcd", ""},
		{"/var/lib/cni", "/var/lib/cni", ""},
		{"/var/lib/docker", "/var/lib/docker", "rw,rslave"},
		{"/var/lib/calico", "/var/lib/calico", ""},
		{"/var/lib/lxc", "/var/lib/lxc", ""},
		{"/var/lib/kubelet", "/var/lib/kubelet", "shared"},
		{"/var/log/pods", "/var/log/pods", ""},
		{"/var/log/containers", "/var/log/containers", ""},
		{"/var/run", "/var/run", ""},

		{"/opt/cni", "/opt/cni", ""},
		{"/opt/fabric", "/opt/fabric", ""},

		{"/usr/libexec/kubernetes", "/usr/libexec/kubernetes", ""},
		{"/run", "/run", ""},
		{"/sys", "/sys", ""},
		{"/proc", "/proc", ""},
		{"/dev", "/dev", ""},
		{"/lib/modules", "/lib/modules", ""},
	}
	for _, dir := range utils.RunKubeletPreCreateDirs {
		exit := utils.Exists(dir)
		if !exit {
			log.Infof("%s not exist, create it", dir)
			if err := os.MkdirAll(dir, utils.RwxRxRx); err != nil {
				return nil
			}
		}
	}
	return mountList
}

// ensureImages ensure images for kubelet and k8s components
func (kp *kubeletPlugin) ensureImages(config map[string]string) error {
	repo := config["imageRepo"]
	clusterVersion := config["kubernetesVersion"]
	// 镜像路径中增加kubernetes前缀，因为kube-apiserver等 k8s components 静态pod yaml中也使用了kubernetes前缀
	repo = fmt.Sprintf("%s/", strings.TrimRight(repo, "/"))

	exporter := imagehelper.NewImageExporter(repo, clusterVersion, "")
	imageMap, err := exporter.ExportImageMapWithBootStrapPhase(config["phase"])
	if err != nil {
		return err
	}

	switch runtime.DetectRuntime() {
	case runtime.ContainerRuntimeContainerd:
		containerdClient, err := containerd.NewContainedClient()
		if err != nil {
			return errors.Wrap(err, "failed to create containerd client")
		}
		kp.containerd = containerdClient
	case runtime.ContainerRuntimeDocker:
		dockerClient, err := docker.NewDockerClient()
		if err != nil {
			return errors.Wrap(err, "failed to create docker client")
		}
		kp.docker = dockerClient
	default:
		return errors.New("unknown container runtime type")
	}

	for _, image := range imageMap {
		if kp.containerd != nil {
			if err := kp.containerd.EnsureImageExists(containerd.ImageRef{Image: image}); err != nil {
				return errors.Wrapf(err, "failed to ensure image %q exists", image)
			}
		}
		if kp.docker != nil {
			if err := kp.docker.EnsureImageExists(docker.ImageRef{Image: image}); err != nil {
				return errors.Wrapf(err, "failed to ensure image %q exists", image)
			}
		}
	}
	config["pauseImage"] = imageMap[bkeinit.DefaultPauseImageName]
	config["kubeletImage"] = imageMap[bkeinit.DefaultKubeletImageName]

	return nil
}

// extraVolumes 为 kubelet 添加额外的挂载，证书目录、资源目录
// master 节点需要挂载 静态 pod 资源目录，worker 节点只需要挂载证书目录
func (kp *kubeletPlugin) extraVolumes(config map[string]string) string {
	// extra volumes
	extraVolumes := config["extraVolumes"]
	volumes := strings.Split(extraVolumes, ";")

	if strings.Contains(config["phase"], "ControlPlane") {
		volume := fmt.Sprintf("%s:%s", config["manifestDir"], mfutil.GetDefaultManifestsPath())
		volumes = append(volumes, volume)
	}
	volume := fmt.Sprintf("%s:%s", config["certificatesDir"], pkiutil.GetDefaultPkiPath())
	volumes = append(volumes, volume)

	volumes = utils.UniqueStringSlice(volumes)

	return strings.Join(volumes, ";")
}

func newKubeletScript(config map[string]string) error {
	return NewRunKubeletCommand().
		SetContainerRuntime("containerd").
		SetKubeletImage(config["kubeletImage"]).
		SetPauseImage(config["pauseImage"]).
		SetKubeConfigPath(config["kubeconfigPath"]).
		SetExtraVolumes(strings.Split(config["extraVolumes"], ";")).
		SetExtraKubeletArgs(strings.Split(config["extraArgs"], ";")).
		SetK8sVersion(config["kubernetesVersion"]).
		SetHostIP(config["hostIP"]).
		SetRootDirPath(config["dataRootDir"]).
		ExportKubeletScript(config["phase"] == utils.UpgradeControlPlane || config["phase"] == utils.UpgradeWorker)
}

func isKubeletActive() (bool, error) {
	time.Sleep(defaultSleepTime * time.Second)
	conn, err := dbus.NewSystemConnectionContext(context.Background())
	if err != nil {
		log.Warnf("failed to connect to systemd: %v", err)
		return false, fmt.Errorf("failed to connect to systemd: %v", err)
	}
	defer conn.Close()

	// 查询 kubelet 服务的状态
	prop, err := conn.GetUnitPropertyContext(context.Background(), utils.KubeletServiceUnitName, "ActiveState")
	if err != nil {
		log.Warnf("failed to get unit property: %v", err)
		return false, fmt.Errorf("failed to get unit property: %v", err)
	}
	log.Infof("get kubelet service ActiveState by dbus: %v", prop.Value.Value())

	// 检查是否为 "active"
	return prop.Value.Value() == "active", nil
}
