package utils

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// E2EProgressLog 同时写入 GinkgoWriter 与 stderr；go test 未加 -v 时 GinkgoWriter 不会实时显示。
func E2EProgressLog(format string, args ...any) {
	msg := fmt.Sprintf(format, args...)
	GinkgoWriter.Println(msg)
	fmt.Fprintln(os.Stderr, msg)
}

// HelmRelease helm release 结构
type HelmRelease struct {
	Name       string `json:"name"`
	Namespace  string `json:"namespace"`
	Chart      string `json:"chart"`
	AppVersion string `json:"app_version"`
	Status     string `json:"status"`
	Updated    string `json:"updated"`
}

// GetHelmList 执行 helm list 获取所有 release
func GetHelmList() ([]HelmRelease, error) {
	cmd := exec.Command("helm", "list", "--all-namespaces", "-o", "json")
	var out bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("helm list 执行失败: %v\nStderr: %s", err, stderr.String())
	}

	output := out.Bytes()
	if len(output) == 0 {
		return nil, fmt.Errorf("helm list 输出为空")
	}

	var releases []HelmRelease
	if err := json.Unmarshal(output, &releases); err != nil {
		return nil, fmt.Errorf("解析 helm list JSON 失败: %v\nOutput: %s", err, string(output))
	}

	return releases, nil
}

// VerifyHelmReleaseStatus 验证指定 release 的 status 为 deployed
func VerifyHelmReleaseStatus(releaseName, namespace string) {
	GinkgoWriter.Printf("验证 helm release %s/%s 状态为 deployed...\n", releaseName, namespace)

	Eventually(func() string {
		releases, err := GetHelmList()
		if err != nil {
			GinkgoWriter.Printf("GetHelmList 失败: %v\n", err)
			return ""
		}
		for _, r := range releases {
			if r.Name == releaseName && r.Namespace == namespace {
				GinkgoWriter.Printf("Helm release status: %s\n", r.Status)
				return r.Status
			}
		}
		return ""
	}, 2*time.Minute, 5*time.Second).Should(Equal("deployed"),
		"helm release %s 应在 2 分钟内变为 deployed 状态", releaseName)

	GinkgoWriter.Printf("helm release %s 状态验证成功: deployed\n", releaseName)
}

// HelmUninstall 执行 helm uninstall
func HelmUninstall(releaseName, namespace string) error {
	args := []string{"uninstall", releaseName}
	if namespace != "" {
		args = append(args, "-n", namespace)
	} else {
		args = append(args, "--all-namespaces")
	}
	cmd := exec.Command("helm", args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		if strings.Contains(string(out), "release: not found") {
			GinkgoWriter.Printf("Helm release %s 不存在（可能已卸载）\n", releaseName)
			return nil
		}
		return fmt.Errorf("helm uninstall 失败: %v\nOutput: %s", err, string(out))
	}
	GinkgoWriter.Printf("helm uninstall %s 成功\n", releaseName)
	return nil
}

// RunElasticScalerHelmInstall 安装 pd-orchestrator（elastic-scaler 子 chart），enableTidal 控制是否启用 tidal 样例。
// 失败且判定为网络/超时类抖动时，清理残留 release 与 hook Job 后重试一次。
func RunElasticScalerHelmInstall(upgradeInstall, enableTidal bool) error {
	var lastErr error
	for attempt := 1; attempt <= ElasticScalerHelmInstallAttempts; attempt++ {
		lastErr = runElasticScalerHelmInstallOnce(upgradeInstall, enableTidal)
		if lastErr == nil {
			return nil
		}
		if attempt == ElasticScalerHelmInstallAttempts || !isTransientHelmFailure(lastErr) {
			break
		}
		E2EProgressLog("[helm] elastic-scaler 第 %d/%d 次安装失败（判定可重试），清理后重试：%v",
			attempt, ElasticScalerHelmInstallAttempts, lastErr)
		if err := PurgeFailedElasticScalerRelease(); err != nil {
			return fmt.Errorf("%v\n重试前清理失败 release 失败: %v", lastErr, err)
		}
	}
	return lastErr
}

func runElasticScalerHelmInstallOnce(upgradeInstall, enableTidal bool) error {
	args := buildElasticScalerHelmArgs(upgradeInstall, enableTidal)
	action := "install"
	if upgradeInstall {
		action = "upgrade --install"
	}
	cmd := exec.Command("helm", args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("helm %s elastic-scaler failed: %v\nOutput: %s", action, err, string(out))
	}
	GinkgoWriter.Printf("helm %s elastic-scaler chart=%s output:\n%s\n", action, resolveHelmChart(EnvElasticScalerHelmChart, ElasticScalerHelmChart), string(out))
	return nil
}

// PurgeFailedElasticScalerRelease 清理上一次失败安装留下的 failed release 与残留 hook Job。
//
// 为什么不直接重试：chart 的 pre-install hook（生成 webhook 证书 Secret）只在 install 分支执行。
// release 停在 failed 时再执行 helm upgrade --install 会走 upgrade 分支、不重跑 pre-install hook，
// 于是 webhook 证书 Secret 缺失、ValidatingWebhookConfiguration 未注册，controller Pod 起不来，
// 后续校验只会报"Pod 未 Ready"这类误导性错误（helm 侧却显示升级成功）。
// 另外这些 hook Job 的 hook-delete-policy 只有 hook-succeeded（无 before-hook-creation），
// 失败/超时后 Job 会残留，且 helm uninstall 不清理 hook 资源，必须显式删除，否则下一次 install 撞 AlreadyExists。
func PurgeFailedElasticScalerRelease() error {
	releases, err := GetHelmList()
	if err != nil {
		return err
	}
	for _, r := range releases {
		if r.Name != ElasticScalerHelmReleaseName || r.Namespace != ElasticScalerHelmNamespace {
			continue
		}
		if r.Status != "deployed" {
			E2EProgressLog("[helm] release %s/%s 状态为 %s，先卸载再重装",
				ElasticScalerHelmReleaseName, ElasticScalerHelmNamespace, r.Status)
			if err := HelmUninstall(ElasticScalerHelmReleaseName, ElasticScalerHelmNamespace); err != nil {
				return err
			}
		}
		break
	}
	return deleteElasticScalerHookJobs()
}

// deleteElasticScalerHookJobs 删除按 release 标签残留的 hook Job（helm uninstall 不会清理 hook 资源）。
func deleteElasticScalerHookJobs() error {
	cmd := exec.Command("kubectl", "delete", "job", "-n", ElasticScalerHelmNamespace,
		"-l", "app.kubernetes.io/instance="+ElasticScalerHelmReleaseName,
		"--ignore-not-found", "--wait=false")
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("清理 hook Job 失败: %v\nOutput: %s", err, string(out))
	}
	GinkgoWriter.Printf("已清理 %s 命名空间下 %s 的残留 hook Job\n", ElasticScalerHelmNamespace, ElasticScalerHelmReleaseName)
	return nil
}

// isTransientHelmFailure 判断 helm 安装失败是否属于网络/镜像/超时类抖动——这类失败重试有意义；
// 参数错误、模板渲染失败等确定性失败直接返回，避免无谓地再等一个 --timeout。
func isTransientHelmFailure(err error) bool {
	message := strings.ToLower(err.Error())
	for _, pattern := range transientHelmFailurePatterns {
		if strings.Contains(message, pattern) {
			return true
		}
	}
	return false
}

// transientHelmFailurePatterns 匹配 helm/hook 的网络与超时类报错关键字。
var transientHelmFailurePatterns = []string{
	"timed out waiting for the condition", // helm --timeout 等待 hook Job 超时
	"context deadline exceeded",
	"i/o timeout",
	"no such host",
	"connection refused",
	"connection reset by peer",
	"tls handshake timeout",
	"unexpected eof",
	"backofflimitexceeded", // hook Job 反复失败，多为拉镜像失败
	"imagepullbackoff",
	"errimagepull",
	"already exists", // 上次失败残留的 hook Job 阻塞本次 install
}

func buildElasticScalerHelmArgs(upgradeInstall, enableTidal bool) []string {
	chart := resolveHelmChart(EnvElasticScalerHelmChart, ElasticScalerHelmChart)
	args := []string{"install"}
	if upgradeInstall {
		args = []string{"upgrade", "--install"}
	}
	args = append(args, ElasticScalerHelmReleaseName, chart)
	args = appendHelmVersionArg(args, EnvElasticScalerHelmVersion, ElasticScalerHelmVersion, chart)
	args = appendHelmTimeoutArg(args, ElasticScalerHelmTimeout)
	args = append(args,
		"-n", ElasticScalerHelmNamespace,
		"--create-namespace",
		// chart 内的 namespace 默认值是 scaling-system，必须显式覆盖，否则资源会落到别人的命名空间；
		// tidal 的资源名还要加前缀：它那批集群级 ClusterRole/ClusterRoleBinding 的名字与命名空间无关，
		// 同集群只要还有另一份 pd-orchestrator（含 tidal）安装就必然撞名（见 tidal_config.go）。
		"--set", "elastic-scaler.namespace.name="+ElasticScalerHelmNamespace,
		"--set", "tidal.namespace.name="+ElasticScalerHelmNamespace,
		"--set", "tidal.fullnameOverride="+TidalFullnameOverride,
		"--set", "tidal.serviceAccount.name="+TidalServiceAccountName,
		"--set", "global.image.pullPolicy=Always",
		"--set", "elastic-scaler.enabled=true",
		"--set", "elastic-scaler.elasticScaler.enabled=false",
		"--set", "elastic-scaler.webhook.enabled=true",
		"--set", "elastic-scaler.webhook.failurePolicy=Fail",
		"--set", "resourcescalinggroup.enabled=false",
	)
	if enableTidal {
		args = append(args, "--set", "tidal.enabled=true")
	} else {
		args = append(args, "--set", "tidal.enabled=false")
	}
	if repo := strings.TrimSpace(envOrDefault(EnvElasticScalerImageRepo, "")); repo != "" {
		args = append(args, "--set", "elastic-scaler.image.repository="+repo)
	}
	if tag := strings.TrimSpace(envOrDefault(EnvElasticScalerImageTag, "")); tag != "" {
		args = append(args, "--set", "elastic-scaler.image.tag="+tag)
	}
	return args
}
