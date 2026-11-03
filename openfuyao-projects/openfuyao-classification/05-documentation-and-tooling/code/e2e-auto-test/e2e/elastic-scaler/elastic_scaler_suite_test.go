package elastic_scaler

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
)

const (
	elasticScalerE2EReexecEnv              = "E2E_ELASTIC_SCALER_REEXEC"
	elasticScalerE2EDefaultTimeout           = "90m"
	elasticScalerE2EMetricsFrameworkMinRun   = 90 * time.Minute
	elasticScalerE2EGinkgoLabelMetricsFramework = "metrics-framework"
)

var (
	clientset     *kubernetes.Clientset
	dynamicClient dynamic.Interface
	cancel        context.CancelFunc
	restConfig    *rest.Config
	ctx           context.Context
)

var _ = BeforeSuite(func() {
	var err error

	ctx, cancel = context.WithTimeout(context.Background(), 90*time.Minute)
	DeferCleanup(cancel)

	kubeconfig := os.Getenv("KUBECONFIG")
	if kubeconfig == "" {
		kubeconfig = filepath.Join(homedir.HomeDir(), ".kube", "config")
	}

	restConfig, err = clientcmd.BuildConfigFromFlags("", kubeconfig)
	Expect(err).NotTo(HaveOccurred())

	clientset, err = kubernetes.NewForConfig(restConfig)
	Expect(err).NotTo(HaveOccurred())

	dynamicClient, err = dynamic.NewForConfig(restConfig)
	Expect(err).NotTo(HaveOccurred())
})

var _ = AfterSuite(func() {
	if cancel != nil {
		cancel()
	}
})

func TestElasticScaler(t *testing.T) {
	if ginkgoLabelFilterIncludes(elasticScalerE2EGinkgoLabelMetricsFramework) {
		fmt.Fprintln(os.Stderr, "[elastic-scaler] metrics-framework 套件启动（编译完成后）；BeforeAll 含 helm/清理，可能 1-5 分钟才有首条用例输出。建议 go test 加 -v。")
	}
	if min := requiredMinTestTimeout(); min > 0 {
		if got := parseTestTimeoutFromArgs(); got < min {
			t.Fatalf(
				"当前 -test.timeout=%s，不足以跑 metrics-framework 全量套件（预计 >15 分钟）。\n"+
					"go test 驱动会在约 10 分钟（+grace）向进程发 SIGQUIT；仅 re-exec 延长测试二进制 -test.timeout 无法绕过该限制。\n"+
					"请改用：\n"+
					"  go test -timeout 90m -v -count=1 ./elastic-scaler/ -run TestElasticScaler "+
					"-args --ginkgo.v --ginkgo.label-filter=\"metrics-framework\"",
				got,
			)
		}
	}
	if os.Getenv(elasticScalerE2EReexecEnv) != "1" && shouldReexecForExtendedTimeout() {
		reexecElasticScalerTest(t)
		return
	}
	RegisterFailHandler(Fail)
	RunSpecs(t, "ElasticScaler Suite")
}

// requiredMinTestTimeout 按 ginkgo label-filter 返回最低进程超时；0 表示无额外要求。
func requiredMinTestTimeout() time.Duration {
	if ginkgoLabelFilterIncludes(elasticScalerE2EGinkgoLabelMetricsFramework) {
		return elasticScalerE2EMetricsFrameworkMinRun
	}
	return 0
}

func ginkgoLabelFilterIncludes(label string) bool {
	for _, a := range os.Args[1:] {
		const prefix = "--ginkgo.label-filter="
		if !strings.HasPrefix(a, prefix) {
			continue
		}
		filter := strings.TrimPrefix(a, prefix)
		filter = strings.Trim(filter, `"'`)
		if filter == "" {
			continue
		}
		if filter == label || strings.Contains(filter, label) {
			return true
		}
	}
	return false
}

func parseTestTimeoutFromArgs() time.Duration {
	for _, a := range os.Args[1:] {
		if strings.HasPrefix(a, "-test.timeout=") {
			raw := strings.TrimPrefix(a, "-test.timeout=")
			if d, err := time.ParseDuration(raw); err == nil {
				return d
			}
		}
	}
	return 10 * time.Minute
}

// shouldReexecForExtendedTimeout 判断是否需 re-exec 以替换 go test 默认 10m 进程级超时。
func shouldReexecForExtendedTimeout() bool {
	for _, a := range os.Args[1:] {
		if strings.HasPrefix(a, "-test.timeout=") {
			return isGoTestDefaultTimeout(a)
		}
	}
	return true
}

// isGoTestDefaultTimeout 判断是否为 go test 默认 10m 进程级超时（需为本套件延长至 90m）。
func isGoTestDefaultTimeout(arg string) bool {
	if !strings.HasPrefix(arg, "-test.timeout=") {
		return false
	}
	switch strings.TrimPrefix(arg, "-test.timeout=") {
	case "10m", "10m0s":
		return true
	default:
		return false
	}
}

// reexecElasticScalerTest 以更长 -test.timeout 重新执行本包测试（默认 90m），缓解测试二进制内部的 testing.M 超时。
// 注意：通过 go test 启动时，go 驱动自身还有独立的 -timeout（默认 10m，约 11m 发 SIGQUIT），
// 必须由用户在 go test 命令行传入 -timeout 90m；metrics-framework 全量套件在 TestElasticScaler 入口会校验。
// syscall.Exec 替换当前进程，避免 cmd.Run 父进程等待子进程时仍被内部 10m 超时杀死（直接运行 .test 二进制时有效）。
func reexecElasticScalerTest(t *testing.T) {
	t.Helper()
	args := make([]string, 0, len(os.Args[1:])+2)
	hasTimeout := false
	for _, a := range os.Args[1:] {
		if strings.HasPrefix(a, "-test.timeout=") {
			hasTimeout = true
			if isGoTestDefaultTimeout(a) {
				a = "-test.timeout=" + elasticScalerE2EDefaultTimeout
			}
		}
		args = append(args, a)
	}
	if !hasTimeout {
		args = append([]string{"-test.timeout=" + elasticScalerE2EDefaultTimeout}, args...)
	}
	env := append(os.Environ(), elasticScalerE2EReexecEnv+"=1")
	execArgs := append([]string{os.Args[0]}, args...)
	if err := syscall.Exec(os.Args[0], execArgs, env); err != nil {
		t.Fatalf("re-exec elastic-scaler test with -test.timeout=%s: %v", elasticScalerE2EDefaultTimeout, err)
	}
}
