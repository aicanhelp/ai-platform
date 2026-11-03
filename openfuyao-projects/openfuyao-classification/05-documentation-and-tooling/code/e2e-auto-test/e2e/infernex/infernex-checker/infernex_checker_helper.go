package infernex_checker

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"gopkg.in/yaml.v3"
)

var (
	checkBinOnce  sync.Once
	checkBinPath  string
	checkRunCache sync.Map
)

// The e2e suite treats infernex-checker as a black-box CLI. These structs are a
// minimal copy of the JSON report shape written by `infernex-checker --output`;
// do not import the product package here.
type checkStatus string

const (
	statusPassed  checkStatus = "passed"
	statusFailed  checkStatus = "failed"
	statusInfo    checkStatus = "info"
	statusWarning checkStatus = "warning"
	statusSkipped checkStatus = "skipped"
)

type checkResult struct {
	ID         string                 `json:"id"`
	Node       string                 `json:"node,omitempty"`
	Status     checkStatus            `json:"status"`
	Message    string                 `json:"message"`
	Suggestion string                 `json:"suggestion,omitempty"`
	Detail     map[string]interface{} `json:"detail,omitempty"`
}

type nodeResult struct {
	Name        string        `json:"name"`
	Status      checkStatus   `json:"status"`
	Is910Series bool          `json:"is910Series"`
	SkippedHCCS bool          `json:"skippedHccs,omitempty"`
	SkipReason  string        `json:"skipReason,omitempty"`
	Checks      []checkResult `json:"checks"`
}

type hardwareResult struct {
	Nodes      []nodeResult  `json:"nodes"`
	CrossNodes []checkResult `json:"cross_nodes,omitempty"`
}

type k8sResult struct {
	Status checkStatus   `json:"status"`
	Checks []checkResult `json:"checks"`
}

type configEnvResult struct {
	Status checkStatus   `json:"status"`
	Checks []checkResult `json:"checks"`
}

type summary struct {
	Total   int `json:"total"`
	Passed  int `json:"passed"`
	Failed  int `json:"failed"`
	Warning int `json:"warning"`
	Info    int `json:"info"`
	Skipped int `json:"skipped"`
}

type report struct {
	Summary            summary          `json:"summary"`
	Hardware           *hardwareResult  `json:"hardware,omitempty"`
	K8s                *k8sResult       `json:"k8s,omitempty"`
	ConfigEnv          *configEnvResult `json:"config_env,omitempty"`
	HardwareTerminated bool             `json:"hardwareTerminated,omitempty"`
}

type checkRun struct {
	stdout string
	stderr string
	err    error
	report *report
}

type envNodeEntry struct {
	Name     string `yaml:"name"`
	Host     string `yaml:"ip"`
	Port     int    `yaml:"port,omitempty"`
	User     string `yaml:"user"`
	Password string `yaml:"password,omitempty"`
	KeyFile  string `yaml:"keyFile,omitempty"`
}

type envNodesDoc struct {
	Nodes []envNodeEntry `yaml:"nodes"`
}

func runInfernexCheckCached(command string, needValues bool, needKubeconfig bool) checkRun {
	key := fmt.Sprintf("%s:%t:%t", command, needValues, needKubeconfig)
	if cached, ok := checkRunCache.Load(key); ok {
		return cached.(checkRun)
	}
	run := runInfernexCheck(command, needValues, needKubeconfig)
	checkRunCache.Store(key, run)
	return run
}

func checkerDebugEnabled() bool {
	value := strings.TrimSpace(os.Getenv("INFERNEX_CHECKER_DEBUG"))
	return value == "1" || strings.EqualFold(value, "true")
}

func writeCommandLogOnFailure(err error) bool {
	return err != nil || checkerDebugEnabled()
}

func runInfernexCheck(command string, needValues bool, needKubeconfig bool) checkRun {
	bin := resolveCheckBinary()
	nodesFile := resolveNodesFile()
	valuesFile := ""
	if needValues {
		valuesFile = resolveValuesFile()
	}

	outputFile := filepath.Join(GinkgoT().TempDir(), command+"-report.json")
	args := []string{command, "--nodes", nodesFile, "--output", outputFile}
	if needValues {
		args = append(args, "--values", valuesFile)
	}
	if needKubeconfig && strings.TrimSpace(InfernexCheckerConfig.KubeconfigPath) != "" {
		args = append(args, "--kubeconfig", InfernexCheckerConfig.KubeconfigPath)
	}
	// 启用连通性检查，以便 H-07 Step 2 和 H-10 跨节点 ping 检查能够执行
	args = append(args, "--enable-connectivity-check")

	cmd := exec.Command(bin, args...)
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()

	if writeCommandLogOnFailure(err) {
		GinkgoWriter.Printf("command: %s %s\n", bin, strings.Join(args, " "))
		GinkgoWriter.Printf("stdout:\n%s\n", stdout.String())
		if stderr.String() != "" {
			GinkgoWriter.Printf("stderr:\n%s\n", stderr.String())
		}
	}

	run := checkRun{stdout: stdout.String(), stderr: stderr.String(), err: err}
	if data, readErr := os.ReadFile(outputFile); readErr == nil && len(data) > 0 {
		var parsed report
		Expect(json.Unmarshal(data, &parsed)).To(Succeed(), "JSON report should be valid")
		run.report = &parsed
	}
	return run
}

func resolveCheckBinary() string {
	checkBinOnce.Do(func() {
		url := strings.TrimSpace(InfernexCheckerConfig.BinaryURL)
		if url == "" {
			url = fmt.Sprintf(InfernexCheckerConfig.BinaryURLPattern, runtime.GOARCH)
		}
		checkBinPath = downloadExecutable(url)
	})
	return checkBinPath
}

func resolveNodesFile() string {
	if path := strings.TrimSpace(InfernexCheckerConfig.NodesYAMLPath); path != "" {
		if _, err := os.Stat(path); err != nil {
			Skip(fmt.Sprintf("configured NodesYAMLPath %s is not accessible: %v", path, err))
		}
		return path
	}
	if url := strings.TrimSpace(InfernexCheckerConfig.NodesYAMLURL); url != "" {
		return resolveCheckerConfigFile("nodes.yaml", url)
	}
	return generateNodesFileFromEnv()
}

func resolveCheckerConfigFile(fileName, url string) string {
	url = strings.TrimSpace(url)
	if url == "" {
		Skip(fmt.Sprintf("set InfernexCheckerConfig.NodesYAMLURL or NodesYAMLPath in constants.go"))
	}
	return downloadFile(url, fileName, 0644)
}

func generateNodesFileFromEnv() string {
	nodes := loadNodeEntriesFromEnv()
	if len(nodes) == 0 {
		Skip("set NodesYAMLPath/NodesYAMLURL or TEST_NODE*_IP in env file")
	}

	data, err := yaml.Marshal(envNodesDoc{Nodes: nodes})
	Expect(err).NotTo(HaveOccurred(), "generated nodes.yaml should be serializable")

	dir := filepath.Join(os.TempDir(), "infernex-checker-e2e")
	Expect(os.MkdirAll(dir, 0755)).To(Succeed())
	path := filepath.Join(dir, "nodes.yaml")
	Expect(os.WriteFile(path, data, 0600)).To(Succeed())
	if checkerDebugEnabled() {
		GinkgoWriter.Printf("generated nodes.yaml from env file: %s\n%s\n", path, string(data))
	}
	return path
}

func loadNodeEntriesFromEnv() []envNodeEntry {
	var nodes []envNodeEntry
	gapCount := 0
	for i := 1; i <= 99; i++ {
		ip := strings.TrimSpace(os.Getenv(fmt.Sprintf("TEST_NODE%d_IP", i)))
		if ip == "" {
			gapCount++
			if len(nodes) > 0 && gapCount >= 5 {
				break
			}
			continue
		}
		gapCount = 0

		port, _ := strconv.Atoi(strings.TrimSpace(os.Getenv(fmt.Sprintf("TEST_NODE%d_PORT", i))))
		if port == 0 {
			port = 22
		}

		name := strings.TrimSpace(os.Getenv(fmt.Sprintf("TEST_NODE%d_HOSTNAME", i)))
		if name == "" {
			name = fmt.Sprintf("node%d", i)
		}

		nodes = append(nodes, envNodeEntry{
			Name:     name,
			Host:     ip,
			Port:     port,
			User:     envOrDefault(fmt.Sprintf("TEST_NODE%d_USER", i), "root"),
			Password: os.Getenv(fmt.Sprintf("TEST_NODE%d_PASSWORD", i)),
			KeyFile:  os.Getenv(fmt.Sprintf("TEST_NODE%d_KEYFILE", i)),
		})
	}
	return nodes
}

func envOrDefault(key, defaultValue string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return defaultValue
}

func resolveValuesFile() string {
	return helmShowValuesFile()
}

func helmShowValuesFile() string {
	chartURL := strings.TrimSpace(InfernexCheckerHelmConfig.ChartURL)
	if chartURL == "" {
		Skip("set InfernexCheckerHelmConfig.ChartURL in constants.go")
	}

	args := []string{"show", "values", chartURL}
	if version := strings.TrimSpace(InfernexCheckerHelmConfig.Version); version != "" && !strings.EqualFold(version, "latest") {
		args = append(args, "--version", version)
	}

	cmd := exec.Command("helm", args...)
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if checkerDebugEnabled() {
		GinkgoWriter.Printf("command: helm %s\n", strings.Join(args, " "))
	}
	if err := cmd.Run(); err != nil {
		Skip(fmt.Sprintf("failed to generate values.yaml from chart %s: %v, stderr: %s", chartURL, err, stderr.String()))
	}
	if strings.TrimSpace(stdout.String()) == "" {
		Skip(fmt.Sprintf("helm show values returned empty output for chart %s", chartURL))
	}

	dir := filepath.Join(os.TempDir(), "infernex-checker-e2e")
	Expect(os.MkdirAll(dir, 0755)).To(Succeed())
	path := filepath.Join(dir, "values.yaml")
	Expect(os.WriteFile(path, stdout.Bytes(), 0644)).To(Succeed())
	return path
}

func downloadExecutable(url string) string {
	return downloadFile(url, filepath.Base(url), 0755)
}

func downloadFile(url, fileName string, perm os.FileMode) string {
	dir := filepath.Join(os.TempDir(), "infernex-checker-e2e")
	Expect(os.MkdirAll(dir, 0755)).To(Succeed())

	if fileName == "." || fileName == "/" || fileName == "" {
		fileName = "artifact"
	}
	path := filepath.Join(dir, fileName)

	if info, err := os.Stat(path); err == nil && info.Size() > 0 {
		Expect(os.Chmod(path, perm)).To(Succeed())
		return path
	}

	client := &http.Client{Timeout: 10 * time.Minute}
	resp, err := client.Get(url)
	Expect(err).NotTo(HaveOccurred(), "failed to download infernex-checker artifact from %s", url)
	defer resp.Body.Close()
	Expect(resp.StatusCode).To(Equal(http.StatusOK), "unexpected HTTP status when downloading %s", url)

	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, perm)
	Expect(err).NotTo(HaveOccurred(), "failed to create artifact file %s", path)
	defer file.Close()

	_, err = io.Copy(file, resp.Body)
	Expect(err).NotTo(HaveOccurred(), "failed to write artifact file %s", path)
	Expect(os.Chmod(path, perm)).To(Succeed())
	return path
}

func hasFailedCheckLine(output string) bool {
	failedIDs := []string{"H-01", "H-02", "H-03", "H-05", "H-06", "H-07", "H-08", "H-09", "H-10", "K-01", "K-02", "B-00", "B-01", "B-02"}
	for _, id := range failedIDs {
		if strings.Contains(output, id) && hasSuggestionLine(output) {
			return true
		}
	}
	return false
}

func hasSuggestionLine(output string) bool {
	return regexp.MustCompile(`(?m)^\s*->\s*.+`).MatchString(output) || strings.Contains(output, "\u2192")
}

func requireK8sCheck(report *report, id string) checkResult {
	checks := findK8sChecks(report, id)
	Expect(checks).NotTo(BeEmpty(), "k8s report should contain %s", id)
	return checks[0]
}

func findK8sChecks(report *report, id string) []checkResult {
	if report == nil || report.K8s == nil {
		return nil
	}
	var checks []checkResult
	for _, check := range report.K8s.Checks {
		if check.ID == id {
			checks = append(checks, check)
		}
	}
	return checks
}

func findConfigChecks(report *report, id string) []checkResult {
	if report == nil || report.ConfigEnv == nil {
		return nil
	}
	var checks []checkResult
	for _, check := range report.ConfigEnv.Checks {
		if check.ID == id {
			checks = append(checks, check)
		}
	}
	return checks
}

func assertConfigFailureOrSkip(report *report, id string, messagePart string, skipReason string) {
	checks := findConfigChecks(report, id)
	for _, check := range checks {
		if check.Status == statusFailed && strings.Contains(check.Message, messagePart) {
			Expect(check.Suggestion).NotTo(BeEmpty(), fmt.Sprintf("%s should include a suggestion", check.ID))
			return
		}
	}
	Skip(skipReason)
}
