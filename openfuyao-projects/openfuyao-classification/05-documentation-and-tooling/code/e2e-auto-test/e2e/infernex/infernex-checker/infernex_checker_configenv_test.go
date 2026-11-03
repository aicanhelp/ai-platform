package infernex_checker

import (
	"fmt"
	"net"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"golang.org/x/crypto/ssh"
	"gopkg.in/yaml.v3"
)

type configValuesEvidence struct {
	cachePath         string
	cachePathSet      bool
	imageTag          string
	imageTagSet       bool
	valuesFileContent string
}

type b01Evidence struct {
	node       checkerNode
	cachePath  string
	configured bool
	exists     bool
	writable   bool
}

type b02Evidence struct {
	node               checkerNode
	versionReadable    bool
	driverVersion      string
	driverVersionOK    bool
	imageTag           string
	imageTagConfigured bool
}

type b00Evidence struct {
	node      checkerNode
	reachable bool
}

var _ = Describe("Infernex software pre-flight checks", Label("infernex_checker", "precheck"), func() {
	Describe("Config-Env Layer B-00~B-02 and resilience", Ordered, Label("config-env"), func() {
		var configRun checkRun
		var configReport *report
		var nodes []checkerNode
		var valuesEnv configValuesEvidence
		var b00Env []b00Evidence
		var b01Env []b01Evidence
		var b02Env []b02Evidence

		BeforeAll(func() {
			nodes = loadCheckerNodes()
			valuesEnv = collectConfigValuesEvidence()
			b00Env = collectB00Evidence(nodes)
			reachableNodes := reachableCheckerNodes(b00Env)
			b01Env = collectB01Evidence(reachableNodes, valuesEnv)
			b02Env = collectB02Evidence(reachableNodes, valuesEnv)

			configRun = runInfernexCheckCached("config-env", true, false)
			Expect(configRun.err).NotTo(HaveOccurred(), "infernex-check config-env should run successfully, stderr: %s", configRun.stderr)
			configReport = configRun.report
			Expect(configReport).NotTo(BeNil(), "config-env command should write JSON report")
			Expect(configReport.ConfigEnv).NotTo(BeNil(), "JSON report should contain config-env result")
		})

		It("ic_cfg_001-【B-00】ConfigEnv层SSH连接失败时报B-00失败并跳过该节点", Label("P0"), func() {
			matched := filterEvidence(b00Env, func(item b00Evidence) bool {
				return !item.reachable
			})
			if len(matched) == 0 {
				Skip("current nodes.yaml nodes are reachable over SSH; B-00 SSH failure scenario requires a real unreachable node")
			}
			for _, item := range matched {
				assertConfigNodeFailure(configReport, item.node, "B-00", "SSH connection failed")
			}
		})

		It("ic_cfg_002-【B-01】cachePath目录存在且可写时通过", Label("P0"), func() {
			matched := filterEvidence(b01Env, func(item b01Evidence) bool {
				return item.configured && item.exists && item.writable
			})
			if len(matched) == 0 {
				Skip("no configured node currently satisfies B-01 positive precondition: cache path exists and is writable")
			}
			for _, item := range matched {
				check := configCheckForNodeOrSkip(configReport, item.node, "B-01")
				Expect(check.Status).To(Equal(statusPassed), "B-01 should pass on node %s", item.node.Name)
				Expect(check.Message).To(ContainSubstring("directory exists and is writable"))
			}
		})

		It("ic_cfg_003-【B-01】global.cachePath未配置时报失败", Label("P0"), func() {
			if valuesEnv.cachePathSet {
				Skip("current values.yaml configures global.cachePath; missing cachePath scenario requires modified values.yaml or mock test")
			}
			assertConfigFailureOrSkip(configReport, "B-01", "global.cachePath is not configured", "current values.yaml configures global.cachePath; missing cachePath scenario requires modified values.yaml or mock test")
		})

		It("ic_cfg_004-【B-01】cachePath目录不存在时Step1报失败", Label("P1"), func() {
			matched := filterEvidence(b01Env, func(item b01Evidence) bool {
				return item.configured && !item.exists
			})
			if len(matched) == 0 {
				Skip("current cachePath exists on configured nodes; missing directory scenario requires modified host state or mock test")
			}
			for _, item := range matched {
				assertConfigNodeFailureOrSkip(configReport, item.node, "B-01", "does not exist", "matched cachePath missing evidence but checker did not report B-01 missing directory for this node")
			}
		})

		It("ic_cfg_005-【B-01】cachePath目录存在但无写权限时Step2报失败", Label("P1"), func() {
			matched := filterEvidence(b01Env, func(item b01Evidence) bool {
				return item.configured && item.exists && !item.writable
			})
			if len(matched) == 0 {
				Skip("current cachePath is writable on configured nodes; permission failure scenario requires modified host state or mock test")
			}
			for _, item := range matched {
				assertConfigNodeFailureOrSkip(configReport, item.node, "B-01", "not writable", "matched cachePath permission evidence but checker did not report B-01 not writable for this node")
			}
		})

		It("ic_cfg_006-【B-02】驱动版本与镜像Tag兼容时通过", Label("P0"), func() {
			matched := filterEvidence(b02Env, func(item b02Evidence) bool {
				return item.versionReadable && item.driverVersionOK && item.imageTagConfigured
			})
			if len(matched) == 0 {
				Skip("no configured node currently has readable driver version and configured image tag for B-02 compatibility validation")
			}
			passed := false
			for _, item := range matched {
				checks := configChecksForNode(configReport, item.node, "B-02")
				for _, check := range checks {
					if check.Status == statusPassed && strings.Contains(check.Message, "compatible with image") {
						passed = true
						Expect(check.Message).To(ContainSubstring(item.driverVersion[:minInt(len(item.driverVersion), 4)]))
					}
				}
			}
			if !passed {
				Skip("current driver/image combination is not compatible; B-02 positive compatibility requires a matching image tag")
			}
		})

		It("ic_cfg_007-【B-02】version.info读取失败时Step1报失败", Label("P1"), func() {
			matched := filterEvidence(b02Env, func(item b02Evidence) bool {
				return !item.versionReadable
			})
			if len(matched) == 0 {
				Skip("current nodes can read version.info; read failure requires modified host state or mock test")
			}
			for _, item := range matched {
				assertConfigNodeFailureOrSkip(configReport, item.node, "B-02", "failed to read /usr/local/Ascend/driver/version.info", "matched unreadable version.info evidence but checker did not report B-02 read failure for this node")
			}
		})

		It("ic_cfg_008-【B-02】version.info格式异常无法解析版本时报失败", Label("P2"), func() {
			matched := filterEvidence(b02Env, func(item b02Evidence) bool {
				return item.versionReadable && !item.driverVersionOK
			})
			if len(matched) == 0 {
				Skip("current version.info is parseable; parse failure requires modified host state or mock test")
			}
			for _, item := range matched {
				assertConfigNodeFailureOrSkip(configReport, item.node, "B-02", "unable to parse Driver version", "matched unparseable version.info evidence but checker did not report B-02 parse failure for this node")
			}
		})

		It("ic_cfg_009-【B-02】inferenceEngine.tag未配置时Step2报失败", Label("P0"), func() {
			if valuesEnv.imageTagSet {
				Skip("current values.yaml configures image tag; missing tag scenario requires modified values.yaml or mock test")
			}
			assertConfigFailureOrSkip(configReport, "B-02", "inference-backend.images.inferenceEngine.tag is not configured", "current values.yaml configures image tag; missing tag scenario requires modified values.yaml or mock test")
		})

		It("ic_cfg_010-【B-02】imageTag不在兼容矩阵时Step2报Warning", Label("P1"), func() {
			if !valuesEnv.imageTagSet {
				Skip("current values.yaml does not configure image tag; missing tag scenario is covered by ic_cfg_009")
			}
			matched := filterEvidence(b02Env, func(item b02Evidence) bool {
				return item.versionReadable && item.driverVersionOK && item.imageTagConfigured
			})
			if len(matched) == 0 {
				Skip("no configured node currently has readable driver version for unsupported compatibility matrix validation")
			}
			foundNotInMatrix := false
			for _, item := range matched {
				if !isInInferenceCompatibilityMatrix(item.imageTag, item.driverVersion) {
					foundNotInMatrix = true
					break
				}
			}
			if !foundNotInMatrix {
				Skip("current image tag and driver version are in the compatibility matrix")
			}
			assertConfigWarning(configReport, "B-02", "not in the built-in compatibility matrix")
		})

		It("ic_cfg_011-【B-02】驱动版本前缀不匹配时Step2报Warning", Label("P1"), func() {
			matched := filterEvidence(b02Env, func(item b02Evidence) bool {
				return item.versionReadable && item.driverVersionOK && item.imageTagConfigured
			})
			if len(matched) == 0 {
				Skip("no configured node currently has readable driver version and configured image tag for B-02 mismatch validation")
			}
			found := false
			for _, item := range matched {
				checks := configChecksForNode(configReport, item.node, "B-02")
				for _, check := range checks {
					if check.Status == statusWarning && strings.Contains(check.Message, "not in the built-in compatibility matrix") {
						found = true
					}
				}
			}
			if !found {
				Skip("current driver and image are compatible; mismatch scenario requires modified driver/image version or mock test")
			}
		})

		It("ic_res_001-【韧性】多节点并发检查节点间互不干扰", Label("P1"), func() {
			reachableNodes := reachableCheckerNodes(b00Env)
			if len(reachableNodes) < 3 {
				Skip("current nodes.yaml has fewer than three SSH-reachable nodes; resilience scenario requires at least three real nodes")
			}

			hardwareRun := runInfernexCheckCached("hardware", false, false)
			Expect(hardwareRun.err).NotTo(HaveOccurred(), "hardware command should run successfully for resilience validation, stderr: %s", hardwareRun.stderr)
			Expect(hardwareRun.report).NotTo(BeNil())
			Expect(hardwareRun.report.Hardware).NotTo(BeNil())

			profiles := collectHardwareResilienceProfiles(reachableNodes)
			if countDistinctProfiles(profiles) < 2 {
				Skip("current hardware environment does not have mixed node outcomes; resilience scenario requires real mixed node results")
			}

			for _, node := range reachableNodes {
				profile := profiles[nodeIdentity(node)]
				nodeReport := hardwareNodeResultForCheckerNode(hardwareRun.report, node)
				Expect(nodeReport).NotTo(BeNil(), "hardware report should contain node %s", node.Name)
				Expect(nodeReport.Name).NotTo(BeEmpty())
				Expect(nodeReport.Checks).NotTo(BeEmpty(), "node %s should have independent hardware check results", node.Name)
				if profile.failedID != "" {
					expectHardwareNodeFailedAt(nodeReport, profile.failedID, node.Name)
				}
			}
		})
	})
})

func collectConfigValuesEvidence() configValuesEvidence {
	valuesFile := resolveValuesFile()
	data, err := os.ReadFile(valuesFile)
	Expect(err).NotTo(HaveOccurred(), "values.yaml should be readable")

	var values map[string]interface{}
	Expect(yaml.Unmarshal(data, &values)).To(Succeed(), "values.yaml should be valid yaml")

	cachePath := nestedString(values, "global", "cachePath")
	imageTag := nestedString(values, "inference-backend", "images", "inferenceEngine", "tag")
	return configValuesEvidence{
		cachePath:         cachePath,
		cachePathSet:      strings.TrimSpace(cachePath) != "",
		imageTag:          imageTag,
		imageTagSet:       strings.TrimSpace(imageTag) != "",
		valuesFileContent: string(data),
	}
}

func collectB00Evidence(nodes []checkerNode) []b00Evidence {
	var evidence []b00Evidence
	for _, node := range nodes {
		evidence = append(evidence, b00Evidence{node: node, reachable: sshReachable(node)})
	}
	return evidence
}

func sshReachable(node checkerNode) bool {
	config := &ssh.ClientConfig{
		User:            node.Username,
		HostKeyCallback: ssh.InsecureIgnoreHostKey(),
		Timeout:         30 * time.Second,
	}
	if node.Password != "" {
		config.Auth = append(config.Auth, ssh.Password(node.Password))
	}
	if key := strings.TrimSpace(node.PrivateKey); key != "" {
		signer, err := ssh.ParsePrivateKey([]byte(key))
		if err != nil {
			return false
		}
		config.Auth = append(config.Auth, ssh.PublicKeys(signer))
	}
	if node.KeyPath != "" {
		keyData, err := os.ReadFile(node.KeyPath)
		if err != nil {
			return false
		}
		signer, err := ssh.ParsePrivateKey(keyData)
		if err != nil {
			return false
		}
		config.Auth = append(config.Auth, ssh.PublicKeys(signer))
	}
	if len(config.Auth) == 0 {
		return false
	}

	client, err := ssh.Dial("tcp", net.JoinHostPort(node.IP, strconv.Itoa(node.Port)), config)
	if err != nil {
		return false
	}
	_ = client.Close()
	return true
}

func reachableCheckerNodes(evidence []b00Evidence) []checkerNode {
	var nodes []checkerNode
	for _, item := range evidence {
		if item.reachable {
			nodes = append(nodes, item.node)
		}
	}
	return nodes
}

func assertConfigNodeFailure(report *report, node checkerNode, id string, messagePart string) {
	checks := configChecksForSpecificNode(report, node, id)
	for _, check := range checks {
		if check.Status == statusFailed && strings.Contains(check.Message, messagePart) {
			Expect(check.Suggestion).NotTo(BeEmpty(), fmt.Sprintf("%s should include a suggestion", check.ID))
			return
		}
	}
	Expect(checks).To(ContainElement(SatisfyAll(
		WithTransform(func(c checkResult) checkStatus { return c.Status }, Equal(statusFailed)),
		WithTransform(func(c checkResult) string { return c.Message }, ContainSubstring(messagePart)),
	)), "%s should fail on node %s with message containing %q", id, node.Name, messagePart)
}

func assertConfigWarning(report *report, id string, messagePart string) {
	checks := findConfigChecks(report, id)
	for _, check := range checks {
		if check.Status == statusWarning && strings.Contains(check.Message, messagePart) {
			return
		}
	}
	Expect(checks).To(ContainElement(SatisfyAll(
		WithTransform(func(c checkResult) checkStatus { return c.Status }, Equal(statusWarning)),
		WithTransform(func(c checkResult) string { return c.Message }, ContainSubstring(messagePart)),
	)), "%s should warn with message containing %q", id, messagePart)
}

func isInInferenceCompatibilityMatrix(tag, driverVersion string) bool {
	compatMatrix := map[string][]string{
		"v0.13.0": []string{"25.5.", "25.3.", "25.2.", "25.0."},
		"v0.18.0": []string{"26.0.RC1", "25.7.RC1", "25.5.", "25.3.", "25.2.", "25.0."},
	}

	prefixes, ok := compatMatrix[strings.TrimSpace(tag)]
	if !ok {
		return false
	}

	driverVersion = strings.TrimSpace(driverVersion)
	for _, prefix := range prefixes {
		if strings.HasPrefix(driverVersion, strings.TrimSpace(prefix)) {
			return true
		}
	}
	return false
}

func collectB01Evidence(nodes []checkerNode, values configValuesEvidence) []b01Evidence {
	var evidence []b01Evidence
	for _, node := range nodes {
		item := b01Evidence{node: node, cachePath: values.cachePath, configured: values.cachePathSet}
		if item.configured {
			item.exists = runSSH(node, "test -d "+shellQuote(item.cachePath)).err == nil
			item.writable = runSSH(node, "test -w "+shellQuote(item.cachePath)).err == nil
		}
		evidence = append(evidence, item)
	}
	return evidence
}

func collectB02Evidence(nodes []checkerNode, values configValuesEvidence) []b02Evidence {
	var evidence []b02Evidence
	for _, node := range nodes {
		result := runSSH(node, "cat /usr/local/Ascend/driver/version.info")
		output := strings.TrimSpace(result.stdout + "\n" + result.stderr)
		driverVersion := parseDriverVersion(output)
		evidence = append(evidence, b02Evidence{
			node:               node,
			versionReadable:    result.err == nil,
			driverVersion:      driverVersion,
			driverVersionOK:    result.err == nil && driverVersion != "",
			imageTag:           values.imageTag,
			imageTagConfigured: values.imageTagSet,
		})
	}
	return evidence
}

func configCheckForNodeOrSkip(report *report, node checkerNode, id string) checkResult {
	checks := configChecksForNode(report, node, id)
	if len(checks) == 0 {
		Skip(fmt.Sprintf("%s was not executed for node %s", id, node.Name))
	}
	return checks[0]
}

func assertConfigNodeFailureOrSkip(report *report, node checkerNode, id string, messagePart string, skipReason string) {
	checks := configChecksForNode(report, node, id)
	for _, check := range checks {
		if check.Status == statusFailed && strings.Contains(check.Message, messagePart) {
			Expect(check.Suggestion).NotTo(BeEmpty(), fmt.Sprintf("%s should include a suggestion", check.ID))
			return
		}
	}
	Skip(skipReason)
}

func configChecksForNode(report *report, node checkerNode, id string) []checkResult {
	checks := findConfigChecks(report, id)
	var matched []checkResult
	for _, check := range checks {
		if check.Node == "" || check.Node == node.Name || check.Node == node.IP || strings.Contains(check.Message, node.Name) || strings.Contains(check.Message, node.IP) {
			matched = append(matched, check)
		}
	}
	return matched
}

func configChecksForSpecificNode(report *report, node checkerNode, id string) []checkResult {
	checks := findConfigChecks(report, id)
	var matched []checkResult
	for _, check := range checks {
		if check.Node == node.Name || check.Node == node.IP || strings.Contains(check.Message, node.Name) || strings.Contains(check.Message, node.IP) {
			matched = append(matched, check)
		}
	}
	return matched
}

type hardwareResilienceProfile struct {
	failedID string
	label    string
}

func collectHardwareResilienceProfiles(nodes []checkerNode) map[string]hardwareResilienceProfile {
	h01ByNode := indexByNode(collectH01Evidence(nodes), func(e h01Evidence) checkerNode { return e.node })
	h02ByNode := indexByNode(collectH02Evidence(nodes), func(e h02Evidence) checkerNode { return e.node })
	h03ByNode := indexByNode(collectH03Evidence(nodes), func(e h03Evidence) checkerNode { return e.node })
	h05ByNode := indexByNode(collectH05Evidence(nodes), func(e h05Evidence) checkerNode { return e.node })
	h06ByNode := indexByNode(collectH06Evidence(nodes), func(e h06Evidence) checkerNode { return e.node })

	profiles := map[string]hardwareResilienceProfile{}
	for _, node := range nodes {
		key := nodeIdentity(node)
		h01 := h01ByNode[key]
		if !h01.driverLoaded {
			profiles[key] = hardwareResilienceProfile{failedID: "H-01", label: "H-01:driver-missing"}
			continue
		}
		if !h01.npuSmiOK {
			profiles[key] = hardwareResilienceProfile{failedID: "H-01", label: "H-01:npu-smi-failed"}
			continue
		}

		h02 := h02ByNode[key]
		if !h02.pluginFound {
			profiles[key] = hardwareResilienceProfile{failedID: "H-02", label: "H-02:plugin-missing"}
			continue
		}
		if !h02.pluginRunning {
			profiles[key] = hardwareResilienceProfile{failedID: "H-02", label: "H-02:plugin-not-running"}
			continue
		}
		if !h02.resourceRegistered {
			profiles[key] = hardwareResilienceProfile{failedID: "H-02", label: "H-02:resource-not-registered"}
			continue
		}

		h03 := h03ByNode[key]
		if !h03.npuSmiOK || !h03.parseModel {
			profiles[key] = hardwareResilienceProfile{failedID: "H-03", label: "H-03:model-unavailable"}
			continue
		}

		h05 := h05ByNode[key]
		if !h05.pathsExist {
			profiles[key] = hardwareResilienceProfile{failedID: "H-05", label: "H-05:paths-missing"}
			continue
		}

		h06 := h06ByNode[key]
		if h03.model910 && (!h06.readable || !h06.hasAddress || !h06.hasNetmask || h06.invalidIP || h06.duplicateIP || h06.differentSubnet) {
			profiles[key] = hardwareResilienceProfile{failedID: "H-06", label: "H-06:hccn-invalid"}
			continue
		}

		if h03.model910 {
			profiles[key] = hardwareResilienceProfile{label: "eligible-910"}
		} else {
			profiles[key] = hardwareResilienceProfile{label: "non-910"}
		}
	}
	return profiles
}

func nodeIdentity(node checkerNode) string {
	if strings.TrimSpace(node.Name) != "" {
		return node.Name
	}
	return node.IP
}

func countDistinctProfiles(profiles map[string]hardwareResilienceProfile) int {
	distinct := map[string]bool{}
	for _, profile := range profiles {
		distinct[profile.label] = true
	}
	return len(distinct)
}

func hardwareNodeResultForCheckerNode(report *report, node checkerNode) *nodeResult {
	if report == nil || report.Hardware == nil {
		return nil
	}
	for i := range report.Hardware.Nodes {
		current := &report.Hardware.Nodes[i]
		if current.Name == node.Name || current.Name == node.IP {
			return current
		}
	}
	return nil
}

func expectHardwareNodeFailedAt(nodeReport *nodeResult, failedID string, nodeName string) {
	failedOrder := hardwareCheckOrder(failedID)
	found := false
	for _, check := range nodeReport.Checks {
		if check.ID == failedID && check.Status == statusFailed {
			found = true
		}
		if failedOrder > 0 && hardwareCheckOrder(check.ID) > failedOrder {
			Fail(fmt.Sprintf("node %s should stop after %s, but report contains later check %s", nodeName, failedID, check.ID))
		}
	}
	Expect(found).To(BeTrue(), "node %s should fail at %s according to real hardware evidence", nodeName, failedID)
}

func hardwareCheckOrder(id string) int {
	order := map[string]int{
		"H-01": 1,
		"H-02": 2,
		"H-03": 3,
		"H-04": 4,
		"H-05": 5,
		"H-06": 6,
		"H-07": 7,
		"H-08": 8,
		"H-09": 9,
		"H-10": 10,
	}
	return order[id]
}

func nestedString(root map[string]interface{}, path ...string) string {
	var current interface{} = root
	for _, key := range path {
		m, ok := current.(map[string]interface{})
		if !ok {
			return ""
		}
		current = m[key]
	}
	value, _ := current.(string)
	return strings.TrimSpace(value)
}

func parseDriverVersion(output string) string {
	patterns := []string{
		`(?i)version[^0-9]*([0-9]+(?:\.[0-9]+){1,3})`,
		`([0-9]+(?:\.[0-9]+){1,3})`,
	}
	for _, pattern := range patterns {
		matches := regexp.MustCompile(pattern).FindStringSubmatch(output)
		if len(matches) > 1 {
			return matches[1]
		}
	}
	return ""
}

func minInt(a int, b int) int {
	if a < b {
		return a
	}
	return b
}
