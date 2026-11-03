package infernex_checker

import (
	"bytes"
	"encoding/json"
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

type checkerNode struct {
	Name       string
	IP         string
	Port       int
	Username   string
	Password   string
	PrivateKey string
	KeyPath    string
}

type commandResult struct {
	stdout string
	stderr string
	err    error
}

type h01Evidence struct {
	node         checkerNode
	driverLoaded bool
	npuSmiOK     bool
	lsmodOutput  string
	npuSmiOutput string
}

type h02Evidence struct {
	node               checkerNode
	k8sNodeName        string
	kubectlOK          bool
	pluginFound        bool
	pluginRunning      bool
	resourceRegistered bool
}

type h03Evidence struct {
	node       checkerNode
	npuSmiOK   bool
	model910   bool
	parseModel bool
	output     string
}

type h04Evidence struct {
	node         checkerNode
	hasNPU       bool
	npuCount     int
	occupiedIDs  []string
	unhealthyIDs []string
}

type h05Evidence struct {
	node       checkerNode
	pathsExist bool
}

type h06Evidence struct {
	node             checkerNode
	readable         bool
	hasAddress       bool
	hasNetmask       bool
	invalidIP        bool
	duplicateIP      bool
	differentSubnet  bool
	hccnConfContents string
}

type h07Evidence struct {
	node           checkerNode
	tlsPassed      bool
	pingPassed     bool
	tlsFailed      bool
	pingFailed     bool
	hccnToolFailed bool
	output         string
}

type h08Evidence struct {
	node         checkerNode
	idleNPUCount int
}

type h10Evidence struct {
	eligibleNodes      []checkerNode
	singleNode         bool
	duplicateIP        bool
	tlsInconsistent    bool
	crossPingFailed    bool
	crossConnectivity  bool
	crossPingCollected bool
}

const npuCheckScript = `#!/bin/bash
set -uo pipefail

PING_PKT=128
PASS="[PASS]"
FAIL="[FAIL]"

mapfile -t DEV_LIST < <(ls /dev/davinci* 2>/dev/null | grep -oP 'davinci\K\d+' | sort -n)

if [[ ${#DEV_LIST[@]} -eq 0 ]]; then
    echo "No NPU devices found under /dev/davinci*"
    exit 1
fi

echo "Detected NPU IDs: ${DEV_LIST[*]}"
echo ""

echo "=== TLS State Check ==="

declare -A TLS_STATE
TLS_FAIL=0
FIRST_TLS=""

for ID in "${DEV_LIST[@]}"; do
    OUT=$(hccn_tool -i "$ID" -tls -g 2>&1)
    if echo "$OUT" | grep -qi "tls switch\[1\]"; then
        STATE="enable"
    elif echo "$OUT" | grep -qi "tls switch\[0\]"; then
        STATE="disable"
    elif echo "$OUT" | grep -qi "no certificate"; then
        STATE="disable"
    else
        STATE="unknown"
    fi
    TLS_STATE[$ID]=$STATE
    echo "  NPU${ID}: ${STATE}"
done

for ID in "${DEV_LIST[@]}"; do
    STATE="${TLS_STATE[$ID]}"
    if [[ -z "$FIRST_TLS" ]]; then
        FIRST_TLS="$STATE"
    elif [[ "$STATE" != "$FIRST_TLS" ]]; then
        TLS_FAIL=1
    fi
done

echo ""
if [[ $TLS_FAIL -eq 0 ]]; then
    echo "$PASS TLS state is consistent across all NPUs (${FIRST_TLS})"
else
    echo "$FAIL TLS state is inconsistent across NPUs:"
    for ID in "${DEV_LIST[@]}"; do
        echo "       NPU${ID} = ${TLS_STATE[$ID]}"
    done
fi

echo ""
echo "=== NIC-to-NIC Ping Check ==="

declare -A IP_MAP
while IFS='=' read -r KEY VAL; do
    KEY="${KEY// /}"
    VAL="${VAL// /}"
    if [[ "$KEY" =~ ^address_([0-9]+)$ ]]; then
        NIC_ID="${BASH_REMATCH[1]}"
        IP_MAP[$NIC_ID]="$VAL"
    fi
done < /etc/hccn.conf

if [[ ${#IP_MAP[@]} -eq 0 ]]; then
    echo "No IP addresses found in /etc/hccn.conf, skipping ping check"
    exit 0
fi

echo "NIC IP mapping:"
for NIC_ID in $(echo "${!IP_MAP[@]}" | tr ' ' '\n' | sort -n); do
    echo "  NPU${NIC_ID}: ${IP_MAP[$NIC_ID]}"
done
echo ""

PING_FAIL=0
PING_FAILED_LIST=()

for SRC_ID in "${DEV_LIST[@]}"; do
    for DST_NIC in $(echo "${!IP_MAP[@]}" | tr ' ' '\n' | sort -n); do
        if [[ "$DST_NIC" == "$SRC_ID" ]]; then
            continue
        fi
        DST_IP="${IP_MAP[$DST_NIC]}"
        if [[ -z "$DST_IP" ]]; then
            continue
        fi

        SUCCESS=0
        for ATTEMPT in 1 2 3; do
            RESULT=$(hccn_tool -i "$SRC_ID" -ping -g address "$DST_IP" pkt "$PING_PKT" 2>&1)
            if echo "$RESULT" | grep -q "100.00% packet loss" || \
               echo "$RESULT" | grep -q "Command execute failed"; then
                if [[ $ATTEMPT -lt 3 ]]; then
                    sleep 1
                fi
            else
                SUCCESS=1
                break
            fi
        done

        PAIR="NPU${SRC_ID} -> NPU${DST_NIC}(${DST_IP})"
        if [[ $SUCCESS -eq 1 ]]; then
            echo "  $PASS $PAIR"
        else
            echo "  $FAIL $PAIR"
            PING_FAILED_LIST+=("$PAIR")
            PING_FAIL=1
        fi
    done
done

echo ""
if [[ $PING_FAIL -eq 0 ]]; then
    echo "$PASS All NIC-to-NIC ping paths passed"
else
    echo "$FAIL The following ping paths failed:"
    for PAIR in "${PING_FAILED_LIST[@]}"; do
        echo "       $PAIR"
    done
fi

echo ""
echo "=== Summary ==="
[[ $TLS_FAIL -eq 0 ]]  && echo "$PASS TLS check passed" || echo "$FAIL TLS check failed"
[[ $PING_FAIL -eq 0 ]] && echo "$PASS Ping check passed" || echo "$FAIL Ping check failed"

if [[ $TLS_FAIL -ne 0 ]] || [[ $PING_FAIL -ne 0 ]]; then
    exit 1
fi
`

const h04EvidenceScript = `#!/bin/bash
set -u

mapfile -t IDS < <(ls /dev/davinci[0-9]* 2>/dev/null | grep -oP 'davinci\K\d+' | sort -n | uniq)
echo "COUNT=${#IDS[@]}"

if [[ ${#IDS[@]} -eq 0 ]]; then
    echo "OCCUPIED="
    echo "UNHEALTHY="
    exit 0
fi

INFO="$(npu-smi info 2>/dev/null || true)"

# 解析 unhealthy NPUs（从设备表）
# 匹配格式：| <id>  <model> | <status> |
UNHEALTHY=()
while IFS= read -r line; do
    if [[ "$line" =~ ^\|[[:space:]]+([0-9]+)[[:space:]]+[^|]+\|[[:space:]]+(Warning|Alarm|Critical|UNKNOWN) ]]; then
        UNHEALTHY+=("NPU${BASH_REMATCH[1]}")
    fi
done <<< "$INFO"

# 解析 occupied NPUs（从进程表）
# 匹配格式：| <npu> <chip> | <pid> | <name> | <mem> |
# 与实现的正则一致：\|\s+(\d+)\s+\d+\s+\|\s+(\d+)\s+\|\s+\S+\s+\|\s+\d+\s+\|
# 注意：\S+ 表示一个或多个非空白字符，\s+ 表示一个或多个空白字符
declare -A OCCUPIED_MAP
while IFS= read -r line; do
    # Bash 等价正则：[^[:space:]]+ 对应 \S+，[[:space:]]+ 对应 \s+
    if [[ "$line" =~ ^\|[[:space:]]+([0-9]+)[[:space:]]+[0-9]+[[:space:]]+\|[[:space:]]+([0-9]+)[[:space:]]+\|[[:space:]]+[^[:space:]]+[[:space:]]+\|[[:space:]]+[0-9]+[[:space:]]+\| ]]; then
        OCCUPIED_MAP["${BASH_REMATCH[1]}"]=1
    fi
done <<< "$INFO"

# 转换为数组
OCCUPIED=()
for npu_id in "${!OCCUPIED_MAP[@]}"; do
    OCCUPIED+=("NPU${npu_id}")
done

IFS=,
echo "OCCUPIED=${OCCUPIED[*]}"
echo "UNHEALTHY=${UNHEALTHY[*]}"
`

// loadCheckerNodes 从标准 nodes.yaml 中读取硬件检查需要 SSH 登录的节点信息。
func loadCheckerNodes() []checkerNode {
	nodesFile := resolveNodesFile()
	data, err := os.ReadFile(nodesFile)
	Expect(err).NotTo(HaveOccurred(), "nodes.yaml should be readable")

	var doc envNodesDoc
	Expect(yaml.Unmarshal(data, &doc)).To(Succeed(), "nodes.yaml should be valid yaml")

	nodes := make([]checkerNode, 0, len(doc.Nodes))
	for _, item := range doc.Nodes {
		node := checkerNode{
			Name:     strings.TrimSpace(item.Name),
			IP:       strings.TrimSpace(item.Host),
			Port:     item.Port,
			Username: strings.TrimSpace(item.User),
			Password: item.Password,
			KeyPath:  strings.TrimSpace(item.KeyFile),
		}
		if node.Port == 0 {
			node.Port = 22
		}
		if node.Name == "" {
			node.Name = node.IP
		}
		if node.IP != "" && node.Username != "" {
			nodes = append(nodes, node)
		}
	}
	if len(nodes) == 0 {
		Skip("nodes.yaml has no SSH node entries for hardware evidence collection")
	}
	return nodes
}

// hardwareChecksForNode 从 checker 的硬件 JSON 结果中取某个节点某个检查项的全部结果。
func hardwareChecksForNode(report *report, node checkerNode, id string) []checkResult {
	if report == nil || report.Hardware == nil {
		return nil
	}
	for _, nodeResult := range report.Hardware.Nodes {
		if nodeResult.Name != node.Name && nodeResult.Name != node.IP {
			continue
		}
		var checks []checkResult
		for _, check := range nodeResult.Checks {
			if check.ID == id {
				check.Node = nodeResult.Name
				checks = append(checks, check)
			}
		}
		return checks
	}
	return nil
}

// requireHardwareChecksForNode 获取节点级硬件检查结果，并断言 checker JSON 中必须包含该检查项。
func requireHardwareChecksForNode(report *report, node checkerNode, id string) []checkResult {
	checks := hardwareChecksForNode(report, node, id)
	Expect(checks).NotTo(BeEmpty(), "hardware report should contain %s for node %s", id, node.Name)
	return checks
}

// hardwareChecksForNodeOrSkip 获取节点级硬件检查结果；如果 checker 因前置失败没执行到该项，则跳过当前用例。
func hardwareChecksForNodeOrSkip(report *report, node checkerNode, id string) []checkResult {
	checks := hardwareChecksForNode(report, node, id)
	if len(checks) == 0 {
		Skip(fmt.Sprintf("%s was not executed because earlier hardware prerequisite failed", id))
	}
	return checks
}

// runSSH 在目标节点上执行一条 SSH 命令并返回 stdout、stderr 和执行错误。
func runSSH(node checkerNode, command string) commandResult {
	return runSSHWithInput(node, command, "")
}

// runSSHWithInput 在目标节点上执行 SSH 命令，并可通过 stdin 传入脚本内容。
func runSSHWithInput(node checkerNode, command string, input string) commandResult {
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
		Expect(err).NotTo(HaveOccurred(), "private key in nodes.yaml should be parseable")
		config.Auth = append(config.Auth, ssh.PublicKeys(signer))
	}
	if node.KeyPath != "" {
		keyData, err := os.ReadFile(node.KeyPath)
		Expect(err).NotTo(HaveOccurred(), "private key path in nodes.yaml should be readable")
		signer, err := ssh.ParsePrivateKey(keyData)
		Expect(err).NotTo(HaveOccurred(), "private key file in nodes.yaml should be parseable")
		config.Auth = append(config.Auth, ssh.PublicKeys(signer))
	}
	if len(config.Auth) == 0 {
		Skip(fmt.Sprintf("node %s has no SSH password/private key in nodes.yaml", node.IP))
	}

	client, err := ssh.Dial("tcp", net.JoinHostPort(node.IP, strconv.Itoa(node.Port)), config)
	Expect(err).NotTo(HaveOccurred(), "should SSH to node %s", node.IP)
	defer client.Close()

	session, err := client.NewSession()
	Expect(err).NotTo(HaveOccurred(), "should create SSH session on node %s", node.IP)
	defer session.Close()

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	session.Stdout = &stdout
	session.Stderr = &stderr
	if input != "" {
		session.Stdin = strings.NewReader(input)
	}
	err = session.Run(command)
	if checkerDebugEnabled() {
		GinkgoWriter.Printf("ssh %s@%s:%d %q\nstdout:\n%s\nstderr:\n%s\n", node.Username, node.IP, node.Port, command, stdout.String(), stderr.String())
	}
	return commandResult{stdout: stdout.String(), stderr: stderr.String(), err: err}
}

// collectH01Evidence 到每个节点上采集驱动模块和 npu-smi 可用性证据。
func collectH01Evidence(nodes []checkerNode) []h01Evidence {
	var evidence []h01Evidence
	for _, node := range nodes {
		lsmod := runSSH(node, "lsmod | grep drv_pcie_host")
		npuSmi := runSSH(node, "command -v npu-smi && npu-smi info")
		evidence = append(evidence, h01Evidence{
			node:         node,
			driverLoaded: lsmod.err == nil && strings.TrimSpace(lsmod.stdout) != "",
			npuSmiOK:     npuSmi.err == nil && strings.Contains(npuSmi.stdout, "NPU"),
			lsmodOutput:  strings.TrimSpace(lsmod.stdout + lsmod.stderr),
			npuSmiOutput: strings.TrimSpace(npuSmi.stdout + npuSmi.stderr),
		})
	}
	return evidence
}

// collectH02Evidence 通过 SSH 到目标节点执行 kubectl，采集 Device Plugin 状态和 NPU 资源注册证据。
func collectH02Evidence(nodes []checkerNode) []h02Evidence {
	var evidence []h02Evidence
	for _, node := range nodes {
		pods := runSSH(node, remoteKubectlCommand("get pods -A -o json"))
		k8sNodes := runSSH(node, remoteKubectlCommand("get nodes -o json"))

		item := h02Evidence{node: node}
		if pods.err != nil || k8sNodes.err != nil {
			evidence = append(evidence, item)
			continue
		}
		item.kubectlOK = true

		var podsJSON map[string]interface{}
		Expect(json.Unmarshal([]byte(pods.stdout), &podsJSON)).To(Succeed(), "kubectl pods output on node %s should be valid JSON", node.Name)
		var nodesJSON map[string]interface{}
		Expect(json.Unmarshal([]byte(k8sNodes.stdout), &nodesJSON)).To(Succeed(), "kubectl nodes output on node %s should be valid JSON", node.Name)

		k8sNode := findK8sNodeForCheckerNode(nodesJSON, node)
		item.k8sNodeName = k8sNode
		if k8sNode != "" {
			item.pluginFound, item.pluginRunning = devicePluginStateForNode(podsJSON, k8sNode)
			item.resourceRegistered = nodeHasNPUResource(nodesJSON, k8sNode)
		}
		evidence = append(evidence, item)
	}
	return evidence
}

// remoteKubectlCommand 生成在目标节点上执行的 kubectl 命令，必要时追加 kubeconfig。
func remoteKubectlCommand(args string) string {
	if strings.TrimSpace(InfernexCheckerConfig.KubeconfigPath) == "" {
		return "kubectl " + args
	}
	return "kubectl --kubeconfig " + shellQuote(InfernexCheckerConfig.KubeconfigPath) + " " + args
}

// shellQuote 对远端 shell 参数做简单单引号转义，避免路径里有特殊字符时命令被拆开。
func shellQuote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", "'\\''") + "'"
}

// collectH03Evidence 到每个节点执行 npu-smi info，判断 NPU 型号是否可解析以及是否为 910 系列。
func collectH03Evidence(nodes []checkerNode) []h03Evidence {
	var evidence []h03Evidence
	for _, node := range nodes {
		result := runSSH(node, "npu-smi info")
		output := strings.TrimSpace(result.stdout + result.stderr)
		evidence = append(evidence, h03Evidence{
			node:       node,
			npuSmiOK:   result.err == nil,
			model910:   result.err == nil && regexp.MustCompile(`(?i)910`).MatchString(output),
			parseModel: result.err == nil && regexp.MustCompile(`(?i)(910|310|ascend|npu)`).MatchString(output),
			output:     output,
		})
	}
	return evidence
}

// collectH04Evidence 统计每个节点上的 /dev/davinci* 设备数量，作为 NPU 可用性证据。
func collectH04Evidence(nodes []checkerNode) []h04Evidence {
	var evidence []h04Evidence
	for _, node := range nodes {
		result := runSSHWithInput(node, "bash -s", h04EvidenceScript)
		values := parseKeyValueLines(result.stdout)
		count, _ := strconv.Atoi(strings.TrimSpace(values["COUNT"]))
		evidence = append(evidence, h04Evidence{
			node:         node,
			hasNPU:       result.err == nil && count > 0,
			npuCount:     count,
			occupiedIDs:  splitCSV(values["OCCUPIED"]),
			unhealthyIDs: splitCSV(values["UNHEALTHY"]),
		})
	}
	return evidence
}

func parseKeyValueLines(output string) map[string]string {
	values := map[string]string{}
	for _, line := range strings.Split(output, "\n") {
		key, value, ok := strings.Cut(strings.TrimSpace(line), "=")
		if ok {
			values[key] = value
		}
	}
	return values
}

func splitCSV(value string) []string {
	var result []string
	for _, item := range strings.Split(value, ",") {
		item = strings.TrimSpace(item)
		if item != "" {
			result = append(result, item)
		}
	}
	return result
}

// collectH05Evidence 检查每个节点上的所有关键 Ascend 驱动路径是否存在（与产品 H-05 检查路径保持一致）。
func collectH05Evidence(nodes []checkerNode) []h05Evidence {
	var evidence []h05Evidence
	for _, node := range nodes {
		result := runSSH(node, "ls /usr/local/dcmi /usr/local/Ascend/driver/lib64 /usr/local/bin/npu-smi /usr/local/Ascend/driver/version.info /etc/ascend_install.info /etc/hccn.conf >/dev/null 2>&1")
		evidence = append(evidence, h05Evidence{node: node, pathsExist: result.err == nil})
	}
	return evidence
}

// collectH06Evidence 读取每个节点的 /etc/hccn.conf，并分析地址、掩码、重复 IP 和网段一致性。
func collectH06Evidence(nodes []checkerNode) []h06Evidence {
	var evidence []h06Evidence
	for _, node := range nodes {
		result := runSSH(node, "cat /etc/hccn.conf")
		item := h06Evidence{node: node, readable: result.err == nil, hccnConfContents: result.stdout}
		if result.err == nil {
			item.hasAddress, item.hasNetmask, item.invalidIP, item.duplicateIP, item.differentSubnet = inspectHCCNConf(result.stdout)
		}
		evidence = append(evidence, item)
	}
	return evidence
}

// inspectHCCNConf 解析 hccn.conf 内容，判断 address/netmask 配置是否完整和合法。
func inspectHCCNConf(content string) (hasAddress bool, hasNetmask bool, invalidIP bool, duplicateIP bool, differentSubnet bool) {
	addresses := map[string]string{}
	netmasks := map[string]string{}
	for _, line := range strings.Split(content, "\n") {
		parts := strings.SplitN(strings.TrimSpace(line), "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		value := strings.TrimSpace(parts[1])
		if strings.HasPrefix(key, "address_") {
			hasAddress = true
			id := strings.TrimPrefix(key, "address_")
			addresses[id] = value
			if net.ParseIP(value) == nil {
				invalidIP = true
			}
		}
		if strings.HasPrefix(key, "netmask_") {
			hasNetmask = true
			id := strings.TrimPrefix(key, "netmask_")
			netmasks[id] = value
			if net.ParseIP(value) == nil {
				invalidIP = true
			}
		}
	}

	seen := map[string]bool{}
	subnets := map[string]bool{}
	for id, address := range addresses {
		if seen[address] {
			duplicateIP = true
		}
		seen[address] = true
		mask := netmasks[id]
		if mask == "" {
			continue
		}
		subnet, ok := subnetKey(address, mask)
		if ok {
			subnets[subnet] = true
		}
	}
	differentSubnet = len(subnets) > 1
	return
}

// subnetKey 根据 IP 和 netmask 计算网段标识，用于判断多张 HCCN 网卡是否在同一网段。
func subnetKey(address, mask string) (string, bool) {
	ip := net.ParseIP(address).To4()
	maskIP := net.ParseIP(mask).To4()
	if ip == nil || maskIP == nil {
		return "", false
	}
	network := net.IPv4(ip[0]&maskIP[0], ip[1]&maskIP[1], ip[2]&maskIP[2], ip[3]&maskIP[3])
	return network.String() + "/" + mask, true
}

// hccnAddresses 从 hccn.conf 中提取 address_N 到 IP 的映射。
func hccnAddresses(content string) map[string]string {
	addresses := map[string]string{}
	for _, line := range strings.Split(content, "\n") {
		parts := strings.SplitN(strings.TrimSpace(line), "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		if strings.HasPrefix(key, "address_") {
			addresses[strings.TrimPrefix(key, "address_")] = strings.TrimSpace(parts[1])
		}
	}
	return addresses
}

// collectH07Evidence 在每个节点执行内嵌 npu_check 脚本，采集单节点 TLS 和 NIC 连通性证据。
func collectH07Evidence(nodes []checkerNode) []h07Evidence {
	var evidence []h07Evidence
	for _, node := range nodes {
		result := runSSHWithInput(node, "bash -s", npuCheckScript)
		output := strings.TrimSpace(result.stdout + result.stderr)
		lower := strings.ToLower(output)
		evidence = append(evidence, h07Evidence{
			node:           node,
			tlsPassed:      strings.Contains(output, "[PASS] TLS check passed"),
			pingPassed:     strings.Contains(output, "[PASS] Ping check passed"),
			tlsFailed:      strings.Contains(output, "[FAIL] TLS check failed") || strings.Contains(lower, "tls state is inconsistent"),
			pingFailed:     strings.Contains(output, "[FAIL] Ping check failed") || strings.Contains(lower, "ping paths failed"),
			hccnToolFailed: strings.Contains(lower, "hccn_tool") && (strings.Contains(lower, "not found") || strings.Contains(lower, "failed to get") || strings.Contains(lower, "unknown")),
			output:         output,
		})
	}
	return evidence
}

// collectH08Evidence 通过复用 H-04 脚本采集每个节点的 idle NPU 数量，作为慢卡检测触发条件的前置证据。
func collectH08Evidence(nodes []checkerNode) []h08Evidence {
	var evidence []h08Evidence
	for _, node := range nodes {
		result := runSSHWithInput(node, "bash -s", h04EvidenceScript)
		values := parseKeyValueLines(result.stdout)
		count, _ := strconv.Atoi(strings.TrimSpace(values["COUNT"]))
		occupied := splitCSV(values["OCCUPIED"])
		idle := count - len(occupied)
		if idle < 0 {
			idle = 0
		}
		evidence = append(evidence, h08Evidence{node: node, idleNPUCount: idle})
	}
	return evidence
}

// collectH10Evidence 采集跨节点 HCCS 检查前置证据，包括 910 节点、重复 IP、TLS 和跨节点 ping。
func collectH10Evidence(nodes []checkerNode) h10Evidence {
	h03 := collectH03Evidence(nodes)
	h06 := collectH06Evidence(nodes)
	h06ByNode := map[string]h06Evidence{}
	for _, item := range h06 {
		h06ByNode[item.node.Name] = item
		h06ByNode[item.node.IP] = item
	}

	ev := h10Evidence{}
	for _, item := range h03 {
		h06Item, ok := h06ByNode[item.node.Name]
		if !ok {
			h06Item = h06ByNode[item.node.IP]
		}
		if item.npuSmiOK && item.model910 && h06Item.readable && h06Item.hasAddress && h06Item.hasNetmask && !h06Item.invalidIP && !h06Item.duplicateIP && !h06Item.differentSubnet {
			ev.eligibleNodes = append(ev.eligibleNodes, item.node)
		}
	}
	ev.singleNode = len(ev.eligibleNodes) < 2
	if ev.singleNode {
		return ev
	}

	ev.duplicateIP = hasDuplicateCrossNodeIP(ev.eligibleNodes, h06ByNode)
	ev.tlsInconsistent = hasCrossNodeTLSInconsistency(ev.eligibleNodes)
	if !ev.duplicateIP && !ev.tlsInconsistent {
		ev.crossPingCollected = true
		ev.crossPingFailed = hasCrossNodePingFailure(ev.eligibleNodes, h06ByNode)
		ev.crossConnectivity = !ev.crossPingFailed
	}
	return ev
}

// hasDuplicateCrossNodeIP 判断不同节点的 hccn.conf 中是否存在重复的 HCCN IP。
func hasDuplicateCrossNodeIP(nodes []checkerNode, h06ByNode map[string]h06Evidence) bool {
	seen := map[string]string{}
	for _, node := range nodes {
		h06Item, ok := h06ByNode[node.Name]
		if !ok {
			h06Item = h06ByNode[node.IP]
		}
		for _, address := range hccnAddresses(h06Item.hccnConfContents) {
			if owner, exists := seen[address]; exists && owner != node.Name {
				return true
			}
			seen[address] = node.Name
		}
	}
	return false
}

// hasCrossNodeTLSInconsistency 判断多个节点的 HCCN TLS 状态是否一致。
func hasCrossNodeTLSInconsistency(nodes []checkerNode) bool {
	states := map[string]bool{}
	for _, node := range nodes {
		for _, state := range collectNodeTLSStates(node) {
			states[state] = true
		}
	}
	return len(states) > 1
}

// collectNodeTLSStates 逐卡执行 hccn_tool -tls -g，归一化得到当前节点的 TLS 状态列表。
func collectNodeTLSStates(node checkerNode) []string {
	result := runSSH(node, `for id in $(ls /dev/davinci* 2>/dev/null | grep -oP 'davinci\K\d+' | sort -n); do out=$(hccn_tool -i "$id" -tls -g 2>&1); if echo "$out" | grep -qi 'tls switch\[1\]'; then echo enable; elif echo "$out" | grep -qi 'tls switch\[0\]'; then echo disable; elif echo "$out" | grep -qi 'no certificate'; then echo disable; else echo unknown; fi; done`)
	var states []string
	for _, line := range strings.Split(result.stdout, "\n") {
		state := strings.TrimSpace(line)
		if state != "" {
			states = append(states, state)
		}
	}
	return states
}

// hasCrossNodePingFailure 通过 hccn_tool -ping 检查不同节点间 HCCN IP 是否互通。
func hasCrossNodePingFailure(nodes []checkerNode, h06ByNode map[string]h06Evidence) bool {
	for _, src := range nodes {
		srcIDs := collectNodeDeviceIDs(src)
		for _, dst := range nodes {
			if src.Name == dst.Name && src.IP == dst.IP {
				continue
			}
			h06Item, ok := h06ByNode[dst.Name]
			if !ok {
				h06Item = h06ByNode[dst.IP]
			}
			for _, srcID := range srcIDs {
				for _, dstIP := range hccnAddresses(h06Item.hccnConfContents) {
					if !crossNodePingOK(src, srcID, dstIP) {
						return true
					}
				}
			}
		}
	}
	return false
}

// collectNodeDeviceIDs 从 /dev/davinci* 设备文件中提取当前节点的 NPU 卡 ID。
func collectNodeDeviceIDs(node checkerNode) []string {
	result := runSSH(node, `ls /dev/davinci* 2>/dev/null | grep -oP 'davinci\K\d+' | sort -n`)
	var ids []string
	for _, line := range strings.Split(result.stdout, "\n") {
		id := strings.TrimSpace(line)
		if id != "" {
			ids = append(ids, id)
		}
	}
	return ids
}

// crossNodePingOK 从源节点指定 NPU 卡向目标 HCCN IP 发起 ping，判断跨节点连通性。
func crossNodePingOK(src checkerNode, srcID string, dstIP string) bool {
	command := fmt.Sprintf(`for attempt in 1 2 3; do result=$(hccn_tool -i %s -ping -g address %s pkt 128 2>&1); if echo "$result" | grep -q "100.00%% packet loss" || echo "$result" | grep -q "Command execute failed"; then sleep 1; else echo "$result"; exit 0; fi; done; echo "$result"; exit 1`, srcID, dstIP)
	result := runSSH(src, command)
	return result.err == nil
}

// findK8sNodeForCheckerNode 根据 nodes.yaml 中的节点名或 IP 匹配 Kubernetes Node 名称。
func findK8sNodeForCheckerNode(nodesJSON map[string]interface{}, node checkerNode) string {
	for _, raw := range listItems(nodesJSON) {
		item, ok := raw.(map[string]interface{})
		if !ok {
			continue
		}
		metadata, _ := item["metadata"].(map[string]interface{})
		name, _ := metadata["name"].(string)
		if name == node.Name || name == node.IP {
			return name
		}
		status, _ := item["status"].(map[string]interface{})
		for _, rawAddress := range listItemsFrom(status, "addresses") {
			address, ok := rawAddress.(map[string]interface{})
			if !ok {
				continue
			}
			if value, _ := address["address"].(string); value == node.IP || value == node.Name {
				return name
			}
		}
	}
	return ""
}

// devicePluginStateForNode 判断指定 K8s 节点上是否存在并运行 Ascend Device Plugin Pod。
func devicePluginStateForNode(podsJSON map[string]interface{}, k8sNode string) (bool, bool) {
	found := false
	running := false
	for _, raw := range listItems(podsJSON) {
		pod, ok := raw.(map[string]interface{})
		if !ok {
			continue
		}
		metadata, _ := pod["metadata"].(map[string]interface{})
		name, _ := metadata["name"].(string)
		if !strings.Contains(strings.ToLower(name), "ascend-device-plugin") {
			continue
		}
		spec, _ := pod["spec"].(map[string]interface{})
		nodeName, _ := spec["nodeName"].(string)
		if nodeName != k8sNode {
			continue
		}
		found = true
		status, _ := pod["status"].(map[string]interface{})
		if phase, _ := status["phase"].(string); phase == "Running" {
			running = true
		}
	}
	return found, running
}

// nodeHasNPUResource 判断指定 K8s Node 的 capacity/allocatable 中是否注册了 NPU 资源。
func nodeHasNPUResource(nodesJSON map[string]interface{}, k8sNode string) bool {
	for _, raw := range listItems(nodesJSON) {
		item, ok := raw.(map[string]interface{})
		if !ok {
			continue
		}
		metadata, _ := item["metadata"].(map[string]interface{})
		name, _ := metadata["name"].(string)
		if name != k8sNode {
			continue
		}
		status, _ := item["status"].(map[string]interface{})
		if hasNPUResourceMap(status, "allocatable") || hasNPUResourceMap(status, "capacity") {
			return true
		}
	}
	return false
}

// hasNPUResourceMap 在 capacity 或 allocatable 字段中查找 NPU 相关资源键。
func hasNPUResourceMap(parent map[string]interface{}, field string) bool {
	values, _ := parent[field].(map[string]interface{})
	for key := range values {
		lower := strings.ToLower(key)
		if strings.Contains(lower, "huawei.com") || strings.Contains(lower, "ascend") || strings.Contains(lower, "npu") {
			return true
		}
	}
	return false
}

// listItems 从 kubectl JSON 输出中取 items 列表。
func listItems(root map[string]interface{}) []interface{} {
	return listItemsFrom(root, "items")
}

// listItemsFrom 从 JSON 对象的指定字段中读取数组。
func listItemsFrom(root map[string]interface{}, field string) []interface{} {
	if root == nil {
		return nil
	}
	items, _ := root[field].([]interface{})
	return items
}

// assertH06EvidenceFailureOrSkip 如果环境命中 H-06 负向场景，则断言 checker 报 failed 且有建议；否则跳过。
func assertH06EvidenceFailureOrSkip(report *report, evidence []h06Evidence, match func(h06Evidence) bool, messagePart string, skipReason string) {
	matched := filterEvidence(evidence, match)
	if len(matched) == 0 {
		Skip(skipReason)
	}
	for _, item := range matched {
		checks := hardwareChecksForNodeOrSkip(report, item.node, "H-06")
		check := checks[0]
		Expect(check.Status).To(Equal(statusFailed), "H-06 should fail on node %s", item.node.Name)
		Expect(check.Message).To(ContainSubstring(messagePart))
		Expect(check.Suggestion).NotTo(BeEmpty(), fmt.Sprintf("%s should include a suggestion", check.ID))
	}
}

// h07ChecksOrSkip 获取 H-07 检查结果；如果 checker 未执行到 H-07，则跳过当前场景。
func h07ChecksOrSkip(report *report, node checkerNode) []checkResult {
	checks := hardwareChecksForNode(report, node, "H-07")
	if len(checks) == 0 {
		Skip("H-07 was not executed because earlier hardware prerequisite failed or node is non-910")
	}
	return checks
}

// assertH07NodeFailure 断言某个节点的 H-07 失败；如果 H-07 未执行则跳过。
func assertH07NodeFailure(report *report, node checkerNode, messagePart string) {
	checks := h07ChecksOrSkip(report, node)
	for _, check := range checks {
		if check.Status == statusFailed && strings.Contains(check.Message, messagePart) {
			Expect(check.Suggestion).NotTo(BeEmpty(), fmt.Sprintf("%s should include a suggestion", check.ID))
			return
		}
	}
	Fail(fmt.Sprintf("H-07 on node %s should fail with message containing %q", node.Name, messagePart))
}

// assertCrossNodeFailure 断言跨节点检查结果中存在预期失败信息和修复建议。
func assertCrossNodeFailure(messagePart string, reports ...*report) {
	for _, report := range reports {
		if report == nil || report.Hardware == nil {
			continue
		}
		for _, check := range report.Hardware.CrossNodes {
			if check.Status == statusFailed && strings.Contains(check.Message, messagePart) {
				Expect(check.Suggestion).NotTo(BeEmpty(), fmt.Sprintf("%s should include a suggestion", check.ID))
				return
			}
		}
	}
	Fail(fmt.Sprintf("H-10 should fail with message containing %q", messagePart))
}

// prepareHardwareReport 统一准备硬件测试前置数据：读取节点并运行 infernex-checker hardware。
func prepareHardwareReport() ([]checkerNode, *report) {
	nodes := loadCheckerNodes()
	hardwareRun := runInfernexCheckCached("hardware", false, false)
	Expect(hardwareRun.err).NotTo(HaveOccurred(),
		"infernex-check hardware should run successfully, stderr: %s", hardwareRun.stderr)
	hardwareReport := hardwareRun.report
	Expect(hardwareReport).NotTo(BeNil(), "hardware command should write JSON report")
	Expect(hardwareReport.Hardware).NotTo(BeNil(), "JSON report should contain hardware result")
	return nodes, hardwareReport
}

var _ = Describe("Infernex software pre-flight checks", Label("infernex_checker", "precheck"), func() {
	Describe("Hardware Layer H-01~H-06", Ordered, Label("hardware"), func() {
		var hardwareReport *report
		var nodes []checkerNode
		var h01Env []h01Evidence
		var h02Env []h02Evidence
		var h03Env []h03Evidence
		var h04Env []h04Evidence
		var h05Env []h05Evidence
		var h06Env []h06Evidence

		BeforeAll(func() {
			nodes, hardwareReport = prepareHardwareReport()
			h01Env = collectH01Evidence(nodes)
			h02Env = collectH02Evidence(nodes)
			h03Env = collectH03Evidence(nodes)
			h04Env = collectH04Evidence(nodes)
			h05Env = collectH05Evidence(nodes)
			h06Env = collectH06Evidence(nodes)
		})

		It("ic_hw_001-【H-01】NPU驱动模块已加载且npu-smi可用", Label("P0"), func() {
			matched := filterEvidence(h01Env, func(item h01Evidence) bool {
				return item.driverLoaded && item.npuSmiOK
			})
			if len(matched) == 0 {
				Skip("no node currently satisfies H-01 positive precondition: driver loaded and npu-smi available")
			}
			for _, item := range matched {
				checks := requireHardwareChecksForNode(hardwareReport, item.node, "H-01")
				check := checks[0]
				Expect(check.Status).To(Equal(statusPassed), "H-01 should pass on node %s", item.node.Name)
				Expect(check.Message).To(ContainSubstring("NPU driver and firmware installed"))
			}
		})

		It("ic_hw_002-【H-01】lsmod无drv_pcie_host输出时报失败", Label("P0"), func() {
			matched := filterEvidence(h01Env, func(item h01Evidence) bool {
				return !item.driverLoaded
			})
			if len(matched) == 0 {
				Skip("no node currently satisfies H-01 missing-driver negative precondition; do not unload NPU driver for e2e")
			}
			for _, item := range matched {
				checks := requireHardwareChecksForNode(hardwareReport, item.node, "H-01")
				check := checks[0]
				Expect(check.Status).To(Equal(statusFailed), "H-01 should fail when driver module is missing on node %s", item.node.Name)
				Expect(check.Message).To(ContainSubstring("no lsmod output"))
				Expect(check.Suggestion).NotTo(BeEmpty())
			}
		})

		It("ic_hw_003-【H-01】驱动模块加载但npu-smi执行失败时报失败", Label("P1"), func() {
			matched := filterEvidence(h01Env, func(item h01Evidence) bool {
				return item.driverLoaded && !item.npuSmiOK
			})
			if len(matched) == 0 {
				Skip("no node currently satisfies H-01 npu-smi failure negative precondition; do not break npu-smi for e2e")
			}
			for _, item := range matched {
				checks := requireHardwareChecksForNode(hardwareReport, item.node, "H-01")
				check := checks[0]
				Expect(check.Status).To(Equal(statusFailed), "H-01 should fail when npu-smi is unavailable on node %s", item.node.Name)
				Expect(check.Message).To(ContainSubstring("npu-smi"))
				Expect(check.Suggestion).NotTo(BeEmpty())
			}
		})

		It("ic_hw_004-【H-02】ascend-device-plugin pod running且NPU资源已注册", Label("P0"), func() {
			matched := filterEvidence(h02Env, func(item h02Evidence) bool {
				return item.kubectlOK && item.pluginFound && item.pluginRunning && item.resourceRegistered
			})
			if len(matched) == 0 {
				Skip("no node currently satisfies H-02 positive precondition: device plugin running and NPU resource registered")
			}
			for _, item := range matched {
				checks := hardwareChecksForNodeOrSkip(hardwareReport, item.node, "H-02")
				check := checks[0]
				Expect(check.Status).To(Equal(statusPassed), "H-02 should pass on node %s", item.node.Name)
				Expect(check.Message).To(ContainSubstring("Ascend Device Plugin Running"))
				Expect(check.Message).To(ContainSubstring("NPU resource registered"))
			}
		})

		It("ic_hw_005-【H-02】ascend-device-plugin pod不存在时报失败", Label("P0"), func() {
			matched := filterEvidence(h02Env, func(item h02Evidence) bool {
				return item.kubectlOK && !item.pluginFound
			})
			if len(matched) == 0 {
				Skip("no node currently satisfies H-02 missing-plugin negative precondition; do not delete plugin pods for e2e")
			}
			for _, item := range matched {
				checks := hardwareChecksForNodeOrSkip(hardwareReport, item.node, "H-02")
				check := checks[0]
				Expect(check.Status).To(Equal(statusFailed), "H-02 should fail when device plugin is missing on node %s", item.node.Name)
				Expect(check.Message).To(ContainSubstring("not found"))
				Expect(check.Suggestion).NotTo(BeEmpty())
			}
		})

		It("ic_hw_006-【H-02】ascend-device-plugin pod存在但非running状态时报失败", Label("P1"), func() {
			matched := filterEvidence(h02Env, func(item h02Evidence) bool {
				return item.kubectlOK && item.pluginFound && !item.pluginRunning
			})
			if len(matched) == 0 {
				Skip("no node currently satisfies H-02 non-running-plugin negative precondition; do not stop plugin pods for e2e")
			}
			for _, item := range matched {
				checks := hardwareChecksForNodeOrSkip(hardwareReport, item.node, "H-02")
				check := checks[0]
				Expect(check.Status).To(Equal(statusFailed), "H-02 should fail when device plugin is not running on node %s", item.node.Name)
				Expect(check.Message).To(ContainSubstring("not in Running"))
				Expect(check.Suggestion).NotTo(BeEmpty())
			}
		})

		It("ic_hw_007-【H-02】ascend-device-pluging pod running但NPU资源未注册时报失败", Label("P1"), func() {
			matched := filterEvidence(h02Env, func(item h02Evidence) bool {
				return item.kubectlOK && item.pluginFound && item.pluginRunning && !item.resourceRegistered
			})
			if len(matched) == 0 {
				Skip("no node currently satisfies H-02 resource-not-registered negative precondition; do not mutate node capacity for e2e")
			}
			for _, item := range matched {
				checks := hardwareChecksForNodeOrSkip(hardwareReport, item.node, "H-02")
				check := checks[0]
				Expect(check.Status).To(Equal(statusFailed), "H-02 should fail when NPU resource is not registered on node %s", item.node.Name)
				Expect(check.Message).To(ContainSubstring("NPU resource not registered"))
				Expect(check.Suggestion).NotTo(BeEmpty())
			}
		})

		It("ic_hw_008-【H-03】NPU型号为910系列时检查通过", Label("P0"), func() {
			matched := filterEvidence(h03Env, func(item h03Evidence) bool {
				return item.npuSmiOK && item.model910
			})
			if len(matched) == 0 {
				Skip("no node currently satisfies H-03 910-series positive precondition")
			}
			for _, item := range matched {
				checks := hardwareChecksForNode(hardwareReport, item.node, "H-03")
				if len(checks) == 0 {
					Skip("H-03 was not executed because earlier hardware prerequisite failed")
				}
				check := checks[0]
				Expect(check.Status).To(Equal(statusPassed), "H-03 should pass on 910-series node %s", item.node.Name)
				Expect(check.Message).To(MatchRegexp(`NPU model: .*910`))
			}
		})

		It("ic_hw_009-【H-03】NPU为非910系列时返回WARNING并跳过H-06~H-10", Label("P1"), func() {
			matched := filterEvidence(h03Env, func(item h03Evidence) bool {
				return item.npuSmiOK && item.parseModel && !item.model910
			})
			if len(matched) == 0 {
				Skip("current environment is 910 series; non-910 scenario requires a dedicated environment or mock test")
			}
			for _, item := range matched {
				checks := hardwareChecksForNode(hardwareReport, item.node, "H-03")
				if len(checks) == 0 {
					Skip("H-03 was not executed because earlier hardware prerequisite failed")
				}
				check := checks[0]
				Expect(check.Status).To(Equal(statusWarning), "H-03 should warn on non-910 node %s", item.node.Name)
				Expect(check.Message).To(ContainSubstring("non-910"))
				// H-06 should not appear in this node's checks regardless of whether
				// HCCS was explicitly skipped (H-05 passed, non-910 skip path) or
				// execution terminated at H-05 (e.g. 310 node missing /etc/hccn.conf)
				h06Checks := hardwareChecksForNode(hardwareReport, item.node, "H-06")
				Expect(h06Checks).To(BeEmpty(), "H-06 should not be executed for non-910 node %s", item.node.Name)
			}
		})

		It("ic_hw_010-【H-03】npu-smi info无法解析型号时报失败", Label("P2"), func() {
			matched := filterEvidence(h03Env, func(item h03Evidence) bool {
				return item.npuSmiOK && !item.parseModel
			})
			if len(matched) == 0 {
				Skip("current environment can parse NPU model; malformed npu-smi output is covered by mock test")
			}
			for _, item := range matched {
				checks := hardwareChecksForNode(hardwareReport, item.node, "H-03")
				if len(checks) == 0 {
					Skip("H-03 was not executed because earlier hardware prerequisite failed")
				}
				check := checks[0]
				Expect(check.Status).To(Equal(statusFailed), "H-03 should fail when model cannot be parsed on node %s", item.node.Name)
				Expect(check.Message).To(ContainSubstring("Unable to parse"))
				Expect(check.Suggestion).NotTo(BeEmpty())
			}
		})

		It("ic_hw_011-【H-04】以Info方式上报NPU可用数量（部分可用）", Label("P1"), func() {
			matched := filterEvidence(h04Env, func(item h04Evidence) bool {
				return item.hasNPU && item.npuCount > 0 && len(item.occupiedIDs) < item.npuCount
			})
			if len(matched) == 0 {
				Skip("no node currently has partially available NPU devices for H-04 info reporting")
			}
			for _, item := range matched {
				checks := hardwareChecksForNode(hardwareReport, item.node, "H-04")
				if len(checks) == 0 {
					Skip("H-04 was not executed because earlier hardware prerequisite failed")
				}
				check := checks[0]
				Expect(check.Status).To(Equal(statusInfo), "H-04 should report info on node %s", item.node.Name)
				Expect(check.Message).To(MatchRegexp(`NPU available: \d+/\d+`))
			}
		})

		It("ic_hw_012-【H-04】所有NPU被占用时上报0可用", Label("P2"), func() {
			matched := filterEvidence(h04Env, func(item h04Evidence) bool {
				return item.hasNPU && item.npuCount > 0 && len(item.occupiedIDs) == item.npuCount
			})
			if len(matched) == 0 {
				Skip("current environment does not have all NPU cards occupied; all-occupied scenario requires real occupied NPU devices")
			}
			for _, item := range matched {
				checks := hardwareChecksForNode(hardwareReport, item.node, "H-04")
				if len(checks) == 0 {
					Skip("H-04 was not executed because earlier hardware prerequisite failed")
				}
				check := checks[0]
				Expect(check.Status).To(Equal(statusWarning), "H-04 should report warning when all NPUs occupied on node %s", item.node.Name)
				Expect(check.Message).To(ContainSubstring(fmt.Sprintf("NPU available: 0/%d", item.npuCount)))
				Expect(check.Suggestion).NotTo(BeEmpty())
			}
		})

		It("ic_hw_013-【H-04】存在unhealthy NPU时消息包含unhealthy列表", Label("P2"), func() {
			matched := filterEvidence(h04Env, func(item h04Evidence) bool {
				return item.hasNPU && len(item.unhealthyIDs) > 0
			})
			if len(matched) == 0 {
				Skip("current environment has no unhealthy NPU; unhealthy hardware scenario requires real unhealthy NPU devices")
			}
			for _, item := range matched {
				checks := hardwareChecksForNode(hardwareReport, item.node, "H-04")
				if len(checks) == 0 {
					Skip("H-04 was not executed because earlier hardware prerequisite failed")
				}
				check := checks[0]
				Expect(check.Status).To(Equal(statusInfo), "H-04 should report info on node %s", item.node.Name)
				Expect(strings.ToLower(check.Message)).To(ContainSubstring("unhealthy"))
				for _, id := range item.unhealthyIDs {
					Expect(check.Message).To(ContainSubstring(id))
				}
			}
		})

		It("ic_hw_014-【H-05】所有关键主机文件和目录均存在", Label("P0"), func() {
			matched := filterEvidence(h05Env, func(item h05Evidence) bool {
				return item.pathsExist
			})
			if len(matched) == 0 {
				Skip("no node currently satisfies H-05 positive precondition: key host paths exist")
			}
			for _, item := range matched {
				checks := hardwareChecksForNodeOrSkip(hardwareReport, item.node, "H-05")
				check := checks[0]
				Expect(check.Status).To(Equal(statusPassed), "H-05 should pass on node %s", item.node.Name)
				Expect(check.Message).To(ContainSubstring("Key host files and directories are complete"))
			}
		})

		It("ic_hw_015-【H-05】缺少任一关键路径时报失败并列出缺失项", Label("P0"), func() {
			matched := filterEvidence(h05Env, func(item h05Evidence) bool {
				return !item.pathsExist
			})
			if len(matched) == 0 {
				Skip("current environment passes H-05; do not remove host files for a negative e2e case")
			}
			for _, item := range matched {
				checks := hardwareChecksForNodeOrSkip(hardwareReport, item.node, "H-05")
				check := checks[0]
				Expect(check.Status).To(Equal(statusFailed), "H-05 should fail when key paths are missing on node %s", item.node.Name)
				Expect(check.Message).To(ContainSubstring("Missing paths or files"))
				Expect(check.Suggestion).NotTo(BeEmpty())
			}
		})

		It("ic_hw_016-【H-06】hccn.conf配置完整正确时检查通过", Label("P0"), func() {
			matched := filterEvidence(h06Env, func(item h06Evidence) bool {
				return item.readable && item.hasAddress && item.hasNetmask && !item.invalidIP && !item.duplicateIP && !item.differentSubnet
			})
			if len(matched) == 0 {
				Skip("no node currently satisfies H-06 positive precondition: hccn.conf valid")
			}
			for _, item := range matched {
				checks := hardwareChecksForNodeOrSkip(hardwareReport, item.node, "H-06")
				check := checks[0]
				Expect(check.Status).To(Equal(statusPassed), "H-06 should pass on node %s", item.node.Name)
				Expect(check.Message).To(ContainSubstring("hccn.conf configuration is correct"))
			}
		})

		It("ic_hw_017-【H-06】hccn.conf缺少address_N时报失败", Label("P0"), func() {
			assertH06EvidenceFailureOrSkip(hardwareReport, h06Env, func(item h06Evidence) bool {
				return item.readable && !item.hasAddress
			}, "Missing IP configuration", "current environment passes H-06; missing address_N scenario requires mock or dedicated fault environment")
		})

		It("ic_hw_018-【H-06】hccn.conf缺少netmask_N时报失败", Label("P0"), func() {
			assertH06EvidenceFailureOrSkip(hardwareReport, h06Env, func(item h06Evidence) bool {
				return item.readable && !item.hasNetmask
			}, "Missing netmask configuration", "current environment passes H-06; missing netmask_N scenario requires mock or dedicated fault environment")
		})

		It("ic_hw_019-【H-06】hccn.conf中IP格式无效时报失败", Label("P1"), func() {
			assertH06EvidenceFailureOrSkip(hardwareReport, h06Env, func(item h06Evidence) bool {
				return item.readable && item.invalidIP
			}, "invalid IP format", "current environment passes H-06; invalid IP scenario requires mock or dedicated fault environment")
		})

		It("ic_hw_020-【H-06】hccn.conf中存在重复IP时报失败", Label("P1"), func() {
			assertH06EvidenceFailureOrSkip(hardwareReport, h06Env, func(item h06Evidence) bool {
				return item.readable && item.duplicateIP
			}, "duplicate IP", "current environment passes H-06; duplicate IP scenario requires mock or dedicated fault environment")
		})

		It("ic_hw_021-【H-06】无法读取hccn.conf文件时报失败", Label("P1"), func() {
			assertH06EvidenceFailureOrSkip(hardwareReport, h06Env, func(item h06Evidence) bool {
				return !item.readable
			}, "Failed to read /etc/hccn.conf", "current environment can read hccn.conf; read failure scenario requires mock or dedicated fault environment")
		})
	})

	Describe("Hardware Layer H-07~H-10", Ordered, Label("hardware"), func() {
		var hardwareReport *report
		var nodes []checkerNode
		var h07Env []h07Evidence
		var h08Env []h08Evidence
		var h10Env h10Evidence

		BeforeAll(func() {
			nodes, hardwareReport = prepareHardwareReport()
			h07Env = collectH07Evidence(nodes)
			h08Env = collectH08Evidence(nodes)
			h10Env = collectH10Evidence(nodes)
		})

		It("ic_hw_022-【H-07】单节点NIC间TLS一致且连通性正常", Label("P0"), func() {
			matched := filterEvidence(h07Env, func(item h07Evidence) bool {
				return item.tlsPassed && item.pingPassed
			})
			if len(matched) == 0 {
				Skip("no node currently satisfies H-07 positive precondition: TLS consistent and NIC-to-NIC ping normal")
			}
			for _, item := range matched {
				checks := h07ChecksOrSkip(hardwareReport, item.node)
				Expect(checks).To(ContainElement(SatisfyAll(
					WithTransform(func(c checkResult) checkStatus { return c.Status }, Equal(statusPassed)),
					WithTransform(func(c checkResult) string { return c.Message }, ContainSubstring("Step 1")),
				)))
				Expect(checks).To(ContainElement(SatisfyAll(
					WithTransform(func(c checkResult) checkStatus { return c.Status }, Equal(statusPassed)),
					WithTransform(func(c checkResult) string { return c.Message }, ContainSubstring("Step 2")),
				)))
			}
		})

		It("ic_hw_023-【H-07】TLS状态不一致时Step1报失败且Step2不执行", Label("P0"), func() {
			matched := filterEvidence(h07Env, func(item h07Evidence) bool {
				return item.tlsFailed
			})
			if len(matched) == 0 {
				Skip("current environment has consistent TLS state; TLS inconsistency requires mock or dedicated fault environment")
			}
			for _, item := range matched {
				assertH07NodeFailure(hardwareReport, item.node, "TLS switch states are inconsistent")
			}
		})

		It("ic_hw_024-【H-07】hccn_tool执行失败时Step1报失败", Label("P1"), func() {
			matched := filterEvidence(h07Env, func(item h07Evidence) bool {
				return item.hccnToolFailed
			})
			if len(matched) == 0 {
				Skip("current environment can run hccn_tool; hccn_tool failure requires mock or dedicated fault environment")
			}
			for _, item := range matched {
				assertH07NodeFailure(hardwareReport, item.node, "Failed to get TLS state")
			}
		})

		It("ic_hw_025-【H-07】NIC间ping失败时Step2报失败", Label("P1"), func() {
			matched := filterEvidence(h07Env, func(item h07Evidence) bool {
				return item.pingFailed
			})
			if len(matched) == 0 {
				Skip("current environment has normal single-node NIC connectivity; ping failure requires mock or dedicated fault environment")
			}
			for _, item := range matched {
				assertH07NodeFailure(hardwareReport, item.node, "NIC pairs failed ping")
			}
		})

		It("ic_hw_026-【H-08】单节点无慢卡", Label("P0"), func() {
			matched := filterEvidence(h08Env, func(item h08Evidence) bool {
				return item.idleNPUCount >= 2
			})
			if len(matched) == 0 {
				Skip("no node currently has >= 2 idle NPUs for H-08 slow card detection")
			}
			found := false
			for _, item := range matched {
				for _, check := range hardwareChecksForNode(hardwareReport, item.node, "H-08") {
					if check.Status == statusPassed {
						Expect(check.Message).To(ContainSubstring("All NPU cards have balanced communication performance"))
						found = true
					}
				}
			}
			if !found {
				Skip("current environment did not produce H-08 passed result; slow card or detection failure may have occurred")
			}
		})

		It("ic_hw_027-【H-08】检测到慢卡时报Warning", Label("P0"), func() {
			matched := filterEvidence(h08Env, func(item h08Evidence) bool {
				return item.idleNPUCount >= 2
			})
			if len(matched) == 0 {
				Skip("no node currently has >= 2 idle NPUs for H-08 slow card detection")
			}
			found := false
			for _, item := range matched {
				for _, check := range hardwareChecksForNode(hardwareReport, item.node, "H-08") {
					if check.Status == statusWarning && strings.Contains(check.Message, "Slow NPU card detected") {
						Expect(check.Suggestion).NotTo(BeEmpty())
						found = true
					}
				}
			}
			if !found {
				Skip("current environment has no slow cards; slow card detection requires real slow NPU hardware")
			}
		})

		It("ic_hw_028-【H-08】idle NPU不足2个时跳过慢卡检测", Label("P1"), func() {
			matched := filterEvidence(h08Env, func(item h08Evidence) bool {
				return item.idleNPUCount < 2
			})
			if len(matched) == 0 {
				Skip("all nodes currently have >= 2 idle NPUs; skip scenario requires fewer idle NPUs")
			}
			for _, item := range matched {
				checks := hardwareChecksForNode(hardwareReport, item.node, "H-08")
				if len(checks) == 0 {
					Skip("H-08 was not executed because earlier hardware prerequisite failed")
				}
				check := checks[0]
				Expect(check.Status).To(Equal(statusSkipped))
				Expect(check.Message).To(ContainSubstring("Slow card detection skipped"))
				Expect(check.Message).To(ContainSubstring("idle NPUs"))
			}
		})

		It("ic_hw_029-【H-08】msprof执行失败时报Warning", Label("P1"), func() {
			matched := filterEvidence(h08Env, func(item h08Evidence) bool {
				return item.idleNPUCount >= 2
			})
			if len(matched) == 0 {
				Skip("no node currently has >= 2 idle NPUs for H-08 slow card detection")
			}
			found := false
			for _, item := range matched {
				for _, check := range hardwareChecksForNode(hardwareReport, item.node, "H-08") {
					if check.Status == statusWarning && strings.Contains(check.Message, "Slow card detection failed") {
						found = true
					}
				}
			}
			if !found {
				Skip("current environment msprof ran successfully; msprof failure requires unavailable msprof binary or unreachable image")
			}
		})

		It("ic_hw_030-【H-08】communication JSON解析失败时报Warning", Label("P1"), func() {
			matched := filterEvidence(h08Env, func(item h08Evidence) bool {
				return item.idleNPUCount >= 2
			})
			if len(matched) == 0 {
				Skip("no node currently has >= 2 idle NPUs for H-08 slow card detection")
			}
			found := false
			for _, item := range matched {
				for _, check := range hardwareChecksForNode(hardwareReport, item.node, "H-08") {
					if check.Status == statusWarning &&
						strings.Contains(check.Message, "Slow card detection failed") &&
						strings.Contains(check.Message, "metrics") {
						found = true
					}
				}
			}
			if !found {
				Skip("current environment did not produce JSON parse failure; this scenario requires msprof output with invalid communication metrics")
			}
		})

		It("ic_hw_031-【H-09】多节点快慢卡检测", Label("P0"), func() {
			var h09Check *checkResult
			for i := range hardwareReport.Hardware.CrossNodes {
				if hardwareReport.Hardware.CrossNodes[i].ID == "H-09" {
					c := hardwareReport.Hardware.CrossNodes[i]
					h09Check = &c
					break
				}
			}
			if h09Check == nil {
				Skip("H-09 was not executed in this run; cross-node slow card check requires hardware mode with multi-node setup")
			}
			if h09Check.Status == statusSkipped {
				Expect(h09Check.Message).To(ContainSubstring("No nodes have slow card metrics"))
			} else {
				Expect(h09Check.Status).To(Equal(statusInfo))
				Expect(h09Check.Message).To(ContainSubstring("Cross-node communication metrics"))
			}
		})

		It("ic_hw_032-【H-10】多节点跨节点IP唯一、TLS一致、跨机连通性", Label("P0"), func() {
			checks := hardwareReport.Hardware.CrossNodes
			if h10Env.singleNode {
				Skip("current nodes.yaml has fewer than two 910-series nodes passing single-node checks; add another NPU node to run H-10 positive e2e")
			}
			if h10Env.duplicateIP {
				Skip("current environment has duplicate cross-node NIC IPs; H-10 positive connectivity requires globally unique NIC IPs")
			}
			if h10Env.tlsInconsistent {
				Skip("current environment has inconsistent cross-node TLS; H-10 positive connectivity requires consistent TLS")
			}
			if !h10Env.crossConnectivity {
				Skip("current environment does not have normal cross-node connectivity; H-10 positive connectivity requires successful cross-node ping")
			}
			Expect(checks).To(ContainElement(SatisfyAll(
				WithTransform(func(c checkResult) checkStatus { return c.Status }, Equal(statusPassed)),
				WithTransform(func(c checkResult) string { return c.Message }, ContainSubstring("Step 1")),
			)))
			Expect(checks).To(ContainElement(SatisfyAll(
				WithTransform(func(c checkResult) checkStatus { return c.Status }, Equal(statusPassed)),
				WithTransform(func(c checkResult) string { return c.Message }, ContainSubstring("Step 2")),
			)))
			Expect(checks).To(ContainElement(SatisfyAll(
				WithTransform(func(c checkResult) checkStatus { return c.Status }, Equal(statusPassed)),
				WithTransform(func(c checkResult) string { return c.Message }, ContainSubstring("Step 3")),
			)))
		})

		It("ic_hw_033-【H-10】仅1个节点通过单节点检查时跳过跨节点检查", Label("P1"), func() {
			checks := hardwareReport.Hardware.CrossNodes
			if !h10Env.singleNode {
				Skip("current environment has enough nodes for H-10 cross-node check; single-node skip scenario requires a one-node nodes.yaml")
			}
			if len(h10Env.eligibleNodes) != 1 {
				Skip("current environment does not have exactly one 910-series node passing single-node checks")
			}
			Expect(checks).NotTo(BeEmpty())

			var h10Check *checkResult
			for i := range checks {
				if checks[i].ID == "H-10" {
					h10Check = &checks[i]
					break
				}
			}
			Expect(h10Check).NotTo(BeNil(), "H-10 check should exist in CrossNodes")
			Expect(h10Check.Status).To(Equal(statusSkipped))
			Expect(h10Check.Message).To(ContainSubstring("Only 1 910-series node"))
		})

		It("ic_hw_034-【H-10】跨节点IP重复时Step1报失败且Step2、Step3不执行", Label("P0"), func() {
			if !h10Env.duplicateIP {
				Skip("current environment has unique cross-node IPs; duplicate IP scenario requires mock or dedicated fault environment")
			}
			assertCrossNodeFailure("Duplicate IPs across nodes", hardwareReport)
		})

		It("ic_hw_035-【H-10】跨节点TLS不一致时Step2报失败且Step3不执行", Label("P0"), func() {
			if !h10Env.tlsInconsistent {
				Skip("current environment has consistent cross-node TLS; TLS inconsistency requires mock or dedicated fault environment")
			}
			assertCrossNodeFailure("TLS configuration inconsistent", hardwareReport)
		})

		It("ic_hw_036-【H-10】跨节点ping失败时Step3报失败", Label("P1"), func() {
			checks := hardwareReport.Hardware.CrossNodes
			if h10Env.singleNode {
				Skip("current nodes.yaml has fewer than two 910-series nodes passing single-node checks; cross-node ping failure scenario requires at least two eligible nodes")
			}
			if h10Env.duplicateIP {
				Skip("current environment fails H-10 Step1 with duplicate cross-node IPs; Step3 ping failure scenario requires unique cross-node IPs")
			}
			if h10Env.tlsInconsistent {
				Skip("current environment fails H-10 Step2 with inconsistent TLS; Step3 ping failure scenario requires consistent cross-node TLS")
			}
			if !h10Env.crossPingCollected {
				Skip("current environment did not collect cross-node ping evidence; Step3 ping failure scenario requires cross-node ping evidence")
			}
			if !h10Env.crossPingFailed {
				Skip("current environment has normal cross-node connectivity; Step3 ping failure scenario requires real cross-node ping failure")
			}
			Expect(checks).To(ContainElement(SatisfyAll(
				WithTransform(func(c checkResult) checkStatus { return c.Status }, Equal(statusPassed)),
				WithTransform(func(c checkResult) string { return c.Message }, ContainSubstring("Step 1")),
			)))
			Expect(checks).To(ContainElement(SatisfyAll(
				WithTransform(func(c checkResult) checkStatus { return c.Status }, Equal(statusPassed)),
				WithTransform(func(c checkResult) string { return c.Message }, ContainSubstring("Step 2")),
			)))
			assertCrossNodeFailure("Cross-node connectivity anomaly", hardwareReport)
		})
	})
})

// filterEvidence returns all items from evidence where match returns true.
func filterEvidence[T any](evidence []T, match func(T) bool) []T {
	var matched []T
	for _, item := range evidence {
		if match(item) {
			matched = append(matched, item)
		}
	}
	return matched
}

// indexByNode indexes evidence items by node identity string using nodeOf to extract the node.
func indexByNode[T any](evidence []T, nodeOf func(T) checkerNode) map[string]T {
	result := map[string]T{}
	for _, item := range evidence {
		result[nodeIdentity(nodeOf(item))] = item
	}
	return result
}
