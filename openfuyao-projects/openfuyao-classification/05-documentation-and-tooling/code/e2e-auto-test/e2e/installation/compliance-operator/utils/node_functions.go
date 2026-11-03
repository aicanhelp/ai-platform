package utils

import (
	"encoding/json"
	"fmt"
	"strings"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
)

// 节点角色判定与集群对账（取值优先级与降级口径见 README「配置复用与节点角色判定」）：
//
//  1. COMPLIANCE_NODE{i}_ROLE 显式指定（最高优先；与集群实测角色不一致时告警不阻塞）；
//  2. entry_node `kubectl get nodes` 角色标签为权威源（node-role.kubernetes.io/master|control-plane
//     存在即 master，否则 worker），并据其对账 .env 节点集合——任一方向对不上即停（防错配）；
//  3. kubectl 不可用时按主机名前缀降级推断（bootstrap/master*/m<数字>→master，worker*/w<数字>→worker），
//     仍无法判定的节点报错即停（禁止静默猜测）。
//
// 前缀规则对本集群（m1/m2/m3/n1）与 .env.template 示例拓扑（bootstrap/m1/w1/w2/w3）的覆盖说明：
// kubectl 权威源对任意命名均正确；前缀降级仅覆盖 m*/w*/master*/worker*/bootstrap 形态，
// 其余命名（如 n1）须依赖 kubectl 或显式 COMPLIANCE_NODE{i}_ROLE。

// k8sNodeInfo kubectl get nodes 解析出的单个节点事实（名称/InternalIP 集合/实测角色）。
type k8sNodeInfo struct {
	Name   string
	IPs    []string
	Master bool
}

// ResolveNodeRoles 为 cfg.Nodes 中未显式指定角色的节点判定 master/worker，并对账集群实际节点。
// 必须在 entry SSH 执行器建立后、任何 Masters()/Workers() 使用前调用一次。
func (c *Config) ResolveNodeRoles(entry *executor.SSHExecutor) (warnings []string, err error) {
	k8sNodes, kubectlErr := queryK8sNodes(entry)
	if kubectlErr != nil {
		warnings = append(warnings, fmt.Sprintf(
			"entry_node kubectl 不可用（%s），已跳过节点集合对账，角色改按主机名前缀推断", kubectlErr))
	}

	if kubectlErr == nil {
		// 节点集合对账：.env 节点集合与 kubectl 实际节点必须一一对应（不一致即停）
		matched, unmatchedEnv, unmatchedK8s := matchEnvNodesToK8s(c.Nodes, k8sNodes)
		if len(unmatchedEnv) > 0 || len(unmatchedK8s) > 0 {
			var b strings.Builder
			b.WriteString("节点对账失败：.env 节点集合与 kubectl get nodes 实际节点不一致（防错配即停）")
			for _, i := range unmatchedEnv {
				fmt.Fprintf(&b, "\n  - .env 节点 %s（IP %s）在集群中无对应节点", c.Nodes[i].Hostname, c.Nodes[i].IP)
			}
			for _, n := range unmatchedK8s {
				fmt.Fprintf(&b, "\n  - 集群节点 %s（InternalIP %s）未出现在 .env 节点清单", n.Name, strings.Join(n.IPs, "/"))
			}
			b.WriteString("\n请核对环境变量中的节点清单（TEST_NODE2~5_*，或 COMPLIANCE_NODE{2..5}_* 显式覆盖）与被测集群实际节点清单")
			return warnings, fmt.Errorf("%s", b.String())
		}
		// 角色填充：kubectl 实测角色（显式指定优先，不一致告警）；matched 与 c.Nodes 下标一一对应
		for i := range c.Nodes {
			observed := roleOfK8sNode(matched[i])
			switch {
			case c.Nodes[i].Role == "":
				if observed == "" {
					return warnings, fmt.Errorf("节点 %s（IP %s）实测角色为空：kubectl 角色标签缺失且未显式指定 COMPLIANCE_NODE%d_ROLE",
						c.Nodes[i].Hostname, c.Nodes[i].IP, c.Nodes[i].Slot)
				}
				c.Nodes[i].Role = observed
			case c.Nodes[i].Role != observed:
				warnings = append(warnings, fmt.Sprintf(
					"节点 %s（IP %s）显式角色 %s 与集群实测角色 %s 不一致，按显式值执行（README 口径）",
					c.Nodes[i].Hostname, c.Nodes[i].IP, c.Nodes[i].Role, observed))
			}
			if c.Nodes[i].Hostname == "" {
				c.Nodes[i].Hostname = matched[i].Name // 以集群实测节点名回填，日志展示可读
			}
		}
		return warnings, nil
	}

	// kubectl 降级路径：主机名前缀推断，仍无法判定即停
	var undetermined []string
	for i := range c.Nodes {
		if c.Nodes[i].Role != "" {
			continue
		}
		if inferred := inferRoleByHostname(c.Nodes[i].Hostname); inferred != "" {
			c.Nodes[i].Role = inferred
		} else {
			undetermined = append(undetermined, fmt.Sprintf(
				"TEST_NODE%d（IP %s）：kubectl 不可用且主机名无 m*/w*/master*/worker*/bootstrap 特征，请显式设置 COMPLIANCE_NODE%d_ROLE",
				c.Nodes[i].Slot, c.Nodes[i].IP, c.Nodes[i].Slot))
		}
	}
	if len(undetermined) > 0 {
		return warnings, fmt.Errorf("节点角色无法判定（禁止静默猜测）：%s", strings.Join(undetermined, "；"))
	}
	return warnings, nil
}

// queryK8sNodes 经 entry SSH 执行 kubectl get nodes -o json 并解析节点事实。
func queryK8sNodes(entry *executor.SSHExecutor) ([]k8sNodeInfo, error) {
	out, err := ExecTrim(entry, "kubectl get nodes -o json")
	if err != nil {
		return nil, err
	}
	var doc struct {
		Items []struct {
			Metadata struct {
				Name   string            `json:"name"`
				Labels map[string]string `json:"labels"`
			} `json:"metadata"`
			Status struct {
				Addresses []struct {
					Type    string `json:"type"`
					Address string `json:"address"`
				} `json:"addresses"`
			} `json:"status"`
		} `json:"items"`
	}
	if err := json.Unmarshal([]byte(out), &doc); err != nil {
		return nil, fmt.Errorf("解析 kubectl get nodes 输出失败: %w", err)
	}
	nodes := make([]k8sNodeInfo, 0, len(doc.Items))
	for _, item := range doc.Items {
		n := k8sNodeInfo{Name: item.Metadata.Name, Master: k8sNodeIsMaster(item.Metadata.Labels)}
		for _, a := range item.Status.Addresses {
			if a.Type == "InternalIP" && a.Address != "" {
				n.IPs = append(n.IPs, a.Address)
			}
		}
		nodes = append(nodes, n)
	}
	return nodes, nil
}

// k8sNodeIsMaster 按角色标签判定 master（control-plane 为新版标签，两者任一存在即算）。
func k8sNodeIsMaster(labels map[string]string) bool {
	_, master := labels["node-role.kubernetes.io/master"]
	_, controlPlane := labels["node-role.kubernetes.io/control-plane"]
	return master || controlPlane
}

// matchEnvNodesToK8s 将 .env 节点逐一匹配到集群节点（先按 InternalIP、再按主机名=节点名），
// 返回与 envNodes 等长的匹配结果（未匹配位为 nil）与两个方向的对账缺口。
func matchEnvNodesToK8s(envNodes []NodeConfig, k8sNodes []k8sNodeInfo) (matched []*k8sNodeInfo, unmatchedEnv []int, unmatchedK8s []k8sNodeInfo) {
	byIP := map[string]*k8sNodeInfo{}
	byName := map[string]*k8sNodeInfo{}
	for i := range k8sNodes {
		for _, ip := range k8sNodes[i].IPs {
			byIP[ip] = &k8sNodes[i]
		}
		byName[k8sNodes[i].Name] = &k8sNodes[i]
	}
	claimed := map[*k8sNodeInfo]bool{}
	matched = make([]*k8sNodeInfo, len(envNodes))
	for i, n := range envNodes {
		m := byIP[n.IP]
		if m == nil && n.Hostname != "" {
			m = byName[n.Hostname]
		}
		if m == nil || claimed[m] {
			unmatchedEnv = append(unmatchedEnv, i)
			continue
		}
		claimed[m] = true
		matched[i] = m
	}
	for i := range k8sNodes {
		if !claimed[&k8sNodes[i]] {
			unmatchedK8s = append(unmatchedK8s, k8sNodes[i])
		}
	}
	return matched, unmatchedEnv, unmatchedK8s
}

func roleOfK8sNode(n *k8sNodeInfo) string {
	if n.Master {
		return RoleMaster
	}
	return RoleWorker
}

// inferRoleByHostname 主机名前缀降级推断（kubectl 不可用时使用）：
// bootstrap/master*/m<数字> → master；worker*/w<数字> → worker；其余返回空串（交由上层判停）。
func inferRoleByHostname(hostname string) string {
	h := strings.ToLower(strings.TrimSpace(hostname))
	if h == "" {
		return ""
	}
	mDigits := len(h) >= 2 && h[0] == 'm' && isAllDigits(h[1:])
	wDigits := len(h) >= 2 && h[0] == 'w' && isAllDigits(h[1:])
	switch {
	case h == "bootstrap" || strings.HasPrefix(h, "master") || mDigits:
		return RoleMaster
	case strings.HasPrefix(h, "worker") || wDigits:
		return RoleWorker
	}
	return ""
}

func isAllDigits(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}
