package utils

import (
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/exec"
	"strings"
)

type clusterNode struct {
	Name        string
	Hostname    string
	InternalIPs []string
}

// detectTestExecutorNode 把测试进程所在机器对应到集群 Node 名（kubernetes.io/hostname）。
// 预测模型/tokenizer 走 hostPath，EPP 必须调度到持有该路径的节点。
func detectTestExecutorNode() (string, error) {
	nodes, err := listClusterNodes()
	if err != nil {
		return "", err
	}
	host, _ := os.Hostname()
	matched := matchK8sNode(host, localNonLoopbackIPs(), nodes)
	if matched == "" {
		return "", fmt.Errorf("无法识别测试执行节点：本机 hostname/IP 与集群 Node 均不匹配")
	}
	return matched, nil
}

func matchK8sNode(hostname string, localIPs map[string]struct{}, nodes []clusterNode) string {
	host := strings.TrimSpace(hostname)
	short := host
	if i := strings.Index(host, "."); i > 0 {
		short = host[:i]
	}
	for _, n := range nodes {
		if host != "" && (strings.EqualFold(n.Name, host) || strings.EqualFold(n.Hostname, host) ||
			strings.EqualFold(n.Name, short) || strings.EqualFold(n.Hostname, short)) {
			return n.Name
		}
	}
	if len(localIPs) == 0 {
		return ""
	}
	for _, n := range nodes {
		for _, ip := range n.InternalIPs {
			if _, ok := localIPs[ip]; ok {
				return n.Name
			}
		}
	}
	return ""
}

func localNonLoopbackIPs() map[string]struct{} {
	out := map[string]struct{}{}
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return out
	}
	for _, addr := range addrs {
		ipNet, ok := addr.(*net.IPNet)
		if !ok || ipNet.IP == nil || ipNet.IP.IsLoopback() {
			continue
		}
		ip := ipNet.IP.To4()
		if ip == nil {
			continue
		}
		out[ip.String()] = struct{}{}
	}
	return out
}

func listClusterNodes() ([]clusterNode, error) {
	cmd := exec.Command("kubectl", "get", "nodes", "-o", "json")
	out, err := cmd.Output()
	if err != nil {
		stderr := ""
		if ee, ok := err.(*exec.ExitError); ok {
			stderr = strings.TrimSpace(string(ee.Stderr))
		}
		return nil, fmt.Errorf("kubectl get nodes 失败: %w, stderr=%s", err, stderr)
	}
	var list struct {
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
	if err := json.Unmarshal(out, &list); err != nil {
		return nil, fmt.Errorf("解析 kubectl get nodes JSON 失败: %w", err)
	}
	nodes := make([]clusterNode, 0, len(list.Items))
	for _, item := range list.Items {
		n := clusterNode{
			Name:     item.Metadata.Name,
			Hostname: item.Metadata.Labels["kubernetes.io/hostname"],
		}
		for _, addr := range item.Status.Addresses {
			if addr.Type == "InternalIP" && addr.Address != "" {
				n.InternalIPs = append(n.InternalIPs, addr.Address)
			}
		}
		nodes = append(nodes, n)
	}
	if len(nodes) == 0 {
		return nil, fmt.Errorf("集群没有 Node")
	}
	return nodes, nil
}
