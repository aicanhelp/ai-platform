package utils

import (
	"fmt"
	"strings"

	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
)

func buildHAProxyTCPConfig(masterIPs []string, vipPort string) string {
	return fmt.Sprintf(`global
    daemon
    maxconn 10000

defaults
    mode tcp
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    retries 3
    balance roundrobin

frontend main_frontend
    bind *:%s
    default_backend backend_servers

backend backend_servers
    mode tcp
    option tcp-check
    balance roundrobin
    server server1 %s:6443 check inter 10s
    server server2 %s:6443 check inter 10s
    server server3 %s:6443 check inter 10s
`, vipPort, masterIPs[0], masterIPs[1], masterIPs[2])
}

// SetupHAProxyTCPProxyOnNode configures and starts haproxy on the LB node.
func SetupHAProxyTCPProxyOnNode(lbNode config.NodeInfo, masterIPs []string, vipPort string) error {
	if len(masterIPs) < 3 {
		return fmt.Errorf("masterIPs 至少需要 3 个节点，当前: %d", len(masterIPs))
	}
	if strings.TrimSpace(lbNode.IP) == "" {
		return fmt.Errorf("LB 节点 IP 为空")
	}

	cfg := buildHAProxyTCPConfig(masterIPs, vipPort)
	cmd := fmt.Sprintf("set -euxo pipefail; mkdir -p /etc/haproxy; command -v haproxy >/dev/null 2>&1; cat > /etc/haproxy/haproxy.cfg <<'EOF'\n%sEOF\nhaproxy -c -f /etc/haproxy/haproxy.cfg; systemctl enable haproxy; systemctl restart haproxy; systemctl is-active haproxy; ss -lntp | grep :%s", cfg, vipPort)
	result, err := ExecuteCommandOnNode(lbNode, cmd)
	if err != nil {
		diagCmd := "set +e; echo '--- command -v haproxy ---'; command -v haproxy 2>&1; echo '--- haproxy version ---'; haproxy -v 2>&1; echo '--- haproxy config check ---'; haproxy -c -f /etc/haproxy/haproxy.cfg 2>&1; echo '--- systemctl status haproxy ---'; systemctl status haproxy --no-pager -l 2>&1; echo '--- journalctl haproxy ---'; journalctl -u haproxy -n 80 --no-pager 2>&1; echo '--- haproxy cfg ---'; ls -l /etc/haproxy/haproxy.cfg 2>&1; sed -n '1,200p' /etc/haproxy/haproxy.cfg 2>&1"
		diagResult, _ := ExecuteCommandOnNode(lbNode, diagCmd)
		if result != nil {
			diagOut := ""
			if diagResult != nil {
				diagOut = strings.TrimSpace(diagResult.Stdout + "\n" + diagResult.Stderr)
			}
			return fmt.Errorf("在节点 %s 配置 haproxy 失败: %w, stdout: %s, stderr: %s, diagnostics: %s", lbNode.IP, err, strings.TrimSpace(result.Stdout), strings.TrimSpace(result.Stderr), diagOut)
		}
		diagOut := ""
		if diagResult != nil {
			diagOut = strings.TrimSpace(diagResult.Stdout + "\n" + diagResult.Stderr)
		}
		return fmt.Errorf("在节点 %s 配置 haproxy 失败: %w, diagnostics: %s", lbNode.IP, err, diagOut)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("haproxy 启动失败: stdout=%s, stderr=%s", strings.TrimSpace(result.Stdout), strings.TrimSpace(result.Stderr))
	}
	return nil
}

// CleanupHAProxyOnNode stops haproxy and removes generated config.
func CleanupHAProxyOnNode(lbNode config.NodeInfo) error {
	if strings.TrimSpace(lbNode.IP) == "" {
		return nil
	}
	cmd := "systemctl stop haproxy || true; rm -f /etc/haproxy/haproxy.cfg"
	_, err := ExecuteCommandOnNode(lbNode, cmd)
	return err
}
