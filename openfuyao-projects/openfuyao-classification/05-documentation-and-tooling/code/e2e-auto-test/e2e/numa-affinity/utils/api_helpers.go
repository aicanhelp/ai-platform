package utils

import (
	"fmt"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
)

// CallWebsiteAPIPut invokes the Volcano config website REST API through curled SSH.
func CallWebsiteAPIPut(sshClient *executor.SSHExecutor, endpoint string, policy string) error {
	// The volcano-config-website is expected to be reachable via the cluster network.
	// Since we are running commands on a node (via SSH), we use the Service ClusterIP or DNS.
	url := fmt.Sprintf("http://volcano-config-website.volcano-system.svc.cluster.local:8080%s", endpoint)

	cmd := fmt.Sprintf("curl -s -X PUT -H 'Content-Type: application/json' -d '{\"policy\": \"%s\"}' %s", policy, url)
	out, err := ExecStdout(sshClient, cmd)
	if err != nil {
		return fmt.Errorf("API invoke failed. cmd: %s output: %s error: %w", cmd, out, err)
	}

	return nil
}

func GetWebsiteAPIPolicy(sshClient *executor.SSHExecutor, endpoint string) (string, error) {
	url := fmt.Sprintf("http://volcano-config-website.volcano-system.svc.cluster.local:8080%s", endpoint)

	cmd := fmt.Sprintf("curl -s -X GET %s", url)
	out, err := ExecStdout(sshClient, cmd)
	if err != nil {
		return "", fmt.Errorf("API invoke failed. cmd: %s output: %s error: %w", cmd, out, err)
	}

	return out, nil
}
