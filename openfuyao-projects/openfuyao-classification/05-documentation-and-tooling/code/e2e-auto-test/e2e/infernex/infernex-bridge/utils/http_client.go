package utils

import (
	"fmt"
	"os"
	"os/exec"
	"time"
)

// PortForwardService starts kubectl port-forward in background; call stop() to terminate.
func PortForwardService(ns, svc string, localPort, remotePort int) (stop func(), err error) {
	kc := localKubeconfig()
	logPath, err := os.CreateTemp("", "pf-log-*")
	if err != nil {
		return nil, err
	}
	cmd := exec.Command("kubectl", "--kubeconfig", kc, "-n", ns, "port-forward",
		fmt.Sprintf("svc/%s", svc), fmt.Sprintf("%d:%d", localPort, remotePort))
	cmd.Stdout = logPath
	cmd.Stderr = logPath
	if err := cmd.Start(); err != nil {
		logPath.Close()
		os.Remove(logPath.Name())
		return nil, err
	}
	time.Sleep(2 * time.Second)
	return func() {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		logPath.Close()
		os.Remove(logPath.Name())
	}, nil
}
