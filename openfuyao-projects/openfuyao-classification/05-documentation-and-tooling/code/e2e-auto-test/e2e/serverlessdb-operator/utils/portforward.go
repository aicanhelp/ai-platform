package utils

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/portforward"
	"k8s.io/client-go/transport/spdy"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
)

const proxyPort = 9090

const proxyScriptTemplate = `use IO::Socket::INET;
use IO::Select;
$| = 1;
my $srv = IO::Socket::INET->new(
    LocalAddr => '127.0.0.1',
    LocalPort => %d,
    Listen    => 50,
    ReuseAddr => 1,
) or die "listen: $!";
open my $pf, '>', '/tmp/e2e-proxy.pid' or die "pid: $!";
print $pf $$;
close $pf;
while (my $c = $srv->accept()) {
    my $t = IO::Socket::INET->new(PeerAddr => '%s', PeerPort => %d);
    unless ($t) { close $c; next }
    my $sel = IO::Select->new($c, $t);
    while (my @r = $sel->can_read()) {
        for my $rh (@r) {
            my $wh = ($rh == $c) ? $t : $c;
            my $buf;
            my $n = $rh->sysread($buf, 65536);
            if (!$n) { $sel->remove($c, $t); close $c; close $t; goto NEXT }
            $wh->syswrite($buf);
        }
    }
    NEXT:
}
`

// StartPortForward establishes a port-forward from a local random port to
// remotePort on the first Ready pod matching labelSelector in ns.
//
// Because some operators bind their HTTP server to the pod IP only (not
// 0.0.0.0), kubelet port-forward (which connects to localhost inside the
// pod) would fail. To work around this, a lightweight perl TCP proxy is
// started inside the pod bridging 127.0.0.1:proxyPort → podIP:remotePort,
// and the port-forward targets the proxy port instead.
func StartPortForward(k8sClient *k8s.K8SClient, ctx context.Context, ns, labelSelector string, remotePort int) (int, func(), error) {
	pods, err := k8sClient.ListPods(ctx, ns, metav1.ListOptions{LabelSelector: labelSelector})
	if err != nil {
		return 0, nil, fmt.Errorf("list pods with selector %q: %w", labelSelector, err)
	}
	var podName, podIP string
	for i := range pods.Items {
		if IsPodReady(&pods.Items[i]) {
			podName = pods.Items[i].Name
			podIP = pods.Items[i].Status.PodIP
			break
		}
	}
	if podName == "" {
		return 0, nil, fmt.Errorf("no ready pod found with selector %q in namespace %q", labelSelector, ns)
	}

	if err := startInPodProxy(ctx, ns, podName, podIP, remotePort); err != nil {
		return 0, nil, err
	}

	transport, upgrader, err := spdy.RoundTripperFor(k8sClient.RestConfig)
	if err != nil {
		return 0, nil, fmt.Errorf("create spdy round tripper: %w", err)
	}
	urlStr := fmt.Sprintf("%s/api/v1/namespaces/%s/pods/%s/portforward", k8sClient.RestConfig.Host, ns, podName)
	parsedURL, err := url.Parse(urlStr)
	if err != nil {
		return 0, nil, fmt.Errorf("parse portforward url: %w", err)
	}
	dialer := spdy.NewDialer(upgrader, &http.Client{Transport: transport}, "POST", parsedURL)

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, nil, fmt.Errorf("find free local port: %w", err)
	}
	localPort := ln.Addr().(*net.TCPAddr).Port
	_ = ln.Close()

	stopCh := make(chan struct{})
	readyCh := make(chan struct{}, 1)
	pf, err := portforward.New(dialer, []string{fmt.Sprintf("%d:%d", localPort, proxyPort)}, stopCh, readyCh, io.Discard, io.Discard)
	if err != nil {
		return 0, nil, fmt.Errorf("create port forwarder: %w", err)
	}

	go func() {
		_ = pf.ForwardPorts()
	}()

	select {
	case <-readyCh:
	case <-time.After(30 * time.Second):
		close(stopCh)
		killInPodProxy(ns, podName)
		return 0, nil, fmt.Errorf("port forward to %s/%s:%d did not become ready in 30s", ns, podName, proxyPort)
	}

	// SPDY tunnel ready ≠ proxy listening; verify end-to-end connectivity.
	if err := waitForLocalPort(localPort, 60*time.Second); err != nil {
		close(stopCh)
		killInPodProxy(ns, podName)
		return 0, nil, err
	}

	stop := func() {
		select {
		case <-stopCh:
		default:
			close(stopCh)
		}
		killInPodProxy(ns, podName)
	}
	return localPort, stop, nil
}

func startInPodProxy(ctx context.Context, ns, podName, podIP string, targetPort int) error {
	killInPodProxy(ns, podName)

	scriptFile, err := writeProxyScript(podIP, targetPort)
	if err != nil {
		return err
	}
	defer os.Remove(scriptFile)

	cpCmd := exec.CommandContext(ctx, "kubectl", "cp", scriptFile, fmt.Sprintf("%s/%s:/tmp/e2e-proxy.pl", ns, podName))
	if out, err := cpCmd.CombinedOutput(); err != nil {
		return fmt.Errorf("copy proxy script to pod: %w (output: %s)", err, string(out))
	}

	execCmd := exec.CommandContext(ctx, "kubectl", "exec", "-n", ns, podName, "--",
		"bash", "-c", "nohup perl /tmp/e2e-proxy.pl > /tmp/e2e-proxy.log 2>&1 & echo $!",
	)
	if out, err := execCmd.CombinedOutput(); err != nil {
		return fmt.Errorf("start proxy in pod: %w (output: %s)", err, string(out))
	}

	time.Sleep(500 * time.Millisecond)

	verifyCmd := exec.CommandContext(ctx, "kubectl", "exec", "-n", ns, podName, "--",
		"bash", "-c", fmt.Sprintf("exec 3<>/dev/tcp/127.0.0.1/%d && exec 3>&-", proxyPort),
	)
	if out, err := verifyCmd.CombinedOutput(); err != nil {
		return fmt.Errorf("TCP proxy not reachable in pod: %w (output: %s)", err, string(out))
	}
	return nil
}

func writeProxyScript(podIP string, targetPort int) (string, error) {
	script := fmt.Sprintf(proxyScriptTemplate, proxyPort, podIP, targetPort)
	f, err := os.CreateTemp("", "e2e-proxy-*.pl")
	if err != nil {
		return "", fmt.Errorf("create temp proxy script: %w", err)
	}
	if _, err := f.WriteString(script); err != nil {
		f.Close()
		os.Remove(f.Name())
		return "", fmt.Errorf("write proxy script: %w", err)
	}
	name := f.Name()
	_ = f.Close()
	return name, nil
}

func killInPodProxy(ns, podName string) {
	cmd := exec.Command("kubectl", "exec",
		"-n", ns, podName, "--",
		"bash", "-c", "test -f /tmp/e2e-proxy.pid && kill $(cat /tmp/e2e-proxy.pid) 2>/dev/null; rm -f /tmp/e2e-proxy.pid /tmp/e2e-proxy.pl /tmp/e2e-proxy.log",
	)
	_ = cmd.Run()
}

func waitForLocalPort(localPort int, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		conn, err := net.DialTimeout("tcp", fmt.Sprintf("127.0.0.1:%d", localPort), 2*time.Second)
		if err == nil {
			_ = conn.Close()
			return nil
		}
		time.Sleep(2 * time.Second)
	}
	return fmt.Errorf("local port %d did not accept TCP connections within %v", localPort, timeout)
}

// IsPodReady reports whether the pod has the Ready condition set to True.
func IsPodReady(p *corev1.Pod) bool {
	for _, c := range p.Status.Conditions {
		if c.Type == corev1.PodReady && c.Status == corev1.ConditionTrue {
			return true
		}
	}
	return false
}
