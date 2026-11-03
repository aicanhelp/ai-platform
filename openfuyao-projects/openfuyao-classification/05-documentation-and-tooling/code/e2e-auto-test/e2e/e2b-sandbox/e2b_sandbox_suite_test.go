/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 */

package e2b_sandbox

import (
	"context"
	"fmt"
	"net/http"
	"testing"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/e2b-sandbox/utils"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/env"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
)

var (
	k8sClient *k8s.K8SClient
	ctx       context.Context
	cancel    context.CancelFunc
	apiClient *APIClient
	sgName    string
)

func init() {
	if err := env.LoadEnv(".env"); err != nil {
		GinkgoWriter.Printf("WARNING: failed to load .env: %v\n", err)
	}
}

var _ = BeforeSuite(func() {
	var err error
	ctx, cancel = context.WithTimeout(context.Background(), 30*time.Minute)
	DeferCleanup(func() {
		if cancel != nil {
			cancel()
		}
	})

	k8sClient, err = k8s.NewK8SClientFromLocalKubeconfig()
	Expect(err).NotTo(HaveOccurred(), "k8s client 应当成功创建")

	// Find OpenSandbox Server pod and get its IP
	serverNs := EnvOr("E2B_SANDBOX_NAMESPACE", ServerNamespace)
	pods, err := k8sClient.ListPods(ctx, serverNs, metav1.ListOptions{LabelSelector: ServerPodLabel})
	Expect(err).NotTo(HaveOccurred(), "列出 opensandbox-server pod 应当成功")
	if len(pods.Items) == 0 {
		Skip(fmt.Sprintf("未发现 opensandbox-server pod (selector %q in %q)，跳过用例", ServerPodLabel, serverNs))
	}

	var podIP string
	for i := range pods.Items {
		pod := &pods.Items[i]
		if pod.Status.Phase == "Running" && pod.Status.PodIP != "" {
			podIP = pod.Status.PodIP
			break
		}
	}
	if podIP == "" {
		Skip("未发现 Running 状态的 opensandbox-server pod，跳过用例")
	}

	baseURL := fmt.Sprintf("http://%s:%d", podIP, ServerPort)
	apiClient = NewAPIClient(baseURL)
	Expect(apiClient).NotTo(BeNil())

	By("等待 OpenSandbox Server API 可达")
	Eventually(func(g Gomega) {
		resp, err := http.Get(fmt.Sprintf("%s/v1/sandboxes?pageSize=1", baseURL))
		g.Expect(err).NotTo(HaveOccurred())
		if resp != nil {
			resp.Body.Close()
		}
	}, 60*time.Second, 3*time.Second).Should(Succeed(), "OpenSandbox Server API 应当可达")

	// Create SandboxGroup if not exists
	sgName = EnvOr("E2B_SANDBOX_GROUP", "e2b-e2e-test")
	By(fmt.Sprintf("确保 SandboxGroup %s 就绪", sgName))
	Expect(EnsureSandboxGroup(ctx, k8sClient, sgName, serverNs)).To(Succeed(), "SandboxGroup 应当就绪")

	// Wait for agent pods Running
	By("等待 agent pod 就绪")
	Eventually(func(g Gomega) {
		agentPods, lerr := k8sClient.ListPods(ctx, serverNs, metav1.ListOptions{LabelSelector: AgentPodLabel})
		g.Expect(lerr).NotTo(HaveOccurred())
		running := 0
		for i := range agentPods.Items {
			if agentPods.Items[i].Status.Phase == "Running" {
				running++
			}
		}
		g.Expect(running).To(BeNumerically(">=", 1), "应当至少有一个 Running 的 agent pod")
	}, 3*time.Minute, 5*time.Second).Should(Succeed())

	By("e2b-sandbox suite 就绪")
})

func TestE2BSandbox(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "E2B Sandbox Suite")
}
