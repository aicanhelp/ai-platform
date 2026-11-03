package utils

import (
	"context"
	"fmt"
	"time"

	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"

	esutils "gitcode.com/openFuyao/e2e-auto-test/e2e/elastic-scaler/utils"
)

const defaultLWSGroupSize int64 = 2

// WaitLeaderWorkerSetReady waits until LeaderWorkerSet status.readyReplicas >= spec.replicas (or exists when skip NPU).
// size is the expected leaderWorkerTemplate.size (pods per group), used for logging only.
func WaitLeaderWorkerSetReady(ctx context.Context, dyn dynamic.Interface, fitID, ns, name string, size int64, timeout time.Duration) {
	LogPhaseBegin(fitID, fmt.Sprintf("leaderworkerset %s templateSize=%d skipNPU=%v", name, size, SkipNPUWorkloadE2E()))
	defer LogPhaseEnd(fitID, fmt.Sprintf("leaderworkerset %s", name))
	if SkipNPUWorkloadE2E() {
		Eventually(func(g Gomega) {
			_, err := dyn.Resource(esutils.LeaderWorkerSetGVR).Namespace(ns).Get(ctx, name, metav1.GetOptions{})
			g.Expect(err).NotTo(HaveOccurred())
		}).WithTimeout(timeout).WithPolling(5 * time.Second).Should(Succeed())
		return
	}
	Eventually(func(g Gomega) {
		obj, err := dyn.Resource(esutils.LeaderWorkerSetGVR).Namespace(ns).Get(ctx, name, metav1.GetOptions{})
		g.Expect(err).NotTo(HaveOccurred(), "get LWS %s/%s", ns, name)
		want, _, _ := unstructured.NestedInt64(obj.Object, "spec", "replicas")
		if want <= 0 {
			want = 1
		}
		ready, found, err := unstructured.NestedInt64(obj.Object, "status", "readyReplicas")
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(found).To(BeTrueBecause("LWS %s status.readyReplicas", name))
		if ready < want {
			LogPollThrottled(fitID, "lws-"+name, defaultPollLogInterval,
				"LWS %s readyReplicas=%d want>=%d", name, ready, want)
		}
		g.Expect(ready).To(BeNumerically(">=", want), "LWS %s readyReplicas", name)
	}).WithTimeout(timeout).WithPolling(10 * time.Second).Should(Succeed())
	LogStep(fitID, "LWS %s readyReplicas>=%d (spec.replicas)", name, 1)
}

func WaitOwnerLeaderWorkerSetWorkloads(ctx context.Context, dyn dynamic.Interface, fitID, ns, owner string, suffixes []string, groupSize int64, timeout time.Duration) {
	for _, suffix := range suffixes {
		name := DeployName(owner, suffix)
		WaitLeaderWorkerSetReady(ctx, dyn, fitID, ns, name, groupSize, timeout)
	}
}

func EnsureLWSOperator(client kubernetes.Interface) error {
	cs, ok := client.(*kubernetes.Clientset)
	if !ok {
		return fmt.Errorf("kubernetes client is not *kubernetes.Clientset")
	}
	return esutils.EnsureLeaderWorkerSetController(cs, context.Background())
}
