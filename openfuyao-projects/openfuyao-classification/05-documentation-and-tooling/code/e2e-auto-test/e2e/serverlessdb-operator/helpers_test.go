package serverlessdb_operator

import (
	"fmt"
	"net/http"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	k8stypes "k8s.io/apimachinery/pkg/types"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/serverlessdb-operator/utils"
)

const (
	pgVersion      = 17
	reconcileWait  = 3 * time.Minute
	reconcilePoll  = 15 * time.Second
	podResizeWait  = 2 * time.Minute
	podResizePoll  = 5 * time.Second
	allocateWait   = 5 * time.Minute
	allocateCpu    = "100m"
	allocateMemory = "128Mi"
)

// cleanupLeftoverCRs 在 suite 启动前清理 namespace 内残留的 DBResourcePool/DBInstance，
// 避免上一轮被中断/超时的测试运行遗留的 CR 污染匹配（特别是 terminating 中的 pool 会被
// Allocate 匹配到，但其 Reconciler 走 delete 路径不会 syncInstanceStatuses）。
func cleanupLeftoverCRs() {
	By("清理上一轮运行残留的 CR")
	list, err := k8sClient.DynamicClient.Resource(PoolGVR).Namespace(testNs).List(ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred())
	if len(list.Items) == 0 {
		return
	}
	for i := range list.Items {
		name := list.Items[i].GetName()
		_ = k8sClient.DynamicClient.Resource(PoolGVR).Namespace(testNs).Delete(ctx, name, metav1.DeleteOptions{})
	}
	Eventually(func(g Gomega) int {
		remaining, lerr := k8sClient.DynamicClient.Resource(PoolGVR).Namespace(testNs).List(ctx, metav1.ListOptions{})
		g.Expect(lerr).NotTo(HaveOccurred())
		for j := range remaining.Items {
			item := &remaining.Items[j]
			if item.GetDeletionTimestamp() != nil {
				patch := []byte(`{"metadata":{"finalizers":null}}`)
				_, _ = k8sClient.DynamicClient.Resource(PoolGVR).Namespace(testNs).Patch(
					ctx, item.GetName(), k8stypes.MergePatchType, patch, metav1.PatchOptions{})
			}
		}
		return len(remaining.Items)
	}, 2*time.Minute, 5*time.Second).Should(Equal(0), "残留 pool 应全部清理完毕")
}

// createPoolAndAwaitWarm creates a DBResourcePool, registers cleanup, and waits
// until at least one DBInstance reaches WarmReady. Returns the created pool.
func createPoolAndAwaitWarm(poolName string, replicas int) *unstructured.Unstructured {
	// best-effort: 清理上一次中断运行残留的同名 pool，避免 AlreadyExists
	_ = DeletePool(ctx, k8sClient, testNs, poolName)
	pool := BuildDBResourcePool(poolName, testNs, replicas, pgVersion)
	created, err := CreatePool(ctx, k8sClient, pool)
	Expect(err).NotTo(HaveOccurred(), "创建 DBResourcePool 应当成功")

	DeferCleanup(func() {
		_ = DeletePool(ctx, k8sClient, testNs, poolName)
	})

	By(fmt.Sprintf("等待 pool %s 出现 WarmReady 实例", poolName))
	Eventually(func(g Gomega) {
		list, err := ListDBInstances(ctx, k8sClient, testNs, fmt.Sprintf("%s=%s", LabelDBPool, poolName))
		g.Expect(err).NotTo(HaveOccurred())
		for i := range list.Items {
			if DBInstancePhase(&list.Items[i]) == "WarmReady" {
				return
			}
		}
		g.Expect(false).To(BeTrue(), "应当至少有一个 WarmReady 实例")
	}, reconcileWait, reconcilePoll).Should(Succeed())
	return created
}

// awaitPoolInstanceCount waits until the number of DBInstances belonging to
// poolName equals want.
func awaitPoolInstanceCount(poolName string, want int) {
	Eventually(func(g Gomega) int {
		list, err := ListDBInstances(ctx, k8sClient, testNs, fmt.Sprintf("%s=%s", LabelDBPool, poolName))
		g.Expect(err).NotTo(HaveOccurred())
		return len(list.Items)
	}, reconcileWait, reconcilePoll).Should(Equal(want))
}

// dbInstanceByAppRef returns the first DBInstance with label app-ref=appRef.
func dbInstanceByAppRef(appRef string) *unstructured.Unstructured {
	list, err := ListDBInstances(ctx, k8sClient, testNs, fmt.Sprintf("%s=%s", LabelAppRef, appRef))
	Expect(err).NotTo(HaveOccurred())
	Expect(list.Items).NotTo(BeEmpty(), "应当存在 app-ref=%s 的 DBInstance", appRef)
	return &list.Items[0]
}

// awaitInstancePhase waits until the DBInstance with app-ref=appRef reaches phase.
func awaitInstancePhase(appRef, phase string) {
	Eventually(func(g Gomega) string {
		list, err := ListDBInstances(ctx, k8sClient, testNs, fmt.Sprintf("%s=%s", LabelAppRef, appRef))
		g.Expect(err).NotTo(HaveOccurred())
		if len(list.Items) == 0 {
			return ""
		}
		return DBInstancePhase(&list.Items[0])
	}, reconcileWait, reconcilePoll).Should(Equal(phase))
}

// computePod returns the backing Pod for the DBInstance with app-ref=appRef.
func computePod(appRef string) interface{} {
	pods, err := k8sClient.ListPods(ctx, testNs, metav1.ListOptions{LabelSelector: fmt.Sprintf("%s=%s", LabelAppRef, appRef)})
	Expect(err).NotTo(HaveOccurred())
	Expect(pods.Items).NotTo(BeEmpty(), "应当存在 app-ref=%s 的 Pod", appRef)
	return &pods.Items[0]
}

func allocateAndAwaitRunning(appRef string) *InstanceResponse {
	const maxAttempts = 2
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		resp, status, err := apiClient.Allocate(ctx, testNs, &CreateInstanceRequest{
			AppRef:    appRef,
			PgVersion: pgVersion,
			Resources: InstanceResources{Cpu: allocateCpu, Memory: allocateMemory},
		})
		Expect(err).NotTo(HaveOccurred())
		Expect(status).To(Equal(200), "申请实例应当返回 200")
		Expect(resp.Endpoint).NotTo(BeEmpty(), "endpoint 应当非空")

		phaseErr := waitForPhaseOrTimeout(appRef, "Running", allocateWait)
		if phaseErr == nil {
			DeferCleanup(func() {
				_, _ = apiClient.Release(ctx, testNs, appRef)
			})
			return resp
		}

		if attempt < maxAttempts {
			_, _ = apiClient.Release(ctx, testNs, appRef)
			time.Sleep(5 * time.Second)
		} else {
			DeferCleanup(func() {
				_, _ = apiClient.Release(ctx, testNs, appRef)
			})
			Fail(fmt.Sprintf("实例 %s 未能进入 Running 阶段 (尝试 %d 次): %v", appRef, maxAttempts, phaseErr))
		}
	}
	return nil
}

func waitForPhaseOrTimeout(appRef, phase string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		list, err := ListDBInstances(ctx, k8sClient, testNs, fmt.Sprintf("%s=%s", LabelAppRef, appRef))
		if err != nil {
			return err
		}
		if len(list.Items) > 0 && DBInstancePhase(&list.Items[0]) == phase {
			return nil
		}
		time.Sleep(reconcilePoll)
	}
	return fmt.Errorf("timed out waiting for %s to reach phase %s", appRef, phase)
}

func clusterSupportsPodResize() bool {
	result := k8sClient.Clientset.CoreV1().RESTClient().
		Get().
		AbsPath(fmt.Sprintf("/api/v1/namespaces/%s/pods/__nonexistent__/resize", testNs)).
		Do(ctx)
	err := result.Error()
	if err == nil {
		return true
	}
	return !apierrors.IsNotFound(err)
}

// restartOperatorPod deletes the current operator pod and waits until a new pod
// becomes Ready. The port-forward tunnel breaks when the old pod dies; callers
// must reestablishPortForward afterwards.
func restartOperatorPod() {
	By("删除 operator pod 触发重建")
	pods, err := k8sClient.ListPods(ctx, opNs, metav1.ListOptions{LabelSelector: OperatorPodLabel})
	Expect(err).NotTo(HaveOccurred(), "列出 operator pod 应当成功")
	Expect(pods.Items).NotTo(BeEmpty(), "应当存在 operator pod")
	oldName := pods.Items[0].Name
	Expect(DeletePod(ctx, k8sClient, opNs, oldName)).To(Succeed(), "删除 operator pod 应当成功")

	By("等待新 operator pod Ready")
	Eventually(func(g Gomega) string {
		pods, lerr := k8sClient.ListPods(ctx, opNs, metav1.ListOptions{LabelSelector: OperatorPodLabel})
		g.Expect(lerr).NotTo(HaveOccurred())
		for i := range pods.Items {
			if pods.Items[i].Name != oldName && IsPodReady(&pods.Items[i]) {
				return pods.Items[i].Name
			}
		}
		return ""
	}, 3*time.Minute, 5*time.Second).ShouldNot(BeEmpty(), "新 operator pod 应当 Ready")
}

// reestablishPortForward tears down the stale port-forward and builds a new one
// to the current operator pod, then rebuilds apiClient with the saved tenant
// token. Waits until the HTTP API is reachable on the new local port.
func reestablishPortForward() {
	if pfStop != nil {
		pfStop()
	}
	var err error
	curLocalPort, pfStop, err = StartPortForward(k8sClient, ctx, opNs, OperatorPodLabel, OperatorGinPort)
	Expect(err).NotTo(HaveOccurred(), "重建 port-forward 应当成功")
	apiClient = NewAPIClient(fmt.Sprintf("http://localhost:%d", curLocalPort), tenantToken)
	Eventually(func(g Gomega) {
		resp, herr := http.Get(fmt.Sprintf("http://localhost:%d/api/v1/instances", curLocalPort))
		g.Expect(herr).NotTo(HaveOccurred())
		if resp != nil {
			resp.Body.Close()
		}
	}, 60*time.Second, 3*time.Second).Should(Succeed(), "重建后 operator HTTP API 应当可达")
}

// poolStatusFields extracts total/warm/allocated counts from a DBResourcePool
// status to verify state recovery across operator restarts.
func poolStatusFields(poolName string) (total, warm, allocated int32) {
	pool, err := GetPool(ctx, k8sClient, testNs, poolName)
	Expect(err).NotTo(HaveOccurred(), "获取 pool 应当成功")
	if v, ok, _ := unstructured.NestedInt64(pool.Object, "status", "totalInstances"); ok {
		total = int32(v)
	}
	if v, ok, _ := unstructured.NestedInt64(pool.Object, "status", "warmInstances"); ok {
		warm = int32(v)
	}
	if v, ok, _ := unstructured.NestedInt64(pool.Object, "status", "allocatedInstances"); ok {
		allocated = int32(v)
	}
	return
}

// operatorDeploymentName resolves the operator Deployment name once and caches it
// in a package var to avoid repeated ownerRef walks.
var operatorDeploymentName = ""

func getOperatorDeploymentName() string {
	if operatorDeploymentName != "" {
		return operatorDeploymentName
	}
	name, err := GetOperatorDeploymentName(ctx, k8sClient, opNs, OperatorPodLabel)
	Expect(err).NotTo(HaveOccurred(), "解析 operator Deployment 名称应当成功")
	operatorDeploymentName = name
	return name
}

// scaleOperator scales the operator Deployment and waits for want Ready pods.
func scaleOperator(want int32) {
	By(fmt.Sprintf("scale operator Deployment 到 %d 副本", want))
	Expect(ScaleDeployment(ctx, k8sClient, opNs, getOperatorDeploymentName(), want)).To(Succeed())
	awaitOperatorReadyCount(int(want))
}

func awaitOperatorReadyCount(want int) {
	Eventually(func(g Gomega) int {
		names, err := ListReadyPodNames(ctx, k8sClient, opNs, OperatorPodLabel)
		g.Expect(err).NotTo(HaveOccurred())
		return len(names)
	}, 3*time.Minute, 5*time.Second).Should(Equal(want), "operator Ready pod 数应当为 %d", want)
}

// leaderPodName returns the pod name of the current lease holder, or "" if no
// holder. The lease lives in the operator namespace (in-cluster default).
func leaderPodName() string {
	leaseNs := EnvOr("SERVERLESSDB_LEADER_ELECTION_NS", opNs)
	holder, err := GetLeaseHolder(ctx, k8sClient, leaseNs, DefaultLeaderElectionID)
	Expect(err).NotTo(HaveOccurred(), "读取 Lease holder 应当成功")
	return LeaderPodName(holder)
}

// deleteLeaderPod deletes the pod whose name matches the current lease holder
// and waits until the lease holder identity changes (standby takeover).
func deleteLeaderPodAndAwaitTakeover() {
	oldLeader := leaderPodName()
	Expect(oldLeader).NotTo(BeEmpty(), "应当存在当前 leader pod")
	By(fmt.Sprintf("删除 leader pod %s 等待备节点接管", oldLeader))
	Expect(DeletePod(ctx, k8sClient, opNs, oldLeader)).To(Succeed())

	// Wait for a new pod to become Ready (the deleted leader is recreated) and
	// the lease holder to differ from the old leader.
	Eventually(func(g Gomega) bool {
		return leaderPodName() != "" && leaderPodName() != oldLeader
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "备节点应当当选新 leader")
}

// computeContainerID returns the containerID of the compute container in the pod
// backing appRef (without the "docker://" prefix).
func computeContainerID(appRef string) string {
	pods, err := k8sClient.ListPods(ctx, testNs, metav1.ListOptions{LabelSelector: LabelEq(LabelAppRef, appRef)})
	Expect(err).NotTo(HaveOccurred())
	Expect(pods.Items).NotTo(BeEmpty())
	for _, c := range pods.Items[0].Status.ContainerStatuses {
		if c.Name == ContainerCompute {
			return c.ContainerID
		}
	}
	return ""
}

// podUID returns the UID of the pod backing appRef.
func podUID(appRef string) string {
	pods, err := k8sClient.ListPods(ctx, testNs, metav1.ListOptions{LabelSelector: LabelEq(LabelAppRef, appRef)})
	Expect(err).NotTo(HaveOccurred())
	Expect(pods.Items).NotTo(BeEmpty())
	return string(pods.Items[0].UID)
}

// podRestartCount returns the total restartCount of the compute container in the
// pod backing appRef.
func podRestartCount(appRef string) int32 {
	pods, err := k8sClient.ListPods(ctx, testNs, metav1.ListOptions{LabelSelector: LabelEq(LabelAppRef, appRef)})
	Expect(err).NotTo(HaveOccurred())
	Expect(pods.Items).NotTo(BeEmpty())
	for _, c := range pods.Items[0].Status.ContainerStatuses {
		if c.Name == ContainerCompute {
			return c.RestartCount
		}
	}
	return 0
}
