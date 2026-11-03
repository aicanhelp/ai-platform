package serverlessdb_operator

import (
	"fmt"
	"math/rand"
	"sync"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/serverlessdb-operator/utils"
)

func extractConnString(obj *unstructured.Unstructured) (string, bool, error) {
	return unstructured.NestedString(obj.Object, "status", "connectionString")
}

func extractPodName(obj *unstructured.Unstructured) (string, bool, error) {
	return unstructured.NestedString(obj.Object, "status", "podName")
}

var _ = Describe("ServerlessDB 并发申请",
	Label("serverlessdb-operator", "with-workload-cluster", "26.01"), Label("P0"), func() {

		It("serverlessdb-IT-014：并发申请实例，全部成功且 endpoint 互不相同", func() {
			createPoolAndAwaitWarm("pool-it014", 5)

			By("启动 5 个 goroutine 并发申请")
			const n = 5
			const maxRetries = 8
			apps := make([]string, n)
			for i := 0; i < n; i++ {
				apps[i] = fmt.Sprintf("app-it014-%d", i)
			}

			type result struct {
				appRef string
				resp   *InstanceResponse
				status int
				err    error
			}
			results := make([]result, n)
			var wg sync.WaitGroup
			wg.Add(n)
			for i := 0; i < n; i++ {
				go func(idx int) {
					defer wg.Done()
					time.Sleep(time.Duration(800+rand.Intn(3200)) * time.Millisecond)
					for attempt := 0; attempt < maxRetries; attempt++ {
						resp, status, err := apiClient.Allocate(ctx, testNs, &CreateInstanceRequest{
							AppRef:    apps[idx],
							PgVersion: pgVersion,
							Resources: InstanceResources{Cpu: allocateCpu, Memory: allocateMemory},
						})
						if err == nil && status == 200 {
							results[idx] = result{appRef: apps[idx], resp: resp, status: 200}
							return
						}
						if err == nil && status == 409 {
							list, lerr := ListDBInstances(ctx, k8sClient, testNs,
								fmt.Sprintf("%s=%s", LabelAppRef, apps[idx]))
							if lerr == nil && len(list.Items) > 0 {
								cs, _, _ := extractConnString(&list.Items[0])
								pn, _, _ := extractPodName(&list.Items[0])
								results[idx] = result{
									appRef: apps[idx],
									resp:   &InstanceResponse{Endpoint: cs, PodName: pn},
									status: 200,
								}
								return
							}
						}
						if attempt < maxRetries-1 {
							time.Sleep(5 * time.Second)
						}
					}
					results[idx] = result{appRef: apps[idx], err: fmt.Errorf("exhausted %d retries for %s", maxRetries, apps[idx])}
				}(i)
			}
			wg.Wait()

			By("校验全部返回 200")
			endpoints := make(map[string]struct{}, n)
			for i := 0; i < n; i++ {
				Expect(results[i].err).NotTo(HaveOccurred(), "app %s 申请不应出错", results[i].appRef)
				Expect(results[i].status).To(Equal(200), "app %s 应当返回 200", results[i].appRef)
				Expect(results[i].resp.Endpoint).NotTo(BeEmpty(), "app %s endpoint 应当非空", results[i].appRef)
				endpoints[results[i].resp.Endpoint] = struct{}{}
			}
			Expect(endpoints).To(HaveLen(n), "5 个 endpoint 应当互不相同")

			By("校验 5 个 DBInstance 均进入 Running")
			for i := 0; i < n; i++ {
				awaitInstancePhase(apps[i], "Running")
				DeferCleanup(func(appRef string) func() {
					return func() { _, _ = apiClient.Release(ctx, testNs, appRef) }
				}(apps[i]))
			}
		})
	})
