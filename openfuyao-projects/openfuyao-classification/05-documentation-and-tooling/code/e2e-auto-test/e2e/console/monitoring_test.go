package console

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

const (
	// 与 monitoring-service/pkg/server/runtime/webservices.go 中 RestApiRootPath + GroupVersion(monitoring/v1) 一致
	monitoringAPIBasePath = "/rest/monitoring/v1"
	// 用于列举 Deployment/StatefulSet/Pod 等探测路径的默认命名空间
	monitoringProbeNamespace = "kube-system"
)

func monitoringAPIPrefix() string {
	return monitoringAPIBasePath
}

func probeNamespace() string {
	return monitoringProbeNamespace
}

func expectJSONKeys(body []byte, keys ...string) {
	var m map[string]json.RawMessage
	Expect(json.Unmarshal(body, &m)).To(Succeed(), string(body))
	for _, k := range keys {
		Expect(m).To(HaveKey(k), "response body: %s", string(body))
	}
}

// expectPromQLQueryBody 断言 /query 返回体：handler 写 client.Metric（见 monitoring-service handleUserQuery），
// 顶层为 data/msg/code 等，不是 services.Metrics 的 results。
func expectPromQLQueryBody(body []byte) {
	var m map[string]json.RawMessage
	Expect(json.Unmarshal(body, &m)).To(Succeed(), string(body))
	Expect(m).To(Or(HaveKey("data"), HaveKey("msg")), "expected client.Metric shape (data or msg): %s", string(body))
}

func expectMonitoringGET(path string) []byte {
	resp, err := makeRequest("GET", path, nil)
	Expect(err).NotTo(HaveOccurred())
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	Expect(err).NotTo(HaveOccurred())
	Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
	return body
}

// --- K8s 列表发现（用于带 path 参数的 metrics 接口）---

type k8sListMeta struct {
	Items []struct {
		Metadata struct {
			Name string `json:"name"`
		} `json:"metadata"`
	} `json:"items"`
}

func k8sGETList(path string) (k8sListMeta, int, []byte, error) {
	resp, err := makeRequest("GET", path, nil)
	if err != nil {
		return k8sListMeta{}, 0, nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return k8sListMeta{}, resp.StatusCode, body, err
	}
	var list k8sListMeta
	if err := json.Unmarshal(body, &list); err != nil {
		return k8sListMeta{}, resp.StatusCode, body, err
	}
	return list, resp.StatusCode, body, nil
}

func firstNodeName() (string, bool) {
	list, code, body, err := k8sGETList("/api/v1/nodes?limit=5")
	if err != nil || code != http.StatusOK || len(list.Items) == 0 {
		GinkgoWriter.Printf("firstNodeName: status=%d err=%v body=%s\n", code, err, string(body))
		return "", false
	}
	return list.Items[0].Metadata.Name, true
}

func allNamespaces() []string {
	list, code, body, err := k8sGETList("/api/v1/namespaces?limit=100")
	if err != nil || code != http.StatusOK || len(list.Items) == 0 {
		GinkgoWriter.Printf("allNamespaces: status=%d err=%v body=%s\n", code, err, string(body))
		return nil
	}
	names := make([]string, 0, len(list.Items))
	for _, item := range list.Items {
		names = append(names, item.Metadata.Name)
	}
	return names
}

// firstNamespacedWorkload 先在 namespace 下查找，找不到则遍历所有命名空间返回第一个匹配的资源名及其所在命名空间。
func firstNamespacedWorkload(kind, namespace string) (name, foundNS string, ok bool) {
	p := fmt.Sprintf("/apis/apps/v1/namespaces/%s/%s?limit=20", url.PathEscape(namespace), kind)
	list, code, body, err := k8sGETList(p)
	if err == nil && code == http.StatusOK && len(list.Items) > 0 {
		return list.Items[0].Metadata.Name, namespace, true
	}
	GinkgoWriter.Printf("firstNamespacedWorkload %s/%s: status=%d err=%v body=%s, falling back to all namespaces\n", kind, namespace, code, err, string(body))

	for _, ns := range allNamespaces() {
		if ns == namespace {
			continue
		}
		p2 := fmt.Sprintf("/apis/apps/v1/namespaces/%s/%s?limit=20", url.PathEscape(ns), kind)
		list2, code2, _, err2 := k8sGETList(p2)
		if err2 == nil && code2 == http.StatusOK && len(list2.Items) > 0 {
			GinkgoWriter.Printf("firstNamespacedWorkload %s: found in namespace %s\n", kind, ns)
			return list2.Items[0].Metadata.Name, ns, true
		}
	}
	return "", "", false
}

// firstPodName 先在 namespace 下查找，找不到则遍历所有命名空间返回第一个 Pod 及其所在命名空间。
func firstPodName(namespace string) (name, foundNS string, ok bool) {
	p := fmt.Sprintf("/api/v1/namespaces/%s/pods?limit=20", url.PathEscape(namespace))
	list, code, body, err := k8sGETList(p)
	if err == nil && code == http.StatusOK && len(list.Items) > 0 {
		return list.Items[0].Metadata.Name, namespace, true
	}
	GinkgoWriter.Printf("firstPodName %s: status=%d err=%v body=%s, falling back to all namespaces\n", namespace, code, err, string(body))

	for _, ns := range allNamespaces() {
		if ns == namespace {
			continue
		}
		p2 := fmt.Sprintf("/api/v1/namespaces/%s/pods?limit=20", url.PathEscape(ns))
		list2, code2, _, err2 := k8sGETList(p2)
		if err2 == nil && code2 == http.StatusOK && len(list2.Items) > 0 {
			GinkgoWriter.Printf("firstPodName: found in namespace %s\n", ns)
			return list2.Items[0].Metadata.Name, ns, true
		}
	}
	return "", "", false
}

type podDetail struct {
	Spec struct {
		Containers []struct {
			Name string `json:"name"`
		} `json:"containers"`
	} `json:"spec"`
}

func firstContainerName(namespace, pod string) (string, bool) {
	p := fmt.Sprintf("/api/v1/namespaces/%s/pods/%s", url.PathEscape(namespace), url.PathEscape(pod))
	resp, err := makeRequest("GET", p, nil)
	if err != nil {
		return "", false
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil || resp.StatusCode != http.StatusOK {
		return "", false
	}
	var detail podDetail
	if err := json.Unmarshal(body, &detail); err != nil || len(detail.Spec.Containers) == 0 {
		return "", false
	}
	return detail.Spec.Containers[0].Name, true
}

// --- 从 Prometheus targets 解析 instance（control-plane / node / addons 组件接口需要）---

type targetsEnvelope struct {
	Targets []struct {
		Endpoint string `json:"endpoint"`
	} `json:"targets"`
}

func normalizeInstance(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}
	if u, err := url.Parse(raw); err == nil && u.Host != "" {
		return u.Host
	}
	return raw
}

func resolveInstanceHint() string {
	body := expectMonitoringGET(monitoringAPIPrefix() + "/targets")
	var env targetsEnvelope
	Expect(json.Unmarshal(body, &env)).To(Succeed(), string(body))
	for _, t := range env.Targets {
		if inst := normalizeInstance(t.Endpoint); inst != "" {
			return inst
		}
	}
	return ""
}

var _ = Describe("Monitoring Service REST API", Label("monitoring"), func() {
	base := monitoringAPIPrefix
	ns := probeNamespace

	Context("元数据与PromQL", func() {
		It("GET /alerting-rules返回200且含alertingRules", func() {
			body := expectMonitoringGET(base() + "/alerting-rules")
			expectJSONKeys(body, "alertingRules")
		})

		It("GET /metric-names返回200且含names", func() {
			body := expectMonitoringGET(base() + "/metric-names")
			expectJSONKeys(body, "names")
		})

		It("GET /targets返回200且含targets", func() {
			body := expectMonitoringGET(base() + "/targets")
			expectJSONKeys(body, "targets")
		})

		It("GET /query?expr=up返回200且为client.Metric结构(data或msg)", func() {
			body := expectMonitoringGET(base() + "/query?expr=up")
			expectPromQLQueryBody(body)
		})

		It("GET /query非法time参数返回400", func() {
			resp, err := makeRequest("GET", base()+"/query?expr=up&time=not-a-valid-time", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})

		It("GET /query区间查询start/end合法时返回200且为client.Metric结构(data或msg)", func() {
			end := time.Now()
			start := end.Add(-10 * time.Minute)
			layout := "2006-01-02 15:04:05"
			q := url.Values{}
			q.Set("expr", "up")
			q.Set("start", start.Format(layout))
			q.Set("end", end.Format(layout))
			q.Set("step", "1m")
			body := expectMonitoringGET(base() + "/query?" + q.Encode())
			expectPromQLQueryBody(body)
		})
	})

	Context("集群/节点/工作负载/Pod/容器metrics", func() {
		It("GET /cluster/metrics返回200且含results", func() {
			body := expectMonitoringGET(base() + "/cluster/metrics")
			expectJSONKeys(body, "results")
		})

		It("GET /nodes/{node}/metrics返回200且含results", func() {
			node, ok := firstNodeName()
			if !ok {
				Skip("未发现Node，跳过节点metrics（需可访问/api/v1/nodes）")
			}
			path := fmt.Sprintf("%s/nodes/%s/metrics", base(), url.PathEscape(node))
			body := expectMonitoringGET(path)
			expectJSONKeys(body, "results")
		})

		It("GET .../deployments/.../metrics返回200且含results", func() {
			dep, depNS, ok := firstNamespacedWorkload("deployments", ns())
			if !ok {
				Skip(fmt.Sprintf("所有命名空间下均未发现Deployment"))
			}
			path := fmt.Sprintf("%s/namespaces/%s/deployments/%s/metrics", base(), url.PathEscape(depNS), url.PathEscape(dep))
			body := expectMonitoringGET(path)
			expectJSONKeys(body, "results")
		})

		It("GET .../statefulsets/.../metrics返回200且含results", func() {
			sts, stsNS, ok := firstNamespacedWorkload("statefulsets", ns())
			if !ok {
				Skip(fmt.Sprintf("所有命名空间下均未发现StatefulSet"))
			}
			path := fmt.Sprintf("%s/namespaces/%s/statefulsets/%s/metrics", base(), url.PathEscape(stsNS), url.PathEscape(sts))
			body := expectMonitoringGET(path)
			expectJSONKeys(body, "results")
		})

		It("GET .../daemonsets/.../metrics返回200且含results", func() {
			ds, dsNS, ok := firstNamespacedWorkload("daemonsets", ns())
			if !ok {
				Skip(fmt.Sprintf("所有命名空间下均未发现DaemonSet"))
			}
			path := fmt.Sprintf("%s/namespaces/%s/daemonsets/%s/metrics", base(), url.PathEscape(dsNS), url.PathEscape(ds))
			body := expectMonitoringGET(path)
			expectJSONKeys(body, "results")
		})

		It("GET .../pods/.../metrics返回200且含results", func() {
			pod, podNS, ok := firstPodName(ns())
			if !ok {
				Skip(fmt.Sprintf("所有命名空间下均未发现Pod"))
			}
			path := fmt.Sprintf("%s/namespaces/%s/pods/%s/metrics", base(), url.PathEscape(podNS), url.PathEscape(pod))
			body := expectMonitoringGET(path)
			expectJSONKeys(body, "results")
		})

		It("GET .../pods/.../containers/.../metrics返回200且含results", func() {
			pod, podNS, ok := firstPodName(ns())
			if !ok {
				Skip(fmt.Sprintf("所有命名空间下均未发现Pod"))
			}
			ctr, ok := firstContainerName(podNS, pod)
			if !ok {
				Skip("无法解析Pod的container名称")
			}
			path := fmt.Sprintf("%s/namespaces/%s/pods/%s/containers/%s/metrics",
				base(), url.PathEscape(podNS), url.PathEscape(pod), url.PathEscape(ctr))
			body := expectMonitoringGET(path)
			expectJSONKeys(body, "results")
		})
	})

	Context("控制面/节点组件/插件组件metrics", func() {
		It("GET control-plane-components/kube-apiserver带verb+resource 返回 200 且含 results", func() {
			q := url.Values{}
			q.Set("verb", "GET")
			q.Set("resource", "pods")
			path := fmt.Sprintf("%s/control-plane-components/%s?%s", base(), url.PathEscape("kube-apiserver"), q.Encode())
			body := expectMonitoringGET(path)
			expectJSONKeys(body, "results")
		})

		It("GET control-plane-components/etcd带instance返回200且含results", func() {
			inst := resolveInstanceHint()
			if inst == "" {
				Skip("/targets无可用endpoint，跳过etcd组件查询")
			}
			q := url.Values{}
			q.Set("instance", inst)
			path := fmt.Sprintf("%s/control-plane-components/%s?%s", base(), url.PathEscape("etcd"), q.Encode())
			body := expectMonitoringGET(path)
			expectJSONKeys(body, "results")
		})

		It("GET control-plane-components/kube-scheduler带instance返回200且含results", func() {
			inst := resolveInstanceHint()
			if inst == "" {
				Skip("/targets无可用endpoint，跳过kube-scheduler组件查询")
			}
			q := url.Values{}
			q.Set("instance", inst)
			path := fmt.Sprintf("%s/control-plane-components/%s?%s", base(), url.PathEscape("kube-scheduler"), q.Encode())
			body := expectMonitoringGET(path)
			expectJSONKeys(body, "results")
		})

		It("GET control-plane-components/kube-controller-manager带instance返回200且含results", func() {
			inst := resolveInstanceHint()
			if inst == "" {
				Skip("/targets无可用endpoint，跳过kube-controller-manager组件查询")
			}
			q := url.Values{}
			q.Set("instance", inst)
			path := fmt.Sprintf("%s/control-plane-components/%s?%s", base(), url.PathEscape("kube-controller-manager"), q.Encode())
			body := expectMonitoringGET(path)
			expectJSONKeys(body, "results")
		})

		It("GET node-components/kubelet 带 instance 返回 200 且含 results", func() {
			inst := resolveInstanceHint()
			if inst == "" {
				Skip("/targets 无可用 endpoint，跳过 kubelet 组件查询")
			}
			q := url.Values{}
			q.Set("instance", inst)
			path := fmt.Sprintf("%s/node-components/%s?%s", base(), url.PathEscape("kubelet"), q.Encode())
			body := expectMonitoringGET(path)
			expectJSONKeys(body, "results")
		})

		It("GET node-components/kube-proxy带instance返回200且含results", func() {
			inst := resolveInstanceHint()
			if inst == "" {
				Skip("/targets无可用endpoint，跳过kube-proxy组件查询")
			}
			q := url.Values{}
			q.Set("instance", inst)
			path := fmt.Sprintf("%s/node-components/%s?%s", base(), url.PathEscape("kube-proxy"), q.Encode())
			body := expectMonitoringGET(path)
			expectJSONKeys(body, "results")
		})

		It("GET add-ons-components/coredns带instance返回200且含results", func() {
			inst := resolveInstanceHint()
			if inst == "" {
				Skip("/targets无可用endpoint，跳过coredns组件查询")
			}
			q := url.Values{}
			q.Set("instance", inst)
			path := fmt.Sprintf("%s/add-ons-components/%s?%s", base(), url.PathEscape("coredns"), q.Encode())
			body := expectMonitoringGET(path)
			expectJSONKeys(body, "results")
		})
	})
})
