/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package service

var clusterLevelMetrics = []string{
	"cluster_cpu_total",
	"cluster_memory_total",
	"cluster_total_cpu_usage",
	"cluster_total_memory_usage",
	"cluster_hls_cpu_usage",
	"cluster_hls_memory_usage",
	"cluster_ls_cpu_usage",
	"cluster_ls_memory_usage",
	"cluster_be_cpu_usage",
	"cluster_be_memory_usage",
	"cluster_noncolo_cpu_usage",
	"cluster_noncolo_memory_usage",
	"cluster_be_cpu_allocatable",
	"cluster_be_memory_allocatable",
	"cluster_be_cpu_request",
	"cluster_be_memory_request",
	"cluster_hls_cpu_request",
	"cluster_hls_memory_request",
	"cluster_ls_cpu_request",
	"cluster_ls_memory_request",
	"cluster_noncolo_cpu_request",
	"cluster_noncolo_memory_request",
}

var nodeLevelMetrics = []string{
	"node_cpu_total",
	"node_memory_total",
	"node_total_cpu_usage",
	"node_total_memory_usage",
	"node_hls_cpu_usage",
	"node_hls_memory_usage",
	"node_ls_cpu_usage",
	"node_ls_memory_usage",
	"node_be_cpu_usage",
	"node_be_memory_usage",
	"node_noncolo_cpu_usage",
	"node_noncolo_memory_usage",
	"node_be_cpu_allocatable",
	"node_be_memory_allocatable",
	"node_be_cpu_request",
	"node_be_memory_request",
	"node_hls_cpu_request",
	"node_hls_memory_request",
	"node_ls_cpu_request",
	"node_ls_memory_request",
	"node_noncolo_cpu_request",
	"node_noncolo_memory_request",
}

var prometheusQLSet = map[string]string{
	"cluster_cpu_total":    `sum(machine_cpu_cores) by (cluster)`,
	"cluster_memory_total": `sum(node_memory_MemTotal_bytes{job="node-exporter"}) by (cluster)`,
	"cluster_total_cpu_usage": `sum(rate(node_cpu_seconds_total{mode!="idle", mode!="iowait", ` +
		`mode!="steal"}[5m])) by (cluster)`,
	"cluster_total_memory_usage": `sum(node_memory_MemTotal_bytes{job="node-exporter"}) by (cluster) - ` +
		`sum(:node_memory_MemAvailable_bytes:sum) by (cluster)`,
	"cluster_hls_cpu_usage":         `sum(pod_cpu_usage_by_colocation{type="HLS"})`,
	"cluster_hls_memory_usage":      `sum(pod_memory_usage_by_colocation{type="HLS"})`,
	"cluster_ls_cpu_usage":          `sum(pod_cpu_usage_by_colocation{type="LS"})`,
	"cluster_ls_memory_usage":       `sum(pod_memory_usage_by_colocation{type="LS"})`,
	"cluster_be_cpu_usage":          `sum(pod_cpu_usage_by_colocation{type="BE"})`,
	"cluster_be_memory_usage":       `sum(pod_memory_usage_by_colocation{type="BE"})`,
	"cluster_noncolo_cpu_usage":     `sum(pod_cpu_usage_by_colocation{type="None"})`,
	"cluster_noncolo_memory_usage":  `sum(pod_memory_usage_by_colocation{type="None"})`,
	"cluster_be_cpu_allocatable":    `sum(node_overquota_cpu_allocatable) / 1000`,
	"cluster_be_memory_allocatable": `sum(node_overquota_memory_allocatable)`,
	"cluster_be_cpu_request":        `sum(pod_overquota_cpu_request) / 1000`,
	"cluster_be_memory_request":     `sum(pod_overquota_memory_request)`,
	"cluster_hls_cpu_request": `sum (
	kube_pod_container_resource_requests{resource="cpu", unit="core"}
	and on(pod, namespace)
	label_replace(
		label_replace(
			pod_cpu_usage_by_colocation{type="HLS"},
			"namespace", "$1", "exported_namespace", "(.*)"
		),
	"pod", "$1", "pod_name", "(.*)"
	)
)`,
	"cluster_hls_memory_request": `sum (
	kube_pod_container_resource_requests{resource="memory", unit="byte"}
	and on(pod, namespace)
	label_replace(
		label_replace(
			pod_memory_usage_by_colocation{type="HLS"},
			"namespace", "$1", "exported_namespace", "(.*)"
		),
	"pod", "$1", "pod_name", "(.*)"
	)
)`,
	"cluster_ls_cpu_request": `sum (
	kube_pod_container_resource_requests{resource="cpu", unit="core"}
	and on(pod, namespace)
	label_replace(
		label_replace(
			pod_cpu_usage_by_colocation{type="LS"},
			"namespace", "$1", "exported_namespace", "(.*)"
		),
	"pod", "$1", "pod_name", "(.*)"
	)
)`,
	"cluster_ls_memory_request": `sum (
	kube_pod_container_resource_requests{resource="memory", unit="byte"}
	and on(pod, namespace)
	label_replace(
		label_replace(
			pod_memory_usage_by_colocation{type="LS"},
			"namespace", "$1", "exported_namespace", "(.*)"
		),
	"pod", "$1", "pod_name", "(.*)"
	)
)`,
	"cluster_noncolo_cpu_request": `sum (
	kube_pod_container_resource_requests{resource="cpu", unit="core"}
	and on(namespace, pod)
	label_replace(
		label_replace(
			pod_cpu_usage_by_colocation{type="None"},
			"namespace", "$1", "exported_namespace", "(.*)"
		),
	"pod", "$1", "pod_name", "(.*)"
	)
)`,
	"cluster_noncolo_memory_request": `sum (
	kube_pod_container_resource_requests{resource="memory", unit="byte"}
	and on(pod, namespace)
	label_replace(
		label_replace(
			pod_memory_usage_by_colocation{type="None"},
			"namespace", "$1", "exported_namespace", "(.*)"
		),
	"pod", "$1", "pod_name", "(.*)"
	)
)`,
	"node_cpu_total": `sum(label_replace(sum(machine_cpu_cores) by (cluster, node), "instance", "$1", ` +
		`"node", "(.*)")) by (cluster, instance)`,
	"node_memory_total": `sum(node_memory_MemTotal_bytes{job="node-exporter"}) by (cluster, instance)`,
	"node_total_cpu_usage": `sum(label_replace(sum(machine_cpu_cores) by (cluster, node), "instance", "$1", ` +
		`"node", "(.*)")) by (cluster, instance) * (1 - (avg(irate(node_cpu_seconds_total{mode="idle"}[5m])) ` +
		`by (cluster, instance)))`,
	"node_total_memory_usage": `sum(node_memory_MemTotal_bytes{job="node-exporter"}) by (cluster, instance) - ` +
		`sum(node_memory_MemFree_bytes{job="node-exporter"} + node_memory_Buffers_bytes{job="node-exporter"} + ` +
		`node_memory_Cached_bytes{job="node-exporter"}) by (cluster, instance)`,
	"node_hls_cpu_usage": `sum by (instance) (label_replace(` +
		`pod_cpu_usage_by_colocation{type="HLS"}, "instance", "$1", "node", "(.*)"))`,
	"node_hls_memory_usage": `sum by (instance) (label_replace(` +
		`pod_memory_usage_by_colocation{type="HLS"}, "instance", "$1", "node", "(.*)"))`,
	"node_ls_cpu_usage": `sum by (instance) (label_replace(` +
		`pod_cpu_usage_by_colocation{type="LS"}, "instance", "$1", "node", "(.*)"))`,
	"node_ls_memory_usage": `sum by (instance) (label_replace(` +
		`pod_memory_usage_by_colocation{type="LS"}, "instance", "$1", "node", "(.*)"))`,
	"node_be_cpu_usage": `sum by (instance) (label_replace(` +
		`pod_cpu_usage_by_colocation{type="BE"}, "instance", "$1", "node", "(.*)"))`,
	"node_be_memory_usage": `sum by (instance) (label_replace(` +
		`pod_memory_usage_by_colocation{type="BE"}, "instance", "$1", "node", "(.*)"))`,
	"node_noncolo_cpu_usage": `sum by (instance) (label_replace(` +
		`pod_cpu_usage_by_colocation{type="None"}, "instance", "$1", "node", "(.*)"))`,
	"node_noncolo_memory_usage": `sum by (instance) (label_replace(` +
		`pod_memory_usage_by_colocation{type="None"}, "instance", "$1", "node", "(.*)"))`,
	"node_be_cpu_allocatable": `sum(label_replace(node_overquota_cpu_allocatable, ` +
		`"instance", "$1", "node_name", "(.*)")) by (instance) / 1000`,
	"node_be_memory_allocatable": `sum(label_replace(node_overquota_memory_allocatable, ` +
		`"instance", "$1", "node_name", "(.*)")) by (instance)`,
	"node_be_cpu_request": `sum(label_replace(pod_overquota_cpu_request, "instance", ` +
		`"$1", "node", "(.*)")) by (instance) / 1000`,
	"node_be_memory_request": `sum(label_replace(pod_overquota_memory_request, "instance", ` +
		`"$1", "node", "(.*)")) by (instance)`,
	"node_hls_cpu_request": `sum by (instance) (
    label_replace(
      kube_pod_container_resource_requests{resource="cpu", unit="core"}
	  and on(pod, namespace)
      label_replace(
        label_replace(
          pod_cpu_usage_by_colocation{type="HLS"},
          "namespace", "$1", "exported_namespace", "(.*)"
        ),
        "pod", "$1", "pod_name", "(.*)"
      ),
      "instance", "$1", "node", "(.*)"
    )
  )`,
	"node_hls_memory_request": `sum by (instance) (
    label_replace(
      kube_pod_container_resource_requests{resource="memory", unit="byte"}
	  and on(pod, namespace)
      label_replace(
        label_replace(
          pod_memory_usage_by_colocation{type="HLS"},
          "namespace", "$1", "exported_namespace", "(.*)"
        ),
        "pod", "$1", "pod_name", "(.*)"
      ),
      "instance", "$1", "node", "(.*)"
    )
  )`,
	"node_ls_cpu_request": `sum by (instance) (
    label_replace(
      kube_pod_container_resource_requests{resource="cpu", unit="core"}
	  and on(pod, namespace)
      label_replace(
        label_replace(
          pod_cpu_usage_by_colocation{type="LS"},
          "namespace", "$1", "exported_namespace", "(.*)"
        ),
        "pod", "$1", "pod_name", "(.*)"
      ),
      "instance", "$1", "node", "(.*)"
    )
  )`,
	"node_ls_memory_request": `sum by (instance) (
    label_replace(
      kube_pod_container_resource_requests{resource="memory", unit="byte"}
	  and on(pod, namespace)
      label_replace(
        label_replace(
          pod_memory_usage_by_colocation{type="LS"},
          "namespace", "$1", "exported_namespace", "(.*)"
        ),
        "pod", "$1", "pod_name", "(.*)"
      ),
      "instance", "$1", "node", "(.*)"
    )
  )`,
	"node_noncolo_cpu_request": `sum by (instance) (
    label_replace(
      kube_pod_container_resource_requests{resource="cpu", unit="core"}
	  and on(pod, namespace)
      label_replace(
        label_replace(
          pod_cpu_usage_by_colocation{type="None"},
          "namespace", "$1", "exported_namespace", "(.*)"
        ),
        "pod", "$1", "pod_name", "(.*)"
      ),
      "instance", "$1", "node", "(.*)"
    )
  )`,
	"node_noncolo_memory_request": `sum by (instance) (
    label_replace(
      kube_pod_container_resource_requests{resource="memory", unit="byte"}
	  and on(pod, namespace)
      label_replace(
        label_replace(
          pod_memory_usage_by_colocation{type="None"},
          "namespace", "$1", "exported_namespace", "(.*)"
        ),
        "pod", "$1", "pod_name", "(.*)"
      ),
      "instance", "$1", "node", "(.*)"
    )
  )`,
}
