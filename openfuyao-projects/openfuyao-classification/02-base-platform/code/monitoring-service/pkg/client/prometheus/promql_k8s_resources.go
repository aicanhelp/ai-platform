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

package prometheus

var promQLTemplatesResources = map[string]string{
	// cluster
	"cluster_nodes_count": `count(kube_node_info{kernel_version!=""}) by (cluster)`,

	"cluster_nodes_not_ready_count": `count(kube_node_info{kernel_version!=""}) ` +
		`- count(kube_node_status_condition{condition="Ready", status="true",node!~"vci.*"} == 1)`,

	"cluster_pods_count": `count(kube_pod_info) by (cluster, node)`,

	"cluster_cpu_usage": `sum(rate(node_cpu_seconds_total{mode!="idle", mode!="iowait", ` +
		`mode!="steal"}[5m])) by (cluster)`,

	"cluster_cpu_usage_by_namespace": `sum(node_namespace_pod_container:container_cpu_usage_seconds_total:sum_irate) ` +
		`by (cluster, namespace)`,

	"cluster_cpu_total": `sum(machine_cpu_cores) by (cluster)`,

	"cluster_cpu_utilisation": `sum(rate(node_cpu_seconds_total{mode!="idle", mode!="iowait", ` +
		`mode!="steal"}[5m])) by (cluster) / sum(machine_cpu_cores) by (cluster)`,

	"cluster_cpu_requests_total": `sum(namespace_cpu:kube_pod_container_resource_requests:sum) by (cluster)`,

	"cluster_cpu_requests_utilisation": `sum(` +
		`node_namespace_pod_container:container_cpu_usage_seconds_total:sum_irate) by (cluster) / ` +
		`sum(namespace_cpu:kube_pod_container_resource_requests:sum) by (cluster)`,

	"cluster_cpu_limits_total": `sum(namespace_cpu:kube_pod_container_resource_limits:sum) by (cluster)`,

	"cluster_cpu_limits_utilisation": `sum(` +
		`node_namespace_pod_container:container_cpu_usage_seconds_total:sum_irate) by (cluster) / ` +
		`sum(namespace_cpu:kube_pod_container_resource_limits:sum) by (cluster)`,

	"cluster_memory_usage": `sum(node_memory_MemTotal_bytes{job="node-exporter"}) by (cluster) - ` +
		`sum(:node_memory_MemAvailable_bytes:sum) by (cluster)`,

	"cluster_memory_usage_without_cache_by_namespace": `sum(container_memory_rss{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor", container!=""}) by(cluster, namespace)`,

	"cluster_memory_total": `sum(node_memory_MemTotal_bytes{job="node-exporter"}) by (cluster)`,

	"cluster_memory_utilisation": `1 - sum(:node_memory_MemAvailable_bytes:sum) by (cluster) / ` +
		`sum(node_memory_MemTotal_bytes{job="node-exporter"}) by (cluster)`,

	"cluster_pod_count_by_namespace": `count(kube_pod_info) by (cluster, namespace)`,

	"cluster_workload_count_by_namespace": `count(avg(namespace_workload_pod:kube_pod_owner:relabel) by (cluster, ` +
		`workload, namespace)) by (cluster, namespace)`,

	"cluster_memory_requests_total_by_namespace": `sum(namespace_memory:kube_pod_container_resource_requests:sum) by ` +
		`(cluster, namespace)`,

	"cluster_memory_requests_total": `sum(namespace_memory:kube_pod_container_resource_requests:sum) by (cluster)`,

	"cluster_memory_requests_utilisation": `sum(container_memory_rss{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor", container!=""}) by (cluster) / ` +
		`sum(namespace_memory:kube_pod_container_resource_requests:sum) by (cluster)`,

	"cluster_memory_requests_utilisation_by_namespace": `sum(container_memory_rss{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor", container!=""}) by (cluster, namespace) / ` +
		`sum(namespace_memory:kube_pod_container_resource_requests:sum) by (cluster, namespace)`,

	"cluster_memory_limits_total_by_namespace": `sum(namespace_memory:kube_pod_container_resource_limits:sum) ` +
		`by (cluster, namespace)`,

	"cluster_memory_limits_total": `sum(namespace_memory:kube_pod_container_resource_limits:sum) by (cluster)`,

	"cluster_memory_limits_utilisation": `sum(container_memory_rss{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor", container!=""}) by (cluster) / ` +
		`sum(namespace_memory:kube_pod_container_resource_limits:sum) by (cluster)`,

	"cluster_memory_limits_utilisation_by_namespace": `sum(container_memory_rss{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor", container!=""}) by (cluster, namespace) / ` +
		`sum(namespace_memory:kube_pod_container_resource_limits:sum) by (cluster, namespace)`,

	"cluster_rate_of_received_packets_by_namespace": `sum(irate(container_network_receive_packets_total{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor", namespace=~".+"}[5m])) by (cluster, namespace)`,

	"cluster_rate_of_transmitted_packets_by_namespace": `sum(` +
		`irate(container_network_transmit_packets_total{job="kubelet", metrics_path="/metrics/cadvisor", ` +
		`namespace=~".+"}[5m])) by (cluster, namespace)`,

	"cluster_rate_of_received_packets_dropped_by_namespace": `sum(` +
		`irate(container_network_receive_packets_dropped_total{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor", namespace=~".+"}[5m])) by (cluster, namespace)`,

	"cluster_rate_of_transmitted_packets_dropped_by_namespace": `sum(` +
		`irate(container_network_transmit_packets_dropped_total{job="kubelet", metrics_path="/metrics/cadvisor", ` +
		`namespace=~".+"}[5m])) by (cluster, namespace)`,

	"cluster_iops_reads_by_namespace": `sum(rate(container_fs_reads_total{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor", ` +
		`device=~"(/dev/)?(mmcblk.p.+|nvme.+|rbd.+|sd.+|vd.+|xvd.+|dm-.+|md.+|dasd.+)", container!="", ` +
		`namespace!=""}[5m])) by (cluster, namespace)`,

	"cluster_iops_writes_by_namespace": `sum(rate(container_fs_writes_total{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor", ` +
		`device=~"(/dev/)?(mmcblk.p.+|nvme.+|rbd.+|sd.+|vd.+|xvd.+|dm-.+|md.+|dasd.+)", container!="", ` +
		`namespace!=""}[5m])) by (cluster, namespace)`,

	"cluster_iops_reads_writes_by_namespace": `sum(rate(container_fs_reads_total{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor", ` +
		`device=~"(/dev/)?(mmcblk.p.+|nvme.+|rbd.+|sd.+|vd.+|xvd.+|dm-.+|md.+|dasd.+)", container!="", ` +
		`namespace!=""}[5m])) by (cluster, namespace) + sum(rate(container_fs_writes_total{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor", ` +
		`device=~"(/dev/)?(mmcblk.p.+|nvme.+|rbd.+|sd.+|vd.+|xvd.+|dm-.+|md.+|dasd.+)", container!="", ` +
		`namespace!=""}[5m])) by (cluster, namespace)`,

	"cluster_throughput_reads_by_namespace": `sum(rate(container_fs_reads_bytes_total{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor", ` +
		`device=~"(/dev/)?(mmcblk.p.+|nvme.+|rbd.+|sd.+|vd.+|xvd.+|dm-.+|md.+|dasd.+)", container!="", ` +
		`namespace!=""}[5m])) by (cluster, namespace)`,

	"cluster_throughput_writes_by_namespace": `sum(rate(container_fs_writes_bytes_total{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor", ` +
		`device=~"(/dev/)?(mmcblk.p.+|nvme.+|rbd.+|sd.+|vd.+|xvd.+|dm-.+|md.+|dasd.+)", container!="", ` +
		`namespace!=""}[5m])) by (cluster, namespace)`,

	"cluster_throughput_reads_writes_by_namespace": `sum(rate(container_fs_reads_bytes_total{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor", ` +
		`device=~"(/dev/)?(mmcblk.p.+|nvme.+|rbd.+|sd.+|vd.+|xvd.+|dm-.+|md.+|dasd.+)", container!="", ` +
		`namespace!=""}[5m])) by (cluster, namespace) + sum(rate(container_fs_writes_bytes_total{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor", ` +
		`device=~"(/dev/)?(mmcblk.p.+|nvme.+|rbd.+|sd.+|vd.+|xvd.+|dm-.+|md.+|dasd.+)", container!="", ` +
		`namespace!=""}[5m])) by (cluster, namespace)`,

	"cluster_disk_size_usage": `sum(node_filesystem_size_bytes{mountpoint="/"}) by (cluster) - ` +
		`sum(node_filesystem_free_bytes{mountpoint="/"}) by (cluster)`,

	"cluster_disk_size_total": `sum(node_filesystem_size_bytes{mountpoint="/"}) by (cluster)`,

	"cluster_disk_size_utilisation": `(sum(node_filesystem_size_bytes{mountpoint="/"}) by (cluster) - ` +
		`sum(node_filesystem_free_bytes{mountpoint="/"}) by (cluster)) / ` +
		`sum(node_filesystem_size_bytes{mountpoint="/"}) by (cluster)`,

	"cluster_disk_read_iops": `sum(rate(node_disk_reads_completed_total[5m])) by (cluster)`,

	"cluster_disk_write_iops": `sum(rate(node_disk_writes_completed_total[5m])) by (cluster)`,

	"cluster_disk_read_throughput": `sum(rate(node_disk_read_bytes_total[5m])) by (cluster)`,

	"cluster_disk_write_throughput": `sum(rate(node_disk_written_bytes_total[5m])) by (cluster)`,

	"cluster_disk_inode_usage": `sum(node_filesystem_files{mountpoint="/"}) by (cluster) - ` +
		`sum(node_filesystem_files_free{mountpoint="/"}) by (cluster)`,

	"cluster_disk_inode_total": `sum(node_filesystem_files{mountpoint="/"}) by (cluster)`,

	"cluster_disk_inode_utilisation": `1 - (sum(node_filesystem_files_free{mountpoint="/"}) by (cluster)) / ` +
		`sum(node_filesystem_files{mountpoint="/"}) by (cluster)`,

	"cluster_network_bytes_transmitted": `sum(rate(node_network_receive_bytes_total{job="node-exporter", ` +
		`device!="lo"}[5m])) by (cluster)`,

	"cluster_network_bytes_received": `sum(rate(node_network_transmit_bytes_total{job="node-exporter", ` +
		`device!="lo"}[5m])) by (cluster)`,

	// node
	"node_cpu_usage": `sum(label_replace(sum(machine_cpu_cores) by (cluster, node), "instance", "$1", ` +
		`"node", "(.*)")) by (cluster, instance) * (1 - (avg(irate(node_cpu_seconds_total{mode="idle"}[5m])) ` +
		`by (cluster, instance)))`,

	"node_cpu_total": `sum(label_replace(sum(machine_cpu_cores) by (cluster, node), "instance", "$1", ` +
		`"node", "(.*)")) by (cluster, instance)`,

	"node_cpu_utilisation": `1 - (avg(irate(node_cpu_seconds_total{mode="idle"}[5m])) by (cluster, instance))`,

	"node_cpu_requests_utilisation": `sum(label_replace(` +
		`node_namespace_pod_container:container_cpu_usage_seconds_total:sum_irate, "instance", "$1", "node", "(.*)")) ` +
		`by (cluster, instance) / ` +
		`sum(label_replace(cluster:namespace:pod_cpu:active:kube_pod_container_resource_requests, ` +
		`"instance", "$1", "node", "(.*)")) by (cluster, instance)`,

	"node_cpu_limits_utilisation": `sum(label_replace(` +
		`node_namespace_pod_container:container_cpu_usage_seconds_total:sum_irate, "instance", "$1", "node", "(.*)")) ` +
		`by (cluster, instance) / ` +
		`sum(label_replace(cluster:namespace:pod_cpu:active:kube_pod_container_resource_limits, ` +
		`"instance", "$1", "node", "(.*)")) by (cluster, instance)`,

	"node_1m_load_average": `sum(node_load1{job="node-exporter"}) by (cluster, instance)`,

	"node_5m_load_average": `sum(node_load5{job="node-exporter"}) by (cluster, instance)`,

	"node_15m_load_average": `sum(node_load15{job="node-exporter"}) by (cluster, instance)`,

	"node_memory_used": `sum(node_memory_MemTotal_bytes{job="node-exporter"}) by (cluster, instance) - ` +
		`sum(node_memory_Buffers_bytes{job="node-exporter"}) by (cluster, instance) - ` +
		`sum(node_memory_Cached_bytes{job="node-exporter"}) by (cluster, instance) - ` +
		`sum(node_memory_MemFree_bytes{job="node-exporter"}) by (cluster, instance)`,

	"node_memory_buffers": `sum(node_memory_Buffers_bytes{job="node-exporter"}) by (cluster, instance)`,

	"node_memory_cached": `sum(node_memory_Cached_bytes{job="node-exporter"}) by (cluster, instance)`,

	"node_memory_free": `sum(node_memory_MemFree_bytes{job="node-exporter"}) by (cluster, instance)`,

	"node_memory_total": `sum(node_memory_MemTotal_bytes{job="node-exporter"}) by (cluster, instance)`,

	"node_memory_usage": `sum(node_memory_MemTotal_bytes{job="node-exporter"}) by (cluster, instance) - ` +
		`sum(node_memory_MemFree_bytes{job="node-exporter"} + node_memory_Buffers_bytes{job="node-exporter"} + ` +
		`node_memory_Cached_bytes{job="node-exporter"}) by (cluster, instance)`,

	"node_memory_utilisation": `(sum(node_memory_MemTotal_bytes{job="node-exporter"}) by (cluster, instance) - ` +
		`sum(node_memory_MemFree_bytes{job="node-exporter"} + node_memory_Buffers_bytes{job="node-exporter"} + ` +
		`node_memory_Cached_bytes{job="node-exporter"}) by (cluster, instance))  / ` +
		`sum(node_memory_MemTotal_bytes{job="node-exporter"}) by (cluster, instance)`,

	"node_memory_requests_utilisation": `sum(label_replace(` +
		`node_namespace_pod_container:container_memory_working_set_bytes, "instance", "$1", "node", "(.*)")) ` +
		`by (cluster, instance)  / sum(label_replace(` +
		`cluster:namespace:pod_memory:active:kube_pod_container_resource_requests, "instance", "$1", ` +
		`"node", "(.*)")) by (cluster, instance)`,

	"node_memory_limits_utilisation": `sum(label_replace(` +
		`node_namespace_pod_container:container_memory_working_set_bytes, "instance", "$1", "node", "(.*)")) ` +
		`by (cluster, instance)  / sum(label_replace(` +
		`cluster:namespace:pod_memory:active:kube_pod_container_resource_limits, "instance", "$1", ` +
		`"node", "(.*)")) by (cluster, instance)`,

	"node_disk_read_bytes_by_device": `sum(rate(node_disk_read_bytes_total{job="node-exporter", ` +
		`device=~"(/dev/)?(mmcblk.p.+|nvme.+|rbd.+|sd.+|vd.+|xvd.+|dm-.+|md.+|dasd.+)"}[5m])) by (cluster, instance, device)`,

	"node_disk_written_bytes_by_device": `sum(rate(node_disk_written_bytes_total{job="node-exporter", ` +
		`device=~"(/dev/)?(mmcblk.p.+|nvme.+|rbd.+|sd.+|vd.+|xvd.+|dm-.+|md.+|dasd.+)"}[5m])) by (cluster, instance, device)`,

	"node_disk_io_time_seconds_by_device": `sum(rate(node_disk_io_time_seconds_total{job="node-exporter", ` +
		`device=~"(/dev/)?(mmcblk.p.+|nvme.+|rbd.+|sd.+|vd.+|xvd.+|dm-.+|md.+|dasd.+)"}[5m])) by (cluster, instance, device)`,

	"node_filesystem_size_bytes_by_mountpoint": `max(node_filesystem_size_bytes{job="node-exporter", fstype!="", ` +
		`mountpoint!=""}) by (cluster, instance, mountpoint)`,

	"node_filesystem_size_bytes_avail_by_mountpoint": `max(node_filesystem_avail_bytes{job="node-exporter", ` +
		`fstype!="", mountpoint!=""}) by (cluster, instance, mountpoint)`,

	"node_filesystem_size_bytes_used_by_mountpoint": `max(node_filesystem_size_bytes{job="node-exporter", ` +
		`fstype!="", mountpoint!=""}) by (cluster, instance, mountpoint) - ` +
		`max(node_filesystem_avail_bytes{job="node-exporter", fstype!="", mountpoint!=""}) ` +
		`by (cluster, instance, mountpoint)`,

	"node_filesystem_size_bytes_utilisation_by_mountpoint": `(max(node_filesystem_size_bytes{job="node-exporter", ` +
		`fstype!="", mountpoint!=""}) by (cluster, instance, mountpoint) - ` +
		`max(node_filesystem_avail_bytes{job="node-exporter", fstype!="", mountpoint!=""}) ` +
		`by (cluster, instance, mountpoint)) / max(node_filesystem_size_bytes{job="node-exporter", fstype!="", ` +
		`mountpoint!=""}) by (cluster, instance, mountpoint)`,

	"node_disk_read_iops": `sum(rate(node_disk_reads_completed_total{job="node-exporter"}[5m])) ` +
		`by (cluster, instance)`,

	"node_disk_write_iops": `sum(rate(node_disk_writes_completed_total{job="node-exporter"}[5m])) ` +
		`by (cluster, instance)`,

	"node_disk_read_throughput": `sum(rate(node_disk_read_bytes_total{job="node-exporter"}[5m])) ` +
		`by (cluster, instance)`,

	"node_disk_write_throughput": `sum(rate(node_disk_written_bytes_tota{job="node-exporter"}[5m])) ` +
		`by (cluster, instance)`,

	"node_network_bytes_received_by_device": `sum(rate(node_network_receive_bytes_total{job="node-exporter", ` +
		`device!="lo"}[5m])) by (cluster, instance, device)`,

	"node_network_bytes_transmitted_by_device": `sum(rate(node_network_transmit_bytes_total{job="node-exporter", ` +
		`device!="lo"}[5m])) by (cluster, instance, device)`,

	"node_disk_size_usage": `sum(node_filesystem_size_bytes{mountpoint="/"}) by (cluster, instance) - ` +
		`sum(node_filesystem_free_bytes{mountpoint="/"}) by (cluster, instance)`,

	"node_disk_size_total": `sum(node_filesystem_size_bytes{mountpoint="/"}) by (cluster, instance)`,

	"node_disk_size_utilisation": `(sum(node_filesystem_size_bytes{mountpoint="/"}) by (cluster, instance) - ` +
		`sum(node_filesystem_free_bytes{mountpoint="/"}) by (cluster, instance)) / ` +
		`sum(node_filesystem_size_bytes{mountpoint="/"}) by (cluster, instance)`,

	"node_disk_inode_utilisation": `1 - (sum(node_filesystem_files_free{mountpoint="/"}) by (cluster, instance)) / ` +
		`sum(node_filesystem_files{mountpoint="/"}) by (cluster, instance)`,

	// workload
	"workload_cpu_usage": `sum by (cluster, namespace, pod) (irate(container_cpu_usage_seconds_total{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor", container!="", ` +
		`image!=""}[5m])) * on (cluster, namespace, pod) group_left(owner_kind, ` +
		`owner_name) kube_pod_owner{job="kube-state-metrics"} * on (cluster, namespace, ` +
		`pod) group_left(node) kube_pod_info{job="kube-state-metrics"}`,

	"workload_memory_usage": `sum by (cluster, namespace, pod) (container_memory_usage_bytes{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor", container!="", image!=""}) * on (cluster, namespace, ` +
		`pod) group_left(owner_kind, owner_name) kube_pod_owner{job="kube-state-metrics"} * on (cluster, ` +
		`namespace, pod) group_left(node) kube_pod_info{job="kube-state-metrics"}`,

	"workload_network_packets_received": `sum by (cluster, namespace, ` +
		`pod) (irate(container_network_receive_packets_total{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor"}[5m])) * on (cluster, namespace, pod) group_left(owner_kind, ` +
		`owner_name) kube_pod_owner{job="kube-state-metrics"} * on (cluster, namespace, ` +
		`pod) group_left(node) kube_pod_info{job="kube-state-metrics"}`,

	"workload_network_packets_transmitted": `sum by (cluster, namespace, ` +
		`pod) (irate(container_network_transmit_packets_total{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor"}[5m])) * on (cluster, namespace, pod) group_left(owner_kind, ` +
		`owner_name) kube_pod_owner{job="kube-state-metrics"} * on (cluster, namespace, ` +
		`pod) group_left(node) kube_pod_info{job="kube-state-metrics"}`,

	"workload_network_packets_received_dropped": `sum by (cluster, namespace, ` +
		`pod) (irate(container_network_receive_packets_dropped_total{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor"}[5m])) * on (cluster, namespace, pod) group_left(owner_kind, ` +
		`owner_name) kube_pod_owner{job="kube-state-metrics"} * on (cluster, namespace, ` +
		`pod) group_left(node) kube_pod_info{job="kube-state-metrics"}`,

	"workload_network_packets_transmitted_dropped": `sum by (cluster, namespace, ` +
		`pod) (irate(container_network_transmit_packets_dropped_total{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor"}[5m])) * on (cluster, namespace, pod) group_left(owner_kind, ` +
		`owner_name) kube_pod_owner{job="kube-state-metrics"} * on (cluster, namespace, ` +
		`pod) group_left(node) kube_pod_info{job="kube-state-metrics"}`,

	// pod
	"pod_cpu_usage": `sum(irate(container_cpu_usage_seconds_total{image!="", job="kubelet", ` +
		`metrics_path="/metrics/cadvisor"}[5m])) by (cluster, namespace, pod, container)`,

	"pod_memory_usage": `sum(container_memory_usage_bytes{job="kubelet", metrics_path="/metrics/cadvisor", ` +
		`container!="", image!=""}) by (cluster, namespace, pod, container)`,

	"pod_network_bytes_received": `sum(rate(container_network_receive_bytes_total{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor", image!=""}[5m])) by (cluster, namespace, pod)`,

	"pod_network_bytes_transmitted": `sum(rate(container_network_transmit_bytes_total{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor", image!=""}[5m])) by (cluster, namespace, pod)`,

	"pod_network_packets_received": `sum(irate(container_network_receive_packets_total{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor"}[5m])) by (cluster, namespace, pod)`,

	"pod_network_packets_transmitted": `sum(irate(container_network_transmit_packets_total{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor"}[5m])) by (cluster, namespace, pod)`,

	"pod_network_packets_received_dropped": `sum(irate(container_network_receive_packets_dropped_total{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor"}[5m])) by (cluster, namespace, pod)`,

	"pod_network_packets_transmitted_dropped": `sum(irate(` +
		`container_network_transmit_packets_dropped_total{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor"}[5m])) by (cluster, namespace, pod)`,

	"pod_iops_reads": `sum(rate(container_fs_reads_total{job="kubelet", metrics_path="/metrics/cadvisor", ` +
		`device=~"(/dev/)?(mmcblk.p.+|nvme.+|rbd.+|sd.+|vd.+|xvd.+|dm-.+|md.+|dasd.+)", ` +
		`container!=""}[5m])) by (cluster, namespace, pod, container)`,

	"pod_iops_writes": `sum(rate(container_fs_writes_total{job="kubelet", metrics_path="/metrics/cadvisor", ` +
		`device=~"(/dev/)?(mmcblk.p.+|nvme.+|rbd.+|sd.+|vd.+|xvd.+|dm-.+|md.+|dasd.+)", ` +
		`container!=""}[5m])) by (cluster, namespace, pod, container)`,

	"pod_iops_reads_writes": `sum(rate(container_fs_reads_total{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor", ` +
		`device=~"(/dev/)?(mmcblk.p.+|nvme.+|rbd.+|sd.+|vd.+|xvd.+|dm-.+|md.+|dasd.+)", ` +
		`container!=""}[5m])) by (cluster, namespace, pod, container) + ` +
		`sum(rate(container_fs_writes_total{job="kubelet", metrics_path="/metrics/cadvisor", ` +
		`device=~"(/dev/)?(mmcblk.p.+|nvme.+|rbd.+|sd.+|vd.+|xvd.+|dm-.+|md.+|dasd.+)", ` +
		`container!=""}[5m]))  by (cluster, namespace, pod, container)`,

	"pod_throughput_reads": `sum(rate(container_fs_reads_bytes_total{job="kubelet", metrics_path="/metrics/cadvisor", ` +
		`device=~"(/dev/)?(mmcblk.p.+|nvme.+|rbd.+|sd.+|vd.+|xvd.+|dm-.+|md.+|dasd.+)", ` +
		`container!=""}[5m])) by (cluster, namespace, pod, container)`,

	"pod_throughput_writes": `sum(rate(container_fs_writes_bytes_total{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor", ` +
		`device=~"(/dev/)?(mmcblk.p.+|nvme.+|rbd.+|sd.+|vd.+|xvd.+|dm-.+|md.+|dasd.+)", ` +
		`container!=""}[5m])) by (cluster, namespace, pod, container)`,

	"pod_throughput_reads_writes": `sum(rate(container_fs_reads_bytes_total{job="kubelet", ` +
		`metrics_path="/metrics/cadvisor", ` +
		`device=~"(/dev/)?(mmcblk.p.+|nvme.+|rbd.+|sd.+|vd.+|xvd.+|dm-.+|md.+|dasd.+)", ` +
		`container!=""}[5m])) by (cluster, namespace, pod, container) + ` +
		`sum(rate(container_fs_writes_bytes_total{job="kubelet", metrics_path="/metrics/cadvisor", ` +
		`device=~"(/dev/)?(mmcblk.p.+|nvme.+|rbd.+|sd.+|vd.+|xvd.+|dm-.+|md.+|dasd.+)", ` +
		`container!=""}[5m])) by (cluster, namespace, pod, container)`,
}
