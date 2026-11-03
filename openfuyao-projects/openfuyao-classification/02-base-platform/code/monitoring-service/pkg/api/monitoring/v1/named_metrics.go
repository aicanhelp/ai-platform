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

package v1

// controlPlaneComponentsKubeApiServerMetrics lists metric identifiers each paired with a custom PromQL query designed
// to monitor specific aspects of Kube Apiserver performance.
var controlPlaneComponentsKubeApiServerMetrics = []string{
	"kube_apiserver_qps",
	"kube_apiserver_qps_by_verb",
	"kube_apiserver_qps_by_verb_resource",
	"kube_apiserver_non_2xx_read_qps",
	"kube_apiserver_non_2xx_write_qps",
	"kube_apiserver_read_request_success_rate",
	"kube_apiserver_write_request_success_rate",
	"kube_apiserver_current_inflight_read_request",
	"kube_apiserver_current_inflight_write_request",
	"kube_apiserver_get_request_latency",
	"kube_apiserver_list_request_latency",
	"kube_apiserver_request_latency_by_verb",
	"kube_apiserver_request_latency_by_verb_resource",
	"kube_apiserver_admission_controller_admission_latency_admit",
	"kube_apiserver_admission_controller_admission_latency_validate",
	"kube_apiserver_admission_webhook_admission_latency_admit",
	"kube_apiserver_admission_webhook_admission_latency_validate",
	"kube_apiserver_admission_webhook_admission_latency_qps",
	"kube_apiserver_cpu_usage",
	"kube_apiserver_memory_usage",
	"kube_apiserver_goroutines",
}

// controlPlaneComponentsETCDMetrics lists metric identifiers each paired with a custom PromQL query designed
// to monitor specific aspects of ETCD performance.
var controlPlaneComponentsETCDMetrics = []string{
	"etcd_server_has_leader",
	"etcd_server_is_leader",
	"etcd_up_instances_count",
	"etcd_down_instances_count",
	"etcd_instances_total",
	"etcd_server_leader_changes_seen_1day",
	"etcd_backend_db_total",
	"etcd_backend_db_usage",
	"etcd_backend_db_commit_latency",
	"etcd_debugging_mvcc_keys_total",
	"etcd_kube_api_request_latency",
	"etcd_cpu_usage",
	"etcd_memory_usage",
	"etcd_goroutines",
}

// controlPlaneComponentsKubeSchedulerMetrics lists metric identifiers each paired with a custom PromQL query designed
// to monitor specific aspects of Kube Scheduler performance.
var controlPlaneComponentsKubeSchedulerMetrics = []string{
	"kube_scheduler_cache_size",
	"kube_scheduler_pending_pods_count",
	"kube_scheduler_pod_scheduling_attempts",
	"kube_scheduler_kube_api_2XX_request_rate",
	"kube_scheduler_kube_api_3XX_request_rate",
	"kube_scheduler_kube_api_4XX_request_rate",
	"kube_scheduler_kube_api_5XX_request_rate",
	"kube_scheduler_kube_api_request_latency",
	"kube_scheduler_cpu_usage",
	"kube_scheduler_memory_usage",
	"kube_scheduler_goroutines",
}

// controlPlaneComponentsKubeControllerManagerMetrics lists metric identifiers each paired with a custom PromQL query
// designed to monitor specific aspects of Kube Controller Manager performance.
var controlPlaneComponentsKubeControllerManagerMetrics = []string{
	"kube_controller_manager_workqueue_adds_rate",
	"kube_controller_manager_workqueue_depth",
	"kube_controller_manager_workqueue_latency",
	"kube_controller_manager_kube_api_2XX_request_rate",
	"kube_controller_manager_kube_api_3XX_request_rate",
	"kube_controller_manager_kube_api_4XX_request_rate",
	"kube_controller_manager_kube_api_5XX_request_rate",
	"kube_controller_manager_kube_api_request_latency",
	"kube_controller_manager_cpu_usage",
	"kube_controller_manager_memory_usage",
	"kube_controller_manager_goroutines",
}

// nodeComponentsKubeletMetrics lists metric identifiers each paired with a custom PromQL query
// designed to monitor specific aspects of Kubelet performance.
var nodeComponentsKubeletMetrics = []string{
	"kubelet_running_count",
	"kubelet_running_pods_count",
	"kubelet_running_containers_count",
	"kubelet_managed_actual_volumes_count",
	"kubelet_managed_desired_volumes_count",
	"kubelet_runtime_operation_rate",
	"kubelet_runtime_operation_error_rate",
	"kubelet_pod_start_rate",
	"kubelet_pod_worker_rate",
	"kubelet_storage_operation_rate",
	"kubelet_storage_operation_error_rate",
	"kubelet_cgroup_manager_operation_rate",
	"kubelet_pleg_relist_rate",
	"kubelet_pleg_relist_interval",
	"kubelet_pleg_relist_latency",
	"kubelet_kube_api_2XX_request_rate",
	"kubelet_kube_api_3XX_request_rate",
	"kubelet_kube_api_4XX_request_rate",
	"kubelet_kube_api_5XX_request_rate",
	"kubelet_kube_api_request_latency",
	"kubelet_cpu_usage",
	"kubelet_memory_usage",
	"kubelet_goroutines",
}

// nodeComponentsKubeProxyMetrics lists metric identifiers each paired with a custom PromQL query
// designed to monitor specific aspects of Kube Proxy performance.
var nodeComponentsKubeProxyMetrics = []string{
	"kube_proxy_up_instances_count",
	"kube_proxy_sync_proxy_rules_rate",
	"kube_proxy_network_programming_rate",
	"kube_proxy_kube_api_2XX_request_rate",
	"kube_proxy_kube_api_3XX_request_rate",
	"kube_proxy_kube_api_4XX_request_rate",
	"kube_proxy_kube_api_5XX_request_rate",
	"kube_proxy_cpu_usage",
	"kube_proxy_memory_usage",
	"kube_proxy_goroutines",
}

// addonsComponentsCorednsMetrics lists metric identifiers each paired with a custom PromQL query
// designed to monitor specific aspects of CoreDNS performance.
var addonsComponentsCorednsMetrics = []string{
	"coredns_requests_rate",
	"coredns_requests_latency",
	"coredns_udp_responses_size_bytes",
	"coredns_tcp_responses_size_bytes",
	"coredns_NXDOMAIN_response_rate",
	"coredns_panic_count",
	"coredns_cache_size",
	"coredns_cache_hitrate",
}

// clusterMetrics lists metric identifiers each paired with a custom PromQL query designed to monitor specific aspects
// of Kubernetes cluster performance.
var clusterMetrics = []string{
	"cluster_nodes_count",
	"cluster_nodes_not_ready_count",
	"cluster_pods_count",
	"cluster_cpu_usage",
	"cluster_cpu_usage_by_namespace",
	"cluster_cpu_total",
	"cluster_cpu_utilisation",
	"cluster_cpu_requests_total",
	"cluster_cpu_requests_utilisation",
	"cluster_cpu_limits_total",
	"cluster_cpu_limits_utilisation",
	"cluster_memory_usage",
	"cluster_memory_usage_without_cache_by_namespace",
	"cluster_memory_total",
	"cluster_memory_utilisation",
	"cluster_pod_count_by_namespace",
	"cluster_workload_count_by_namespace",
	"cluster_memory_requests_total_by_namespace",
	"cluster_memory_requests_total",
	"cluster_memory_requests_utilisation",
	"cluster_memory_requests_utilisation_by_namespace",
	"cluster_memory_limits_total_by_namespace",
	"cluster_memory_limits_total",
	"cluster_memory_limits_utilisation",
	"cluster_memory_limits_utilisation_by_namespace",
	"cluster_rate_of_received_packets_by_namespace",
	"cluster_rate_of_transmitted_packets_by_namespace",
	"cluster_rate_of_received_packets_dropped_by_namespace",
	"cluster_rate_of_transmitted_packets_dropped_by_namespace",
	"cluster_iops_reads_by_namespace",
	"cluster_iops_writes_by_namespace",
	"cluster_iops_reads_writes_by_namespace",
	"cluster_throughput_reads_by_namespace",
	"cluster_throughput_writes_by_namespace",
	"cluster_throughput_reads_writes_by_namespace",
	"cluster_disk_size_usage",
	"cluster_disk_size_total",
	"cluster_disk_size_utilisation",
	"cluster_disk_read_iops",
	"cluster_disk_write_iops",
	"cluster_disk_read_throughput",
	"cluster_disk_write_throughput",
	"cluster_disk_inode_usage",
	"cluster_disk_inode_total",
	"cluster_disk_inode_utilisation",
	"cluster_network_bytes_transmitted",
	"cluster_network_bytes_received",
}

// nodeMetrics lists metric identifiers each paired with a custom PromQL query designed to monitor specific aspects
// of Kubernetes node performance.
var nodeMetrics = []string{
	"node_cpu_usage",
	"node_cpu_total",
	"node_cpu_utilisation",
	"node_cpu_requests_utilisation",
	"node_cpu_limits_utilisation",
	"node_1m_load_average",
	"node_5m_load_average",
	"node_15m_load_average",
	"node_memory_used",
	"node_memory_buffers",
	"node_memory_cached",
	"node_memory_free",
	"node_memory_total",
	"node_memory_usage",
	"node_memory_utilisation",
	"node_memory_requests_utilisation",
	"node_memory_limits_utilisation",
	"node_disk_read_bytes_by_device",
	"node_disk_written_bytes_by_device",
	"node_disk_io_time_seconds_by_device",
	"node_filesystem_size_bytes_by_mountpoint",
	"node_filesystem_size_bytes_avail_by_mountpoint",
	"node_filesystem_size_bytes_used_by_mountpoint",
	"node_filesystem_size_bytes_utilisation_by_mountpoint",
	"node_disk_read_iops",
	"node_disk_write_iops",
	"node_disk_read_throughput",
	"node_disk_write_throughput",
	"node_network_bytes_received_by_device",
	"node_network_bytes_transmitted_by_device",
	"node_disk_size_usage",
	"node_disk_size_total",
	"node_disk_size_utilisation",
	"node_disk_inode_utilisation",
}

// workloadMetrics lists metric identifiers each paired with a custom PromQL query designed to monitor specific aspects
// of Kubernetes workload performance.
var workloadMetrics = []string{
	"workload_cpu_usage",
	"workload_memory_usage",
	"workload_network_packets_received",
	"workload_network_packets_transmitted",
	"workload_network_packets_received_dropped",
	"workload_network_packets_transmitted_dropped",
}

// podMetrics lists metric identifiers each paired with a custom PromQL query designed to monitor specific aspects
// of Kubernetes pod performance.
var podMetrics = []string{
	"pod_cpu_usage",
	"pod_memory_usage",
	"pod_network_bytes_received",
	"pod_network_bytes_transmitted",
	"pod_network_packets_received",
	"pod_network_packets_transmitted",
	"pod_network_packets_received_dropped",
	"pod_network_packets_transmitted_dropped",
	"pod_iops_reads",
	"pod_iops_writes",
	"pod_iops_reads_writes",
	"pod_throughput_reads",
	"pod_throughput_writes",
	"pod_throughput_reads_writes",
}

// containerMetrics lists metric identifiers each paired with a custom PromQL query designed to monitor specific aspects
// of Kubernetes container performance.
var containerMetrics = podMetrics
