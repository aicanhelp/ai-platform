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

var promQLTemplatesComponents = map[string]string{
	// kube-apiserver
	"kube_apiserver_qps": `sum(rate(apiserver_request_total{job="apiserver"}[5m])) by (cluster)`,

	"kube_apiserver_qps_by_verb": `sum(rate(apiserver_request_total{job="apiserver", verb=~"$verb"}[5m])) ` +
		`by (cluster, verb)`,

	"kube_apiserver_qps_by_verb_resource": `sum(rate(apiserver_request_total{job="apiserver", verb=~"$verb", ` +
		`resource=~"$resource"}[5m])) by (cluster, verb, resource)`,

	"kube_apiserver_non_2xx_read_qps": `sum(irate(apiserver_request_total{job="apiserver", verb=~"GET|LIST", ` +
		`verb=~"$verb", resource=~"$resource", code!~"2.*"}[5m])) by (cluster, verb, resource, code)`,

	"kube_apiserver_non_2xx_write_qps": `sum(irate(apiserver_request_total{job="apiserver", ` +
		`verb=~"POST|PUT|PATCH|DELETE", verb=~"$verb", resource=~"$resource", code!~"2.*"}[5m])) ` +
		`by (cluster, verb, resource, code)`,

	"kube_apiserver_read_request_success_rate": `sum(code_resource:apiserver_request_total:rate5m{verb="read", ` +
		`code=~"20.*"}) by (cluster) / sum(code_resource:apiserver_request_total:rate5m{verb="read"}) by (cluster)`,

	"kube_apiserver_write_request_success_rate": `sum(code_resource:apiserver_request_total:rate5m{verb="write", ` +
		`code=~"20.*"}) by (cluster) / sum(code_resource:apiserver_request_total:rate5m{verb="write"}) by (cluster)`,

	"kube_apiserver_current_inflight_read_request": `sum(apiserver_current_inflight_requests{job="apiserver", ` +
		`request_kind="readOnly"}) by (cluster)`,

	"kube_apiserver_current_inflight_write_request": `sum(apiserver_current_inflight_requests{job="apiserver", ` +
		`request_kind="mutating"}) by (cluster)`,

	"kube_apiserver_get_request_latency": `histogram_quantile(0.99, ` +
		`sum(irate(apiserver_request_duration_seconds_bucket{job="apiserver", verb="GET", ` +
		`resource!="",subresource!~"log|proxy"}[5m])) by (cluster, pod, verb, resource, subresource, scope, le))`,

	"kube_apiserver_list_request_latency": `histogram_quantile(0.99, ` +
		`sum(irate(apiserver_request_duration_seconds_bucket{job="apiserver", ` +
		`verb="LIST"}[5m])) by (cluster, pod_name, verb, resource, scope, le))`,

	"kube_apiserver_request_latency_by_verb": `histogram_quantile(0.99, ` +
		`sum(irate(apiserver_request_duration_seconds_bucket{job="apiserver", verb=~"$verb", ` +
		`verb!~"WATCH|CONNECT", resource!=""}[5m])) by (cluster, le, verb, resource))`,

	"kube_apiserver_request_latency_by_verb_resource": `histogram_quantile(0.99, ` +
		`sum(irate(apiserver_request_duration_seconds_bucket{job="apiserver", verb=~"$verb", ` +
		`verb!~"WATCH|CONNECT", resource=~"$resource", resource!=""}[5m])) by (cluster, le, verb, resource))`,

	"kube_apiserver_admission_controller_admission_latency_admit": `histogram_quantile(0.99, ` +
		`sum(irate(apiserver_admission_controller_admission_duration_seconds_bucket{job="apiserver", ` +
		`type="admit"}[5m])) by (cluster, operation, name, le, type, rejected))`,

	"kube_apiserver_admission_controller_admission_latency_validate": `histogram_quantile(0.99, ` +
		`sum(irate(apiserver_admission_controller_admission_duration_seconds_bucket{job="apiserver", ` +
		`type="validate"}[5m])) by (cluster, operation, name, le, type, rejected))`,

	"kube_apiserver_admission_webhook_admission_latency_admit": `histogram_quantile(0.99, ` +
		`sum(irate(apiserver_admission_webhook_admission_duration_seconds_bucket{job="apiserver", ` +
		`type="admit"}[5m])) by (cluster, operation, name, le, type, rejected))`,

	"kube_apiserver_admission_webhook_admission_latency_validate": `histogram_quantile(0.99, ` +
		`sum(irate(apiserver_admission_webhook_admission_duration_seconds_bucket{job="apiserver", ` +
		`type="validating"}[5m])) by (cluster, operation, name, le, type, rejected))`,

	"kube_apiserver_admission_webhook_admission_latency_qps": `sum(irate(` +
		`apiserver_admission_webhook_admission_duration_seconds_count{job="apiserver"}[5m]))by(cluster, ` +
		`name, operation, type, rejected)`,

	"kube_apiserver_cpu_usage": `sum(rate(process_cpu_seconds_total{job="apiserver"}[5m])) by (cluster, instance)`,

	"kube_apiserver_memory_usage": `sum(process_resident_memory_bytes{job="apiserver"}) by (cluster, instance)`,

	"kube_apiserver_goroutines": `sum(go_goroutines{job="apiserver"}) by (cluster, instance)`,

	// etcd
	"etcd_server_has_leader": `sum(etcd_server_has_leader{job="etcd"}) by (cluster)`,

	"etcd_server_is_leader": `sum(etcd_server_is_leader{job="etcd"}) by (cluster, instance)`,

	"etcd_up_instances_count": `count(up{job="etcd"} == 1) by (cluster)`,

	"etcd_down_instances_count": `count(up{job="etcd"} < 1) by (cluster)`,

	"etcd_instances_total": `count(up{job="etcd"}) by (cluster)`,

	"etcd_server_leader_changes_seen_1day": `sum(changes(` +
		`etcd_server_leader_changes_seen_total{job="etcd"}[1d])) by (cluster)`,

	"etcd_backend_db_total": `sum(etcd_mvcc_db_total_size_in_bytes{job="etcd"}) by (cluster)`,

	"etcd_backend_db_usage": `sum(etcd_mvcc_db_total_size_in_use_in_bytes{job="etcd"}) by (cluster)`,

	"etcd_backend_db_commit_latency": `histogram_quantile(0.99, ` +
		`sum(rate(etcd_disk_backend_commit_duration_seconds_bucket{job="etcd", ` +
		`instance=~"$instance"}[5m])) by (cluster, instance, le))`,

	"etcd_debugging_mvcc_keys_total": `sum(etcd_debugging_mvcc_keys_total{job="etcd", ` +
		`instance=~"$instance"}) by (cluster, instance)`,

	"etcd_kube_api_request_latency": `histogram_quantile(0.99, ` +
		`sum(irate(etcd_request_duration_seconds_bucket[5m])) by (cluster, le,operation,type,instance))`,

	"etcd_cpu_usage": `sum(rate(process_cpu_seconds_total{job="etcd", ` +
		`instance=~"$instance"}[5m])) by (cluster, instance)`,

	"etcd_memory_usage": `sum(process_resident_memory_bytes{job="etcd", ` +
		`instance=~"$instance"}) by (cluster, instance)`,

	"etcd_goroutines": `sum(go_goroutines{job="etcd", ` +
		`instance=~"$instance"}) by (cluster, instance)`,

	// kube-scheduler
	"kube_scheduler_cache_size": `sum(scheduler_scheduler_cache_size{job="kube-scheduler", ` +
		`instance=~"$instance"}) by (cluster, instance, type)`,

	"kube_scheduler_pending_pods_count": `sum(scheduler_pending_pods{job="kube-scheduler", ` +
		`instance=~"$instance"})by(cluster, instance, queue)`,

	"kube_scheduler_pod_scheduling_attempts": `histogram_quantile(0.99, ` +
		`sum(rate(scheduler_pod_scheduling_attempts_bucket{job="kube-scheduler", ` +
		`instance=~"$instance"}[5m])) by (cluster, instance, pod, le))`,

	"kube_scheduler_kube_api_2XX_request_rate": `sum(rate(rest_client_requests_total{job="kube-scheduler", ` +
		`code=~"2..", instance=~"$instance"}[5m])) by (cluster, instance)`,

	"kube_scheduler_kube_api_3XX_request_rate": `sum(rate(rest_client_requests_total{job="kube-scheduler", ` +
		`code=~"3..", instance=~"$instance"}[5m])) by (cluster, instance)`,

	"kube_scheduler_kube_api_4XX_request_rate": `sum(rate(rest_client_requests_total{job="kube-scheduler", ` +
		`code=~"4..", instance=~"$instance"}[5m])) by (cluster, instance)`,

	"kube_scheduler_kube_api_5XX_request_rate": `sum(rate(rest_client_requests_total{job="kube-scheduler", ` +
		`code=~"5..", instance=~"$instance"}[5m])) by (cluster, instance)`,

	"kube_scheduler_kube_api_request_latency": `histogram_quantile(0.99, ` +
		`sum(rate(rest_client_request_duration_seconds_bucket{job="kube-scheduler", ` +
		`instance=~"$instance"}[5m])) by (cluster, instance, verb, url, le))`,

	"kube_scheduler_cpu_usage": `sum(rate(process_cpu_seconds_total{job="kube-scheduler", ` +
		`instance=~"$instance"}[5m])) by (cluster, instance)`,

	"kube_scheduler_memory_usage": `sum(process_resident_memory_bytes{job="kube-scheduler", ` +
		`instance=~"$instance"}) by (cluster, instance)`,

	"kube_scheduler_goroutines": `sum(go_goroutines{job="kube-scheduler", ` +
		`instance=~"$instance"}) by (cluster, instance)`,

	// kube-controller-manager
	"kube_controller_manager_workqueue_adds_rate": `sum(rate(workqueue_adds_total{job="kube-controller-manager", ` +
		`instance=~"$instance"}[5m])) by (cluster, instance, name)`,

	"kube_controller_manager_workqueue_depth": `sum(rate(workqueue_depth{job="kube-controller-manager", ` +
		`instance=~"$instance"}[5m])) by (cluster, instance, name)`,

	"kube_controller_manager_workqueue_latency": `histogram_quantile(0.99, ` +
		`sum(rate(workqueue_queue_duration_seconds_bucket{job="kube-controller-manager", ` +
		`instance=~"$instance"}[5m])) by (cluster, instance, name, le))`,

	"kube_controller_manager_kube_api_2XX_request_rate": `sum(rate(` +
		`rest_client_requests_total{job="kube-controller-manager", code=~"2..", ` +
		`instance=~"$instance"}[5m])) by (cluster, instance)`,

	"kube_controller_manager_kube_api_3XX_request_rate": `sum(rate(` +
		`rest_client_requests_total{job="kube-controller-manager", code=~"3..", ` +
		`instance=~"$instance"}[5m])) by (cluster, instance)`,

	"kube_controller_manager_kube_api_4XX_request_rate": `sum(rate(` +
		`rest_client_requests_total{job="kube-controller-manager", code=~"4..", ` +
		`instance=~"$instance"}[5m])) by (cluster, instance)`,

	"kube_controller_manager_kube_api_5XX_request_rate": `sum(` +
		`rate(rest_client_requests_total{job="kube-controller-manager", code=~"5..", ` +
		`instance=~"$instance"}[5m])) by (cluster, instance)`,

	"kube_controller_manager_kube_api_request_latency": `histogram_quantile(0.99, ` +
		`sum(rate(rest_client_request_duration_seconds_bucket{job="kube-controller-manager", ` +
		`instance=~"$instance"}[5m])) by (cluster, instance, verb, url, le))`,

	"kube_controller_manager_cpu_usage": `sum(rate(` +
		`process_cpu_seconds_total{job="kube-controller-manager", ` +
		`instance=~"$instance"}[5m])) by (cluster, instance)`,

	"kube_controller_manager_memory_usage": `sum(` +
		`process_resident_memory_bytes{job="kube-controller-manager", ` +
		`instance=~"$instance"}) by (cluster, instance)`,

	"kube_controller_manager_goroutines": `sum(go_goroutines{job="kube-controller-manager", ` +
		`instance=~"$instance"}) by (cluster, instance)`,

	// kubelet
	"kubelet_running_count": `sum(kubelet_node_name{job="kubelet", metrics_path="/metrics"}) by (cluster)`,

	"kubelet_running_pods_count": `sum(kubelet_running_pods{job="kubelet", metrics_path="/metrics", ` +
		`instance=~"$instance"}) by (cluster, instance)`,

	"kubelet_running_containers_count": `sum(kubelet_running_containers{job="kubelet", metrics_path="/metrics", ` +
		`instance=~"$instance"}) by (cluster, instance)`,

	"kubelet_managed_actual_volumes_count": `sum(volume_manager_total_volumes{job="kubelet", metrics_path="/metrics", ` +
		`instance=~"$instance", state="actual_state_of_world"}) by (cluster, instance)`,

	"kubelet_managed_desired_volumes_count": `sum(volume_manager_total_volumes{job="kubelet", metrics_path="/metrics", ` +
		`instance=~"$instance", state="desired_state_of_world"}) by (cluster, instance)`,

	"kubelet_runtime_operation_rate": `sum(rate(kubelet_runtime_operations_total{job="kubelet", ` +
		`metrics_path="/metrics", instance=~"$instance"}[5m])) by (cluster, operation_type, instance)`,

	"kubelet_runtime_operation_error_rate": `sum(rate(kubelet_runtime_operations_errors_total{job="kubelet", ` +
		`metrics_path="/metrics", instance=~"$instance"}[5m])) by (cluster, instance, operation_type)`,

	"kubelet_pod_start_rate": `sum(rate(kubelet_pod_start_duration_seconds_count{job="kubelet", ` +
		`metrics_path="/metrics", instance=~"$instance"}[5m])) by (cluster, instance)`,

	"kubelet_pod_worker_rate": `sum(rate(kubelet_pod_worker_duration_seconds_count{job="kubelet", ` +
		`metrics_path="/metrics",instance=~"$instance"}[5m])) by (cluster, instance)`,

	"kubelet_storage_operation_rate": `sum(rate(storage_operation_duration_seconds_count{job="kubelet", ` +
		`metrics_path="/metrics", instance=~"$instance"}[5m])) by (cluster, instance, operation_name, volume_plugin)`,

	"kubelet_storage_operation_error_rate": `sum(rate(storage_operation_errors_total{job="kubelet", ` +
		`metrics_path="/metrics",instance=~"$instance"}[5m])) by (cluster, instance, operation_name, volume_plugin)`,

	"kubelet_cgroup_manager_operation_rate": `sum(rate(kubelet_cgroup_manager_duration_seconds_count{job="kubelet",  ` +
		`metrics_path="/metrics", instance=~"$instance"}[5m])) by (cluster, instance, operation_type)`,

	"kubelet_pleg_relist_rate": `sum(rate(kubelet_pleg_relist_duration_seconds_count{job="kubelet", ` +
		`metrics_path="/metrics", instance=~"$instance"}[5m])) by (cluster, instance)`,

	"kubelet_pleg_relist_interval": `histogram_quantile(0.99, ` +
		`sum(rate(kubelet_pleg_relist_interval_seconds_bucket{job="kubelet",  ` +
		`metrics_path="/metrics",instance=~"$instance"}[5m])) by (cluster, instance, le))`,

	"kubelet_pleg_relist_latency": `histogram_quantile(0.99,` +
		`sum(rate(kubelet_pleg_relist_duration_seconds_bucket{job="kubelet",` +
		`metrics_path="/metrics",instance=~"$instance"}[5m])) by (cluster, instance, le))`,

	"kubelet_kube_api_2XX_request_rate": `sum(rate(` +
		`rest_client_requests_total{job="kubelet", metrics_path="/metrics", code=~"2..", ` +
		`instance=~"$instance"}[5m])) by (cluster, instance)`,

	"kubelet_kube_api_3XX_request_rate": `sum(rate(` +
		`rest_client_requests_total{job="kubelet", metrics_path="/metrics", code=~"3..", ` +
		`instance=~"$instance"}[5m])) by (cluster, instance)`,

	"kubelet_kube_api_4XX_request_rate": `sum(rate(` +
		`rest_client_requests_total{job="kubelet", metrics_path="/metrics", code=~"4..", ` +
		`instance=~"$instance"}[5m])) by (cluster, instance)`,

	"kubelet_kube_api_5XX_request_rate": `sum(` +
		`rate(rest_client_requests_total{job="kubelet", metrics_path="/metrics", code=~"5..", ` +
		`instance=~"$instance"}[5m])) by (cluster, instance)`,

	"kubelet_kube_api_request_latency": `histogram_quantile(0.99, ` +
		`sum(rate(rest_client_request_duration_seconds_bucket{job="kubelet", metrics_path="/metrics", ` +
		`instance=~"$instance"}[5m])) by (cluster, instance, verb, url, le))`,

	"kubelet_cpu_usage": `sum(rate(` +
		`process_cpu_seconds_total{job="kubelet", metrics_path="/metrics", ` +
		`instance=~"$instance"}[5m])) by (cluster, instance)`,

	"kubelet_memory_usage": `sum(` +
		`process_resident_memory_bytes{job="kubelet", metrics_path="/metrics", ` +
		`instance=~"$instance"}) by (cluster, instance)`,

	"kubelet_goroutines": `sum(go_goroutines{job="kubelet", metrics_path="/metrics", ` +
		`instance=~"$instance"}) by (cluster, instance)`,

	// kube-proxy
	"kube_proxy_up_instances_count": `sum(up{job="kube-proxy", instance=~"$instance"}) by (cluster, instance)`,

	"kube_proxy_sync_proxy_rules_rate": `sum(rate(kubeproxy_sync_proxy_rules_duration_seconds_count{` +
		`job="kube-proxy", instance=~"$instance"}[5m])) by (cluster, instance)`,

	"kube_proxy_network_programming_rate": `sum(rate(kubeproxy_network_programming_duration_seconds_count{` +
		`job="kube-proxy", instance=~"$instance"}[5m])) by (cluster, instance)`,

	"kube_proxy_kube_api_2XX_request_rate": `sum(rate(` +
		`rest_client_requests_total{job="kube-proxy", code=~"2..", ` +
		`instance=~"$instance"}[5m])) by (cluster, instance)`,

	"kube_proxy_kube_api_3XX_request_rate": `sum(rate(` +
		`rest_client_requests_total{job="kube-proxy", code=~"3..", ` +
		`instance=~"$instance"}[5m])) by (cluster, instance)`,

	"kube_proxy_kube_api_4XX_request_rate": `sum(rate(` +
		`rest_client_requests_total{job="kube-proxy", code=~"4..", ` +
		`instance=~"$instance"}[5m])) by (cluster, instance)`,

	"kube_proxy_kube_api_5XX_request_rate": `sum(` +
		`rate(rest_client_requests_total{job="kube-proxy", code=~"5..", ` +
		`instance=~"$instance"}[5m])) by (cluster, instance)`,

	"kube_proxy_cpu_usage": `sum(rate(` +
		`process_cpu_seconds_total{job="kube-proxy", instance=~"$instance"}[5m])) by (cluster, instance)`,

	"kube_proxy_memory_usage": `sum(` +
		`process_resident_memory_bytes{job="kube-proxy", ` +
		`instance=~"$instance"}) by (cluster, instance)`,

	"kube_proxy_goroutines": `sum(go_goroutines{job="kube-proxy", instance=~"$instance"}) by (cluster, instance)`,

	// coredns
	"coredns_requests_rate": `sum(increase(coredns_dns_requests_total{job="kube-dns",  ` +
		`instance=~"$instance"}[5m])) by (cluster, instance, proto)`,

	"coredns_requests_latency": `histogram_quantile(0.99, ` +
		`sum(rate(coredns_dns_request_duration_seconds_bucket{job="kube-dns",  ` +
		`instance=~"$instance"}[5m])) by(cluster, instance, server, zone, le))`,

	"coredns_udp_responses_size_bytes": `histogram_quantile(0.99, ` +
		`sum(rate(coredns_dns_response_size_bytes_bucket{job="kube-dns",  ` +
		`instance=~"$instance", proto="udp"}[5m])) by(cluster, instance, server, zone, le, proto))`,

	"coredns_tcp_responses_size_bytes": `histogram_quantile(0.99, ` +
		`sum(rate(coredns_dns_response_size_bytes_bucket{job="kube-dns",  ` +
		`instance=~"$instance", proto="tcp"}[5m])) by(cluster, instance, server, zone, le, proto))`,

	"coredns_NXDOMAIN_response_rate": `sum(rate(coredns_dns_responses_total{job="kube-dns", ` +
		`rcode="NXDOMAIN", instance=~"$instance"}[5m])) by (cluster, instance) ` +
		`/ sum(rate(coredns_dns_responses_total{job="kube-dns", instance=~"$instance"}[5m])) by (cluster, instance)`,

	"coredns_panic_count": `sum(increase(coredns_panics_total{instance=~"$instance"}[1m])) by (cluster, instance)`,

	"coredns_cache_size": `sum(coredns_cache_entries{job="kube-dns", instance=~"$instance"}) by (cluster, ` +
		`instance, type)`,

	"coredns_cache_hitrate": `sum(rate(coredns_cache_hits_total{instance=~"$instance"}[5m])) by (cluster, instance) / ` +
		`sum(rate(coredns_cache_requests_total{instance=~"$instance"}[5m])) by (cluster, instance)`,
}
