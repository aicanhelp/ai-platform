import logging
import time
from typing import Dict, Optional, Any

from playwright.sync_api import CDPSession

from performance.runner.jsonl_writer import JsonlWriter
from performance.runner.network_monitor.network_request import NetworkRequest
from performance.runner.network_monitor.util import first_positive, match_url_to_entrypoint
from performance.runner.session import SessionContext


class NetworkMonitor:
    """网络监控类，用于监控和分析网络请求性能"""

    def __init__(
        self,
        ctx: SessionContext,
        feature_category: str,
        step_logger: logging.Logger,
    ):
        self._ctx = ctx
        self._feature_category = feature_category
        self._target_apis = []
        self._is_monitoring = False
        self._requests: Dict[str, NetworkRequest] = {}
        self._step_logger = step_logger
        self._timing_writer = ctx.network_timing_writer(feature_category)
        self._cdp_writer: Optional[JsonlWriter] = None
        self._current_case: Optional[str] = None

    def start_monitoring(self, client: Optional[CDPSession], test_case_name: str, target_apis: list[Any]):
        """开始监控网络请求"""
        if not client:
            error = RuntimeError('Browser not started. Call startBrowser() first.')
            self._step_logger.error(str(error))
            raise error

        try:
            self._target_apis = target_apis
            self._is_monitoring = True
            self._current_case = test_case_name
            client.send('Performance.enable')
            client.send('Network.enable')
            self._step_logger.info(f'Network monitoring started! Monitor test case: {test_case_name}')

            if self._ctx.enable_cdp_log:
                self._cdp_writer = self._ctx.network_cdp_writer(self._feature_category, test_case_name)
                self._step_logger.info(f'CDP logging for test case: {test_case_name} enabled')

            client.on('Network.requestWillBeSent', self._monitor_request_will_be_sent)
            client.on('Network.requestWillBeSentExtraInfo', self._monitor_request_extra_info)
            client.on('Network.requestServedFromCache', self._monitor_request_from_cache)
            client.on('Network.resourceChangedPriority', self._monitor_priority_change)
            client.on('Network.responseReceived', self._monitor_response_received)
            client.on('Network.dataReceived', self._monitor_data_received)
            client.on('Network.loadingFinished', self._monitor_loading_finished)

        except Exception as error:
            self._step_logger.error(f'Monitoring process failed: {str(error)}')
            raise

    def stop_monitoring(self, client: Optional[CDPSession]):
        """停止网络监控"""
        if client and self._is_monitoring:
            try:
                self._target_apis = []
                client.send('Performance.disable')
                client.send('Network.disable')
                self._is_monitoring = False
                self._cdp_writer = None
                self._current_case = None
                self._step_logger.info('Network monitoring stopped')
            except Exception as error:
                self._step_logger.error(f'Failed to stop monitoring: {str(error)}')
                raise

    def _write_cdp_event(self, event: str, data: Dict[str, Any]) -> None:
        if self._cdp_writer is None:
            return
        timestamp = data.get('timestamp')
        if timestamp is not None:
            ts = float(timestamp)
        else:
            ts = time.time()
        self._cdp_writer.write({
            'event': event,
            'timestamp': ts,
            'data': data,
        })

    def _monitor_request_will_be_sent(self, data: Dict[str, Any]):
        """处理请求发送事件"""
        resource_type = data.get('type', 'Other')
        if resource_type not in ('Fetch', 'XHR'):
            return

        self._write_cdp_event('Network.requestWillBeSent', data)

        request = data['request']
        matched_api = None
        for api in self._target_apis:
            if request['method'].lower() == api['method'] and \
                match_url_to_entrypoint(request['url'], api['path']):
                matched_api = api
        if not matched_api:
            return

        request_id = data['requestId']
        document_url = data['documentURL']
        timestamp = data['timestamp'] * 1000
        wall_time = data['wallTime']

        network_request = NetworkRequest(request_id, request['url'], document_url)
        network_request.set_issue_time(timestamp, wall_time)
        network_request.resource_type = resource_type
        network_request.request_time_ranges['requestWillBeSent'] = timestamp
        self._update_network_request_with_request(network_request, request)
        self._requests[request_id] = network_request

    def _monitor_request_extra_info(self, data: Dict[str, Any]):
        """处理请求额外信息事件"""
        request_id = data['requestId']
        request = self._requests.get(request_id)
        if request:
            self._write_cdp_event('Network.requestWillBeSentExtraInfo', data)
            extra_request_info = {
                'connectTiming': data.get('connectTiming', {})
            }
            request.add_extra_request_info(extra_request_info)

    def _monitor_request_from_cache(self, data: Dict[str, Any]):
        """处理缓存请求事件"""
        request_id = data['requestId']
        request = self._requests.get(request_id)
        if request:
            self._write_cdp_event('Network.requestServedFromCache', data)
            request.set_from_memory_cache()

    def _monitor_priority_change(self, data: Dict[str, Any]):
        """处理优先级变化事件"""
        request_id = data['requestId']
        request = self._requests.get(request_id)
        if request:
            self._write_cdp_event('Network.resourceChangedPriority', data)
            request.priority = data['newPriority']

    def _monitor_response_received(self, data: Dict[str, Any]):
        """处理响应接收事件"""
        request_id = data['requestId']
        request = self._requests.get(request_id)
        if request:
            self._write_cdp_event('Network.responseReceived', data)

            timestamp = data['timestamp'] * 1000
            request.request_time_ranges['responseReceived'] = timestamp
            request.response_received_time = timestamp
            request.resource_type = data['type']

            response = data['response']
            self._update_network_with_response(request, response)

    def _monitor_data_received(self, data: Dict[str, Any]):
        """处理数据接收事件"""
        request_id = data['requestId']
        request = self._requests.get(request_id)
        if request:
            self._write_cdp_event('Network.dataReceived', data)
            timestamp = data['timestamp'] * 1000
            data_length = data['dataLength']
            encoded_data_length = data['encodedDataLength']
            request.add_data_received(timestamp, data_length, encoded_data_length)

    def _monitor_loading_finished(self, data: Dict[str, Any]):
        """处理加载完成事件"""
        request_id = data['requestId']
        request = self._requests.get(request_id)
        if request:
            self._write_cdp_event('Network.loadingFinished', data)

            timestamp = data['timestamp'] * 1000
            request.request_time_ranges['loadingFinished'] = timestamp
            request.end_time = timestamp
            self._analyze_request_timing(request_id)
            self._requests.pop(request_id, None)

    def _analyze_request_timing(self, request_id: str):
        """分析请求时间信息"""
        request = self._requests.get(request_id)
        if not request:
            return

        result_timing = {
            'queuing': 0.0,
            'stalled': 0.0,
            'dnsLookup': 0.0,
            'initialConnection': 0.0,
            'ssl': 0.0,
            'requestSent': 0.0,
            'waitingForServer': 0.0,
            'contentDownload': 0.0,
            'total': 0.0
        }

        timing_info = request.timing
        if not timing_info:
            start = request.issue_time if request.issue_time != -1 else (
                request.start_time if request.start_time != -1 else 0
            )
            middle = request.response_received_time if request.response_received_time != -1 else (
                request.start_time if request.start_time != -1 else float('inf')
            )
            end = float('inf') if request.end_time == -1 else request.end_time

            result_timing['total'] = end - start
            result_timing['stalled'] = middle - start
            result_timing['contentDownload'] = end - middle

            self._log_timing_info(request, result_timing)
            return

        issue_time = request.issue_time
        start_time = (timing_info.get('requestTime') or 0) * 1000
        end_time = first_positive(request.end_time, request.response_received_time)

        result_timing['total'] = end_time - min(issue_time, start_time)
        if issue_time < start_time:
            result_timing['queuing'] = start_time - issue_time

        response_received = request.response_received_time - start_time
        blocking_end = first_positive(
            timing_info.get('dnsStart') or 0,
            timing_info.get('connectStart') or 0,
            timing_info.get('sendStart') or 0,
            response_received
        )

        result_timing['stalled'] = blocking_end or 0.0
        result_timing['dnsLookup'] = (timing_info.get('dnsEnd') or 0) - (timing_info.get('dnsStart') or 0)
        result_timing['initialConnection'] = \
            (timing_info.get('connectEnd') or 0) - (timing_info.get('connectStart') or 0)
        result_timing['ssl'] = (timing_info.get('sslEnd') or 0) - (timing_info.get('sslStart') or 0)
        result_timing['requestSent'] = (timing_info.get('sendEnd') or 0) - (timing_info.get('sendStart') or 0)
        result_timing['waitingForServer'] = response_received - max(
            timing_info.get('sendEnd') or 0,
            timing_info.get('connectEnd') or 0,
            timing_info.get('dnsEnd') or 0,
            blocking_end
        )

        if request.end_time != -1:
            result_timing['contentDownload'] = end_time - request.response_received_time

        self._log_timing_info(request, result_timing)

    def _log_timing_info(self, request: NetworkRequest, result_timing: Dict[str, float]):
        """记录 API 时延采样到 api_timing.jsonl"""
        timestamp_sec = request.end_time / 1000.0 if request.end_time != -1 else time.time()
        record = {
            'run_id': self._ctx.paths.run_id,
            'feature': self._feature_category,
            'case': self._current_case or '',
            'timestamp': timestamp_sec,
            'request_id': request.request_id,
            'method': request.request_method,
            'url': request.url,
            'status': request.status_code,
            'timing': {
                'total': result_timing['total'] / 1000.0,
                'waitingForServer': result_timing['waitingForServer'] / 1000.0,
            },
        }
        self._timing_writer.write(record)

    @staticmethod
    def _update_network_with_response(network_request: NetworkRequest, response: Dict[str, Any]):
        """使用响应信息更新网络请求对象"""
        if response.get('url') and response['url'] != network_request.url:
            network_request.url = response['url']
        network_request.mime_type = response.get('mimeType', '')
        network_request.charset = response.get('charset', '')
        network_request.status_code = response.get('status', 0)
        network_request.status_text = response.get('statusText', '')
        network_request.transfer_size = response.get('encodedDataLength', 0)
        network_request.connection_id = response.get('connectionId', 0)
        network_request.connection_reused = response.get('connectionReused', False)
        network_request.set_remote_address(
            response.get('remoteIPAddress', ''),
            response.get('remotePort', -1)
        )
        network_request.protocol = response.get('protocol', '')
        network_request.timing = response.get('timing', {})

    @staticmethod
    def _update_network_request_with_request(network_request: NetworkRequest, request: Dict[str, Any]):
        """使用请求信息更新网络请求对象"""
        network_request.request_method = request.get('method', '')
