class NetworkRequest:
    """网络请求信息类"""
    def __init__(self, request_id, url, document_url):
        self._request_id = request_id
        self.url = url
        self._document_url = document_url
        self.request_method = ''
        self._remote_address = ''
        self._start_time = -1
        self._issue_time = -1
        self._pseudo_wall_time = -1
        self.response_received_time = -1
        self._end_time = -1
        self.resource_size = 0
        self.transfer_size = 0
        self._timing = {}
        self.mime_type = ''
        self.resource_type = 'Other'
        self.domain = ''
        self.request_time_ranges = {
            'requestWillBeSent': 0,
            'responseReceived': 0,
            'dataReceived': [],
            'loadingFinished': 0
        }
        self.status_code = 0
        self.status_text = ''
        self.charset = ''
        self.connection_id = '0'
        self.connection_reused = False
        self.protocol = ''
        self._from_memory_cache = False
        self._current_priority = None

    @property
    def request_id(self):
        """获取请求ID，只读属性"""
        return self._request_id

    @property
    def document_url(self):
        """获取文档URL，只读属性"""
        return self._document_url

    @property
    def pseudo_wall_time(self):
        """获取伪墙时间，只读属性"""
        return self._pseudo_wall_time

    @property
    def issue_time(self):
        """获取请求发出时间"""
        return self._issue_time

    def set_issue_time(self, timestamp, wall_time):
        self._issue_time = timestamp
        self._start_time = timestamp
        self._pseudo_wall_time = wall_time

    @property
    def start_time(self):
        """获取请求开始时间，只读属性"""
        return self._start_time

    @property
    def end_time(self):
        """获取请求结束时间"""
        return self._end_time

    @end_time.setter
    def end_time(self, timestamp):
        """设置end_time"""
        if self._timing.get('requestTime', 0) != 0:
            self._end_time = max(self.response_received_time, timestamp)
        else:
            self._end_time = timestamp
            if self.response_received_time > timestamp:
                self.response_received_time = timestamp

    @property
    def timing(self):
        """获取timing信息"""
        return self._timing

    @timing.setter
    def timing(self, timing_info):
        """设置timing信息
        
        Args:
            timing_info: timing信息对象
        """
        if not timing_info or self._from_memory_cache:  # 如果timing_info是空对象{}，直接返回
            return
        self._start_time = timing_info.get('requestTime', 0) * 1000
        # header_received_time 单位为ms
        header_received_time = self._start_time + timing_info.get('receiveHeadersEnd', 0)
        if (self.response_received_time == -1 or self.response_received_time > header_received_time): 
            self.response_received_time = header_received_time
        if (self._start_time > self.response_received_time):
            self.response_received_time = self._start_time
        self._timing = timing_info

    @property
    def remote_address(self):
        """获取远程地址"""
        return self._remote_address

    def set_remote_address(self, ip, port):
        """设置远程地址
        """
        self._remote_address = f"{ip}:{port}"

    def add_extra_request_info(self, extra_request_info):
        if not extra_request_info:
            return
        if 'connectTiming' in extra_request_info:
            connect_timing = extra_request_info['connectTiming']
            if (connect_timing.get('requestTime', 0) != 0):
                self._start_time = connect_timing.get('requestTime', 0) * 1000
    
    def add_data_received(self, timestamp, data_length, encoded_data_length):
        self.request_time_ranges['dataReceived'].append(timestamp)
        self.resource_size += data_length
        if (encoded_data_length != -1):
            self.transfer_size += encoded_data_length
        self.end_time = timestamp
    
    def get_latency(self): 
        if self.response_received_time == -1 or self._start_time == -1:
            return -1
        return self.response_received_time - self._start_time
    
    def get_duration(self): 
        if self._end_time == -1 or self._start_time == -1:
            return -1
        return self._end_time - self._start_time

    def set_from_memory_cache(self):
        """设置是否从内存缓存加载"""
        self._from_memory_cache = True
        self.timing = {}

    @property
    def priority(self):
        """获取请求优先级"""
        return self._current_priority

    @priority.setter
    def priority(self, priority):
        """设置请求优先级
        
        Args:
            value: 优先级值
        """
        self._current_priority = priority

    def to_dict(self):
        """将对象转换为字典格式"""
        return {
            'request_id': self._request_id,
            'url': self.url,
            'documentUrl': self._document_url,
            'method': self.request_method,
            'remoteAddress': self._remote_address,
            'startTime': self._start_time,
            'issueTime': self._issue_time,
            'pseudoWallTime': self._pseudo_wall_time,
            'responseReceivedTime': self.response_received_time,
            'endTime': self._end_time,
            'duration': self.get_duration(),
            'latency': self.get_latency(),
            'resourceSize': self.resource_size,
            'transferSize': self.transfer_size,
            'timing': self._timing,
            'mimeType': self.mime_type,
            'resourceType': self.resource_type,
            'domain': self.domain,
            'requestTimeRanges': self.request_time_ranges,
            'status': self.status_code,
            'statusText': self.status_text,
            'connection_id': self.connection_id,
            'connection_reused': self.connection_reused,
            'protocol': self.protocol,
            'charset': self.charset,
            'from_memory_cache': self._from_memory_cache,
            'priority': self._current_priority
        } 