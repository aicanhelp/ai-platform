# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""
This module defines the CacheManager class which provides a thread-safe
cache for storing and retrieving data in memory. The cache is designed to
be a singleton, ensuring that only one instance of CacheManager exists
across the application.
"""

import threading

from typing import Any, Dict


class CacheManager:
    """
    The CacheManager ensures that cache operations are atomic by using a lock.
    It supports adding, retrieving, and clearing cache entries. It maintains
    a dictionary where keys are associated with lists of values.
    """
    _instance = None
    cache: Dict[str, Any]
    lock: threading.Lock

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance.cache = {}
            cls._instance.lock = threading.Lock()
        return cls._instance

    def get(self, key: str):
        """
        Retrieves the value associated with the given key from the cache.
        """
        with self.lock:
            return self.cache.get(key)

    def set(self, key: str, value: Any):
        """
        Adds a value to the cache under the given key. If the key already exists,
        the value is appended to the list.
        """
        with self.lock:
            if key not in self.cache:
                self.cache[key] = []
            self.cache[key].append(value)

    def clear(self, key: str):
        """
        Removes the given key from the cache.
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]

    def clear_all(self):
        """
        Clears all entries from the cache.
        """
        with self.lock:
            self.cache.clear()
