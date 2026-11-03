# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""
This module defines the CollectorRegistry base class.
"""


class CollectorRegistry:
    """
    The base class of collector registry.
    """

    def __init__(self):
        self._registry = {}

    def register(self, name: str, collector, group: str):
        """
        Registers a collector under a specific group and name.
        """
        if group not in self._registry:
            self._registry[group] = {}
        self._registry[group][name] = collector

    def get(self, name: str, group: str):
        """
        Retrieves a collector by its name and group.
        """
        return self._registry.get(group, {}).get(name)

    def get_all(self):
        """
        Returns all the registered collectors, grouped by their respective groups.
        """
        return self._registry


registry = CollectorRegistry()
