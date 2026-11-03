/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */


import { getExpressQuery, getPodByUUID } from '@/api/numaApi';

export const getPrometheusQuery = async (query) => {
  try {
    const res = await getExpressQuery(query);
    return res?.data?.data ?? null;
  } catch (error) {
    return null;
  }
};

export const mergePrometheusData = (dataGroups, initializedNodeData) => {
  dataGroups.forEach((group, groupIndex) => {
    if (!group || !group.result) {
      return;
    }

    group.result.forEach(entry => {
      const newNodeName = entry.metric.node;
      const value = entry.value?.[1];

      if (!newNodeName || value === undefined) {
        return;
      }

      if (initializedNodeData[newNodeName]) {
        initializedNodeData[newNodeName][groupIndex] = value;
      }
    });
  });
  return initializedNodeData;
};

export const getDatArr = async (queryCheckArr) => {
  return await Promise.all(queryCheckArr.map(query => getPrometheusQuery(query)));
};

export const processContainerAssignments = async (containerAssignments, processAssignmentData) => {
  const dataSourceArray = await Promise.all(
    containerAssignments.map(processAssignmentData)
  ).then(results => results.filter(item => item.podName !== 'N/A'));

  const containerNum = dataSourceArray.length;
  const cpuOccupied = Array.from(new
  Set(dataSourceArray.flatMap(item => item.cpuList)));
  const cpuOccupiedNum = cpuOccupied.length;
  const crossNumaContainerNum = dataSourceArray.filter(item => item.crossNuma === '是').length;

  return {
    containerNum: containerNum || 0,
    crossNumaContainerNum: crossNumaContainerNum || 0,
    cpuOccupied,
    cpuOccupiedNum: cpuOccupiedNum || 0,
    dataSourceArray,
  };
};

export const processAssignmentData = async (assignment) => {
  try {
    const { metric } = assignment;
    const podUUID = metric.pod_uuid || 'N/A';
    const podName = await getPodByUUID(podUUID);

    return {
      container: metric.container || 'N/A',
      podName: podName?.data?.data || 'N/A',
      cpuList: metric.cpu_list ? metric.cpu_list.split(',').map(cpu => cpu.trim()) : [],
      crossNuma: metric.cpu_numa_node.includes(',') ? '是' : '否',
      cpuNumaNode: metric.cpu_numa_node ? metric.cpu_numa_node.split(',').map(node => node.trim()) : [],
    };
  } catch (error) {
    return {
      container: 'N/A',
      podName: 'N/A',
      cpuList: [],
      crossNuma: '否',
      cpuNumaNode: [],
    };
  }
};