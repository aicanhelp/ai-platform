/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */

import { getNodeList } from '@/api/numaApi';

export const initializeNodeData = (nodes, length, defaultValue) => {
  const nodeData = {};
  nodes.forEach(node => {
    const nodeName = node.metadata.name;
    nodeData[nodeName] = new Array(length).fill(defaultValue);
  });
  return nodeData;
};

export const initializeNumaNodeData = (numaNodeCpuMapping) => {
  const initializedNumaNodeData = {};
  Object.keys(numaNodeCpuMapping).forEach(numaNodeName => {
    initializedNumaNodeData[numaNodeName] = {
      cpus: numaNodeCpuMapping[numaNodeName].cpus,
      allocatableCpus: numaNodeCpuMapping[numaNodeName].allocatableCpus,
    };
  });
  return initializedNumaNodeData;
};


export const parseAllocatable = (allocatable) => {
  const cpuIds = new Set();

  if (allocatable && typeof allocatable === 'string') {
    allocatable.split(',').forEach(part => {
      if (part.includes('-')) {
        const [start, end] = part.split('-').map(Number);
        for (let i = start; i <= end; i++) {
          cpuIds.add(i.toString());
        }
      } else {
        cpuIds.add(part);
      }
    });
  }

  return cpuIds;
};

export function convertToRanges(nums) {
  if (nums.length === 0) {
    return '';
  }

  const numListInt = nums.map(Number).sort((a, b) => a - b);

  let result = [];
  let [start] = numListInt;
  let [end] = numListInt;

  for (let i = 1; i < numListInt.length; i++) {
    if (numListInt[i] === end + 1) {
      end = numListInt[i];
    } else {
      if (start === end) {
        result.push(`${start}`);
      } else {
        result.push(`${start}-${end}`);
      }
      start = numListInt[i];
      end = numListInt[i];
    }
  }

  if (start === end) {
    result.push(`${start}`);
  } else {
    result.push(`${start}-${end}`);
  }

  return result.join(', ');
}

export function formatBytes(bytes) {
  if (bytes === 0) {
    return '0 B';
  }

  if (isNaN(bytes) || bytes === 'N/A' || bytes == null) {
    return 'N/A';
  }

  const sizes = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  const formattedSize = (bytes / Math.pow(1024, i)).toFixed(2);
  return `${formattedSize} ${sizes[i]}`;
}

export function transformPercent(value) {
  if (value === 'N/A') {
    return '0%';
  }
  const numericValue = parseFloat(value);
  return isNaN(numericValue) ? '0%' : `${(numericValue * 100).toFixed(2)}%`;
}

export const getNodeListInfo = async () => {
  try {
    const res = await getNodeList(name);
    if (res?.data?.data?.items) {
      return res.data.data.items;
    } else {
      return null;
    }
  } catch (error) {
    throw new Error(`Failed to fetch node list: ${error.message}`);
  }
};