/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { containerRouterPrefix } from '@/common/constants';
import { Address4 } from 'ip-address';
/**
 * 处理params
 */
import Dayjs from 'dayjs';
import { filterStatusName } from '../utils/common';
import dayjs from 'dayjs';

export function solveParams(params) {
  for (let key in params) {
    if (params[key] === '') {
      params[key] = null;
    }
  }
  return params;
}

/**
 * antd表格按名字首字母顺序排序
 */
export function sorterFirstAlphabe(a = '--', b = '--') {
  if (typeof a === 'number' && typeof b === 'number') {
    return a - b;
  }
  return a.toString().localeCompare(b.toString());
}

/**
 * antd表格按时间顺序排序
 */
export function sortJobByTime(a, b) {
  let firstTime = a.creationTimestamp ? a.creationTimestamp : '';
  let endTime = b.creationTimestamp ? b.creationTimestamp : '';
  return Dayjs(firstTime) - Dayjs(endTime);
}

/**
 * antd表格筛选项去重过滤
 */
export function filterRepeat(arr) {
  const newArr = arr.filter((item, index, self) => {
    return self.findIndex((obj) => obj.value === item.value) === index;
  });
  return newArr;
}

/**
 * 工作负载表格按名称升序
 */
export function sortWorkLoadByNameAscend(type, arr) {
  arr.sort((a, b) => {
    let nameA = a.name;
    let nameB = b.name;
    if (nameA < nameB) {
      return -1;
    }
    if (nameA > nameB) {
      return 1;
    }
    return 0;
  });
}

/**
 * 工作负载表格按名称降序
 */
export function sortWorkLoadByNameDescend(type, arr) {
  arr.sort((a, b) => {
    let nameA = a.name;
    let nameB = b.name;
    if (nameA < nameB) {
      return 1;
    }
    if (nameA > nameB) {
      return -1;
    }
    return 0;
  });
}

/**
 * 工作负载表格按时间降序
 */
export function sortWorkLoadByTimeDescend(type, arr) {
  arr.sort((a, b) => {
    let timeA = a.creationTimestamp;
    let timeB = b.creationTimestamp;
    if (timeA < timeB) {
      return 1;
    }
    if (timeA > timeB) {
      return -1;
    }
    return 0;
  });
}

/**
 * 工作负载表格按时间升序
 */
export function sortWorkLoadByTimeAscend(type, arr) {
  arr.sort((a, b) => {
    let nameA = a.creationTimestamp;
    let nameB = b.creationTimestamp;
    if (nameA < nameB) {
      return -1;
    }
    if (nameA > nameB) {
      return 1;
    }
    return 0;
  });
}

/**
 * 工作负载表格按命名空间升序
 */
export function sortWorkLoadByNamespaceAscend(type, arr) {
  arr.sort((a, b) => {
    let namespaceA = a.namespace;
    let namespaceB = b.namespace;
    if (namespaceA < namespaceB) {
      return -1;
    }
    if (namespaceA > namespaceB) {
      return 1;
    }
    return 0;
  });
}

/**
 * 工作负载表格按命名空间降序
 */
export function sortWorkLoadByNamespaceDescend(type, arr) {
  arr.sort((a, b) => {
    let namespaceA = a.namespace;
    let namespaceB = b.namespace;
    if (namespaceA < namespaceB) {
      return 1;
    }
    if (namespaceA > namespaceB) {
      return -1;
    }
    return 0;
  });
}

/**
 * 工作负载表格按属性升序
 */
export function sortWorkLoadByStatusAscend(type, arr) {
  arr.sort((a, b) => {
    let typeA = filterStatusName(a.colocationType);
    let typeB = filterStatusName(b.colocationType);
    if (typeA < typeB) {
      return 1;
    }
    if (typeA > typeB) {
      return -1;
    }
    return 0;
  });
}

/**
 * 工作负载表格按属性降序
 */
export function sortWorkLoadByStatusDescend(type, arr) {
  arr.sort((a, b) => {
    let typeA = filterStatusName(a.colocationType);
    let typeB = filterStatusName(b.colocationType);
    if (typeA < typeB) {
      return -1;
    }
    if (typeA > typeB) {
      return 1;
    }
    return 0;
  });
}

/**
 * 工作负载表格按所属混部节点升序
 */
export function sortWorkLoadByNodeAscend(type, arr) {
  arr.sort((a, b) => {
    let nodeA = a.hostIP;
    let nodeB = b.hostIP;
    if (nodeA < nodeB) {
      return -1;
    }
    if (nodeA > nodeB) {
      return 1;
    }
    return 0;
  });
}

/**
 * 工作负载表格按所属混部节点降序
 */
export function sortWorkLoadByNodeDescend(type, arr) {
  arr.sort((a, b) => {
    let nodeA = a.hostIP;
    let nodeB = b.hostIP;
    if (nodeA < nodeB) {
      return 1;
    }
    if (nodeA > nodeB) {
      return -1;
    }
    return 0;
  });
}

/**
 * 工作负载表格按ip升序
 */
export function sortWorkLoadByIpAscend(type, arr) {
  arr.sort((a, b) => {
    const [podIpA, ...resets] = a.podIPs;
    const [podIpB, ...resets1] = b.podIPs;
    if (podIpA < podIpB) {
      return -1;
    }
    if (podIpA > podIpB) {
      return 1;
    }
    return 0;
  });
}

/**
 * 工作负载表格按ip降序
 */
export function sortWorkLoadByIpDescend(type, arr) {
  arr.sort((a, b) => {
    const [podIpA, ...resets] = a.podIPs;
    const [podIpB, ...resets1] = b.podIPs;
    if (podIpA < podIpB) {
      return 1;
    }
    if (podIpA > podIpB) {
      return -1;
    }
    return 0;
  });
}

/**
 * 判断 daemonsets
 */
export function getDaemonSetStatus(statusObj) {
  let status = 'Failed';
  if (!statusObj) {
    status === 'Failed';
    return status;
  }
  if (statusObj.desiredNumberScheduled) {
    if (statusObj.numberAvailable === statusObj.desiredNumberScheduled) {
      status = 'Active';
    } else {
      if (statusObj.currentNumberScheduled !== statusObj.desiredNumberScheduled) {
        status = 'Updating';
      }
    }
  } else {
    status = 'Failed';
  }
  return status;
}

/**
 * 增加判断状态工作负载
 * // status
 */
export function getWorkloadStatusJudge(statusObj) {
  let status = 'Failed';
  if (!statusObj) {
    status === 'Failed';
    return status;
  }
  if (statusObj.replicas) {
    if (statusObj.availableReplicas === statusObj.replicas) {
      status = 'Active';
    } else {
      if (statusObj.updatedReplicas !== statusObj.replicas) {
        status = 'Updating';
      }
    }
  } else {
    status = 'Failed';
  }
  return status;
}

export function sorterFirstAlphabet(a = '--', b = '--') {
  if (typeof a === 'number' && typeof b === 'number') {
    return a - b;
  }
  return a.toString().localeCompare(b.toString());
}

/** 首字母大写 */
export function firstAlphabetUp(word) {
  return word.charAt(0).toUpperCase() + word.slice(1);
}

export function skipSendMainMsg(event, key, url, history, isAddLocalPrefix = true) {
  event ? event.preventDefault() : '';
  window.parent.postMessage({
    type: 'subToMainMsg',
    data: key,
    url: isAddLocalPrefix ? `/${containerRouterPrefix}${url}` : url,
    isInsertMainPrefix: true,
  });
  window.__OPENFUYAO__ ? '' : history.push(url);
}

/**
 * 通过数量生成
 */
export function solveDataAreaNumPalette(num) {
  let colorList = ['#43cbbb', '#F9a975', '#4B8BEA', '#B4A2FF', '#Eb96F5'];
  const colLength = colorList.length;
  try {
    for (let range = 0; range < num - colLength; range++) {
      let color = '';
      color = `rgb( ${[
        Math.round((window.crypto.getRandomValues(new Uint8Array(1))[0] / 256) * 255), // 替换math.random()
        Math.round((window.crypto.getRandomValues(new Uint8Array(1))[0] / 256) * 255),
        Math.round((window.crypto.getRandomValues(new Uint8Array(1))[0] / 256) * 255),
      ].join(',')})`;
      while (colorList.some((i) => i === color)) {
        color = `rgb( ${[
          Math.round((window.crypto.getRandomValues(new Uint8Array(1))[0] / 256) * 255), // 替换math.random()
          Math.round((window.crypto.getRandomValues(new Uint8Array(1))[0] / 256) * 255),
          Math.round((window.crypto.getRandomValues(new Uint8Array(1))[0] / 256) * 255),
        ].join(',')})`;
      }
      colorList.push(color);
    }
  } catch (e) {
    //
  }
  return colorList;
}

/**
 * 优化颜色
 */
export function solveDataAreaNumPalettePlus(num) {
  // 预设颜色列表
  const presetColors = ['#43CBBB', '#F9A975', '#4B8BEA', '#B4A2FF', '#Eb96F5'];

  // 如果请求数量小于等于预设颜色数量，直接返回切片
  if (num <= presetColors.length) {
    return presetColors.slice(0, num);
  }

  // 使用Set提高查找效率
  const colorSet = new Set(presetColors);
  const result = [...presetColors];

  // 生成随机RGB颜色的辅助函数
  const generateRandomRgb = () => {
    const randomValue = () => Math.floor(Math.random() * 256); // 使用Math.random性能更好
    return `rgb(${randomValue()}, ${randomValue()}, ${randomValue()})`;
  };

  // 预先生成足够数量的颜色
  const neededColors = num - presetColors.length;
  const batchSize = Math.min(1000, neededColors); // 分批处理避免内存问题
  let generated = 0;

  while (generated < neededColors) {
    const batch = [];
    const currentBatchSize = Math.min(batchSize, neededColors - generated);

    // 批量生成颜色
    for (let i = 0; i < currentBatchSize * 1.2; i++) {
      // 多生成一些以防重复
      batch.push(generateRandomRgb());
    }

    // 过滤掉重复颜色
    const uniqueBatch = batch.filter((color) => !colorSet.has(color));

    // 添加到结果中
    const addCount = Math.min(uniqueBatch.length, neededColors - generated);
    for (let i = 0; i < addCount; i++) {
      const color = uniqueBatch[i];
      colorSet.add(color);
      result.push(color);
      generated++;
    }
  }

  return result;
}

export function solveColorMap(mockData, colors) {
  const groupedData = {};
  mockData.forEach((item) => {
    if (!groupedData[item.name]) {
      groupedData[item.name] = [];
    }
    groupedData[item.name].push(item);
  });
  const series = Object.keys(groupedData).map((type, index) => ({
    name: type,
    color: colors[index % colors.length], // 动态分配颜色
  }));
  return series;
}

const completeTimeSeries = (dataArray) => {
  if (dataArray.length === 0) return [];
  // 解析时间字符串为分钟数（例如 "09:30" → 570）
  const timeToMinutes = (timeStr) => {
    const [hours, minutes] = timeStr.split(':').map(Number);
    return hours * 60 + minutes;
  };

  // 将分钟数转换回时间字符串（例如 570 → "09:30"）
  const minutesToTime = (totalMinutes) => {
    const hours = Math.floor(totalMinutes / 60);
    const mins = totalMinutes % 60;
    return `${String(hours).padStart(2, '0')}:${String(mins).padStart(2, '0')}`;
  };
  // 创建时间点映射表便于快速查找
  // 处理原始数据
  const processedData = [];
  const timeMap = new Map();
  dataArray.forEach(([time, value]) => {
    const minutes = timeToMinutes(time);
    processedData.push(minutes);
    timeMap.set(minutes, value);
  });
  if (processedData.length === 0) return [];
  processedData.sort((a, b) => a - b);
  // 确定时间范围
  const min = processedData[0];
  const max = processedData[processedData.length - 1];
  const interval = 1; // 分钟间隔
  const allMinutes = new Set(processedData);
  for (let m = min; m <= max; m += interval) {
    allMinutes.add(m);
  }
  const sortedMinutes = Array.from(allMinutes).sort((a, b) => a - b);
  return sortedMinutes.map((m) => [minutesToTime(m), timeMap.has(m) ? timeMap.get(m) : null]);
};

export function solveSeriesMap(mockData) {
  const seriesMap = {};
  mockData.forEach((item) => {
    if (!seriesMap[item.name]) {
      seriesMap[item.name] = [];
    }
    seriesMap[item.name].push([dayjs(item.timestamp * 1000).format('HH:mm'), item.value]);
  });
  // 转换为 ECharts 需要的 series 格式
  const series = Object.keys(seriesMap).map((name) => ({
    name,
    type: 'line',
    smooth: true,
    symbol: 'none',
    data: completeTimeSeries(seriesMap[name]),
    lineStyle: {
      width: 3,
    },
    connectNulls: false,
  }));
  return series;
}

const cpuMapping = {
  clusterCpuTotal: '物理资源总量',
  clusterMemoryTotal: '物理资源总量',
  clusterNoncoloCpuUsage: '非混部Pod使用量',
  clusterNoncoloMemoryUsage: '非混部Pod使用量',
  clusterHlsCpuUsage: 'HLS级别Pod使用量',
  clusterHlsMemoryUsage: 'HLS级别Pod使用量',
  clusterLsCpuUsage: 'LS级别Pod使用量',
  clusterLsMemoryUsage: 'LS级别Pod使用量',
  clusterBeCpuUsage: 'BE级别Pod使用量',
  clusterBeMemoryUsage: 'BE级别Pod使用量',
  clusterBeCpuAllocatable: '超卖资源总量',
  clusterBeMemoryAllocatable: '超卖资源总量',
  clusterBeCpuRequest: '超卖资源已申请量',
  clusterBeMemoryRequest: '超卖资源已申请量',
  clusterHlsCpuRequest: 'HLS级别Pod资源申请量',
  clusterHlsMemoryRequest: 'HLS级别Pod资源申请量',
  clusterLsCpuRequest: 'LS级别Pod资源申请量',
  clusterLsMemoryRequest: 'LS级别Pod资源申请量',
  clusterNoncoloCpuRequest: '非混部Pod资源申请量',
  clusterNoncoloMemoryRequest: '非混部Pod资源申请量',
};

const cpuMappingEn = {
  clusterCpuTotal: 'Total Physical Resources',
  clusterMemoryTotal: 'Total Physical Resources',
  clusterNoncoloCpuUsage: 'Non-Colocation Pod Usage',
  clusterNoncoloMemoryUsage: 'Non-Colocation Pod Usage',
  clusterHlsCpuUsage: 'HLS-Level Pod Usage',
  clusterHlsMemoryUsage: 'HLS-Level Pod Usage',
  clusterLsCpuUsage: 'LS-Level Pod Usage',
  clusterLsMemoryUsage: 'LS-Level Pod Usage',
  clusterBeCpuUsage: 'BE-Level Pod Usage',
  clusterBeMemoryUsage: 'BE-Level Pod Usage',
  clusterBeCpuAllocatable: 'Total Oversold Resources',
  clusterBeMemoryAllocatable: 'Total Oversold Resources',
  clusterBeCpuRequest: 'Oversold Resources Requested',
  clusterBeMemoryRequest: 'Oversold Resources Requested',
  clusterHlsCpuRequest: 'HLS-Level Pod Resource Request',
  clusterHlsMemoryRequest: 'HLS-Level Pod Resource Request',
  clusterLsCpuRequest: 'LS-Level Pod Resource Request',
  clusterLsMemoryRequest: 'LS-Level Pod Resource Request',
  clusterNoncoloCpuRequest: 'Non-Colocation Pod Resource Request',
  clusterNoncoloMemoryRequest: 'Non-Colocation Pod Resource Request',
};

const nodeMapping = {
  nodeCpuTotal: '物理资源总量',
  nodeMemoryTotal: '物理资源总量',
  nodeNoncoloCpuUsage: '非混部Pod使用量',
  nodeNoncoloMemoryUsage: '非混部Pod使用量',
  nodeHlsCpuUsage: 'HLS级别Pod使用量',
  nodeHlsMemoryUsage: 'HLS级别Pod使用量',
  nodeLsCpuUsage: 'LS级别Pod使用量',
  nodeLsMemoryUsage: 'LS级别Pod使用量',
  nodeBeCpuUsage: 'BE级别Pod使用量',
  nodeBeMemoryUsage: 'BE级别Pod使用量',
  nodeBeCpuAllocatable: '超卖资源总量',
  nodeBeMemoryAllocatable: '超卖资源总量',
  nodeBeCpuRequest: '超卖资源已申请量',
  nodeBeMemoryRequest: '超卖资源已申请量',
  nodeHlsCpuRequest: 'HLS级别Pod资源申请量',
  nodeHlsMemoryRequest: 'HLS级别Pod资源申请量',
  nodeLsCpuRequest: 'LS级别Pod资源申请量',
  nodeLsMemoryRequest: 'LS级别Pod资源申请量',
  nodeNoncoloCpuRequest: '非混部Pod资源申请量',
  nodeNoncoloMemoryRequest: '非混部Pod资源申请量',
};

const nodeMappingEn = {
  nodeCpuTotal: 'Total Physical Resources',
  nodeMemoryTotal: 'Total Physical Resources',
  nodeNoncoloCpuUsage: 'Non-Colocation Pod Usage',
  nodeNoncoloMemoryUsage: 'Non-Colocation Pod Usage',
  nodeHlsCpuUsage: 'HLS-Level Pod Usage',
  nodeHlsMemoryUsage: 'HLS-Level Pod Usage',
  nodeLsCpuUsage: 'LS-Level Pod Usage',
  nodeLsMemoryUsage: 'LS-Level Pod Usage',
  nodeBeCpuUsage: 'BE-Level Pod Usage',
  nodeBeMemoryUsage: 'BE-Level Pod Usage',
  nodeBeCpuAllocatable: 'Total Oversold Resources',
  nodeBeMemoryAllocatable: 'Total Oversold Resources',
  nodeBeCpuRequest: 'Oversold Resources Requested',
  nodeBeMemoryRequest: 'Oversold Resources Requested',
  nodeHlsCpuRequest: 'HLS-Level Pod Resource Request',
  nodeHlsMemoryRequest: 'HLS-Level Pod Resource Request',
  nodeLsCpuRequest: 'LS-Level Pod Resource Request',
  nodeLsMemoryRequest: 'LS-Level Pod Resource Request',
  nodeNoncoloCpuRequest: 'Non-Colocation Pod Resource Request',
  nodeNoncoloMemoryRequest: 'Non-Colocation Pod Resource Request',
};

const isEnLocale = () => (window.__OPENFUYAO_LOCALE__ || 'zh-CN') === 'en-US';

const getMetricNameMap = () => (isEnLocale() ? cpuMappingEn : cpuMapping);
const getNodeMetricNameMap = () => (isEnLocale() ? nodeMappingEn : nodeMapping);

const getPhysicalResourceTotalName = () => (isEnLocale() ? 'Total Physical Resources' : '物理资源总量');
const getGeneralResourceTotalName = () => (isEnLocale() ? 'Total General Resources' : '通用资源总量');

const nodeZhMap = {
  nodeCpuTotal: 'cpuNodeUse',
  nodeMemoryTotal: 'memoryNodeUse',
  nodeNoncoloCpuUsage: 'cpuNodeUse',
  nodeNoncoloMemoryUsage: 'memoryNodeUse',
  nodeHlsCpuUsage: 'cpuNodeUse',
  nodeHlsMemoryUsage: 'memoryNodeUse',
  nodeLsCpuUsage: 'cpuNodeUse',
  nodeLsMemoryUsage: 'memoryNodeUse',
  nodeBeCpuUsage: 'cpuNodeUse',
  nodeBeMemoryUsage: 'memoryNodeUse',
  nodeBeCpuAllocatable: 'cpuNodeApply',
  nodeBeMemoryAllocatable: 'memoryNodeApply',
  nodeBeCpuRequest: 'cpuNodeApply',
  nodeBeMemoryRequest: 'memoryNodeApply',
  nodeHlsCpuRequest: 'normalNodeCpu',
  nodeHlsMemoryRequest: 'normalNodeMemory',
  nodeLsCpuRequest: 'normalNodeCpu',
  nodeLsMemoryRequest: 'normalNodeMemory',
  nodeNoncoloCpuRequest: 'normalNodeCpu',
  nodeNoncoloMemoryRequest: 'normalNodeMemory',
};

const newZhMap = {
  clusterCpuTotal: 'cpuUse',
  clusterMemoryTotal: 'memoryUse',
  clusterNoncoloCpuUsage: 'cpuUse',
  clusterNoncoloMemoryUsage: 'memoryUse',
  clusterHlsCpuUsage: 'cpuUse',
  clusterHlsMemoryUsage: 'memoryUse',
  clusterLsCpuUsage: 'cpuUse',
  clusterLsMemoryUsage: 'memoryUse',
  clusterBeCpuUsage: 'cpuUse',
  clusterBeMemoryUsage: 'memoryUse',
  clusterBeCpuAllocatable: 'cpuApply',
  clusterBeMemoryAllocatable: 'memoryApply',
  clusterBeCpuRequest: 'cpuApply',
  clusterBeMemoryRequest: 'memoryApply',
  clusterHlsCpuRequest: 'normalCpu',
  clusterHlsMemoryRequest: 'normalMemory',
  clusterLsCpuRequest: 'normalCpu',
  clusterLsMemoryRequest: 'normalMemory',
  clusterNoncoloCpuRequest: 'normalCpu',
  clusterNoncoloMemoryRequest: 'normalMemory',
};

const backMetricName = (name) => {
  let finallyName = getMetricNameMap()[convertObjectKeys(name)];
  return finallyName;
};

const backNodeMetricName = (name) => {
  let finallyName = getNodeMetricNameMap()[convertObjectKeys(name)];
  return finallyName;
};

const formatNumber = (num) => {
  return num % 1 === 0 ? num : parseFloat(num.toFixed(2));
};

const convertObjectKeys = (name) => name.replace(/_(?<line>[a-z])/g, (_, letter) => letter.toUpperCase());

export function solveEchartsClean(data) {
  let clusterResult = {
    cpuUse: [],
    memoryUse: [],
    cpuApply: [],
    memoryApply: [],
    normalCpu: [],
    normalMemory: [],
  };
  data.map((item) => {
    const metricName = backMetricName(item.metricName);
    if (metricName) {
      item.data.result.map((resultItem) => {
        resultItem.values.map(([timestamp, value]) => {
          clusterResult[newZhMap[convertObjectKeys(item.metricName)]].push({
            name: metricName,
            timestamp,
            value: item.metricName.includes('memory') && value ? formatNumber(value / 1024 / 1024) : value,
          });
        });
      });
    }
  });
  clusterResult.normalCpu = [
    ...clusterResult.cpuUse
      .filter((item) => item.name === getPhysicalResourceTotalName())
      .map((item) => ({ ...item, name: getGeneralResourceTotalName() })),
    ...clusterResult.normalCpu,
  ];
  clusterResult.normalMemory = [
    ...clusterResult.memoryUse
      .filter((item) => item.name === getPhysicalResourceTotalName())
      .map((item) => ({ ...item, name: getGeneralResourceTotalName() })),
    ...clusterResult.normalMemory,
  ];
  return clusterResult;
}

export function solveEchartsNodeClean(data) {
  let nodeResult = {
    cpuNodeUse: [],
    memoryNodeUse: [],
    cpuNodeApply: [],
    memoryNodeApply: [],
    normalNodeCpu: [],
    normalNodeMemory: [],
  };
  data.map((item) => {
    const metricName = backNodeMetricName(item.metricName);
    if (metricName) {
      item.data.result.map((resultItem) => {
        resultItem.values.map(([timestamp, value]) => {
          nodeResult[nodeZhMap[convertObjectKeys(item.metricName)]].push({
            name: `${resultItem.metric.instance}-${metricName}`,
            timestamp,
            value: item.metricName.includes('memory') && value ? formatNumber(value / 1024 / 1024) : value,
          });
        });
      });
    }
  });
  nodeResult.normalNodeCpu = [
    ...nodeResult.cpuNodeUse
      .filter((item) => item.name.includes(getPhysicalResourceTotalName()))
      .map((item) => ({
        ...item,
        name: item.name.replace(getPhysicalResourceTotalName(), getGeneralResourceTotalName()),
      })),
    ...nodeResult.normalNodeCpu,
  ];
  nodeResult.normalNodeMemory = [
    ...nodeResult.memoryNodeUse
      .filter((item) => item.name.includes(getPhysicalResourceTotalName()))
      .map((item) => ({
        ...item,
        name: item.name.replace(getPhysicalResourceTotalName(), getGeneralResourceTotalName()),
      })),
    ...nodeResult.normalNodeMemory,
  ];
  return nodeResult;
}

export function compareIPs(ipA, ipB) {
  let addrA = '';
  let addrB = '';
  if (ipA && ipA.length) {
    addrA = new Address4(ipA[0]);
  }
  if (ipB && ipB.length) {
    addrB = new Address4(ipB[0]);
  }
  return addrA.bigInteger() - addrB.bigInteger();
}

export function forbiddenMsg(message, errorMsg) {
  const isEn = isEnLocale();
  if (errorMsg.response.data.message.includes('User') || errorMsg.response.data.message.includes('user')) {
    return message.error(
      isEn
        ? 'Operation failed, current user does not have permission. Please contact the administrator to add permissions!'
        : '操作失败，当前用户没有操作权限，请联系管理员添加权限!',
    );
  } else {
    return message.error(
      isEn
        ? 'Request rejected by the server. Please consider modifying the request!'
        : '请求被服务端拒绝，建议修改该请求!',
    );
  }
}
