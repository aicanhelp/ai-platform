/**
 *  Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  openFuyao is licensed under Mulan PSL v2.
 *  You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  You may obtain a copy of Mulan PSL v2 at:
  
 *       http://license.coscl.org.cn/MulanPSL2
  
 *   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *   EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *   MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *   See the Mulan PSL v2 for more details.
 */
import jsyaml from 'js-yaml';

/**
 * 处理params
 */
export function solveParams(params) {
  for (let key in params) {
    if (params[key] === '') {
      params[key] = null;
    }
  }
  return params;
}

export function jsonToYaml(text) {
  let yaml = '';
  try {
    let jsonObj = JSON.parse(text);
    yaml = jsyaml.dump(jsonObj);
  } catch (e) {
    const str = `json转yaml失败，详细信息为:${e}`;
  }
  return yaml;
}

export function yamlTojson(text) {
  return JSON.parse(JSON.stringify(jsyaml.load(text)));
}

/** 首字母大写 */
export function firstAlphabetUp(word) {
  return word.charAt(0).toUpperCase() + word.slice(1);
}

/** 处理标签注解数据 */
export function solveAnnotation(data = {}) {
  const keyArr = Object.keys(data); // 遍历数组
  let dataList = [];
  if (keyArr.length) {
    keyArr.map((item) => {
      dataList.push({ key: item, value: data[item] });
    });
  }
  return dataList;
}

/** 处理实例 */
export function judgeStatusNormalCount(conditions) {
  let normalCount = 0;
  conditions.map((item) => {
    if (item.status === 'True') {
      normalCount++;
    }
  });
  return normalCount;
}

export function gainStatus(data) {
  let status = '';
  data.map((item) => {
    if (item.manager.includes('controller-manager')) {
      if (item.operation === 'Update') {
        status = 'Updating';
      }
      if (item.operation === 'Apply') {
        status = 'Applying';
      }
      if (item.operation === 'Patch') {
        status = 'Patching';
      }
      if (item.operation === 'Create') {
        status = 'Creating';
      }
      if (item.operation === 'Delete') {
        status = 'Deleted';
      }
      if (item.operation === 'Remove') {
        status = 'Removed';
      }
    }
  });
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
};

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
};

/**
 * 判断job
 */
export function getJobStatus(statusObj) {
  let status = 'Failed';
  if (!statusObj) {
    status === 'Failed';
    return status;
  }
  if (statusObj.conditions && statusObj.conditions.length && statusObj.conditions[0].type) {
    status = statusObj.conditions[0].type;
  } else {
    if (statusObj.active === 1) {
      status = 'Active';
    }
  }
  return status;
};

/**
 * 判断cronjob
 */
export function getCronJobStatus(statusObj) {
  let status = 'Completed';
  if (!statusObj) {
    status === 'Completed';
    return status;
  }
  if (statusObj.active) {
    status = 'Active';
  }
  return status;
}

/**
 * 处理key
 */
export function solvePathKeyEscape(key) {
  return key.replace('/', '~1');
}

/**
 * 处理标签注解
 * @param oldData 旧数据
 * @param newData 新数据
 * @param type label || annotation
 * @param isLinning 是否内层 默认外层
 * @return data 返回处理对象数组
 */
export function solveAnnotationOrLabelDiff(
  oldData,
  newData,
  type = 'annotation',
  isLinning = false
) {
  let data = []; // 返回数组
  let commonPath = '';
  if (type === 'annotation') {
    commonPath = '/metadata/annotations/';
  } else {
    commonPath = isLinning
      ? '/spec/template/metadata/labels/'
      : '/metadata/labels/';
  }
  // 是否为全新添加
  if (!oldData.length) {
    let valueObj = {};
    newData.map((item) => {
      valueObj[item.key] = item.value;
    });
    data.push({
      op: 'add',
      path: `${commonPath.slice(0, -1)}`,
      value: valueObj,
    });
    return data;
  }
  // 遍历老数组和新数组判断key
  oldData.map((item) => {
    const result = newData.filter((newItem) => newItem.key === item.key); // 遍历新数组是否存在这项key
    if (result.length) {
      const [{ key, ...resets }, ...remainArr] = result;
      if (
        oldData[oldData.findIndex((oldItem) => oldItem.key === key)].value !==
        result[0].value
      ) {
        data.push({
          op: 'replace',
          path: `${commonPath}${solvePathKeyEscape(key)}`,
          value: result[0].value,
        });
      }
    } else {
      // 删除
      const { key, ...resets } = item;
      data.push({
        op: 'remove',
        path: `${commonPath}${solvePathKeyEscape(key)}`,
      });
    }
  });

  newData.map((item) => {
    const result = oldData.filter((oldItem) => oldItem.key === item.key);
    if (!result.length) {
      data.push({
        op: 'add',
        path: `${commonPath}${solvePathKeyEscape(item.key)}`,
        value: item.value,
      });
    }
  });
  return data;
}
/**
 * 处理节点类型
 */
export function solveNodeType(labels) {
  let type = 'workerNode';
  const keyArr = Object.keys(labels);
  const isManage = keyArr.filter(filterItem => filterItem.includes('control-plane'));
  const isWorker = keyArr.filter(filterItem => filterItem.includes('worker'));
  if (isManage.length) {
    type = 'manageNode'; // host
    return type;
  }
  if (isWorker.length) {
    type = 'workerNode'; // worker
    return type;
  }
  return type;
}
/**
 * 处理异常正常节点数以及管理和工作节点
 * @param data []
 * @param isOverView 是否来自概览页
 */
export function solveNodeBaseInfo(data = [], isOverView = false) {
  if (!isOverView) {
    // 判断是否为主从节点
    let obj = {
      unknownNode: 0,
      unnormalNode: 0,
      totalNode: 0,
      manageNode: 0,
      workerNode: 0,
    };
    data.map((item) => {
      const keyArr = Object.keys(item.metadata.labels);
      const isManage = keyArr.filter(filterItem => filterItem.includes('control-plane'));
      const isWorker = keyArr.filter(filterItem => filterItem.includes('worker'));
      if (isManage.length) {
        obj.manageNode++; // host
      } else {
        obj.workerNode++;
      }
      const unknown = item.status.conditions.some(unknownItem => unknownItem.status === 'Unknown');
      if (!unknown) {
        const even = item.status.conditions.some(
          (statusItem) =>
            (statusItem.type !== 'Ready' && statusItem.status === 'True') ||
            (statusItem.type === 'Ready' && statusItem.status === 'False')
        ); // 过滤至少一项有压力
        even && obj.unnormalNode++;
      } else {
        obj.unknownNode++;
      }
      obj.totalNode++;
    });
    return obj;
  } else {
    // 判断是否为主从节点
    let obj = {
      unnormalNode: 0,
      totalNode: 0,
      manageNode: 0,
      workerNode: 0,
      manageUnnormalNode: 0,
    };
    data.map((item) => {
      const keyArr = Object.keys(item.metadata.labels);
      const isManage = keyArr.filter(filterItem => filterItem.includes('control-plane'));
      const isWorker = keyArr.filter(filterItem => filterItem.includes('worker'));
      if (isManage.length) {
        obj.manageNode++; // host
        const overviewEven = item.status.conditions.some(
          (statusItem) =>
            (statusItem.type !== 'Ready' && statusItem.status === 'True') ||
            (statusItem.type === 'Ready' && statusItem.status === 'False')
        );
        overviewEven && obj.manageUnnormalNode++;
      } else {
        obj.workerNode++; // worker
        const even = item.status.conditions.some(
          (statusItem) =>
            (statusItem.type !== 'Ready' && statusItem.status === 'True') ||
            (statusItem.type === 'Ready' && statusItem.status === 'False')
        );
        even && obj.unnormalNode++;
      }
      obj.totalNode++;
    });
    return obj;
  }
}

/**
 * 内层判断是否繁忙
 */
export function nodeDetailIsBusy(statusList) {
  let busy = false;
  const unknown = statusList.some(item => item.status === 'Unknown');
  if (unknown) {
    busy = false;
  } else {
    const pressure = statusList.some(
      (statusItem) =>
        (statusItem.type !== 'Ready' && statusItem.status === 'True') ||
        (statusItem.type === 'Ready' && statusItem.status === 'False')
    );
    busy = pressure ? true : false;
  }
  return busy;
};

/**
 * 处理ip地址
 */
export function solveNodeIpAddress(address = []) {
  let ip = '--';
  const [filterList, ...resets] = address.filter(
    (item) => item.type === 'InternalIP'
  );
  filterList ? (ip = filterList.address) : (ip = '--');
  return ip;
}

/**
 * 处理节点健康状态
 * @param conditions 健康数组
 */
export function solveNodeHealthyStatus(conditions) {
  let healthyObj = {
    networkUnavailable: false,
    networkReason: '',
    memoryPressure: false,
    memoryReason: '',
    diskPressure: false,
    diskReason: '',
    pidPressure: false,
    pidReason: '',
    ready: true,
    readyReason: '',
  };
  conditions.map((item) => {
    if (item.type === 'NetworkUnavailable') {
      healthyObj.networkUnavailable = item.status === 'True';
      healthyObj.networkReason = item.message;
    }
    if (item.type === 'MemoryPressure') {
      healthyObj.memoryPressure = item.status === 'True';
      healthyObj.memoryReason = item.message;
    }
    if (item.type === 'DiskPressure') {
      healthyObj.diskPressure = item.status === 'True';
      healthyObj.diskReason = item.message;
    }
    if (item.type === 'PIDPressure') {
      healthyObj.pidPressure = item.status === 'True';
      healthyObj.pidReason = item.message;
    }
    if (item.type === 'Ready') {
      healthyObj.ready = item.status === 'True';
      healthyObj.readyReason = item.message;
    }
  });
  return healthyObj;
}

/**
 * 处理资源配额
 * @param allocatable 可用
 * @param capacity 存量
 */
export function solveNodeResource(allocatable, capacity) {
  let resourceObj = {
    cpuUse: parseInt(capacity.cpu) - parseInt(allocatable.cpu) || 0,
    cpuMax: parseInt(capacity.cpu) || 0,
    cpuPercent: 0,
    memoryUse: parseInt(allocatable.memory) || 0,
    memoryMax: parseInt(capacity.memory) || 0,
    memoryPercent: 0,
    podUse: parseInt(capacity.pods) - parseInt(allocatable.pods) || 0,
    podMax: parseInt(capacity.pods) || 0,
    podPercent: 0,
  };

  resourceObj.memoryUse = (
    (resourceObj.memoryMax - resourceObj.memoryUse) /
    1024 /
    1024
  ).toFixed(2);
  resourceObj.memoryMax = (resourceObj.memoryMax / 1024 / 1024).toFixed(2);
  resourceObj.cpuPercent = (resourceObj.cpuUse / resourceObj.cpuMax) * 100;
  resourceObj.cpuPercent = Number.isInteger(resourceObj.cpuPercent)
    ? resourceObj.cpuPercent
    : Number(resourceObj.cpuPercent.toFixed(2));
  resourceObj.podPercent = (resourceObj.podUse / resourceObj.podMax) * 100;
  resourceObj.podPercent = Number.isInteger(resourceObj.podPercent)
    ? resourceObj.podPercent
    : Number(resourceObj.podPercent.toFixed(2));
  resourceObj.memoryPercent =
    (resourceObj.memoryUse / resourceObj.memoryMax) * 100;
  resourceObj.memoryPercent = Number.isInteger(resourceObj.memoryPercent)
    ? resourceObj.memoryPercent
    : Number(resourceObj.memoryPercent.toFixed(2));

  return resourceObj;
}

/**
 * 计算页面总数
 * @param page 当前页
 * @param pageSize
 * @param currentTotal
 */
export function calculatePageTotal(page, pageSize, currentTotal, remain = 0) {
  return ((page - 1) * pageSize) + currentTotal + remain;
}

/**
 * 处理限制范围内table
 */
export function solveLimitTableList(data) {
  let finnalData = [];
  data.map((item) => {
    let keys = [];
    if (item.max) {
      keys = Object.keys(item.max); // 资源类型
    } else {
      keys = Object.keys(item.default); // 取key
    }
    keys.map((itemKey) => {
      finnalData.push({
        type: item.type,
        resource: itemKey,
        min: item.min ? item.min[itemKey] : '--',
        max: item.max ? item.max[itemKey] : '--',
        defaultRequest: item.defaultRequest ? (item.defaultRequest[itemKey] || '--') : '--',
        default: item.default ? item.default[itemKey] || '--' : '--',
        rate: item.defaultRequest && item.default ? item.default[itemKey] / item.defaultRequest[itemKey] : '--',
      });
    });
  });
  return finnalData;
}

/**
 * 处理命名空间
 * @param dataList 注解列表
 */
export function solveNamepaceShowName(dataList) {
  let showName = '';
  dataList.map((item) => {
    if (item.key.includes('display-name')) {
      showName = item.value;
    }
  });
  return showName;
}

/**
 * 处理resource
 */
export function solveResourceQuota(dataStatusList) {
  let status = '';
  if (dataStatusList.hard && dataStatusList.used) {
    Object.keys(dataStatusList.hard).map((item) => {
      let used = judgeStatusValue(dataStatusList.used[item]);
      let hard = judgeStatusValue(dataStatusList.hard[item]);
      if (used >= hard) {
        status = 'exist';
      }
    });
  }
  return status;
}

/**
 * 返回纯int
 * 处理单位换算
 */
export function judgeStatusValue(str) {
  let value = 0;
  if (str.includes('g') || str.includes('Gi')) {
    value = parseInt(str) * 1024 * 1024;
  } else if (str.includes('m') || str.includes('Mi')) {
    value = parseInt(str) * 1024;
  } else if (str.includes('k') || str.includes('Ki')) {
    value = parseInt(str);
  } else {
    value = parseInt(str) * 1024 * 1024;
  }
  return value;
}

export function judgeCpuValue(str) {
  let value = 0;
  if (str.includes('m')) {
    value = parseInt(str) * 1000 * 1000;
  } else if (str.includes('u')) {
    value = parseInt(str) * 1000;
  } else if (str.includes('n')) {
    value = parseInt(str);
  } else {
    value = parseInt(str) * 1000 * 1000 * 1000;
  }
  return value;
}

/**
 * 计算资源配额详情
 */
export function solveResourceEcharts(dataStatusList) {
  let resourceList = [];
  if (dataStatusList.hard && dataStatusList.used) {
    Object.keys(dataStatusList.hard).map((item) => {
      let used = item.includes('cpu') ? judgeCpuValue(dataStatusList.used[item]) : judgeStatusValue(dataStatusList.used[item]);
      let hard = item.includes('cpu') ? judgeCpuValue(dataStatusList.hard[item]) : judgeStatusValue(dataStatusList.hard[item]);
      resourceList.push({
        title: item,
        rate: Number.isInteger((used / hard) * 100)
          ? (used / hard) * 100
          : Number(((used / hard) * 100).toFixed(2)),
        used: dataStatusList.used[item],
        max: dataStatusList.hard[item],
      });
    });
  }
  return resourceList;
}

/**
 * 计算自定义资源
 */
export function solveCustomResourceStatus(data = []) {
  let status = false;
  data.map((item) => {
    if (item.type === 'Established') {
      item.status === 'True' && (status = true);
    }
  });
  return status;
}

/**
 * 处理路径中带.的名称
 */
export function solveEncodePath(name) {
  let str = '';
  str = name.replaceAll('.', '_'); // replaceAll
  return str;
}

/**
 * 解码
 */
export function solveDecodePath(name) {
  let str = '';
  str = name.replaceAll('_', '.');
  return str;
}

/**
 * 通过数量生成
 */
export function solveDataAreaNumPalette(num) {
  let colorList = ['#43CBBB', '#F9A975', '#4B8BEA', '#B4A2FF', '#Eb96F5'];
  const colLength = colorList.length;
  try {
    for (let range = 0; range < num - colLength; range++) {
      let color = '';
      color = `rgb( ${[
        Math.round(
          (window.crypto.getRandomValues(new Uint8Array(1))[0] / 256) * 255
        ), // 替换math.random()
        Math.round(
          (window.crypto.getRandomValues(new Uint8Array(1))[0] / 256) * 255
        ),
        Math.round(
          (window.crypto.getRandomValues(new Uint8Array(1))[0] / 256) * 255
        ),
      ].join(',')})`;
      while (colorList.some((i) => i === color)) {
        color = `rgb( ${[
          Math.round(
            (window.crypto.getRandomValues(new Uint8Array(1))[0] / 256) * 255
          ), // 替换math.random()
          Math.round(
            (window.crypto.getRandomValues(new Uint8Array(1))[0] / 256) * 255
          ),
          Math.round(
            (window.crypto.getRandomValues(new Uint8Array(1))[0] / 256) * 255
          ),
        ].join(',')})`;
      }
      colorList.push(color);
    }
  } catch (e) {
    // 
  }
  return colorList;
};

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
    for (let i = 0; i < currentBatchSize * 1.2; i++) { // 多生成一些以防重复
      batch.push(generateRandomRgb());
    }

    // 过滤掉重复颜色
    const uniqueBatch = batch.filter(color => !colorSet.has(color));

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
};

/**
 * 处理返回数学模型 最大最小平均值
 * @param data 数据
 * @param typeList 类型
 * @param colorList 对应颜色
 */
export function getMathValueList(data, typeList, colorList) {
  let tableData = [];
  let tableDataGroupObj = {};
  // 处理数据成某组
  typeList.map((item) => {
    if (!Object.prototype.hasOwnProperty.call(tableDataGroupObj, item)) {
      tableDataGroupObj[item] = [];
    }
    data.map(({ type, timestamp, value }) => {
      if (type === item) {
        tableDataGroupObj[item].push({
          timestamp,
          value,
        });
      }
    });
  });
  Object.keys(tableDataGroupObj).map((item, index) => {
    let min = 0;
    let max = 0;
    let average = 0;
    let current = 0;
    if (!tableDataGroupObj[item].length) {
      min = NaN;
      max = NaN;
      average = NaN;
      current = NaN;
    } else {
      tableDataGroupObj[item] = tableDataGroupObj[item].filter(filterItem => !isNaN(filterItem.value));
      if (tableDataGroupObj[item].length) {
        tableDataGroupObj[item].sort((a, b) => a.value - b.value); // 升序 从小到大
        max = (tableDataGroupObj[item][tableDataGroupObj[item].length - 1].value).toFixed(4);
        min = (tableDataGroupObj[item][0].value).toFixed(4);
        average = (
          tableDataGroupObj[item].reduce((acc, val) => acc + val.value, 0) /
          tableDataGroupObj[item].length
        ).toFixed(4);
        let timestampMedium = 0;
        tableDataGroupObj[item].map((timeItem) => {
          if (timeItem.timestamp > timestampMedium) {
            timestampMedium = timeItem.timestamp;
            current = (timeItem.value).toFixed(4);
          }
        });
      } else {
        min = NaN;
        max = NaN;
        average = NaN;
        current = NaN;
      }
    }
    tableData.push({
      legend: {
        type: item,
        color: colorList[index],
      },
      min,
      max,
      average,
      current,
    });
  });

  return tableData;
}

// 拼接多个相同参数
export function buildQueryString(name, list) {
  let queryString = '';
  if (list && list.length > 0) {
    const values = list.map((item) => `&${name}=${item}`).join('');
    queryString = `${values}`;
  }
  return queryString;
}
/**
 * 随机生成uuid
 */
export function randomUuid() {
  return window.crypto.getRandomValues(new Uint16Array(8))[0];
};

/**
 * 查找表
 */
export const hashMap = {
  etcd: data => getDataToEtcd(data),
  kubeApiserver: data => getDataToKubeApiServer(data),
  kubeControllerManager: data => getDataToKubeNode(data),
  kubeScheduler: data => getDataToKubeNode(data),
  kubelet: data => getDataToKubeNode(data),
  kubeProxy: data => getDataToKubeNode(data),
  coredns: data => data,
  cluster: data => getDataToCluster(data),
  node: data => getDataToNode(data),
  workload: data => data,
  pod: data => getDataToPod(data),
  container: data => getDataToPod(data),
};

/**
 * 处理监控控制组件数据元
 */
export function solveMonitorComponentData(value, data) {
  let newData = {};
  data.map(item => {
    const key = item.metricName;
    newData[key] = [];
    if (item.data.result) {
      item.data?.result.map(subItem => {
        let type = key;
        let newType = '';
        if (subItem.labels) {
          const labelResult = Object.values(subItem.labels);
          type = labelResult.join(` - `);
          newType = labelResult.join(` - `);
        }
        if (key.includes('load_average')) {
          type = `${key.slice(key.indexOf('node_') + 5, key.indexOf('load_average') - 1)}${newType ? ` - ${newType}` : ''}`;
        }
        if (key.includes('XX_request_rate')) {
          type = `${key.slice(key.indexOf('XX_request_rate') - 1, key.indexOf('XX_request_rate') + 2)}${newType ? ` - ${newType}` : ''}`;
        }
        if (key.includes('etcd_server_has_leader') || key.includes('etcd_server_is_leader')) {
          type = `${key.slice(key.indexOf('server') + 7)}${newType ? ` - ${newType}` : ''}`;
        }
        subItem.series = subItem.series.sort((a, b) => a.timestamp - b.timestamp); // 从小到大排序时间
        subItem.series.map(timeItem => {
          if (timeItem.value !== -1) {
            newData[key].push({
              type,
              timestamp: timeItem.timestamp,
              value: timeItem.value,
              labels: subItem.labels,
            });
          } else {
            newData[key].push({
              type,
              timestamp: timeItem.timestamp,
              value: NaN,
              labels: subItem.labels,
            });
          }
        });
      });
    }
  });
  // 处理逻辑
  let newValue = '';
  value.split('-').map((item, index) => {
    index ? newValue += firstAlphabetUp(item) : newValue += item;
  });
  newData = hashMap[newValue](newData);
  return newData;
};

/**
 * 处理etcd
 */
export function getDataToEtcd(data) {
  data.up = data.etcd_up_instances_count.length ? data.etcd_up_instances_count[data.etcd_up_instances_count.length - 1].value : '--';
  data.down = data.etcd_down_instances_count.length ? data.etcd_down_instances_count[data.etcd_down_instances_count.length - 1].value : '--';
  data.total = data.etcd_instances_total.length ? data.etcd_instances_total[data.etcd_instances_total.length - 1].value : '--';
  return data;
};

/**
 * 处理kube-apiserver
 */
export function getDataToKubeApiServer(data) {
  data.apiQps = data.kube_apiserver_qps.length ? Number((data.kube_apiserver_qps[data.kube_apiserver_qps.length - 1].value).toFixed(1)) : '--';
  data.readRequestSuccessRate =
    data.kube_apiserver_read_request_success_rate.length ?
      Number((data.kube_apiserver_read_request_success_rate[data.kube_apiserver_read_request_success_rate.length - 1].value * 100).toFixed(1)) : '--';
  data.writeRequestSuccessRate =
    data.kube_apiserver_write_request_success_rate.length ?
      Number((data.kube_apiserver_write_request_success_rate[data.kube_apiserver_write_request_success_rate.length - 1].value * 100).toFixed(1)) : '--';
  return data;
};

/**
 * 处理kube-controller以及kubelet\kube-proxy\kube-scheduler
 */
export function getDataToKubeNode(data) {
  const keysList = Object.keys(data);
  data.xxRequestRate = [];
  keysList.map(item => {
    if (item.includes('kube_api_') && item.includes('XX_request_rate')) {
      if (data.xxRequestRate.length) {
        data.xxRequestRate = data.xxRequestRate.concat(data[item]);
      } else {
        data.xxRequestRate = data[item];
      }
    }
  });
  return data;
}

/**
 * 处理cluster
 */
export function getDataToCluster(data) {
  data.nodesCount = data.cluster_nodes_count.length
    ? (data.cluster_nodes_count).slice(-1)[0]?.value
    : '--';
  data.nodesReadyCount = data.cluster_nodes_not_ready_count
    ? (data.cluster_nodes_not_ready_count).slice(-1)[0]?.value
    : '--';
  let podValue = 0;
  if (data.cluster_pods_count) {
    let podCountArr = {};
    data.cluster_pods_count.map(item => {
      const temporyPodType = item.type;
      if (!Object.keys(podCountArr).includes(temporyPodType)) {
        podCountArr[temporyPodType] = [];
      }
      podCountArr[temporyPodType].push(item);
    });
    const temporyKeyList = Object.keys(podCountArr);
    temporyKeyList.map(podItem => {
      podValue += podCountArr[podItem].length && podCountArr[podItem].slice(-1)[0].value;
    });
  } else {
    podValue = '--';
  }
  data.podsCount = podValue;
  data.clusterCpuUtilisationRate = data.cluster_cpu_utilisation.length ? Number(((data.cluster_cpu_utilisation).slice(-1)[0].value).toFixed(4)) : '--';
  data.requests = [];
  let namespaceList = [];
  data.cluster_pod_count_by_namespace.map(item => {
    if (!namespaceList.includes(item.type)) {
      namespaceList.push(item.type);
    }
  });
  namespaceList.map(item => {
    const podsList = data.cluster_pod_count_by_namespace.filter(filterItem => filterItem.type === item);
    const workloadList = data.cluster_workload_count_by_namespace.filter(filterItem => filterItem.type === item);
    const usageList = data.cluster_memory_usage_without_cache_by_namespace.filter(filterItem => filterItem.type === item);
    const requestsList = data.cluster_memory_requests_utilisation_by_namespace.filter(filterItem => filterItem.type === item);
    const memoryLimitsList = data.cluster_memory_limits_total_by_namespace.filter(filterItem => filterItem.type === item);
    const memoryLimitsRateList = data.cluster_memory_limits_utilisation_by_namespace.filter(filterItem => filterItem.type === item);
    data.requests.push({
      namespace: item,
      pods: podsList.length ? podsList.slice(-1)[0].value : '--',
      workloads: workloadList.length ? (workloadList.slice(-1)[0].value) : '--',
      usage: usageList.length ? Number((usageList.slice(-1)[0].value / 1024 / 1024).toFixed(2)) : '--',
      requests: requestsList.length ? (requestsList.slice(-1)[0].value).toFixed(2) : '--',
      memoryLimits: memoryLimitsList.length ? Number((memoryLimitsList.slice(-1)[0].value / 1024 / 1024).toFixed(2)) : '--',
      memoryLimitsRate: memoryLimitsRateList.length ? Number((memoryLimitsRateList.slice(-1)[0].value).toFixed(2)) : '--',
    });
  });
  return data;
};

// 瞬时数据统一处理
export function timelayDataSolve(key, originalData) {
  let data = {};
  data.requests = [];
  if (key === 'cluster') {
    originalData.map(item => {
      // 处理瞬时数据
      switch (item.metricName) {
        case 'cluster_nodes_count': data.nodesCount = item.data.result.reduce((accumulator, currentTarget) => { return accumulator + currentTarget.sample.value }, 0); break;
        case 'cluster_nodes_not_ready_count': data.nodesReadyCount = item.data.result.reduce((accumulator, currentTarget) => { return accumulator + currentTarget.sample.value }, 0); break;
        case 'cluster_pods_count': data.podsCount = item.data.result.reduce((accumulator, currentTarget) => { return accumulator + currentTarget.sample.value }, 0); break;
        case 'cluster_cpu_utilisation': data.clusterCpuUtilisationRate = Number(item.data.result.reduce((accumulator, currentTarget) => { return accumulator + currentTarget.sample.value }, 0)).toFixed(4); break;
        case 'cluster_pod_count_by_namespace': data.pods = item.data.result; break;
        case 'cluster_workload_count_by_namespace': data.workloads = item.data.result; break;
        case 'cluster_memory_usage_without_cache_by_namespace': data.usage = item.data.result; break;
        case 'cluster_memory_requests_utilisation_by_namespace': data.memopryRequest = item.data.result; break;
        case 'cluster_memory_limits_total_by_namespace': data.memoryLimit = item.data.result; break;
        case 'cluster_memory_limits_utilisation_by_namespace': data.memoryLimitsRate = item.data.result; break;
        default: '';
      }
    });
    // 处理表格数据
    let namespaceList = [];
    data.pods.map(item => {
      if (!namespaceList.includes(item.labels.namespace)) {
        namespaceList.push(item.labels.namespace);
      }
    });
    namespaceList.map(item => {
      const podsList = data.pods.filter(filterItem => filterItem.labels.namespace === item);
      const workloadList = data.workloads.filter(filterItem => filterItem.labels.namespace === item);
      const usageList = data.usage.filter(filterItem => filterItem.labels.namespace === item);
      const requestsList = data.memopryRequest.filter(filterItem => filterItem.labels.namespace === item);
      const memoryLimitsList = data.memoryLimit.filter(filterItem => filterItem.labels.namespace === item);
      const memoryLimitsRateList = data.memoryLimitsRate.filter(filterItem => filterItem.labels.namespace === item);
      data.requests.push({
        namespace: item,
        pods: podsList.length ? podsList.slice(-1)[0].sample.value : '--',
        workloads: workloadList.length ? (workloadList.slice(-1)[0].sample.value) : '--',
        usage: usageList.length ? Number((usageList.slice(-1)[0].sample.value / 1024 / 1024).toFixed(2)) : '--',
        requests: requestsList.length ? (requestsList.slice(-1)[0].sample.value).toFixed(2) : '--',
        memoryLimits: memoryLimitsList.length ? Number((memoryLimitsList.slice(-1)[0].sample.value / 1024 / 1024).toFixed(2)) : '--',
        memoryLimitsRate: memoryLimitsRateList.length ? Number((memoryLimitsRateList.slice(-1)[0].sample.value).toFixed(2)) : '--',
      });
    });
  }
  if (key === 'node') {
    originalData.map(item => {
      switch (item.metricName) {
        case 'node_memory_utilisation': data.nodeMemoryUtilisationRate = Number(item.data.result.reduce((accumulator, currentTarget) => { return accumulator + currentTarget.sample.value }, 0)).toFixed(4); break;
        case 'node_filesystem_size_bytes_by_mountpoint': data.nodeFilesystemSize = item.data.result; break;
        case 'node_filesystem_size_bytes_avail_by_mountpoint': data.nodeAvailSize = item.data.result; break;
        case 'node_filesystem_size_bytes_used_by_mountpoint': data.nodeSizeUsed = item.data.result; break;
        case 'node_filesystem_size_bytes_utilisation_by_mountpoint': data.nodeUtilisation = item.data.result; break;
        default: '';
      }
    });
    data.diskUsage = [];
    let monutedList = [];
    data.nodeFilesystemSize.map(item => {
      if (!monutedList.includes(`${item.labels.instance}-${item.labels.mountpoint}`)) {
        monutedList.push(`${item.labels.instance}-${item.labels.mountpoint}`);
      }
    });
    monutedList.map(item => {
      const sizeList = data.nodeFilesystemSize.filter(filterItem => `${filterItem.labels.instance}-${filterItem.labels.mountpoint}` === item);
      const avaliableList = data.nodeAvailSize.filter(filterItem => `${filterItem.labels.instance}-${filterItem.labels.mountpoint}` === item);
      const usedList = data.nodeSizeUsed.filter(filterItem => `${filterItem.labels.instance}-${filterItem.labels.mountpoint}` === item);
      const usedRateList = data.nodeUtilisation.filter(filterItem => `${filterItem.labels.instance}-${filterItem.labels.mountpoint}` === item);
      data.diskUsage.push({
        mounted: item,
        size: sizeList.length ? sizeList.slice(-1)[0].sample.value : '--',
        available: avaliableList.length ? avaliableList.slice(-1)[0].sample.value : '--',
        used: usedList.length ? usedList.slice(-1)[0].sample.value : '--',
        usedRate: usedRateList.length ? usedRateList.slice(-1)[0].sample.value : '--',
      });
    });
  }
  if (key === 'pod' || key === 'container') {
    originalData.map(item => {
      switch (item.metricName) {
        case 'pod_iops_reads': data.podReads = item.data.result; break;
        case 'pod_iops_writes': data.podWrites = item.data.result; break;
        case 'pod_iops_reads_writes': data.podReadsWrites = item.data.result; break;
        case 'pod_throughput_reads': data.podThoughtReads = item.data.result; break;
        case 'pod_throughput_writes': data.podThoughtWrites = item.data.result; break;
        case 'pod_throughput_reads_writes': data.podThoughtReadsWrites = item.data.result; break;
        default: '';
      }
    });
    data.storage = [];
    let monutedList = [];
    let podList = [];
    data.podReads.map(item => {
      let type = '';
      if (item.labels) {
        const labelResult = Object.values(item.labels);
        type = labelResult.join(` - `);
      }
      if (!monutedList.includes(type)) {
        monutedList.push(type);
        podList.push(item?.labels?.pod || '--');
      }
    });
    monutedList.map((item, index) => {
      const readList = data.podReads.filter(filterItem => filterItem.labels ? Object.values(filterItem.labels).join(' - ') === item : '');
      const writeList = data.podWrites.filter(filterItem => filterItem.labels ? Object.values(filterItem.labels).join(' - ') === item : '');
      const readWriteList = data.podReadsWrites.filter(filterItem => filterItem.labels ? Object.values(filterItem.labels).join(' - ') === item : '');
      const throughReadList = data.podThoughtReads.filter(filterItem => filterItem.labels ? Object.values(filterItem.labels).join(' - ') === item : '');
      const throughWriteList = data.podThoughtWrites.filter(filterItem => filterItem.labels ? Object.values(filterItem.labels).join(' - ') === item : '');
      const throughReadWriteList = data.podThoughtReadsWrites.filter(filterItem => filterItem.labels ? Object.values(filterItem.labels).join(' - ') === item : '');
      data.storage.push({
        pod: podList[index],
        iopsRead: readList.length ? Number((readList.slice(-1)[0].sample.value).toFixed(3)) ?? '--' : '--',
        iopsWrite: writeList.length ? Number(writeList.slice(-1)[0].sample.value.toFixed(3)) ?? '--' : '--',
        iopsReadWrite: readWriteList.length ? Number(readWriteList.slice(-1)[0].sample.value.toFixed(3) ?? '--') : '--',
        throughputRead: throughReadList.length ? Number(throughReadList.slice(-1)[0].sample.value.toFixed(2)) ?? '--' : '--',
        throughputWrite: throughWriteList.length ? Number(throughWriteList.slice(-1)[0].sample.value.toFixed(2)) ?? '--' : '--',
        throughputReadWrite: throughReadWriteList.length ? Number(throughReadWriteList.slice(-1)[0].sample.value.toFixed(2)) ?? '--' : '--',
      });
    });
  }
  return data;
};

// compoennt顺势数据
export function timelayDataComponentSolve(key, originalData) {
  let data = {};
  if (key === 'etcd') {
    originalData.map(item => {
      switch (item.metricName) {
        case 'etcd_up_instances_count': data.up = item.data.result.reduce((accumulator, currentTarget) => { return accumulator + currentTarget.sample.value }, 0); break;
        case 'etcd_down_instances_count': data.down = item.data.result.reduce((accumulator, currentTarget) => { return accumulator + currentTarget.sample.value }, 0); break;
        case 'etcd_instances_total': data.total = item.data.result.reduce((accumulator, currentTarget) => { return accumulator + currentTarget.sample.value }, 0); break;
        default: '';
      }
    });
  }
  if (key === 'kube-apiserver') {
    originalData.map(item => {
      switch (item.metricName) {
        case 'kube_apiserver_qps': data.apiQps = item.data.result ? Number((item.data.result[0].sample.value).toFixed(1)) : '--'; break;
        case 'kube_apiserver_read_request_success_rate': data.readRequestSuccessRate = item.data.result ? Number((item.data.result[0].sample.value * 100).toFixed(1)) : '--'; break;
        case 'kube_apiserver_write_request_success_rate': data.writeRequestSuccessRate = item.data.result ? Number((item.data.result[0].sample.value * 100).toFixed(1)) : '--'; break;
        default: '';
      }
    });
  }
  return data;
};


/**
 * 处理node
 */
export function getDataToNode(data) {
  const keysList = Object.keys(data);
  data.nodeLoadUsage = [];
  keysList.map(item => {
    if (item.includes('load_average')) {
      if (data.nodeLoadUsage.length) {
        data.nodeLoadUsage = data.nodeLoadUsage.concat(data[item]);
      } else {
        data.nodeLoadUsage = data[item];
      }
    }
  });
  data.nodeMemoryUtilisationRate = data.node_memory_utilisation.length ? Number((data.node_memory_utilisation.slice(-1)[0].value).toFixed(4)) : 0;
  data.disk = [];
  data.node_disk_read_bytes_by_device.length && (data.disk = [...data.disk, ...sovleTypeData(data.node_disk_read_bytes_by_device, 'node_disk_read_bytes_by_device')]);
  data.node_disk_written_bytes_by_device.length && (data.disk = [...data.disk, ...sovleTypeData(data.node_disk_written_bytes_by_device, 'node_disk_written_bytes_by_device')]);
  data.node_disk_io_time_seconds_by_device.length && (data.disk = [...data.disk, ...sovleTypeData(data.node_disk_io_time_seconds_by_device, 'node_disk_io_time_seconds_by_device')]);
  data.diskUsage = [];
  let monutedList = [];
  data.node_filesystem_size_bytes_by_mountpoint.map(item => {
    if (!monutedList.includes(item.type)) {
      monutedList.push(item.type);
    }
  });
  monutedList.map(item => {
    const sizeList = data.node_filesystem_size_bytes_by_mountpoint.filter(filterItem => filterItem.type === item);
    const avaliableList = data.node_filesystem_size_bytes_avail_by_mountpoint.filter(filterItem => filterItem.type === item);
    const usedList = data.node_filesystem_size_bytes_used_by_mountpoint.filter(filterItem => filterItem.type === item);
    const usedRateList = data.node_filesystem_size_bytes_utilisation_by_mountpoint.filter(filterItem => filterItem.type === item);
    data.diskUsage.push({
      mounted: item,
      size: sizeList.length ? sizeList.slice(-1)[0].value : '--',
      available: avaliableList.length ? avaliableList.slice(-1)[0].value : '--',
      used: usedList.length ? usedList.slice(-1)[0].value : '--',
      usedRate: usedRateList.length ? usedRateList.slice(-1)[0].value : '--',
    });
  });
  return data;
};

/**
 * 处理pod
 */
export function getDataToPod(data) {
  data.storage = [];
  let monutedList = [];
  let podList = [];
  data.pod_iops_reads.map(item => {
    if (!monutedList.includes(item.type)) {
      monutedList.push(item.type);
      podList.push(item?.labels?.pod || '--');
    }
  });
  monutedList.map((item, index) => {
    const readList = data.pod_iops_reads.filter(filterItem => filterItem.type === item);
    const writeList = data.pod_iops_writes.filter(filterItem => filterItem.type === item);
    const readWriteList = data.pod_iops_reads_writes.filter(filterItem => filterItem.type === item);
    const throughReadList = data.pod_throughput_reads.filter(filterItem => filterItem.type === item);
    const throughWriteList = data.pod_throughput_writes.filter(filterItem => filterItem.type === item);
    const throughReadWriteList = data.pod_throughput_reads_writes.filter(filterItem => filterItem.type === item);
    data.storage.push({
      pod: podList[index],
      iopsRead: readList.length ? Number((readList.slice(-1)[0].value).toFixed(3)) || '--' : '--',
      iopsWrite: writeList.length ? Number(writeList.slice(-1)[0].value.toFixed(3)) || '--' : '--',
      iopsReadWrite: readWriteList.length ? Number(readWriteList.slice(-1)[0].value.toFixed(3) || '--') : '--',
      throughputRead: throughReadList.length ? Number(throughReadList.slice(-1)[0].value.toFixed(2)) || '--' : '--',
      throughputWrite: throughWriteList.length ? Number(throughWriteList.slice(-1)[0].value.toFixed(2)) || '--' : '--',
      throughputReadWrite: throughReadWriteList.length ? Number(throughReadWriteList.slice(-1)[0].value.toFixed(2)) || '--' : '--',
    });
  });
  return data;
};

/**
 * 处理类型数据
 */
export function sovleTypeData(data, propsType) {
  let result = [];
  data.map(({ type, timestamp, value }) => {
    result.push({ type: `${propsType}-${type}`, timestamp, value });
  });
  return result;
};

/**
 * 处理自定义查询数据清洗
 */
export function solveQueryMonitor(data) {
  let result = {
    chartDataList: [],
    chartTableList: [],
    tableColumns: [],
  };
  if (data && data.length) {
    result.tableColumns = ['idSelf', 'timestamp', ...Object.keys(data[0].labels), 'value'];
    data.map((item, index) => {
      item.series.map(seriesItem => {
        result.chartDataList.push({
          type: index,
          timestamp: seriesItem.timestamp,
          value: seriesItem.value,
        });
      });
      const valueListTempory = (result.chartDataList.filter(filterItem => filterItem.type === index).slice(-1));
      result.chartTableList.push(
        {
          ...item.labels,
          idSelf: index,
          value: valueListTempory.length ? valueListTempory[0].value : '--',
          timestamp: valueListTempory.length ? valueListTempory[0].timestamp : '--',
        }
      ); // 获取值
    });
  }
  return result;
};

/**
 * 转化格式
 */
export function solveBytesToMiB(data, isBytes = true) {
  let result = data;
  if (data && data.length) {
    if (isBytes) {
      const solveData = data.filter(filterItem => !isNaN(filterItem.value));
      const isSolve = solveData.every(item => item.value > 10485);
      if (isSolve) {
        result = data.map(({ type, timestamp, value }) => {
          return { type, timestamp, value: Number((value / 1024 / 1024).toFixed(2)) };
        });
      } else {
        result = data.map(({ type, timestamp, value }) => {
          return { type, timestamp, value: Number((value).toFixed(4)) };
        });
      }
    } else {
      result = data.map(({ type, timestamp, value }) => {
        if (value) { // 0.01MiB
          return { type, timestamp, value: Number((value).toFixed(4)) };
        } else {
          return { type, timestamp, value };
        }
      });
    }
  }
  return result;
};

/**
 * 处理比率
 */
export function solveRate(data) {
  let result = data;
  if (data && data.length) {
    result = data.map(({ type, timestamp, value }) => {
      return { type, timestamp, value: Number((value * 100).toFixed(2)) };
    });
  }
  return result;
};

/**
 * 处理单位u
 */
export function solveUnitMonitor(data) {
  let unit = 'bytes';
  if (data && data.length) {
    const unitList = data.some(item => item.value > 10485);
    if (unitList) {
      unit = 'MiB';
    }
  }
  return unit;
};

/**
 * 单个数据转化
 */
export function solveEachBytesToMB(each) {
  let result = each;
  if (each) {
    result = Number((each / 1024 / 1024).toFixed(2));
  }
  if (each === -1 || isNaN(each) || each === '--') {
    result = '--';
  }
  return result;
};

/**
 * 处理概览节点图中的数据
 * @param data []
 */
export function solveNodeChartInfo(data = []) {
  let _arr = [];
  data.map((item) => {
    let isNormal = false;
    let fcount = 0;
    let kcount = 0;
    let ucount = 0;
    for (let i = 0; i < item.status.conditions.length; i++) {
      if (item.status.conditions[i].status === 'Unknown') {
        ucount++;
        break;
      }
      const networkUnavailable = item.status.conditions[i].type === 'NetworkUnavailable' && item.status.conditions[i].status === 'False';
      const memoryPressure = item.status.conditions[i].type === 'MemoryPressure' && item.status.conditions[i].status === 'False';
      const diskPressure = item.status.conditions[i].type === 'DiskPressure' && item.status.conditions[i].status === 'False';
      const pidPressure = item.status.conditions[i].type === 'PIDPressure' && item.status.conditions[i].status === 'False';
      const ready = item.status.conditions[i].type === 'Ready' && item.status.conditions[i].status === 'True';
      if (networkUnavailable || memoryPressure || diskPressure) {
        isNormal = true;
      } else if (pidPressure || ready) {
        isNormal = true;
      } else {
        isNormal = false;
        fcount++;
        break;
      }
    }
    if (isNormal && !ucount) {
      kcount++;
    }
    if (kcount) {
      _arr.push({ item: '空闲', count: kcount, color: '#77aef7' });
    }
    if (fcount) {
      _arr.push({ item: '繁忙', count: fcount, color: '#f4840c' });
    }
    if (ucount) {
      _arr.push({ item: '其他', count: ucount, color: '#ccc' });
    }
  });
  return _arr;
};


/**
 * 导出
 * name 导出名称
 * data 数据
 */
export function exportYamlOutPut(name, data) {
  const link = document.createElement('a');
  link.setAttribute('href', `data:text/txt;charset=utf-8,${encodeURIComponent(data)}`);
  link.setAttribute('download', `${name}.yaml`);
  link.style.display = 'none';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

/**
 * 导出日志 txt
 * name 导出名称
 * data 数据
 */
export function exportHelmLogOutPut(name, data) {
  const link = document.createElement('a');
  link.setAttribute('href', `data:text/txt;charset=utf-8,${encodeURIComponent(data)}`);
  link.setAttribute('download', `${name}.txt`);
  link.style.display = 'none';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

/**
 * 解决限制范围比率
 */
export function solveLimitRangeRate(defaultFirst, defaultRequest) {
  let rate = '--';
  if (!defaultFirst || !defaultRequest) {
    return rate;
  } else if (defaultFirst === '--' || defaultRequest === '--') {
    return rate;
  } else {
    let defaultCpu = parseFloat(defaultFirst);
    let requestCpu = parseFloat(defaultRequest);
    if (defaultFirst.includes('G')) {
      defaultCpu = parseFloat(defaultFirst) * 1024;
    }
    if (defaultRequest.includes('G')) {
      requestCpu = parseFloat(defaultRequest) * 1024;
    }

    return `${((defaultCpu / requestCpu) * 100)}%`;
  }
}

/**
 * 解决服务端时区相差多少分钟
 */
export function solveCompareServerAndClientTime(server, client) {
  let offset = 0;
  if (server && client) {
    let serverDelimiter = server.slice(3, 4) === '+' ? true : false;
    let clientDelimiter = client.slice(3, 4) === '+' ? true : false;

    let serverMinutes = parseInt(server.slice(server.indexOf(':') + 1));
    let clientMinutes = parseFloat(client.slice(4)) * 60; // 对应分钟
    const serverTime = serverDelimiter
      ? +Number((parseInt(server.slice(4)) * 60) + serverMinutes)
      : -Number((parseInt(server.slice(4)) * 60) + serverMinutes); // +8
    const clientTime = clientDelimiter ? +Number(clientMinutes) : -Number(clientMinutes); // +9
    offset = clientTime - serverTime;
  }
  return offset;
};

// 随机生成色板
export function solveDataAreaNumPaletteTrash(num) {
  let colorList = ['#43CBBB', '#F9A975', '#4B8BEA', '#B4A2FF', '#EB96F5'];
  for (let i = 0; i < num - 5; i++) {
    const hue = i * 360 / num;
    const saturation = 50 + ((window.crypto.getRandomValues(new Uint8Array(1))[0] / 256) * 50);
    const lightness = 50 + ((window.crypto.getRandomValues(new Uint8Array(1))[0] / 256) * 25);
    const rgb = randomRgb(hue, saturation, lightness);
    const color = rgbToHex(rgb);
    if (colorList.filter(item => item === color).length) {
      continue;
    }
    colorList.push(color);
  }
  return colorList;
};

function randomRgb(hue, saturation, lightness) {
  let s = s / 100;
  let l = lightness / 100;
  const k = n => (n + (hue / 30)) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = n => l - (a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1))));
  return [Math.round(f(0) * 255), Math.round(f(8) * 255), Math.round(f(4) * 255)];
};

function rgbToHex([r, g, b]) {
  return `#${((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1).toUpperCase()}`;
};

/**
 * 获取对象数组某一个key值下的聚合
 */
export function getMappingValueArrayOfKey(array, keyName) {
  if (Object.prototype.toString.call(array) === '[object Array]') {
    return array.map(item => {
      return item[keyName];
    });
  }
  return [];
}

/**
 * antd表格按名字首字母顺序排序(全排序)
 */
export function sorterFirstAlphabe(a, b) {
  return a.localeCompare(b);
}
/**
 * 判断seesion是否含有某个属性
 */
export function checkSessionStorage(key) {
  return sessionStorage.getItem(key) !== null;
}

// 字符串转ASCII码
export function stringToAscii(str) {
  let asciiArray = [];
  for (let i = 0; i < str.length; i++) {
    asciiArray.push(str.charCodeAt(i));
  }
  return asciiArray;
}

export function oldSort(a, b) {
  return a.slice(0, 1).charCodeAt(0) - b.slice(0, 1).charCodeAt(0); // 首字母排序
}

export function sorterFirstAlphabet(a = '--', b = '--') {
  if (typeof (a) === 'number' && typeof (b) === 'number') {
    return a - b;
  }
  return (a.toString()).localeCompare(b.toString());
}

/**
 * antd表格pv pvc容量排序
 */
export function sorterStorage(a, b) {
  const beginStorage = a.slice(0, -2);
  const endStorage = b.slice(0, -2);
  return beginStorage - endStorage;
}

/**
 * 函数防抖
 */
export function debounce(func, delay) {
  let timer = null;
  return (...args) => {
    if (timer) {
      clearTimeout(timer);
    }
    timer = setTimeout(() => {
      func.apply(this, args);
    }, delay);
  };
}

/**
 * 转换数字显示（NaN,-1为--)
 */
export function transformUnifiedUnit(data) {
  let isTransform = false;
  if (isNaN(data)) {
    isTransform = true;
  }
  if (data === -1) {
    isTransform = true;
  }
  return isTransform;
}

/**
 * 截取第一个带数字的-前的数据
 */
export function getContentBeforeFirstDigitDash(str) {
  // 替换操作，将匹配到的内容返回
  let arr = str.split('-');
  let backStr = '';
  let backArr = [];
  for (let i = 0; i < arr.length - 1; i++) {
    if (!/\d/.test(arr[i])) {
      backArr.push(arr[i]);
    }
  }
  backStr = backArr.join('-');
  return backStr;
};

/**
 * 将403整体抽出
 */
export function forbiddenMsg(message, errorMsg) {
  if (errorMsg.response.data.message.includes('User')) {
    return message.error('操作失败，当前用户没有操作权限，请联系管理员添加权限!');
  } else {
    return message.error('请求被服务端拒绝，建议修改该请求!');
  }
};

export function generateDistinctColors(count) {
  const colors = new Set();
  const goldenRatioConjugate = 0.618033988749895;
  let hue = Math.random(); // 随机起始点
  while (colors.size < count) {
    hue += goldenRatioConjugate;
    hue %= 1; // 保持在[0,1)范围内
    const saturation = 0.7 + (Math.random() * 0.3); // 0.7-1.0
    const value = 0.5 + (Math.random() * 0.5); // 0.5-1.0

    const hI = Math.floor(hue * 6);
    const f = (hue * 6) - hI;
    const p = value * (1 - saturation);
    const q = value * (1 - (f * saturation));
    const t = value * (1 - ((1 - f) * saturation));

    let r = 0;
    let g = 0;
    let b = 0;
    switch (hI % 6) {
      case 0: r = value; g = t; b = p; break;
      case 1: r = q; g = value; b = p; break;
      case 2: r = p; g = value; b = t; break;
      case 3: r = p; g = q; b = value; break;
      case 4: r = t; g = p; b = value; break;
      case 5: r = value; g = p; b = q; break;
      default: break;
    }
    const toHex = (x) => Math.floor(x * 255).toString(16).padStart(2, '0');
    const color = `#${toHex(r)}${toHex(g)}${toHex(b)}`;
    colors.add(color);
  }

  return Array.from(colors);
};

export function rgbToHexOut(r, g, b) {
  // 确保数值在0-255范围内
  const clamp = (val) => Math.max(0, Math.min(255, val));

  // 转换为16进制并补零
  const toHex = (val) => {
    const hex = clamp(val).toString(16);
    return hex.length === 1 ? `0${hex}` : hex;
  };

  return `#${toHex(r)}${toHex(g)}${toHex(b)}`.toUpperCase();
};