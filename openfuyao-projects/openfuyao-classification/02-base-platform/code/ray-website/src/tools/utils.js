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
import jsYaml from 'js-yaml';
import dayjs from 'dayjs';
import { containerRouterPrefix, lokiResultWord } from '@/common/constants';

// 去重过滤value
export function filterRepeatVolume(arr) {
  const newArr = arr.filter((item, index, self) => {
    return self.findIndex((obj) => obj.value === item.value && obj.name === item.name) === index;
  });
  return newArr;
}

export function sorterFirstAlphabet(a = '--', b = '--') {
  if (typeof a === 'number' && typeof b === 'number') {
    return a - b;
  }
  return a.toString().localeCompare(b.toString());
}

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

export function exportYamlOutPutLog(name, data) {
  const link = document.createElement('a');
  link.setAttribute('href', `data:text/txt;charset=utf-8,${encodeURIComponent(data)}`);
  link.setAttribute('download', `${name}.log`);
  link.style.display = 'none';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

export function jsonToYaml(text) {
  let yaml = '';
  try {
    let jsonObj = JSON.parse(text);
    yaml = jsYaml.dump(jsonObj);
  } catch (error) {
    const str = `json转yaml失败,详细信息为:${e}`;
  }
  return yaml;
}

export function yamlTojson(text) {
  return JSON.parse(JSON.stringify(jsYaml.load(text)));
}

export function filterClusterStatusValue(data) {
  let background = '';
  switch (data) {
    case '未知':
      background = '#89939b';
      break;
    case '运行中':
      background = '#09aa71';
      break;
    case '未运行':
      background = '#fcd72e';
      break;
    case '失败':
      background = '#e7434a';
      break;
    case '不健康':
      background = '#f4840c';
      break;
    default: {
    }
  }
  return background;
}
export function filterServiceStatusValue(data) {
  let background = '';
  switch (data) {
    case '未知':
      background = '#89939b';
      break;
    case '运行中':
      background = '#09aa71';
      break;
    case '未运行':
      background = '#fcd72e';
      break;
    case '失败':
      background = '#e7434a';
      break;
    case '重启中':
      background = '#3f66f5';
      break;
    case '服务部署中':
      background = '#715afb';
      break;
    default: {
    }
  }
  return background;
}
export function filterJobStatusValue(data) {
  let background = '';
  switch (data) {
    case '未知':
      background = '#89939b';
      break;
    case '运行中':
      background = '#09aa71';
      break;
    case '未运行':
      background = '#fcd72e';
      break;
    case '失败':
      background = '#e7434a';
      break;
    case '完成':
      background = '#87c859';
      break;
    case '初始化':
      background = '#77aef7';
      break;
    case '重试中':
      background = '#3f66f5';
      break;
    case '暂停中':
      background = '#cccccc';
      break;
    default: {
    }
  }
  return background;
}

export function workHeadConfigTh(data) {
  let th = '';
  switch (data) {
    case 'nodeConfigName':
      th = '节点配置名称';
      break;
    case 'npu':
      th = 'NPU';
      break;
    case 'cpu':
      th = 'CPU';
      break;
    case 'memory':
      th = '内存';
      break;
    case 'maxCopy':
      th = '最大副本数';
      break;
    case 'minCopy':
      th = '最小副本数';
      break;
    default: {
    }
  }
  return th;
}

export function filterTimeValue(data) {
  let timeArr = [dayjs(), dayjs().unix()];
  switch (data) {
    case '10m':
      timeArr[0] = timeArr[0].subtract(10, 'm').unix();
      break;
    case '30m':
      timeArr[0] = timeArr[0].subtract(30, 'm').unix();
      break;
    case '1h':
      timeArr[0] = timeArr[0].subtract(1, 'h').unix();
      break;
    case '3h':
      timeArr[0] = timeArr[0].subtract(3, 'h').unix();
      break;
    case '6h':
      timeArr[0] = timeArr[0].subtract(6, 'h').unix();
      break;
    case '1d':
      timeArr[0] = timeArr[0].subtract(1, 'd').unix();
      break;
    case '3d':
      timeArr[0] = timeArr[0].subtract(3, 'd').unix();
      break;
    case '7d':
      timeArr[0] = timeArr[0].subtract(7, 'd').unix();
      break;
    case '14d':
      timeArr[0] = timeArr[0].subtract(14, 'd').unix();
      break;
    default: {
    }
  }
  return timeArr;
}

export function validateName(_rule, value) {
  const regex = /^[a-z0-9]([a-z0-9.-]{0,251}[a-z0-9])?$/;
  if (value && !regex.test(value)) {
    return Promise.reject(
      new Error('名称由小写字母、数字和.-组成，以字母或者数字开头且以字母或者数字结尾，不能超过253个字符！'),
    );
  } else {
    return Promise.resolve();
  }
}
export function validateImage(_rule, value) {
  const regex = /[a-zA-Z0-9_][a-zA-Z0-9._-]{0,127}/;
  if (value && !regex.test(value)) {
    return Promise.reject(new Error('名称由字母、数字或_，后续可包含字母，数字和_.-，不能超过128个字符！'));
  } else {
    return Promise.resolve();
  }
}
/**
 * 跳转加发送信息
 */
export function skipSendMainMsg(event, key, url, history) {
  event && event.preventDefault();
  window.parent.postMessage({
    type: 'subToMainMsg',
    data: key,
    url: `/${containerRouterPrefix}${url}`,
    isInsertMainPrefix: true,
  });
  !window.__OPENFUYAO__ && history.push(url);
}

/**
 * 通过数量生成
 */
export function solveDataAreaNumPalette(num) {
  let colorList = ['#77AEF7', '#2CB8C9', '#715AFB', '#F4840C', '#Eb96F5'];
  const colLength = colorList.length;
  try {
    for (let range = 0; range < num - colLength; range++) {
      let color = '';
      color = `rgb( ${[
        Math.round((window.crypto.getRandomValues(new Uint8Array(1))[0] / 256) * 255), // 替换math.random()
        Math.round((window.crypto.getRandomValues(new Uint8Array(1))[0] / 256) * 255),
        Math.round((window.crypto.getRandomValues(new Uint8Array(1))[0] / 256) * 255),
      ].join(',')})`;
      colorList.push(color);
    }
  } catch (e) {
    //
  }
  return colorList;
}

const formatterXLabel = (step, value, type = '') => {
  let returnValue = dayjs(value).format('HH:mm:ss');
  if (step.includes('m')) {
    returnValue = type ? dayjs(value).format('HH:mm') : dayjs(value).format('HH:mm');
  }
  if (step.includes('h')) {
    returnValue = dayjs(value).format('HH:mm');
  }
  if (step.includes('d')) {
    returnValue = dayjs(value).format('MM/DD\nHH:mm');
  }
  return returnValue;
};

/**
 * 封装方法
 */
export function birthOptions(time, xData = [], series = null, config = {}, type = '') {
  let options = {
    xAxis: {
      type: series && series.length ? 'time' : 'time',
      min: xData[0],
      max: xData[xData.length - 1],
      splitNumber: 5,
      axisLabel: {
        formatter: (value) => formatterXLabel(time, value, type),
      },
      data: xData,
    },
    tooltip: {
      trigger: 'axis',
    },
    grid: {
      left: '12%',
      top: '10%',
      bottom: '5%',
      containLabel: true,
    },
    legend: false,
    yAxis: {
      type: 'value',
      ...config,
    },
    series,
  };
  return options;
}

const hashMap = {
  cpu: { defaultName: 'MAX + PENDING', name: '', sub: '' },
  nodeCpu: { defaultName: 'MAX', name: 'CPU Usage', sub: '（head）' },
  manageMemory: { defaultName: 'MAX + PENDING', name: '', sub: '' },
  nodeMemory: { defaultName: 'MAX', name: 'Memory Used', sub: '（head）' },
  manageGpu: { defaultName: 'MAX + PENDING', name: '', sub: '' },
  nodeGpu: { defaultName: 'MAX', name: 'GPU Usage', sub: '（head）' },
  nodeGpuMemory: { defaultName: 'MAX', name: 'Memory Used', sub: '（head）' },
  manageNpu: { defaultName: 'MAX + PENDING', name: '', sub: '' },
  nodeNpu: { defaultName: 'MAX', name: 'NPU Usage', sub: '（head）' },
  nodeNpuMemory: { defaultName: 'MAX', name: 'Memory Used', sub: '（head）' },
};

export const birthEmptyDayTime = (step, start, end) => {
  let xAxis = [];
  let count = parseInt(step);
  let x = start;
  while (x < end) {
    xAxis.push(dayjs((x + count) * 1000).format('YYYY-MM-DD HH:mm:ss'));
    x += count;
  }
  return xAxis;
};

export function solveEchartsData(type, result, colorList = []) {
  let series = [];
  let xData = [];
  let echartsSeries = [];
  const colorMap = new Map(); // 映射表
  if (result && result.length) {
    let count = 0;
    result.map((item, index) => {
      let newSeries = [];
      let metric =
        item.metric && Object.keys(item.metric).length
          ? `${hashMap[type].name}:${item.metric.instance}@${item.metric.job}@${hashMap[type].sub}`
          : hashMap[type].defaultName;
      if (!colorMap.get(metric)) {
        // 防止替换颜色
        colorMap.set(metric, colorList[count]);
        count++;
      }
      item.values.map(([time, value]) => {
        xData.push(time);
        newSeries.push({
          name: metric,
          time,
          value,
        });
      });
      series = series.concat(newSeries);
    });

    // 处理数据
    let aimList = [];
    series.forEach((serieItem) => {
      aimList.push(serieItem.name);
    });
    aimList = [...new Set(aimList)];
    aimList.forEach((aimItem) => {
      const needData = series.filter((seItem) => aimItem === seItem.name);
      let data = [];
      needData.forEach((needItem) => {
        data.push([dayjs(needItem.time * 1000).format('YYYY-MM-DD HH:mm:ss'), needItem.value]);
      });
      echartsSeries.push({
        name: aimItem.replace(/@[^@]+@/g, ''),
        type: 'line',
        stack: 'Total',
        data,
        emphasis: {
          focus: 'series', // 悬停时高亮整个系列
        },
        itemStyle: {
          color: colorMap.get(aimItem),
        },
        lineStyle: {
          type: 'solid',
        },
        showSymbol: false,
      });
    });
  }
  let legend = [];
  colorMap.forEach((value, key) => {
    legend.push({ color: value, name: key.replace(/@[^@]+@/g, '') });
  });
  return { xData, series: echartsSeries, legend };
}

const gcsHashMap = {
  gcsStateTotal: 'GCS Global State Total Count',
  gcsStateActive: 'GCS Global State Active Total Count',
  gcsQueueTime: 'GCS Event Queueing Time',
};

const lokiHashMap = {
  gcsStateTotal: 'GCS Global State Total Count',
  gcsStateActive: 'GCS Global State Active Total Count',
  gcsQueueTime: 'GCS Event Queueing Time',
  gcsHealthyCheck: 'GCS health',
  gcsRayLet: 'Raylet health',
  jobGet: 'GET /api/jobs/-',
  jobPost: 'POST /api/jobs/-',
  jobDelete: 'DELETE /api/jobs/-',
  jobLogGet: 'GET /api/jobs/-/logs',
  jobLogDelete: 'DELETE /api/jobs/-/logs',
  dashboardNode: 'GET /nodes',
  dashboardLog: 'GET /logical/actors',
  dashboardJob: 'GET /api/jobs/',
};

const yNameMap = {
  gcsStateTotal: '',
  gcsStateActive: '',
  gcsQueueTime: '排队用时(ms)',
  gcsHealthyCheck: '接口用时(ms)',
  gcsRayLet: '接口用时(ms)',
  jobGet: '接口用时(ms)',
  jobPost: '接口用时(ms)',
  jobDelete: '接口用时(ms)',
  jobLogGet: '接口用时(ms)',
  jobLogDelete: '接口用时(ms)',
  dashboardNode: '接口用时(ms)',
  dashboardLog: '接口用时(ms)',
  dashboardJob: '接口用时(ms)',
};

const dashboardHashMap = {
  dashboardNode: 0,
  dashboardLog: 1,
  dashboardJob: 2,
};

export function solveLokiEchartsData(type, result, colorList = []) {
  let series = [];
  let xData = [];
  let echartsSeries = [];
  const colorMap = new Map(); // 映射表
  let max = 500;
  let min = 0;
  if (result && result.length) {
    let count = 0;

    result.map((item, index) => {
      let newSeries = [];
      let metric = lokiHashMap[type];
      if (!colorMap.get(metric)) {
        // 防止替换颜色
        if (type === 'dashboardNode' || type === 'dashboardLog' || type === 'dashboardJob') {
          colorMap.set(metric, colorList[dashboardHashMap[type]]);
        } else {
          colorMap.set(metric, colorList[count]);
          count++;
        }
      }
      item.values.map(([time, value]) => {
        if (!xData.filter((xitem) => xitem === Math.trunc(time / 1000000)).length) {
          xData.push(Math.trunc(time / 1000000));
        }
        newSeries.push({
          name: metric,
          time,
          value: Object.keys(gcsHashMap).includes(type) ? JSON.parse(value)[lokiResultWord.gcs[type]] : value,
        });
      });
      series = series.concat(newSeries);
    });
    // 处理数据
    let aimList = [];
    series.forEach((serieItem) => {
      aimList.push(serieItem.name);
    });
    aimList = [...new Set(aimList)];
    aimList.forEach((aimItem) => {
      const needData = series.filter((seItem) => aimItem === seItem.name);
      let data = [];
      needData.forEach((needItem) => {
        // 判断重复塞值
        const [isExist, ...resets] = data.filter(
          ([oldTime, oldValue]) => oldTime === dayjs(Math.trunc(needItem.time / 1000000)).format('YYYY-MM-DD HH:mm:ss'),
        );
        if (isExist) {
          data.map((item) => {
            if (item[0] === isExist[0]) {
              if (!item[1]) {
                item[1] = needItem.value || '';
              }
            }
          });
        } else {
          data.push([dayjs(Math.trunc(needItem.time / 1000000)).format('YYYY-MM-DD HH:mm:ss'), needItem.value || '']);
        }
      });
      data.sort((a, b) => a[0].localeCompare(b[0]));
      if (type === 'gcsRayLet') {
        data = data.map(([timestamp, value]) => [timestamp, value === '' ? 0 : value]);
      }
      const validValues = data
        .map((item) => item[1])
        .filter((value) => value !== null && value !== undefined && value !== '' && !isNaN(value));
      if (validValues.length) {
        max = Math.max(...validValues);
        min = Math.min(...validValues);
      }
      echartsSeries.push({
        name: aimItem,
        type: 'line',
        stack: 'Total',
        data,
        emphasis: {
          focus: 'series', // 悬停时高亮整个系列
        },
        itemStyle: {
          color: colorMap.get(aimItem),
        },
        lineStyle: {
          type: 'solid',
        },
        showSymbol: false,
        connectNulls: true,
      });
    });
  }
  xData.sort((a, b) => a - b);
  let legend = [];
  colorMap.forEach((value, key) => {
    legend.push({ color: value, name: key });
  });
  const yName = yNameMap[type];
  return {
    xData,
    series: echartsSeries,
    legend,
    config: {
      max,
      min,
      name: yName,
      nameLocation: 'center',
      nameRoate: '90',
      nameGap: 70,
      nameTextStyle: { padding: [500, 0, 0, 50] },
    },
  };
}

export const birthLokiDayTime = (step, start, end) => {
  let xAxis = [];
  let count = parseInt(step);
  let x = start;
  while (x < end) {
    xAxis.push((x + count) * 1000);
    x += count;
  }
  return xAxis;
};

export function forbiddenMsg(message, errorMsg) {
  if (errorMsg.response.data.message.includes('User') || errorMsg.response.data.message.includes('user')) {
    return message.error('操作失败，当前用户没有操作权限，请联系管理员添加权限!');
  } else {
    return message.error('请求被服务端拒绝，建议修改该请求!');
  }
}
