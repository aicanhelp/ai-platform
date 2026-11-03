/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import pluralize from 'pluralize';
import dayjs from 'dayjs';
import { containerRouterPrefix } from '@/constant.js';
// 去重过滤value
function filterRepeat(arr) {
  const newArr = arr.filter((item, index, self) => {
    return self.findIndex(obj => obj.value === item.value) === index;
  });
  return newArr;
}

// 去重过滤标签label
function filterLabelRepeat(arr) {
  const newArr = arr.filter((item, index, self) => {
    return self.findIndex(obj => obj.label === item.label) === index;
  });
  return newArr;
}

function filterRepeatByid(arrObj) {
  let newArr = [];
  let obj = {};
  for (let i = 0; i < arrObj.length; i++) {
    if (!obj[arrObj[i].id]) {
      obj[arrObj[i].id] = true;
      newArr.push(arrObj[i]);
    }
  }
  return newArr;
}
function filterColor(data) {
  let color = '';
  switch (data) {
    case '严重':
      color = 'red';
      break;
    case '提示':
      color = 'blue';
      break;
    case '一般':
      color = 'yellow';
      break;
    default: { }
  }
  return color;
}

function filterHelmType(data) {
  let type = '';
  switch (data) {
    case 'Turbo':
      type = 'turbo';
      break;
    case 'export':
      type = 'export';
      break;
    default: { }
  }
  return type;
}

function filterManageState(data) {
  let state = '';
  switch (data) {
    case 'deployed':
      state = '部署成功';
      break;
    case 'failed':
      state = '部署失败';
      break;
    case 'unknown':
      state = '未知';
      break;
    case 'uninstalling':
      state = '卸载中';
      break;
    case 'pending-install':
      state = '处理中';
      break;
    case 'pending-upgrade':
      state = '处理中';
      break;
    case 'pending-rollback':
      state = '处理中';
      break;
    default: {
      state = '未知';
    }
  }
  return state;
}

function filterManageStateBack(data) {
  let state = '';
  switch (data) {
    case '部署成功':
      state = 'deployed';
      break;
    case '部署失败':
      state = 'failed';
      break;
    case '未知':
      state = 'unknown';
      break;
    case '处理中':
      state = 'pending';
      break;
    case '卸载中':
      state = 'uninstalling';
      break;
    default: { }
  }
  return state;
}

function filterAlertStatus(data) {
  let type = '';
  switch (data) {
    case 'warning':
      type = '警告';
      break;
    case 'info':
      type = '提示';
      break;
    case 'critical':
      type = '严重';
      break;
    default: { }
  }
  return type;
}

function filterAlertStatusStyle(data) {
  let style = {};
  switch (data) {
    case 'warning':
      style = {
        borderRadius: '16px',
        background: '#f4840c1a',
        border: '1px solid #f4840c',
        fontSize: '14px',
        padding: '5px 14px',
        color: '#f4840cff',
        width: 'max-content',
      };
      break;
    case 'info':
      style = {
        borderRadius: '16px',
        background: '#77aef71a',
        border: '1px solid #4b8cea',
        fontSize: '14px',
        padding: '5px 14px',
        color: '#4B8BEAff',
        width: 'max-content',
      };
      break;
    case 'critical':
      style = {
        borderRadius: '16px',
        background: '#e7434a1a',
        border: '1px solid #e7434a',
        fontSize: '14px',
        padding: '5px 14px',
        color: '#e7434aff',
        width: 'max-content',
      };
      break;
    default: { }
  }
  return style;
}

function filterResourceJumpStyle(data) {
  let style = {
    color: '#333333',
    cursor: 'default',
  };
  switch (data) {
    case 'Pod':
      style = {
        color: '#3f66f5',
        cursor: 'pointer',
      };
      break;
    case 'Deployment':
      style = {
        color: '#3f66f5',
        cursor: 'pointer',
      };
      break;
    case 'StatefulSet':
      style = {
        color: '#3f66f5',
        cursor: 'pointer',
      };
      break;
    case 'DaemonSet':
      style = {
        color: '#3f66f5',
        cursor: 'pointer',
      };
      break;
    case 'Job':
      style = {
        color: '#3f66f5',
        cursor: 'pointer',
      };
      break;
    case 'CronJob':
      style = {
        color: '#3f66f5',
        cursor: 'pointer',
      };
      break;
    case 'Service':
      style = {
        color: '#3f66f5',
        cursor: 'pointer',
      };
      break;
    case 'Ingress':
      style = {
        color: '#3f66f5',
        cursor: 'pointer',
      };
      break;
    case 'Secret':
      style = {
        color: '#3f66f5',
        cursor: 'pointer',
      };
      break;
    case 'ConfigMap':
      style = {
        color: '#3f66f5',
        cursor: 'pointer',
      };
      break;
    case 'ServiceAccount':
      style = {
        color: '#3f66f5',
        cursor: 'pointer',
      };
      break;
    case 'Role':
      style = {
        color: '#3f66f5',
        cursor: 'pointer',
      };
      break;
    case 'RoleBinding':
      style = {
        color: '#3f66f5',
        cursor: 'pointer',
      };
      break;
    case 'Namespace':
      style = {
        color: '#3f66f5',
        cursor: 'pointer',
      };
      break;
    default: { }
  }
  return style;
}

function changeCrName(type, apiVersion) {
  let str = `${pluralize(type.toLowerCase())}_${apiVersion.split('/')[0].split('.').join('_')}`;
  return str;
}

function filterAlertTime(data) {
  let day = 0;
  let hour = 0;
  let minute = 0;
  let seconds = 0;
  let string = '';
  day = parseInt(data / 86400);
  hour = parseInt((data % 86400) / 3600);
  minute = parseInt(((data % 86400) % 3600) / 60);
  seconds = parseInt(((data % 86400) % 3600) % 60);
  string = (day === 0 ? '' : `${day}天`) + (hour === 0 ? '' : `${hour}小时`) + (minute === 0 ? '' : `${minute}分钟`);
  return string;
}

function filterOverViewAreaEcharts(data1, data2) {
  let _data = [];
  for (let i = 0; i < data1.length; i++) {
    let _singleArr = [];
    let _singObj = { name: '', time: [], data: [] };
    _singleArr = data2.filter(item => item.type === data1[i]);
    _singleArr.forEach(item2 => {
      _singObj.name = item2.type;
      _singObj.time.push(dayjs(item2.timestamp * 1000).format('MM/DD HH:mm'));
      _singObj.data.push(item2.value);
    });
    _data.push(_singObj);
  }
  return _data;
}

function filterContinueTime(data) {
  let time = '';
  switch (data) {
    case '1d':
      time = [0, 86400];
      break;
    case '1-5d':
      time = [86400, 432000];
      break;
    case '5-10d':
      time = [432000, 864000];
      break;
    case '10d':
      time = [864000];
      break;
    default: { }
  }
  return time;
}

function filterOverviewRouter(type, data) {
  let obj = {
    url: '',
    status: '',
  };
  switch (type) {
    case 'node':
      obj = {
        url: `/${containerRouterPrefix}/nodeManage`,
        status: data.seriesName,
      };
      break;
    case 'pod':
      obj = {
        url: `/${containerRouterPrefix}/workload/pod`,
        status: data.seriesName,
      };
      break;
    case 'application':
      obj = {
        url: `/${containerRouterPrefix}/applicationManageHelm`,
        status: data.name,
      };
      break;
    case 'extend':
      obj = {
        url: `/${containerRouterPrefix}/extendManage`,
        status: data.name,
      };
      break;
    case 'Deployment':
      obj = {
        url: `/${containerRouterPrefix}/workload/deployment`,
        status: data.seriesName,
      };
      break;
    case 'StatefulSet':
      obj = {
        url: `/${containerRouterPrefix}/workload/statefulSet`,
        status: data.seriesName,
      };
      break;
    case 'DaemonSet':
      obj = {
        url: `/${containerRouterPrefix}/workload/daemonSet`,
        status: data.seriesName,
      };
      break;
    default: { }
  }
  return obj;
}

function filterOverviewType(type, params) {
  let data = '';
  data = type === 'workload' ? params.name : type;
  return data;
}

function getOverviewChartList(arr) {
  let runningCount = 0;
  let pendingCount = 0;
  let FailedCount = 0;
  let SucceededCount = 0;
  let UnknownCount = 0;
  let chartArr = [];
  let legArr = [];
  arr.forEach(i => {
    if (i.status.phase === 'Running') {
      runningCount++;
      chartArr.push({ item: 'Running', count: runningCount, color: '#09aa71' });
    } else if (i.status.phase === 'Pending') {
      pendingCount++;
      chartArr.push({ item: 'Pending', count: pendingCount, color: '#fcd72e' });
    } else if (i.status.phase === 'Failed') {
      FailedCount++;
      chartArr.push({ item: 'Failed', count: FailedCount, color: '#e7434a' });
    } else if (i.status.phase === 'Succeeded') {
      SucceededCount++;
      chartArr.push({ item: 'Succeeded', count: SucceededCount, color: '#77aef7' });
    } else {
      UnknownCount++;
      chartArr.push({ item: 'Unknown', count: UnknownCount, color: '#cccccc' });
    }
    legArr.push(i.status.phase);
  });
  return chartArr;
};

function sortResourceByTime(a, b) {
  let firstTime = a.creationTimestamp ? a.creationTimestamp : '';
  let endTime = b.creationTimestamp ? b.creationTimestamp : '';
  return dayjs(firstTime) - dayjs(endTime);
}

function getTimeType(data) {
  let typeData = {
    type: '',
    num: 0,
  };
  if (data >= 86400 && (data % 86400 === 0)) {
    typeData = {
      type: 'd',
      num: Math.floor(data / 86400),
    };
  } else if (data >= 3600 && (data % 3600 === 0)) {
    typeData = {
      type: 'h',
      num: Math.floor(data / 3600),
    };
  } else {
    typeData = {
      type: 'm',
      num: Math.floor(data / 60),
    };
  }

  return typeData;
}

export {
  filterRepeat,
  filterLabelRepeat,
  filterColor,
  filterHelmType,
  filterManageState,
  filterRepeatByid,
  filterManageStateBack,
  filterAlertStatus,
  filterAlertStatusStyle,
  filterResourceJumpStyle,
  changeCrName,
  filterAlertTime,
  filterOverViewAreaEcharts,
  filterContinueTime,
  filterOverviewRouter,
  filterOverviewType,
  getOverviewChartList,
  sortResourceByTime,
  getTimeType,
};