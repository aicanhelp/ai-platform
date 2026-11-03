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
import {
  CheckCircleFilled,
  CloseCircleFilled,
  ExclamationCircleFilled,
  QuestionCircleFilled,
  LoadingOutlined,
} from '@ant-design/icons';
import { containerRouterPrefix } from '@/constant.js';
import i18n from '@/i18n';
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
  // Now accepts backend keys instead of Chinese strings
  switch (data) {
    case 'critical':
      return 'red';
    case 'info':
      return 'blue';
    case 'warning':
      return 'yellow';
    default: {
      return '';
    }
  }
}

function filterHelmType(data) {
  switch (data) {
    case 'Turbo':
      return 'turbo';
    case 'export':
      return 'export';
    default: {
      return '';
    }
  }
}

/** Helm release phase string from API → stable key for i18n (status.*) */
function getHelmManageStateKey(helmStatus) {
  if (!helmStatus) { return 'unknown'; }
  switch (helmStatus) {
    case 'deployed':
      return 'deployed';
    case 'failed':
      return 'failed';
    case 'unknown':
      return 'unknown';
    case 'uninstalling':
      return 'uninstalling';
    case 'pending-install':
    case 'pending-upgrade':
    case 'pending-rollback':
      return 'pending';
    default:
      return 'unknown';
  }
}

/** Status icon for helm/extension release states; prefix: 'helm_state_' or 'extend_state_' */
function getHelmManageStateIcon(statusKey, prefix = 'helm_state_') {
  const iconStyle = { marginRight: '8px' };
  switch (statusKey) {
    case 'deployed':
      return <CheckCircleFilled className={`${prefix}successful`} style={iconStyle} />;
    case 'failed':
      return <CloseCircleFilled className={`${prefix}failed`} style={iconStyle} />;
    case 'uninstalling':
      return <ExclamationCircleFilled className={`${prefix}unknown`} style={iconStyle} />;
    case 'pending':
      return <LoadingOutlined className={`${prefix}pending`} style={iconStyle} />;
    case 'unknown':
    default:
      return <QuestionCircleFilled className={`${prefix}unknown`} style={iconStyle} />;
  }
}

function filterAlertStatus(data) {
  const key = typeof data === 'string' ? data.toLowerCase() : '';
  const navMap = {
    warning: i18n.t('alert.warning', { ns: 'nav' }),
    info: i18n.t('alert.info', { ns: 'nav' }),
    critical: i18n.t('alert.critical', { ns: 'nav' }),
  };
  if (key && navMap[key]) { return navMap[key]; }
  const alertLabel = key ? i18n.t(`level.${key}`, { ns: 'alert' }) : '';
  if (alertLabel && alertLabel !== `level.${key}`) { return alertLabel; }
  return data != null && data !== '' ? String(data) : '';
}

function filterAlertStatusStyle(data) {
  const normalized = typeof data === 'string' ? data.toLowerCase() : data;
  switch (normalized) {
    case 'warning':
      return {
        borderRadius: '16px',
        background: '#f4840c1a',
        border: '1px solid #f4840c',
        fontSize: '14px',
        padding: '5px 14px',
        color: '#f4840cff',
        width: 'max-content',
      };
    case 'info':
      return {
        borderRadius: '16px',
        background: '#77aef71a',
        border: '1px solid #4b8cea',
        fontSize: '14px',
        padding: '5px 14px',
        color: '#4b8beaff',
        width: 'max-content',
      };
    case 'critical':
      return {
        borderRadius: '16px',
        background: '#e7434a1a',
        border: '1px solid #e7434a',
        fontSize: '14px',
        padding: '5px 14px',
        color: '#e7434aff',
        width: 'max-content',
      };
    default: {
      return {};
    }
  }
}

const JUMPABLE_RESOURCE_KINDS = new Set(['Pod', 'Deployment', 'StatefulSet', 'DaemonSet', 'Job', 'CronJob', 'Service', 'Ingress', 'Secret', 'ConfigMap', 'ServiceAccount', 'Role', 'RoleBinding', 'Namespace']);

function filterResourceJumpStyle(data) {
  if (JUMPABLE_RESOURCE_KINDS.has(data)) {
    return {
      color: '#3f66f5',
      cursor: 'pointer',
    };
  }
  return {
    color: '#333333',
    cursor: 'default',
  };
}

function changeCrName(type, apiVersion) {
  let str = `${pluralize(type.toLowerCase())}_${apiVersion.split('/')[0].split('.').join('_')}`;
  return str;
}

function filterAlertTime(data) {
  const day = parseInt(data / 86400, 10);
  const hour = parseInt((data % 86400) / 3600, 10);
  const minute = parseInt(((data % 86400) % 3600) / 60, 10);
  return (day === 0 ? '' : `${day}${i18n.t('timeUnit.day', { ns: 'alert' })}`) +
    (hour === 0 ? '' : `${hour}${i18n.t('timeUnit.hour', { ns: 'alert' })}`) +
    (minute === 0 ? '' : `${minute}${i18n.t('timeUnit.minute', { ns: 'alert' })}`);
}

function filterOverViewAreaEcharts(data1, data2) {
  let _data = [];
  for (let i = 0; i < data1.length; i++) {
    let _singleArr;
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
  switch (data) {
    case '1d':
      return [0, 86400];
    case '1-5d':
      return [86400, 432000];
    case '5-10d':
      return [432000, 864000];
    case '10d':
      return [864000];
    default: {
      return '';
    }
  }
}

function filterOverviewRouter(type, data) {
  switch (type) {
    case 'node':
      return {
        url: `/${containerRouterPrefix}/nodeManage`,
        status: data.seriesName,
      };
    case 'pod':
      return {
        url: `/${containerRouterPrefix}/workload/pod`,
        status: data.seriesName,
      };
    case 'application':
      return {
        url: `/${containerRouterPrefix}/applicationManageHelm`,
        status: data.name,
      };
    case 'extend':
      return {
        url: `/${containerRouterPrefix}/extendManage`,
        status: data.name,
      };
    case 'Deployment':
      return {
        url: `/${containerRouterPrefix}/workload/deployment`,
        status: data.seriesName,
      };
    case 'StatefulSet':
      return {
        url: `/${containerRouterPrefix}/workload/statefulSet`,
        status: data.seriesName,
      };
    case 'DaemonSet':
      return {
        url: `/${containerRouterPrefix}/workload/daemonSet`,
        status: data.seriesName,
      };
    default: {
      return {
        url: '',
        status: '',
      };
    }
  }
}

function filterOverviewType(type, params) {
  return type === 'workload' ? params.name : type;
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
}

function sortResourceByTime(a, b) {
  let firstTime = a.creationTimestamp ? a.creationTimestamp : '';
  let endTime = b.creationTimestamp ? b.creationTimestamp : '';
  return dayjs(firstTime) - dayjs(endTime);
}

function getTimeType(data) {
  if (data >= 86400 && (data % 86400 === 0)) {
    return {
      type: 'd',
      num: Math.floor(data / 86400),
    };
  } else if (data >= 3600 && (data % 3600 === 0)) {
    return {
      type: 'h',
      num: Math.floor(data / 3600),
    };
  } else {
    return {
      type: 'm',
      num: Math.floor(data / 60),
    };
  }
}

/**
 * Get translated label for time period options
 * @param {string} key - stable key like '10m', '30m', '1h', etc.
 * @returns {string} translated label
 */
function getTimePeriodLabel(key) {
  const labels = {
    '10m': i18n.t('timePeriod.10m', { ns: 'common' }),
    '30m': i18n.t('timePeriod.30m', { ns: 'common' }),
    '1h': i18n.t('timePeriod.1h', { ns: 'common' }),
    '3h': i18n.t('timePeriod.3h', { ns: 'common' }),
    '6h': i18n.t('timePeriod.6h', { ns: 'common' }),
    '1d': i18n.t('timePeriod.1d', { ns: 'common' }),
    '3d': i18n.t('timePeriod.3d', { ns: 'common' }),
    '7d': i18n.t('timePeriod.7d', { ns: 'common' }),
    '14d': i18n.t('timePeriod.14d', { ns: 'common' }),
  };
  return labels[key] || key;
}

/**
 * Get translated label for event level
 * @param {string} key - stable key like '', 'Warning', 'Normal'
 * @returns {string} translated label
 */
function getEventLevelLabel(key) {
  const labels = {
    all: i18n.t('eventLevel.all', { ns: 'common' }),
    warning: i18n.t('eventLevel.warning', { ns: 'common' }),
    normal: i18n.t('eventLevel.normal', { ns: 'common' }),
  };
  return labels[key] || key;
}

/**
 * Get translated label for silent time unit
 * @param {string} key - stable key like 'm', 'h', 'd'
 * @returns {string} translated label
 */
function getSilentTimeUnitLabel(key) {
  const labels = {
    m: i18n.t('timeUnit.minute', { ns: 'common' }),
    h: i18n.t('timeUnit.hour', { ns: 'common' }),
    d: i18n.t('timeUnit.day', { ns: 'common' }),
  };
  return labels[key] || key;
}

/**
 * Get translated label for terminal type
 * @param {string} key - stable key like 'cluster', 'container'
 * @returns {string} translated label
 */
function getTerminalTypeLabel(key) {
  const labels = {
    cluster: i18n.t('terminalType.cluster', { ns: 'common' }),
    container: i18n.t('terminalType.container', { ns: 'common' }),
  };
  return labels[key] || key;
}

/**
 * Get translated label for node status
 * @param {string} key - stable key like 'normal', 'failed'
 * @returns {string} translated label
 */
function getNodeStatusLabel(key) {
  const labels = {
    normal: i18n.t('nodeStatus.normal', { ns: 'common' }),
    failed: i18n.t('nodeStatus.failed', { ns: 'common' }),
  };
  return labels[key] || key;
}

/**
 * Get translated label for node type
 * @param {string} key - stable key like 'workerNode', 'manageNode'
 * @returns {string} translated label
 */
function getNodeTypeLabel(key) {
  const labels = {
    workerNode: i18n.t('nodeType.workerNode', { ns: 'common' }),
    manageNode: i18n.t('nodeType.manageNode', { ns: 'common' }),
  };
  return labels[key] || key;
}

/**
 * Transform constants with stable keys to localized options array
 * @param {Array} options - options array with stable key labels
 * @param {Function} labelFn - function to get localized label
 * @returns {Array} options array with localized labels
 */
function localizeOptions(options, labelFn) {
  if (!labelFn) { return options; }
  return options.map(opt => ({
    ...opt,
    label: labelFn(opt.label),
  }));
}

export {
  filterRepeat,
  filterLabelRepeat,
  filterColor,
  filterHelmType,
  getHelmManageStateKey,
  getHelmManageStateIcon,
  filterRepeatByid,
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
  getTimePeriodLabel,
  getEventLevelLabel,
  getSilentTimeUnitLabel,
  getTerminalTypeLabel,
  getNodeStatusLabel,
  getNodeTypeLabel,
  localizeOptions,
};