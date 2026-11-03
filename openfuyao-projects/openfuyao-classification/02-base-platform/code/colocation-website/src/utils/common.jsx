/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */

function filterOverviewType(type, params) {
  let data = '';
  data = type === 'workload' ? params.name : type;
  return data;
}

function filterStatusName(status, intl) {
  if (!intl) return status;

  let name = '';
  switch (status) {
    case 'online': {
      name = intl.formatMessage({ id: 'workload.status.online' });
      break;
    }
    case 'offline': {
      name = intl.formatMessage({ id: 'workload.status.offline' });
      break;
    }
    case 'oversold': {
      name = intl.formatMessage({ id: 'workload.status.oversold' });
      break;
    }
    case 'active': {
      name = intl.formatMessage({ id: 'node.status.normal' });
      break;
    }
    case 'inactive': {
      name = intl.formatMessage({ id: 'node.status.abnormal' });
      break;
    }
    default: {
      name = '';
      break;
    }
  }
  return name;
}

function reseverStatusName(name, intl) {
  if (!intl) return name;

  let reseverName = '';
  const onlineText = intl.formatMessage({ id: 'workload.status.online' });
  const offlineText = intl.formatMessage({ id: 'workload.status.offline' });
  const oversoldText = intl.formatMessage({ id: 'workload.status.oversold' });

  switch (name) {
    case onlineText: {
      reseverName = 'online';
      break;
    }
    case offlineText: {
      reseverName = 'offline';
      break;
    }
    case oversoldText: {
      reseverName = 'oversold';
      break;
    }
    default: {
      reseverName = '';
      break;
    }
  }
  return reseverName;
}

function overviewUsedStatus(name, intl) {
  if (!intl) return name;

  let translaterName = '';
  switch (name) {
    case 'normal': {
      translaterName = intl.formatMessage({ id: 'node.status.normal' });
      break;
    }
    case 'oversold': {
      translaterName = intl.formatMessage({ id: 'workload.status.oversold' });
      break;
    }
    case 'colocation': {
      translaterName = intl.formatMessage({ id: 'workload.colocation' });
      break;
    }
    default: {
      translaterName = '';
      break;
    }
  }
  return translaterName;
}

function compareObjKeys(data, searchName) {
  let arr = [];
  let result = false;
  if (data.labels) {
    Object.keys(data.labels).forEach((key) => {
      arr.push(key);
    });
    arr.forEach((i) => {
      if (i.replace(/\s+/g, '') === searchName.split(':')[0].replace(/\s+/g, '')) {
        if (data.labels[i].replace(/\s+/g, '') === searchName.split(':')[1].replace(/\s+/g, '')) {
          result = true;
        }
      }
    });
  }
  return result;
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
    default: {
    }
  }
  return obj;
}

// echarts配置项公共部分
function filterEchartsOption(series, leftName, theme) {
  let option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow',
      },
    },
    legend: {
      right: '10%',
      top: '10%',
      borderRadius: '50%',
      borderWidth: 0,
      padding: 5,
      itemGap: 10,
      itemWidth: 8,
      itemHeight: 8,
      icon: 'circle',
      formatter: (name) => {
        let _name = '';
        series.forEach((item) => {
          if (item.name === name) {
            _name = item.name;
          }
        });
        return _name;
      },
      textStyle: {
        color: theme === 'light' ? 'black' : '#f7f7f7',
      },
    },
    grid: {
      left: '4%',
      bottom: '',
      containLabel: true,
    },
    xAxis: {
      type: 'value',
      axisLabel: {
        show: true,
      },
      splitLine: {
        show: true,
        lineStyle: {
          type: 'dashed',
        },
      },
      axisTick: {
        show: false,
      },
      axisLine: {
        show: false,
      },
    },
    yAxis: {
      type: 'category',
      axisLine: {
        show: false,
      },
      axisLabel: {
        textStyle: {
          color: theme === 'light' ? 'black' : '#89939b',
        },
      },
      axisTick: {
        show: false,
      },
      data: leftName,
    },
    series,
  };
  return option;
}

function generateRandomColorHex() {
  const letters = '0123456789ABCDEF';
  let color = '#';
  for (let i = 0; i < 6; i++) {
    color += letters[Math.floor(Math.random() * 16)];
  }
  return color;
}

// 处理概览的资源使用率图表数据
function filterOverViewAreaEcharts(data1, data2) {
  let _data = [];
  for (let i = 0; i < data1.length; i++) {
    let _singleArr = [];
    let _singObj = { name: '', time: [], data: [] };
    _singleArr = data2.filter((item) => item.type === data1[i]);
    _singleArr.forEach((item2) => {
      _singObj.name = item2.type;
      _singObj.time.push(dayjs(item2.timestamp * 1000).format('MM/DD HH:mm'));
      _singObj.data.push(item2.value);
    });
    _data.push(_singObj);
  }
  return _data;
}

export {
  filterOverviewType,
  filterStatusName,
  reseverStatusName,
  compareObjKeys,
  filterOverviewRouter,
  filterEchartsOption,
  generateRandomColorHex,
  overviewUsedStatus,
  filterOverViewAreaEcharts,
};
