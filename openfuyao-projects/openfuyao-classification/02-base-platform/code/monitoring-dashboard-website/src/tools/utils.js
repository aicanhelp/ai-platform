/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */

import dayjs from 'dayjs';
import Big from 'big.js';

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
 * 聚合chart加自定义字段
 */
export function getMappingValueArrayOfKeyChart(array, keyName) {
  if (Object.prototype.toString.call(array) === '[object Array]') {
    return array.map(item => {
      return { value: item[keyName], aimList: item.aimList };
    });
  }
  return [];
}

/**
 * HTML 转义，防止 XSS
 */
export function escapeHtml(str) {
  if (typeof str !== 'string') return str;
  const map = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#x27;',
  };
  return str.replace(/[&<>"']/g, m => map[m]);
}

/**
 * 展示字段图例
 */
export function displayTooltip(tooltip) {
  let str = '';
  const keys = Object.keys(tooltip);
  keys.map((item, index) => {
    if (item !== '__name__') {
      str += `<b>${escapeHtml(item)}</b>: ${escapeHtml(String(tooltip[item]))}<br/>`;
    }
  });
  return str;
}

/**
 * 处理async/await错误捕获
 */
export async function requestResource(fn) {
  try {
    let response = await fn;
    return [null, response];
  } catch (err) {
    return [err, null];
  }
}

/**
 * 生成固定8色
 */
export function fixedColor() {
  let colorList = ['#2CB8C9', '#77AEF7', '#F4840C', '#E049CE', '#87C859', '#09AA71', '#715AFB', '#FCD72E'];
  return colorList;
}

/**
 * 元素去重结合
 */
export function repeatFilter(typeList, dataList) {
  let newData = [];
  typeList.map(item => {
    const signalList = dataList.filter(filterItem => filterItem.type === item);
    const sumWithInitialValue = signalList[signalList.length - 1].value;
    const total = signalList.reduce((accumulator, currentValue) => accumulator + currentValue?.value ?? 0, 0);
    newData.push({ name: item, value: sumWithInitialValue, type: item, aimList: signalList[0].aimList });
  })
  return newData;
}

/**
 * 数据清洗
 */
export function solveWidgetData(indicatorList) {
  let data = [];
  let num = 0;
  if (indicatorList) {
    indicatorList.map(item => {
      let type = item.metric.__name__;
      const isExistList = data.filter(filterItem => filterItem.displayName === type);
      if (isExistList.length) {
        num += 1;
        type = type + num;
      }
      if (item.value && item.value.length) {
        item.value.map(([timeStamp = '--', value = '--']) => {
          let obj = {};
          obj.displayName = item.metric.__name__;
          obj.type = type;
          obj.timeStamp = timeStamp;
          obj.value = Number(Number(value).toFixed(2));
          obj.aimList = item.metric; // 暂存键值对
          data.push(obj);
        });
      } else {
        data.push({
          displayName: item.metric.__name__,
          type,
          timeStamp: '--',
          value: '--',
          aimList: item.metric,
        });
      }
    });
  }
  return data;
}

/**
 * 处理创建数据
 */
export function createWidgetDataChart(data) {
  let finallyData = [];
  let num = 0;
  data.map(item => {
    let type = item.metric.__name__;
    const isExistList = finallyData.filter(filterItem => filterItem.displayName === type);
    if (isExistList.length) {
      num += 1;
      type = type + num;
    }
    if (item.value && item.value.length) {
      item.value.map(([timeStamp, value]) => {
        let obj = {};
        obj.displayName = item.metric.__name__;
        obj.type = type;
        obj.timeStamp = timeStamp;
        obj.value = Number(Number(value).toFixed(2));
        obj.aimList = item.metric; // 暂存键值对
        finallyData.push(obj);
      });
    } else {
      finallyData.push({
        displayName: item.metric.__name__,
        type,
        timeStamp: '--',
        value: '--',
        aimList: item.metric,
      });
    }
  });
  return finallyData;
}

/**
 * 获取instance实例个数
 */
export function getDataRowKeysInstance(data, temporyLength, name) {
  let instanceCount = temporyLength || 0;
  if (name) {
    data.map(item => {
      if (item.aimName !== name) {
        instanceCount += item.selectedRowKeys.length;
      }
    });
  } else {
    data.map(item => {
      instanceCount += item.selectedRowKeys ? item.selectedRowKeys.length : 0;
    });
  }
  return instanceCount;
}

/**
 * 返回选择的指标数
 */
export function getReturnAims(data) {
  let aimCount = 0;
  data.map(item => {
    if (item.selectedRowKeys.length > 0) {
      aimCount++;
    }
  });
  return aimCount;
}

/**
 * 生成随机id
 */
export function randomUuid() {
  return window.crypto.getRandomValues(new Uint16Array(8))[0];
};

// 处理实例
export function solveInstanceData(data, oldId = 0) {
  let tableData = [];
  data.instances.map((item, index) => {
    let str = data.displayName;
    let keyValueList = [];
    const keys = Object.keys(item.metric);
    keys.map(keyItem => {
      if (keyItem !== '__name__') {
        keyValueList.push(`${keyItem} = "${item.metric[keyItem]}"`);
      }
    });
    const newInstanceStr = keyValueList.join(',');
    str += `{${newInstanceStr}}`;
    tableData.push({ instance: str, id: index + oldId, disabled: oldId ? true : false });
  });
  return tableData;
}

// 添加禁用实例
export function solveDisabledData(data, newData) {
  let solveData = [];
  let newSolveData = [];
  data.instances.map(item => {
    const keys = Object.keys(item.metric);
    let str = data.displayName;
    let keyValueList = [];
    keys.map(keyItem => {
      if (keyItem !== '__name__') {
        keyValueList.push(`${keyItem} = "${item.metric[keyItem]}"`);
      }
    });
    const newInstanceStr = keyValueList.join(',');
    str += `{${newInstanceStr}}`;
    solveData.push(str);
  });
  newData.map(item => {
    const keys = Object.keys(item.metric);
    let str = data.displayName;
    let keyValueList = [];
    keys.map(keyItem => {
      if (keyItem !== '__name__') {
        keyValueList.push(`${keyItem} = "${item.metric[keyItem]}"`);
      }
    });
    const newInstanceStr = keyValueList.join(',');
    str += `{${newInstanceStr}}`;
    newSolveData.push(str);
  });
  const map = {};
  solveData.forEach(n => map[n] = true);
  const result = newSolveData.filter(n => !map[n]);
  // 解码
  let finallyInstanceList = [];
  result.map(item => {
    let obj = {};
    obj.__name__ = item.slice(0, item.indexOf('{'));
    let str = item.slice(item.indexOf('{') + 1, item.lastIndexOf('}'));
    str = str.replaceAll('"', '');
    let arrList = str.split(',');
    arrList.map(arrItem => {
      let [key, value] = arrItem.split('=');
      key = key.trim();
      value = value.trim();
      obj[key] = value;
    });
    finallyInstanceList.push({ metric: obj, value: [] });
  });
  return { instances: finallyInstanceList, displayName: data.displayName };
}

// 获取所有键
export function getAllKeys(data) {
  let labelList = [];
  let keyList = [];
  data.instances.map(item => {
    const keys = Object.keys(item.metric);
    keyList = keyList.concat(keys);
  });
  keyList = [...new Set(keyList)];
  keyList.map(item => {
    if (item !== '__name__') {
      labelList.push({
        label: item,
        value: item,
      });
    }
  });
  return labelList;
}

// 键值对映射
export function getKeyMap(data) {
  let keyList = [];
  data.instances.map(item => {
    const keys = Object.keys(item.metric);
    keys.map(keyItem => {
      if (keyItem !== '__name__') {
        keyList.push({
          key: keyItem,
          value: item.metric[keyItem],
        });
      }
    });
  });
  return keyList;
}

// 获取对应建对应值列表
export function getMappingValue(key, data) {
  let result = [];
  let valueList = [];
  data.map(item => {
    if (item.key === key) {
      valueList.push(item.value);
    }
  });
  valueList = [...new Set(valueList)];
  valueList.map(item => {
    result.push({
      label: item,
      value: item,
    });
  });
  return result;
}

// 数据过滤
export function cleanFilterTableData(passValue, data, type = 'key') {
  let finallyData = [];
  // 反序列化
  data.map(({ instance }, index) => {
    let str = instance.slice(instance.indexOf('{') + 1, instance.lastIndexOf('}'));
    str = str.replaceAll('"', '');
    let arrList = str.split(',');
    arrList.map(arrItem => {
      let [key, value] = arrItem.split('=');
      if (type === 'key') {
        if (key.trim() === passValue) {
          finallyData.push(data[index]);
        }
      } else {
        if (value.trim() === passValue) {
          finallyData.push(data[index]);
        }
      }
    });
  });
  return finallyData;
}

// 过滤点击删除后数据检索
export function filterTableData(mapList, data) {
  let finallyData = [];
  let testData = {};
  if (!mapList.length) {
    finallyData = finallyData.concat(data);
  } else {
    data.map(({ instance }, index) => {
      let str = instance.slice(instance.indexOf('{') + 1, instance.lastIndexOf('}'));
      str = str.replaceAll('"', '');
      let arrList = str.split(',');
      mapList.map((mapItem, mapIndex) => {
        const { key, value } = mapItem;
        const original = arrList.filter(filterItem => {
          let [oldKey, oldValue] = filterItem.split('=');
          return value ? (oldKey.trim() === key && oldValue.trim() === value) : (oldKey.trim() === key);
        });
        if (original.length) {
          if (!testData[mapIndex]) {
            testData[mapIndex] = [];
          }
          testData[mapIndex].push(data[index]);
        }
      });
    });
    // 数组项聚合
    const keys = Object.keys(testData);
    let newArrList = [];
    keys.map(item => {
      if (!newArrList.length) {
        newArrList = newArrList.concat(testData[item]);
      }
      let resultArr = [...newArrList].filter(x => [...testData[item]].some(y => y.instance === x.instance));
      newArrList = [...resultArr];
    });
    finallyData = [...newArrList];
  }
  return finallyData;
}

// 将表格数据转为instance
export function solveTableDataToInstance(data) {
  let finallyInstanceList = [];
  let instanceList = [];
  data.map(item => {
    let newList = [];
    item.selectInstances.map(instanceItem => {
      newList.push(instanceItem.instance);
    });
    instanceList = instanceList.concat(newList);
  });
  instanceList.map(item => {
    let obj = {};
    obj.__name__ = item.slice(0, item.indexOf('{'));
    let str = item.slice(item.indexOf('{') + 1, item.lastIndexOf('}'));
    str = str.replaceAll('"', '');
    let arrList = str.split(',');
    arrList.map(arrItem => {
      let [key, value] = arrItem.split('=');
      if (key && value) {
        key = key.trim();
        value = value.trim();
        obj[key] = value;
      }
    });
    finallyInstanceList.push(JSON.stringify(obj));
  });
  return finallyInstanceList;
}

export function birthDefaultTimestamp(num = 1) {
  let defaultData = [];
  for (let i = 0; i < num; i++) {
    defaultData.push({
      timeStamp: dayjs() / 1000,
    });
  }
  return defaultData;
};

export function gainAimType(instanceList) {
  let finallyList = [];
  instanceList.map(item => {
    finallyList.push(item.metric.__name__);
  });
  return [...new Set(finallyList)];
}

/**
 * 获取select key值
 */
export function solveSelectedRowKeys(instanceList, data) {
  const newInstanceList = [];
  const newDataList = [];
  const keys = [];
  instanceList.map(instanceItem => {
    newInstanceList.push(JSON.stringify(instanceItem.metric));
  });
  data.instances.map(item => {
    newDataList.push(JSON.stringify(item.metric));
  });
  newInstanceList.map(item => {
    keys.push(newDataList.indexOf(item));
  });
  return keys;
}

/**
 * 处理echarts仪表盘数据
 */
export function solveEchartsDashboardLength(value) {
  let solveValue = String(value);
  let le = solveValue.length;
  let tip1 = '';
  let tip = '';
  if (false) {
    let l = Math.ceil(le / 5);
    for (let i = 1; i <= l; i++) {
      if (i < l) {
        tip1 += `${solveValue.slice((i * 5) - 5, i * 5)}\n`;
      } else {
        tip = tip1 + solveValue.slice((l - 1) * 5, le);
      }
    }
    return `${tip}`;
  } else {
    tip = solveValue;
    return `${tip}`;
  }
}

// 处理dashbaord值
export function solveDashboardValue(value) {
  if (String(value).length > 5) {
    let remainValue = value;
    let count = 0;
    let finallyStr = '';
    do {
      remainValue = new Big(remainValue / 10);
      count++;
    }
    while (Math.abs(remainValue) >= 10);
    finallyStr = `${Math.round(remainValue)}e+${count}`;
    return finallyStr;
  } else {
    return value;
  }
}

export function sortNormalTime(a, b) {
  return dayjs(a.lastUpdateTime) - dayjs(b.lastUpdateTime);
}

export function sortReserveTime(a, b) {
  return dayjs(b.lastUpdateTime) - dayjs(a.lastUpdateTime);
}

export function sorterFirstAlphabet(a = '--', b = '--') {
  if (typeof (a) === 'number' && typeof (b) === 'number') {
    return a - b;
  }
  return (a.toString()).localeCompare(b.toString());
}

// 排序时间
export function sortData(data, obj) {
  let finallyData = [];
  if (obj.columnKey === 'lastUpdateTime') {
    if (!obj.order) {
      finallyData = data;
    }
    if (obj.order === 'ascend') {
      finallyData = data.sort(sortNormalTime);
    }
    if (obj.order === 'descend') {
      finallyData = data.sort(sortReserveTime);
    }
  } else {
    if (!obj.columnKey || !obj.order) {
      finallyData = data;
    } else {
      let mapObj = {
        name: 'displayName',
        monitorComponentNum: 'totalWidgets',
        creator: 'creator',
      };
      let key = mapObj[obj.columnKey];
      if (obj.order === 'ascend') {
        finallyData = data.sort((a, b) => sorterFirstAlphabet(a[key], b[key]));
      }
      if (obj.order === 'descend') {
        finallyData = data.sort((a, b) => sorterFirstAlphabet(b[key], a[key]));
      }
    }
  }
  return finallyData;
}

/**
 * 过滤项目
 */
export function filterData(data, value) {
  let finallyData = [];
  finallyData = data.filter(item => item.creator === value);
  return finallyData;
}