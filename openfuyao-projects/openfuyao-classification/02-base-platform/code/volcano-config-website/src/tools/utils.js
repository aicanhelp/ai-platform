/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */

import jsyaml from 'js-yaml';
import { containerRouterPrefix } from '@/common/constants';
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

/**
 * antd表格按名字首字母顺序排序
 */
export function sorterFirstAlphabe(a, b) {
  return a.slice(0, 1).charCodeAt(0) - b.slice(0, 1).charCodeAt(0);
}

/**
 * antd表格筛选项去重过滤
 */
export function filterRepeat(arr) {
  const newArr = arr.filter((item, index, self) => {
    return self.findIndex(obj => obj.value === item.value) === index;
  });
  return newArr;
}

export function yamlTojson(text) {
  return JSON.parse(JSON.stringify(jsyaml.load(text)));
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

/**
 * 处理key
 */
export function solvePathKeyEscape(key) {
  return key.replace('/', '~1');
}

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

/**
 * antd表格按名字首字母顺序排序(全排序)
 */
export function sorterFirstAlphabet(a, b) {
  if (typeof (a) === 'number' && typeof (b) === 'number') {
    return a - b;
  }
  return (a.toString()).localeCompare(b.toString());
}
// 自定义用户名校验函数
export async function validateUsername(_rule, value) {
  const regex = /^[a-zA-Z\u4e00-\u9fa5]{1,255}$/; // 只能包含中文、英文，且长度在1到255之间
  if (value && !regex.test(value)) {
    return Promise.reject(new Error('用户名格式不正确，请输入中文或英文，且长度在1到255之间'));
  } else {
    return Promise.resolve();
  }
};

export function sortNaN(a, b, sorter) {
  if (isNaN(a) && isNaN(b)) {
    return 0;
  }
  if (isNaN(a)) {
    return sorter === 'ascend' ? 1 : -1;
  }
  if (isNaN(b)) {
    return sorter === 'ascend' ? -1 : 1;
  }

  return a - b;
};


export function percenteHeight(numaNodeMemoryUsed, numaNodeMemoryTotal) {
  const availableMemoryPercentage = (numaNodeMemoryUsed) / (numaNodeMemoryTotal);
  return `${availableMemoryPercentage * 139}px`;
}

export function skipSendMainMsg(event, key, url, history, isAddLocalPrefix = true) {
  event ? event.preventDefault() : '';
  window.parent.postMessage({ type: 'subToMainMsg', data: key, url: isAddLocalPrefix ? `/${containerRouterPrefix}${url}` : url, isInsertMainPrefix: true });
  window.__OPENFUYAO__ ? '' : history.push(url);
}