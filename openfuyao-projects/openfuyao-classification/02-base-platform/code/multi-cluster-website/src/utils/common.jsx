/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */

import jsyaml from 'js-yaml';

/**
 * antd表格按名字首字母顺序排序
 */
function sorterFirstAlphabe(a = '--', b = '--') {
  if (typeof (a) === 'number' && typeof (b) === 'number') {
    return a - b;
  }
  return (a.toString()).localeCompare(b.toString());
}

/**
 * 处理params
 */
function solveParams(params) {
  for (let key in params) {
    if (params[key] === '') {
      params[key] = null;
    }
  }
  return params;
}

/**
 * antd表格筛选项去重过滤
 */
function filterRepeat(arr) {
  const newArr = arr.filter((item, index, self) => {
    return self.findIndex(obj => obj.value === item.value) === index;
  });
  return newArr;
}

function yamlTojson(text) {
  return JSON.parse(JSON.stringify(jsyaml.load(text)));
}

function getkeysList(data) {
  const arr = Object.keys(data);
  return arr;
}
function jsonToYaml(text) {
  let yaml = '';
  try {
    let jsonObj = JSON.parse(text);
    yaml = jsyaml.dump(jsonObj);
  } catch (e) {
    const str = `json转yaml失败，详细信息为:${e}`;
  }
  return yaml;
}

// 自定义排序函数
function cpuSorter(a, b) {
  if (a.clusterresources.allocatablecpu === b.clusterresources.allocatablecpu) {
    // 当分母相同时，比较分子
    return a.clusterresources.allocatedcpu - b.clusterresources.allocatedcpu;
  }
  // 先按分母排序
  return a.clusterresources.allocatablecpu - b.clusterresources.allocatablecpu;
}

// 自定义排序函数
function memorySorter(a, b) {
  if (a.clusterresources.allocatablemem === b.clusterresources.allocatablemem) {
    // 当分母相同时，比较分子
    return a.clusterresources.allocatedmem.substring(0, a.clusterresources.allocatedmem.length - 2) -
        b.clusterresources.allocatedmem.substring(0, b.clusterresources.allocatedmem.length - 2);
  }
  // 先按分母排序
  return a.clusterresources.allocatablemem.substring(0, a.clusterresources.allocatablemem.length - 2) -
      b.clusterresources.allocatablemem.substring(0, a.clusterresources.allocatablemem.length - 2);
}

export {
  sorterFirstAlphabe,
  solveParams,
  yamlTojson,
  filterRepeat,
  getkeysList,
  cpuSorter,
  memorySorter,
  jsonToYaml,
};