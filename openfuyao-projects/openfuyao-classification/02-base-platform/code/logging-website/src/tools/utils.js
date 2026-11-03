/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */

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
 * 处理标签注解数据
 */
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

/**
 * 导出日志 txt
 * name 导出名称
 * data 数据
 */
export function exportHelmLogOutPut(name, data) {
  const link = document.createElement('a');
  link.setAttribute('href', `data:text/txt;charset=utf-8,${encodeURIComponent(data)}`);
  link.setAttribute('download', `${name}.log`);
  link.style.display = 'none';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

export function sorterFirstAlphabet(a = '--', b = '--') {
  if (typeof a === 'number' && typeof b === 'number') {
    return a - b;
  }
  return a.toString().localeCompare(b.toString());
}

export function getLogcontextBk(index, num, type) {
  let color = '';
  if (index === num) {
    color = type === 'light' ? '#cfe7ffff' : '#2a2d34ff';
  }

  return color;
}
