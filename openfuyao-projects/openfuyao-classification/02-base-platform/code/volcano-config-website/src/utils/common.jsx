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

function filterStatusName(status) {
  let name = '';
  switch (status) {
    case 'online': {
      name = '在线';
      break;
    }
    case 'offline': {
      name = '离线';
      break;
    }
    default: {
      name = '';
      break;
    }
  }
  return name;
}

function reseverStatusName(name) {
  let reseverName = '';
  switch (name) {
    case '在线': {
      reseverName = '离线';
      break;
    }
    case '离线': {
      reseverName = '在线';
      break;
    }
    default: {
      reseverName = '';
      break;
    }
  }
  return reseverName;
}


export {
  filterOverviewType,
  filterStatusName,
  reseverStatusName,
};