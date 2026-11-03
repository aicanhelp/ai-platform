/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import ir from 'inula-request';

function request(method,
  url,
  data = null,
  headers = {
    'Content-Type': 'application/json',
  },
  args) {
  return new Promise((resolve, reject) => {
    ir({
      method,
      url,
      data,
      headers,
      withCredentials: true,
      ...args,
    })
      .then(response => {
        // 请求成功，处理响应数据
        resolve(response);
      })
      .catch(error => {
        // 请求失败，处理错误
        reject(error);
      });
  });
};

export function GET(url, ...args) {
  return request('get', url, null, '', ...args);
}

export function POST(url, data, ...args) {
  return request('post', url, data, ...args);
}

export function DELETE(url, ...args) {
  return request('delete', url, ...args);
}

export function PUT(url, data, ...args) {
  return request('put', url, data, ...args);
}

export function PATCH(url, data, ...args) {
  return request('patch', url, data, ...args);
}