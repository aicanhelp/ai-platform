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

/**
 * get请求
 */
export function GET(url, headers = { 'Content-Type': 'application/json' }) {
  return new Promise((resolve, reject) => {
    ir.get(url, {
      headers,
      withCredentials: true,
    })
      .then(response => {
        // 请求成功，处理响应数据
        resolve(response);
      })
      .catch(error => {
        // 请求失败，处理错误
        reject(error);
      });
  }
  );
}

/**
 * post请求
 */
export function POST(url, data, headers = { 'Content-Type': 'application/json' }) {
  return new Promise((resolve, reject) => {
    ir.post(url, data, {
      headers,
      withCredentials: true,
    }).then(response => {
      // 请求成功，处理响应数据
      resolve(response);
    })
      .catch(error => {
        // 请求失败，处理错误
        reject(error);
      });
  });
}

/**
 * delete请求
 */
export function DELETE(url, headers = { 'Content-Type': 'application/json' }) {
  return new Promise((resolve, reject) => {
    ir.delete(url, {
      headers,
      withCredentials: true,
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
}

/**
 * put请求
 */
export function PUT(url, data, headers = { 'Content-Type': 'application/json' }) {
  return new Promise((resolve, reject) => {
    ir.put(url, data, {
      headers,
      withCredentials: true,
    }).then(response => {
      // 请求成功，处理响应数据
      resolve(response);
    })
      .catch(error => {
        // 请求失败，处理错误
        reject(error);
      });
  });
}

/**
 * patch请求
 */
export function PATCH(url, data, headers = { 'Content-Type': 'application/json' }) {
  return new Promise((resolve, reject) => {
    ir.patch(url, data, {
      headers,
      withCredentials: true,
    }).then(response => {
      // 请求成功，处理响应数据
      resolve(response);
    })
      .catch(error => {
        // 请求失败，处理错误
        reject(error);
      });
  });
}