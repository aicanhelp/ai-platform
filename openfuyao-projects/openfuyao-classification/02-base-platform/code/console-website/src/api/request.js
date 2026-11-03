/**
 *  Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  openFuyao is licensed under Mulan PSL v2.
 *  You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  You may obtain a copy of Mulan PSL v2 at:
  
 *       http://license.coscl.org.cn/MulanPSL2
  
 *   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *   EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *   MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *   See the Mulan PSL v2 for more details.
 */
import ir from 'inula-request';
import { logout } from '@/api/loginApi';
import { ResponseCode } from '@/common/constants';
import { useStore } from 'openinula';
import i18n from '@/i18n';

function request(method,
  url,
  data = null,
  headers = {
    'Content-Type': 'application/json',
  },
  ...args) {
  const intervalStore = useStore('interval');
  const terminalStore = useStore('terminal');
  return new Promise((resolve, reject) => {
    ir({
      method,
      url,
      data,
      headers,
      withCredentials: true,
      ...args[0],
    })
      .then(response => {
        // 请求成功，处理响应数据
        resolve(response);
      })
      .catch(error => {
        // 请求失败，处理错误
        let code = error?.response?.status;
        // 除了修改密码/登出以外的401错误，跳转到登陆界面（排除 logout 自身，避免登出 401 再次触发拦截器形成死循环）
        if (url !== '/password/modify' && url !== '/rest/auth/logout' && code === ResponseCode.UnAuthorized) {
          intervalStore.$a.setInterval(false);
          logout().then(() => {
            if (window.location.pathname !== '/rest/auth/login' && !window.location.pathname.includes('/auth/login')) {
              localStorage.clear();
              terminalStore.$a.setTerminalOpen(false);
              createModal(i18n.t('error.sessionTimeout', { ns: 'common' }));
            }
          }).catch(() => { }).finally(() => reject(error)); // 保证 Promise 最终 settle，调用方 await 不再永久挂起
        } else {
          reject(error);
        }
      });
  });
}

// 创建弹出框的函数
function createModal(message) {
  const themeStore = useStore('theme');
  if (!document.getElementById('myModal')) {
    // 创建弹出框
    let modal = document.createElement('div');
    modal.id = 'myModal';
    modal.style.position = 'absolute';
    modal.style.top = '20%';
    modal.style.left = '50%';
    modal.style.height = '160px';
    modal.style.width = '400px';
    themeStore.$s.theme === 'dark' ? modal.style.color = '#fff' : modal.style.color = '#666';
    modal.style.fontSize = '16px';
    modal.style.transform = 'translate(-50%, -50%)';
    themeStore.$s.theme === 'dark' ? modal.style.backgroundColor = '#171a1f' : modal.style.backgroundColor = 'white';
    modal.style.padding = '20px';
    modal.style.border = '1px solid #00000028';
    modal.style.borderRadius = '6px';
    modal.style.boxShadow = '0 0 6px 0 #00000028';
    modal.style.zIndex = 1001;
    modal.textContent = message;
    document.body.appendChild(modal);

    // 创建关闭按钮
    let closeButton = document.createElement('button');
    closeButton.id = 'closeButton';
    closeButton.textContent = i18n.t('actions.confirm', { ns: 'common' });
    closeButton.style.position = 'absolute';
    closeButton.style.bottom = '20px';
    closeButton.style.right = '20px';
    closeButton.style.fontSize = '16px';
    closeButton.style.cursor = 'pointer';
    closeButton.style.backgroundColor = '#3f66f5';
    closeButton.style.padding = '5px 10px';
    closeButton.style.border = '0px';
    closeButton.style.borderRadius = '6px';
    closeButton.style.color = '#fff';
    closeButton.style.width = '60px';
    closeButton.onclick = function () { removeModal(); };
    modal.appendChild(closeButton);
  }
}

// 移除弹出框的函数
function removeModal() {
  let modal = document.getElementById('myModal');
  let closeButton = document.getElementById('closeButton');
  if (modal && closeButton) {
    document.body.removeChild(modal);
    setTimeout(() => {
      window.location.href = '/rest/auth/login';
    }, 1000);
  }
}

export function GET(url, ...args) {
  return request('get', url, null, '', ...args);
}

export function POST(url, data, ...args) {
  return request('post', url, data, ...args);
}

export function DELETE(url, data, ...args) {
  return request('delete', url, data, ...args);
}

export function PUT(url, data, ...args) {
  return request('put', url, data, ...args);
}

export function PATCH(url, data, ...args) {
  return request('patch', url, data, ...args);
}