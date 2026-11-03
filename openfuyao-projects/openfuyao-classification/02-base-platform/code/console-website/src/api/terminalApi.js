/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { getUserName } from '@/api/loginApi';

// 统一公共前缀
function terminalPrefix() {
  let clusterName = sessionStorage.getItem('cluster');
  let paramObj = {
    terminalUrl: `/clusters/${clusterName}/rest/webterminal/v1`,
  };
  return paramObj;
}

/**
 * 判断是否有权限
 */
export async function checkTerminalAccess(str) {
  let newCount = 0;
  let auth = false; // 鉴权
  return new Promise((resolve) => {
    // 兜底：无论何种路径，15s 内必须 settle，避免调用方永久挂起
    const fallbackTimer = setTimeout(() => {
      settle({ data: { Code: 500 } });
    }, 15000);
    const settle = (value) => {
      clearTimeout(fallbackTimer);
      resolve(value);
    };
    const socket = new WebSocket(`${terminalPrefix().terminalUrl}${str}`);
    socket.onmessage = e => {
      if (!newCount) {
        if (e.data.includes('User has no access')) {
          settle({ data: { Code: 403 } });
          socket.close();
        } else if (e.data.includes('User has access')) {
          auth = true; // 鉴权成功
          setTimeout(() => {
            settle({ data: { Code: 200, message: 'Time too Long!' } });
            socket.close();
          }, 3000);
        } else {
          settle({ data: { Code: 500 } });
          socket.close();
        }
      } else {
        if (auth) {
          if (e.data.includes('404')) {
            settle({ data: { Code: 404, data: e.data } });
            socket.close();
          } else {
            settle({ data: { Code: 200 } });
            socket.close();
          }
        }
      }
      newCount++;
    };

    socket.onerror = () => {
      // 调用户接口判断是否401
      getUserName().then(() => settle({ data: { Code: 500 } })).catch(() => settle({ data: { Code: 500 } }));
    };
  });
}