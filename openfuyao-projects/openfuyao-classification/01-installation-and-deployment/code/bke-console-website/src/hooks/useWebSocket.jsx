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
 * @name useWebSocket
 * @export
 * @param {url} -（必需）ws/wss地址
 * @param {info} -（不必需）socket运行过程中的一些参数，进入参数joinParams，心跳参数heartParams
 * @returns
 * {ws} -socket对象
 * {wsData} -接收的socket消息
 */
import { useEffect, useState, useRef } from 'openinula';
let lockReconnect = false; // 避免重复连接
const heartCheckSecond = 30 * 1000; // 心跳检测间隔时长
let count = 0; // 连接次数
export default function useWebSocket(url = '', info = {}) {
  const [wsData, setWsData] = useState({}); // 接收的消息
  const [wsState, setWsState] = useState(''); // 连接
  const wsRef = useRef(null);
  const { joinParams, heartParams } = info ?? {};
  useEffect(() => {
    if (url) {
      createWebSocket();
    }
  }, [url]);

  const createWebSocket = () => { // 创建socket连接
    setWsState('');
    wsRef.current = new WebSocket(url);
    initWebSocket();
  };

  const reconnect = () => { // 创建连接，lockReconnect使当前项目只会存在一个socket
    if (!lockReconnect) {
      lockReconnect = true;
      if (count < 10) {
        setTimeout(() => { // 没连接上会一直重连，设置延迟避免请求过多
          createWebSocket();
          count++;
          lockReconnect = false;
        }, 2000);
      }
    }
  };

  const initWebSocket = () => { // 初始化socket
    wsRef.current.onclose = function (evt) { // 关闭
      if (wsRef.current.url.includes(url)) {
        setWsState('closed');
      }
    };

    wsRef.current.onerror = function (evt) { // 连接错误
      setWsState('fail');
      reconnect();
    };

    wsRef.current.onopen = function () {
      setWsState('open');
      let dt = new Date();
      let str = `${dt.getFullYear()}-${(dt.getMonth() + 1)}-${dt.getDate()}  ${dt.getHours()}:${dt.getMinutes()}:${dt.getSeconds()}`;
      if (joinParams && Object.keys(joinParams)) {
        wsRef.current.send(JSON.stringify(joinParams));
      }
      if (heartParams) {
        heartCheck.reset().start();
      }
    };

    wsRef.current.onmessage = function ({ data }) {
      setWsData(JSON.stringify(data));
      if (heartParams) {
        heartCheck.reset().start();
      }
    };
  };

  let heartCheck = { // 心跳检测
    timeout: heartCheckSecond,
    timeoutObj: null,
    serverTimeoutObj: null,
    reset: () => {
      clearTimeout(heartCheck.timeoutObj);
      clearTimeout(heartCheck.serverTimeoutObj);
      return;
    },
    start: () => {
      heartCheck.timeoutObj = setTimeout(() => {
        // 这里发送一个心跳，后端收到后，返回一个心跳消息，
        wsRef.current.send(heartParams);
        heartCheck.serverTimeoutObj = setTimeout(() => { // 如果超过一定时间还没重置，说明后端主动断开了
          wsRef.current.close();
        }, heartCheck.timeout);
      }, timeout);
    },
  };

  return [wsRef.current, wsData, wsState];
}