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
const heartCheckSecond = 30 * 1000; // 心跳检测间隔时长
export default function useWebSocket(url = '', info = {}) {
  const [wsData, setWsData] = useState(null); // 接收的消息
  const [wsState, setWsState] = useState(''); // 连接
  const wsRef = useRef(null);
  const { joinParams, heartParams } = info ?? {};
  const lockReconnectRef = useRef(false); // 避免重复连接
  const countRef = useRef(0); // 连接次数
  const heartCheckRef = useRef(null); // 心跳检测单例，避免每次渲染重建导致定时器漏清
  useEffect(() => {
    if (url) {
      createWebSocket();
    }
    // 修复：添加清理函数，组件卸载时关闭连接、清除定时器
    return () => {
      clearTimeout(heartCheckRef.current?.timeoutObj);
      clearTimeout(heartCheckRef.current?.serverTimeoutObj);
      const ws = wsRef.current;
      if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        ws.close();
      }
    };
  }, [url]);

  const createWebSocket = () => { // 创建socket连接
    setWsState('');
    // 相对路径转绝对地址，浏览器 WebSocket 构造器要求绝对 URL
    const wsUrl = url.startsWith('ws://') || url.startsWith('wss://')
      ? url
      : `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}${url}`;
    wsRef.current = new WebSocket(wsUrl);
    initWebSocket();
  };

  const reconnect = () => { // 创建连接，lockReconnect使当前项目只会存在一个socket
    if (!lockReconnectRef.current) {
      lockReconnectRef.current = true;
      if (countRef.current < 10) {
        setTimeout(() => { // 没连接上会一直重连，设置延迟避免请求过多
          createWebSocket();
          countRef.current++;
          lockReconnectRef.current = false;
        }, 2000);
      }
    }
  };

  const initWebSocket = () => { // 初始化socket
    wsRef.current.onclose = function () { // 关闭
      if (wsRef.current.url.includes(url)) {
        setWsState('closed');
      }
    };

    wsRef.current.onerror = function () { // 连接错误
      setWsState('fail');
      reconnect();
    };

    wsRef.current.onopen = function () {
      setWsState('open');
      countRef.current = 0; // 修复：连接成功后重置计数，避免达到上限后永不重连
      if (joinParams) {
        wsRef.current.send(JSON.stringify(joinParams));
      }
      if (heartParams) {
        getHeartCheck().reset().start();
      }
    };

    wsRef.current.onmessage = function ({ data }) {
      setWsData(JSON.stringify(data));
      if (heartParams) {
        getHeartCheck().reset().start();
      }
    };
  };

  const getHeartCheck = () => { // 心跳检测（单例）
    if (!heartCheckRef.current) {
      heartCheckRef.current = {
        timeout: heartCheckSecond,
        timeoutObj: null,
        serverTimeoutObj: null,
        reset() {
          clearTimeout(this.timeoutObj);
          clearTimeout(this.serverTimeoutObj);
          return this; // 返回自身以支持链式调用 .start()
        },
        start() {
          this.timeoutObj = setTimeout(() => {
            // 这里发送一个心跳，后端收到后，返回一个心跳消息，
            wsRef.current.send(heartParams);
            this.serverTimeoutObj = setTimeout(() => { // 如果超过一定时间还没重置，说明后端主动断开了
              wsRef.current.close();
            }, this.timeout);
          }, this.timeout);
        },
      };
    }
    return heartCheckRef.current;
  };

  return [wsRef.current, wsData, wsState];
}
