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
import Inula, { useCallback, useEffect, useLayoutEffect, useRef, useState, useStore } from 'openinula';
import { Button } from 'antd';
import { useTranslation } from 'react-i18next';
import EmptyData from './EmptyData';
import { getConsolePlugin } from '@/api/containerApi';

const MicroAppPage = Inula.memo(({ name, subKey }) => {
  const { t } = useTranslation(['application', 'common']);
  const consolePluginStore = useStore('consolePlugins');
  const mod = consolePluginStore.$s.modules[name];
  const mountRef = useRef(null);
  const [loadingFailed, setLoadingFailed] = useState(false);
  const loadSeqRef = useRef(0);

  const loadPlugin = useCallback(() => {
    setLoadingFailed(false);
    // 扩展组件 name 切换后旧请求 resolve 不再覆盖新实例状态
    loadSeqRef.current += 1;
    const seq = loadSeqRef.current;
    getConsolePlugin(name).then((res3) => {
      if (seq !== loadSeqRef.current) { return; }
      const resMod = new Function(res3.data);
      consolePluginStore.$a.setModule(name, resMod);
    }).catch(_ => {
      if (seq !== loadSeqRef.current) { return; }
      setLoadingFailed(true);
    });
  }, [name]);

  useEffect(() => {
    if (!consolePluginStore.$s.modules[name]) {
      loadPlugin();
    }
  }, [loadPlugin]);

  useEffect(() => {
    if (mountRef.current) {
      mountRef.current.className = subKey || '';
    }
  }, [subKey]);

  useLayoutEffect(() => {
    if (mod && mountRef.current) {
      const unmount = mod();
      return () => {
        if (typeof unmount === 'function') {
          unmount();
        }
      };
    }
    return undefined;
  }, [mod]);

  useEffect(() => {
    // 挂载时快照 pathname：卸载时路由已切换，读 window.location.pathname 会误判
    const pathname = window.location.pathname;
    if (pathname.includes('ColocationMonitor')) {
      window.postMessage({ type: 'colocationOverviewPageDestory', isDestory: false });
    }
    if (pathname.includes('clusterManage') && pathname.split('/').pop() === 'clusterManage') {
      window.postMessage({ type: 'clusterPageDestory', isDestory: false });
    }
    if (pathname.includes('/clusterManage/detail')) {
      window.postMessage({ type: 'clusterDetailPageDestory', isDestory: false });
    }
    return () => {
      if (pathname.includes('ColocationMonitor')) {
        window.postMessage({ type: 'colocationOverviewPageDestory', isDestory: true });
      }
      if (pathname.includes('/clusterManage/detail')) {
        window.postMessage({ type: 'clusterDetailPageDestory', isDestory: true });
      }
      if (pathname.includes('clusterManage') && pathname.split('/').pop() === 'clusterManage') {
        window.postMessage({ type: 'clusterPageDestory', isDestory: true });
      }
    };
  }, [subKey]);

  const blankMsg = loadingFailed ? t('plugin.notStarted') : t('plugin.loading');

  return (
    <div style={{ height: '100%', position: 'relative' }}>
      <div ref={mountRef} id={`${name}_root`} style={{ height: '100%' }} />
      {!mod && (
        <div style={{ position: 'absolute', inset: 0 }}>
          <EmptyData>
            <p>{blankMsg}</p>
            {loadingFailed && (
              <Button type="link" onClick={loadPlugin}>
                {t('common.retry', { ns: 'common' })}
              </Button>
            )}
          </EmptyData>
        </div>
      )}
    </div>
  );
});

export default MicroAppPage;
