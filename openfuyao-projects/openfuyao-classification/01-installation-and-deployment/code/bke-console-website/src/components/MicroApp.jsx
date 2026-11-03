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
import { GET } from '@/api/request';
import Inula, { useEffect, useStore } from 'openinula';
import { useLocation } from 'inula-router';

const MicroAppPage = Inula.memo(({ name, key, subKey }) => {
  const location = useLocation();
  const consolePluginStore = useStore('consolePlugins');
  const mod = consolePluginStore.$s.modules[name];
  let rendered = false;

  useEffect(() => {
    if (!rendered) {
      mod();
      rendered = true;
    }
    if (window.location.pathname.includes('ColocationMonitor')) {
      window.postMessage({ type: 'colocationOverviewPageDestory', isDestory: false });
    }
    if (window.location.pathname.includes('clusterManage') && window.location.pathname.split('/').pop() === 'clusterManage') {
      window.postMessage({ type: 'clusterPageDestory', isDestory: false });
    }
    if (window.location.pathname.includes('/clusterManage/detail')) {
      window.postMessage({ type: 'clusterDetailPageDestory', isDestory: false });
    }
    return () => {
      // 注册事件
      window.postMessage({ type: 'colocationOverviewPageDestory', isDestory: true });
      window.postMessage({ type: 'clusterDetailPageDestory', isDestory: true });
      window.postMessage({ type: 'clusterPageDestory', isDestory: true });
    };
  }, [mod]);

  return <div id={`${name}_root`} className={subKey} style={{ height: '100%' }}></div>;
});

export default MicroAppPage;