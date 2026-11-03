/*
 * Copyright (c) 2023 Huawei Technologies Co.,Ltd.
 *
 * openInula is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

import Inula, { useStore } from 'openinula';
import '@/index.less';
import App from '@/App';
import { message } from 'antd';
import { getConsolePlugins } from '@/api/containerApi';
import { GET } from '@/api/request';
import { checkSessionStorage } from '@/tools/utils';
import { getClustersList } from '@/api/clusterApi';
import { ResponseCode } from '@/common/constants';
import { IntlProvider } from 'inula-intl';
import { getMessages, getInitialLocale } from '@/i18n';

window.__OPENFUYAO__ = true;
window.__OPENFUYAO_LOCALE__ = getInitialLocale();

const cpStore = useStore('consolePlugins');

cpStore.$a.setConsolePlugins([]);

let cycleCount = 0;
let setTimeoutId = 0;
const timeSearch = () => {
  clearTimeout(setTimeoutId);
  const id = setTimeout(() => {
    timeCycle();
  }, 10000);
  setTimeoutId = id;
};
const timeCycle = async () => {
  if (cycleCount < 61) {
    await getClustersList()
      .then((res) => res.data)
      .then(async (res) => {
        if (Object.prototype.hasOwnProperty.call(res, 'info')) {
          if (!checkSessionStorage('cluster')) {
            sessionStorage.setItem(
              'cluster',
              Object.keys(res.info).filter((item) => item === 'host').length > 0 ? 'host' : Object.keys(res.info)[0],
            );
          }
        }
        if (cycleCount > 0) {
          window.location.reload();
        }
        cycleCount = 0;
      })
      .catch((error) => {
        if (error.response.status === ResponseCode.BadGateway || error.response.status === 503) {
          message.error(getMessages(getInitialLocale())['cluster.preparingMultiCluster']);
          cycleCount++;
          timeSearch();
        } else {
          message.error(error.response?.data?.Message || error.response?.data?.message);
          cycleCount = 0;
          clearTimeout(setTimeoutId);
        }
      });
  } else {
    message.error(getMessages(getInitialLocale())['cluster.multiClusterStartupFailed']);
    clearTimeout(setTimeoutId);
  }
};
timeCycle();

await getClustersList()
  .then((res) => res.data)
  .then(async (res) => {
    let clusterName = '';
    if (Object.prototype.hasOwnProperty.call(res, 'info')) {
      if (!checkSessionStorage('cluster')) {
        clusterName =
          Object.keys(res.info).filter((item) => item === 'host').length > 0 ? 'host' : Object.keys(res.info)[0];
      } else {
        clusterName = sessionStorage.getItem('cluster');
      }
      if (clusterName) {
        await getConsolePlugins(sessionStorage.getItem('cluster'))
          .then((res2) => res2.data)
          .then((res2) => (Array.isArray(res2.data) ? res2.data : []))
          .then((res2) => cpStore.$a.setConsolePlugins(res2))
          .catch(() => cpStore.$a.setConsolePlugins([]));
        cpStore.$s.consolePlugins.forEach((cp) => {
          const name = cp.pluginName;
          GET(`${window.location.origin}/clusters/${clusterName}/proxy/${name}/dist/${name}.mjs`).then((res3) =>
            cpStore.$a.setModule(name, new Function(res3.data)),
          );
        });
      }
    }
  });

const locale = getInitialLocale();
Inula.render(
  <IntlProvider locale={locale} messages={getMessages(locale)}>
    <App />
  </IntlProvider>,
  document.getElementById('root'),
);
