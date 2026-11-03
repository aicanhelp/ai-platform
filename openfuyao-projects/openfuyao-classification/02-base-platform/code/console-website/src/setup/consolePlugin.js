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

import { useStore } from 'openinula';
import { getClustersList } from '@/api/clusterApi';
import { listConsolePlugins } from '@/api/containerApi';
import { checkSessionStorage } from '@/tools/utils';

const loadConsolePlugins = async () => {
  const cpStore = useStore('consolePlugins');
  cpStore.$a.setConsolePlugins([]);

  await getClustersList().
    then(res => res.data).
    then(async (res) => {
      let clusterName;
      if (Object.prototype.hasOwnProperty.call(res, 'info')) {
        if (!checkSessionStorage('cluster')) {
          clusterName = Object.keys(res.info).filter(item => item === 'host').length > 0 ? 'host' : Object.keys(res.info)[0];
        } else {
          clusterName = sessionStorage.getItem('cluster');
        }
        if (clusterName) {
          await listConsolePlugins(clusterName).
            then(res2 => res2.data).
            then(res2 => Array.isArray(res2.data) ? res2.data : []).
            then(res2 => cpStore.$a.setConsolePlugins(res2)).
            catch(() => cpStore.$a.setConsolePlugins([]));
        }
      }
    }).
    catch(() => {
      cpStore.$a.setConsolePlugins([]);
    });
};

export default loadConsolePlugins;