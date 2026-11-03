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
import { message } from 'antd';
import { getClustersList } from '@/api/clusterApi';
import { ResponseCode } from '@/common/constants';
import { checkSessionStorage } from '@/tools/utils';
import i18n from '@/i18n';

const getCluster = () => {
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
      await getClustersList().
        then(res => res.data).
        then(async (res) => {
          if (Object.prototype.hasOwnProperty.call(res, 'info')) {
            if (!checkSessionStorage('cluster')) {
              sessionStorage.setItem('cluster', Object.keys(res.info).filter(item => item === 'host').length > 0
                ? 'host' : Object.keys(res.info)[0]);
            }
          }
          if (cycleCount > 0) {
            window.location.reload();
          }
          cycleCount = 0;
        }).
        catch(error => {
          const status = error.response?.status;
          if (status === ResponseCode.BadGateway || status === ResponseCode.ServiceUnavailable) {
            message.error(i18n.t('error.multiClusterPreparing', { ns: 'common' }));
            cycleCount++;
            timeSearch();
          } else {
            message.error(error.response?.data?.Message || error.response?.data?.message);
            cycleCount = 0;
            clearTimeout(setTimeoutId);
          }
        });
    } else {
      message.error(i18n.t('error.multiClusterStartFailed', { ns: 'common' }));
      clearTimeout(setTimeoutId);
    }
  };
  timeCycle();
};

export default getCluster;