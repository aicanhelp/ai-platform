/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { containerRouterPrefix, ResponseCode } from '@/common/constants';
import { useIntl } from 'inula-intl';
import { useHistory, Link } from 'inula-router';
import { useCallback, useEffect, useState } from 'openinula';
import useLocalStorage from '@/hooks/useLocalStorage';
import QuickStart from '@/components/QuickStart';
import '@/styles/rayOverview.less';
import { message } from 'antd';
import rayclusters from '@/assets/images/rayclusters.png';
import raycluster from '@/assets/images/raycluster.png';
import rayjob from '@/assets/images/rayjob.png';
import rayservice from '@/assets/images/rayservice.png';
import { getOverviewNum } from '@/api/rayApi.js';
import { skipSendMainMsg } from '@/tools/utils';
import RayCharts from '@/pages/overview/RayCharts';
export default function RayOverview() {
  const intl = useIntl();
  const history = useHistory();
  const [messageApi, contextHolder] = message.useMessage();
  const [isQuickstart, setIsQuickstart] = useState(false);
  const [overviewSeletor, setOverviewSeletor] = useState([
    {
      key: 'RayCluster',
      img: rayclusters,
      num: 0,
      title: intl.formatMessage({ id: 'overview.activeRayClusters' }),
      path: '/RayCluster',
    },
    {
      key: 'RayService',
      img: rayservice,
      num: 0,
      title: intl.formatMessage({ id: 'overview.activeRayService' }),
      path: '/RayServices',
    },
    {
      key: 'RayJob',
      img: rayjob,
      num: 0,
      title: intl.formatMessage({ id: 'overview.activeRayJobs' }),
      path: '/RayJob',
    },
    {
      img: raycluster,
      num: 0,
      title: intl.formatMessage({ id: 'overview.activeRayCluster' }),
      path: '',
    },
  ]);
  const getOverviewData = async () => {
    try {
      const res = await getOverviewNum();
      if (res.data.code === ResponseCode.OK) {
        setOverviewSeletor([
          {
            key: 'RayCluster',
            img: rayclusters,
            num: res.data.data.rayclusters,
            title: intl.formatMessage({ id: 'overview.activeRayClusters' }),
            path: '/RayCluster',
          },
          {
            key: 'RayService',
            img: rayservice,
            num: res.data.data.rayservices,
            title: intl.formatMessage({ id: 'overview.activeRayService' }),
            path: '/RayService',
          },
          {
            key: 'RayJob',
            img: rayjob,
            num: res.data.data.rayjobs,
            title: intl.formatMessage({ id: 'overview.activeRayJobs' }),
            path: '/RayJob',
          },
          {
            img: raycluster,
            num: res.data.data.total,
            title: intl.formatMessage({ id: 'overview.activeRayCluster' }),
            path: '',
          },
        ]);
      } else {
        let msg = '';
        msg = JSON.stringify(res.data.data) !== '{}' ? `${res.data.msg + ':' + res.data.data}` : res.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'overview.fetchDataFail' })},${msg}`);
      }
    } catch (error) {
      if (error) {
        let msg = '';
        msg =
          JSON.stringify(error.response.data.data) !== '{}'
            ? `${error.response.data.msg + ':' + error.response.data.data}`
            : error.response.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'overview.fetchDataFail' })},${msg}`);
      }
    }
  };

  const handleSkip = (event, record) => {
    if (record.path) {
      skipSendMainMsg(event, record.key, record.path, history);
    }
  };

  useEffect(() => {
    getOverviewData();
  }, []);
  return (
    <div className="child_content">
      <BreadCrumbCom
        className="create_bread"
        items={[
          { title: 'openFuyao Ray', disabled: true },
          { title: intl.formatMessage({ id: 'overview.quickStart' }), path: `/` },
        ]}
      />
      <div className="overview_content">
        {contextHolder}
        <div className="overview_top">
          <p className="overview_top_begin" onClick={() => setIsQuickstart(true)}>
            {intl.formatMessage({ id: 'overview.quickStart' })}
          </p>
          <p className="overview_top_title">{intl.formatMessage({ id: 'overview.welcome' })}</p>
          <div className="overview_top_seletor">
            {overviewSeletor.map((item) => {
              return (
                <div className="overview_top_oneseletor" onClick={(e) => handleSkip(e, item)}>
                  <img src={item.img} className="overview_top_oneseletor_img" alt="" />
                  <div>
                    <p className="overview_top_oneseletor_num">{item.num}</p>
                    <p className="overview_top_oneseletor_title">{item.title}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
        <RayCharts />
        <QuickStart isQuickOpen={isQuickstart} onCancelQuick={() => setIsQuickstart(false)} />
      </div>
    </div>
  );
}
