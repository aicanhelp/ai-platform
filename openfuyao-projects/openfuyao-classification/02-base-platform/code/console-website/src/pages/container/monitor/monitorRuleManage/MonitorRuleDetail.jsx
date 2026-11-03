/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { useCallback, useEffect, useState, useMemo } from 'openinula';
import { useParams } from 'inula-router';
import { message } from 'antd';
import { useTranslation } from 'react-i18next';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { containerRouterPrefix } from '@/constant';
import {
  ResponseCode,
  alertStatusEx,
} from '@/common/constants';
import { getAlertListData } from '@/api/monitorApi';
import { filterAlertStatus, filterAlertStatusStyle } from '@/utils/common';
import SkeletonArea from '@/components/SkeletonArea';
import EmptyData from '@/components/EmptyData';
import DetailBaseInfoList from '@/components/DetailBaseInfoList';
import ToastMsg from '@/components/ToastMsg';

export default function MonitorRuleDetail() {
  const { name } = useParams();
  const { t } = useTranslation('monitoring', 'common');
  const [messageApi, contextHolder] = message.useMessage();
  const [alertDetail, setAlertDetail] = useState();
  const [detailLoaded, setDetailLoaded] = useState(false);

  const getAlertDetail = useCallback(async () => {
    try {
      const res = await getAlertListData();
      if (res.status === ResponseCode.OK) {
        const [data] = (res.data?.alertingRules ?? []).filter(item => item.name === name);
        setAlertDetail(data);
      }
    } catch (e) {
      messageApi.error(t('messages.fetchFailed', { ns: 'common', message: e?.response?.data?.message }));
    } finally {
      setDetailLoaded(true);
    }
  }, [name]);

  useEffect(() => {
    getAlertDetail();
  }, [getAlertDetail]);

  const baseInfoItems = useMemo(() => {
    return [
      { key: t('alertRule.detail.group'), value: alertDetail?.group },
      {
        key: t('alertRule.detail.status'),
        value: <>
          <span className={`${alertDetail?.status ? alertDetail?.status.toLowerCase() : ''}_circle`}></span>
          <span>{alertStatusEx[alertDetail?.status] ?? alertDetail?.status}</span>
        </>,
      },
      { key: t('alertRule.detail.duration'), value: alertDetail?.duration },
      { key: t('alertRule.detail.name'), value: alertDetail?.name },
      {
        key: t('alertRule.detail.level'),
        value: (
          <span style={filterAlertStatusStyle(alertDetail?.severity)}>
            {filterAlertStatus(alertDetail?.severity)}
          </span>
        ),
      },
    ];
  }, [alertDetail, t]);

  const detailInfoItems = useMemo(() => {
    return [
      { key: t('alertRule.detail.description'), value: alertDetail?.description },
      { key: t('alertRule.detail.summary'), value: alertDetail?.summary },
      { key: t('alertRule.detail.expression'), value: alertDetail?.query },
    ];
  }, [alertDetail, t]);

  let detailContent;
  if (!detailLoaded) {
    detailContent = <SkeletonArea />;
  } else if (alertDetail) {
    detailContent = <div className={'tab_container container_margin_box monitor_rule_container_width'}>
      <div className='detail_card'>
        <h3>{t('alertRule.detail.ruleInfo')}</h3>
        <div className="detail_info_box">
          <DetailBaseInfoList items={baseInfoItems} columns={2} />
        </div>
        <div className='detail_info_box'>
          <DetailBaseInfoList items={detailInfoItems} columns={1} />
        </div>
      </div>
    </div>;
  } else {
    detailContent = <EmptyData />;
  }

  return <div className="child_content">
    <ToastMsg contextHolder={contextHolder} />
    <BreadCrumbCom className='create_bread' items={[
      { title: t('module.title'), path: `/${containerRouterPrefix}/monitor`, disabled: true },
      { title: t('module.ruleList'), path: '/monitorRuleManage' },
      { title: t('alertRule.detail.title'), path: '/detail' },
    ]} />
    {detailContent}
  </div >;
}
