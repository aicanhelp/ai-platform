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

import { Fragment, useMemo } from 'openinula';
import Dayjs from 'dayjs';
import { getHelmManageStateKey, getHelmManageStateIcon } from '@/utils/common';
import { useTranslation } from 'react-i18next';
import '@/styles/pages/helm.less';
import DetailBaseInfoList from '@/components/DetailBaseInfoList';

export default function HelmDetailInfo({ helmName, helmDetailDataProps }) {
  const { t } = useTranslation('application');

  // Get status key and translated label
  const statusKey = getHelmManageStateKey(helmDetailDataProps?.info?.status);
  const statusLabel = t(`status.${statusKey}`);

  const baseInfoItems = useMemo(() => [
    { key: t('detail.appName'), value: helmName },
    { key: t('detail.appVersion'), value: helmDetailDataProps.chart?.metadata?.appVersion ? helmDetailDataProps.chart.metadata.appVersion : '--' },
    { key: t('detail.namespace'), value: helmDetailDataProps?.namespace },
    { key: t('detail.chartVersion'), value: helmDetailDataProps.chart?.metadata?.version ? helmDetailDataProps.chart.metadata.version : '--' },
    {
      key: t('detail.status'),
      value: <span>{getHelmManageStateIcon(statusKey)}{statusLabel}</span>,
    },
    { key: t('detail.createTime'), value: Dayjs(helmDetailDataProps?.info?.firstDeployed).format('YYYY-MM-DD HH:mm') },
    { key: t('detail.chartName'), value: helmDetailDataProps.chart?.metadata?.name },
    { key: t('detail.updateTime'), value: Dayjs(helmDetailDataProps?.info?.lastDeployed).format('YYYY-MM-DD HH:mm') },
  ], [helmDetailDataProps, t, statusKey, statusLabel]);

  return <Fragment>
    <div className="helm_tab_container container_margin_box tooltip_container_height">
      <div className="detail_card">
        <h3>{t('detail.basicInfo')}</h3>
        <div className="detail_info_box">
          <DetailBaseInfoList items={baseInfoItems} columns={2} />
        </div>
      </div>
    </div>
  </Fragment>;
}