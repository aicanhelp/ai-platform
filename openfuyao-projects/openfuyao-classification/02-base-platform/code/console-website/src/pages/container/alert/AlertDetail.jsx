/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { containerRouterPrefix } from '@/constant';
import { useParams } from 'inula-router';
import { useMemo, useStore } from 'openinula';
import '@/styles/pages/alert.less';
import { useLocation } from 'inula-router';
import dayjs from 'dayjs';
import { solveAnnotation } from '@/tools/utils';
import { filterAlertStatus, filterAlertStatusStyle } from '@/utils/common';
import LabelTag from '@/components/LabelTag';
import DetailBaseInfoList from '@/components/DetailBaseInfoList';
import { useTranslation } from 'react-i18next';

export default function AlertDetail() {
  const { t } = useTranslation('alert');
  const param = useParams();
  const name = decodeURI(param.alert_name);
  const location = useLocation();
  const themeStore = useStore('theme');
  const detailsInfo = location.state?.currentdetail || {};

  const baseInfoItems = useMemo(() => ([
    { key: t('detail.alertRule'), value: detailsInfo?.name },
    {
      key: t('detail.alertLevel'),
      value: (
        <div style={filterAlertStatusStyle(detailsInfo?.level)}>
          {filterAlertStatus(detailsInfo?.level)}
        </div>
      ),
    },
    { key: t('detail.description'), value: detailsInfo?.desc },
    { key: t('detail.summary'), value: detailsInfo?.summary },
    { key: t('detail.triggerTime'), value: detailsInfo?.time ? dayjs(detailsInfo.time).format('YYYY-MM-DD HH:mm:ss') : '--' },
  ]), [detailsInfo, t]);

  return (
    <div className='alertDetail child_content'>
      <div className='alertDetail_top'>
        <BreadCrumbCom items={[
          { title: t('module.title'), disabled: true },
          { title: t('currentAlert.title'), path: `/${containerRouterPrefix}/alert/alertIndex` },
          { title: t('detail.detail'), path: `/${containerRouterPrefix}/alert/alertIndex/detail` + `${name}` },
        ]} />
        <div className='alertDetail_top_row' style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
          <p className='alertDetail_top_name' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{name}</p>
        </div>
      </div>
      <div className='tab_container container_margin_box normal_container_height'>
        <div className="detail_card">
          <h3>{t('detail.basicInfo')}</h3>
          <div className="detail_info_box">
            <DetailBaseInfoList items={baseInfoItems} columns={2} />
            <div className="annotation">
              <div className='ann_title'>
                <p>{t('detail.labels')}</p>
              </div>
              <div className='key_value'>
                {solveAnnotation(detailsInfo?.labels).map(item => (
                  <LabelTag key={item.key} labelKey={item.key} labelValue={item.value} />),
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}