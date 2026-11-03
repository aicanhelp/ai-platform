/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Tabs } from 'antd';
import { useStore } from 'openinula';
import DetailedInformation from '@/pages/applicationMarket/DetailedInformation';
import DefaultParameter from '@/pages/applicationMarket/DefaultParameter';
import { useTranslation } from 'react-i18next';

export default function DetailsTabComponent({ detailInfo, defaultParams, chart, loading }) {
  const themeStore = useStore('theme');
  const { t } = useTranslation('appMarket');
  const items = [
    {
      key: '1',
      label: t('tab.detailInfo'),
      children: <DetailedInformation detailInfo={detailInfo} loading={loading} />,
    },
    {
      key: '2',
      label: t('tab.defaultParams'),
      children: <DefaultParameter defaultParams={defaultParams} chart={chart} loading={loading} />,
    },
  ];

  return (
    <div style={{ paddingBottom: '20px', backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
      <Tabs items={items} tabBarStyle={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff', color: '#000' }}></Tabs>
    </div>
  );
}
