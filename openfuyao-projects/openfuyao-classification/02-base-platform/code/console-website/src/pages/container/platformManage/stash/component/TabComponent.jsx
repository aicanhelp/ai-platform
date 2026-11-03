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
import WarehouseConfiguration from '@/pages/applicationMarket/component/WarehouseConfiguration';
import ApplicationPackageManagement from '@/pages/applicationMarket/component/ApplicationPackageManagement';
import { containerRouterPrefix } from '@/constant.js';
import { useParams, useHistory } from 'inula-router';
import { useEffect, useState } from 'openinula';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { useTranslation } from 'react-i18next';

export default function TabComponent() {
  const history = useHistory();
  const { tabIndex } = useParams();
  const [activeIndex, setActiveIndex] = useState('wareHouse');
  const { t } = useTranslation('stash');
  const handleChangeTabIndex = (key) => {
    setActiveIndex(key);
    history.push(`/${containerRouterPrefix}/appMarket/stash/${key}`);
  };
  const items = [
    {
      key: 'wareHouse',
      label: t('tab.wareHouse'),
      children: <WarehouseConfiguration />,
    },
    {
      key: 'applicationPackage',
      label: t('tab.applicationPackage'),
      children: <ApplicationPackageManagement />,
    },
  ];

  useEffect(() => {
    setActiveIndex(tabIndex ?? 'wareHouse');
  }, [tabIndex]);
  return (
    <div className='configmap'>
      <BreadCrumbCom
        items={[
          { title: t('module.wareHouseConfig'), path: `/${containerRouterPrefix}/appMarket/stash/wareHouse`, disabled: true },
          { title: t('module.wareHouse'), path: `/${containerRouterPrefix}/appMarket/stash/wareHouse` },
        ]}
      />
      <div style={{ marginBottom: '20px' }}>
        <Tabs
          activeKey={activeIndex}
          onChange={handleChangeTabIndex}
          items={items}
        />
      </div>
    </div>
  );
}