/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import EventsCard from '@/components/layouts/EventsCard';
import '@/styles/pages/event.less';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { containerRouterPrefix } from '@/constant.js';
import { useTranslation } from 'react-i18next';

export default function Events() {
  const { t } = useTranslation('events');

  return (
    <div className='event child_content'>
      <div className="alert-bread">
        <BreadCrumbCom className='alert-bread-auto' items={[{ title: t('module.title'), path: `/${containerRouterPrefix}/event` }]} />
      </div>
      <EventsCard />
    </div>
  );
}
