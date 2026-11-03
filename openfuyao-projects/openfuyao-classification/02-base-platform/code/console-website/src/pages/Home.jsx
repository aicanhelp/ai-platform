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

import { Link } from 'inula-router';
import { useStore } from 'openinula';
import { useTranslation } from 'react-i18next';
import AppMarket from '@/assets/images/home/appMarket.png';
import Migrant from '@/assets/images/home/migrant.png';
import Pluggable from '@/assets/images/home/pluggable.png';
import AppManage from '@/assets/images/home/appManage.png';
import Engine from '@/assets/images/home/engine.png';
import Turbo from '@/assets/images/home/turbo.png';
import Monitoring from '@/assets/images/home/monitoring.png';
import Overview from '@/assets/images/home/overview.png';
import boxBack from '@/assets/images/home/boxBackground.png';
import DarkBoxBack from '@/assets/images/home/darkBoxBackground.png';
import boxBackHover from '@/assets/images/home/boxBackgroundHover.png';
import DarkBoxBackHover from '@/assets/images/home/darkBoxBackgroundHover.png';
import '@/styles/pages/home.less';

export default function Home() {
  const { t } = useTranslation('home');
  const themeStore = useStore('theme');

  const features = [
    {
      title: t('title.install'),
      display: 'block',
      items: [
        {
          img: AppMarket,
          name: t('subtitle.applicationMarket'),
          desc: t('message.applicationMarket'),
          linkText: t('link.applicationMarket'),
          target: '/container_platform/appMarket/appOverview',
        },
        {
          img: Migrant,
          name: t('subtitle.migration'),
          desc: t('message.migration'),
          linkText: t('link.migration'),
          target: undefined,
        },
      ],
    },
    {
      title: t('title.maintanance'),
      display: 'block',
      items: [
        {
          img: Pluggable,
          name: t('subtitle.pluggability'),
          desc: t('message.pluggability'),
          linkText: t('link.pluggability'),
          target: '/container_platform/extendManage',
        },
        {
          img: AppManage,
          name: t('subtitle.applicationManagement'),
          desc: t('message.applicationManagement'),
          linkText: t('link.applicationManagement'),
          target: '/container_platform/applicationManageHelm',
        },
      ],
    },
    {
      title: t('title.acceleration'),
      display: 'none',
      items: [
        {
          img: Engine,
          name: t('subtitle.engine'),
          desc: t('message.engine'),
          linkText: t('link.engine'),
          target: undefined,
        },
        {
          img: Turbo,
          name: t('subtitle.turbo'),
          desc: t('message.turbo'),
          linkText: t('link.turbo'),
          target: undefined,
        },
      ],
    },
    {
      title: t('title.observability'),
      display: 'block',
      items: [
        {
          img: Monitoring,
          name: t('subtitle.dashboard'),
          desc: t('message.dashboard'),
          linkText: t('link.dashboard'),
          target: '/container_platform/monitor/monitorDashboard',
        },
        {
          img: Overview,
          name: t('subtitle.clusterOverview'),
          desc: t('message.clusterOverview'),
          linkText: t('link.clusterOverview'),
          target: '/container_platform/overview',
        },
      ],
    },
  ];

  return (
    <div className='home_container'>
      <p className='home_main_title'>
        {t('mainTitle')}
      </p>
      <div className='feature_boxes'>
        {
          features.map((item) => (
            <div className='feature_box' style={{ display: item.display }}>
              <img className='feature_box_back' src={themeStore.$s.theme === 'light' ? boxBack : DarkBoxBack} alt='' />
              <img className='feature_box_back_hover' src={themeStore.$s.theme === 'light' ? boxBackHover : DarkBoxBackHover} alt='' />
              <p className='feature_title'>{item.title}</p>
              <div className='feature_items'>
                <div className='feature_item'>
                  <img src={item.items[0].img} alt='' />
                  <p className='feature_name'>{item.items[0].name}</p>
                  <p className='feature_desc'>{item.items[0].desc}</p>
                  {
                    item.items[0].target ?
                      <p className='feature_link'>
                        <Link to={item.items[0].target}>{item.items[0].linkText}</Link>
                      </p> :
                      <p className='feature_link'>{item.items[0].linkText}</p>
                  }
                </div>
                <div className='feature_item'>
                  <img src={item.items[1].img} alt='' />
                  <p className='feature_name'>{item.items[1].name}</p>
                  <p className='feature_desc'>{item.items[1].desc}</p>
                  {
                    item.items[1].target ?
                      <p className='feature_link'>
                        <Link to={item.items[1].target}>{item.items[1].linkText}</Link>
                      </p> :
                      <p className='feature_link'>{item.items[1].linkText}</p>
                  }
                </div>
              </div>
            </div>
          ))
        }
      </div>
    </div>
  );
}