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
import BreadCrumbCom from '@/components/BreadCrumbCom';
import '../styles/overview.less';

import { useEffect, useState } from 'openinula';
import { useIntl } from 'inula-intl';
import ResourceWorkload from './overview/ResourceWorkload';
import ResourceUsed from './overview/ResourceUsed';
import ResourceManage from './overview/ResourceManage';
import useLocalStorage from '@/hooks/useLocalStorage';

export default function OverviewPage() {
  const intl = useIntl();

  const [currentTheme] = useLocalStorage('theme', 'light');
  const [hiddenPage, setHiddenPage] = useState(false);

  const handlePassMessage = (event) => {
    if (event.data.type === 'colocationOverviewPageDestory') {
      if (event.data.isDestory) {
        setHiddenPage(true);
        // 清理定时器
        const colocationTime1ID = localStorage.getItem('colocationTime1Id') || '';
        const colocationTime2ID = localStorage.getItem('colocationTime2Id') || '';
        const colocationTime3ID = localStorage.getItem('colocationTime3Id') || '';

        clearTimeout(colocationTime1ID);
        clearTimeout(colocationTime2ID);
        clearTimeout(colocationTime3ID);

        localStorage.removeItem('colocationTime1Id');
        localStorage.removeItem('colocationTime2Id');
        localStorage.removeItem('colocationTime3Id');
      } else {
        setHiddenPage(false);
      }
    }
  };

  useEffect(() => {
    window.addEventListener('message', handlePassMessage);
    return () => window.removeEventListener('message', handlePassMessage);
  }, []);

  return (
    <div className="child_content overview">
      <BreadCrumbCom
        className="create_bread"
        items={[
          { title: intl.formatMessage({ id: 'colocation.title' }), path: `/ColocationOverview` },
          { title: intl.formatMessage({ id: 'colocation.overview' }), path: `/` },
        ]}
      />
      {!hiddenPage && (
        <div className="overview_card" style={{ backgroundColor: currentTheme === 'light' ? '#f7f7f7' : 'black' }}>
          <div className="overview_flex">
            <div
              className="overview_flex_infomation"
              style={{ backgroundColor: currentTheme === 'light' ? '#fff' : '#2a2d34ff' }}
            >
              <p className="infomation_title" style={{ color: currentTheme === 'light' ? '#2a2d34ff' : '#fff' }}>
                {intl.formatMessage({ id: 'overview.usageInstructions' })}
              </p>
              <div className="infomation_box">
                <div className="infomation_single">
                  <div className="infomation_single_top">
                    <div className="infomation_single_top_number" style={{ color: '#3f66f5' }}>
                      1
                    </div>
                    <div className="infomation_single_top_text" style={{ color: '#3f66f5' }}>
                      {intl.formatMessage({ id: 'overview.step1Title' })}
                    </div>
                    <div className="infomation_single_top_bord"></div>
                  </div>
                  <div className="infomation_single_bottom">
                    <p className="infomation_single_bottom_text">
                      {intl.formatMessage({ id: 'overview.step1Description' })}
                    </p>
                  </div>
                </div>
                <div className="infomation_single">
                  <div className="infomation_single_top">
                    <div className="infomation_single_top_number" style={{ color: '#3f66f5' }}>
                      2
                    </div>
                    <div className="infomation_single_top_text" style={{ color: '#3f66f5' }}>
                      {intl.formatMessage({ id: 'overview.step2Title' })}
                    </div>
                    <div className="infomation_single_top_bord"></div>
                  </div>
                  <div className="infomation_single_bottom">
                    <p className="infomation_single_bottom_text">
                      {intl.formatMessage({ id: 'overview.step2Description' })}
                    </p>
                  </div>
                </div>
                <div className="infomation_single">
                  <div className="infomation_single_top">
                    <div className="infomation_single_top_number" style={{ color: '#3f66f5' }}>
                      3
                    </div>
                    <div className="infomation_single_top_text" style={{ color: '#3f66f5' }}>
                      {intl.formatMessage({ id: 'overview.step3Title' })}
                    </div>
                    <div className="infomation_single_top_bord"></div>
                  </div>
                  <div className="infomation_single_bottom">
                    <p className="infomation_single_bottom_text">
                      {intl.formatMessage({ id: 'overview.step3Description' })}
                    </p>
                  </div>
                </div>
                <div className="infomation_single" style={{ marginRight: '0px' }}>
                  <div className="infomation_single_top">
                    <div className="infomation_single_top_number" style={{ color: '#3f66f5' }}>
                      4
                    </div>
                    <div className="infomation_single_top_text" style={{ color: '#3f66f5' }}>
                      {intl.formatMessage({ id: 'overview.step4Title' })}
                    </div>
                    <div className="infomation_single_top_bord"></div>
                  </div>
                  <div className="infomation_single_bottom">
                    <p className="infomation_single_bottom_text">
                      {intl.formatMessage({ id: 'overview.step4Description' })}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <ResourceWorkload />
          <ResourceUsed />
          <ResourceManage />
        </div>
      )}
    </div>
  );
}
