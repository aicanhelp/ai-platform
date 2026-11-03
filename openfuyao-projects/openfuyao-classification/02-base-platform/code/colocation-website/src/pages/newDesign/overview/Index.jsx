/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */

import { useIntl } from 'inula-intl';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { useHistory } from 'inula-router';
import '@/styles/cololcationOverview.less';
import { QuestionCircleOutlined } from '@ant-design/icons';
import { Drawer } from 'antd';
import useLocalStorage from '@/hooks/useLocalStorage';
import { useEffect, useState } from 'openinula';
import overviewStep1 from '@/assets/images/overviewStep1.png';
import overviewStep2 from '@/assets/images/overviewStep2.png';
import overviewStep3 from '@/assets/images/overviewStep3.png';
import overviewStep4 from '@/assets/images/overviewStep4.png';
import overviewStepDark1 from '@/assets/images/overviewStepDark1.png';
import overviewStepDark2 from '@/assets/images/overviewStepDark2.png';
import overviewStepDark3 from '@/assets/images/overviewStepDark3.png';
import overviewStepDark4 from '@/assets/images/overviewStepDark4.png';
import { getWorkload, getNewEnvPrepare } from '@/assets/md';
import { skipSendMainMsg } from '@/tools/utils';

import PriorityPreemption from '@/components/PriorityPreemption';

export default function ColocationOverviewPage() {
  const intl = useIntl();
  const [currentTheme] = useLocalStorage('theme', 'light');
  const history = useHistory();
  const [className, setClassName] = useState('');
  const [open, setOpen] = useState(false);
  const showDrawer = (data, temporyClass = '') => {
    setClassName(temporyClass);
    setExplanatoryInfo(data);
    setOpen(true);
  };
  const onClose = () => {
    setOpen(false);
  };
  const isIndependence = !window.location.href.includes('container_platform');
  const [explanatoryInfo, setExplanatoryInfo] = useState();
  const overviewData = [
    {
      index: 1,
      img: currentTheme === 'light' ? overviewStep1 : overviewStepDark1,
      title: intl.formatMessage({ id: 'overview.envPreparation' }),
      desc: intl.formatMessage({ id: 'overview.envPreparationDesc' }),
      link: '',
      path: '',
      className: 'break_code_md_body',
      showDrawers: true,
      md: getNewEnvPrepare(intl.locale),
    },
    {
      index: 2,
      key: 'configuration',
      img: currentTheme === 'light' ? overviewStep2 : overviewStepDark2,
      title: intl.formatMessage({ id: 'overview.colocationStrategyConfig' }),
      desc: intl.formatMessage({ id: 'overview.colocationStrategyConfigDesc' }),
      link: intl.formatMessage({ id: 'overview.configureColocationStrategy' }),
      path: '/ColocationConfiguration',
      href: '/container_platform/colocation/ColocationConfiguration',
      showDrawers: false,
      md: '',
    },
    {
      index: 3,
      key: 'pod',
      img: currentTheme === 'light' ? overviewStep3 : overviewStepDark3,
      title: intl.formatMessage({ id: 'overview.workloadDeployment' }),
      desc: intl.formatMessage({ id: 'overview.workloadDeploymentDesc' }),
      link: intl.formatMessage({ id: 'overview.deployWorkload' }),
      path: '/workload/pod',
      href: '/container_platform/workload/pod',
      showDrawers: true,
      md: getWorkload(intl.locale),
    },
    {
      index: 4,
      key: 'ColocationMonitor',
      img: currentTheme === 'light' ? overviewStep4 : overviewStepDark4,
      title: intl.formatMessage({ id: 'overview.colocationMonitor' }),
      desc: intl.formatMessage({ id: 'overview.colocationMonitorDesc' }),
      link: intl.formatMessage({ id: 'overview.viewColocationMonitor' }),
      path: '/ColocationMonitor',
      href: '/container_platform/colocation/ColocationMonitor',
      showDrawers: false,
      md: '',
    },
  ];
  const goOtherPage = (e, data) => {
    skipSendMainMsg(e, data.key, data.path, history, !data.showDrawers);
  };
  useEffect(() => {}, []);
  return (
    <div id="colocation_overview" className="colocation_overview">
      <BreadCrumbCom
        className="create_bread"
        items={[
          { title: intl.formatMessage({ id: 'overview.colocation' }), path: '/colocation/ColocationOverview' },
          { title: intl.formatMessage({ id: 'overview.overview' }), path: '/' },
        ]}
      />
      <div className="colocation_overview_content">
        <div className="colocation_overview_content_banner">
          <p className="colocation_overview_content_banner_title">
            {intl.formatMessage({ id: 'overview.colocation' })}
          </p>
          <p className="colocation_overview_content_banner_desc">
            {intl.formatMessage({ id: 'overview.colocationDesc' })}
          </p>
        </div>
        <div className="colocation_overview_content_process">
          <div className="colocation_overview_content_process_all">
            {overviewData.map((item) => (
              <div className="colocation_overview_content_process_single">
                <img src={item.img} alt="" className="colocation_overview_content_process_single_img" />
                <div>
                  <div className="colocation_overview_content_process_single_rightTop">
                    <p className="colocation_overview_content_process_single_rightTop_num">{item.index}</p>
                    <p className="colocation_overview_content_process_single_rightTop_title">{item.title}</p>
                    <QuestionCircleOutlined
                      className="question-icon-hover"
                      onClick={() => showDrawer(item.md, item.className)}
                      style={{ marginLeft: '7px', display: item.showDrawers ? 'flex' : 'none' }}
                    />
                  </div>
                  <p className="colocation_overview_content_process_single_rightBottom">
                    {item.desc}
                    <a
                      href={item.href}
                      style={{ display: item.index === 3 && isIndependence ? 'none' : 'inline-block' }}
                      onClick={(e) => goOtherPage(e, item)}
                      className="colocation_overview_content_process_single_rightBottom_link"
                    >
                      {item.link}
                    </a>
                  </p>
                </div>
              </div>
            ))}
          </div>
          <Drawer
            title=""
            placement="right"
            closable={false}
            onClose={onClose}
            width={600}
            open={open}
            getContainer={() => document.getElementById('colocation_overview')}
          >
            <PriorityPreemption className={className} onClose={onClose} data={explanatoryInfo} />
          </Drawer>
        </div>
      </div>
    </div>
  );
}
