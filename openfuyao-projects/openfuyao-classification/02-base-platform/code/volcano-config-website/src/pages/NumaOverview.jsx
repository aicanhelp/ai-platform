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
import { useHistory } from 'inula-router';
import '@/styles/numaOverview.less';
import { QuestionCircleOutlined } from '@ant-design/icons';
import { Drawer } from 'antd';
import useLocalStorage from '@/hooks/useLocalStorage';
import { useEffect, useState } from 'openinula';
import { useIntl } from 'inula-intl';
import overviewStep1 from '@/assets/images/overviewStep1.png';
import overviewStep2 from '@/assets/images/overviewStep2.png';
import overviewStep3 from '@/assets/images/overviewStep3.png';
import overviewStep4 from '@/assets/images/overviewStep4.png';
import overviewStepDark1 from '@/assets/images/overviewStepDark1.png';
import overviewStepDark2 from '@/assets/images/overviewStepDark2.png';
import overviewStepDark3 from '@/assets/images/overviewStepDark3.png';
import overviewStepDark4 from '@/assets/images/overviewStepDark4.png';
import { getEnvironmnetPreparation, getWorkload, getNewEnvPrepare } from '@/assets/md';
import { skipSendMainMsg } from '@/tools/utils';

import PriorityPreemption from './SchedulingPolicyConfiguration/PriorityPreemption';

export default function NumaOverview() {
  const intl = useIntl();
  const [currentTheme, setCurrentTheme] = useLocalStorage('theme', 'light');
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
      md: getNewEnvPrepare(),
    },
    {
      index: 2,
      key: 'AffinityPolicyConfiguration',
      img: currentTheme === 'light' ? overviewStep2 : overviewStepDark2,
      title: intl.formatMessage({ id: 'overview.affinityConfig' }),
      desc: intl.formatMessage({ id: 'overview.affinityConfigDesc' }),
      link: intl.formatMessage({ id: 'overview.configAffinity' }),
      path: '/AffinityPolicyConfiguration',
      href: '/container_platform/scheduling/AffinityPolicyConfiguration',
      showDrawers: false,
      md: '',
    },
    {
      index: 3,
      key: 'pod',
      img: currentTheme === 'light' ? overviewStep3 : overviewStepDark3,
      title: intl.formatMessage({ id: 'overview.workloadDeploy' }),
      desc: intl.formatMessage({ id: 'overview.workloadDeployDesc' }),
      link: intl.formatMessage({ id: 'overview.deployWorkload' }),
      path: '/workload/pod',
      href: '/container_platform/workload/pod',
      showDrawers: true,
      md: getWorkload(),
    },
    {
      index: 4,
      key: 'numaMonitor',
      img: currentTheme === 'light' ? overviewStep4 : overviewStepDark4,
      title: intl.formatMessage({ id: 'overview.numaMonitor' }),
      desc: intl.formatMessage({ id: 'overview.numaMonitorDesc' }),
      link: intl.formatMessage({ id: 'overview.viewNumaMonitor' }),
      path: '/numaMonitor',
      href: '/container_platform/scheduling/numaMonitor',
      showDrawers: false,
      md: '',
    },
  ];
  const goOtherPage = (e, data) => {
    skipSendMainMsg(e, data.key, data.path, history, !data.showDrawers);
  };
  useEffect(() => {}, []);
  return (
    <div id="numa_overview" className="numa_overview">
      <BreadCrumbCom
        items={[
          { title: intl.formatMessage({ id: 'common.numaAffinityScheduling' }), path: '/scheduling/NumaOverview' },
          { title: intl.formatMessage({ id: 'overview.overviewTitle' }), path: '/' },
        ]}
      />
      <div className="numa_overview_content">
        <div className="numa_overview_content_banner">
          <p className="numa_overview_content_banner_title">
            {intl.formatMessage({ id: 'common.numaAffinityScheduling' })}
          </p>
          <p className="numa_overview_content_banner_desc">{intl.formatMessage({ id: 'overview.bannerDesc' })}</p>
        </div>
        <div className="numa_overview_content_process">
          <div className="numa_overview_content_process_all">
            {overviewData.map((item) => (
              <div className="numa_overview_content_process_single">
                <img src={item.img} alt="" className="numa_overview_content_process_single_img" />
                <div>
                  <div className="numa_overview_content_process_single_rightTop">
                    <p className="numa_overview_content_process_single_rightTop_num">{item.index}</p>
                    <p className="numa_overview_content_process_single_rightTop_title">{item.title}</p>
                    <QuestionCircleOutlined
                      className="question-icon-hover"
                      onClick={() => showDrawer(item.md, item.className)}
                      style={{ marginLeft: '7px', display: item.showDrawers ? 'flex' : 'none' }}
                    />
                  </div>
                  <p className="numa_overview_content_process_single_rightBottom">
                    {item.desc}
                    <a
                      href={item.href}
                      style={{ display: item.index === 3 && isIndependence ? 'none' : 'inline-block' }}
                      onClick={(e) => goOtherPage(e, item)}
                      className="numa_overview_content_process_single_rightBottom_link"
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
            getContainer={() => document.getElementById('numa_overview')}
          >
            <PriorityPreemption className={className} onClose={onClose} data={explanatoryInfo} />
          </Drawer>
        </div>
      </div>
    </div>
  );
}
