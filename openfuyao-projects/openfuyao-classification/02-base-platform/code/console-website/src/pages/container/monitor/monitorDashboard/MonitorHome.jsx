/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { useEffect, useState } from 'openinula';
import { useTranslation } from 'react-i18next';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import ComponentMonitor from '@/pages/container/monitor/monitorDashboard/ComponentMonitor';
import ResourceMonitor from '@/pages/container/monitor/monitorDashboard/ResourceMonitor';
import { Button, Tabs, FloatButton } from 'antd';
import { containerRouterPrefix } from '@/constant';
import '@/styles/pages/monitor.less';
import { useHistory } from 'inula-router';
import { solveDataAreaNumPalettePlus } from '@/tools/utils';
import { ArrowUpOutlined, ArrowDownOutlined } from '@ant-design/icons';

export default function MonitorHomePage() {
  const { t } = useTranslation('monitoring');
  const [intervalType, setIntervalType] = useState('');
  const [rangeColorList, setRangeColorList] = useState([]); // 生成随机颜色
  const history = useHistory();
  const [monitorTabKey, setMonitorTabKey] = useState('resource');
  const handleSetMonitorTabKey = (key) => {
    setIntervalType(key);
    setMonitorTabKey(key);
  };

  const items = [
    {
      key: 'resource',
      label: t('tabs.resourceMonitor'),
      children: <ResourceMonitor propsType={intervalType} propsColor={rangeColorList} />,
    },
    {
      key: 'component',
      label: t('tabs.componentMonitor'),
      children: <ComponentMonitor propsType={intervalType} propsColor={rangeColorList} />,
    },
  ];

  const toTop = () => {
    const target = document.querySelector('.container_content');
    target.scrollTo({ top: 0, left: 0, behavior: 'smooth' });
  };

  const toDown = () => {
    const target = document.querySelector('.container_content');
    target.scrollTo({ top: target.scrollHeight - target.clientHeight, left: 0, behavior: 'smooth' });
  };

  useEffect(() => {
    const rangeBirthColorList = solveDataAreaNumPalettePlus(1000); // 颜色区间根据类别生成
    setRangeColorList(rangeBirthColorList);
  }, []);

  return <div className="child_content">
    <BreadCrumbCom className="create_bread" items={[
      { title: t('module.title'), path: `/${containerRouterPrefix}/monitor/monitorDashboard`, disabled: true },
      { title: t('module.dashboard'), path: '/' },
    ]} />
    <div className='monitor_header'>
      <Button
        type="primary"
        className='primary_btn'
        style={{ width: 'auto' }}
        onClick={() => history.push(`/${containerRouterPrefix}/monitor/monitorDashboard/customize/query`)}>
        {t('actions.customizeQuery')}
      </Button>
    </div>
    <Tabs items={items} onChange={handleSetMonitorTabKey} activeKey={monitorTabKey} destroyInactiveTabPane={true}></Tabs>
    <FloatButton.Group
      className='MonitorHome float_btn_right'
      shape="square"
      style={{ right: 20 }}>
      <FloatButton icon={<ArrowUpOutlined />} onClick={toTop} />
      <FloatButton icon={<ArrowDownOutlined />} onClick={toDown} />
    </FloatButton.Group>
  </div>;
}