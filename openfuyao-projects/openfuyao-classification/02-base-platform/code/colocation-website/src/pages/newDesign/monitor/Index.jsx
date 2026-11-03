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
import { Tabs } from 'antd';
import { useHistory } from 'inula-router';
import { useIntl } from 'inula-intl';
import { useState, useEffect } from 'openinula';
import { solveDataAreaNumPalettePlus } from '@/tools/utils';
import ClusterMonitor from '@/pages/newDesign/monitor/ClusterMonitor';
import NodeMonitor from '@/pages/newDesign/monitor/NodeMonitor';
import '@/styles/monitor.less';
export default function ColocationMonitorPage() {
  const intl = useIntl();
  const [intervalType, setIntervalType] = useState('');
  const [rangeColorList, setRangeColorList] = useState([]); // 生成随机颜色
  const history = useHistory();
  const [hiddenPage, setHiddenPage] = useState(false);
  const [monitorTabKey, setMonitorTabKey] = useState('cluster');
  const handleSetMonitorTabKey = (key) => {
    setIntervalType(key);
    setMonitorTabKey(key);
  };
  const items = [
    {
      key: 'cluster',
      label: intl.formatMessage({ id: 'monitor.clusterLevelColocationMonitor' }),
      children: <ClusterMonitor propsType={intervalType} propsColor={rangeColorList} />,
    },
    {
      key: 'node',
      label: intl.formatMessage({ id: 'monitor.nodeLevelColocationMonitor' }),
      children: <NodeMonitor propsType={intervalType} propsColor={rangeColorList} />,
    },
  ];

  const handlePassMessage = (event) => {
    if (event.data.type === 'colocationOverviewPageDestory') {
      if (event.data.isDestory) {
        setHiddenPage(true);
        // 清理定时器
        const colocationTime1ID = localStorage.getItem('colocationTime1Id') || '';
        const colocationTime2ID = localStorage.getItem('colocationTime2Id') || '';

        clearTimeout(colocationTime1ID);
        clearTimeout(colocationTime2ID);

        localStorage.removeItem('colocationTime1Id');
        localStorage.removeItem('colocationTime2Id');
      } else {
        setHiddenPage(false);
      }
    }
  };

  useEffect(() => {
    window.addEventListener('message', handlePassMessage);
    return () => window.removeEventListener('message', handlePassMessage);
  }, []);

  useEffect(() => {
    const rangeBirthColorList = solveDataAreaNumPalettePlus(1000); // 颜色区间根据类别生成
    setRangeColorList(rangeBirthColorList);
  }, []);

  return (
    <div className="child_content">
      <BreadCrumbCom
        className="create_bread noshadow"
        items={[
          { title: intl.formatMessage({ id: 'monitor.colocation' }), path: `/ColocationMonitor` },
          { title: intl.formatMessage({ id: 'monitor.monitor' }), path: `/` },
        ]}
      />
      {!hiddenPage && (
        <Tabs items={items} onChange={handleSetMonitorTabKey} activeKey={monitorTabKey} destroyOnHidden={true}></Tabs>
      )}
    </div>
  );
}
