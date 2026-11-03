/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import BreadCrumbCom from '../components/BreadCrumbCom';

import { Tabs } from 'antd';
import PodPage from '../pages/workload/PodPage';
import DeploymentPage from '../pages/workload/DeploymentPage';
import StatefulSetPage from '../pages/workload/StatefulSetPage';
import DaemonSetPage from '../pages/workload/DaemonSetPage';
import JobPage from '../pages/workload/JobPage';
import CronJobPage from '../pages/workload/CronJobPage';
import { useState } from 'openinula';
import { useIntl } from 'inula-intl';
import '../styles/ColocationWorkloadManagementPage.less';

export default function ColocationWorkloadManagementPage() {
  const intl = useIntl();
  const [helmDetailTabKey, setHelmDetailTabKey] = useState('1');
  const items = [
    {
      key: '1',
      label: 'Pod',
      children: <PodPage />,
    },
    {
      key: '2',
      label: 'Deployment',
      children: <DeploymentPage />,
    },
    {
      key: '3',
      label: 'StatefulSet',
      children: <StatefulSetPage />,
    },
    {
      key: '5',
      label: 'Job',
      children: <JobPage />,
    },
    {
      key: '6',
      label: 'CronJob',
      children: <CronJobPage />,
    },
  ];

  const handleSetHelmDetailTabKey = (key) => {
    setHelmDetailTabKey(key);
  };

  return (
    <div className="child_content">
      <BreadCrumbCom
        className="create_bread"
        items={[
          { title: intl.formatMessage({ id: 'colocation.title' }), path: `/ColocationWorkloadManagement` },
          { title: intl.formatMessage({ id: 'workload.management' }), path: `/` },
        ]}
      />
      <Tabs
        items={items}
        onChange={handleSetHelmDetailTabKey}
        activeKey={helmDetailTabKey}
        destroyInactiveTabPane={true}
      ></Tabs>
    </div>
  );
}
