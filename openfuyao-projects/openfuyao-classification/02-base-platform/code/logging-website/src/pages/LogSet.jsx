/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import '@/styles/log.less';
import { useIntl } from 'inula-intl';
import { Tabs } from 'antd';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import LogAlerts from '@/pages/LogAlerts';
import LogCollecting from '@/pages/LogCollecting';
import LogJob from '@/pages/LogJob';
import '@/styles/dark.less';

export default function LogSet() {
  const intl = useIntl();
  const items = [
    {
      key: '1',
      label: intl.formatMessage({ id: 'logging.collectionSource' }),
      children: <LogCollecting />,
    },
    {
      key: '2',
      label: intl.formatMessage({ id: 'logging.collectionTask' }),
      children: <LogJob />,
    },
    {
      key: '3',
      label: intl.formatMessage({ id: 'logging.alertRules' }),
      children: <LogAlerts />,
    },
  ];
  const onChange = (key) => {};
  return (
    <div className="logSetContainer child_content">
      <BreadCrumbCom items={[{ title: intl.formatMessage({ id: 'logSet.breadcrumb' }), path: `/logSet` }]} />
      <Tabs defaultActiveKey="1" items={items} onChange={onChange} />
    </div>
  );
}
