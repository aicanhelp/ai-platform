/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import BreadCrumbCom from '../../components/BreadCrumbCom';
import { containerRouterPrefix } from '../../common/constants';
import { Tabs } from 'antd';
import CreateLabelSelectModel from './CreateLabelSelectModel';
import CreateExactMatchModel from './CreateExactMatchModel';
import { useEffect, useState, useCallback } from 'openinula';
import { useIntl } from 'inula-intl';
import { useParams, Link, useHistory } from 'inula-router';

export default function RuleDetail() {
  const intl = useIntl();
  const param = useParams();
  const [ruleDetailTabKey, setRuleDetailTabKey] = useState('1');
  const items = [
    {
      key: '1',
      label: intl.formatMessage({ id: 'rule.labelSelectorMode' }),
      children: <CreateLabelSelectModel tabKey={ruleDetailTabKey} />,
    },
    {
      key: '2',
      label: intl.formatMessage({ id: 'rule.exactMatchMode' }),
      children: <CreateExactMatchModel tabKey={ruleDetailTabKey} />,
    },
  ];

  const handleSetRuleDetailTabKey = (key) => {
    setRuleDetailTabKey(key);
  };

  useEffect(() => {
    if (param.way === 'exact') {
      setRuleDetailTabKey('2');
    } else {
      setRuleDetailTabKey('1');
    }
  }, []);

  return (
    <div className="child_content">
      <BreadCrumbCom
        className="create_bread"
        items={[
          { title: intl.formatMessage({ id: 'colocation.title' }), path: `/ColocationRulesManagement` },
          { title: intl.formatMessage({ id: 'rule.management' }), path: `/` },
          { title: intl.formatMessage({ id: 'common.create' }), path: `/ColocationRulesManagement/create` },
        ]}
      />
      <Tabs items={items} onChange={handleSetRuleDetailTabKey} activeKey={ruleDetailTabKey}></Tabs>
    </div>
  );
}
