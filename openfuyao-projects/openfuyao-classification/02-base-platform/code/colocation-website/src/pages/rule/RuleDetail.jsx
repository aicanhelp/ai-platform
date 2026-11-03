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
import LabelSelectModel from '../rule/LabelSelectModel';
import ExactMatchModel from '../rule/ExactMatchModel';
import { useEffect, useState, useCallback } from 'openinula';
import { useIntl } from 'inula-intl';
import { useParams, Link, useHistory } from 'inula-router';

export default function RuleDetail() {
  const intl = useIntl();
  const [ruleDetailTabKey, setRuleDetailTabKey] = useState('1');
  const param = useParams();
  const [labelShow, setLabelShow] = useState(false);
  const [exactShow, setExactShow] = useState(false);

  useEffect(() => {
    if (param.editWay === 'exact') {
      setRuleDetailTabKey('2');
      setLabelShow(false);
      setExactShow(true);
    } else {
      setRuleDetailTabKey('1');
      setExactShow(false);
      setLabelShow(true);
    }
  }, []);

  return (
    <div className="child_content">
      <BreadCrumbCom
        className="create_bread"
        items={[
          { title: intl.formatMessage({ id: 'colocation.title' }), path: `/ColocationRulesManagement` },
          { title: intl.formatMessage({ id: 'rule.management' }), path: `/` },
          {
            title: `${param.onlyRead ? intl.formatMessage({ id: 'rule.detail' }) : intl.formatMessage({ id: 'common.edit' })}`,
            path: `/detail`,
          },
        ]}
      />
      {ruleDetailTabKey === '1' ? <LabelSelectModel /> : <ExactMatchModel />}
    </div>
  );
}
