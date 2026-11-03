/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { useParams } from 'inula-router';
import { useState } from 'openinula';
import { useIntl } from 'inula-intl';
import '../../styles/colocationRuleManagementPage.less';
import CreateModel from './CreateModel';

export default function CreateLabelSelectModel({ tabKey }) {
  const intl = useIntl();
  const param = useParams();
  const [tabKeyType, setTabKeyType] = useState(tabKey);
  const [selectType, setSelectType] = useState(param.type);

  return (
    <div className="child_content">
      <CreateModel tabKeyType={tabKeyType} stateType={selectType} />
    </div>
  );
}
