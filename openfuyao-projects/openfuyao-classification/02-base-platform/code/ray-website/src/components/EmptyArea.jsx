/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Fragment } from 'openinula';
import OverviewEcharts from '@/components/OverviewEcharts.jsx';
import { useIntl } from 'inula-intl';

export default function EmptyArea({ options }) {
  const intl = useIntl();
  return (
    <Fragment>
      <div className="empty_text_pop-area_sub">{intl.formatMessage({ id: 'common.noData' })}</div>
      <OverviewEcharts overOption={options} />
    </Fragment>
  );
}
