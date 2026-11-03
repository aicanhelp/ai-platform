/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import '../styles/emptyData.less';
import { useIntl } from 'inula-intl';
import NoContent from '../assets/images/node/noContent.png';

export default function EmptyData({ className = '' }) {
  const intl = useIntl();
  return (
    <div className={`flexBox ${className}`}>
      <img src={NoContent} />
      <p>{intl.formatMessage({ id: 'empty.noData' })}</p>
    </div>
  );
}
