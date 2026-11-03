/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Popover } from 'antd';
import { Fragment } from 'openinula';
import { StockOutlined } from '@ant-design/icons';
import '@/styles/components/dataDisplayList.less';
import { useIntl } from 'inula-intl';
export default function DataDisplayList({ data = [] }) {
  const intl = useIntl();
  return (
    <Fragment>
      <Popover
        placement="bottomLeft"
        trigger="click"
        getPopupContainer={() => document.querySelector('.legend_com')}
        content={
          data.length ? (
            <div className="legend_pop">
              {data.map((item) => (
                <div className="each_line">
                  <div style={{ background: item.color }}></div>
                  <div>{item.name}</div>
                </div>
              ))}
            </div>
          ) : (
            <p className="empty_text_pop">{intl.formatMessage({ id: 'common.noData' })}</p>
          )
        }
      >
        <div className="legend_com">
          <StockOutlined />
          <p>{intl.formatMessage({ id: 'common.legend' })}</p>
        </div>
      </Popover>
    </Fragment>
  );
}
