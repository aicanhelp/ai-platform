/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { QuestionCircleOutlined } from '@ant-design/icons';
import { useHistory } from 'inula-router';
import { containerRouterPrefix } from '@/constant.js';

export default function MarketCategoryItem({ icon, categoryTitle, tipIcon, value }) {
  const history = useHistory();

  const handleClick = () => {
    history.push(`/${containerRouterPrefix}/appMarket/marketCategory/${encodeURIComponent(value)}`);
  };
  return (
    <div className='category_label' onClick={handleClick}>
      <div className='category_icon'><img src={icon} /></div>
      <div className='category_text'>{categoryTitle}</div>
      <div>{tipIcon}</div>
      {tipIcon && <div style={{ marginLeft: '8px' }}><QuestionCircleOutlined style={{ color: '#89939bff' }} /></div>}
    </div>
  );
}