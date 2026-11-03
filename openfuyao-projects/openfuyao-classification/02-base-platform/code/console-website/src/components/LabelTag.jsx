/**
 *  Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  openFuyao is licensed under Mulan PSL v2.
 *  You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  You may obtain a copy of Mulan PSL v2 at:
  
 *       http://license.coscl.org.cn/MulanPSL2
  
 *   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *   EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *   MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *   See the Mulan PSL v2 for more details.
 */

import { Tag } from 'antd';
import { useStore } from 'openinula';

export default function LabelTag({ labelKey, labelValue }) {
  const keyColorTable = {light: '#cce6ff', dark: '#234c9e'};
  const valueColorTable = {light: '#e5e5e5', dark: '#444444'};
  const themeStore = useStore('theme');

  // 纯派生值，直接随 store 渲染，无需 state/effect 同步
  const keyColor = keyColorTable[themeStore.$s.theme] || '#cce6ff';
  const valueColor = valueColorTable[themeStore.$s.theme] || '#e5e5e5';

  return (
    <div className="label_item">
      <Tag color={keyColor} className="label_tag_key">
        {labelKey}
      </Tag>
      <Tag color={valueColor} className="label_tag_value">
        {labelValue}
      </Tag>
    </div>
  );
}

