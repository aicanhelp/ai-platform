/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import nodeDarkIcon from '@/assets/images/menu/nodeDarkIcon.png';
import nodeSelectedIcon from '@/assets/images/menu/nodeSelectedIcon.png';
import nodeIcon from '@/assets/images/menu/nodeIcon.png';
const nodeFilled = (theme) => (
  <img src={theme === 'light' ? nodeSelectedIcon : nodeDarkIcon} />
);

const nodeOutlined = (_theme) => (
  <img src={nodeIcon} />
);

export default function NodeIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? nodeFilled(theme) : nodeOutlined(theme)}
    </div>
  );
}