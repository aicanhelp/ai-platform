/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import colocationDarkIcon from '@/assets/images/menu/colocationDarkIcon.png';
import colocationSelectedIcon from '@/assets/images/menu/colocationSelectedIcon.png';
import colocationIcon from '@/assets/images/menu/colocationIcon.png';
const colocationFilled = (theme) => (
  <img src={theme === 'light' ? colocationSelectedIcon : colocationDarkIcon} />
);

const colocationOutlined = (_theme) => (
  <img src={colocationIcon} />
);

export default function ColocationIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? colocationFilled(theme) : colocationOutlined(theme)}
    </div>
  );
}