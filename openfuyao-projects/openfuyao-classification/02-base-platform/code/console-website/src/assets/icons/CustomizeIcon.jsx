/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import customizeDarkIcon from '@/assets/images/menu/customizeDarkIcon.png';
import customizeSelectedIcon from '@/assets/images/menu/customizeSelectedIcon.png';
import customizeIcon from '@/assets/images/menu/customizeIcon.png';
const customizeFilled = (theme) => (
  <img src={theme === 'light' ? customizeSelectedIcon : customizeDarkIcon} />
);

const customizeOutlined = (_theme) => (
  <img src={customizeIcon} />
);

export default function CustomizeIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? customizeFilled(theme) : customizeOutlined(theme)}
    </div>
  );
}