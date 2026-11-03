/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import misManagementDarkIcon from '@/assets/images/menu/misManagementDarkIcon.png';
import misManagementSelectedIcon from '@/assets/images/menu/misManagementSelectedIcon.png';
import misManagementIcon from '@/assets/images/menu/misManagementIcon.png';
const misManagementFilled = (theme) => (
  <img src={theme === 'light' ? misManagementSelectedIcon : misManagementDarkIcon} />
);

const misManagementOutlined = (_theme) => (
  <img src={misManagementIcon} />
);

export default function MisManagementIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? misManagementFilled(theme) : misManagementOutlined(theme)}
    </div>
  );
}