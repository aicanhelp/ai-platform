/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import extensionDarkIcon from '@/assets/images/menu/extensionDarkIcon.png';
import extensionSelectedIcon from '@/assets/images/menu/extensionSelectedIcon.png';
import extensionIcon from '@/assets/images/menu/extensionIcon.png';
const extensionFilled = (theme) => (
  <img src={theme === 'light' ? extensionSelectedIcon : extensionDarkIcon} />
);

const extensionOutlined = (_theme) => (
  <img src={extensionIcon} />
);

export default function ExtensionIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? extensionFilled(theme) : extensionOutlined(theme)}
    </div>
  );
}