/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import networkDarkIcon from '@/assets/images/menu/networkDarkIcon.png';
import networkSelectedIcon from '@/assets/images/menu/networkSelectedIcon.png';
import networkIcon from '@/assets/images/menu/networkIcon.png';
const networkFilled = (theme) => (
  <img src={theme === 'light' ? networkSelectedIcon : networkDarkIcon} />
);

const networkOutlined = (_theme) => (
  <img src={networkIcon} />
);

export default function NetworkIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? networkFilled(theme) : networkOutlined(theme)}
    </div>
  );
}