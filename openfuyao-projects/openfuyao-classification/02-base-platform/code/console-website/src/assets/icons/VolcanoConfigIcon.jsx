/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import volcanoConfigDarkIcon from '@/assets/images/menu/volcanoConfigDarkIcon.png';
import volcanoConfigSelectedIcon from '@/assets/images/menu/volcanoConfigSelectedIcon.png';
import volcanoConfigIcon from '@/assets/images/menu/volcanoConfigIcon.png';
const volcanoConfigFilled = (theme) => (
  <img src={theme === 'light' ? volcanoConfigSelectedIcon : volcanoConfigDarkIcon} />
);

const volcanoConfigOutlined = (_theme) => (
  <img src={volcanoConfigIcon} />
);

export default function VolcanoConfigIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? volcanoConfigFilled(theme) : volcanoConfigOutlined(theme)}
    </div>
  );
}