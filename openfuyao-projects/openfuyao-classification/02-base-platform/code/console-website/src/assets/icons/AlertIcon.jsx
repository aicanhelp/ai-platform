/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import alertDarkIcon from '@/assets/images/menu/alertDarkIcon.png';
import alertSelectedIcon from '@/assets/images/menu/alertSelectedIcon.png';
import alertIcon from '@/assets/images/menu/alertIcon.png';
const alertFilled = (theme) => (
  <img src={theme === 'light' ? alertSelectedIcon : alertDarkIcon} />
);

const alertOutlined = (_theme) => (
  <img src={alertIcon} />
);

export default function AlertIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? alertFilled(theme) : alertOutlined(theme)}
    </div>
  );
}