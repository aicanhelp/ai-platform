/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import applicationDarkIcon from '@/assets/images/menu/applicationDarkIcon.png';
import applicationSelectedIcon from '@/assets/images/menu/applicationSelectedIcon.png';
import applicationIcon from '@/assets/images/menu/applicationIcon.png';
const applicationFilled = (theme) => (
  <img src={theme === 'light' ? applicationSelectedIcon : applicationDarkIcon} />
);

const applicationOutlined = (_theme) => (
  <img src={applicationIcon} />
);

export default function ApplicationIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? applicationFilled(theme) : applicationOutlined(theme)}
    </div>
  );
}