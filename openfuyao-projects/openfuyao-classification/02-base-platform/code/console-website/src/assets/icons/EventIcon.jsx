/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import eventDarkIcon from '@/assets/images/menu/eventDarkIcon.png';
import eventSelectedIcon from '@/assets/images/menu/eventSelectedIcon.png';
import eventIcon from '@/assets/images/menu/eventIcon.png';
const eventFilled = (theme) => (
  <img src={theme === 'light' ? eventSelectedIcon : eventDarkIcon} />
);

const eventOutlined = (_theme) => (
  <img src={eventIcon} />
);

export default function EventIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? eventFilled(theme) : eventOutlined(theme)}
    </div>
  );
}