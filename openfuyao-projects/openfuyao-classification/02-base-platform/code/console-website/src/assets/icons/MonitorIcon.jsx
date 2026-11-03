/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import monitorDarkIcon from '@/assets/images/menu/monitorDarkIcon.png';
import monitorSelectedIcon from '@/assets/images/menu/monitorSelectedIcon.png';
import monitorIcon from '@/assets/images/menu/monitorIcon.png';
const monitorFilled = (theme) => (
  <img src={theme === 'light' ? monitorSelectedIcon : monitorDarkIcon} />
);

const monitorOutlined = (_theme) => (
  <img src={monitorIcon} />
);

export default function MonitorIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? monitorFilled(theme) : monitorOutlined(theme)}
    </div>
  );
}