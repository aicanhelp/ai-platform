/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import workloadDarkIcon from '@/assets/images/menu/workloadDarkIcon.png';
import workloadSelectedIcon from '@/assets/images/menu/workloadSelectedIcon.png';
import workloadIcon from '@/assets/images/menu/workloadIcon.png';
const workloadFilled = (theme) => (
  <img src={theme === 'light' ? workloadSelectedIcon : workloadDarkIcon} />
);

const workloadOutlined = (_theme) => (
  <img src={workloadIcon} />
);

export default function WorkloadIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? workloadFilled(theme) : workloadOutlined(theme)}
    </div>
  );
}