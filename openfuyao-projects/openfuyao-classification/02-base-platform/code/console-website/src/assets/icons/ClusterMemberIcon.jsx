/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import clusterMemberDarkIcon from '@/assets/images/menu/clusterMemberDarkIcon.png';
import clusterMemberSelectedIcon from '@/assets/images/menu/clusterMemberSelectedIcon.png';
import clusterMembertIcon from '@/assets/images/menu/clusterMemberIcon.png';
const clusterMemberFilled = (theme) => (
  <img src={theme === 'light' ? clusterMemberSelectedIcon : clusterMemberDarkIcon} />
);

const clusterMemberOutlined = (_theme) => (
  <img src={clusterMembertIcon} />
);
export default function ClusterMemberIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? clusterMemberFilled(theme) : clusterMemberOutlined(theme)}
    </div>
  );
}