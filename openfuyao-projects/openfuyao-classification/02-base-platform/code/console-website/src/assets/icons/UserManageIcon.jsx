/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import userManageDarkIcon from '@/assets/images/menu/userManageDarkIcon.png';
import userManageSelectedIcon from '@/assets/images/menu/userManageSelectedIcon.png';
import userManageIcon from '@/assets/images/menu/userManageIcon.png';
const userManageFilled = (theme) => (
  <img src={theme === 'light' ? userManageSelectedIcon : userManageDarkIcon} />
);

const userManageOutlined = (_theme) => (
  <img src={userManageIcon} />
);

export default function UserManageIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? userManageFilled(theme) : userManageOutlined(theme)}
    </div>
  );
}