/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import storageDarkIcon from '@/assets/images/menu/storageDarkIcon.png';
import storageSelectedIcon from '@/assets/images/menu/storageSelectedIcon.png';
import storageOutlinedIcon from '@/assets/images/menu/storageIcon.png';
const storageFilled = (theme) => (
  <img src={theme === 'light' ? storageSelectedIcon : storageDarkIcon} />
);

const storageOutlined = (_theme) => (
  <img src={storageOutlinedIcon} />
);

export default function StorageIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? storageFilled(theme) : storageOutlined(theme)}
    </div>
  );
}
