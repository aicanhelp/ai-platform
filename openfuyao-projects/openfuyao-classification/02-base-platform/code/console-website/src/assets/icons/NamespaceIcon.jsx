/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import namespaceDarkIcon from '@/assets/images/menu/namespaceDarkIcon.png';
import namespaceSelectedIcon from '@/assets/images/menu/namespaceSelectedIcon.png';
import namespaceIcon from '@/assets/images/menu/namespaceIcon.png';
const namespaceFilled = (theme) => (
  <img src={theme === 'light' ? namespaceSelectedIcon : namespaceDarkIcon} />
);

const namespaceOutlined = (_theme) => (
  <img src={namespaceIcon} />
);
export default function NamespaceIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? namespaceFilled(theme) : namespaceOutlined(theme)}
    </div>
  );
}