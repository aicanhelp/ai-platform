/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { useStore } from 'openinula';
import { InfoCircleFilled } from '@ant-design/icons';
import '@/styles/pages/helm.less';

export default function ToolTipComponent({ children, className }) {
  const themeStore = useStore('theme');
  return (
    <div className={`${className} prompt`} style={{ backgroundColor: themeStore.$s.theme !== 'light' && 'rgba(119,174,247, 0.12)', border: '1px solid rgba(119,174,247, 0.50)', color: themeStore.$s.theme !== 'light' && '#fff' }}>
      <div style={{ display: 'flex' }}>
        <InfoCircleFilled className='prompt-icon' />
        <div style={{ flex: 1 }}>
          {children}
        </div>
      </div>
    </div>
  );
}