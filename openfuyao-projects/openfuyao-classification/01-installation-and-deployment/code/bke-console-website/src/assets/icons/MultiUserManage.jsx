/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
const multiUserManageFilled = (theme) => (
  <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 20 20">
    <defs><style>.cls-1{{ fill: 'none' }}.cls-2{{ fill: 'url(#未命名的渐变_2)' }}</style>
      <linearGradient id="未命名的渐变_2" x1="1.27" y1="10.15" x2="18.36" y2="10.15" gradientUnits="userSpaceOnUse">
        <stop offset="0" stop-color="#477dd8" /><stop offset="1" stop-color="#234c9e" />
      </linearGradient></defs><g id="图层_2" data-name="图层 2"><g id="图层_1-2" data-name="图层 1">
        <rect class="cls-1" fill="none" width="20" height="20" />
        <path class="cls-2" fill={`${theme === 'light' ? 'url(#未命名的渐变_2)' : 'white'}`} d="M18.07,17l-1.22-1.22.76-.77a1,1,0,1,0-1.41-1.42l-.77.77-1-1,2.46-2.47a1,1,0,0,0,0-1.41,1,1,0,0,0-1.42,0L13,11.92l-1.09-1.09a5.85,5.85,0,1,0-1.41,1.42l6.15,6.15a1,1,0,0,0,1.41,0A1,1,0,0,0,18.07,17ZM8.69,9A2.21,2.21,0,0,1,5.56,5.91a2.21,2.21,0,0,1,3.13,0A2.22,2.22,0,0,1,8.69,9Z" />
      </g></g>
  </svg>
);

const multiUserManageOutlined = (theme) => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><defs><style>.cls-1{{ fill: 'none' }}.cls-2{{ fill: '#89939b' }}</style>
  </defs><g id="图层_2" data-name="图层 2"><g id="图层_1-2" data-name="图层 1">
    <rect class="cls-1" fill="none" width="20" height="20" />
    <path class="cls-2" fill="#89939b" d="M7.12,13.33a5.86,5.86,0,1,1,4.15-10h0a5.86,5.86,0,0,1-4.15,10Zm0-9.71A3.85,3.85,0,0,0,4.4,10.2a3.94,3.94,0,0,0,5.45,0A3.85,3.85,0,0,0,7.12,3.62Z" />
    <rect class="cls-2" fill="#89939b" x="13.11" y="8.84" width="2" height="11.2" rx="1" transform="translate(-6.08 14.21) rotate(-45)" />
    <rect class="cls-2" fill="#89939b" x="11.71" y="10.6" width="6.07" height="2" rx="1" transform="translate(-3.89 13.82) rotate(-45)" />
    <rect class="cls-2" fill="#89939b" x="14.48" y="13.87" width="3.67" height="2" rx="1" transform="translate(-5.74 15.89) rotate(-45)" />
  </g></g>
  </svg>
);

export default function MultiUserManageIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? multiUserManageFilled(theme) : multiUserManageOutlined(theme)}
    </div>
  );
}