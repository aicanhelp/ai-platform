/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
const resourceManageFilled = (theme) => (
  <svg id="_图层_2" data-name="图层 2" xmlns="http://www.w3.org/2000/svg" xmlns: xlink="http://www.w3.org/1999/xlink" viewBox="0 0 20 20">
    <style>.cls-1{{ fill: 'url(#未命名的渐变_2-3)' }}.cls-2{{ fill: 'none' }}.cls-3{{ fill: 'url(#_未命名的渐变_2)' }}.cls-4{{ fill: 'url(#_未命名的渐变_2-2)' }}.cls-1, .cls-2, .cls-3, .cls-4{{ strokeWidth: '0px' }}
    </style>
    <defs>
      <linearGradient id="_未命名的渐变_2" data-name="未命名的渐变 2" x1="10" y1=".29" x2="10" y2="8.67" gradientUnits="userSpaceOnUse">
        <stop offset="0" stop-color="#477dd8" />
        <stop offset="1" stop-color="#234c9e" />
      </linearGradient>
      <linearGradient id="_未命名的渐变_2-2" data-name="未命名的渐变 2" x1="10" y1="7.19" x2="10" y2="14.04" xlink: href="#_未命名的渐变_2" />
      <linearGradient id="_未命名的渐变_2-3" data-name="未命名的渐变 2" x1="10" y1="13.14" x2="10" y2="28.38" gradientUnits="userSpaceOnUse">
        <stop offset="0" stop-color="#477dd8" />
        <stop offset="1" stop-color="#234c9e" />
      </linearGradient>
      {/* <linearGradient id="_未命名的渐变_2-3" data-name="未命名的渐变 2" x1="10" y1="12.85" x2="10" y2="19.71" xlink:href="#_未命名的渐变_2" /> */}
    </defs>
    <g id="_图层_1-2" data-name="图层 1">
      <rect class="cls-2" fill={`${theme === 'light' ? 'none' : 'none'}`} x="0" width="20" height="20" />
      <g>
        <ellipse class="cls-3" fill={`${theme === 'light' ? 'url(#_未命名的渐变_2)' : '#fff'}`} cx="10" cy="4.48" rx="10" ry="4.19" />
        <path class="cls-4" fill={`${theme === 'light' ? 'url(#_未命名的渐变_2)' : '#fff'}`} d="m19.98,7.19c-1.31,1.71-5.29,2.96-9.98,2.96S1.33,8.9.02,7.19v2.53S0,9.81,0,9.86s.01.09.02.14v.32h.05c.54,2.1,4.78,3.73,9.94,3.73s9.39-1.63,9.94-3.73h.05v-.32s.02-.09.02-.14-.01-.09-.02-.14v-2.53Z" />
        <path class="cls-1" fill={`${theme === 'light' ? 'url(#_未命名的渐变_2-3)' : '#fff'}`} d="m19.98,12.85c-1.31,1.71-5.29,2.96-9.98,2.96S1.33,14.56.02,12.85v2.53S0,15.47,0,15.52s.01.09.02.14v.32h.05c.54,2.1,4.78,3.73,9.94,3.73s9.39-1.63,9.94-3.73h.05v-.32s.02-.09.02-.14-.01-.09-.02-.14v-2.53Z" />
      </g>
    </g>
  </svg>
);

const resourceManageOutlined = (theme) => (
  <svg id="_图层_2" data-name="图层 2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
    <defs>
      <style>.cls-1{{ fill: 'none' }}.cls-1, .cls-2{{ strokeWidth: '0px' }}.cls-2 {{ fill: '#89939b' }}
      </style>
    </defs>
    <g id="_图层_1-2" data-name="图层 1">
      <rect class="cls-1" fill={`${theme === 'light' ? 'none' : 'none'}`} x="0" width="20" height="20" />
      <g>
        <path class="cls-2" fill={'#89939b'} d="m10,2.29c4.9,0,7.66,1.54,7.99,2.19-.33.65-3.09,2.19-7.99,2.19s-7.66-1.54-7.99-2.19c.33-.65,3.09-2.19,7.99-2.19m0-2C4.48.29,0,2.17,0,4.48s4.48,4.19,10,4.19,10-1.88,10-4.19S15.52.29,10,.29h0Z" />
        <path class="cls-2" fill={'#89939b'} d="m18,4.48v2h0v3.32c-.2.62-2.98,2.24-8,2.24s-7.8-1.62-8-2.19v-5.37H0v5.37C0,12.17,4.48,14.04,10,14.04s10-1.88,10-4.19v-5.37h-2Z" />
        <path class="cls-2" fill={'#89939b'} d="m18,10.14v2h0v3.32c-.2.62-2.98,2.24-8,2.24s-7.8-1.62-8-2.19v-5.37H0v5.37C0,17.83,4.48,19.71,10,19.71s10-1.88,10-4.19v-5.37h-2Z" />
      </g>
    </g>
  </svg>
);

export default function resourceManageIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? resourceManageFilled(theme) : resourceManageOutlined(theme)}
    </div>
  );
}