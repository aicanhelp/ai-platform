/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
const customizeIconFilled = (theme) => (
  <svg id="_图层_2" data-name="图层 2" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 20 20">
    <defs>
      <style>
        .cls-1 {{
          fill: 'url(#_未命名的渐变_2-3)',
        }}

        .cls-1, .cls-2, .cls-3 {{
          strokeWidth: '0px',
        }}

        .cls-2 {{
          fill: 'url(#_未命名的渐变_2)',
        }}

        .cls-3 {{
          fill: 'url(#_未命名的渐变_2-2)',
        }}
      </style>
      <linearGradient id="_未命名的渐变_2" data-name="未命名的渐变 2" x1="10" y1="0" x2="10" y2="12" gradientUnits="userSpaceOnUse">
        <stop offset="0" stop-color="#477dd8" />
        <stop offset="1" stop-color="#234c9e" />
      </linearGradient>
      <linearGradient id="_未命名的渐变_2-2" data-name="未命名的渐变 2" x1="10" y1="8.99" x2="10" y2="16" xlink:href="#_未命名的渐变_2" />
      <linearGradient id="_未命名的渐变_2-3" data-name="未命名的渐变 2" x1="10" y1="12.99" x2="10" y2="20" xlink:href="#_未命名的渐变_2" />
    </defs>
    <g id="_图层_1-2" data-name="图层 1">
      <path class="cls-2" fill={theme === 'light' ? 'url(#_未命名的渐变_2)' : '#fff'} d="m8.36,11.59L.68,6.99c-.91-.54-.91-1.43,0-1.97L8.36.41c.91-.54,2.38-.54,3.29,0l7.67,4.6c.91.54.91,1.43,0,1.97l-7.67,4.6c-.91.54-2.38.54-3.29,0Z" />
      <path class="cls-3" fill={theme === 'light' ? 'url(#_未命名的渐变_2)' : '#fff'} d="m10,16c-.76,0-1.51-.19-2.12-.56L.47,10.84c-.47-.29-.61-.91-.32-1.38.29-.47.91-.61,1.38-.32l7.41,4.6c.55.34,1.56.35,2.12,0l7.41-4.6c.47-.29,1.09-.15,1.38.32.29.47.15,1.09-.32,1.38l-7.41,4.6c-.6.37-1.36.56-2.12.56Z" />
      <path class="cls-1" fill={theme === 'light' ? 'url(#_未命名的渐变_2)' : '#fff'} d="m10,20c-.76,0-1.51-.19-2.12-.56L.47,14.84c-.47-.29-.61-.91-.32-1.38.29-.47.91-.61,1.38-.32l7.41,4.6c.55.34,1.56.35,2.12,0l7.41-4.6c.47-.29,1.09-.15,1.38.32.29.47.15,1.09-.32,1.38l-7.41,4.6c-.6.37-1.36.56-2.12.56Z" />
    </g>
  </svg>
);

const customizeIconOutlined = (theme) => (
  <svg id="自定义资源" data-name="图层 2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
    <defs>
      <style>
        .cls-1 {{
          fill: '#89939b',
          strokeWidth: '0px',
        }}
      </style>
    </defs>
    <g id="_图层_1-2" data-name="图层 1">
      <path class="cls-1" fill={'#89939b'} d="m10,2c.32,0,.54.08.62.12l6.46,3.88-6.46,3.88c-.07.04-.29.12-.62.12s-.54-.08-.62-.12l-6.46-3.88,6.46-3.88c.07-.04.29-.12.62-.12m0-2c-.6,0-1.19.14-1.64.41L.68,5.01c-.91.54-.91,1.43,0,1.97l7.67,4.6c.45.27,1.05.41,1.64.41s1.19-.14,1.64-.41l7.67-4.6c.91-.54.91-1.43,0-1.97L11.64.41c-.45-.27-1.05-.41-1.64-.41h0Z" />
      <path class="cls-1" fill={'#89939b'} d="m10,16c-.76,0-1.51-.19-2.12-.56L.47,10.84c-.47-.29-.61-.91-.32-1.38.29-.47.91-.61,1.38-.32l7.41,4.6c.55.34,1.56.35,2.12,0l7.41-4.6c.47-.29,1.09-.15,1.38.32.29.47.15,1.09-.32,1.38l-7.41,4.6c-.6.37-1.36.56-2.12.56Z" />
      <path class="cls-1" fill={'#89939b'} d="m10,20c-.76,0-1.51-.19-2.12-.56L.47,14.84c-.47-.29-.61-.91-.32-1.38s.91-.61,1.38-.32l7.41,4.6c.55.34,1.56.35,2.12,0l7.41-4.6c.47-.29,1.09-.15,1.38.32s.15,1.09-.32,1.38l-7.41,4.6c-.6.37-1.36.56-2.12.56Z" />
    </g>
  </svg>
);

export default function CustomizeIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? customizeIconFilled(theme) : customizeIconOutlined(theme)}
    </div>
  );
}