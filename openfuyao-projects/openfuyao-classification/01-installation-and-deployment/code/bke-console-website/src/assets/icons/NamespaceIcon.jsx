/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
const namespaceIconFilled = (theme) => (
  <svg id="命名空间选中" data-name="图层 2" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 20 20">
    <defs>
      <style>
        .cls-1 {{
          fill: 'url(#_未命名的渐变_2-3)',
        }}

        .cls-1, .cls-2, .cls-3, .cls-4, .cls-5 {{
          strokeWidth: '0px',
        }}

        .cls-2 {{
          fill: 'url(#_未命名的渐变_2)',
        }}

        .cls-3 {{
          fill: 'url(#_未命名的渐变_2-4)',
        }}

        .cls-4 {{
          fill: 'url(#_未命名的渐变_2-2)',
        }}

        .cls-5 {{
          fill: 'url(#_未命名的渐变_2-5)',
        }}
      </style>
      <linearGradient id="_未命名的渐变_2" data-name="未命名的渐变 2" x1="16.07" y1="17.73" x2="16.07" y2="18" gradientUnits="userSpaceOnUse">
        <stop offset="0" stop-color="#477dd8" />
        <stop offset="1" stop-color="#234c9e" />
      </linearGradient>
      <linearGradient id="_未命名的渐变_2-2" data-name="未命名的渐变 2" x1="3.93" y1="2" x2="3.93" y2="2.27" xlink:href="#_未命名的渐变_2" />
      <linearGradient id="_未命名的渐变_2-3" data-name="未命名的渐变 2" x1="3" y1="0" x2="3" y2="20" xlink:href="#_未命名的渐变_2" />
      <linearGradient id="_未命名的渐变_2-4" data-name="未命名的渐变 2" x1="17" y1="0" x2="17" y2="20" xlink:href="#_未命名的渐变_2" />
      <linearGradient id="_未命名的渐变_2-5" data-name="未命名的渐变 2" x1="10" y1="0" x2="10" y2="20" xlink:href="#_未命名的渐变_2" />
    </defs>
    <g id="_图层_1-2" data-name="图层 1">
      <g>
        <polygon className="cls-2" fill={theme === 'light' ? 'url(#_未命名的渐变_2)' : '#fff'} points="16 18 16.15 18 16 17.73 16 18" />
        <polygon className="cls-4" fill={theme === 'light' ? 'url(#_未命名的渐变_2)' : '#fff'} points="3.85 2 4 2.27 4 2 3.85 2" />
        <path className="cls-1" fill={theme === 'light' ? `url(#_未命名的渐变_2)` : '#fff'} d="m4,18h-2V2.78l-.43-.78c-.61-1.1-.21-2,.89-2h-.46C.9,0,0,.9,0,2v16c0,1.1.9,2,2,2h2c1.1,0,2-.9,2-2v-8l-2-3.61v11.61Z" />
        <path className="cls-3" fill={theme === 'light' ? `url(#_未命名的渐变_2)` : '#fff'} d="m18,0h-2c-1.1,0-2,.9-2,2v8l2,3.61V2h2v15.22l.43.78c.61,1.1.21,2-.89,2h.46c1.1,0,2-.9,2-2V2c0-1.1-.9-2-2-2Z" />
        <path className="cls-5" fill={theme === 'light' ? `url(#_未命名的渐变_2)` : '#fff'} d="m18.43,18l-.43-.78-2-3.61-2-3.61-4.43-8c-.61-1.1-2-2-3.11-2H2.46c-1.1,0-1.5.9-.89,2l.43.78,2,3.61,2,3.61,4.43,8c.61,1.1,2,2,3.11,2h4c1.1,0,1.5-.9.89-2Zm-2.43,0h-2.46c-.28,0-1.05-.41-1.36-.97L6,5.87l-2-3.61-.15-.27h2.6c.28,0,1.05.41,1.36.97l6.18,11.16,2,3.61.15.27h-.15Z" />
      </g>
    </g>
  </svg>
);

const namespaceIconOutlined = (theme) => (
  <svg id="命名空间" data-name="图层 2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
    <defs>
      <style>.cls-1 {{ fill: '#89939b', strokeWidth: '0px' }}</style>
    </defs>
    <g id="_图层_1-2" data-name="图层 1">
      <g>
        <polygon className="cls-1" fill='#89939b' strokeWidth='0px' points="16 18 16.15 18 16 17.73 16 18" />
        <polygon className="cls-1" fill='#89939b' strokeWidth='0px' points="3.85 2 4 2.27 4 2 3.85 2" />
        <path className="cls-1" fill='#89939b' strokeWidth='0px' d="m4,18h-2V2.78l-.43-.78c-.61-1.1-.21-2,.89-2h-.46C.9,0,0,.9,0,2v16c0,1.1.9,2,2,2h2c1.1,0,2-.9,2-2v-8l-2-3.61v11.61Z" />
        <path className="cls-1" fill='#89939b' strokeWidth='0px' d="m18,0h-2c-1.1,0-2,.9-2,2v8l2,3.61V2h2v15.22l.43.78c.61,1.1.21,2-.89,2h.46c1.1,0,2-.9,2-2V2c0-1.1-.9-2-2-2Z" />
        <path className="cls-1" fill='#89939b' strokeWidth='0px' d="m18.43,18l-.43-.78-2-3.61-2-3.61-4.43-8c-.61-1.1-2-2-3.11-2H2.46c-1.1,0-1.5.9-.89,2l.43.78,2,3.61,2,3.61,4.43,8c.61,1.1,2,2,3.11,2h4c1.1,0,1.5-.9.89-2Zm-2.43,0h-2.46c-.28,0-1.05-.41-1.36-.97L6,5.87l-2-3.61-.15-.27h2.6c.28,0,1.05.41,1.36.97l6.18,11.16,2,3.61.15.27h-.15Z" />
      </g>
    </g>
  </svg>
);

export default function NamespaceIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? namespaceIconFilled(theme) : namespaceIconOutlined(theme)}
    </div>
  );
}