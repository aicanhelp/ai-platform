/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
const clusterMemberIconFilled = (theme) => (
  <svg id="_图层_2" data-name="图层 2" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 19.64 19.37">
    <defs>
      <style>
        .cls-1 {{
          fill: 'url(#_未命名的渐变_2-3)',
        }}

        .cls-1, .cls-2, .cls-3, .cls-4 {{
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
      </style>
      <linearGradient id="_未命名的渐变_2" data-name="未命名的渐变 2" x1="7.84" y1="1.87" x2="7.84" y2="10.02" gradientUnits="userSpaceOnUse">
        <stop offset="0" stop-color="#477dd8" />
        <stop offset="1" stop-color="#234c9e" />
      </linearGradient>
      <linearGradient id="_未命名的渐变_2-2" data-name="未命名的渐变 2" x1="7.84" y1="11.21" x2="7.84" y2="19.37" xlink:href="#_未命名的渐变_2" />
      <linearGradient id="_未命名的渐变_2-3" data-name="未命名的渐变 2" x1="13.96" y1="0" x2="13.96" y2="7.14" xlink:href="#_未命名的渐变_2" />
      <linearGradient id="_未命名的渐变_2-4" data-name="未命名的渐变 2" x1="16.3" y1="8.33" x2="16.3" y2="16.65" xlink:href="#_未命名的渐变_2" />
    </defs>
    <g id="_图层_1-2" data-name="图层 1">
      <circle class="cls-2" fill={theme === 'light' ? 'url(#_未命名的渐变_2)' : '#fff'} cx="7.84" cy="5.95" r="4.08" />
      <path class="cls-4" fill={theme === 'light' ? 'url(#_未命名的渐变_2)' : '#fff'} d="m15.67,17.32c0,2.25-3.51,2.04-7.84,2.04s-7.84.21-7.84-2.04,1.45-6.11,7.84-6.11,7.84,3.86,7.84,6.11Z" />
      <path class="cls-1" fill={theme === 'light' ? 'url(#_未命名的渐变_2)' : '#fff'} d="m14.68,7.14c-.34,0-.67-.17-.85-.48-.29-.47-.14-1.09.33-1.38.53-.33.85-.89.85-1.51,0-.98-.79-1.77-1.77-1.77-.3,0-.58.07-.85.21-.49.26-1.09.08-1.36-.4-.26-.49-.08-1.09.4-1.36.55-.3,1.17-.46,1.8-.46,2.08,0,3.77,1.69,3.77,3.77,0,1.32-.67,2.53-1.8,3.22-.16.1-.34.15-.52.15Z" />
      <path class="cls-3" fill={theme === 'light' ? 'url(#_未命名的渐变_2)' : '#fff'} d="m17.36,16.65c-.43,0-.83-.29-.96-.72-.15-.53.15-1.08.68-1.24.56-.16.56-.31.56-.45,0-.77-.28-3.33-3.85-3.93-.54-.09-.91-.61-.82-1.15s.6-.91,1.15-.82c4.07.69,5.52,3.61,5.52,5.9,0,.67-.26,1.86-2,2.37-.09.03-.19.04-.28.04Z" />
    </g>
  </svg>
);

const clusterMemberIconOutlined = (theme) => (
  <svg id="集群成员" data-name="图层 2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 19.64 19.37">
    <defs>
      <style>
        .cls-1 {{
          fill: '#89939b',
          strokeWidth: '0px',
        }}
      </style>
    </defs>
    <g id="_图层_1-2" data-name="图层 1">
      <path class="cls-1" fill="#89939b" d="m7.84,3.87c1.14,0,2.08.93,2.08,2.08s-.93,2.08-2.08,2.08-2.08-.93-2.08-2.08.93-2.08,2.08-2.08m0-2c-2.25,0-4.08,1.82-4.08,4.08s1.82,4.08,4.08,4.08,4.08-1.82,4.08-4.08S10.09,1.87,7.84,1.87h0Z" />
      <path class="cls-1" fill="#89939b" d="m7.84,13.21c5.02,0,5.75,2.71,5.83,3.87-.81.28-3.05.28-3.83.28h-.95s-1.04,0-1.04,0h-1.04s-.95,0-.95,0c-.78,0-3.02,0-3.83-.28.08-1.16.8-3.87,5.83-3.87m0-2c-6.39,0-7.84,3.86-7.84,6.11,0,1.89,2.48,2.04,5.84,2.04.64,0,1.3,0,1.99,0s1.36,0,1.99,0c3.36,0,5.84-.15,5.84-2.04,0-2.25-1.45-6.11-7.84-6.11h0Z" />
      <path class="cls-1" fill="#89939b" d="m14.68,7.14c-.34,0-.67-.17-.85-.48-.29-.47-.14-1.09.33-1.38.53-.33.85-.89.85-1.51,0-.98-.79-1.77-1.77-1.77-.3,0-.58.07-.85.21-.49.26-1.09.08-1.36-.4-.26-.49-.08-1.09.4-1.36.55-.3,1.17-.46,1.8-.46,2.08,0,3.77,1.69,3.77,3.77,0,1.32-.67,2.53-1.8,3.22-.16.1-.34.15-.52.15Z" />
      <path class="cls-1" fill="#89939b" d="m17.36,16.65c-.43,0-.83-.29-.96-.72-.15-.53.15-1.08.68-1.24.56-.16.56-.31.56-.45,0-.77-.28-3.33-3.85-3.93-.54-.09-.91-.61-.82-1.15s.6-.91,1.15-.82c4.07.69,5.52,3.61,5.52,5.9,0,.67-.26,1.86-2,2.37-.09.03-.19.04-.28.04Z" />
    </g>
  </svg>
);

export default function ClusterMemberIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? clusterMemberIconFilled(theme) : clusterMemberIconOutlined(theme)}
    </div>
  );
}