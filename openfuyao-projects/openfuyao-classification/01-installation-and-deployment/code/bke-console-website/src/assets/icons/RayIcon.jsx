/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
const rayIconFilled = (theme) => (
  <svg id="ray" data-name="图层 2" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 20 20">
    <defs>
      <linearGradient id="_未命名的渐变_2" data-name="未命名的渐变 2" x1="10" y1="0" x2="10" y2="20" gradientUnits="userSpaceOnUse">
        <stop offset="0" stop-color="#477dd8" />
        <stop offset="1" stop-color="#234c9e" />
      </linearGradient>
    </defs>
    <g id="_图层_1-2" data-name="图层 1">
      <path class="cls-1" fill={theme === 'light' ? 'url(#_未命名的渐变_2)' : '#fff'} strokeWidth={'0px'}
        d="m17.67,7.66c-.4,0-.77.11-1.1.29l-4.53-4.52c.18-.33.29-.7.29-1.1,0-1.29-1.04-2.33-2.33-2.33s-2.33,1.04-2.33,2.33,1.04,2.33,2.33,2.33c.4,0,.77-.11,1.1-.29l4.53,4.52c-.11.2-.19.4-.24.63l-3.13-.05c-.24-1.04-1.16-1.81-2.27-1.81s-1.98.74-2.24,1.74l-3.2-.05c-.28-.98-1.16-1.7-2.23-1.7-1.29,0-2.33,1.04-2.33,2.33s1.04,2.33,2.33,2.33c1.04,0,1.91-.69,2.21-1.63l3.27.05c.31.92,1.17,1.58,2.19,1.58s1.84-.63,2.17-1.52l3.34.05c.03.08.08.16.12.24l-4.54,4.55c-.33-.18-.7-.29-1.1-.29-1.29,0-2.33,1.04-2.33,2.33s1.04,2.33,2.33,2.33,2.33-1.04,2.33-2.33c0-.4-.11-.77-.29-1.1l4.54-4.55c.33.18.7.29,1.1.29,1.29,0,2.33-1.04,2.33-2.33s-1.04-2.33-2.33-2.33Z" />
    </g>
  </svg>
);

const rayIconOutlined = (theme) => (
  <svg id="ray" data-name="图层 2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
    <defs>
    </defs>
    <g id="_图层_1-2" data-name="图层 1">
      <path class="cls-1"
        fill="#89939b"
        strokeWidth={'0px'}
        d="m17.67,7.66c-.4,0-.77.11-1.1.29l-4.53-4.52c.18-.33.29-.7.29-1.1,0-1.29-1.04-2.33-2.33-2.33s-2.33,1.04-2.33,2.33,1.04,2.33,2.33,2.33c.4,0,.77-.11,1.1-.29l4.53,4.52c-.11.2-.19.4-.24.63l-3.13-.05c-.24-1.04-1.16-1.81-2.27-1.81s-1.98.74-2.24,1.74l-3.2-.05c-.28-.98-1.16-1.7-2.23-1.7-1.29,0-2.33,1.04-2.33,2.33s1.04,2.33,2.33,2.33c1.04,0,1.91-.69,2.21-1.63l3.27.05c.31.92,1.17,1.58,2.19,1.58s1.84-.63,2.17-1.52l3.34.05c.03.08.08.16.12.24l-4.54,4.55c-.33-.18-.7-.29-1.1-.29-1.29,0-2.33,1.04-2.33,2.33s1.04,2.33,2.33,2.33,2.33-1.04,2.33-2.33c0-.4-.11-.77-.29-1.1l4.54-4.55c.33.18.7.29,1.1.29,1.29,0,2.33-1.04,2.33-2.33s-1.04-2.33-2.33-2.33Zm-7.67,3.14c-.46,0-.83-.37-.83-.83s.37-.83.83-.83.83.37.83.83-.37.83-.83.83Z" />
    </g>
  </svg>
);

export default function RayIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? rayIconFilled(theme) : rayIconOutlined(theme)}
    </div>
  );
}