/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
const clusterLifeIconFilled = (theme) => (
  <svg id="集群生命周期管理" data-name="图层 2" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 20 20">
    <defs>
      <linearGradient id="_未命名的渐变_2" data-name="未命名的渐变 2" x1="10" y1="18.19" x2="10" y2="20" gradientUnits="userSpaceOnUse">
        <stop offset="0" stop-color="#477dd8" />
        <stop offset="1" stop-color="#234c9e" />
      </linearGradient>
      <linearGradient id="_未命名的渐变_2-2" data-name="未命名的渐变 2" y1="0" x2="10" y2="16.08" xlink:href="#_未命名的渐变_2" />
    </defs>
    <g id="_图层_1-2" data-name="图层 1">
      <g>
        <path class="cls-1" fill={theme === 'light' ? 'url(#_未命名的渐变_2)' : '#fff'}
          d="m15,18.19H5c-.5,0-.9.4-.9.9s.4.9.9.9h10c.5,0,.9-.4.9-.9s-.4-.9-.9-.9Z" />
        <path class="cls-2" fill={theme === 'light' ? 'url(#_未命名的渐变_2)' : '#fff'}
          d="m18,0H2C.9,0,0,.9,0,2.01v12.06c0,1.11.9,2.01,2,2.01h16c1.1,0,2-.9,2-2.01V2.01c0-1.11-.9-2.01-2-2.01Zm-3,7.54c-.45,0-.87-.15-1.2-.4l-2.2,2.21c.25.34.4.75.4,1.21,0,1.11-.9,2.01-2,2.01s-2-.9-2-2.01c0-.43.14-.83.37-1.16l-2.21-2.22c-.33.23-.73.37-1.16.37-1.1,0-2-.9-2-2.01s.9-2.01,2-2.01,2,.9,2,2.01c0,.31-.07.61-.2.87l2.33,2.34c.26-.13.56-.2.87-.2.29,0,.56.06.81.17l2.36-2.37c-.11-.25-.17-.52-.17-.81,0-1.11.9-2.01,2-2.01s2,.9,2,2.01-.9,2.01-2,2.01Z" />
      </g>
    </g>
  </svg>
);

const clusterLifeIconOutlined = (theme) => (
  <svg id="集群生命周期管理" data-name="图层 2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
    <defs>
      <style>
        .cls-1 {{
          fill: '#89939b',
          fillRule: 'evenodd',
          strokeWidth: '0px',
        }}
      </style>
    </defs>
    <g id="_图层_1-2" data-name="图层 1">
      <path
        id="_集群生命周期管理正常"
        data-name="集群生命周期管理正常"
        fill='#89939b'
        className="cls-1"
        fillRule='evenodd'
        strokeWidth='0px'
        d="m18,0c1.1,0,2,.9,2,2.02v12.12c0,1.12-.9,2.02-2,2.02H2c-1.1,0-2-.9-2-2.02V2.02C0,.9.9,0,2,0h16Zm0,1.52c.28,0,.5.23.5.51v12.12c0,.28-.22.51-.5.51H2c-.28,0-.5-.23-.5-.51V2.02c0-.28.22-.51.5-.51h16ZM3,6.06c0,1.12.9,2.02,2,2.02.43,0,.83-.14,1.16-.37l2.21,2.24c-.23.33-.37.73-.37,1.17,0,1.12.9,2.02,2,2.02s2-.9,2-2.02c0-.46-.15-.88-.4-1.22l2.2-2.22c.33.26.75.41,1.2.41,1.1,0,2-.9,2-2.02s-.9-2.02-2-2.02-2,.9-2,2.02c0,.29.06.57.17.82l-2.36,2.39c-.25-.11-.52-.17-.81-.17-.31,0-.6.07-.87.2l-2.33-2.35c.13-.26.2-.56.2-.87,0-1.12-.9-2.02-2-2.02s-2,.9-2,2.02Zm11.9,12.12H4.9c-.5,0-.9.4-.9.91s.4.91.9.91h10c.5,0,.9-.4.9-.91s-.4-.91-.9-.91Z" />
    </g>
  </svg>
);

export default function ClusterLifeIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? clusterLifeIconFilled(theme) : clusterLifeIconOutlined(theme)}
    </div>
  );
}