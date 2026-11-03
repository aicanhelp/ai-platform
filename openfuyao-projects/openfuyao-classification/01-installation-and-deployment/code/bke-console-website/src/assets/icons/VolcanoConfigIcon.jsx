/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
const volcanoConfigFilled = (theme) => (
  <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 20 20">
    <defs><style>.cls-1{{ fill: 'none' }}.cls-2{{ fill: 'url(#未命名的渐变_2)' }}.cls-3{{ fill: 'url(#未命名的渐变_2-2)' }}</style>
      <linearGradient id="未命名的渐变_2" y1="15" x2="20" y2="15" gradientUnits="userSpaceOnUse">
        <stop offset="0" stop-color="#477dd8" />
        <stop offset="1" stop-color="#234c9e" />
      </linearGradient>
      <linearGradient id="未命名的渐变_2-2" x1="0" y1="5" x2="20" y2="5" xlink:href="#未命名的渐变_2" /></defs>
    <g id="配置策略选中" data-name="图层 2"><g id="图层_1-2" data-name="图层 1"><rect className="cls-1" fill="none" width="20" height="20" />
      <path className="cls-2" fill={`${theme === 'light' ? 'url(#未命名的渐变_2)' : 'white'}`} d="M16,11H4a4,4,0,0,0,0,8H16a4,4,0,0,0,0-8ZM4.14,18a3,3,0,1,1,3-3A3,3,0,0,1,4.14,18Z" />
      <path className="cls-3" fill={`${theme === 'light' ? 'url(#未命名的渐变_2)' : 'white'}`} d="M16,1H4A4,4,0,0,0,4,9H16a4,4,0,0,0,0-8Zm-.36,7a3,3,0,1,1,3-3A3,3,0,0,1,15.64,8Z" /></g></g>
  </svg>
);

const volcanoConfigOutlined = (theme) => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
    <defs><style>.cls-1{{ fill: 'none' }}.cls-2{{ fill: '#89939b' }}</style></defs>
    <g id="配置策略正常" data-name="图层 2"><g id="图层_1-2" data-name="图层 1">
      <rect className="cls-1" fill="none" width="20" height="20" />
      <path className="cls-2" fill="#89939b" d="M16,1H4A4,4,0,0,0,4,9H16a4,4,0,0,0,0-8Zm.15,6h-12a2,2,0,0,1-2-2h0a2,2,0,0,1,2-2h12a2,2,0,0,1,2,2h0A2,2,0,0,1,16.15,7Z" />
      <circle className="cls-2" fill="#89939b" cx="15.56" cy="5.04" r="1.25" />
      <path className="cls-2" fill="#89939b" d="M16,11H4a4,4,0,0,0,0,8H16a4,4,0,0,0,0-8Zm.15,6h-12a2,2,0,0,1-2-2h0a2,2,0,0,1,2-2h12a2,2,0,0,1,2,2h0A2,2,0,0,1,16.15,17Z" />
      <circle className="cls-2" fill="#89939b" cx="4.51" cy="15.04" r="1.25" /></g></g>
  </svg>
);

export default function VolcanoConfigIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? volcanoConfigFilled(theme) : volcanoConfigOutlined(theme)}
    </div>
  );
}