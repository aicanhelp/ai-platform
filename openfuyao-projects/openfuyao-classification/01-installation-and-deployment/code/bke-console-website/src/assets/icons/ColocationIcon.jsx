/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
const colocationFilled = (theme) => (
  <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 20 20">
    <defs><style>.cls-1{{ fill: 'url(#未命名的渐变_34)' }}</style>
      <linearGradient id="未命名的渐变_34" x1="10" x2="10" y2="20" gradientUnits="userSpaceOnUse">
        <stop offset="0" stop-color="#477dd8" />
        <stop offset="0.01" stop-color="#467cd7" />
        <stop offset="0.42" stop-color="#3362b8" />
        <stop offset="0.76" stop-color="#2752a5" />
        <stop offset="1" stop-color="#234c9e" />
      </linearGradient>
    </defs>
    <g id="在离线混部选中" data-name="图层 2"><g id="图层_1-2" data-name="图层 1">
      <path className="cls-1" fill={`${theme === 'light' ? 'url(#未命名的渐变_34)' : 'white'}`} d="M16.94,14.32l1.8-2A1,1,0,0,0,17.31,11l-.25.28V6.79a4.86,4.86,0,0,0-4.85-4.85H6.82V1.32A1.32,1.32,0,0,0,5.5,0H2.32A1.32,1.32,0,0,0,1,1.32V4.5A1.32,1.32,0,0,0,2.32,5.82h.62v5.5L2.69,11a1,1,0,0,0-1.43,1.31l1.8,2A2.9,2.9,0,0,0,3.91,20,2.89,2.89,0,0,0,6,19.15H6a2.89,2.89,0,0,0-1.21-4.83l1.81-2A1,1,0,0,0,5.14,11l-.26.28V5.82H5.5A1.32,1.32,0,0,0,6.82,4.5V3.88h5.39a2.92,2.92,0,0,1,2.91,2.91v4.53L14.86,11a1,1,0,1,0-1.43,1.31l1.81,2A2.89,2.89,0,0,0,14,19.15a2.92,2.92,0,0,0,4.12,0h0a2.89,2.89,0,0,0-1.21-4.83ZM4.6,17.78A1,1,0,0,1,3.22,16.4a1,1,0,0,1,1.38,0A1,1,0,0,1,4.6,17.78Zm.28-13.9H2.94V1.94H4.88Zm11.89,13.9a1,1,0,1,1-.68-1.66.94.94,0,0,1,.68.28A1,1,0,0,1,16.77,17.78Z" /></g></g>
  </svg>
);

const colocationOutlined = (theme) => (
  <svg id="在离线混部正常" data-name="图层 1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
    <defs><style>.cls-1{{ fill: '#89939b' }}</style></defs>
    <path class="cls-1" fill="#89939b" d="M3.91,14.34a1,1,0,0,1-1-1V5.27a1,1,0,0,1,1.94,0v8.1A1,1,0,0,1,3.91,14.34Z" />
    <path class="cls-1" fill="#89939b" d="M16.09,14.28a1,1,0,0,1-1-1V6.79a2.92,2.92,0,0,0-2.91-2.91H6.75a1,1,0,0,1,0-1.94h5.46a4.86,4.86,0,0,1,4.85,4.85v6.52A1,1,0,0,1,16.09,14.28Z" />
    <path class="cls-1" fill="#89939b" d="M3.91,20A2.91,2.91,0,1,1,6,19.15H6A2.89,2.89,0,0,1,3.91,20Zm0-3.88a1,1,0,0,0-.69.28,1,1,0,1,0,1.38,0A1,1,0,0,0,3.91,16.12Z" />
    <path class="cls-1" fill="#89939b" d="M5.51,5.82H2.32A1.32,1.32,0,0,1,1,4.5V1.32A1.32,1.32,0,0,1,2.32,0H5.51A1.32,1.32,0,0,1,6.82,1.32V4.5A1.32,1.32,0,0,1,5.51,5.82ZM2.94,3.88H4.88V1.94H2.94Z" />
    <path class="cls-1" fill="#89939b" d="M16.09,20A2.91,2.91,0,0,1,14,15a2.91,2.91,0,0,1,4.12,4.12h0A2.91,2.91,0,0,1,16.09,20Zm0-3.88a1,1,0,0,0-.69.28,1,1,0,1,0,1.37,0A.94.94,0,0,0,16.09,16.12Z" />
    <path class="cls-1" fill="#89939b" d="M3.91,14.78a1,1,0,0,1-.71-.31L1.26,12.35A1,1,0,0,1,2.69,11l1.22,1.34L5.14,11a1,1,0,0,1,1.43,1.31L4.63,14.47A1,1,0,0,1,3.91,14.78Z" />
    <path class="cls-1" fill="#89939b" d="M16.09,14.78a1,1,0,0,1-.72-.31l-1.94-2.12A1,1,0,1,1,14.86,11l1.23,1.34L17.31,11a1,1,0,0,1,1.43,1.31L16.8,14.47A1,1,0,0,1,16.09,14.78Z" />
  </svg>
);

export default function ColocationIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? colocationFilled(theme) : colocationOutlined(theme)}
    </div>
  );
}