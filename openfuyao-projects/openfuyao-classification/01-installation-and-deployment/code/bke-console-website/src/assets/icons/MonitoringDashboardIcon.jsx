/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
const monitoringDashboardFilled = (theme) => (
  <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 20 20">
    <defs><style>.cls-1{{ fill: 'none' }}.cls-2{{ fill: 'url(#未命名的渐变_2)' }}.cls-3{{ fill: 'url(#未命名的渐变_2-2)' }}</style>
      <linearGradient id="未命名的渐变_2" x1="10" y1="20" x2="10" y2="18" gradientUnits="userSpaceOnUse">
        <stop offset="0" stop-color="#477dd8" />
        <stop offset="1" stop-color="#234c9e" />
      </linearGradient>
      <linearGradient id="未命名的渐变_2-2" y1="16" y2="0" xlink:href="#未命名的渐变_2" />
    </defs>
    <g id="自定义监控看板选中" data-name="图层 2"><g id="图层_1-2" data-name="图层 1">
      <rect id="矩形_831" data-name="矩形 831" className="cls-1" fill="none" width="20" height="20" />
      <path className="cls-2" fill={`${theme === 'light' ? 'url(#未命名的渐变_2)' : 'white'}`} d="M15,18H5a1,1,0,0,0,0,2H15a1,1,0,0,0,0-2Z" />
      <path className="cls-3" fill={`${theme === 'light' ? 'url(#未命名的渐变_2)' : 'white'}`} d="M17,0H3A3,3,0,0,0,0,3V13a3,3,0,0,0,3,3H17a3,3,0,0,0,3-3V3A3,3,0,0,0,17,0ZM13.07,9h-2v2a1,1,0,0,1-2,0V9h-2a1,1,0,0,1,0-2h2V5a1,1,0,0,1,2,0V7h2a1,1,0,0,1,0,2Z" /></g></g>
  </svg>
);

const monitoringDashboardOutlined = (theme) => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
    <defs><style>.cls-1{{ fill: 'none' }}.cls-2{{ fill: '#89939b' }}</style></defs>
    <g id="自定义监控" data-name="图层 2"><g id="图层_1-2" data-name="图层 1">
      <rect id="矩形_831" data-name="矩形 831" className="cls-1" fill="none" width="20" height="20" />
      <path id="联合_52" data-name="联合 52" className="cls-2" fill="#89939b" d="M5,20a1,1,0,0,1,0-2H15a1,1,0,0,1,0,2ZM3,16a3,3,0,0,1-3-3V3A3,3,0,0,1,3,0H17a3,3,0,0,1,3,3h0V13a3,3,0,0,1-3,3ZM2,3V13a1,1,0,0,0,1,1H17a1,1,0,0,0,1-1V3a1,1,0,0,0-1-1H3A1,1,0,0,0,2,3Z" />
      <path className="cls-2" fill="#89939b" d="M13,7H11V5A1,1,0,0,0,9,5V7H7A1,1,0,0,0,7,9H9v2a1,1,0,0,0,2,0V9h2a1,1,0,0,0,0-2Z" /></g></g>
  </svg>
);

export default function MonitoringDashboardIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? monitoringDashboardFilled(theme) : monitoringDashboardOutlined(theme)}
    </div>
  );
}