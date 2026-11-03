/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
const overviewFilled = (theme) => (
  <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="20.001" height="20.001" viewBox="0 0 20.001 20.001">
    <defs>
      <linearGradient id="linear-gradient" x1="0.5" x2="0.5" y2="1" gradientUnits="objectBoundingBox">
        <stop offset="0" stop-color="#477dd8" />
        <stop offset="1" stop-color="#234c9e" />
      </linearGradient>
    </defs>
    <g id="概览选中" transform="translate(15319.001 13435)">
      <rect id="矩形_831" data-name="矩形 831" width="20" height="20" transform="translate(-15319 -13435)" fill="none" />
      <path id="减去_75" data-name="减去 75" d="M8773,1075h-14c-1.654,0-3-1.683-3-3.751v-12.5c0-2.068,1.346-3.751,3-3.751h14c1.654,0,3,1.683,3,3.751v12.5C8776,1073.318,8774.654,1075,8773,1075Zm-3-10a1,1,0,0,0-1,1v3a1,1,0,0,0,2,0v-3A1,1,0,0,0,8770,1065Zm-4-5a1,1,0,0,0-1,1v8a1,1,0,0,0,2,0v-8A1,1,0,0,0,8766,1060Zm-4,4a1,1,0,0,0-1,1v4a1,1,0,0,0,2,0v-4A1,1,0,0,0,8762,1064Zm8-2a1,1,0,1,0,1,1A1,1,0,0,0,8770,1062Z" transform="translate(-24075 -14490)" fill={`${theme === 'light' ? 'url(#linear-gradient' : 'white'}`} />
    </g>
  </svg>
);

const overviewOutlined = (theme) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">
    <g id="概览正常" transform="translate(15319 13435)">
      <rect id="矩形_831" data-name="矩形 831" width="20" height="20" transform="translate(-15319 -13435)" fill="none" />
      <path id="联合_53" data-name="联合 53" d="M-2097,47c-1.654,0-3-1.682-3-3.749V30.75c0-2.068,1.345-3.75,3-3.75h14c1.654,0,3,1.682,3,3.75v12.5c0,2.067-1.345,3.749-3,3.749Zm-1-16.25v12.5a1.147,1.147,0,0,0,1,1.249h14a1.147,1.147,0,0,0,1-1.249V30.75a1.148,1.148,0,0,0-1-1.25h-14A1.147,1.147,0,0,0-2098,30.75ZM-2087,41V38a1,1,0,0,1,1-1,1,1,0,0,1,1,1v3a1,1,0,0,1-1,1A1,1,0,0,1-2087,41Zm-4,0V33a1,1,0,0,1,1-1,1,1,0,0,1,1,1v8a1,1,0,0,1-1,1A1,1,0,0,1-2091,41Zm-4,0V37a1,1,0,0,1,1-1,1,1,0,0,1,1,1v4a1,1,0,0,1-1,1A1,1,0,0,1-2095,41Zm8-6a1,1,0,0,1,1-1,1,1,0,0,1,1,1,1,1,0,0,1-1,1A1,1,0,0,1-2087,35Z" transform="translate(-13219 -13462)" fill="#89939b" />
    </g>
  </svg>
);

export default function OverviewIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? overviewFilled(theme) : overviewOutlined(theme)}
    </div>
  );
}