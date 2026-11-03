/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
const applicationFilled = (theme) => (
  <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="20" height="20" viewBox="0 0 20 20">
    <defs>
      <linearGradient id="linear-gradient" x1="0.5" x2="0.5" y2="1" gradientUnits="objectBoundingBox">
        <stop offset="0" stop-color="#477dd8" />
        <stop offset="1" stop-color="#234c9e" />
      </linearGradient>
    </defs>
    <g id="应用管理选中" transform="translate(15319 13435)">
      <rect id="矩形_831" data-name="矩形 831" width="20" height="20" transform="translate(-15319 -13435)" fill="none" />
      <path id="联合_81" data-name="联合 81" d="M8358,5597a3,3,0,0,1-3-3v-3a3,3,0,0,1,3-3h3a3,3,0,0,1,3,3v3a3,3,0,0,1-3,3Zm-11,0a3,3,0,0,1-3-3v-3a3,3,0,0,1,3-3h3a3,3,0,0,1,3,3v3a3,3,0,0,1-3,3Zm11-11a3,3,0,0,1-3-3v-3a3,3,0,0,1,3-3h3a3,3,0,0,1,3,3v3a3,3,0,0,1-3,3Zm-1-6v3a1,1,0,0,0,1,1h3a1,1,0,0,0,1-1v-3a1,1,0,0,0-1-1h-3A1,1,0,0,0,8357,5580Zm-10,6a3,3,0,0,1-3-3v-3a3,3,0,0,1,3-3h3a3,3,0,0,1,3,3v3a3,3,0,0,1-3,3Z" transform="translate(-23663 -19012)" fill={`${theme === 'light' ? 'url(#linear-gradient' : 'white'}`} />
    </g>
  </svg>
);

const applicationOutlined = (theme) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">
    <g id="应用管理正常" transform="translate(15319 13435)">
      <rect id="矩形_831" data-name="矩形 831" width="20" height="20" transform="translate(-15319 -13435)" fill="none" />
      <path id="联合_80" data-name="联合 80" d="M8358,5660a3,3,0,0,1-3-3v-3a3,3,0,0,1,3-3h3a3,3,0,0,1,3,3v3a3,3,0,0,1-3,3Zm-1-6v3a1,1,0,0,0,1,1h3a1,1,0,0,0,1-1v-3a1,1,0,0,0-1-1h-3A1,1,0,0,0,8357,5654Zm-10,6a3,3,0,0,1-3-3v-3a3,3,0,0,1,3-3h3a3,3,0,0,1,3,3v3a3,3,0,0,1-3,3Zm-1-6v3a1,1,0,0,0,1,1h3a1,1,0,0,0,1-1v-3a1,1,0,0,0-1-1h-3A1,1,0,0,0,8346,5654Zm12-5a3,3,0,0,1-3-3v-3a3,3,0,0,1,3-3h3a3,3,0,0,1,3,3v3a3,3,0,0,1-3,3Zm-1-6v3a1,1,0,0,0,1,1h3a1,1,0,0,0,1-1v-3a1,1,0,0,0-1-1h-3A1,1,0,0,0,8357,5643Zm-10,6a3,3,0,0,1-3-3v-3a3,3,0,0,1,3-3h3a3,3,0,0,1,3,3v3a3,3,0,0,1-3,3Zm-1-6v3a1,1,0,0,0,1,1h3a1,1,0,0,0,1-1v-3a1,1,0,0,0-1-1h-3A1,1,0,0,0,8346,5643Z" transform="translate(-23663 -19075)" fill="#89939b" />
    </g>
  </svg>
);

export default function ApplicationIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? applicationFilled(theme) : applicationOutlined(theme)}
    </div>
  );
}