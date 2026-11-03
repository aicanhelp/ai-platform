/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
const multiRoleManageFilled = (theme) => (
  <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="20" height="20" viewBox="0 0 20 20">
    <defs>
      <linearGradient id="linear-gradient" x1="0.5" x2="0.5" y2="1" gradientUnits="objectBoundingBox">
        <stop offset="0" stop-color="#477dd8" />
        <stop offset="1" stop-color="#234c9e" />
      </linearGradient>
    </defs>
    <g id="角色管理选中" transform="translate(-999 -233)">
      <rect id="矩形_1314" data-name="矩形 1314" width="20" height="20" transform="translate(999 233)" fill="none" />
      <path id="联合_129" data-name="联合 129" d="M24005,19833.5a1,1,0,0,1-1-1c0-3.545,2.564-6.588,6.2-7.861a6,6,0,1,1,7.605,0c3.633,1.273,6.2,4.316,6.2,7.861a1,1,0,0,1-1,1Zm5-13.506a4,4,0,1,0,4-4A4.008,4.008,0,0,0,24010,19820Z" transform="translate(-23005 -19581)" fill={`${theme === 'light' ? 'url(#linear-gradient)' : 'white'}`} />
    </g>
  </svg>

);

const multiRoleManageOutlined = (theme) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">
    <g id="角色管理正常" transform="translate(-999 -172)">
      <rect id="矩形_1313" data-name="矩形 1313" width="20" height="20" transform="translate(999 172)" fill="none" />
      <g id="椭圆_176" data-name="椭圆 176" transform="translate(1003 172)" fill="none" stroke="#89939b" stroke-width="2">
        <circle cx="6" cy="6" r="6" stroke="none" />
        <circle cx="6" cy="6" r="5" fill="none" />
      </g>
      <path id="减去_51" data-name="减去 51" d="M24024,19821.5h-18c0-4.137,4.037-7.5,9-7.5s9,3.365,9,7.5Z" transform="translate(-23006 -19631)" fill="none" stroke="#89939b" stroke-linejoin="round" stroke-width="2" />
    </g>
  </svg>
);

export default function MultiRoleManageIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? multiRoleManageFilled(theme) : multiRoleManageOutlined(theme)}
    </div>
  );
}