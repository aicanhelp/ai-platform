/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
const platformManegeFilled = (theme) => (
  <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="20" height="20" viewBox="0 0 20 20">
    <defs>
      <linearGradient id="linear-gradient" x1="0.5" x2="0.5" y2="1" gradientUnits="objectBoundingBox">
        <stop offset="0" stop-color="#477dd8" />
        <stop offset="1" stop-color="#234c9e" />
      </linearGradient>
    </defs>
    <g id="平台管理选中" transform="translate(15319 13435)">
      <rect id="矩形_831" data-name="矩形 831" width="20" height="20" transform="translate(-15319 -13435)" fill="none" />
      <path id="减去_25" data-name="减去 25" d="M7783,5596.71a1,1,0,0,1-.486-.126l-8-4.443a1,1,0,0,1-.514-.874v-8.823a1,1,0,0,1,.514-.875l8-4.445a1,1,0,0,1,.972,0l8,4.445a1,1,0,0,1,.515.875v8.823a1,1,0,0,1-.515.874l-8,4.443A1,1,0,0,1,7783,5596.71Zm0-13.521h0l3,1.8v3.735l-2,1.2v2.333l3.514-2.108a1.007,1.007,0,0,0,.487-.858v-4.867a1.007,1.007,0,0,0-.487-.857l-4-2.4a1,1,0,0,0-1.03,0l-3.64,2.187,1.971,1.15,2.184-1.312Zm-5,1.983h0v4.118a1.007,1.007,0,0,0,.485.858l3.473,2.084.041-.025v-2.284l-2-1.2v-2.386l-2-1.167Z" transform="translate(-23092 -19011.855)" fill={`${theme === 'light' ? 'url(#linear-gradient' : 'white'}`} />
    </g>
  </svg>
);

const platformManegeOutlined = (theme) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">
    <g id="平台管理正常" transform="translate(15319 13435)">
      <rect id="矩形_831" data-name="矩形 831" width="20" height="20" transform="translate(-15319 -13435)" fill="none" />
      <path id="联合_48" data-name="联合 48" d="M7.514-262.413l-8-4.445A1,1,0,0,1-1-267.732v-8.823a1,1,0,0,1,.514-.875l8-4.444a1,1,0,0,1,.971,0l8,4.444a1,1,0,0,1,.515.875v8.823a1,1,0,0,1-.515.874l-8,4.445a1,1,0,0,1-.485.126A1,1,0,0,1,7.514-262.413ZM1-275.967v7.646l7,3.889,7-3.889v-7.646l-7-3.889Zm8,9.222v-2.331l2-1.2v-3.735l-3-1.8L5.816-274.5,3.845-275.65l3.641-2.185A1,1,0,0,1,8-277.978a1.008,1.008,0,0,1,.515.142l4,2.4a1.006,1.006,0,0,1,.486.858v4.867a1,1,0,0,1-.486.858L9-266.744H9Zm-2.042-.023-3.472-2.084A1,1,0,0,1,3-269.71v-4.118l2,1.167v2.385l2,1.2v2.283l-.041.024Z" transform="translate(-15317 -13152.856)" fill="#89939b" />
    </g>
  </svg>
);

export default function PlatformManegeIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? platformManegeFilled(theme) : platformManegeOutlined(theme)}
    </div>
  );
}