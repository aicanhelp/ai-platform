/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
const eventFilled = (theme) => (
  <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="20" height="20.002" viewBox="0 0 20 20.002">
    <defs>
      <linearGradient id="linear-gradient" x1="0.5" x2="0.5" y2="1" gradientUnits="objectBoundingBox">
        <stop offset="0" stop-color="#477dd8" />
        <stop offset="1" stop-color="#234c9e" />
      </linearGradient>
    </defs>
    <g id="事件选中" transform="translate(-1145 -234.998)">
      <rect id="矩形_1349" data-name="矩形 1349" width="20" height="20" transform="translate(1145 235)" fill="none" />
      <path id="联合_136" data-name="联合 136" d="M24007,19834a3,3,0,0,1-3-3v-12.992a3,3,0,0,1,3-3h.084a1.409,1.409,0,0,0-.084.5v2a1.5,1.5,0,0,0,1.5,1.5h7.006a1.5,1.5,0,0,0,1.494-1.5v-2a1.666,1.666,0,0,0-.084-.5h.084a3.007,3.007,0,0,1,3,3V19831a3.007,3.007,0,0,1-3,3Zm1-6.994a1,1,0,0,0,1,1h6a1,1,0,0,0,0-2h-6A1,1,0,0,0,24008,19827.006Zm0-4a1,1,0,0,0,1,1h2a1,1,0,0,0,0-2.006h-2A1,1,0,0,0,24008,19823Zm2-6a1,1,0,0,1-1-1v-1a1,1,0,0,1,1-1h4a1,1,0,0,1,1,1v1a1,1,0,0,1-1,1Z" transform="translate(-22857.004 -19579)" fill={`${theme === 'light' ? 'url(#linear-gradient' : 'white'}`} />
    </g>
  </svg>
);

const eventOutlined = (theme) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">
    <g id="事件正常" transform="translate(-1145 -170)">
      <rect id="矩形_1348" data-name="矩形 1348" width="20" height="20" transform="translate(1145 170)" fill="none" />
      <path id="联合_135" data-name="联合 135" d="M24006.994,19834a3,3,0,0,1-3-3v-13a3.007,3.007,0,0,1,3-3v2a1,1,0,0,0-.994,1v13a1,1,0,0,0,.994.994h10a1,1,0,0,0,1-.994v-13a1,1,0,0,0-1-1v-2a3,3,0,0,1,3,3v13a3.005,3.005,0,0,1-3,3Zm2-6a1,1,0,0,1,0-2h6a1,1,0,0,1,0,2Zm0-4a1,1,0,0,1,0-2h2a1,1,0,0,1,0,2Zm1-7a1,1,0,0,1-1-1v-1a1,1,0,0,1,1-1h4a1,1,0,0,1,1,1v1a1,1,0,0,1-1,1Z" transform="translate(-22857 -19644)" fill="#89939b" />
    </g>
  </svg>
);

export default function EventIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? eventFilled(theme) : eventOutlined(theme)}
    </div>
  );
}