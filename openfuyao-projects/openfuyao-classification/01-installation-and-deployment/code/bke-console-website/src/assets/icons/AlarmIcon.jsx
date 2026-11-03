/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
const alarmFilled = (theme) => (
  <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="20" height="20" viewBox="0 0 20 20">
    <defs>
      <linearGradient id="linear-gradient" x1="0.5" x2="0.5" y2="1" gradientUnits="objectBoundingBox">
        <stop offset="0" stop-color="#477dd8" />
        <stop offset="1" stop-color="#234c9e" />
      </linearGradient>
    </defs>
    <path id="告警选中" d="M24010,19834h0l-4,0h-1a1,1,0,1,1,0-2h1v-6a8,8,0,0,1,16,0v6h1a1,1,0,1,1,0,2h-11v-7a1,1,0,0,0-2,0v7Zm9.75-15.068a1.031,1.031,0,0,1-.5-.133,1,1,0,0,1-.361-1.363l.5-.867a.994.994,0,0,1,.865-.5,1.017,1.017,0,0,1,.5.133.993.993,0,0,1,.465.609.983.983,0,0,1-.1.76l-.5.861A.987.987,0,0,1,24019.75,19818.932Zm-11.5,0a1,1,0,0,1-.869-.5l-.5-.861a.983.983,0,0,1-.1-.76.993.993,0,0,1,.465-.609,1.017,1.017,0,0,1,.5-.133.994.994,0,0,1,.865.5l.5.867a1,1,0,0,1-.361,1.363A1.018,1.018,0,0,1,24008.252,19818.932Zm5.746-1.93a1,1,0,0,1-1-1v-1a1,1,0,1,1,2,0v1A1,1,0,0,1,24014,19817Z" transform="translate(-24004 -19814)" fill={`${theme === 'light' ? 'url(#linear-gradient' : 'white'}`} />
  </svg>
);

const alarmOutlined = (theme) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20.001" height="20" viewBox="0 0 20.001 20">
    <path id="告警正常" d="M2574-6537h-17a1,1,0,0,1-1-1,1,1,0,0,1,1-1h1v-6a8,8,0,0,1,8-8,8,8,0,0,1,8,8v6h1a1,1,0,0,1,1,1,1,1,0,0,1-1,1Zm-2-2v-6a6.007,6.007,0,0,0-6-6,6.008,6.008,0,0,0-6,6v6h2v-5a1,1,0,0,1,1-1,1,1,0,0,1,1,1v5Zm-.75-13.2a1,1,0,0,1-.365-1.368l.5-.866a1,1,0,0,1,1.366-.365,1,1,0,0,1,.365,1.366l-.5.866a1,1,0,0,1-.867.5A1,1,0,0,1,2571.249-6552.2Zm-11.865-.367-.5-.866a1,1,0,0,1,.365-1.366,1,1,0,0,1,1.366.365l.5.866a1,1,0,0,1-.365,1.368,1,1,0,0,1-.5.133A1,1,0,0,1,2559.385-6552.567ZM2565-6555v-1a1,1,0,0,1,1-1,1,1,0,0,1,1,1v1a1,1,0,0,1-1,1A1,1,0,0,1,2565-6555Z" transform="translate(-2556 6557)" fill="#89939b" />
  </svg>
);

export default function AlarmIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? alarmFilled(theme) : alarmOutlined(theme)}
    </div>
  );
}