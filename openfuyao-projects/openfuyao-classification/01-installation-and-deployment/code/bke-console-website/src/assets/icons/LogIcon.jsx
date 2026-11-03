/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
const logFilled = (theme) => (
  <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="20.001" height="20.001" viewBox="0 0 20.001 20.001">
    <defs>
      <linearGradient id="linear-gradient" x1="0.5" x2="0.5" y2="1" gradientUnits="objectBoundingBox">
        <stop offset="0" stop-color="#477dd8" />
        <stop offset="1" stop-color="#234c9e" />
      </linearGradient>
    </defs>
    <g id="日志选中" transform="translate(15319.001 13435.001)">
      <rect id="矩形_831" data-name="矩形 831" width="20" height="20" transform="translate(-15319 -13435)" fill="none" />
      <path id="减去_21" data-name="减去 21" d="M8035,5595h-14a3,3,0,0,1-3-3v-12a3.005,3.005,0,0,1,3-3h.5v2a1.5,1.5,0,0,0,3,0v-2h7v2a1.5,1.5,0,0,0,3,0v-2h.5a3,3,0,0,1,3,3v12A3,3,0,0,1,8035,5595Zm-12-8a1,1,0,0,0,0,2h6a1,1,0,0,0,0-2Zm0-4a1,1,0,1,0,0,2h10a1,1,0,1,0,0-2Z" transform="translate(-23337 -19010.002)" fill={`${theme === 'light' ? 'url(#linear-gradient' : 'white'}`} />
      <rect id="矩形_853" data-name="矩形 853" width="5" height="2" rx="1" transform="translate(-15313 -13435.001) rotate(90)" fill="#3f66f5" />
      <rect id="矩形_854" data-name="矩形 854" width="5" height="2" rx="1" transform="translate(-15303 -13435.001) rotate(90)" fill="#3f66f5" />
    </g>
  </svg>
);

const logOutlined = (theme) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">
    <g id="日志正常" transform="translate(15319 13435)">
      <rect id="矩形_831" data-name="矩形 831" width="20" height="20" transform="translate(-15319 -13435)" fill="none" />
      <path id="联合_46" data-name="联合 46" d="M1-158a3,3,0,0,1-3-3v-12a3,3,0,0,1,3-3H2v-1a1,1,0,0,1,1-1,1,1,0,0,1,1,1v1h8v-1a1,1,0,0,1,1-1,1,1,0,0,1,1,1v1h1a3,3,0,0,1,3,3v12a3,3,0,0,1-3,3ZM0-173v12a1,1,0,0,0,1,1H15a1,1,0,0,0,1-1v-12a1,1,0,0,0-1-1H14a1,1,0,0,1-1,1,1,1,0,0,1-1-1H4a1,1,0,0,1-1,1,1,1,0,0,1-1-1H1A1,1,0,0,0,0-173Zm3,9a1,1,0,0,1-1-1,1,1,0,0,1,1-1H9a1,1,0,0,1,1,1,1,1,0,0,1-1,1Zm0-4a1,1,0,0,1-1-1,1,1,0,0,1,1-1H13a1,1,0,0,1,1,1,1,1,0,0,1-1,1Z" transform="translate(-15317 -13257)" fill="#89939b" />
    </g>
  </svg>
);

export default function LogIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? logFilled(theme) : logOutlined(theme)}
    </div>
  );
}