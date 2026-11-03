/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
const monitorFilled = (theme) => (
  <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="20.001" height="20" viewBox="0 0 20.001 20">
    <defs>
      <linearGradient id="linear-gradient" x1="0.5" x2="0.5" y2="1" gradientUnits="objectBoundingBox">
        <stop offset="0" stop-color="#477dd8" />
        <stop offset="1" stop-color="#234c9e" />
      </linearGradient>
    </defs>
    <g id="监控选中" transform="translate(15319.001 13435)">
      <rect id="矩形_831" data-name="矩形 831" width="20" height="20" transform="translate(-15319 -13435)" fill="none" />
      <path id="联合_83" data-name="联合 83" d="M7860,5597a1,1,0,0,1,0-2h10a1,1,0,0,1,0,2Zm-2-4a3,3,0,0,1-3-3v-10a3,3,0,0,1,3-3h14a3,3,0,0,1,3,3v10a3,3,0,0,1-3,3Zm3.595-8.882-3.3,3.3a1,1,0,0,0,0,1.415,1,1,0,0,0,1.413,0l2.593-2.6,2.081,2.079a.995.995,0,0,0,1.413,0l4.858-4.858a1,1,0,1,0-1.413-1.413l-4.15,4.15-2.081-2.079a.995.995,0,0,0-1.413,0Z" transform="translate(-23174 -19012)" fill={`${theme === 'light' ? 'url(#linear-gradient' : 'white'}`} />
    </g>
  </svg>
);

const monitorOutlined = (theme) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">
    <g id="监控正常" transform="translate(15319 13435)">
      <rect id="矩形_831" data-name="矩形 831" width="20" height="20" transform="translate(-15319 -13435)" fill="none" />
      <path id="联合_52" data-name="联合 52" d="M5-225a1,1,0,0,1-1-1,1,1,0,0,1,1-1H15a1,1,0,0,1,1,1,1,1,0,0,1-1,1Zm-2-4a3,3,0,0,1-3-3v-10a3,3,0,0,1,3-3H17a3,3,0,0,1,3,3v10a3,3,0,0,1-3,3ZM2-242v10a1,1,0,0,0,1,1H17a1,1,0,0,0,1-1v-10a1,1,0,0,0-1-1H3A1,1,0,0,0,2-242Zm1.293,8.831a1,1,0,0,1,0-1.414l3.3-3.3a1,1,0,0,1,1.415,0l2.08,2.079,4.15-4.151a1,1,0,0,1,1.414,0,1,1,0,0,1,0,1.414l-4.857,4.858a1,1,0,0,1-1.414,0l-2.08-2.08-2.594,2.594a1,1,0,0,1-.707.293A1,1,0,0,1,3.293-233.169Z" transform="translate(-15319 -13190)" fill="#89939b" />
    </g>
  </svg>
);

export default function MonitorIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? monitorFilled(theme) : monitorOutlined(theme)}
    </div>
  );
}