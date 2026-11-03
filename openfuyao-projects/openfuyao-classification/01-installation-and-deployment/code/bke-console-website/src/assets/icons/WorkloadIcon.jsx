/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
const workloadFilled = (theme) => (
  <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="20" height="20" viewBox="0 0 20 20">
    <defs>
      <linearGradient id="linear-gradient" x1="0.5" x2="0.5" y2="1" gradientUnits="objectBoundingBox">
        <stop offset="0" stop-color="#477dd8" />
        <stop offset="1" stop-color="#234c9e" />
      </linearGradient>
    </defs>
    <g id="工作负载选中" transform="translate(15319 13435)">
      <rect id="矩形_831" data-name="矩形 831" width="20" height="20" transform="translate(-15319 -13435)" fill="none" />
      <path id="联合_43" data-name="联合 43" d="M7-55a3.006,3.006,0,0,1,.091-.735L4.445-57.37A2.986,2.986,0,0,1,3-57a3,3,0,0,1-3-3,3,3,0,0,1,3-3,3,3,0,0,1,.909.14l3.57-4.514A2.988,2.988,0,0,1,7-69a3,3,0,0,1,3-3,3,3,0,0,1,3,3,2.987,2.987,0,0,1-.431,1.55L16.1-62.863A3,3,0,0,1,17-63a3,3,0,0,1,3,3,3,3,0,0,1-3,3,2.985,2.985,0,0,1-1.43-.362l-2.648,1.68A3.01,3.01,0,0,1,13-55a3,3,0,0,1-3,3A3,3,0,0,1,7-55Zm4.843-2.367,2.376-1.507A2.993,2.993,0,0,1,14-60a2.986,2.986,0,0,1,.5-1.662l-3.479-4.517L11-66.17v8.342A2.993,2.993,0,0,1,11.843-57.367ZM8.2-57.4a2.992,2.992,0,0,1,.8-.429v-8.244l-3.5,4.42A2.985,2.985,0,0,1,6-60a2.991,2.991,0,0,1-.212,1.109Z" transform="translate(-15319 -13363)" fill={`${theme === 'light' ? 'url(#linear-gradient' : 'white'}`} />
    </g>
  </svg>
);

const workloadOutlined = (theme) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">
    <g id="工作负载正常" transform="translate(15319 13435)">
      <rect id="矩形_831" data-name="矩形 831" width="20" height="20" transform="translate(-15319 -13435)" fill="none" />
      <path id="联合_43" data-name="联合 43" d="M7-55a3.006,3.006,0,0,1,.091-.735L4.445-57.37A2.986,2.986,0,0,1,3-57a3,3,0,0,1-3-3,3,3,0,0,1,3-3,3,3,0,0,1,.909.14l3.57-4.514A2.988,2.988,0,0,1,7-69a3,3,0,0,1,3-3,3,3,0,0,1,3,3,2.987,2.987,0,0,1-.431,1.55L16.1-62.863A3,3,0,0,1,17-63a3,3,0,0,1,3,3,3,3,0,0,1-3,3,2.985,2.985,0,0,1-1.43-.362l-2.648,1.68A3.01,3.01,0,0,1,13-55a3,3,0,0,1-3,3A3,3,0,0,1,7-55Zm2,0a1,1,0,0,0,1,1,1,1,0,0,0,1-1,1,1,0,0,0-1-1A1,1,0,0,0,9-55Zm2.843-2.367,2.376-1.507A2.993,2.993,0,0,1,14-60a2.986,2.986,0,0,1,.5-1.662l-3.479-4.517L11-66.17v8.342A2.993,2.993,0,0,1,11.843-57.367ZM8.2-57.4a2.992,2.992,0,0,1,.8-.429v-8.244l-3.5,4.42A2.985,2.985,0,0,1,6-60a2.991,2.991,0,0,1-.212,1.109ZM16-60a1,1,0,0,0,1,1,1,1,0,0,0,1-1,1,1,0,0,0-1-1A1,1,0,0,0,16-60ZM2-60a1,1,0,0,0,1,1,1,1,0,0,0,1-1,1,1,0,0,0-1-1A1,1,0,0,0,2-60Zm7-9a1,1,0,0,0,1,1,1,1,0,0,0,1-1,1,1,0,0,0-1-1A1,1,0,0,0,9-69Z" transform="translate(-15319 -13363)" fill="#89939b" />
    </g>
  </svg>
);

export default function WorkloadIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? workloadFilled(theme) : workloadOutlined(theme)}
    </div>
  );
}