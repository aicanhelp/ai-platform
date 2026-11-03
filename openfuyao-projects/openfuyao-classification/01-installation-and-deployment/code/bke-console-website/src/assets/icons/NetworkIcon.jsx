/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
const networkFilled = (theme) => (
  <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="20" height="20" viewBox="0 0 20 20">
    <defs>
      <linearGradient id="linear-gradient" x1="0.5" x2="0.5" y2="1" gradientUnits="objectBoundingBox">
        <stop offset="0" stop-color="#477dd8" />
        <stop offset="1" stop-color="#234c9e" />
      </linearGradient>
    </defs>
    <g id="网络选中" transform="translate(15318 13435)">
      <rect id="矩形_831" data-name="矩形 831" width="20" height="20" transform="translate(-15318 -13435)" fill="none" />
      <path id="减去_24" data-name="减去 24" d="M8192.037,5597v0c.966-.052,1.891-.977,2.6-2.6a18.255,18.255,0,0,0,1.288-6.395h5.971a10,10,0,0,1-9.863,9Zm-.175,0a10,10,0,0,1-9.863-9h5.971a18.293,18.293,0,0,0,1.288,6.395c.714,1.626,1.639,2.55,2.6,2.6Zm.088-2.056h0a4.9,4.9,0,0,1-1.163-2.132,17.791,17.791,0,0,1-.816-4.811h3.956a17.732,17.732,0,0,1-.816,4.811,4.9,4.9,0,0,1-1.16,2.132ZM8201.9,5586h-5.972a18.253,18.253,0,0,0-1.288-6.393c-.736-1.677-1.69-2.6-2.687-2.6a10,10,0,0,1,9.947,9Zm-7.974,0h-3.956a17.778,17.778,0,0,1,.816-4.808,1.383,1.383,0,1,1,2.324,0,17.691,17.691,0,0,1,.816,4.807Zm-5.957,0H8182a10,10,0,0,1,9.943-9c-.995,0-1.948.928-2.685,2.6a18.269,18.269,0,0,0-1.288,6.392Z" transform="translate(-23499.949 -19012)" fill={`${theme === 'light' ? 'url(#linear-gradient' : 'white'}`} />
    </g>
  </svg>
);

const networkOutlined = (theme) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">
    <g id="网络正常" transform="translate(15318 13435)">
      <rect id="矩形_831" data-name="矩形 831" width="20" height="20" transform="translate(-15318 -13435)" fill="none" />
      <path id="联合_44" data-name="联合 44" d="M10-88A10,10,0,0,1,0-98a10,10,0,0,1,10-10A10,10,0,0,1,20-98,10,10,0,0,1,10-88ZM8.838-92.19A5.062,5.062,0,0,0,10-90.057a5.062,5.062,0,0,0,1.162-2.133A17.808,17.808,0,0,0,11.978-97H8.022A17.845,17.845,0,0,0,8.838-92.19ZM7.382-90.44A17.93,17.93,0,0,1,6.02-97H2.062A8.024,8.024,0,0,0,7.382-90.44Zm5.237,0A8.024,8.024,0,0,0,17.938-97H13.981A17.929,17.929,0,0,1,12.619-90.44ZM17.938-99a8.024,8.024,0,0,0-5.319-6.56A17.927,17.927,0,0,1,13.981-99Zm-5.96,0a17.817,17.817,0,0,0-.816-4.811A5.063,5.063,0,0,0,10-105.944a5.063,5.063,0,0,0-1.162,2.133A17.854,17.854,0,0,0,8.022-99ZM6.02-99a17.932,17.932,0,0,1,1.362-6.56A8.025,8.025,0,0,0,2.062-99Z" transform="translate(-15318 -13327)" fill="#89939b" />
    </g>
  </svg>
);

export default function NetworkIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? networkFilled(theme) : networkOutlined(theme)}
    </div>
  );
}