/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
const appMarketFilled = (theme) => (
  <svg width="19.338867" height="20.000000" viewBox="0 0 19.3389 20" fill="none" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <desc>
      Created with Pixso.
    </desc>
    <defs>
      <linearGradient x1="9.669605" y1="0.000000" x2="9.669605" y2="20.000000" id="paint_linear_960_23605_0" gradientUnits="userSpaceOnUse">
        <stop stop-color="#477DD8" />
        <stop offset="1.000000" stop-color="#234C9E" />
      </linearGradient>
    </defs>
    <path id="合并" d="M6.39209 2L5.83008 9L7.83643 9L8.39844 2L6.39209 2ZM5.66943 0L3.02979 0C2.58447 0 2.19287 0.294189 2.06934 0.72168L0.0400391 7.72168C-0.14502 8.36133 0.334473 9 1.00049 9L1.66943 9L1.66943 19C1.66943 19.5522 2.11719 20 2.66943 20L16.6694 20C17.2217 20 17.6694 19.5522 17.6694 19L17.6694 9L18.3384 9C19.0044 9 19.4844 8.36133 19.2988 7.72168L17.27 0.72168C17.146 0.294189 16.7549 0 16.3096 0L5.66943 0ZM10.9409 2L11.5024 9L13.5093 9L12.9473 2L10.9409 2ZM12.6694 15L12.6694 18L6.66943 18L6.66943 15C6.66943 13.3433 8.0127 12 9.66943 12C11.3267 12 12.6694 13.3433 12.6694 15Z" clip-rule="evenodd" fill={`${theme === 'light' ? 'url(#paint_linear_960_23605_0' : 'white'}`} fill-opacity="1.000000" fill-rule="evenodd" />
  </svg>
);

const appMarketOutlined = (theme) => (
  <svg width="19.338867" height="20.000000" viewBox="0 0 19.3389 20" fill="none" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <desc>
      Created with Pixso.
    </desc>
    <defs />
    <path id="合并" d="M3.02979 0L16.3096 0C16.7549 0 17.146 0.294189 17.27 0.72168L19.2988 7.72168C19.4844 8.36133 19.0044 9 18.3384 9L17.6694 9L17.6694 19C17.6694 19.5522 17.2217 20 16.6694 20L2.66943 20C2.11719 20 1.66943 19.5522 1.66943 19L1.66943 9L1.00049 9C0.334473 9 -0.14502 8.36133 0.0400391 7.72168L2.06934 0.72168C2.19287 0.294189 2.58447 0 3.02979 0ZM8.66943 18L10.6694 18L10.6694 15C10.6694 14.4478 10.2217 14 9.66943 14C9.11719 14 8.66943 14.4478 8.66943 15L8.66943 18ZM12.6694 18L12.6694 15C12.6694 13.3433 11.3267 12 9.66943 12C8.0127 12 6.66943 13.3433 6.66943 15L6.66943 18L3.66943 18L3.66943 9L15.6694 9L15.6694 18L12.6694 18ZM10.8447 2L11.4243 7L7.91504 7L8.49463 2L10.8447 2ZM15.5581 2L17.0078 7L13.4375 7L12.8579 2L15.5581 2ZM6.48096 2L5.90137 7L2.33154 7L3.78076 2L6.48096 2Z" clip-rule="evenodd" fill="#89939B" fill-opacity="1.000000" fill-rule="evenodd" />
  </svg>
);

export default function AppMarketIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? appMarketFilled(theme) : appMarketOutlined(theme)}
    </div>
  );
}