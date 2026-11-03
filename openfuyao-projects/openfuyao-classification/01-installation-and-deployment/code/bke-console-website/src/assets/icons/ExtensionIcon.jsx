/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
const extensionFilled = (theme) => (
  <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="20" height="20" viewBox="0 0 20 20">
    <defs>
      <linearGradient id="linear-gradient" x1="0.364" y1="0.38" x2="0.905" y2="0.887" gradientUnits="objectBoundingBox">
        <stop offset="0" stop-color="#477dd8" />
        <stop offset="1" stop-color="#234c9e" />
      </linearGradient>
    </defs>
    <path id="扩展组件管理选中" d="M997.584,501.285l2.781-2.781a1.621,1.621,0,0,0,0-2.309l-2.311-2.312.866-.866a2.042,2.042,0,1,0-2.888-2.888l-.866.866-3.069-3.068a1.622,1.622,0,0,0-2.309,0l-2.67,2.67.866.866a2.2,2.2,0,0,1-3.091,3.129l-.019-.019-.867-.866-2.677,2.677a1.623,1.623,0,0,0,0,2.312l2.956,2.956.866-.866a2.187,2.187,0,0,1,3.111,0c.866.866,1.344,1.765.477,2.632l-1.344,1.344,2.2,2.2a1.621,1.621,0,0,0,2.31,0l2.81-2.809.866.866a2.026,2.026,0,1,0,2.865-2.865l-.866-.867Z" transform="translate(-980.848 -487.445)" fill={`${theme === 'light' ? 'url(#linear-gradient' : 'white'}`} />
  </svg>
);

const extensionOutlined = (theme) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">
    <path id="扩展组件管理正常" d="M989.761,506.445a2.38,2.38,0,0,1-1.692-.7l-2.643-2.643,1.865-1.865c.177-.177.157-.251.139-.317a2.037,2.037,0,0,0-.573-.791,1.079,1.079,0,0,0-1.542,0l-1.43,1.43-3.331-3.33a2.384,2.384,0,0,1,0-3.387l3.076-3.076,1.45,1.449a1.079,1.079,0,0,0,1.519-.016l0,0a1.077,1.077,0,0,0,0-1.541l-1.43-1.43,3.07-3.07a2.384,2.384,0,0,1,3.385,0l2.147,2.146.145-.145a2.765,2.765,0,1,1,3.911,3.911l-.145.145,1.458,1.459a2.383,2.383,0,0,1,0,3.385l-1.886,1.886.145.145a2.747,2.747,0,0,1-1.934,4.7h0a2.749,2.749,0,0,1-1.956-.812l-.145-.145-1.912,1.911A2.38,2.38,0,0,1,989.761,506.445ZM988,503.1l1.357,1.357a.565.565,0,0,0,.814,0l3.2-3.2,1.43,1.43a.943.943,0,0,0,.67.279h0a.929.929,0,0,0,.649-1.6l-1.49-1.491.732-.612,2.5-2.5a.565.565,0,0,0,0-.813l-2.744-2.744,1.43-1.43a.947.947,0,1,0-1.34-1.34l-1.43,1.43-3.432-3.431a.566.566,0,0,0-.814,0l-1.784,1.784.145.145A2.908,2.908,0,0,1,983.8,494.5l-.166-.167-1.791,1.791a.566.566,0,0,0,0,.816l2.045,2.045.145-.145a2.9,2.9,0,0,1,4.113,0,3.672,3.672,0,0,1,1.043,1.6,2.087,2.087,0,0,1-.609,2.077Z" transform="translate(-979.848 -486.445)" fill="#89939b" />
  </svg>
);

export default function ExtesnionIcon(selected, theme) {
  return (
    <div className="menu-icon">
      {selected ? extensionFilled(theme) : extensionOutlined(theme)}
    </div>
  );
}