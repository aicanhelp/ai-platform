/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import Inula from 'openinula';
import './index.less';
import App from './App';
import extensionConfig from './extension';
import { ConfigProvider } from 'antd';
import { containerStylePrefix } from './common/constants';
import enUS from 'antd/es/locale/en_US';
import zhCN from 'antd/es/locale/zh_CN';

const locale = window.__OPENFUYAO_LOCALE__ || localStorage.getItem('locale') || 'zh-CN';
const antdLocale = locale === 'en-US' ? enUS : zhCN;

Inula.render(
  <ConfigProvider prefixCls={containerStylePrefix} locale={antdLocale}>
    <App />
  </ConfigProvider>,
  document.querySelector(`#${extensionConfig.menu.pluginName}_root`),
);
