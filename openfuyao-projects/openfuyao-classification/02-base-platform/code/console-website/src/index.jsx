/*
 * Copyright (c) 2023 Huawei Technologies Co.,Ltd.
 *
 * openInula is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

import Inula from 'openinula';
import '@/index.less';
import { i18nReady } from '@/i18n';
import App from '@/App';
import '@/store/consolePluginStore';
import '@/store/namespaceStore';
import '@/store/themeStore';
import '@/store/userStore';
import '@/store/intervalStore';
import '@/store/collapsedStore';
import '@/store/terminalStore';
import getCluster from '@/setup/getCluster';
import loadConsolePlugins from '@/setup/consolePlugin';

window.__OPENFUYAO__ = true;

getCluster();
await i18nReady;
await loadConsolePlugins();

Inula.render(<App />, document.getElementById('root'));
