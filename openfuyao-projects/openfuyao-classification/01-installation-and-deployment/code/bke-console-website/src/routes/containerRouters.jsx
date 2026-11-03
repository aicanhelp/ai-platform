/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Route, Switch, Redirect } from 'inula-router';
import Inula, { useStore } from 'openinula';
import { containerRouterPrefix } from '@/constant.js';

import MicroAppPage from '@/components/MicroApp';

export default function ContainerRouters() {
  const consolePluginStore = useStore('consolePlugins');

  const getPluginRoute = (item) => {
    return item.subPages ?
      item.subPages.map((sub) => (<Route
        path={`/${containerRouterPrefix}/${item.pluginName}/${sub.pageName}`}
        render={props => <MicroAppPage {...props} key={sub.pageName} subKey={window.location.pathname} name={item.pluginName} />}
      />)) :
      <Route
        path={`/${containerRouterPrefix}/${item.pluginName}`}
        render={props => <MicroAppPage {...props} key={item.pluginName} subKey={window.location.pathname} name={item.pluginName} />}
      />;
  };

  return (
    <Switch>
      {
        ...consolePluginStore.$s.consolePlugins.filter(item => item.entrypoint === `/${containerRouterPrefix}` && item.enabled)
          .map((item) => getPluginRoute(item))
      }
      
    </Switch>
  );
}
