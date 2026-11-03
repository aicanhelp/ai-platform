/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import Inula, { useStore } from 'openinula'; //
import { Switch, Route, Redirect } from 'inula-router';
import { containerRouterPrefix, usermanageRouterPrefix } from '@/constant.js';
import MicroAppPage from '@/components/MicroApp';
import UserManage from '@/pages/userManage';
import ContainerPlatFormIndex from '@/pages/container';
import UserInfo from '@/pages/setting/UserInfo';
import Password from '@/pages/setting/Password';

const routeItems = [
  {
    path: `/${containerRouterPrefix}`,
    component: <ContainerPlatFormIndex />,
  },
  {
    path: `/${usermanageRouterPrefix}`,
    component: <UserManage />,
  },
  {
    path: '/setting/userinfo',
    component: <UserInfo />,
  },
  {
    path: '/setting/password',
    component: <Password />,
  },
];

export default function router() {
  const consolePluginStore = useStore('consolePlugins');
  return (
    <Switch>
      {
        routeItems.map(item => (
          <Route path={item.path} render={() => item.component} />
        ))
      }
      {
        ...consolePluginStore.$s.consolePlugins.filter(item => item.entrypoint === '/' && item.enabled)
          .map(item => (
            <Route path={`/${item.pluginName}`} render={props => <MicroAppPage {...props} key={item.pluginName} subKey={window.location.pathname} name={item.pluginName} />} />
          ))
      }
      <Redirect path='*' to={`/${containerRouterPrefix}/cluster`} />
    </Switch>
  );
}
