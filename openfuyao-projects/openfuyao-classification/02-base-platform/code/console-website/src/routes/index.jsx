/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { useStore } from 'openinula';
import { Switch, Route, Redirect } from 'inula-router';
import MicroAppPage from '@/components/MicroApp';
import UserManage from '@/pages/userManage';
import ContainerPlatFormIndex from '@/pages/container';
import NoPermissions from '@/components/NoPermissions';
import Setting from './settingRouters';
import Home from '@/pages/Home';

export default function RootRouters({ permitted = false }) {
  const consolePluginStore = useStore('consolePlugins');
  return (
    <Switch>
      <Route path="/setting" component={Setting} />
      <Route exact path="/home" component={Home} />
      {permitted ? (<Switch>
        <Route path="/container_platform" component={ContainerPlatFormIndex} />
        <Route path="/user_manage" component={UserManage} />
        {
          consolePluginStore.$s.consolePlugins.filter(item => item.entrypoint === '/' && item.enabled)
            .map(item => (
              <Route path={`/${item.pluginName}`} render={props => <MicroAppPage {...props} key={item.pluginName} subKey={props.location.pathname} name={item.pluginName} />} />
            ))
        }
        <Redirect to="/container_platform/overview" />
      </Switch>) : (
        <Route component={NoPermissions} />
      )}
    </Switch >
  );
}
