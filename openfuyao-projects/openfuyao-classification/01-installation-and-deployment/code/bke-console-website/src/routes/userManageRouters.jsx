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
import { usermanageRouterPrefix } from '@/constant.js';
import UserIndex from '@/pages/userManage/user/index';
import UserDetail from '@/pages/userManage/user/userDetail';
import RoleIndex from '@/pages/userManage/role/index';

export default function userManageRouters() {
  return (
    <Switch>
      <Route
        exact
        path={`/${usermanageRouterPrefix}/user`}
        component={UserIndex}
      />
      <Route
        exact
        path={`/${usermanageRouterPrefix}/user/detail`}
        component={UserDetail}
      />
      <Route
        exact
        path={`/${usermanageRouterPrefix}/role`}
        component={RoleIndex}
      />
      <Redirect path='*' to={`/${usermanageRouterPrefix}/user`} />
    </Switch>
  );
}
