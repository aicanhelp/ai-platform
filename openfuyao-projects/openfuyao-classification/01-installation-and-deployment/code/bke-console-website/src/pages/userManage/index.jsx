/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import '@/styles/pages/containerIndex.less';
import '@/styles/components/menu.less';
import UserManageRouters from '@/routes/userManageRouters.jsx';
import { useState, useEffect, useStore } from 'openinula';
import UserMenu from './userMenu';
import { getUserList } from '@/api/clusterApi';
import { ResponseCode } from '@/common/constants';
import NoPermissions from '@/components/NoPermissions';
export default function user() {
  const [collapsed, setCollapsed] = useState(false); // 折叠
  const [isShowNopermission, setIsShowNopermission] = useState(true);
  const handleCollapseTrigger = (status) => {
    setCollapsed(status); // 切换
  };
  const getUserManagesList = async () => {
    const res = await getUserList();
    if (res.status !== ResponseCode.Forbidden) {
      setIsShowNopermission(false);
    }
  };
  useEffect(() => {
    getUserManagesList();
  }, []);
  return <div style={{ display: 'flex', height: '100%' }}>
    <div className='container_platform_box_content container_platform_box' style={{ display: 'flex', flexDirection: 'row' }}>
      <div className={collapsed ? 'container_nav_bar collapse_self_bar' : 'container_nav_bar'} style={{ paddingTop: '0' }}>
        <UserMenu collapsed={collapsed} handleUserCollapseChange={handleCollapseTrigger} />
      </div>
      <div className={collapsed ? 'container_content collapse_self_content' : 'container_content'}>
        {/* 后续根据路由取值隐藏该命名空间 */}
        {isShowNopermission ? < NoPermissions /> : <UserManageRouters />}
      </div>
    </div>
  </div>;
}
