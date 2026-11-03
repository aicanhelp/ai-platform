/**
 *  Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  openFuyao is licensed under Mulan PSL v2.
 *  You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  You may obtain a copy of Mulan PSL v2 at:
  
 *       http://license.coscl.org.cn/MulanPSL2
  
 *   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *   EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *   MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *   See the Mulan PSL v2 for more details.
 */

import '@/styles/pages/containerIndex.less';
import '@/styles/components/namespaceBar.less';
import MenuBar from '@/components/Menu.jsx';
import ContainerRouters from '@/routes/containerRouters.jsx';
import { useState, useEffect, useStore, Fragment } from 'openinula';
import '@/styles/components/navbar.less';
import NoPermissionsModal from '@/components/NoPermissionsModal';

export default function ContainerPlatFormIndex() {
  const themeStore = useStore('theme');
  const collapsedStore = useStore('collapsed');
  const [collapsed, setCollapsed] = useState(false); // 折叠
  const [noPermissionsModalOpen, setNoPermissionsModalOpen] = useState(false); // 无权限弹窗
  const handleCollapseTrigger = (status) => {
    setCollapsed(status); // 切换
    // 存入storage
    localStorage.setItem('collapsed', status);
    collapsedStore.$a.setCollapsed(status);
  };

  const handleClosePermissionModal = () => {
    setNoPermissionsModalOpen(false);
  };

  useEffect(() => {
    const collapsedStroage = localStorage.getItem('collapsed');
    if (collapsedStroage === 'true') {
      setCollapsed(true);
    }
  }, []);

  return <Fragment>
    <div className={`container_platform_box ${themeStore.$s.theme === 'dark' ? 'dark_box' : ''}`}>
      <div className='container_platform_box_content'>
        <div className={collapsed ? 'container_nav_bar collapse_self_bar' : 'container_nav_bar'}>
          <MenuBar collapsed={collapsed} handleCollapseChange={handleCollapseTrigger} />
        </div>
        <div className={collapsed ? 'container_content collapse_self_content' : 'container_content'}>
          {/* 后续根据路由取值隐藏该命名空间 */}

          <ContainerRouters />
        </div>
      </div>
      <NoPermissionsModal open={noPermissionsModalOpen} handleOpen={handleClosePermissionModal} />
    </div>
  </Fragment>;
}