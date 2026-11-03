/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Menu } from 'antd';
import { useEffect, useState, useLayoutEffect, Fragment, useStore } from 'openinula';
import { useIntl } from 'inula-intl';
import '@/styles/components/menu.less';
import { Link, useLocation } from 'inula-router';
import { usermanageRouterPrefix } from '@/constant.js';
import { LeftOutlined, RightOutlined } from '@ant-design/icons';
import MultiRoleManageIcon from '@/assets/icons/MultiRoleManage';
import MultiUserManageIcon from '@/assets/icons/MultiUserManage';

function getUserItem(label, userKey, imgIcon, children, kind) {
  return {
    key: userKey,
    icon: imgIcon,
    children,
    label,
    type: kind,
  };
}

export default function MenuBar({ collapsed, handleUserCollapseChange }) {
  const location = useLocation();
  const intl = useIntl();
  const [openUserKeys, setOpenUserKeys] = useState([]);
  const [selectedUserKeys, setSelectedUserKeys] = useState([]);
  const themeStore = useStore('theme');
  const fixItems = [
    {
      key: 'user',
      name: intl.formatMessage({ id: 'userManage.userManage' }),
      icon: MultiUserManageIcon,
      path: '/user',
    },
    {
      key: 'role',
      name: intl.formatMessage({ id: 'userManage.roleManage' }),
      icon: MultiRoleManageIcon,
      path: '/role',
    },
  ];

  const getUserMenuItems = () => {
    const userItems = [...fixItems];
    return userItems;
  };

  let menuItems = getUserMenuItems();
  // 渲染菜单项
  const getUserIcon = (item) => item.icon(location.pathname.split('/')[2] === item.path.slice(1), themeStore.$s.theme);

  const getUserLink = (item, child = null) => {
    const path = child
      ? `/${usermanageRouterPrefix}${item.path}${child.path}`
      : `/${usermanageRouterPrefix}${item.path}`;
    return <Link to={path}>{child ? child.name : item.name}</Link>;
  };

  const getUserChildItems = (item) => item.children.map((child) => getUserItem(getUserLink(item, child), child.key));

  const getUserMenuItem = (item) => {
    if (item.children) {
      return getUserItem(item.name, item.key, getUserIcon(item), getUserChildItems(item));
    }
    return getUserItem(getUserLink(item), item.key, getUserIcon(item));
  };

  const userItems = menuItems.map(getUserMenuItem);

  const getUserLevelKeysList = (menuItem) => {
    const key = {};
    const func = (menuSubUserItem, level = 1) => {
      menuSubUserItem.forEach((item) => {
        if (item.key) {
          key[item.key] = level;
        }
        if (item.children) {
          func(item.children, level + 1);
        }
      });
    };
    func(menuItem);
    return key;
  };

  const keyUserLevelList = getUserLevelKeysList(userItems);

  const onOpenUserChange = (keys) => {
    const currentOpenKey = keys.find((key) => openUserKeys.indexOf(key) === -1);
    if (currentOpenKey !== undefined) {
      const repeatIndex = keys
        .filter((key) => key !== currentOpenKey)
        .findIndex((key) => keyUserLevelList[key] === keyUserLevelList[currentOpenKey]);
      setOpenUserKeys(
        keys
          .filter((_, index) => index !== repeatIndex)
          .filter((key) => keyUserLevelList[key] <= keyUserLevelList[currentOpenKey]), // 小于等于
      );
    } else {
      setOpenUserKeys(keys);
    }
  };

  // 判断屏幕尺寸
  useLayoutEffect(() => {
    //
    window.addEventListener('resize', () => {
      if (window.screen.width < 1280) {
        // 尺寸小于1280
        handleUserCollapseChange(true);
      } else {
        handleUserCollapseChange(false);
      }
    });

    // 首次小屏进入
    if (window.screen.width < 1280) {
      handleUserCollapseChange(true);
    }
  }, []);

  useEffect(() => {
    let [prefix, firstKey, linkKey, childKey] = location.pathname.split('/');
    // 例外
    let result = menuItems.find((item) => item.path.slice(1) === linkKey);
    if (result) {
      if (result.children) {
        setSelectedUserKeys([childKey]);
        setOpenUserKeys([linkKey]);
      } else {
        setSelectedUserKeys([linkKey]);
        setOpenUserKeys([]);
      }
    } else {
      setSelectedUserKeys([]);
      setOpenUserKeys([]);
    }
  }, [location]);

  const handleCollapseTrigger = () => {
    if (window.screen.width < 1280) {
      // 1280下禁止点击
      return;
    }
    handleUserCollapseChange(!collapsed); // 切换
  };

  return (
    <Fragment>
      <Menu
        mode="inline"
        style={{ paddingTop: '16px' }}
        className="menu"
        openKeys={openUserKeys}
        selectedKeys={selectedUserKeys}
        onOpenChange={onOpenUserChange}
        inlineCollapsed={collapsed}
        items={userItems}
      />
      <div className="collapse_arrow" onClick={handleCollapseTrigger}>
        {!collapsed ? <LeftOutlined /> : <RightOutlined />}
      </div>
    </Fragment>
  );
}
