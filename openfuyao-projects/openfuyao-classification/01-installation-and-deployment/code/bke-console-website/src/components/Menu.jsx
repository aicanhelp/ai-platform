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
import { useEffect, useState, useLayoutEffect, Fragment, useStore, useCallback } from 'openinula';
import '@/styles/components/menu.less';
import { Link, useLocation, useRouteMatch, useHistory } from 'inula-router';
import { containerRouterPrefix } from '@/constant.js';
import { LeftOutlined, RightOutlined } from '@ant-design/icons';
import OverviewIcon from '@/assets/icons/OverviewIcon';
import ApplicationIcon from '@/assets/icons/ApplicationIcon';
import ExtensionIcon from '@/assets/icons/ExtensionIcon';
import WorkloadIcon from '@/assets/icons/WorkloadIcon';
import NetworkIcon from '@/assets/icons/NetworkIcon';
import ConfigIcon from '@/assets/icons/ConfigIcon';
import LogIcon from '@/assets/icons/LogIcon';
import AlarmIcon from '@/assets/icons/AlarmIcon';
import MonitorIcon from '@/assets/icons/MonitorIcon';
import PlatformManageIcon from '@/assets/icons/PlatformManageIcon';
import UserManageIcon from '@/assets/icons/UserManageIcon';
import EventIcon from '@/assets/icons/EventIcon';
import ComputingPowerEngineIcon from '@/assets/icons/ComputingPowerEngineIcon';
import MonitoringDashboardIcon from '@/assets/icons/MonitoringDashboardIcon';
import ColocationIcon from '@/assets/icons/ColocationIcon';
import AppMarketIcon from '@/assets/icons/AppMarketIcon';
import VolcanoConfigIcon from '@/assets/icons/VolcanoConfigIcon';
import ResourceManageIcon from '@/assets/icons/ResourceManageIcon';
import NamespaceIcon from '@/assets/icons/NamespaceIcon';
import NodeIcon from '@/assets/icons/NodeIcon';
import CustomizeIcon from '@/assets/icons/CustomizeIcon';
import ClusterMemberIcon from '@/assets/icons/ClusterMemberIcon';
import ClusterRoleIcon from '@/assets/icons/ClusterRoleIcon';
import ClusterLifeIcon from '@/assets/icons/ClusterLifeIcon';
import RayIcon from '@/assets/icons/RayIcon';
import { expandComponent } from '@/common/constants';
import { useIntl } from 'inula-intl';
import { getConsolePluginDisplayName } from '@/i18n/utils';

export default function MenuBar({ collapsed, handleCollapseChange }) {
  const intl = useIntl();
  const location = useLocation();
  const history = useHistory();
  const [openKeys, setOpenKeys] = useState([]);
  const [selectedKeys, setSelectedKeys] = useState([]);
  const consolePluginStore = useStore('consolePlugins');
  const [collapsedHidden, setCollapsedHidden] = useState(false);
  const [items, setItems] = useState([]);

  function getItem(label, key, icon, children, type) {
    return {
      key,
      icon,
      children,
      label,
      type,
    };
  }

  const getPluginConfig = (item) => ({
    key: item.pluginName,
    isPlugin: true,
    name: getConsolePluginDisplayName(item, intl.locale),
    path: `/${item.pluginName}`,
    icon: ExtensionIcon,
    children: item.subPages
      ? item.subPages.map((sub) => ({
          key: sub.pageName,
          name: getConsolePluginDisplayName(sub, intl.locale),
          path: `/${sub.pageName}`,
        }))
      : null,
  });

  const themeStore = useStore('theme');
  const fixItems = [];

  const getMenuItems = () => {
    let temporyItems = [...fixItems];
    const plugins = consolePluginStore.$s.consolePlugins.filter(
      (item) => item.entrypoint === `/${containerRouterPrefix}` && item.enabled,
    );
    let clusterConfig = {};
    for (const plugin of plugins) {
      const pluginConfig = getPluginConfig(plugin);
      if (pluginConfig.key === expandComponent.Cluster) {
        clusterConfig = pluginConfig;
      }
    }

    // 最后执行
    if (Object.keys(clusterConfig).length) {
      clusterConfig.icon = ClusterLifeIcon;
      temporyItems.push(clusterConfig);
    }
    return temporyItems;
  };

  let menuItems = getMenuItems();
  // 渲染菜单项
  const getIcon = (item) => item.icon(location.pathname.split('/')[2] === item.path.slice(1), themeStore.$s.theme);

  const handleClickAnotherHistory = (e, child) => {
    if (!child) {
      window.postMessage({ type: 'relayRouter', isHome: true });
    }
  };

  const getLink = (item, child = null) => {
    const path = child ? `/${containerRouterPrefix}${item.path}${child.path}` : `/${containerRouterPrefix}${item.path}`;
    if (item.isPlugin) {
      return (
        <Link
          className={`${child ? child.key : item.key}_scrollView`}
          to={path}
          onClick={(e) => handleClickAnotherHistory(e, child)}
        >
          {child ? child.name : item.name}
        </Link>
      );
    }
    return (
      <Link className={`${child ? child.key : item.key}_scrollView`} to={path}>
        {child ? child.name : item.name}
      </Link>
    );
  };

  const getChildItems = (item, isCol = false) =>
    item.children.map((child) => {
      if (child.children) {
        return isCol
          ? getItem(child.name, child.key, getIcon(child), [
              getItem(child.name, null, null, [...getChildItems(child)], 'group'),
            ])
          : getItem(child.name, child.key, getIcon(child), getChildItems(child));
      }
      return child.icon ? getItem(getLink(child), child.key, getIcon(child)) : getItem(getLink(item, child), child.key);
    });

  const getMenuItem = (item, isCol = false) => {
    if (item.children) {
      if (item.type === 'group') {
        return getItem(item.name, item.key, null, getChildItems(item, isCol), 'group');
      }
      return isCol
        ? getItem(item.name, item.key, getIcon(item), [
            getItem(item.name, null, null, [...getChildItems(item)], 'group'),
          ])
        : getItem(item.name, item.key, getIcon(item), getChildItems(item));
    }
    return getItem(getLink(item), item.key, getIcon(item));
  };

  useEffect(() => {
    setItems([...menuItems.map((item) => getMenuItem(item, collapsed))]);
  }, [location, themeStore.$s.theme, collapsed]);

  const solveMenuScroll = (scrollItem) => {
    if (scrollItem) {
      const menuContainer = document.querySelector('.container_nav_bar');
      const domMenuKey = document.querySelector(`.${scrollItem}_scrollView`);
      const selectedMenuItemRect = domMenuKey.getBoundingClientRect();
      const menuContainerRect = menuContainer.getBoundingClientRect();
      const offsetTop = selectedMenuItemRect.top - menuContainerRect.top;
      if (offsetTop + selectedMenuItemRect.height > menuContainerRect.height) {
        menuContainer.scrollTop += offsetTop;
      } else if (offsetTop < 0) {
        menuContainer.scrollTop = offsetTop;
      }
    }
  };

  useEffect(() => {
    const [selectedMenuItem, ...resets] = selectedKeys;
    solveMenuScroll(selectedMenuItem);
  }, [selectedKeys]);

  const handlePassMessage = useCallback((event) => {
    if (event.data.type === 'subToMainMsg') {
      const url = event.data.isInsertMainPrefix ? `/${containerRouterPrefix}${event.data.url}` : event.data.url;
      history.push(url);
      solveMenuScroll(event.data.data);
    }
  }, []);

  useEffect(() => {
    window.addEventListener('message', handlePassMessage);
    return () => window.removeEventListener('message', handlePassMessage);
  }, []);

  const getLevelKeysList = (menuItem) => {
    const key = {};
    const func = (menuSubItem, level = 1) => {
      menuSubItem.forEach((item) => {
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

  const keyLevelList = getLevelKeysList(items);
  const onOpenChange = (keys) => {
    const currentOpenKey = keys.find((key) => openKeys.indexOf(key) === -1);
    if (currentOpenKey !== undefined) {
      const repeatIndex = keys
        .filter((key) => key !== currentOpenKey)
        .findIndex((key) => keyLevelList[key] === keyLevelList[currentOpenKey]);
      setOpenKeys(
        keys
          .filter((_, index) => index !== repeatIndex)
          .filter((key) => keyLevelList[key] <= keyLevelList[currentOpenKey]), // 小于等于
      );
    } else {
      setOpenKeys(keys);
    }
  };

  // 判断屏幕尺寸
  useLayoutEffect(() => {
    //
    window.addEventListener('resize', () => {
      if (window.innerWidth < 1280) {
        // 尺寸小于1280
        setCollapsedHidden(true);
        handleCollapseChange(true);
      } else {
        setCollapsedHidden(false);
        handleCollapseChange(false);
      }
    });

    // 首次小屏进入
    if (window.innerWidth < 1280) {
      setCollapsedHidden(true);
      handleCollapseChange(true);
    }
  }, []);

  useEffect(() => {
    let [prefix, firstKey, linkKey, childKey] = location.pathname.split('/');
    let result = menuItems.find((item) =>
      item.type === 'group'
        ? item.children.find((childItem) => childItem.path.slice(1) === linkKey)
        : item.path.slice(1) === linkKey,
    );
    if (result) {
      if (result.children) {
        if (linkKey === 'event' || linkKey === expandComponent.MonitoringDashboard) {
          childKey = linkKey;
        }
        setSelectedKeys([childKey]);
        window.screen.width < 1280 || collapsed ? setOpenKeys([]) : setOpenKeys([linkKey]);
      } else {
        setSelectedKeys([linkKey]);
        setOpenKeys([]);
      }
    } else {
      setSelectedKeys([]);
      setOpenKeys([]);
    }
  }, [location, collapsed]);

  const handleCollapseTrigger = () => {
    if (window.screen.width < 1280) {
      // 1280下禁止点击
      setCollapsedHidden(true);
      return;
    }
    handleCollapseChange(!collapsed); // 切换
  };

  const onSelectChange = (e) => {
    setSelectedKeys([e.key]);
  };

  return (
    <Fragment>
      <Menu
        mode="inline"
        className="menu"
        openKeys={openKeys}
        selectedKeys={selectedKeys}
        onOpenChange={onOpenChange}
        onSelect={onSelectChange}
        inlineCollapsed={collapsed}
        items={items}
      />
      {!collapsedHidden && (
        <div className="collapse_arrow" onClick={handleCollapseTrigger}>
          {!collapsed ? <LeftOutlined /> : <RightOutlined />}
        </div>
      )}
    </Fragment>
  );
}
