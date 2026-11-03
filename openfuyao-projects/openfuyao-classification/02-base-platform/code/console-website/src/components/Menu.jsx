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
import { useEffect, useState, useLayoutEffect, Fragment, useStore, useCallback, useMemo } from 'openinula';
import '@/styles/components/menu.less';
import { Link, useLocation, useHistory } from 'inula-router';
import { containerRouterPrefix } from '@/constant.js';
import { LeftOutlined, RightOutlined } from '@ant-design/icons';
import OverviewIcon from '@/assets/icons/OverviewIcon';
import ApplicationIcon from '@/assets/icons/ApplicationIcon';
import ExtensionIcon from '@/assets/icons/ExtensionIcon';
import WorkloadIcon from '@/assets/icons/WorkloadIcon';
import NetworkIcon from '@/assets/icons/NetworkIcon';
import ConfigIcon from '@/assets/icons/ConfigIcon';
import LogIcon from '@/assets/icons/LogIcon';
import AlertIcon from '@/assets/icons/AlertIcon';
import MonitorIcon from '@/assets/icons/MonitorIcon';
import UserManageIcon from '@/assets/icons/UserManageIcon';
import EventIcon from '@/assets/icons/EventIcon';
import ComputingPowerEngineIcon from '@/assets/icons/ComputingPowerEngineIcon';
import MonitoringDashboardIcon from '@/assets/icons/MonitoringDashboardIcon';
import ColocationIcon from '@/assets/icons/ColocationIcon';
import AppMarketIcon from '@/assets/icons/AppMarketIcon';
import VolcanoConfigIcon from '@/assets/icons/VolcanoConfigIcon';
import StorageIcon from '@/assets/icons/StorageIcon';
import NamespaceIcon from '@/assets/icons/NamespaceIcon';
import NodeIcon from '@/assets/icons/NodeIcon';
import CustomizeIcon from '@/assets/icons/CustomizeIcon';
import ClusterMemberIcon from '@/assets/icons/ClusterMemberIcon';
import ClusterRoleIcon from '@/assets/icons/ClusterRoleIcon';
import ClusterLifeIcon from '@/assets/icons/ClusterLifeIcon';
import RayIcon from '@/assets/icons/RayIcon';
import MisManagementIcon from '@/assets/icons/MisManagementIcon';
import { expandComponent } from '@/common/constants';
import { useTranslation } from 'react-i18next';
import { getConsolePluginDisplayName } from '@/i18n/utils';

export default function MenuBar({ collapsed, handleCollapseChange }) {
  const { t, i18n } = useTranslation('menu');
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
    name: getConsolePluginDisplayName(item, i18n.language),
    path: `/${item.pluginName}`,
    icon: ExtensionIcon,
    children: item.subPages ? item.subPages.map(sub => ({
      key: sub.pageName,
      name: getConsolePluginDisplayName(sub, i18n.language),
      path: `/${sub.pageName}`,
    })) : null,
  });

  const themeStore = useStore('theme');
  const fixItems = useMemo(() => ([
    {
      key: 'overview',
      name: t('overview'),
      icon: OverviewIcon,
      path: '/overview',
    },
    {
      key: 'appMarket',
      name: t('appMarket.title'),
      icon: AppMarketIcon,
      path: '/appMarket',
      children: [
        {
          key: 'appOverview',
          name: t('appMarket.overview'),
          path: '/appOverview',
        },
        {
          key: 'marketCategory',
          name: t('appMarket.appList'),
          path: '/marketCategory',
        },
        {
          key: 'stash',
          name: t('appMarket.repoConfig'),
          path: '/stash/wareHouse',
        },
      ],
    },
    {
      key: 'applicationManageHelm',
      name: t('appManagement'),
      icon: ApplicationIcon,
      path: '/applicationManageHelm',
    },
    {
      key: 'extendManage',
      name: t('extensionManagement'),
      icon: ExtensionIcon,
      path: '/extendManage',
    },
    {
      key: 'storage',
      name: t('resourceManagement'),
      type: 'group',
      children: [
        {
          key: 'workload',
          name: t('workload.title'),
          icon: WorkloadIcon,
          path: '/workload',
          children: [
            {
              key: 'pod',
              name: t('workload.pod'),
              path: '/pod',
            },
            {
              key: 'deployment',
              name: t('workload.deployment'),
              path: '/deployment',
            },
            {
              key: 'statefulSet',
              name: t('workload.statefulSet'),
              path: '/statefulSet',
            },
            {
              key: 'daemonSet',
              name: t('workload.daemonSet'),
              path: '/daemonSet',
            },
            {
              key: 'job',
              name: t('workload.job'),
              path: '/job',
            },
            {
              key: 'cronJob',
              name: t('workload.cronJob'),
              path: '/cronJob',
            },
          ],
        },
        {
          key: 'network',
          name: t('network.title'),
          icon: NetworkIcon,
          path: '/network',
          children: [
            {
              key: 'service',
              name: t('network.service'),
              path: '/service',
            },
            {
              key: 'ingress',
              name: t('network.ingress'),
              path: '/ingress',
            },
          ],
        },
        {
          key: 'storage',
          name: t('storage.title'),
          icon: StorageIcon,
          path: '/storage',
          children: [
            {
              key: 'pv',
              name: t('storage.pv'),
              path: '/pv',
            },
            {
              key: 'pvc',
              name: t('storage.pvc'),
              path: '/pvc',
            },
            {
              key: 'sc',
              name: t('storage.sc'),
              path: '/sc',
            },
          ],
        },
        {
          key: 'nodeManage',
          name: t('node'),
          icon: NodeIcon,
          path: '/nodeManage',
        },
        {
          key: 'configuration',
          name: t('configuration.title'),
          icon: ConfigIcon,
          path: '/configuration',
          children: [
            {
              key: 'configMap',
              name: t('configuration.configMap'),
              path: '/configMap',
            },
            {
              key: 'secret',
              name: t('configuration.secret'),
              path: '/secret',
            },
          ],
        },
        {
          key: 'namespace',
          name: t('namespace.title'),
          path: '/namespace',
          icon: NamespaceIcon,
          children: [
            {
              key: 'namespaceManage',
              name: t('namespace.namespace'),
              path: '/namespaceManage',
            },
            {
              key: 'limitRange',
              name: t('namespace.limitRange'),
              path: '/limitRange',
            },
            {
              key: 'resourceQuota',
              name: t('namespace.resourceQuota'),
              path: '/resourceQuota',
            },
          ],
        },
        {
          key: 'customResourceDefinition',
          name: t('customResource'),
          icon: CustomizeIcon,
          path: '/customResourceDefinition',
        },
      ],
    },
    {
      key: 'observeCenter',
      name: t('observability.title'),
      type: 'group',
      children: [
        {
          key: 'monitor',
          name: t('observability.monitor.title'),
          icon: MonitorIcon,
          path: '/monitor',
          children: [
            {
              key: 'monitorDashboard',
              name: t('observability.monitor.dashboard'),
              path: '/monitorDashboard',
            },
            {
              key: 'monitorGoalManage',
              name: t('observability.monitor.targets'),
              path: '/monitorGoalManage',
            },
            {
              key: 'monitorRuleManage',
              name: t('observability.monitor.rules'),
              path: '/monitorRuleManage',
            },
          ],
        },
        {
          key: 'alert',
          name: t('observability.alert.title'),
          icon: AlertIcon,
          path: '/alert',
          children: [
            {
              key: 'alertIndex',
              name: t('observability.alert.current'),
              path: '/alertIndex',
            },
            {
              key: 'silentAlert',
              name: t('observability.alert.silent'),
              path: '/silentAlert',
            },
          ],
        },
        {
          key: 'event',
          name: t('observability.event'),
          icon: EventIcon,
          path: '/event',
        },
      ],
    },
    {
      key: 'roleManagementCenter',
      name: t('rbac.title'),
      type: 'group',
      children: [
        {
          key: 'clusterUser',
          name: t('rbac.clusterRole'),
          icon: ClusterRoleIcon,
          path: '/clusterUser',
        },
        {
          key: 'clusterMember',
          name: t('rbac.clusterMember'),
          icon: ClusterMemberIcon,
          path: '/clusterMember',
        },
        {
          key: 'userManage',
          name: t('rbac.rbacManagement.title'),
          icon: UserManageIcon,
          path: '/userManage',
          children: [
            {
              key: 'serviceAccount',
              name: t('rbac.rbacManagement.serviceAccount'),
              path: '/serviceAccount',
            },
            {
              key: 'role',
              name: t('rbac.rbacManagement.role'),
              path: '/role',
            },
            {
              key: 'roleBinding',
              name: t('rbac.rbacManagement.roleBinding'),
              path: '/roleBinding',
            },
          ],
        },
      ],
    },
  ]), [t, i18n.language]);

  const getMenuItems = () => {
    let temporyItems = [...fixItems];
    const plugins = consolePluginStore.$s.consolePlugins.filter(item => item.entrypoint === `/${containerRouterPrefix}` && item.enabled);
    const isExistComputing = plugins.filter(item =>
      item.pluginName === expandComponent.ComputingPowerEngine || item.pluginName === expandComponent.Volcano ||
      item.pluginName === expandComponent.Colocation || item.pluginName === expandComponent.Ray);
    // 按 key 定位插入点，避免菜单项增删后索引错位
    let computingGroup = null;
    if (isExistComputing.length) {
      const observeIndex = temporyItems.findIndex(indexItem => indexItem.key === 'observeCenter');
      if (observeIndex >= 0) {
        computingGroup = {
          key: 'computingOptimizationCenter',
          name: t('computingOptimization'),
          type: 'group',
          children: [],
        };
        temporyItems.splice(observeIndex, 0, computingGroup);
      }
    }
    let clusterConfig = {};
    plugins.sort((a, b) => a.pluginName.localeCompare(b.pluginName));
    for (const plugin of plugins) {
      const pluginConfig = getPluginConfig(plugin);
      if (pluginConfig.key === expandComponent.Logging) {
        // 更换日志icon
        pluginConfig.icon = LogIcon;
        const index = temporyItems.findIndex(indexItem => indexItem.key === 'observeCenter');
        if (index < 0) { continue; }
        temporyItems[index].children.splice(-2, 0, pluginConfig);
      } else if (pluginConfig.key === expandComponent.ComputingPowerEngine) {
        pluginConfig.icon = ComputingPowerEngineIcon;
        computingGroup?.children.unshift(pluginConfig);
      } else if (pluginConfig.key === expandComponent.MonitoringDashboard) {
        pluginConfig.icon = MonitoringDashboardIcon;
        const index = temporyItems.findIndex(indexItem => indexItem.key === 'observeCenter');
        if (index < 0) { continue; }
        temporyItems[index].children.splice(1, 0, pluginConfig);
      } else if (pluginConfig.key === expandComponent.Cluster) {
        clusterConfig = pluginConfig;
      } else if (pluginConfig.key === expandComponent.Colocation) {
        pluginConfig.icon = ColocationIcon;
        computingGroup?.children.splice(1, 0, pluginConfig);
      } else if (pluginConfig.key === expandComponent.Volcano) {
        pluginConfig.icon = VolcanoConfigIcon;
        computingGroup?.children.push(pluginConfig);
      } else if (pluginConfig.key === expandComponent.Ray) {
        pluginConfig.icon = RayIcon;
        computingGroup?.children.push(pluginConfig);
      } else if (pluginConfig.key === expandComponent.MisManagement) {
        pluginConfig.icon = MisManagementIcon;
        const index = temporyItems.findIndex(indexItem => indexItem.key === 'storage');
        if (index < 0) { continue; }
        temporyItems[index].children.push(pluginConfig);
      } else {
        const index = temporyItems.findIndex(indexItem => indexItem.key === 'storage');
        if (index < 0) { continue; }
        temporyItems[index].children.push(pluginConfig);
      }
    }

    // 最后执行
    if (Object.keys(clusterConfig).length) {
      clusterConfig.icon = ClusterLifeIcon;
      const storageIndex = temporyItems.findIndex(indexItem => indexItem.key === 'storage');
      if (storageIndex >= 0) {
        temporyItems.splice(storageIndex, 0, clusterConfig);
      }
    }
    return temporyItems;
  };

  const menuItems = useMemo(() => getMenuItems(), [fixItems, consolePluginStore.$s.consolePlugins]);
  // 渲染菜单项
  const getIcon = (item) => (
    item.icon(location.pathname.split('/')[2] === item.path.slice(1), themeStore.$s.theme)
  );

  const handleClickAnotherHistory = (item, child) => {
    if (!child && item?.isPlugin) {
      window.postMessage({ type: 'relayRouter', isHome: true, pluginName: item.key });
    }
  };

  const getLink = (item, child = null) => {
    const path = child ? `/${containerRouterPrefix}${item.path}${child.path}` : `/${containerRouterPrefix}${item.path}`;
    if (item.isPlugin) {
      return <Link className={`${child ? child.key : item.key}_scrollView`} to={path} onClick={() => handleClickAnotherHistory(item, child)}>{child ? child.name : item.name}</Link>;
    }
    return <Link className={`${child ? child.key : item.key}_scrollView`} to={path}>{child ? child.name : item.name}</Link>;
  };

  const getChildItems = (item, isCol = false) => (
    item.children.map(child => {
      if (child.children) {
        return isCol ? getItem(child.name, child.key, getIcon(child), [getItem(child.name, child.key, null, [...getChildItems(child)], 'group')]) : getItem(child.name, child.key, getIcon(child), getChildItems(child));
      }
      return child.icon ? getItem(getLink(child), child.key, getIcon(child)) : getItem(getLink(item, child), child.key);
    },
    )
  );

  const getMenuItem = (item, isCol = false) => {
    if (item.children) {
      if (item.type === 'group') {
        return getItem(item.name, item.key, null, getChildItems(item, isCol), 'group');
      }
      return isCol
        ? getItem(item.name, item.key, getIcon(item), [getItem(item.name, item.key, null, [...getChildItems(item)], 'group')])
        : getItem(item.name, item.key, getIcon(item), getChildItems(item));
    }
    return getItem(getLink(item), item.key, getIcon(item));
  };

  useEffect(() => {
    setItems([...menuItems.map(item => getMenuItem(item, collapsed))]);
  }, [location, themeStore.$s.theme, collapsed, t, i18n.language, consolePluginStore.$s.consolePlugins]);

  const solveMenuScroll = scrollItem => {
    if (scrollItem) {
      const menuContainer = document.querySelector('.container_nav_bar');
      const domMenuKey = document.querySelector(`.${scrollItem}_scrollView`);
      if (!menuContainer || !domMenuKey) {
        return;
      }
      const selectedMenuItemRect = domMenuKey.getBoundingClientRect();
      const menuContainerRect = menuContainer.getBoundingClientRect();
      const offsetTop = selectedMenuItemRect.top - menuContainerRect.top;
      if (offsetTop + selectedMenuItemRect.height > menuContainerRect.height) {
        menuContainer.scrollTop += offsetTop;
      } else if (offsetTop < 0) {
        menuContainer.scrollTop = offsetTop;
      } else {
        menuContainer.scrollTop = 0;
      }
    }
  };

  useEffect(() => {
    const [selectedMenuItem] = selectedKeys;
    solveMenuScroll(selectedMenuItem);
  }, [selectedKeys]);

  const handlePassMessage = useCallback(event => {
    // 三方脚本可能 postMessage 任意字符串/非对象数据，先校验类型避免 event.data.type 抛错
    if (event.data && typeof event.data === 'object' && event.data.type === 'subToMainMsg') {
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

  const keyLevelList = useMemo(() => getLevelKeysList(items), [items]);
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
    const handleResize = () => {
      if (window.innerWidth < 1280) { // 尺寸小于1280
        setCollapsedHidden(true);
        handleCollapseChange(true);
      } else {
        setCollapsedHidden(false);
        handleCollapseChange(false);
      }
    };
    window.addEventListener('resize', handleResize);

    // 首次小屏进入
    if (window.innerWidth < 1280) {
      setCollapsedHidden(true);
      handleCollapseChange(true);
    }
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    let [, , linkKey, childKey] = location.pathname.split('/');
    // 例外
    if (childKey === 'clusterRole') {
      childKey = 'role';
    }
    if (childKey === 'clusterRoleBinding') {
      childKey = 'roleBinding';
    }
    if (linkKey === 'nodeManage' || linkKey === 'customResourceDefinition') {
      childKey = linkKey;
    }
    if (linkKey === 'clusterUser' || linkKey === 'clusterMember') {
      childKey = linkKey;
    }
    let result = menuItems.find(item => item.type === 'group' ? item.children.find(childItem => childItem.path.slice(1) === linkKey) : item.path.slice(1) === linkKey);
    if (result) {
      if (result.children) {
        if (linkKey === 'event' || linkKey === expandComponent.MonitoringDashboard) {
          childKey = linkKey;
        }
        setSelectedKeys([childKey]);
        window.innerWidth < 1280 || collapsed ? setOpenKeys([]) : setOpenKeys([linkKey]);
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
    if (window.innerWidth < 1280) { // 1280下禁止点击
      setCollapsedHidden(true);
      return;
    }
    handleCollapseChange(!collapsed); // 切换
  };

  const onSelectChange = (e) => {
    setSelectedKeys([e.key]);
  };

  return <Fragment>
    <Menu
      mode='inline'
      className='menu'
      openKeys={openKeys}
      selectedKeys={selectedKeys}
      onOpenChange={onOpenChange}
      onSelect={onSelectChange}
      inlineCollapsed={collapsed}
      items={items}
    />
    {!collapsedHidden && <div className='collapse_arrow' onClick={handleCollapseTrigger}>{!collapsed ? <LeftOutlined /> :
      <RightOutlined />}</div>}
  </Fragment>;
}