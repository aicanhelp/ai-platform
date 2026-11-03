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
import { Link, useLocation } from 'inula-router';
import { useState, useEffect } from 'openinula';
import { useIntl } from 'inula-intl';
import ExportIcon from '@/assets/images/menu/exportIcon.png';
import ExportSelectedIcon from '@/assets/images/menu/exportSelectedIcon.png';
import '@/styles/components/menu.less';

export default function LocalMenu() {
  const intl = useIntl();
  const location = useLocation();
  const [openKeys, setOpenKeys] = useState(['monitoring-dashboard']);
  const [selectedKeys, setSelectedKeys] = useState(['monitoring-dashboard']);

  const items = [
    {
      key: 'monitoring-dashboard',
      label: <Link to="/dashboard">{intl.formatMessage({ id: 'dashboard.title' })}</Link>,
      icon: <img src={ExportSelectedIcon} />,
    },
  ];

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

  useEffect(() => {
    let [prefix, firstKey, linkKey, childKey] = location.pathname.split('/');
    let [result, ...others] = items;
    if (result) {
      if (result.children) {
        setSelectedKeys([firstKey]);
        setOpenKeys(['monitoring-dashboard']);
      }
    } else {
      setSelectedKeys([]);
      setOpenKeys([]);
    }
  }, [location]);

  return (
    <Menu
      mode="inline"
      className="menu"
      openKeys={openKeys}
      selectedKeys={selectedKeys}
      onOpenChange={onOpenChange}
      items={items}
    />
  );
}
