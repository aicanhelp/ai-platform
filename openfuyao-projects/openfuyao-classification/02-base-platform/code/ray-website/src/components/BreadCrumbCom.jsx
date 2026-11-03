/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Breadcrumb } from 'antd';
import { Link } from 'inula-router';

export default function BreadCrumbCom({
  items,
  className = 'breadCrumb_self',
}) {
  const itemRender = (currentRoute, params, itemList, paths) => {
    const isLast = currentRoute?.path === itemList[itemList.length - 1]?.path;
    return (isLast || currentRoute.disabled) ? (
      <span >{currentRoute.title}</span>
    ) : (
      <Link to={`/${paths.join('')}`}>{currentRoute.title}</Link>
    );
  };

  return <Breadcrumb itemRender={itemRender} items={items} className={className} />;
}