/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { NavLink } from 'inula-router';
import '@/styles/navbar.less';

export default function NavBar() {
  const navItems = [
  ];

  return (
    <div className='nav'>
      <div className='navList'>
        {
          navItems.map((item) => (
            <NavLink
              to={item.path}
              isActive={() => item.name === window.location.pathname.split('/')[2]}
              activeClassName='active'
            >
              {item.displayName}
            </NavLink>
          ))
        }
      </div>
    </div>
  );
}