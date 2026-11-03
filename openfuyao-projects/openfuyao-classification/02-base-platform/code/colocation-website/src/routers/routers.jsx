/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import Inula from 'openinula';
import ColocationOverviewPage from '@/pages/newDesign/overview/Index';
import ColocationMonitorPage from '@/pages/newDesign/monitor/Index';
import ColocationConfigurationPage from '@/pages/newDesign/configuration/Index';
import ColocationNodeManagementPage from '../pages/ColocationNodeManagementPage';
import ColocationWorkloadManagementPage from '../pages/ColocationWorkloadManagementPage';
import ColocationRulesManagementPage from '../pages/ColocationRuleManagementPage';
import RuleDetail from '../pages/rule/RuleDetail';
import RuleCreate from '../pages/rule/RuleCreate';
import { containerRouterPrefix } from '@/common/constants';

const routers = [
  {
    path: '/ColocationOverview',
    component: ColocationOverviewPage,
  },
  {
    path: '/ColocationMonitor',
    component: ColocationMonitorPage,
  },
  {
    path: '/ColocationConfiguration',
    component: ColocationConfigurationPage,
  },
  {
    path: '/ColocationNodeManagement',
    component: ColocationNodeManagementPage,
  },
  {
    path: '/ColocationWorkloadManagement',
    component: ColocationWorkloadManagementPage,
  },
  {
    path: '/ColocationRulesManagement',
    component: ColocationRulesManagementPage,
  },
  {
    path: `/ColocationRulesManagement/detail/:rule_name?/:editWay?/:onlyRead?`,
    component: RuleDetail,
  },
  {
    path: `/ColocationRulesManagement/create/:way?/:type?`,
    component: RuleCreate,
  },
];

export default routers;
