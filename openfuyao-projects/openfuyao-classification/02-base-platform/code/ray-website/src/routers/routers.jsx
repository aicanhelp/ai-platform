/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import RayOverviewPage from '@/pages/overview/RayOverview';
import RayClusterPage from '@/pages/cluster/RayCluster';
import RayClusterAddConfig from '@/pages/cluster/RayClusterAddConfig';
import RayClusterDetail from '@/pages/cluster/RayClusterDetail';
import RayServicePage from '@/pages/service/RayService';
import RayAddConfig from '@/pages/service/RayAddConfig';
import RayServiceDetail from '@/pages/service/RayServiceDetail';
import RayJobPage from '@/pages/job/RayJob';
import RayJobAddConfig from '@/pages/job/RayJobAddConfig';
import RayJobDetail from '@/pages/job/RayJobDetail';

const routers = [
  {
    path: '/RayOverview',
    component: RayOverviewPage,
  },
  {
    path: '/RayCluster',
    component: RayClusterPage,
  },
  {
    path: '/RayCluster/addConfig',
    component: RayClusterAddConfig,
  },
  {
    path: '/RayCluster/detail',
    component: RayClusterDetail,
  },
  {
    path: '/RayService',
    component: RayServicePage,
  },
  {
    path: '/RayService/detail',
    component: RayServiceDetail,
  },
  {
    path: '/RayService/addConfig',
    component: RayAddConfig,
  },
  {
    path: '/RayJob',
    component: RayJobPage,
  },
  {
    path: '/RayJob/addConfig',
    component: RayJobAddConfig,
  },
  {
    path: '/RayJob/detail',
    component: RayJobDetail,
  },
];

export default routers;
