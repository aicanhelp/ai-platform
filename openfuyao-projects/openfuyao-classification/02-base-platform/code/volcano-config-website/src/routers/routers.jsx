/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import NumaList from '@/pages/NUMA/NumaList';
import NumaDetails from '@/pages/NUMA/NumaDetails';
import SchedulingPolicyConfiguration from '@/pages/SchedulingPolicyConfiguration';
import AffinityPolicyConfiguration from '@/pages/AffinityPolicyConfiguration';
import PriorityClass from '@/pages/SchedulingPolicyConfiguration/PriorityClass';
import PriorityClassDetails from '@/pages/SchedulingPolicyConfiguration/PriorityClassDetails';
import CreatePriority from '@/pages/SchedulingPolicyConfiguration/CreatePriority';
import PriorityClassUpdate from '@/pages/SchedulingPolicyConfiguration/PriorityClassUpdate';
import NumaOverview from '@/pages/NumaOverview';
const routers = [
  {
    path: '/numaMonitor',
    component: NumaList,
  },
  {
    path: '/numaMonitor/numaDetails/:numaDetailName?',
    component: NumaDetails,
  },
  {
    path: '/SchedulingPolicyConfiguration',
    component: SchedulingPolicyConfiguration
  },
  {
    path: '/SchedulingPolicyConfiguration/priorityClass',
    component: PriorityClass,
  },
  {
    path: '/SchedulingPolicyConfiguration/priorityClassDetails',
    component: PriorityClassDetails,
  },
  {
    path: '/SchedulingPolicyConfiguration/priorityClassUpdate',
    component: PriorityClassUpdate,
  },
  {
    path: '/SchedulingPolicyConfiguration/createPriority',
    component: CreatePriority
  },
  {
    path: '/AffinityPolicyConfiguration',
    component: AffinityPolicyConfiguration
  },
  {
    path: '/NumaOverview',
    component: NumaOverview
  },
];

export default routers;
