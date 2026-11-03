/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import BreadCrumbCom from '../components/BreadCrumbCom';
import { containerRouterPrefix } from '../common/constants';

export default function ColocationRulesManagementPage() {
  return <div className="child_content">
    <BreadCrumbCom className="create_bread" items={[
      { title: '在离线混部 / 混部工作负载管理', path: `/${containerRouterPrefix}/colocation/ColocationWorkloadManagement` },
    ]} />
  </div >;
}