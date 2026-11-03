/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Route, Switch, Redirect } from 'inula-router';
import { useStore } from 'openinula';
import { containerRouterPrefix } from '@/constant.js';

import MicroAppPage from '@/components/MicroApp';
import OverviewPage from '@/pages/container/overview';

// 应用市场
import Market from '@/pages/applicationMarket/Market';
import ApplicationDetails from '@/pages/applicationMarket/ApplicationDetails';
import Deploy from '@/pages/applicationMarket/Deploy';
import OneClickDeploy from '@/pages/applicationMarket/OneClickDeploy';
import MarketCategory from '@/pages/applicationMarket/MarketCategory';
import Stash from '@/pages/container/platformManage/stash';
import StashDetails from '@/pages/container/platformManage/stash/component/StashDetails';
import PackageManagementDetail from '@/pages/applicationMarket/component/PackageManagementDetail';

// 应用管理 & 扩展组件管理
import HelmPage from '@/pages/container/applicationManage/helm/HelmIndex';
import HelmDetail from '@/pages/container/applicationManage/helm/detail/HelmDetail';
import HelmUpgrade from '@/pages/container/applicationManage/helm/HelmUpgrade';
import ExtendPage from '@/pages/container/extendManage/extend/ExtendIndex';
import ExtendUpgrade from '@/pages/container/extendManage/extend/ExtendUpgrade';
import ExtendDetail from '@/pages/container/extendManage/extend/detail/ExtendDetail';

// 工作负载
import Pod from '@/pages/container/workload/pod/Index';
import PodDetail from '@/pages/container/workload/pod/detail/Index';
import PodCreate from '@/pages/container/workload/pod/PodCreate';
import Deployment from '@/pages/container/workload/deployment/Index.jsx';
import DeploymentDetail from '@/pages/container/workload/deployment/detail/Index';
import DeploymentCreate from '@/pages/container/workload/deployment/DeploymentCreate';
import StatefulSet from '@/pages/container/workload/statefulSet/Index';
import StatefulSetDetail from '@/pages/container/workload/statefulSet/detail/Index';
import StatefulSetCreate from '@/pages/container/workload/statefulSet/StatefulSetCreate';
import DaemonSet from '@/pages/container/workload/daemonSet/Index';
import DaemonSetDetail from '@/pages/container/workload/daemonSet/detail/Index';
import DaemonSetCreate from '@/pages/container/workload/daemonSet/DaemonSetCreate';
import Job from '@/pages/container/workload/job/Index';
import JobDetail from '@/pages/container/workload/job/detail/Index';
import JobCreate from '@/pages/container/workload/job/JobCreate';
import CronJob from '@/pages/container/workload/cronJob/Index';
import CronJobDetail from '@/pages/container/workload/cronJob/detail/Index';
import CronJobCreate from '@/pages/container/workload/cronJob/CronJobCreate';

// 网络
import Service from '@/pages/container/network/service/Index';
import ServiceDetail from '@/pages/container/network/service/detail/Index';
import ServiceCreate from '@/pages/container/network/service/ServiceCreate';
import Ingress from '@/pages/container/network/ingress/Index';
import IngressDetail from '@/pages/container/network/ingress/detail/Index';
import IngressCreate from '@/pages/container/network/ingress/IngressCreate';

// 配置
import ConfigMap from '@/pages/container/configuration/configMap';
import Secret from '@/pages/container/configuration/secret';
import SecretCreate from '@/pages/container/configuration/secret/SecretCreate';
import ConfigMapCreate from '@/pages/container/configuration/configMap/ConfigMapCreate';
import ConfigMapDetail from '@/pages/container/configuration/configMap/detail/Index';
import SecretDetail from '@/pages/container/configuration/secret/detail/Index';

// 告警
import AlertIndex from '@/pages/container/alert/AlertIndex';
import SilentAlert from '@/pages/container/alert/SilentAlert';
import AlertDetail from '@/pages/container/alert/AlertDetail';
import SilentDetail from '@/pages/container/alert/SilentDetail';

// 监控
import MonitorHomePage from '@/pages/container/monitor/monitorDashboard/MonitorHome';
import MonitorGoalList from '@/pages/container/monitor/monitorGoalManage/MonitorGoalList';
import MonitorServiceMonitor from '@/pages/container/monitor/monitorGoalManage/MonitorServiceMonitor';
import MonitorServiceDetail from '@/pages/container/monitor/monitorGoalManage/MonitorServiceDetail';
import MonitorRuleList from '@/pages/container/monitor/monitorRuleManage/MonitorRuleList';
import MonitorRuleDetail from '@/pages/container/monitor/monitorRuleManage/MonitorRuleDetail';
import CustomizeMonitorQuery from '@/pages/container/monitor/monitorDashboard/CustomizeMonitorQuery';
import MonitorCreateService from '@/pages/container/monitor/monitorGoalManage/MonitorCreateService';

// RBAC
import ServiceAccount from '@/pages/container/userManage/serviceAccount/Index';
import ServiceAccountCreate from '@/pages/container/userManage/serviceAccount/ServiceAccountCreate';
import ServiceAccountDetail from '@/pages/container/userManage/serviceAccount/detail/Index';
import Role from '@/pages/container/userManage/role/Index';
import RoleDetail from '@/pages/container/userManage/role/role/Index';
import RoleCreate from '@/pages/container/userManage/role/RoleCreate';
import ClusterRoleDetail from '@/pages/container/userManage/role/clusterRole/Index';
import ClusterRoleCreate from '@/pages/container/userManage/role/ClusterRoleCreate';
import RoleBinding from '@/pages/container/userManage/roleBinding/Index';
import RoleBindCreate from '@/pages/container/userManage/roleBinding/RoleBindCreate';
import RoleBindDetail from '@/pages/container/userManage/roleBinding/roleBinding/Index';
import ClusterRoleBindCreate from '@/pages/container/userManage/roleBinding/ClusterRoleBindCreate';
import ClusterRoleBindDetail from '@/pages/container/userManage/roleBinding/clusterRoleBind/Index';

// 事件
import Events from '@/pages/container/event/Index';

// 管理
import Node from '@/pages/container/platformManage/node/Index';
import NodeDetail from '@/pages/container/platformManage/node/detail/Index';
import LimitRange from '@/pages/container/platformManage/limitRange/Index';
import LimitRangeCreate from '@/pages/container/platformManage/limitRange/LimitRangeCreate';
import LimitRangeDetail from '@/pages/container/platformManage/limitRange/detail/Index';
import Namespace from '@/pages/container/platformManage/namespace/Index';
import NamespaceDetail from '@/pages/container/platformManage/namespace/detail/Index';
import NamespaceCreate from '@/pages/container/platformManage/namespace/NamespaceCreate';
import ResourceQuota from '@/pages/container/platformManage/resourceQuota/Index';
import ResourceQuotaCreate from '@/pages/container/platformManage/resourceQuota/ResourceQuotaCreate';
import ResourceQuotaDetail from '@/pages/container/platformManage/resourceQuota/detail/Index';
import CrdIndex from '@/pages/container/crd/Index';
import CrdCreate from '@/pages/container/crd/CrdCreate';
import CrdDetail from '@/pages/container/crd/detail/Index';
import CrDetail from '@/pages/container/crd/crDetail/Index';
import CrCreate from '@/pages/container/crd/detail/CrCreate';
import ClusterUser from '@/pages/container/userManage/clusterUser';
import ClusterMember from '@/pages/container/userManage/clusterMember';

// 存储
import Pv from '@/pages/container/storage/pv/Index';
import PvDetail from '@/pages/container/storage/pv/detail/Index';
import PvCreate from '@/pages/container/storage/pv/PvCreate';
import Pvc from '@/pages/container/storage/pvc/Index';
import PvcDetail from '@/pages/container/storage/pvc/detail/Index';
import PvcCreate from '@/pages/container/storage/pvc/PvcCreate';
import Sc from '@/pages/container/storage/sc/Index';
import ScDetail from '@/pages/container/storage/sc/detail/Index';
import ScCreate from '@/pages/container/storage/sc/ScCreate';

export default function ContainerRouters() {
  const consolePluginStore = useStore('consolePlugins');

  const getPluginRoute = (item) => {
    return item.subPages ?
      item.subPages.map((sub) => (<Route
        path={`/${containerRouterPrefix}/${item.pluginName}/${sub.pageName}`}
        render={props => <MicroAppPage {...props} key={sub.pageName} subKey={props.location.pathname} name={item.pluginName} />}
      />)) :
      <Route
        path={`/${containerRouterPrefix}/${item.pluginName}`}
        render={props => <MicroAppPage {...props} key={item.pluginName} subKey={props.location.pathname} name={item.pluginName} />}
      />;
  };

  return (
    <Switch>
      {
        consolePluginStore.$s.consolePlugins.filter(item => item.entrypoint === `/${containerRouterPrefix}` && item.enabled)
          .map((item) => getPluginRoute(item))
      }
      <Route
        exact
        path={`/${containerRouterPrefix}/overview`}
        component={OverviewPage}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/appMarket/appOverview`}
        component={Market}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/appMarket/marketCategory/ApplicationDetails/:chart?/:repo?/:versionRepo?`}
        component={ApplicationDetails}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/appMarket/marketCategory/Deploy/:repo?/:chart?/:versionSelect?/:defaultNameSpace?`}
        component={Deploy}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/appMarket/oneClickDeploy`}
        component={OneClickDeploy}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/appMarket/marketCategory/:scene?/:isFuyaoExtension?/:isCompute?`}
        component={MarketCategory}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/appMarket/stash/:tabIndex?`}
        component={Stash}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/appMarket/stash/wareHouse/stashDetails/:repo?`}
        component={StashDetails}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/appMarket/stash/packageManageDetails/:repo?/:chart?`}
        component={PackageManagementDetail}
      />
      <Redirect
        path={`/${containerRouterPrefix}/appMarket`}
        to={`/${containerRouterPrefix}/appMarket/appOverview`}
      />
      {/** 应用管理 */}
      <Route
        exact
        path={`/${containerRouterPrefix}/applicationManageHelm`}
        component={HelmPage}
      />
      <Route
        path={`/${containerRouterPrefix}/applicationManageHelm/:helm_namespace?/:helm_name?/upgrade`}
        component={HelmUpgrade}
      />
      <Route
        path={`/${containerRouterPrefix}/applicationManageHelm/:helm_namespace?/:helm_name?`}
        component={HelmDetail}
      />
      {/** 拓展管理 */}
      <Route
        exact
        path={`/${containerRouterPrefix}/extendManage`}
        component={ExtendPage}
      />
      <Route
        path={`/${containerRouterPrefix}/extendManage/:extend_namespace?/:extend_name?/upgrade`}
        component={ExtendUpgrade}
      />
      <Route
        path={`/${containerRouterPrefix}/extendManage/:extend_namespace?/:extend_name?`}
        component={ExtendDetail}
      />
      {/* 配置 */}
      <Route
        exact
        path={`/${containerRouterPrefix}/configuration/configMap`}
        component={ConfigMap}
      />
      <Route
        path={`/${containerRouterPrefix}/configuration/configMap/create`}
        component={ConfigMapCreate}
      />
      <Route
        path={`/${containerRouterPrefix}/configuration/configMap/:namespace/:name/:activeKey?`}
        component={ConfigMapDetail}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/configuration/secret`}
        component={Secret}
      />
      <Route
        path={`/${containerRouterPrefix}/configuration/secret/create`}
        component={SecretCreate}
      />
      <Route
        path={`/${containerRouterPrefix}/configuration/secret/:namespace/:name/:activeKey?`}
        component={SecretDetail}
      />
      {/** 告警 */}
      <Route
        exact
        path={`/${containerRouterPrefix}/alert/alertIndex`}
        component={AlertIndex}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/alert/silentAlert`}
        component={SilentAlert}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/alert/alertIndex/detail/:alert_name?`}
        component={AlertDetail}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/alert/silentAlert/detail/:silent_name?`}
        component={SilentDetail}
      />
      {/** 工作负载 */}
      <Route
        path={`/${containerRouterPrefix}/workload/pod/:namespace/:name/:activeKey?`}
        component={PodDetail}
      />
      <Route
        path={`/${containerRouterPrefix}/workload/pod/create`}
        component={PodCreate}
      />
      <Route
        path={`/${containerRouterPrefix}/workload/pod`}
        component={Pod}
      />
      <Route
        path={`/${containerRouterPrefix}/workload/deployment/create`}
        component={DeploymentCreate}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/workload/deployment/:namespace/:name/:activeKey?`}
        component={DeploymentDetail}
      />
      <Route
        path={`/${containerRouterPrefix}/workload/deployment`}
        component={Deployment}
      />
      <Route
        path={`/${containerRouterPrefix}/workload/statefulSet/create`}
        component={StatefulSetCreate}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/workload/statefulSet/:namespace/:name/:activeKey?`}
        component={StatefulSetDetail}
      />
      <Route
        path={`/${containerRouterPrefix}/workload/statefulSet`}
        component={StatefulSet}
      />
      <Route
        path={`/${containerRouterPrefix}/workload/daemonSet/create`}
        component={DaemonSetCreate}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/workload/daemonSet/:namespace/:name/:activeKey?`}
        component={DaemonSetDetail}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/workload/daemonSet`}
        component={DaemonSet}
      />
      <Route
        path={`/${containerRouterPrefix}/workload/job/create`}
        component={JobCreate}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/workload/job/:namespace/:name/:activeKey?`}
        component={JobDetail}
      />
      <Route
        path={`/${containerRouterPrefix}/workload/job`}
        component={Job}
      />
      <Route
        path={`/${containerRouterPrefix}/workload/cronJob/create`}
        component={CronJobCreate}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/workload/cronJob/:namespace/:name/:activeKey?`}
        component={CronJobDetail}
      />
      <Route
        path={`/${containerRouterPrefix}/workload/cronJob`}
        component={CronJob}
      />
      <Redirect
        path={`/${containerRouterPrefix}/workload`}
        to={`/${containerRouterPrefix}/workload/pod`}
      />
      {/** 事件 */}
      <Route
        path={`/${containerRouterPrefix}/event`}
        component={Events}
      />
      {/** 管理 */}
      <Route
        exact
        path={`/${containerRouterPrefix}/nodeManage`}
        component={Node}
      />
      <Route
        path={`/${containerRouterPrefix}/nodeManage/:nodeName?`}
        component={NodeDetail}
      />

      <Route
        path={`/${containerRouterPrefix}/namespace/namespaceManage/create`}
        component={NamespaceCreate}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/namespace/namespaceManage/:name/:activeKey?`}
        component={NamespaceDetail}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/namespace/namespaceManage`}
        component={Namespace}
      />

      <Route
        path={`/${containerRouterPrefix}/namespace/limitRange/create`}
        component={LimitRangeCreate}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/namespace/limitRange/:namespace/:name/:activeKey?`}
        component={LimitRangeDetail}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/namespace/limitRange`}
        component={LimitRange}
      />

      <Route
        path={`/${containerRouterPrefix}/namespace/resourceQuota/create`}
        component={ResourceQuotaCreate}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/namespace/resourceQuota/:namespace/:name/:activeKey?`}
        component={ResourceQuotaDetail}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/namespace/resourceQuota`}
        component={ResourceQuota}
      />

      <Route
        exact
        path={`/${containerRouterPrefix}/customResourceDefinition`}
        component={CrdIndex}
      />
      <Route
        path={`/${containerRouterPrefix}/customResourceDefinition/createCustomResourceDefinition`}
        component={CrdCreate}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/customResourceDefinition/:customResourceName?/createCR`}
        component={CrCreate}
      />
      <Route
        path={`/${containerRouterPrefix}/customResourceDefinition/:customResourceName/cr/namespace/:namespace/:instanceName?/:activeKey?`}
        component={CrDetail}
      />
      <Route
        path={`/${containerRouterPrefix}/customResourceDefinition/:customResourceName/cr/:instanceName?/:activeKey?`}
        component={CrDetail}
      />
      <Route
        path={`/${containerRouterPrefix}/customResourceDefinition/:customResourceName?/:activeKey?`}
        component={CrdDetail}
      />

      <Route
        exact
        path={`/${containerRouterPrefix}/clusterUser`}
        component={ClusterUser}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/clusterMember`}
        component={ClusterMember}
      />

      {/** 网络 */}
      <Route
        path={`/${containerRouterPrefix}/network/service/create`}
        component={ServiceCreate}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/network/service/:namespace/:name/:activeKey?`}
        component={ServiceDetail}
      />
      <Route
        path={`/${containerRouterPrefix}/network/service`}
        component={Service}
      />
      <Route
        path={`/${containerRouterPrefix}/network/ingress/create`}
        component={IngressCreate}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/network/ingress/:namespace/:name/:activeKey?`}
        component={IngressDetail}
      />
      <Route
        path={`/${containerRouterPrefix}/network/ingress`}
        component={Ingress}
      />
      <Redirect
        path={`/${containerRouterPrefix}/network`}
        to={`/${containerRouterPrefix}/network/service`}
      />

      {/** 存储 */}
      <Route
        exact
        path={`/${containerRouterPrefix}/storage/pv/create`}
        component={PvCreate}
      />
      <Route
        path={`/${containerRouterPrefix}/storage/pv/:name/:activeKey?`}
        component={PvDetail}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/storage/pv`}
        component={Pv}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/storage/pvc/create`}
        component={PvcCreate}
      />
      <Route
        path={`/${containerRouterPrefix}/storage/pvc/:namespace/:name/:activeKey?`}
        component={PvcDetail}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/storage/pvc`}
        component={Pvc}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/storage/sc/create`}
        component={ScCreate}
      />
      <Route
        path={`/${containerRouterPrefix}/storage/sc/:name/:activeKey?`}
        component={ScDetail}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/storage/sc`}
        component={Sc}
      />
      <Redirect
        exact
        path={`/${containerRouterPrefix}/storage`}
        to={`/${containerRouterPrefix}/storage/pv`}
      />

      {/** 监控 */}
      <Route
        exact
        path={`/${containerRouterPrefix}/monitor/monitorDashboard`}
        component={MonitorHomePage}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/monitor/monitorDashboard/customize/query`}
        component={CustomizeMonitorQuery}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/monitor/monitorGoalManage`}
        component={MonitorGoalList}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/monitor/monitorGoalManage/serviceMonitor`}
        component={MonitorServiceMonitor}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/monitor/monitorGoalManage/serviceMonitor/create`}
        component={MonitorCreateService}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/monitor/monitorGoalManage/serviceMonitor/:namespace?/:exampleName?/:activeKey?`}
        component={MonitorServiceDetail}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/monitor/monitorRuleManage`}
        component={MonitorRuleList}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/monitor/monitorRuleManage/detail/:name?`}
        component={MonitorRuleDetail}
      />

      <Redirect
        path={`/${containerRouterPrefix}/monitor`}
        to={`/${containerRouterPrefix}/monitor/monitorDashboard`}
      />
      {/** 服务账号 */}
      <Route
        path={`/${containerRouterPrefix}/userManage/serviceAccount/:namespace/:name/:activeKey?`}
        component={ServiceAccountDetail}
      />
      <Route
        path={`/${containerRouterPrefix}/userManage/serviceAccount/create`}
        component={ServiceAccountCreate}
      />
      <Route
        path={`/${containerRouterPrefix}/userManage/serviceAccount`}
        component={ServiceAccount}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/userManage/roleBinding`}
        component={RoleBinding}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/userManage/roleBinding/create`}
        component={RoleBindCreate}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/userManage/clusterRoleBinding/create`}
        component={ClusterRoleBindCreate}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/userManage/roleBinding/:roleBindNamespace?/:roleBindName?/:activeKey?`}
        component={RoleBindDetail}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/userManage/clusterRoleBinding/:roleBindName?/:activeKey?`}
        component={ClusterRoleBindDetail}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/userManage/role`}
        component={Role}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/userManage/role/create`}
        component={RoleCreate}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/userManage/clusterRole/create`}
        component={ClusterRoleCreate}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/userManage/role/:roleNamespace?/:roleName?/:activeKey?`}
        component={RoleDetail}
      />
      <Route
        exact
        path={`/${containerRouterPrefix}/userManage/clusterRole/:roleName?/:activeKey?`}
        component={ClusterRoleDetail}
      />
      <Redirect to={`/${containerRouterPrefix}/overview`} />
    </Switch>
  );
}
