/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { GET, POST, DELETE } from '@/api/request';

function apiPrefix() {
  const clusterPrefix = '/rest/cluster/v1';
  return clusterPrefix;
}

export function getInstallerConfig() {
  return GET(`${apiPrefix()}/configs`);
}

export function listcluster() {
  return GET(`${apiPrefix()}/clusters`);
}

export function createCluster(data) {
  return POST(`${apiPrefix()}/clusters`, data);
}

export function getCluster(clusteName) {
  return GET(`${apiPrefix()}/clusters/${clusteName}`);
}

export function deleteCluster(clusterName) {
  return DELETE(`${apiPrefix()}/clusters/${clusterName}`);
}

export function autoUpgrade(cluster, data) {
  return POST(`${apiPrefix()}/clusters/${cluster}/auto-upgrade`, data);
}

export function clusterScaleDown(cluster, data) {
  return POST(`${apiPrefix()}/clusters/${cluster}/scale-down`, data);
}

export function clusterScaleUp(cluster, data) {
  return POST(`${apiPrefix()}/clusters/${cluster}/scale-up`, data);
}

export function upgradeCluster(cluster, data) {
  return POST(`${apiPrefix()}/clusters/${cluster}/upgrade`, data);
}

export function getClusterUpgradeVersions(cluster) {
  return GET(`${apiPrefix()}/clusters/${cluster}/upgrade-versions`);
}

export function validateNodes(data) {
  return POST(`${apiPrefix()}/nodes/validate`, data);
}

export function createPatch(data) {
  return POST(`${apiPrefix()}/patches`, data);
}

export function listVersions() {
  return GET(`${apiPrefix()}/versions`);
}
