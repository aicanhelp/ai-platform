/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { GET, POST, PUT, DELETE, PATCH } from '@/api/request';
import qs from 'query-string';
function commonApiPrefix() {
  let clusterName = sessionStorage.getItem('cluster');
  let paramObj = {
    clusterUrl: `/rest/multicluster/v1beta1`,
    userUrl: '/rest/user/v1',
    clusterUserUrl: `/clusters/${clusterName}/rest/user/v1`,
    clustername: clusterName,
  };
  return paramObj;
};

/**
 * 获取集群列表
 */
export function getClustersList() {
  const url = `${commonApiPrefix().clusterUrl}/resources/clusters`;
  return GET(url);
}

/**
 * 用户管理-获取平台用户列表
 */
export function getUserList() {
  const url = `${commonApiPrefix().userUrl}/users`;
  return GET(url);
}

/**
 * 用户管理-获取平台用户详情
 */
export function getUserDetail(name) {
  const url = `${commonApiPrefix().userUrl}/users/${name}`;
  return GET(url);
}

/**
 * 用户管理-创建用户
 */
export function createUser(data) {
  const url = qs.stringifyUrl(
    { url: `${commonApiPrefix().userUrl}/users` }
  );
  return POST(url, data, { 'Content-Type': 'application/json' });
}

/**
 * 用户管理-编辑用户
 */
export function editUser(name, data) {
  const url = qs.stringifyUrl(
    { url: `${commonApiPrefix().userUrl}/users/${name}` }
  );
  return PATCH(url, data, { 'Content-Type': 'application/json' });
}

/**
 * 用户管理-删除用户
 */
export function deleteUser(data) {
  const url = qs.stringifyUrl(
    { url: `${commonApiPrefix().userUrl}/users/${data}` }
  );
  return DELETE(url);
}

/**
 * 获取平台角色列表
 */
export function getRoleList() {
  const url = `${commonApiPrefix().userUrl}/platform-roles`;
  return GET(url);
}

/**
 * 管理-获取集群角色列表
 */
export function getClusterRoleList() {
  const url = `${commonApiPrefix().clusterUserUrl}/cluster-roles`;
  return GET(url);
}


/**
 * 管理-集群成员-获取集群成员列表
 */
export function getInvitedClusterMemberList() {
  const url = `${commonApiPrefix().userUrl}/cluster-members?cluster-name=${commonApiPrefix().clustername}`;
  return GET(url);
}

/**
 * 管理-集群成员-获取待邀请集群成员列表
 */
export function getWaitInviteClusterMemberList() {
  const url = `${commonApiPrefix().userUrl}/invite-users?cluster-name=${commonApiPrefix().clustername}`;
  return GET(url);
}

/**
 * 管理-集群成员-邀请集群成员角色
 */
export function inviteMemberRole(member, role) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().userUrl}/invite-users/${member}/${commonApiPrefix().clustername}/${role}`,
  });
  return PUT(url, { 'Content-Type': 'application/json' });
}

/**
 * 管理-集群成员-修改集群成员角色
 */
export function editWaitInviteClusterMemberList(member, role) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().clusterUserUrl}/cluster-rolebindings/${member}/${role}`,
  });
  return PATCH(url, { 'Content-Type': 'application/json' });
}

/**
 * 管理-集群成员-删除用户
 */
export function deleteClusterUser(member) {
  const url = qs.stringifyUrl(
    { url: `${commonApiPrefix().userUrl}/remove-users/${member}/${commonApiPrefix().clustername}` }
  );
  return DELETE(url);
}