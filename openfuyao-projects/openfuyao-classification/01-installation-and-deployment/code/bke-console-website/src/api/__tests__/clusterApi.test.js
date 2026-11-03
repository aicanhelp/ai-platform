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
import {
  getClustersList,
  getUserList,
  getUserDetail,
  createUser,
  editUser,
  deleteUser,
  getRoleList,
  getClusterRoleList,
  getInvitedClusterMemberList,
  getWaitInviteClusterMemberList,
  inviteMemberRole,
  editWaitInviteClusterMemberList,
  deleteClusterUser,
} from '../clusterApi';

// Mock sessionStorage and API requests
jest.mock('@/api/request', () => ({
  GET: jest.fn(),
  POST: jest.fn(),
  PUT: jest.fn(),
  DELETE: jest.fn(),
  PATCH: jest.fn(),
}));

jest.mock('query-string');

describe('clusterApi', () => {
  beforeEach(() => {
    sessionStorage.setItem('cluster', 'test-cluster');
    jest.clearAllMocks();
  });

  describe('commonApiPrefix', () => {
    it('should return correct API prefixes', () => {
      const result = commonApiPrefix();
      expect(result).toEqual({
        clusterUrl: '/rest/multicluster/v1beta1',
        userUrl: '/rest/user/v1',
        clusterUserUrl: '/clusters/test-cluster/rest/user/v1',
        clustername: 'test-cluster',
      });
    });
  });

  describe('getClustersList', () => {
    it('should call GET with correct URL', () => {
      getClustersList();
      expect(GET).toHaveBeenCalledWith('/rest/multicluster/v1beta1/resources/clusters');
    });
  });

  describe('getUserList', () => {
    it('should call GET with correct URL', () => {
      getUserList();
      expect(GET).toHaveBeenCalledWith('/rest/user/v1/users');
    });
  });

  describe('getUserDetail', () => {
    it('should call GET with correct URL', () => {
      getUserDetail('test-user');
      expect(GET).toHaveBeenCalledWith('/rest/user/v1/users/test-user');
    });
  });

  describe('createUser', () => {
    it('should call POST with correct URL and data', () => {
      const mockData = { name: 'test-user' };
      qs.stringifyUrl.mockReturnValue('/rest/user/v1/users');
      createUser(mockData);
      expect(POST).toHaveBeenCalledWith(
        '/rest/user/v1/users',
        mockData,
        { 'Content-Type': 'application/json' }
      );
    });
  });

  describe('editUser', () => {
    it('should call PATCH with correct URL and data', () => {
      const mockData = { name: 'test-user' };
      qs.stringifyUrl.mockReturnValue('/rest/user/v1/users/test-user');
      editUser('test-user', mockData);
      expect(PATCH).toHaveBeenCalledWith(
        '/rest/user/v1/users/test-user',
        mockData,
        { 'Content-Type': 'application/json' }
      );
    });
  });

  describe('deleteUser', () => {
    it('should call DELETE with correct URL', () => {
      qs.stringifyUrl.mockReturnValue('/rest/user/v1/users/test-user');
      deleteUser('test-user');
      expect(DELETE).toHaveBeenCalledWith('/rest/user/v1/users/test-user');
    });
  });

  describe('getRoleList', () => {
    it('should call GET with correct URL', () => {
      getRoleList();
      expect(GET).toHaveBeenCalledWith('/rest/user/v1/platform-roles');
    });
  });

  describe('getClusterRoleList', () => {
    it('should call GET with correct URL', () => {
      getClusterRoleList();
      expect(GET).toHaveBeenCalledWith('/clusters/test-cluster/rest/user/v1/cluster-roles');
    });
  });

  describe('getInvitedClusterMemberList', () => {
    it('should call GET with correct URL', () => {
      getInvitedClusterMemberList();
      expect(GET).toHaveBeenCalledWith('/rest/user/v1/cluster-members?cluster-name=test-cluster');
    });
  });

  describe('getWaitInviteClusterMemberList', () => {
    it('should call GET with correct URL', () => {
      getWaitInviteClusterMemberList();
      expect(GET).toHaveBeenCalledWith('/rest/user/v1/invite-users?cluster-name=test-cluster');
    });
  });

  describe('inviteMemberRole', () => {
    it('should call PUT with correct URL', () => {
      qs.stringifyUrl.mockReturnValue('/rest/user/v1/invite-users/test-member/test-cluster/test-role');
      inviteMemberRole('test-member', 'test-role');
      expect(PUT).toHaveBeenCalledWith(
        '/rest/user/v1/invite-users/test-member/test-cluster/test-role',
        { 'Content-Type': 'application/json' }
      );
    });
  });

  describe('editWaitInviteClusterMemberList', () => {
    it('should call PATCH with correct URL', () => {
      qs.stringifyUrl.mockReturnValue('/clusters/test-cluster/rest/user/v1/cluster-rolebindings/test-member/test-role');
      editWaitInviteClusterMemberList('test-member', 'test-role');
      expect(PATCH).toHaveBeenCalledWith(
        '/clusters/test-cluster/rest/user/v1/cluster-rolebindings/test-member/test-role',
        { 'Content-Type': 'application/json' }
      );
    });
  });

  describe('deleteClusterUser', () => {
    it('should call DELETE with correct URL', () => {
      qs.stringifyUrl.mockReturnValue('/rest/user/v1/remove-users/test-member/test-cluster');
      deleteClusterUser('test-member');
      expect(DELETE).toHaveBeenCalledWith('/rest/user/v1/remove-users/test-member/test-cluster');
    });
  });
});
