/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { Button, Form, Space, Input, Popover, Table, message, Modal, Select } from 'antd';
import { MoreOutlined } from '@ant-design/icons';
import '@/styles/userManage/user.less';
import { useCallback, useEffect, useState, useStore, useMemo } from 'openinula';
import { useTranslation } from 'react-i18next';
import { DEFAULT_CURRENT_PAGE, ResponseCode } from '@/common/constants';
import { sorterFirstAlphabet, showSuccessThen } from '@/tools/utils';
import { 
  getInvitedClusterMemberList,
  editWaitInviteClusterMemberList,
  getClusterRoleList,
  inviteMemberRole,
  deleteClusterUser,
} from '@/api/clusterApi';
import InvitedModal from './invitedModal';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import NoPermissions from '@/components/NoPermissions';
import { filterRepeat } from '@/utils/common';
import useButtonLoading from '@/hooks/useButtonLoading';

export default function ClusterMember() {
  const { t } = useTranslation('rbac');
  const themeStore = useStore('theme');
  const [clusterForm] = Form.useForm();
  const [messageApi, contextHolder] = message.useMessage();
  const [clusterMemberPage, setClusterMemberPage] = useState(DEFAULT_CURRENT_PAGE);
  const [editRole, setEditRole] = useState('');
  const [roleOption, setRoleOption] = useState();
  const [clusterMemberListData, setClusterMemberListData] = useState();
  const [clusterMemberLoading, setClusterMemberLoading] = useState(false);
  const [popOpen, setPopOpen] = useState('');
  const [clusterMemberDelModalOpen, setClusterMemberDelModalOpen] = useState(false);
  const [clusterMemberEditModalOpen, setClusterMemberEditModalOpen] = useState(false);
  const [editLoading, setEditLoading] = useButtonLoading();
  const [clusterMemberEditName, setClusterMemberEditName] = useState('');
  const [clusterMemberDelName, setClusterMemberDelName] = useState('');
  const [originalList, setOriginalList] = useState([]);
  const [isresetInvite, setIsresetInvite] = useState(false);
  const [isMemberDelCheck, setIsMemberDelCheck] = useState(false);
  const [isShowNopermission, setIsShowNopermission] = useState(false);
  const [filterMemberValue, setFilterMemberValue] = useState();
  const [filterMemberOption, setFilterMemberOption] = useState([]);

  const clusterMemberColumns = useMemo(() => [
    {
      title: t('clusterMember.table.name'),
      key: 'member_name',
      sorter: (a, b) => sorterFirstAlphabet(a.Username, b.Username),
      render: (_, record) => record.Username || '--',
    },
    {
      title: t('clusterMember.table.role'),
      key: 'member_user',
      filters: filterMemberOption,
      filterMultiple: false,
      filteredValue: filterMemberValue ? [filterMemberValue] : [],
      sorter: (a, b) => sorterFirstAlphabet(a.ClusterRole, b.ClusterRole),
      onFilter: (value, record) => (record.ClusterRole || '').toLowerCase() === value.toLowerCase(),
      render: (_, record) => record.ClusterRole || '--',
    },
    {
      title: t('clusterMember.table.operation'),
      key: 'namespace_handle',
      fixed: 'right',
      width: 120,
      render: (_, record) => <Space>
        <Popover placement='bottom' content={<div className='pop_modal'>
          <Button type='link' onClick={() => handleOpenEdit(record)}>{t('clusterMember.actions.modifyRole')}</Button>
          <Button type='link' onClick={() => handleOpenDelete(record.Username)}>{t('clusterMember.actions.remove')}</Button></div>
        } trigger='click' open={popOpen === record.Username} onOpenChange={newOpen => newOpen ? setPopOpen(record.Username) : setPopOpen('')}>
          <MoreOutlined className='common_antd_icon primary_color' />
        </Popover>
      </Space>,
    },
  ], [t, filterMemberOption, filterMemberValue, popOpen]);

  const [isNoInivtedModalOpen, setIsNoInvitedModalOpen] = useState(false);

  const cancelInvited = () => {
    setIsNoInvitedModalOpen(false);
    setIsresetInvite(true);
  };

  const handleClickInvitedModel = () => {
    setIsNoInvitedModalOpen(true);
  };

  const getInvitedListData = useCallback(async (isChange = true) => {
    setClusterMemberLoading(true);
    try {
      const res = await getInvitedClusterMemberList();
      if (res.status === ResponseCode.OK) {
        setOriginalList([...(res.data?.Data || [])]);
      }
      handleSearchClusterName([].concat(res.data?.Data || []), isChange);
    } catch (error) {
      if (error.response?.status === ResponseCode.Forbidden) {
        setIsShowNopermission(true);
      }
      if (error.response?.status === ResponseCode.NotFound) {
        setClusterMemberListData([]);
      }
    }
    setClusterMemberLoading(false);
  }, []);

  const handleOpenDelete = (data) => {
    setPopOpen('');
    setClusterMemberDelModalOpen(true);
    setClusterMemberDelName(data);
  };

  const handleOpenEdit = (data) => {
    setPopOpen('');
    setEditRole(data.ClusterRole);
    setClusterMemberEditModalOpen(true);
    setClusterMemberEditName(data.Username);
  };

  const handleCancelDelete = () => {
    setClusterMemberDelModalOpen(false);
    setIsMemberDelCheck(false);
    setClusterMemberDelName('');
  };

  const handleClusterMemberTableChange = useCallback(
    (_pagination, filter, _sorter, extra) => {
      if (extra.action === 'filter') {
        setFilterMemberValue(filter.member_user);
      }
    },
    [],
  );

  const handleDelMember = async () => {
    try {
      setIsresetInvite(false);
      const res = await deleteClusterUser(clusterMemberDelName);
      if (res.status === ResponseCode.OK) {
        await showSuccessThen(messageApi, t('clusterMember.messages.removeSuccess', { name: clusterMemberDelName }), () => {
          setClusterMemberDelModalOpen(false);
          setClusterMemberDelName('');
          setIsMemberDelCheck(false);
          setIsresetInvite(true);
          getInvitedListData();
        });
      }
    } catch (e) {
      messageApi.error(t('clusterMember.messages.removeFailed', { name: clusterMemberDelName, reason: e.response?.data.Msg }));
    }
  };

  const handleCancelEdit = () => {
    setClusterMemberEditModalOpen(false);
    setClusterMemberEditName('');
    setEditRole('');
  };

  const handleEditMember = async () => {
    setEditLoading(true);
    try {
      const res = await editWaitInviteClusterMemberList(clusterMemberEditName, editRole);
      if (res.status === ResponseCode.OK) {
        await showSuccessThen(messageApi, t('clusterMember.messages.editSuccess', { name: clusterMemberEditName }), () => {
          setClusterMemberEditModalOpen(false);
          setClusterMemberEditName('');
          setEditRole('');
          getInvitedListData();
        });
      }
    } catch {
      messageApi.error(t('clusterMember.messages.editFailed', { name: clusterMemberEditName }));
    } finally {
      setEditLoading(false);
    }
  };

  const handleSelectRole = (value) => {
    setEditRole(value);
  };

  const handleClickNoinvitedData = async (data) => {
    setIsresetInvite(false);
    const results = await Promise.allSettled(
      data.map(async item => {
        const res = await inviteMemberRole(item.member, item.role);
        if (res.status === ResponseCode.OK) {
          messageApi.success(t('clusterMember.messages.inviteSuccess', { name: item.member }));
        } else {
          messageApi.error(t('clusterMember.messages.inviteFailed', { name: item.member }));
        }
      }),
    );
    results.forEach(result => {
      if (result.status === 'rejected') {
        messageApi.error(t('clusterMember.messages.inviteFailed', { name: '' }));
      }
    });
    setIsresetInvite(true);
    getInvitedListData();
    setIsNoInvitedModalOpen(false);
  };

  const handleSearchClusterName = (totalData = originalList, isChange = true) => {
    const memberFormName = clusterForm.getFieldValue('memebr_name');
    let temporyList = totalData;
    if (memberFormName) {
      temporyList = temporyList.filter(item => (item.Username).toLowerCase().includes(memberFormName.toLowerCase()));
    }
    setClusterMemberListData(temporyList);
    let arr = [];
    temporyList.map(item => {
      arr.push({ text: item.ClusterRole, value: item.ClusterRole });
    });
    setFilterMemberOption(filterRepeat(arr));
    isChange ? setClusterMemberPage(DEFAULT_CURRENT_PAGE) : null;
  };

  const handleMemberCheckFn = (e) => {
    setIsMemberDelCheck(e.target.checked);
  };

  const getRoleOption = async () => {
    const res = await getClusterRoleList();
    let option = [];
    if (res.status === ResponseCode.OK) {
      res.data.Data.map(item => {
        option.push({
          label: item.metadata.name,
          value: item.metadata.name,
        });
      });
    }
    setRoleOption(option);
  };

  useEffect(() => {
    getRoleOption();
  }, []);

  useEffect(() => {
    getInvitedListData();
  }, [getInvitedListData]);

  return <div className='child_content withBread_content'>
    <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
      {contextHolder}
    </div>
    {isShowNopermission ? '' : <BreadCrumbCom className='create_bread' items={[{ title: t('clusterMember.title'), path: '/', disabled: true }]} />}
    {isShowNopermission ? < NoPermissions /> :
      <div className="container_margin_box" style={{ display: 'flex', flexDirection: 'column' }}>
        <Form className="searchForm form_padding_bottom" form={clusterForm}>
          <Form.Item name="memebr_name" className="search_input">
            <Input.Search placeholder={t('searchPlaceholder', { ns: 'common', name: t('searchName.member', { ns: 'common' }) })} onSearch={() => handleSearchClusterName()} autoComplete="off" />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button className='primary_btn' onClick={handleClickInvitedModel}>{t('clusterMember.actions.invite')}</Button>
            </Space>
          </Form.Item>
        </Form>
        <div className='tab_table_flex cluster_container_height'>
          <Table
            className='table_padding'
            loading={clusterMemberLoading}
            onChange={handleClusterMemberTableChange}
            columns={clusterMemberColumns}
            dataSource={clusterMemberListData}
            pagination={{
              className: 'page',
              current: clusterMemberPage,
              showTotal: (total) => t('pagination.total', { ns: 'common', total }),
              showSizeChanger: true,
              showQuickJumper: true,
              pageSizeOptions: [10, 20, 50],
              onChange: page => setClusterMemberPage(page),
            }}
            scroll={{ x: 1280 }}
          />
        </div>
        <InvitedModal isresetInvited={isresetInvite} isShowModel={isNoInivtedModalOpen}
          submitNoinvitedModel={handleClickNoinvitedData} cancelEdit={cancelInvited} />
        <DeleteInfoModal
          title={t('clusterMember.modal.removeTitle')}
          open={clusterMemberDelModalOpen}
          cancelFn={handleCancelDelete}
          content={[
            t('clusterMember.modal.removeWarning'),
            t('clusterMember.modal.removeConfirm', { name: clusterMemberDelName }),
          ]}
          showCheck={true}
          confirmText={t('clusterMember.actions.remove')}
          isCheck={isMemberDelCheck}
          checkFn={handleMemberCheckFn}
          confirmFn={handleDelMember} />
        <Modal title={t('clusterMember.modal.modifyRoleTitle')} getContainer={false} width={682} height={270} open={clusterMemberEditModalOpen} confirmLoading={editLoading} onOk={handleEditMember} onCancel={handleCancelEdit}
          footer={[
            <Button onClick={handleCancelEdit} className='cancel_btn'>
              {t('actions.cancel', { ns: 'common' })}
            </Button>,
            <Button className='primary_btn' loading={editLoading} onClick={handleEditMember}>
              {t('actions.confirm', { ns: 'common' })}
            </Button>,
          ]}>
          <p style={{ margin: '10px 0 8px', color: '#666 ' }}>{t('clusterMember.modal.roleLabel')}</p>
          <Select style={{ width: '100%' }} value={editRole} options={roleOption} onChange={handleSelectRole} />
        </Modal>
      </div>
    }
  </div>;
}