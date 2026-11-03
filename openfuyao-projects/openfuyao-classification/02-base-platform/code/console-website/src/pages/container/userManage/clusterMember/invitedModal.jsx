/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Button, Form, Space, Input, Popover, Table, Modal, Select } from 'antd';
import { useEffect, useState, useCallback, useRef, useMemo } from 'openinula';
import { useTranslation } from 'react-i18next';
import { PlusCircleOutlined, CheckCircleFilled } from '@ant-design/icons';
import '@/styles/userManage/user.less';
import { ResponseCode } from '@/common/constants';
import { getWaitInviteClusterMemberList, getClusterRoleList } from '@/api/clusterApi';
import { sorterFirstAlphabet } from '@/tools/utils';

export default function InvitedModal(props) {
  const { t } = useTranslation('rbac');
  const [noInvitedForm] = Form.useForm();
  const [originalList, setOriginalList] = useState([]);
  const [noInvitedList, setNoInvitedList] = useState([]);
  const [noInvitedMemberLoading, setNoInvitedMemberLoading] = useState(false);
  const [noInvitedPopOpen, setNoInvitedPopOpen] = useState('');
  const [roleOption, setRoleOption] = useState();
  const [isSubmitBtn, setIsSubmitBtn] = useState(true);
  const [invitedModalForm] = Form.useForm();
  const optionsRef = useRef(null);

  const cancelModelEdit = () => {
    props.cancelEdit();
  };

  const submitNoInvited = () => {
    let data = [];
    noInvitedList.map(item => {
      if (item.clusterrole) {
        data.push({
          member: item.username,
          role: item.clusterrole,
        });
      }
    });
    props.submitNoinvitedModel(data);
  };

  const resetNoInvited = () => {
    noInvitedForm.setFieldValue('noInvited_name', '');
    getNoInvitedListData();
  };

  const handelChangeRole = (e, record) => {
    let data = [...noInvitedList];
    data.map(item => {
      if (item.username === record.username) {
        item.clusterrole = e;
      }
    });
    invitedModalForm?.setFieldsValue({ role: '' });
    setNoInvitedList(data);
    setNoInvitedPopOpen('');
  };

  const noInvitedMemberColumns = useMemo(() => [
    {
      title: t('clusterMember.invitedModal.table.user'),
      key: 'noInvited_name',
      sorter: (a, b) => sorterFirstAlphabet(a.username, b.username),
      render: (_, record) => record.username || '--',
    },
    {
      title: t('clusterMember.invitedModal.table.assignRole'),
      key: 'noInvited_role',
      render: (_, record) => (
        <Space>
          <p style={{ display: record.clusterrole ? 'none' : 'block' }}>--</p>
          <Select style={{ display: record.clusterrole ? 'block' : 'none' }}
            options={roleOption} value={record.clusterrole} onChange={(e) => handelChangeRole(e, record)} />
        </Space >
      ),
    },
    {
      title: t('clusterMember.invitedModal.table.operation'),
      key: 'namespace_handle',
      fixed: 'right',
      width: 120,
      render: (_, record) => (
        <Space>
          <Popover placement="bottom" getPopupContainer={() => document.getElementById('invitedTable')}
            content={
              <div className="pop_modal">
                <Form form={invitedModalForm} ref={optionsRef} autoComplete="off" initialValues={{
                  role: '',
                }}>
                  <Form.Item name='role'>
                    <Select options={roleOption} placeholder={t('clusterMember.invitedModal.selectPlaceholder')} style={{ width: '120px' }} onChange={(e) => handelChangeRole(e, record)} />
                  </Form.Item>
                </Form>
              </div>
            }
            trigger="click"
            open={noInvitedPopOpen === record.username}
            onOpenChange={e => handleUserManageOpenChange(e, record)}
          >
            <PlusCircleOutlined style={{ display: record.clusterrole ? 'none' : 'block', marginLeft: '8px' }} className="common_antd_icon primary_color" />
          </Popover >
          <CheckCircleFilled style={{ display: record.clusterrole ? 'block' : 'none' }} className="common_antd_icon icon_selected" />
        </Space >
      ),
    },
  ], [t, roleOption, noInvitedList, noInvitedPopOpen]);

  const handleUserManageOpenChange = (newOpen, record) => {
    if (newOpen) {
      setNoInvitedPopOpen(record.username);
    } else {
      setNoInvitedPopOpen('');
    }
  };

  const getNoInvitedListData = useCallback(async () => {
    setNoInvitedMemberLoading(true);
    try {
      const res = await getWaitInviteClusterMemberList();
      if (res.status === ResponseCode.OK) {
        setOriginalList([...(res.data?.Data || [])]);
        handleSearchNoInvitedName(res.data?.Data ?? []);
      }
    } catch (e) {
      if (e.response?.data?.code === ResponseCode.NotFound) {
        setNoInvitedList([]);
      }
    }
    setNoInvitedMemberLoading(false);
  }, []);

  const handleSearchNoInvitedName = (totalData = originalList) => {
    const noInvitedFormName = noInvitedForm.getFieldValue('noInvited_name');
    let temporyList = totalData;
    if (noInvitedFormName) {
      temporyList = temporyList.filter(item => (item.username).toLowerCase().includes(noInvitedFormName.toLowerCase()));
    }
    setNoInvitedList([...temporyList]);
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
    if (props.isresetInvited) {
      getNoInvitedListData();
    }
  }, [props.isresetInvited]);

  useEffect(() => {
    getRoleOption();
  }, []);

  useEffect(() => {
    getNoInvitedListData();
  }, [getNoInvitedListData]);

  useEffect(() => {
    getNoInvitedListData();
    if (optionsRef.current) {
      invitedModalForm?.setFieldsValue({ role: '' });
    }
  }, [props.isShowModel, optionsRef]);

  useEffect(() => {
    setIsSubmitBtn(true);
    noInvitedList.map(item => {
      if (Object.prototype.hasOwnProperty.call(item, 'clusterrole')) {
        setIsSubmitBtn(false);
      }
    });
  }, [noInvitedList]);

  return (
    <Modal width={1094} forceRender getContainer={false} title={t('clusterMember.modal.inviteTitle')} open={props.isShowModel} onCancel={cancelModelEdit} className='silentModelCommon'
      footer={[
        <Button className='cancel_btn' onClick={cancelModelEdit} style={{ marginRight: '14px' }}>{t('actions.cancel', { ns: 'common' })}</Button>,
        <Button className='cancel_btn' style={{ marginRight: '14px' }} onClick={resetNoInvited}>{t('actions.reset', { ns: 'common' })}</Button>,
        <Button className='primary_btn' disabled={isSubmitBtn} onClick={submitNoInvited}>{t('actions.confirm', { ns: 'common' })}</Button>,
      ]}
    >
      <Form className="searchForm form_padding_bottom" form={noInvitedForm}>
        <Form.Item name="noInvited_name" className="search_input">
          <Input.Search placeholder={t('searchPlaceholder', { ns: 'common', name: t('searchName.member', { ns: 'common' }) })} onSearch={() => handleSearchNoInvitedName()} autoComplete="off" />
        </Form.Item>
      </Form>
      <div className='tab_table_flex'>
        <Table
          id='invitedTable'
          className='table_padding'
          loading={noInvitedMemberLoading}
          columns={noInvitedMemberColumns}
          dataSource={noInvitedList}
          pagination={false}
        />
      </div>
    </Modal >
  );
}