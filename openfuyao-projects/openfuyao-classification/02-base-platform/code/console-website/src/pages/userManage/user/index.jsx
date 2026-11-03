/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { useTranslation } from 'react-i18next';
import { Button, Form, Space, Input, Table, Popover, message } from 'antd';
import { SyncOutlined, MoreOutlined } from '@ant-design/icons';
import { useEffect, useState, useStore, useCallback, useMemo, useRef } from 'openinula';
import { DEFAULT_CURRENT_PAGE, ResponseCode, ADMIN_USERNAME } from '@/common/constants';
import { Link } from 'inula-router';
import { usermanageRouterPrefix } from '@/constant.js';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { sorterFirstAlphabet, showSuccessThen } from '@/tools/utils';
import EditUser from './editUser';
import { getUserList, createUser, editUser, deleteUser } from '@/api/clusterApi';
import '@/styles/pages/workload.less';
import '@/styles/userManage/dark.less';

export default function UserManage() {
  const { t, i18n } = useTranslation('setting');
  const themeStore = useStore('theme');
  const [userManageForm] = Form.useForm();
  const [messageApi, contextHolder] = message.useMessage();
  const [userManagePage, setUserManagePage] = useState(DEFAULT_CURRENT_PAGE);
  const [userManageList, setUserManageList] = useState([]); // 数据集
  const [userManageLoading, setUserManageLoading] = useState(false); // 加载中
  const [popOpen, setPopOpen] = useState(''); // 悬浮框是否展示
  const [userManageDelModalOpen, setUserManageDelModalOpen] = useState(false); // 删除对话框展示
  const [userManageDelName, setUserManageDelName] = useState(''); // 删除的名称
  const [isUserManageDelCheck, setIsUserManageDelCheck] = useState(false); // 是否选中
  const [originalList, setOriginalList] = useState([]); // 原始数据
  const [formType, setFormType] = useState('');
  const [isUserModalOpen, setIsUserModalOpen] = useState(false);
  const [propsChildData, setPropsChildData] = useState();
  const [editName, setEditName] = useState();
  const [filterRoleValue, setFilterRoleValue] = useState();
  const [filterUserRole, setFilterUserRole] = useState([]);

  const handleUserManageTableChange = useCallback(
    (
      _pagination,
      filter,
      _sorter,
      extra,
    ) => {
      if (extra.action === 'filter' && filter) {
        setFilterRoleValue(filter.plate_role);
      }
    },
    [],
  );

  const cancelEdit = () => {
    setIsUserModalOpen(false);
  };

  const handleChildClick = (data) => {
    let arr = data;
    if (formType === 'add') {
      createUserRequest(arr);
    } else {
      editUserRequest(arr);
    }
  };

  const handleEditUser = (data) => {
    setPopOpen('');
    setFormType('edit');
    setEditName(data.Username);
    setIsUserModalOpen(true);
    setPropsChildData({
      userName: data.Username,
      plateRole: data.PlatformRole,
      desc: data.Description,
    });
  };

  const editUserRequest = async (data) => {
    try {
      const res = await editUser(editName, data);
      if (res.status === ResponseCode.OK) {
        showSuccessThen(messageApi, t('messages.editUserSuccess'), () => {
          setIsUserModalOpen(false);
          setEditName('');
          getUserManagesList();
        });
      }
    } catch (e) {
      let tip = e?.response?.data?.Msg ? t('messages.editUserFailedReason', { reason: e.response.data.Msg }) : t('messages.editUserFailed');
      messageApi.error(tip);
    }
  };

  const createUserRequest = async (data) => {
    try {
      const res = await createUser(data);
      if (res.status === ResponseCode.OK) {
        showSuccessThen(messageApi, t('messages.createUserSuccess'), () => {
          setIsUserModalOpen(false);
          getUserManagesList();
        });
      }
    } catch (e) {
      let tip = e?.response?.data?.Msg ? t('messages.createUserFailedReason', { reason: e.response.data.Msg }) : t('messages.createUserFailed');
      messageApi.error(tip);
    }
  };

  const addUser = () => {
    setFormType('add');
    setPropsChildData(null);
    setIsUserModalOpen(true);
  };

  // 重置按钮
  const handleUserManageReset = () => {
    userManageForm.resetFields();
    setFilterRoleValue(undefined);
    getUserManagesList(false);
  };

  // 删除按钮
  const handleDeleteUserManage = (record) => {
    // 隐藏气泡框
    setPopOpen('');
    setUserManageDelModalOpen(true); // 打开弹窗
    setUserManageDelName(record.Username);
    setIsUserManageDelCheck(false);
  };

  const handleDelUserManageCancel = () => {
    setUserManageDelModalOpen(false);
    setUserManageDelName('');
  };

  const delPendingRef = useRef(false);
  const handleDelUserManageConfirm = async () => {
    if (delPendingRef.current) { return; }
    delPendingRef.current = true;
    try {
      const res = await deleteUser(userManageDelName);
      if (res.status === ResponseCode.OK) {
        showSuccessThen(messageApi, t('messages.deleteUserSuccess', { name: userManageDelName }), () => {
          setUserManageDelModalOpen(false);
          setIsUserManageDelCheck(false);
          setUserManageDelName('');
          getUserManagesList();
        });
      }
    } catch (e) {
      let tip = e?.response?.data?.Msg ? t('messages.deleteUserFailedReason', { reason: e.response.data.Msg }) : t('messages.deleteUserFailed', { name: userManageDelName });
      messageApi.error(tip);
    } finally {
      delPendingRef.current = false;
    }
  };

  const handleUserManageCheckFn = (e) => {
    setIsUserManageDelCheck(e.target.checked);
  };

  const handleUserManageOpenChange = (newOpen, record) => {
    if (newOpen) {
      setPopOpen(record.Username);
    } else {
      setPopOpen('');
    }
  };

  // 列表项
  const userManageColumns = useMemo(() => [
    {
      title: t('table.userName'),
      key: 'user_name',
      sorter: (a, b) => sorterFirstAlphabet(a.Username, b.Username),
      render: (_, record) => <Link to={{ pathname: `/${usermanageRouterPrefix}/user/detail`, state: { userName: record.Username } }}>{record.Username}</Link>,
    },
    {
      title: t('table.platformRole'),
      key: 'plate_role',
      filters: filterUserRole,
      filterMultiple: false,
      filteredValue: filterRoleValue ? [filterRoleValue] : [],
      sorter: (a, b) => sorterFirstAlphabet(a.PlatformRole, b.PlatformRole),
      onFilter: (value, record) => record.PlatformRole.toLowerCase() === value.toLowerCase(),
      render: (_, record) => record.PlatformRole || '-',
    },
    {
      title: t('table.description'),
      key: 'user_desc',
      render: (_, record) => record.Description || '-',
    },
    {
      title: t('table.operation'),
      width: 120,
      key: 'handle',
      fixed: 'right',
      render: (_, record) => (
        <Space>
          <Popover placement="bottom"
            content={
              <div className="pop_modal">
                <Button type="link" disabled={record.Username === ADMIN_USERNAME} onClick={() => handleEditUser(record)}>{t('actions.modify')}</Button>
                <Button type="link" disabled={record.Username === ADMIN_USERNAME} onClick={() => handleDeleteUserManage(record)}>{t('actions.delete')}</Button>
              </div>
            }
            trigger="click"
            open={popOpen === record.Username}
            onOpenChange={e => handleUserManageOpenChange(e, record)}
          >
            <MoreOutlined className="common_antd_icon primary_color" />
          </Popover >
        </Space >
      ),
    },
  ], [t, i18n.language, filterUserRole, filterRoleValue, popOpen]);

  // 获取userManageList
  const getUserManagesList = async (isChange = true) => {
    setUserManageLoading(true);
    try {
      const res = await getUserList();
      if (res.status === ResponseCode.OK) {
        let statusArr = [];
        let roleOption = {};
        res.data.Data.map(item => {
          if (!Object.keys(roleOption).includes(item.PlatformRole)) {
            roleOption[item.PlatformRole] = item.PlatformRole;
          }
        });
        const statusKey = Object.keys(roleOption);
        statusKey.map(item => {
          statusArr.push({ text: item, value: roleOption[item] });
        });
        setFilterUserRole([...statusArr]);
        setUserManageLoading(false);
        setOriginalList([...res.data.Data]);
        handleSearchUserManage(res.data.Data, isChange); // 先搜索
      }
    } catch (e) {
      setUserManageLoading(false);
      if (e?.response?.status === ResponseCode.Forbidden) {
        return;
      }
      if (e?.response?.data?.code === ResponseCode.NotFound) {
        setUserManageList([]); // 数组为空
      } else {
        messageApi.error(t('messages.fetchDataError', { message: e.response?.data?.Msg }));
      }
    }
  };

  // 检索
  const handleSearchUserManage = (totalData = originalList || [], isChange = true) => {
    const serviceAccountFormName = userManageForm.getFieldValue('user_name');
    let temporyList = totalData;
    if (serviceAccountFormName) {
      temporyList = totalData.filter(item => (item.Username).toLowerCase().includes(serviceAccountFormName.toLowerCase()));
    }
    setUserManageList([...temporyList]);
    isChange ? setUserManagePage(DEFAULT_CURRENT_PAGE) : null;
  };

  useEffect(() => {
    getUserManagesList();
  }, []);

  return <div className="child_content">
    <BreadCrumbCom items={[{ title: t('module.userManage'), path: `/${usermanageRouterPrefix}/user` }]} />
    <div className="tab_container container_margin_box userManageMinHeight">
      <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
        {contextHolder}
      </div>
      <Form className="pod_searchForm form_padding_bottom" form={userManageForm}>
        <Form.Item name="user_name" className="pod_search_input">
          <Input.Search placeholder={t('searchPlaceholder', { ns: 'common', name: t('searchName.user', { ns: 'common' }) })} onSearch={() => handleSearchUserManage()} autoComplete="off" />
        </Form.Item>
        <Form.Item>
          <Space>
            <Button className="primary_btn" onClick={addUser}>{t('actions.createUser')}</Button>
            <Button icon={<SyncOutlined />} onClick={handleUserManageReset} className="reset_btn" style={{ marginLeft: '16px' }}></Button>
          </Space>
        </Form.Item>
      </Form>
      <div className="tab_table_flex">
        <Table
          className="table_padding"
          loading={userManageLoading}
          columns={userManageColumns}
          dataSource={userManageList}
          onChange={handleUserManageTableChange}
          pagination={{
            className: 'page',
            current: userManagePage,
            showTotal: (total) => t('pagination.total', { ns: 'common', total }),
            showSizeChanger: true,
            showQuickJumper: true,
            pageSizeOptions: [10, 20, 50],
            onChange: page => setUserManagePage(page),
          }}
          scroll={{ x: 1280 }}
        />
      </div>
      <EditUser formType={formType} isShowModel={isUserModalOpen}
        data={propsChildData} cancelEdit={cancelEdit} submitModel={handleChildClick} />
      <DeleteInfoModal
        title={t('modal.deleteUserTitle')}
        open={userManageDelModalOpen}
        cancelFn={handleDelUserManageCancel}
        content={[
          t('modal.deleteUserWarning'),
          t('modal.deleteUserConfirm', { name: userManageDelName }),
        ]}
        isCheck={isUserManageDelCheck}
        showCheck={true}
        checkFn={handleUserManageCheckFn}
        confirmFn={handleDelUserManageConfirm} />
    </div>
  </div>;
}