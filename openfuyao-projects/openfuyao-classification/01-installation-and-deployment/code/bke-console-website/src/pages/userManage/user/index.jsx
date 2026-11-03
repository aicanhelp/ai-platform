/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Button, Form, Space, Input, Table, ConfigProvider, Popover, message } from 'antd';
import { SyncOutlined, MoreOutlined } from '@ant-design/icons';
import { useEffect, useState, useStore, useCallback } from 'openinula';
import { DEFAULT_CURRENT_PAGE, ResponseCode } from '@/common/constants';
import zhCN from 'antd/es/locale/zh_CN';
import { Link } from 'inula-router';
import { usermanageRouterPrefix } from '@/constant.js';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { sorterFirstAlphabet } from '@/tools/utils';
import EditUser from './editUser';
import { getUserList, getRoleList, createUser, editUser, deleteUser } from '@/api/clusterApi';
import { useIntl } from 'inula-intl';
import '@/styles/pages/workload.less';
import '@/styles/userManage/dark.less';
export default function UserManage() {
  const intl = useIntl();
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
  const [isReset, setIsReset] = useState(false);
  const [filterRoleValue, setFilterRoleValue] = useState();
  const [filterUserRole, setFilterUserRole] = useState([]);

  const handleUserManageTableChange = useCallback((_pagination, filter, _sorter, extra) => {
    if (extra.action === 'filter') {
      setFilterRoleValue(filter.plate_role);
    }
  }, []);

  const cancelEdit = () => {
    setIsUserModalOpen(false);
    setIsReset(true);
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
    setIsReset(false);
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
        messageApi.success(intl.formatMessage({ id: 'userManage.editUserSuccess' }));
        setIsUserModalOpen(false);
        setEditName('');
        setIsReset(true);
        getUserManagesList();
      }
    } catch (e) {
      let tip = e.response.data.Msg
        ? intl.formatMessage({ id: 'userManage.editUserFailedReason' }, { msg: e.response.data.Msg })
        : intl.formatMessage({ id: 'userManage.editUserFailed' });
      messageApi.error(tip);
    }
  };

  const createUserRequest = async (data) => {
    try {
      const res = await createUser(data);
      if (res.status === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'userManage.createUserSuccess' }));
        setIsReset(true);
        setIsUserModalOpen(false);
        getUserManagesList();
      }
    } catch (e) {
      let tip = e.response.data.Msg
        ? intl.formatMessage({ id: 'userManage.createUserFailedReason' }, { msg: e.response.data.Msg })
        : intl.formatMessage({ id: 'userManage.createUserFailed' });
      messageApi.error(tip);
    }
  };
  const addUser = () => {
    setIsReset(false);
    setFormType('add');
    setIsUserModalOpen(true);
  };
  // 重置按钮
  const handleUserManageReset = () => {
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

  const handleDelUserManageConfirm = async () => {
    try {
      const res = await deleteUser(userManageDelName);
      if (res.status === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'userManage.deleteUserSuccess' }, { name: userManageDelName }));
        setUserManageDelModalOpen(false);
        setIsUserManageDelCheck(false);
        setUserManageDelName('');
        getUserManagesList();
      }
    } catch (e) {
      let tip = e.response.data.Msg
        ? intl.formatMessage({ id: 'userManage.deleteUserFailedReason' }, { msg: e.response.data.Msg })
        : intl.formatMessage({ id: 'userManage.deleteUserFailed' });
      messageApi.error(tip);
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
  const userManageColumns = [
    {
      title: '用户名称',
      key: 'user_name',
      sorter: (a, b) => sorterFirstAlphabet(a.Username, b.Username),
      render: (_, record) => (
        <Link to={{ pathname: `/${usermanageRouterPrefix}/user/detail`, state: { userName: record.Username } }}>
          {record.Username}
        </Link>
      ),
    },
    {
      title: '平台角色',
      key: 'plate_role',
      filters: filterUserRole,
      filterMultiple: false,
      filteredValue: filterRoleValue ? [filterRoleValue] : [],
      sorter: (a, b) => sorterFirstAlphabet(a.PlatformRole, b.PlatformRole),
      onFilter: (value, record) => record.PlatformRole.toLowerCase() === value.toLowerCase(),
      render: (_, record) => record.PlatformRole || '-',
    },
    {
      title: '描述',
      key: 'user_desc',
      render: (_, record) => record.Description || '-',
    },
    {
      title: '操作',
      width: 120,
      key: 'handle',
      fixed: 'right',
      render: (_, record) => (
        <Space>
          <Popover
            placement="bottom"
            content={
              <div className="pop_modal">
                <Button
                  type="link"
                  disabled={record.Username === 'admin' ? true : false}
                  onClick={() => handleEditUser(record)}
                >
                  修改
                </Button>
                <Button
                  type="link"
                  disabled={record.Username === 'admin' ? true : false}
                  onClick={() => handleDeleteUserManage(record)}
                >
                  删除
                </Button>
              </div>
            }
            trigger="click"
            open={popOpen === record.Username}
            onOpenChange={(e) => handleUserManageOpenChange(e, record)}
          >
            <MoreOutlined className="common_antd_icon primary_color" />
          </Popover>
        </Space>
      ),
    },
  ];

  // 获取userManageList
  const getUserManagesList = async (isChange = true) => {
    setUserManageLoading(true);
    try {
      const res = await getUserList();
      if (res.status === ResponseCode.OK) {
        let statusArr = [];
        let roleOption = {};
        res.data.Data.map((item) => {
          if (!Object.keys(roleOption).includes(item.PlatformRole)) {
            roleOption[item.PlatformRole] = item.PlatformRole;
          }
        });
        const statusKey = Object.keys(roleOption);
        statusKey.map((item) => {
          statusArr.push({ text: item, value: roleOption[item] });
        });
        setFilterUserRole([...statusArr]);
        setUserManageLoading(false);
        setOriginalList([...res.data.Data]);
        handleSearchUserManage(res.data.Data, isChange); // 先搜索
      }
    } catch (e) {
      setUserManageLoading(false);
      if (e.response.data.code === ResponseCode.NotFound) {
        setUserManageList([]); // 数组为空
      }
    }
  };

  // 检索
  const handleSearchUserManage = (totalData = originalList, isChange = true) => {
    const serviceAccountFormName = userManageForm.getFieldValue('user_name');
    let temporyList = totalData;
    if (serviceAccountFormName) {
      temporyList = totalData.filter((item) =>
        item.Username.toLowerCase().includes(serviceAccountFormName.toLowerCase()),
      );
    }
    setUserManageList([...temporyList]);
    isChange ? setUserManagePage(DEFAULT_CURRENT_PAGE) : null;
  };

  useEffect(() => {
    getUserManagesList();
  }, []);

  return (
    <div className="child_content">
      <BreadCrumbCom items={[{ title: '用户管理', path: `/${usermanageRouterPrefix}/user` }]} />
      <div className="tab_container container_margin_box userManageMinHeight">
        <div
          style={{
            background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff',
            color: themeStore.$s.theme === 'dark' ? '#fff' : '#333',
          }}
        >
          {contextHolder}
        </div>
        <Form className="pod_searchForm form_padding_bottom" form={userManageForm}>
          <Form.Item name="user_name" className="pod_search_input">
            <Input.Search placeholder="搜索用户名称" onSearch={() => handleSearchUserManage()} autoComplete="off" />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button className="primary_btn" onClick={addUser}>
                创建用户
              </Button>
              <Button
                icon={<SyncOutlined />}
                onClick={handleUserManageReset}
                className="reset_btn"
                style={{ marginLeft: '16px' }}
              ></Button>
            </Space>
          </Form.Item>
        </Form>
        <div className="tab_table_flex">
          <ConfigProvider locale={zhCN}>
            <Table
              className="table_padding"
              loading={userManageLoading}
              columns={userManageColumns}
              dataSource={userManageList}
              onChange={handleUserManageTableChange}
              pagination={{
                className: 'page',
                current: userManagePage,
                showTotal: (total) => `共${total}条`,
                showSizeChanger: true,
                showQuickJumper: true,
                pageSizeOptions: [10, 20, 50],
                onChange: (page) => setUserManagePage(page),
              }}
              scroll={{ x: 1280 }}
            />
          </ConfigProvider>
        </div>
        <EditUser
          formType={formType}
          isResetForm={isReset}
          isShowModel={isUserModalOpen}
          data={propsChildData}
          cancelEdit={cancelEdit}
          submitModel={handleChildClick}
        />
        <DeleteInfoModal
          title="删除用户"
          open={userManageDelModalOpen}
          cancelFn={handleDelUserManageCancel}
          content={['删除用户后将无法恢复，请谨慎操作。', `确定删除 ${userManageDelName} 吗？`]}
          isCheck={isUserManageDelCheck}
          showCheck={true}
          checkFn={handleUserManageCheckFn}
          confirmFn={handleDelUserManageConfirm}
        />
      </div>
    </div>
  );
}
