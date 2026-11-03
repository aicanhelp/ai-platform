/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { usermanageRouterPrefix } from '@/constant.js';
import zhCN from 'antd/es/locale/zh_CN';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { Button, Popover, Space, message, Table, ConfigProvider } from 'antd';
import { useState, useEffect, useStore } from 'openinula';
import { useHistory, useLocation } from 'inula-router';
import { useIntl } from 'inula-intl';
import { DownOutlined } from '@ant-design/icons';
import '@/styles/pages/podDetail.less';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { ResponseCode } from '@/common/constants';
import '@/styles/userManage/user.less';
import '@/styles/userManage/dark.less';
import EditUser from './editUser';
import { getUserDetail, editUser, deleteUser } from '@/api/clusterApi';
export default function UserDetail() {
  const history = useHistory();
  const location = useLocation();
  const intl = useIntl();
  const [messageApi, contextHolder] = message.useMessage();
  const [userPopOpen, setUserPopOpen] = useState(false); // 气泡悬浮
  const themeStore = useStore('theme');
  const { userName } = location.state;
  const [userDetailTableList, setUserDetailTableList] = useState();
  const [userDetail, setUserDetail] = useState();
  const [userManageDelModalOpen, setUserManageDelModalOpen] = useState(false); // 删除对话框展示
  const [isUserManageDelCheck, setIsUserManageDelCheck] = useState(false);
  const [isUserModalOpen, setIsUserModalOpen] = useState(false);
  const [isReset, setIsReset] = useState(false);
  const [propsChildData, setPropsChildData] = useState();
  const userDetailcolumns = [
    {
      title: intl.formatMessage({ id: 'userDetail.clusterName' }),
      dataIndex: 'name',
      render: (_, record) => record.name,
    },
    {
      title: intl.formatMessage({ id: 'userDetail.clusterRole' }),
      dataIndex: 'role',
      render: (_, record) => record.role,
    },
  ];
  // 气泡
  const handleServiceAccountPopOpenChange = (open) => {
    setUserPopOpen(open);
  };

  // 删除按钮
  const handleDeleteUser = () => {
    setUserPopOpen(false); // 气泡框
    setUserManageDelModalOpen(true);
    setIsUserManageDelCheck(false);
  };
  const handleEditUser = (data) => {
    setIsReset(false);
    setIsUserModalOpen(true);
    setPropsChildData({
      userName: userDetail?.Username,
      plateRole: userDetail?.PlatformRole,
      desc: userDetail?.Description,
    });
  };
  const cancelEdit = () => {
    setIsUserModalOpen(false);
  };
  const handleChildClick = (data) => {
    let arr = data;
    editUserRequest(arr);
  };
  const editUserRequest = async (data) => {
    try {
      const res = await editUser(userDetail?.Username, data);
      if (res.status === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'userDetail.editUserSuccess' }));
        setIsUserModalOpen(false);
        setIsReset(true);
        getPlateUserDetail();
      }
    } catch (e) {
      let tip = e.response.data.Msg
        ? intl.formatMessage({ id: 'userDetail.editUserFailedReason' }, { msg: e.response.data.Msg })
        : intl.formatMessage({ id: 'userDetail.editUserFailed' });
      messageApi.error(tip);
    }
  };
  const getPlateUserDetail = async () => {
    try {
      const res = await getUserDetail(userName);
      let datalist = [];
      if (res.status === ResponseCode.OK) {
        setUserDetail(res.data.Data);
        Object.keys(res.data.Data.InvitedByClustersMap).map((item) => {
          datalist.push({
            name: item,
            role: res.data.Data.InvitedByClustersMap[item],
          });
        });
      }
      setUserDetailTableList(datalist);
    } catch (e) {
      messageApi.error(intl.formatMessage({ id: 'userDetail.roleDetailFetchError' }));
    }
  };

  const handleDelUserManageConfirm = async () => {
    const res = await deleteUser(userName);
    if (res.status === ResponseCode.OK) {
      messageApi.success(intl.formatMessage({ id: 'userDetail.deleteUserSuccess' }, { name: userName }));
      setUserManageDelModalOpen(false);
      setIsUserManageDelCheck(false);
      setTimeout(() => {
        history.push(`/${usermanageRouterPrefix}/user`);
      }, 1000);
    } else {
      messageApi.error(intl.formatMessage({ id: 'userDetail.deleteUserFailed' }, { name: userName }));
    }
  };
  const handleDelUserManageCancel = () => {
    setUserManageDelModalOpen(false);
  };
  const handleUserManageCheckFn = (e) => {
    setIsUserManageDelCheck(e.target.checked);
  };
  useEffect(() => {
    getPlateUserDetail();
  }, []);

  return (
    <div className="child_content withBread_content user_detail">
      <div
        style={{
          background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff',
          color: themeStore.$s.theme === 'dark' ? '#fff' : '#333',
        }}
      >
        {contextHolder}
      </div>
      <BreadCrumbCom
        items={[
          { title: intl.formatMessage({ id: 'userManage.userManage' }), path: `/${usermanageRouterPrefix}/user` },
          { title: intl.formatMessage({ id: 'userDetail.userDetail' }), path: `/detail` },
        ]}
      />
      <div
        className="pod_title"
        style={{
          border: themeStore.$s.theme !== 'light' && 'none',
          backgroundColor: themeStore.$s.theme !== 'light' && '#2a2d34',
        }}
      >
        <h3>{userName}</h3>
        <Popover
          placement="bottom"
          content={
            <Space className="column_pop">
              <Button type="link" onClick={handleEditUser} disabled={userName === 'admin' ? true : false}>
                {intl.formatMessage({ id: 'common.edit' })}
              </Button>
              <Button type="link" onClick={handleDeleteUser} disabled={userName === 'admin' ? true : false}>
                {intl.formatMessage({ id: 'common.delete' })}
              </Button>
            </Space>
          }
          open={userPopOpen}
          onOpenChange={handleServiceAccountPopOpenChange}
        >
          <Button className="primary_btn">
            {intl.formatMessage({ id: 'common.action' })} <DownOutlined className="small_margin_adjust" />
          </Button>
        </Popover>
      </div>
      <div className="user_detail_info">
        <h3 className="user_detail_info_title">{intl.formatMessage({ id: 'userDetail.userInfo' })}</h3>
        <div className="user_detail_info_other">
          <div className="user_detail_info_other_single">
            <p className="user_detail_info_other_single_label">{intl.formatMessage({ id: 'userDetail.userName' })}</p>
            <p className="user_detail_info_other_single_value">{userDetail?.Username}</p>
          </div>
          <div className="user_detail_info_other_single">
            <p className="user_detail_info_other_single_label">
              {intl.formatMessage({ id: 'userDetail.platformRole' })}
            </p>
            <p className="user_detail_info_other_single_value">{userDetail?.PlatformRole}</p>
          </div>
          <div className="user_detail_info_other_single">
            <p className="user_detail_info_other_single_label">
              {intl.formatMessage({ id: 'userDetail.userDescription' })}
            </p>
            <p className="user_detail_info_other_single_value">{userDetail?.Description}</p>
          </div>
        </div>
        <h3 className="user_detail_info_title">{intl.formatMessage({ id: 'userDetail.belongingClusters' })}</h3>
        <ConfigProvider locale={zhCN}>
          <Table
            className="table_padding user_detail_info_table"
            columns={userDetailcolumns}
            dataSource={userDetailTableList}
            pagination={false}
          />
        </ConfigProvider>
      </div>
      <EditUser
        formType={'edit'}
        isResetForm={isReset}
        isShowModel={isUserModalOpen}
        data={propsChildData}
        cancelEdit={cancelEdit}
        submitModel={handleChildClick}
      />
      <DeleteInfoModal
        title={intl.formatMessage({ id: 'userDetail.deleteUserTitle' })}
        open={userManageDelModalOpen}
        cancelFn={handleDelUserManageCancel}
        content={[
          intl.formatMessage({ id: 'userDetail.deleteUserWarning1' }),
          intl.formatMessage({ id: 'userDetail.deleteUserConfirm' }, { name: userDetail?.Username }),
        ]}
        isCheck={isUserManageDelCheck}
        showCheck={true}
        checkFn={handleUserManageCheckFn}
        confirmFn={handleDelUserManageConfirm}
      />
    </div>
  );
}
