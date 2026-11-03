/**
 *  Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  openFuyao is licensed under Mulan PSL v2.
 *  You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  You may obtain a copy of Mulan PSL v2 at:

 *       http://license.coscl.org.cn/MulanPSL2

 *   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *   EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *   MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *   See the Mulan PSL v2 for more details.
 */

import '@/styles/pages/setUserInfo.less';
import SetUserMenu from '@/components/SetUserMenu';
import { Avatar, Button, Form, Input, Upload, message } from 'antd';
import { useState, useStore, useEffect } from 'openinula';
import { useIntl } from 'inula-intl';
import { UserOutlined, PlusCircleOutlined } from '@ant-design/icons';
import UserStore from '@/store/userStore';
import { getUserInfoByname, editUserInfo } from '@/api/loginApi';
import { ResponseCode } from '@/common/constants';
export default function UserInfo() {
  const intl = useIntl();
  const [setUserForm] = Form.useForm();
  const [userAvatar, setUserAvatar] = useState('');
  const [messageApi, contextHolder] = message.useMessage();
  const { TextArea } = Input;
  const userStore = useStore('user');
  const themeStore = useStore('theme');
  const uploadEnable = false;
  const [isEdit, setIsEdit] = useState(true);
  const getInfo = async () => {
    const res = await getUserInfoByname(userStore.$s.user.name);
    if (res.data.Code === ResponseCode.OK) {
      if (setUserForm) {
        setUserAvatar(userStore.$s.user.avatar);
        setUserForm.setFieldValue('username', res.data.Data.Username);
        setUserForm.setFieldValue('description', res.data.Data.Description);
      }
    }
  };
  const handleEditInfo = () => {
    setIsEdit(false);
  };
  const handleCancelInfo = () => {
    setIsEdit(true);
  };
  const handleSubmitInfo = async () => {
    let data = {
      Username: setUserForm.getFieldValue('username'),
      Description: setUserForm.getFieldValue('description'),
    };
    const res = await editUserInfo(data.Username, data);
    if (res.data.Code === ResponseCode.OK) {
      messageApi.success(intl.formatMessage({ id: 'settings.editDescSuccess' }, { name: data.Username }));
      getInfo();
      setIsEdit(true);
    } else {
      messageApi.error(intl.formatMessage({ id: 'settings.editDescFailed' }, { name: data.Username }));
    }
  };
  useEffect(() => {
    getInfo();
  }, [userStore.$s.user.name]);

  const props = {
    name: 'file',
    action: '',
    onChange(info) {
      if (info.file.status === 'done') {
        setUserAvatar('');
        messageApi.success(intl.formatMessage({ id: 'settings.uploadSuccess' }, { name: info.file.name }));
      } else if (info.file.status === 'error') {
        messageApi.error(intl.formatMessage({ id: 'settings.uploadFailed' }, { name: info.file.name }));
      } else {
      }
    },
  };

  return (
    <div className="user_set_box">
      <div
        style={{
          background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff',
          color: themeStore.$s.theme === 'dark' ? '#fff' : '#333',
        }}
      >
        {contextHolder}
      </div>
      <p className="user_title">{intl.formatMessage({ id: 'settings.userSettings' })}</p>
      <div className="user_set_content">
        <SetUserMenu />
        <div className="user_info_content">
          <h3 style={{ color: '#fff' }}>{intl.formatMessage({ id: 'settings.basicInfo' })}</h3>
          <Form className="formList userinfo_form" form={setUserForm} layout="vertical">
            <Form.Item name="avatar" className="modify_password_item">
              <div className="avatar_pic">
                {userAvatar ? (
                  <Avatar shape="square" size={120} src={userAvatar}></Avatar>
                ) : (
                  <Avatar shape="square" size={120} icon={<UserOutlined />}></Avatar>
                )}
                {uploadEnable && (
                  <Upload {...props} className="upload">
                    <PlusCircleOutlined />
                    <span className="upload_word">{intl.formatMessage({ id: 'settings.uploadFile' })}</span>
                  </Upload>
                )}
              </div>
            </Form.Item>
            <Form.Item label={intl.formatMessage({ id: 'settings.username' })} name="username">
              <Input className="user_input" disabled />
            </Form.Item>
            <Form.Item label={intl.formatMessage({ id: 'settings.description' })} name="description">
              <TextArea rows={4} disabled={isEdit} maxLength={256} showCount />
            </Form.Item>
            <Form.Item style={{ display: isEdit ? 'flex' : 'none', justifyContent: 'flex-end' }}>
              <Button className="primary_btn" onClick={handleEditInfo}>
                {intl.formatMessage({ id: 'common.edit' })}
              </Button>
            </Form.Item>
            <Form.Item style={{ display: isEdit ? 'none' : 'flex', justifyContent: 'flex-end' }}>
              <Button className="cancel_btn" onClick={handleCancelInfo} style={{ marginRight: '16px' }}>
                {intl.formatMessage({ id: 'common.cancel' })}
              </Button>
              <Button className="primary_btn" onClick={handleSubmitInfo}>
                {intl.formatMessage({ id: 'common.confirm' })}
              </Button>
            </Form.Item>
          </Form>
        </div>
      </div>
    </div>
  );
}
