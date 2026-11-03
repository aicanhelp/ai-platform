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

import { useTranslation } from 'react-i18next';
import '@/styles/pages/setUserInfo.less';
import SetUserMenu from '@/components/SetUserMenu';
import { Avatar, Button, Form, Input, message } from 'antd';
import { useState, useStore, useEffect } from 'openinula';
import { UserOutlined } from '@ant-design/icons';
import { getUserInfoByname, editUserInfo } from '@/api/loginApi';
import { showSuccessThen } from '@/tools/utils';
import { ResponseCode } from '@/common/constants';
import useButtonLoading from '@/hooks/useButtonLoading';

export default function UserInfo() {
  const { t } = useTranslation('setting');
  const [setUserForm] = Form.useForm();
  const [userAvatar, setUserAvatar] = useState('');
  const [messageApi, contextHolder] = message.useMessage();
  const { TextArea } = Input;
  const userStore = useStore('user');
  const themeStore = useStore('theme');
  const [isReadonly, setIsReadonly] = useState(true);
  const [submitLoading, setSubmitLoading, submitPending] = useButtonLoading();

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
    setIsReadonly(false);
  };

  const handleCancelInfo = () => {
    setIsReadonly(true);
  };

  const handleSubmitInfo = async () => {
    setSubmitLoading(true);
    try {
      let data = {
        Username: setUserForm.getFieldValue('username'),
        Description: setUserForm.getFieldValue('description'),
      };
      const res = await editUserInfo(data.Username, data);
      if (res.data.Code === ResponseCode.OK) {
        await showSuccessThen(messageApi, t('messages.editDescriptionSuccess', { name: data.Username }), () => {
          getInfo();
          setIsReadonly(true);
        });
      } else {
        messageApi.error(t('messages.editDescriptionFailed', { name: data.Username }));
      }
    } catch (e) {
      messageApi.error(e?.response?.data?.Msg || t('messages.editDescriptionFailed', { name: setUserForm.getFieldValue('username') }));
    } finally {
      setSubmitLoading(false);
    }
  };

  useEffect(() => {
    getInfo();
  }, [userStore.$s.user.name]);

  return <div className="user_set_box">
    <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
      {contextHolder}
    </div>
    <p className='user_title'>{t('module.userSettings')}</p>
    <div className='user_set_content'>
      <SetUserMenu />
      <div className='user_info_content'>
        <h3 style={{ color: '#fff' }}>{t('title.basicInfo')}</h3>
        <Form className='formList userinfo_form' form={setUserForm} layout='vertical'>
          <Form.Item name="avatar" className='modify_password_item'>
            <div className='avatar_pic'>
              {userAvatar ? <Avatar shape='square' size={120} src={userAvatar}></Avatar> : <Avatar shape='square' size={120} icon={<UserOutlined />}></Avatar>}
            </div>
          </Form.Item>
          <Form.Item label={t('label.username')} name="username">
            <Input className='user_input' disabled />
          </Form.Item>
          <Form.Item label={t('table.description')} name="description">
            <TextArea rows={4} disabled={isReadonly} maxLength={256} showCount />
          </Form.Item>
          <Form.Item style={{ display: isReadonly ? 'flex' : 'none', justifyContent: 'flex-end' }}>
            <Button className="primary_btn" onClick={handleEditInfo}>{t('actions.edit')}</Button>
          </Form.Item>
          <Form.Item style={{ display: isReadonly ? 'none' : 'flex', justifyContent: 'flex-end' }}>
            <Button className="cancel_btn" onClick={handleCancelInfo} style={{ marginRight: '16px' }}>{t('actions.cancel')}</Button>
            <Button className="primary_btn" onClick={handleSubmitInfo} loading={submitLoading} disabled={submitPending}>{t('actions.confirm')}</Button>
          </Form.Item>
        </Form>
      </div>
    </div>
  </div>;
}