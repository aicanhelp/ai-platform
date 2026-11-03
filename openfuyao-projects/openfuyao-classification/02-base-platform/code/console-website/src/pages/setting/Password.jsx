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

import { useState, useStore } from 'openinula';
import { useTranslation } from 'react-i18next';
import '@/styles/pages/setUserInfo.less';
import SetUserMenu from '@/components/SetUserMenu';
import { Button, Form, Input, message } from 'antd';
import { ResponseCode, PASSWORD_PATTERN } from '@/common/constants';
import { logoutSSO, updatePassword } from '@/api/loginApi';
import { showSuccessThen } from '@/tools/utils';
import { CheckCircleFilled, CloseCircleFilled } from '@ant-design/icons';
import useButtonLoading from '@/hooks/useButtonLoading';

function PromptLine({ valid, msg }) {
  return (
    <div className="prompt-line">
      <div className="prompt-icon">
        {valid ?
          <CheckCircleFilled style={{ color: '#09aa71' }} /> :
          <CloseCircleFilled style={{ color: '#e7434a' }} />}
      </div>
      <div className="prompt-text">{msg}</div>
    </div>
  );
}

export default function Password() {
  const { t } = useTranslation('setting');
  const userStore = useStore('user');
  const themeStore = useStore('theme');
  const [passwordForm] = Form.useForm();
  const [messageApi, contextHolder] = message.useMessage();
  const [newPwd, setNewPwd] = useState('');
  const [confirmPwd, setConfirmPwd] = useState('');
  const [passwordValid1, setPasswordValid1] = useState(false);
  const [passwordValid2, setPasswordValid2] = useState(false);
  const [passwordValid3, setPasswordValid3] = useState(false);
  const [confirmValid, setConfirmValid] = useState(false);
  const [submitLoading, setSubmitLoading, submitPending] = useButtonLoading();

  const reverseStr = (s) => s.split('').reverse().join('');

  const handleNewInput = (e) => {
    const newPassword = e.target.value;
    setNewPwd(newPassword);

    setPasswordValid1(newPassword.length >= 8 && newPassword.length <= 32);
    setPasswordValid2(PASSWORD_PATTERN.test(newPassword));
    const username = userStore.$s.user.name;
    setPasswordValid3(newPassword !== username && reverseStr(newPassword) !== username);
    setConfirmValid(newPassword === confirmPwd);
  };

  const handleConfirmInput = (e) => {
    const confirmPassword = e.target.value;
    setConfirmPwd(confirmPassword);
    setConfirmValid(confirmPassword === newPwd);
  };

  const handleSubmit = async () => {
    setSubmitLoading(true);
    let encoder = new TextEncoder();
    try {
      const res = await updatePassword(
        userStore.$s.user.name,
        Object.values(encoder.encode(passwordForm.getFieldValue('old_password'))),
        Object.values(encoder.encode(passwordForm.getFieldValue('new_password'))),
      );
      if (res.status === ResponseCode.OK) {
        await showSuccessThen(messageApi, t('messages.passwordChangeSuccess'), () => {
          logoutSSO().then(() => {
            window.location.href = '/rest/auth/login';
          });
        });
      }
    } catch (error) {
      if (error.response?.status === ResponseCode.Found) {
        messageApi.error(t('messages.passwordChangeLocked'), 1, () => {
          logoutSSO()
            .then(() => {
              setTimeout(() => {
                window.location.href = '/rest/auth/login';
              }, 1000);
            });
        });
      } else if (error.response?.status === ResponseCode.UnAuthorized) {
        messageApi.error(`${t('messages.passwordChangeFailed')}: ${error.response?.data?.message || error.message}`);
      } else {
        messageApi.error(error.response?.data?.message || t('messages.passwordChangeFailed'));
      }
    } finally {
      setSubmitLoading(false);
    }
  };

  return <div className="user_set_box">
    <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
      {contextHolder}
    </div>
    <p className="user_title">{t('module.userSettings')}</p>
    <div className="user_set_content">
      <SetUserMenu />
      <div className="user_info_content">
        <h3>{t('title.passwordSettings')}</h3>
        <Form className="formList update_form" form={passwordForm} layout="vertical" onFinish={handleSubmit}>
          <Form.Item
            label={t('label.currentPassword')}
            name="old_password"
            className="modify_password_item"
            rules={[
              {
                required: true,
                message: t('validation.currentPasswordRequired'),
              },
            ]}
          >
            <Input.Password className="user_input" placeholder={t('placeholder.inputPassword')} onCopy={(e) => e.preventDefault()} />
          </Form.Item>

          <Form.Item
            className="modify_password_item"
          >
            <Form.Item
              label={t('label.newPassword')}
              name="new_password"
              rules={[
                {
                  required: true,
                  message: t('validation.newPasswordRequired'),
                },
              ]}
            >
              <Input.Password className="user_input" placeholder={t('placeholder.inputNewPassword')} onInput={handleNewInput} onCopy={(e) => e.preventDefault()} />
            </Form.Item>
            <PromptLine valid={passwordValid1} msg={t('validation.passwordLength')} />
            <PromptLine valid={passwordValid2} msg={t('validation.passwordComplexity')} />
            <PromptLine valid={passwordValid3} msg={t('validation.passwordNotSameAsAccount')} />
          </Form.Item>

          <Form.Item
            className="modify_password_item"
          >
            <Form.Item
              label={t('label.confirmPassword')}
              name="confirm_password"
              dependencies={['new_password']}
              rules={[
                {
                  required: true,
                  message: t('validation.confirmPasswordRequired'),
                },
              ]}>
              <Input.Password className="user_input" placeholder={t('placeholder.inputConfirmPassword')} onInput={handleConfirmInput} onCopy={(e) => e.preventDefault()} />
            </Form.Item>
            <PromptLine valid={confirmValid} msg={t('validation.passwordMustMatch')} />
          </Form.Item>
          <Form.Item className="modify_password_btn">
            <Button className="primary_btn" htmlType="submit" loading={submitLoading} disabled={
              submitPending || !(passwordValid1 && passwordValid2 && passwordValid3 && confirmValid)
            }>{t('actions.confirm')}</Button>
          </Form.Item>
        </Form>
      </div>
    </div>
  </div>;
}