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

import Inula, { useState, useStore } from 'openinula';
import { useIntl } from 'inula-intl';
import '@/styles/pages/setUserInfo.less';
import SetUserMenu from '@/components/SetUserMenu';
import { Button, Form, Input, message } from 'antd';
import { ResponseCode } from '@/common/constants';
import { logoutSSO, updatePassword } from '@/api/loginApi';
import { CheckCircleFilled, CloseCircleFilled } from '@ant-design/icons';

function PromptLine({ valid, msg }) {
  return (
    <div className="prompt-line">
      <div className="prompt-icon">
        {valid ? (
          <CheckCircleFilled style={{ color: '#09aa71' }} />
        ) : (
          <CloseCircleFilled style={{ color: '#e7434a' }} />
        )}
      </div>
      <div className="prompt-text">{msg}</div>
    </div>
  );
}

export default function Password() {
  const userStore = useStore('user');
  const themeStore = useStore('theme');
  const intl = useIntl();

  const validPrompt1 = intl.formatMessage({ id: 'settings.passwordLengthHint' });
  const validPrompt2 = intl.formatMessage({ id: 'settings.passwordCharHint' });
  const validPrompt3 = intl.formatMessage({ id: 'settings.passwordNoUsername' });
  const confirmPrompt = intl.formatMessage({ id: 'settings.passwordsMustMatch' });
  const [passwordForm] = Form.useForm();
  const [messageApi, contextHolder] = message.useMessage();
  const [newPwd, setNewPwd] = useState('');
  const [confirmPwd, setConfitmPwd] = useState('');
  const [passwordValid1, setPasswordValid1] = useState(false);
  const [passwordValid2, setPasswordValid2] = useState(false);
  const [passwordValid3, setPasswordValid3] = useState(false);
  const [confirmValid, setConfirmValid] = useState(false);

  const passwordPattern =
    /^(?=.*[0-9])(?=.*[A-Za-z])(?=.*[`~!@#$%^&*()\-_=+\\|\[{}\];:'",<.>/?])[A-Za-z0-9`~!@#$%^&*()\-_=+\\|\[{}\];:'",<.>/?]+$/;
  const reverseStr = (s) => s.split('').reverse().join('');

  const handleNewInput = (e) => {
    const newPassword = e.target.value;
    setNewPwd(newPassword);

    setPasswordValid1(newPassword.length >= 8 && newPassword.length <= 32);
    setPasswordValid2(passwordPattern.test(newPassword));
    const username = userStore.$s.user.name;
    setPasswordValid3(newPassword !== username && reverseStr(newPassword) !== username);
    setConfirmValid(newPassword === confirmPwd);
  };

  const handleConfirmInput = (e) => {
    const confirmPassword = e.target.value;
    setConfitmPwd(confirmPassword);
    setConfirmValid(confirmPassword === newPwd);
  };

  const handleSubmit = async () => {
    let encoder = new TextEncoder();
    updatePassword(
      userStore.$s.user.name,
      Object.values(encoder.encode(passwordForm.getFieldValue('old_password'))),
      Object.values(encoder.encode(passwordForm.getFieldValue('new_password'))),
    )
      .then((res) => {
        if (res.status === ResponseCode.OK) {
          messageApi.success(intl.formatMessage({ id: 'settings.passwordChangeSuccessLogout' }), 1, () => {
            logoutSSO().then(() => {
              setTimeout(() => {
                window.location.href = '/rest/auth/login';
              }, 1000);
            });
          });
        }
      })
      .catch((error) => {
        if (error.response.status === ResponseCode.Found) {
          messageApi.error(intl.formatMessage({ id: 'settings.passwordChangeLockout' }), 1, () => {
            logoutSSO().then(() => {
              setTimeout(() => {
                window.location.href = '/rest/auth/login';
              }, 1000);
            });
          });
        }
        if (error.response.status === ResponseCode.UnAuthorized) {
          messageApi.error(intl.formatMessage({ id: 'settings.passwordChangeFailed' }), 1);
        }
      });
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
          <h3>{intl.formatMessage({ id: 'settings.passwordSettings' })}</h3>
          <Form className="formList update_form" form={passwordForm} layout="vertical" onFinish={handleSubmit}>
            <Form.Item
              label={intl.formatMessage({ id: 'settings.currentPassword' })}
              name="old_password"
              className="modify_password_item"
              rules={[
                {
                  required: true,
                  message: intl.formatMessage({ id: 'settings.inputCurrentPassword' }),
                },
              ]}
            >
              <Input.Password
                className="user_input"
                placeholder={intl.formatMessage({ id: 'common.pleaseInput' })}
                onCopy={(e) => e.preventDefault()}
              />
            </Form.Item>

            <Form.Item label={intl.formatMessage({ id: 'settings.newPassword' })} className="modify_password_item">
              <Form.Item
                name="new_password"
                rules={[
                  {
                    required: true,
                    message: intl.formatMessage({ id: 'settings.inputNewPassword' }),
                  },
                ]}
              >
                <Input.Password
                  className="user_input"
                  placeholder={intl.formatMessage({ id: 'common.pleaseInput' })}
                  onInput={handleNewInput}
                  onCopy={(e) => e.preventDefault()}
                />
              </Form.Item>
              <PromptLine valid={passwordValid1} msg={validPrompt1} />
              <PromptLine valid={passwordValid2} msg={validPrompt2} />
              <PromptLine valid={passwordValid3} msg={validPrompt3} />
            </Form.Item>

            <Form.Item label={intl.formatMessage({ id: 'settings.confirmPassword' })} className="modify_password_item">
              <Form.Item
                name="confirm_password"
                dependencies={['new_password']}
                rules={[
                  {
                    required: true,
                    message: intl.formatMessage({ id: 'settings.confirmNewPassword' }),
                  },
                ]}
              >
                <Input.Password
                  className="user_input"
                  placeholder={intl.formatMessage({ id: 'common.pleaseInput' })}
                  onInput={handleConfirmInput}
                  onCopy={(e) => e.preventDefault()}
                />
              </Form.Item>
              <PromptLine valid={confirmValid} msg={confirmPrompt} />
            </Form.Item>
            <Form.Item className="modify_password_btn">
              <Button
                className="primary_btn"
                htmlType="submit"
                disabled={!(passwordValid1 && passwordValid2 && passwordValid3 && confirmValid)}
              >
                {intl.formatMessage({ id: 'common.confirm' })}
              </Button>
            </Form.Item>
          </Form>
        </div>
      </div>
    </div>
  );
}
