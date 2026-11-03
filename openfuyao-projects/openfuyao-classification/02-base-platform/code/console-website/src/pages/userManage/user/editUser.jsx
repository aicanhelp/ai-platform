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
import { Select, Button, Form, Input, Modal, message } from 'antd';
import { useEffect, useState, useMemo, useStore } from 'openinula';
import '@/styles/userManage/user.less';
import { ResponseCode, PASSWORD_PATTERN } from '@/common/constants';
import { getRoleList } from '@/api/clusterApi';
import { stringToAscii } from '@/tools/utils';
import '@/styles/userManage/dark.less';
import useButtonLoading from '@/hooks/useButtonLoading';

export default function EditUserModel(props) {
  const { t } = useTranslation('setting');
  const { TextArea } = Input;
  const [userForm] = Form.useForm();
  const [messageApi, contextHolder] = message.useMessage();
  const themeStore = useStore('theme');
  const [plateRoleOption, setPlateRoleOption] = useState([]);
  const [submitLoading, setSubmitLoading, submitPending] = useButtonLoading();
  const formType = props?.formType;

  const cancelModelEdit = () => {
    props.cancelEdit();
  };

  const submitModelUser = async () => {
    try {
      await userForm.validateFields();
    } catch {
      return; // 校验失败：antd 自动标红，阻断提交
    }
    setSubmitLoading(true);
    try {
      await submitModelUserInner();
    } finally {
      setSubmitLoading(false);
    }
  };

  const submitModelUserInner = () => {
    if (formType === 'add') {
      props.submitModel({
        Username: userForm.getFieldsValue().userName,
        PlatformRole: userForm.getFieldsValue().plateRole,
        Description: userForm.getFieldsValue().desc,
        UnEncryptedPassword: stringToAscii(userForm.getFieldsValue().password),
      });
      return;
    }
    props.submitModel({
      Username: userForm.getFieldsValue().userName,
      PlatformRole: userForm.getFieldsValue().plateRole,
      Description: userForm.getFieldsValue().desc,
    });
  };

  // 自定义密码校验函数
  const validatePassword = async (_rule, value) => {
    const regex = PASSWORD_PATTERN;
    if (!value) {
      throw new Error(t('validation.passwordEmpty'));
    }
    if (value.length > 32 || value.length < 8) {
      throw new Error(t('validation.passwordLengthRange'));
    }
    if (!regex.test(value)) {
      throw new Error(t('validation.passwordChars'));
    }
    return Promise.resolve();
  };

  const getPlateRoleList = async () => {
    try {
      const res = await getRoleList();
      let option = [];
      if (res.status === ResponseCode.OK) {
        res.data.Data.map(item => {
          option.push({
            label: item.metadata.name,
            value: item.metadata.name,
          });
        });
      }
      setPlateRoleOption(option);
    } catch {
      messageApi.error(t('messages.roleListError'));
    }
  };

  // 自定义用户名校验函数
  const validateName = async (_rule, value) => {
    const regex = /^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$/;
    if (!value) {
      throw new Error(t('validation.usernameEmpty'));
    }
    if (value.length > 32) {
      throw new Error(t('validation.usernameMaxLength'));
    }
    if (!regex.test(value)) {
      return Promise.reject(new Error(t('validation.usernameFormat')));
    }
    return Promise.resolve();
  };

  useEffect(() => {
    getPlateRoleList();
  }, []);

  useEffect(() => {
    userForm.setFieldsValue({
      userName: props.data?.userName,
      plateRole: props.data?.plateRole,
      password: '',
      desc: props.data?.desc,
    });
  }, [props.data?.userName, props.data?.plateRole, props.data?.desc]);

  const modalTitle = useMemo(() =>
    formType === 'add' ? t('modal.createUserTitle') : t('modal.modifyUserTitle')
  , [formType, t]);

  return (
    <Modal width={682} forceRender getContainer={false} title={modalTitle} open={props.isShowModel} onCancel={cancelModelEdit} className='silentModelCommon'
      footer={[]}
    >
      <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
        {contextHolder}
      </div>
      <Form form={userForm} name='userForm' layout="vertical" style={{ marginTop: '20px' }}>
        <Form.Item name='userName' label={t('table.userName')} rules={[{ required: true, validator: validateName }]}>
          <Input placeholder={t('placeholder.inputUserName')} disabled={formType !== 'add'} />
        </Form.Item>
        {formType === 'add' && (
          <Form.Item name='password' label={t('label.initialPassword')} rules={[{ required: true, validator: validatePassword }]}>
            <Input.Password placeholder={t('placeholder.inputPassword')} type='password' />
          </Form.Item>
        )}
        <Form.Item name='plateRole' label={t('label.platformRole')} rules={[{ required: true, message: t('validation.platformRoleRequired') }]}>
          <Select options={plateRoleOption} />
        </Form.Item>
        <Form.Item name='desc' label={t('label.userDescription')}>
          <TextArea rows={4} maxLength={256} showCount />
        </Form.Item>
        <Form.Item style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <Button className='cancel_btn' onClick={cancelModelEdit} style={{ marginRight: '14px' }}>{t('actions.cancel')}</Button>
          <Button className='primary_btn' onClick={submitModelUser} loading={submitLoading} disabled={submitPending}>{t('actions.confirm')}</Button>
        </Form.Item>
      </Form>
    </Modal >
  );
}