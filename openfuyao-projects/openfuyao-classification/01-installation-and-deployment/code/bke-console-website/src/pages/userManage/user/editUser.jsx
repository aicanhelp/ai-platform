/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Select, Button, Form, Input, Modal } from 'antd';
import { useEffect, useState } from 'openinula';
import { useIntl } from 'inula-intl';
import '@/styles/userManage/user.less';
import { ResponseCode } from '@/common/constants';
import { getRoleList } from '@/api/clusterApi';
import { stringToAscii } from '@/tools/utils';
import '@/styles/userManage/dark.less';
export default function EditUserModel(props) {
  const intl = useIntl();
  const { TextArea } = Input;
  const [userForm] = Form.useForm();
  const [plateRoleOption, setPlateRoleOption] = useState();
  const [plateRole, setPlateRole] = useState(props?.plateRole);
  const formType = props?.formType;
  const handleChangePlateRole = (value) => {
    setPlateRole(value);
  };
  const cancelModelEdit = () => {
    props.cancelEdit();
  };
  const submitModelUser = () => {
    let arr = {};
    if (formType === 'add') {
      arr = {
        Username: userForm.getFieldsValue().userName,
        PlatformRole: userForm.getFieldsValue().plateRole,
        Description: userForm.getFieldsValue().desc,
        UnEncryptedPassword: stringToAscii(userForm.getFieldsValue().password),
      };
    } else {
      arr = {
        Username: userForm.getFieldsValue().userName,
        PlatformRole: userForm.getFieldsValue().plateRole,
        Description: userForm.getFieldsValue().desc,
      };
    }
    props.submitModel(arr);
  };
  // 自定义密码校验函数
  const validatePassword = async (_rule, value) => {
    const regex =
      /^(?=.*[0-9])(?=.*[A-Za-z])(?=.*[`~!@#$%^&*()\-_=+\\|\[{}\];:'",<.>/?])[A-Za-z0-9`~!@#$%^&*()\-_=+\\|\[{}\];:'",<.>/?]+$/;
    if (!value) {
      throw new Error(intl.formatMessage({ id: 'editUser.passwordRequired' }));
    }
    if (value.length > 32 || value.length < 8) {
      throw new Error(intl.formatMessage({ id: 'editUser.passwordLengthError' }));
    }
    if (!regex.test(value)) {
      throw new Error(intl.formatMessage({ id: 'editUser.passwordCharError' }));
    }
    return Promise.resolve();
  };

  const getPlateRoleList = async () => {
    try {
      const res = await getRoleList();
      let option = [];
      if (res.status === ResponseCode.OK) {
        res.data.Data.map((item) => {
          option.push({
            label: item.metadata.name,
            value: item.metadata.name,
          });
        });
      }
      setPlateRoleOption(option);
    } catch (e) {
      messageApi.error(intl.formatMessage({ id: 'editUser.roleListFetchError' }));
    }
  };
  // 自定义用户名校验函数
  const validateName = async (_rule, value) => {
    const regex = /^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$/;
    if (!value) {
      throw new Error(intl.formatMessage({ id: 'editUser.usernameRequired' }));
    }
    if (value.length > 32) {
      throw new Error(intl.formatMessage({ id: 'editUser.usernameLengthError' }));
    }
    if (!regex.test(value)) {
      return Promise.reject(new Error(intl.formatMessage({ id: 'editUser.usernameFormatError' })));
    }
    return Promise.resolve();
  };

  useEffect(() => {
    getPlateRoleList();
  }, []);
  useEffect(() => {
    if (props.isResetForm) {
      userForm.resetFields();
    }
  }, [props.isResetForm]);

  useEffect(() => {
    userForm.setFieldsValue({
      userName: props.data?.userName,
      plateRole: props.data?.plateRole,
      password: '',
      desc: props.data?.desc,
    });
  }, [props.data?.userName, props.data?.plateRole, props.data?.desc]);
  return (
    <Modal
      width={682}
      forceRender
      getContainer={false}
      title={
        formType === 'add'
          ? intl.formatMessage({ id: 'editUser.createUser' })
          : intl.formatMessage({ id: 'editUser.editUser' })
      }
      open={props.isShowModel}
      onCancel={cancelModelEdit}
      className="silentModelCommon"
      footer={[]}
    >
      <Form form={userForm} name="userForm" layout="vertical" style={{ marginTop: '20px' }}>
        <Form.Item
          name="userName"
          label={intl.formatMessage({ id: 'editUser.usernameLabel' })}
          rules={[{ required: true, validator: validateName }]}
        >
          <Input placeholder={intl.formatMessage({ id: 'common.pleaseInput' })} disabled={formType !== 'add'} />
        </Form.Item>
        <Form.Item
          name="password"
          label={intl.formatMessage({ id: 'editUser.initialPassword' })}
          rules={[{ required: true, validator: validatePassword }]}
          style={{ display: formType === 'edit' ? 'none' : 'block' }}
        >
          <Input.Password placeholder={intl.formatMessage({ id: 'editUser.inputPassword' })} type="password" />
        </Form.Item>
        <Form.Item
          name="plateRole"
          label={intl.formatMessage({ id: 'editUser.platformRoleLabel' })}
          rules={[{ required: true, message: intl.formatMessage({ id: 'editUser.platformRoleRequired' }) }]}
        >
          <Select value={plateRole} options={plateRoleOption} onChange={handleChangePlateRole} />
        </Form.Item>
        <Form.Item name="desc" label={intl.formatMessage({ id: 'editUser.descriptionLabel' })}>
          <TextArea rows={4} maxLength={256} showCount />
        </Form.Item>
        <Form.Item style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <Button className="cancel_btn" onClick={cancelModelEdit} style={{ marginRight: '14px' }}>
            {intl.formatMessage({ id: 'common.cancel' })}
          </Button>
          <Button className="primary_btn" onClick={submitModelUser}>
            {intl.formatMessage({ id: 'common.confirm' })}
          </Button>
        </Form.Item>
      </Form>
    </Modal>
  );
}
