/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { useParams } from 'inula-router';
import { Form, Modal, Input, Button, Table, ConfigProvider, message, Tooltip, InputNumber } from 'antd';
import { DeleteOutlined, PlusCircleOutlined, QuestionCircleOutlined } from '@ant-design/icons';
import '@/styles/pages/createNode.less';
import useLocalStorage from '@/hooks/useLocalStorage';
import useButtonLoading from '@/hooks/useButtonLoading';
import { useIntl } from '../../i18n';

export default function AddNode({
  open,
  okFn,
  cancelFn,
  errorIP,
  checkLoading,
  handleCheck,
  isCheckPass,
  handleNodeValueChange,
}) {
  const [addNodeForm] = Form.useForm();
  const intl = useIntl();
  const nodeWatchList = Form.useWatch('node', addNodeForm);
  const [currentTheme, setCurrentTheme] = useLocalStorage('theme', 'light');
  const [messageApi, contextHolder] = message.useMessage();
  const [submitLoading, setSubmitLoading, submitPending] = useButtonLoading();

  const handleOk = async (form) => {
    if (submitPending) return;
    setSubmitLoading(true);
    try {
      await okFn(form);
    } catch (error) {
      // 调用方未自行捕获异常时兜底，避免 unhandled rejection 导致按钮卡死
    } finally {
      setSubmitLoading(false);
    }
  };

  const modifyStatus = (index) => {
    if (nodeWatchList && errorIP) {
      if (nodeWatchList[index] && nodeWatchList[index].ip === errorIP) {
        return 'error';
      }
    }
    return '';
  };

  const addNodeColumns = [
    {
      title: intl.formatMessage({ id: 'node.name' }),
      width: 220,
      key: 'hostname',
      render: (value, { field }) => {
        return (
          <Form.Item
            name={[field.name, 'hostname']}
            rules={[
              { required: true, message: intl.formatMessage({ id: 'node.nameRequired' }) },
              {
                min: 1,
                message: intl.formatMessage({ id: 'node.nameValidation' }),
              },
              {
                max: 63,
                message: intl.formatMessage({ id: 'node.nameValidation' }),
              },
              {
                pattern: /^(?!\d+$)[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$/,
                message: intl.formatMessage({ id: 'node.nameValidation' }),
              },
            ]}
          >
            <Input placeholder={intl.formatMessage({ id: 'node.namePlaceholder' })} autoComplete="off" />
          </Form.Item>
        );
      },
    },
    {
      title: intl.formatMessage({ id: 'node.ip' }),
      width: 220,
      key: 'ip',
      render: (value, { field }) => {
        return (
          <Form.Item
            name={[field.name, 'ip']}
            validateStatus={modifyStatus(field.name)}
            rules={[
              { required: true, message: intl.formatMessage({ id: 'node.ipRequired' }) },
              {
                pattern:
                  /^(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])$/,
                message: intl.formatMessage({ id: 'node.ipv4Only' }),
              },
            ]}
          >
            <Input placeholder={intl.formatMessage({ id: 'node.ipPlaceholder' })} autoComplete="off" />
          </Form.Item>
        );
      },
    },
    {
      title: intl.formatMessage({ id: 'node.port' }),
      width: 220,
      key: 'port',
      render: (value, { field }) => {
        return (
          <Form.Item
            name={[field.name, 'port']}
            rules={[
              { required: true, message: intl.formatMessage({ id: 'node.portRequired' }) },
              {
                validator: (_, val) => {
                  if (val >= 0 && val <= 65535) {
                    return Promise.resolve();
                  }
                  return Promise.reject(new Error(intl.formatMessage({ id: 'node.portRange' })));
                },
              },
            ]}
          >
            <InputNumber
              style={{ width: '100%' }}
              placeholder={intl.formatMessage({ id: 'node.portPlaceholder' })}
              autoComplete="off"
            />
          </Form.Item>
        );
      },
    },
    {
      title: (
        <div>
          <span style={{ marginRight: '16px' }}>{intl.formatMessage({ id: 'node.username' })}</span>
          <Tooltip title={intl.formatMessage({ id: 'node.usernameTooltip' })}>
            <QuestionCircleOutlined style={{ width: '16px', height: '16px', color: '#89939B', cursor: 'pointer' }} />
          </Tooltip>
        </div>
      ),
      width: 220,
      key: 'userName',
      render: (value, { field }) => {
        return (
          <Form.Item
            name={[field.name, 'userName']}
            rules={[{ required: true, message: intl.formatMessage({ id: 'node.usernameRequired' }) }]}
          >
            <Input placeholder={intl.formatMessage({ id: 'node.usernamePlaceholder' })} autoComplete="off" />
          </Form.Item>
        );
      },
    },
    {
      title: (
        <div>
          <span style={{ marginRight: '16px' }}>{intl.formatMessage({ id: 'node.password' })}</span>
          <Tooltip title={intl.formatMessage({ id: 'node.passwordTooltip' })}>
            <QuestionCircleOutlined style={{ width: '16px', height: '16px', color: '#89939B', cursor: 'pointer' }} />
          </Tooltip>
        </div>
      ),
      width: 220,
      key: 'password',
      render: (value, { field }) => {
        return (
          <Form.Item
            name={[field.name, 'password']}
            rules={[{ required: true, message: intl.formatMessage({ id: 'node.passwordRequired' }) }]}
          >
            <Input.Password autoComplete="off" />
          </Form.Item>
        );
      },
    },
    {
      title: intl.formatMessage({ id: 'node.action' }),
      key: 'handle',
      fixed: 'right',
      width: 150,
      render: (value, { operation, field, isRemoveAble }) => (
        <Form.Item>
          <DeleteOutlined
            className="common_antd_icon primary_color"
            onClick={() => {
              if (isRemoveAble) {
                operation.remove(field.name);
              } else {
                messageApi.error(intl.formatMessage({ id: 'node.atLeastOne' }));
              }
            }}
          />
        </Form.Item>
      ),
    },
  ];
  return (
    <>
      {contextHolder}
      <Modal
        getContainer={() => document.querySelector('#cluster_root .container_content')}
        title={intl.formatMessage({ id: 'node.addNode' })}
        open={open}
        onCancel={() => cancelFn(addNodeForm)}
        className={currentTheme === 'dark' ? 'dark_box addNodeBoxModal' : 'addNodeBoxModal'}
        width={1132}
        footer={[
          <div style={{ marginTop: '32px' }}>
            <Button className="cancel_btn" onClick={() => cancelFn(addNodeForm)} disabled={submitPending}>
              {intl.formatMessage({ id: 'common.cancel' })}
            </Button>
            <Tooltip title={intl.formatMessage({ id: 'node.validateTooltip' })} placement="top">
              <Button
                className="primary_btn"
                style={{ marginLeft: '16px' }}
                onClick={() => handleCheck(addNodeForm)}
                loading={checkLoading}
              >
                {intl.formatMessage({ id: 'node.validate' })}
              </Button>
            </Tooltip>
            <Button
              className={isCheckPass && !submitPending ? 'cluster_primary_btn' : 'disable_btn'}
              htmlType="submit"
              style={{ marginLeft: '16px' }}
              onClick={() => handleOk(addNodeForm)}
              disabled={!isCheckPass || submitPending}
              loading={submitLoading}
            >
              {intl.formatMessage({ id: 'common.ok' })}
            </Button>
          </div>,
        ]}
        okText={intl.formatMessage({ id: 'common.ok' })}
        cancelText={intl.formatMessage({ id: 'common.cancel' })}
        destroyOnHidden={true}
      >
        <Form
          preserve={true}
          preserveState={true}
          style={{ marginTop: '35px' }}
          form={addNodeForm}
          onValuesChange={handleNodeValueChange}
          initialValues={{
            node: [
              {
                name: '',
                port: 22,
                userName: 'root',
              },
            ],
          }}
        >
          <Form.List name="node">
            {(fields, operation) => {
              let isRemoveAble = true;
              if (fields && fields.length === 1) {
                isRemoveAble = false;
              }
              const dataSources = fields.map((field) => ({
                field,
                operation,
                isRemoveAble,
              }));
              return (
                <ConfigProvider>
                  <Table
                    className="form_table"
                    rowKey={(row) => row.field.key}
                    dataSource={dataSources}
                    columns={addNodeColumns}
                    pagination={{
                      className: 'page',
                      style: { paddingBottom: 0 },
                      pageSizeOptions: [10, 20, 50],
                      showTotal: (total) => intl.formatMessage({ id: 'node.totalItems' }, { total }),
                      showSizeChanger: true,
                      showQuickJumper: true,
                    }}
                    scroll={{ x: 720 }}
                    summary={() => (
                      <Table.Summary>
                        <Table.Summary.Row className="summary_line">
                          <Table.Summary.Cell className="summary_cell" colSpan={7}>
                            <Button
                              className="summary_btn"
                              icon={<PlusCircleOutlined />}
                              style={{ width: '100%' }}
                              onClick={() => operation.add({ port: 22, userName: 'root' })}
                            >
                              {intl.formatMessage({ id: 'node.addNewNode' })}
                            </Button>
                          </Table.Summary.Cell>
                        </Table.Summary.Row>
                      </Table.Summary>
                    )}
                  />
                </ConfigProvider>
              );
            }}
          </Form.List>
        </Form>
      </Modal>
    </>
  );
}
