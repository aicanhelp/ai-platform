/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { useEffect, useState, useCallback, useMemo } from 'openinula';
import { useHistory } from 'inula-router';
import { Form, Input, Button, Table, Tooltip, message, Select, InputNumber } from 'antd';
import { DeleteOutlined, PlusCircleOutlined, QuestionCircleOutlined } from '@ant-design/icons';
import { createCluster, getInstallerConfig, listcluster, validateNodes, listVersions } from '@/api/clusterApi';
import { ResponseCode, ADDONLIST, DEFAULT_REPO } from '@/common/constants';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import CreateClusterLog from '@/pages/clusterManage/CreateClusterLog';
import { requestResource, extractIPv4 } from '@/tools/utils';
import useButtonLoading from '@/hooks/useButtonLoading';
import '@/styles/pages/createCluster.less';
import { useIntl } from '../../i18n';

export default function CreateClusterPage() {
  const history = useHistory();
  const intl = useIntl();
  const [checkLoading, setCheckLoading] = useState(false);
  const [createLoading, setCreateLoading, createPending] = useButtonLoading();

  const [addClusterForm] = Form.useForm();
  const masterWatchList = Form.useWatch('master', addClusterForm);
  const deploymentIP = Form.useWatch('deployment_ip_url', addClusterForm);
  const nodeWatchList = Form.useWatch('node', addClusterForm);
  const [logOpen, setLogOpen] = useState(false);
  const [isCheckPass, setIsCheckPass] = useState(false);
  const [errorIP, setErrorIP] = useState('');

  const [imageRepoOptions, setImageRepoOptions] = useState([]);
  const [imageRepoIpOptions, setImageRepoIpOptions] = useState([]);
  const [chartRepoOptions, setChartRepoOptions] = useState([]);

  const [messageApi, contextHolder] = message.useMessage();

  const handleLogExit = () => {
    setLogOpen(false);
    history.push('/cluster');
  };

  const modifyStatus = (value, index) => {
    if (masterWatchList && errorIP) {
      if (value.includes('master')) {
        if (masterWatchList[index] && masterWatchList[index].ip === errorIP) {
          return 'error';
        }
      } else {
        if (nodeWatchList[index] && nodeWatchList[index].ip === errorIP) {
          return 'error';
        }
      }
    }
    return '';
  };

  const modifyDeploymentStatus = () => {
    if (deploymentIP && errorIP) {
      if (deploymentIP === errorIP) {
        return 'error';
      }
    }
    return '';
  };

  const masterColumns = (omitted = false) => {
    return [
      {
        title: intl.formatMessage({ id: 'node.name' }),
        key: 'name',
        render: (value, { field }) => {
          return (
            <Form.Item
              name={[field.name, 'name']}
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
        key: 'ip',
        render: (value, { field }) => {
          return (
            <Form.Item
              name={[field.name, 'ip']}
              validateStatus={modifyStatus(value.reallyKey, field.name)}
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
        key: 'password',
        render: (value, { field }) => {
          return (
            <Form.Item
              name={[field.name, 'password']}
              rules={[{ required: true, message: intl.formatMessage({ id: 'node.passwordRequired' }) }]}
            >
              <Input.Password
                placeholder={intl.formatMessage({ id: 'node.passwordPlaceholder' })}
                autoComplete="new-password"
              />
            </Form.Item>
          );
        },
      },
      {
        title: intl.formatMessage({ id: 'node.action' }),
        key: 'handle',
        render: (value, { operation, field, isRemoveAble }) => (
          <Form.Item>
            <DeleteOutlined
              className="common_antd_icon primary_color"
              onClick={() => {
                if (!omitted) {
                  if (isRemoveAble) {
                    operation.remove(field.name);
                  } else {
                    messageApi.error(intl.formatMessage({ id: 'node.atLeastOne' }));
                  }
                } else {
                  operation.remove(field.name);
                }
              }}
            />
          </Form.Item>
        ),
      },
    ];
  };

  const handleCreateCluster = async () => {
    let createClusterData = {};
    const formObj = addClusterForm.getFieldsValue();
    createClusterData.cluster = {
      name: formObj?.cluster_name,
      openFuyaoVersion: formObj?.openFuyaoVersion,
      imageRepo: {
        url: formObj?.image_repo_url,
        ip: formObj?.image_repo_ip_url,
      },
    };
    createClusterData.controlPlaneEndpoint = formObj?.deployment_ip_url;
    const allNodeList = [
      ...formObj.master.map((item) => ({
        hostname: item.name,
        ip: item.ip,
        port: String(item.port),
        username: item.userName,
        password: item.password,
        role: ['master', 'etcd'],
      })),
      ...formObj.node.map((item) => ({
        hostname: item.name,
        ip: item.ip,
        port: String(item.port),
        username: item.userName,
        password: item.password,
        role: ['node'],
      })),
    ];
    createClusterData.nodes = allNodeList;
    createClusterData.addons = [];
    if (formObj?.addonChoice === ADDONLIST.OPENFUYAOSYSTEMCONTROLLER) {
      createClusterData.addons.push({
        name: ADDONLIST.OPENFUYAOSYSTEMCONTROLLER,
        params: {
          helmRepo: formObj?.chart_url,
        },
      });
    }
    if (createPending) return;
    setCreateLoading(true);
    try {
      const [error, res] = await requestResource(createCluster(createClusterData));
      if (error) {
        messageApi.error(
          intl.formatMessage({ id: 'cluster.create.createFailed' }, { error: error.response.data.message }),
        );
      }
      if (res) {
        if (res.data.code === ResponseCode.OK) {
          setLogOpen(true);
        }
      }
    } catch (error) {
      // requestResource 已内聚错误分支，此处兜底非请求异常，避免 unhandled rejection
    } finally {
      setCreateLoading(false);
    }
  };

  const getClusterConfig = useCallback(async () => {
    const [error, res] = await requestResource(getInstallerConfig());
    if (res) {
      if (res.data.code === ResponseCode.OK) {
        setImageRepoOptions([
          {
            label: res.data.data.imageRepo.domain,
            value: res.data.data.imageRepo.domain,
          },
        ]);
        if (res.data.data.imageRepo.domain === DEFAULT_REPO.imageRepo) {
          // 在线
          setImageRepoIpOptions([
            {
              label: res.data.data.imageRepo.ip,
              value: res.data.data.imageRepo.ip,
            },
          ]);
          setChartRepoOptions([
            {
              label: DEFAULT_REPO.helmRepo,
              value: DEFAULT_REPO.helmRepo,
            },
          ]);
        } else {
          // 离线
          setImageRepoIpOptions([
            {
              label: res.data.data.ip,
              value: res.data.data.ip,
            },
          ]);
          setChartRepoOptions([
            {
              label: `http://${res.data.data.ip}:38080`,
              value: `http://${res.data.data.ip}:38080`,
            },
          ]);
        }
      }
    }
  }, []);

  const handleCheck = async () => {
    setIsCheckPass(false);
    // 验证表单
    const formObj = addClusterForm.getFieldsValue();
    await addClusterForm.validateFields();
    // 验证集群名称是否已存在
    try {
      const res = await listcluster();
      if (res.data.code === ResponseCode.OK) {
        const clusterNameList = res.data.data.items?.map((item) => item.name) || [];
        if (clusterNameList.find((item) => item === formObj?.cluster_name)) {
          messageApi.error(intl.formatMessage({ id: 'cluster.create.clusterNameExists' }));
          return;
        }
      }
    } catch (e) {
      messageApi.error(
        intl.formatMessage({ id: 'cluster.create.getNameListFailed' }, { error: e.response.data.message }),
      );
      return;
    }
    // 验证Master节点数量是否为奇数
    if (!(formObj.master.length % 2)) {
      messageApi.error(intl.formatMessage({ id: 'cluster.create.masterEvenCount' }));
      return;
    }
    // 验证节点名称、IP是否重复
    const allNodeList = [
      ...formObj.master.map((item) => ({
        hostname: item.name,
        ip: item.ip,
        port: String(item.port),
        username: item.userName,
        password: item.password,
      })),
      ...formObj.node.map((item) => ({
        hostname: item.name,
        ip: item.ip,
        port: String(item.port),
        username: item.userName,
        password: item.password,
      })),
    ];
    let nameSet = new Set();
    let ipSet = new Set();
    for (let node of allNodeList) {
      if (nameSet.has(node.hostname)) {
        messageApi.error(intl.formatMessage({ id: 'cluster.create.duplicateNodeName' }, { name: node.hostname }));
        setCheckLoading(false);
        return;
      }
      if (ipSet.has(node.ip)) {
        messageApi.error(intl.formatMessage({ id: 'cluster.create.duplicateNodeIp' }, { ip: node.ip }));
        setCheckLoading(false);
        return;
      }
      nameSet.add(node.hostname);
      ipSet.add(node.ip);
    }

    setCheckLoading(true);
    try {
      const jsonData = {
        namespace: formObj?.cluster_name,
        nodes: allNodeList,
        balanceIp: formObj?.deployment_ip_url || '',
      };
      const res = await validateNodes(jsonData);
      if (res.data.code === ResponseCode.OK) {
        setIsCheckPass(true);
        messageApi.success(intl.formatMessage({ id: 'node.validateSuccess' }));
        setErrorIP('');
      }
    } catch (e) {
      messageApi.error(intl.formatMessage({ id: 'node.validateFailed' }, { error: e.response.data.message }));
      const [ip, ...resets] = extractIPv4(e.response.data.message);
      setErrorIP(ip);
    }
    setCheckLoading(false);
  };

  useEffect(() => {
    getClusterConfig();
  }, [getClusterConfig]);

  const handleChangeValue = (changedValues, _allValues) => {
    if (changedValues.deployment_ip_url) {
      setIsCheckPass(false);
      return;
    }
    if (changedValues.image_repo_url || changedValues.image_repo_ip_url || changedValues.chart_url) {
      return;
    }
    // 判断是否改变或者删除
    if (changedValues.master) {
      const deleteList = changedValues.master.some((item) => item && Object.keys(item).length > 1); // 删除
      if (deleteList) {
        return;
      }
    }
    if (changedValues.node) {
      const deleteList = changedValues.node.some((item) => item && Object.keys(item).length > 1); // 删除
      if (deleteList) {
        return;
      }
    }
    setIsCheckPass(false);
  };

  const [patchList, setPatchList] = useState([]);
  useEffect(() => {
    listVersions().then((res) => {
      if (res.status === ResponseCode.OK) {
        const {
          data: {
            data: { versions },
          },
        } = res;
        if (versions) {
          setPatchList(versions);
        }
      }
    });
  }, []);
  const patchOptions = useMemo(() => patchList.map((item) => ({ label: item, value: item })), [patchList]);

  return (
    <div className="child_content">
      <BreadCrumbCom
        items={[
          { title: intl.formatMessage({ id: 'cluster.breadcrumb' }), path: `/cluster` },
          { title: intl.formatMessage({ id: 'cluster.create.title' }), path: `/addCluster` },
        ]}
      />
      {contextHolder}
      <Form
        initialValues={{
          master: [{ ip: '', name: '', port: 22, userName: 'root', password: '' }],
        }}
        onValuesChange={handleChangeValue}
        preserve={true}
        preserveState={true}
        autoComplete="off"
        onFinish={handleCreateCluster}
        form={addClusterForm}
      >
        <div className="container_margin_box container_padding_margin_box">
          <div className="form_each_title require_title"> {intl.formatMessage({ id: 'cluster.create.basicInfo' })}</div>
          <div className="col_flex">
            <Form.Item
              layout="vertical"
              label={intl.formatMessage({ id: 'cluster.create.clusterName' })}
              name="cluster_name"
              required={false}
              rules={[
                { required: true, message: intl.formatMessage({ id: 'cluster.create.clusterNameValidation' }) },
                {
                  pattern: /^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$/,
                  message: intl.formatMessage({ id: 'cluster.create.clusterNameValidation' }),
                },
              ]}
            >
              <Input showCount maxLength={63} autoComplete="off" />
            </Form.Item>
            <Form.Item
              layout="vertical"
              label={intl.formatMessage({ id: 'cluster.create.openFuyaoVersion' })}
              name="openFuyaoVersion"
              required={false}
              rules={[{ required: true, message: intl.formatMessage({ id: 'cluster.create.selectVersion' }) }]}
            >
              <Select placeholder={intl.formatMessage({ id: 'cluster.create.selectVersion' })} options={patchOptions} />
            </Form.Item>
          </div>
        </div>
        <div className="container_margin_box container_padding_margin_box">
          <div className="form_each_title require_title">
            {' '}
            {intl.formatMessage({ id: 'cluster.create.masterInfo' })}
          </div>
          <Form.List name="master">
            {(fields, operation) => {
              let isRemoveAble = true;
              if (fields && fields.length === 1) {
                isRemoveAble = false;
              }
              const dataSources = fields.map((field) => ({
                reallyKey: `master_${field.key}`,
                field,
                operation,
                isRemoveAble,
              }));
              return (
                <Table
                  className="form_table"
                  rowKey={(row) => row.reallyKey}
                  dataSource={dataSources}
                  columns={masterColumns()}
                  pagination={false}
                  summary={() => (
                    <Table.Summary>
                      <Table.Summary.Row className="summary_line">
                        <Table.Summary.Cell index={0} className="summary_cell" colSpan={6}>
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
              );
            }}
          </Form.List>
          <Form.Item
            noStyle
            shouldUpdate={(prevValues, curValues) => prevValues.master.length !== curValues.master.length}
          >
            {({ getFieldValue }) => {
              let fieldMasterList = getFieldValue('master');
              if (fieldMasterList.length > 1) {
                if (fieldMasterList.length % 2) {
                  return (
                    <Form.Item
                      layout="vertical"
                      label={intl.formatMessage({ id: 'cluster.create.lbIpAddress' })}
                      rules={[
                        {
                          pattern:
                            /^(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])$/,
                          message: intl.formatMessage({ id: 'node.ipv4Only' }),
                        },
                      ]}
                      validateStatus={modifyDeploymentStatus()}
                      name="deployment_ip_url"
                    >
                      <Input
                        autoComplete="off"
                        placeholder={intl.formatMessage({ id: 'cluster.create.inputPlaceholder' })}
                        style={{ width: '30%' }}
                      />
                    </Form.Item>
                  );
                }
                return (
                  <Form.Item
                    layout="vertical"
                    label={intl.formatMessage({ id: 'cluster.create.lbIpAddress' })}
                    name="deployment_ip_url_disabled"
                  >
                    <Input
                      autoComplete="off"
                      placeholder={intl.formatMessage({ id: 'cluster.create.lbIpDisabledPlaceholder' })}
                      disabled={true}
                      style={{ width: '30%' }}
                    />
                  </Form.Item>
                );
              }
            }}
          </Form.Item>
        </div>
        <div className="container_margin_box container_padding_margin_box padding_bottom_no_box">
          <div className="form_each_title"> {intl.formatMessage({ id: 'cluster.create.nodeInfo' })}</div>
          <Form.List name="node" initialValue={[]}>
            {(fields, operation) => {
              const dataSources = fields.map((field) => ({
                reallyKey: `node_${field.key}`,
                field,
                operation,
              }));
              return (
                <Table
                  className="form_table node_form"
                  rowKey={(row) => row.reallyKey}
                  dataSource={dataSources}
                  columns={masterColumns(true)}
                  locale={{ emptyText: null }}
                  pagination={false}
                  summary={() => (
                    <Table.Summary>
                      <Table.Summary.Row className="summary_line">
                        <Table.Summary.Cell index={0} className="summary_cell" colSpan={6}>
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
              );
            }}
          </Form.List>
        </div>
        <div className="container_margin_box container_padding_margin_box special_suffix">
          <div className="form_each_title require_title">
            {' '}
            {intl.formatMessage({ id: 'cluster.create.advancedConfig' })}
          </div>
          <div className="col_flex">
            <Form.Item
              layout="vertical"
              label={intl.formatMessage({ id: 'cluster.create.registryAddress' })}
              name="image_repo_url"
              required={false}
              rules={[{ required: true, message: intl.formatMessage({ id: 'cluster.create.selectRegistryAddress' }) }]}
            >
              <Select
                options={imageRepoOptions}
                placeholder={intl.formatMessage({ id: 'cluster.create.selectPlaceholder' })}
              />
            </Form.Item>
            <Form.Item
              label={intl.formatMessage({ id: 'cluster.create.registryIp' })}
              layout="vertical"
              name="image_repo_ip_url"
              required={false}
              rules={[{ required: true, message: intl.formatMessage({ id: 'cluster.create.selectRegistryIp' }) }]}
            >
              <Select
                options={imageRepoIpOptions}
                placeholder={intl.formatMessage({ id: 'cluster.create.selectPlaceholder' })}
              />
            </Form.Item>
          </div>
          <div className="col_flex" style={{ marginTop: '24px' }}>
            <Form.Item
              layout="vertical"
              label={intl.formatMessage({ id: 'cluster.create.addons' })}
              name="addonChoice"
              required={false}
              className="addonchoiceInput"
              rules={[{ required: true, message: intl.formatMessage({ id: 'cluster.create.selectAddon' }) }]}
              initialValue={ADDONLIST.OPENFUYAOSYSTEMCONTROLLER}
            >
              <Select
                allowClear
                options={[
                  {
                    label: intl.formatMessage({ id: 'cluster.create.none' }),
                    value: ADDONLIST.NONE,
                  },
                  {
                    label: 'openFuyao Cores',
                    value: ADDONLIST.OPENFUYAOSYSTEMCONTROLLER,
                  },
                ]}
                placeholder={intl.formatMessage({ id: 'cluster.create.selectPlaceholder' })}
              />
            </Form.Item>
            <Form.Item noStyle shouldUpdate={(prev, current) => prev.addonChoice !== current.addonChoice}>
              {({ getFieldValue }) => {
                const addon = getFieldValue('addonChoice') || [];
                if (addon.includes(ADDONLIST.OPENFUYAOSYSTEMCONTROLLER)) {
                  return (
                    <Form.Item
                      layout="vertical"
                      label={intl.formatMessage({ id: 'cluster.create.chartRepoAddress' })}
                      name="chart_url"
                      required={false}
                      rules={[
                        {
                          required: true,
                          message: intl.formatMessage({ id: 'cluster.create.selectChartRepoAddress' }),
                        },
                      ]}
                    >
                      <Select
                        options={chartRepoOptions}
                        placeholder={intl.formatMessage({ id: 'cluster.create.selectPlaceholder' })}
                      />
                    </Form.Item>
                  );
                }
                return null;
              }}
            </Form.Item>
          </div>
        </div>
        <div className="stricy_bottom">
          <Button className="cancel_btn" onClick={() => history.go(-1)}>
            {intl.formatMessage({ id: 'common.cancel' })}
          </Button>
          <Tooltip title={intl.formatMessage({ id: 'node.validateTooltip' })} placement="top">
            <Button className="primary_btn" onClick={handleCheck} loading={checkLoading}>
              {intl.formatMessage({ id: 'node.validate' })}
            </Button>
          </Tooltip>
          <Button
            className={isCheckPass && !createPending ? 'primary_btn cluster_primary_btn' : 'disable_btn'}
            htmlType="submit"
            disabled={!isCheckPass || createPending}
            loading={createLoading}
          >
            {intl.formatMessage({ id: 'cluster.create.submit' })}
          </Button>
        </div>
      </Form>
      {logOpen && (
        <CreateClusterLog
          clusterName={addClusterForm.getFieldValue('cluster_name')}
          open={logOpen}
          okFn={handleLogExit}
          cancelFn={handleLogExit}
        />
      )}
    </div>
  );
}
