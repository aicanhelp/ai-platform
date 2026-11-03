/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { containerRouterPrefix } from '../common/constants';
import { useEffect, useState, createRef } from 'openinula';
import { QuestionCircleOutlined, PlusCircleOutlined, DeleteOutlined } from '@ant-design/icons';
import { Form, Space, Input, Button, message, Spin, Flex, Modal } from 'antd';
import BreadCrumbCom from '../components/BreadCrumbCom';
import { useLocation, useHistory } from 'inula-router';
import CodeMirrorEditor from '@/components/CodeMirrorEditor';
import { yamlTojson } from '@/utils/common';
import '@/styles/cluster.less';
import { creatCluster } from '@/api/multiclusterApi';
import '@/styles/dark.less';
import { useIntl } from 'inula-intl';
export default function EditCluster() {
  const intl = useIntl();
  const location = useLocation();
  const history = useHistory();
  const childCodeMirrorRef = createRef(null);
  const [clusetForm] = Form.useForm();
  const { type } = location?.state || {};
  const [messageApi, contextHolder] = message.useMessage();
  const { detailInfo } = location?.state || {};
  const [config, SetConfig] = useState();
  const [loading, setLoading] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const layout = {
    labelCol: { span: 8 },
    wrapperCol: { span: 16 },
  };
  const showModal = () => {
    setIsModalOpen(true);
  };
  const handleCancel = () => {
    setIsModalOpen(false);
  };
  const onFinish = async () => {
    await clusetForm.validateFields();
    let arr = {
      name: clusetForm.getFieldsValue().cluster_name,
      kubeconfig: config ? yamlTojson(config) : '',
      clusterlabels: {},
    };
    if (
      clusetForm.getFieldsValue().labelList.length > 0 &&
      clusetForm.getFieldsValue().labelList[0].key &&
      clusetForm.getFieldsValue().labelList[0].value
    ) {
      clusetForm.getFieldsValue().labelList.map((item) => {
        arr.clusterlabels[item.key] = item.value;
      });
    }
    setLoading(true);
    if (!arr.name || !config) {
      setLoading(false);
      messageApi.error(intl.formatMessage({ id: 'multicluster.fillRequiredInfo' }));
    } else {
      try {
        await creatCluster(arr);
        setLoading(false);
        messageApi.success(
          intl.formatMessage(
            { id: 'multicluster.manageClusterSuccess' },
            { name: clusetForm.getFieldsValue().cluster_name },
          ),
        );
        setTimeout(() => {
          history.push(`/${containerRouterPrefix}/`);
          clusetForm.resetFields();
        }, 1000);
      } catch (error) {
        setLoading(false);
        messageApi.error(
          intl.formatMessage(
            { id: 'multicluster.manageClusterFailed' },
            { name: clusetForm.getFieldsValue().cluster_name },
          ),
        );
      }
    }
  };
  const oncancelEdit = () => {
    clusetForm.resetFields();
    history.push(`/${containerRouterPrefix}/`);
  };
  const handleChangeYaml = (str) => {
    SetConfig(str);
  };

  useEffect(() => {
    clusetForm.setFieldsValue({
      labelList: detailInfo?.labelList,
    });
  }, [detailInfo?.labelList]);

  return (
    <div className={`child_content multi_cluster_edit ${localStorage.getItem('theme') === 'dark' ? 'dark_box' : ''}`}>
      {contextHolder}
      <Flex gap="middle" vertical className="flex_content">
        <Spin tip={intl.formatMessage({ id: 'multicluster.deploying' })} spinning={loading}>
          <BreadCrumbCom
            items={[
              {
                title: intl.formatMessage({ id: 'multicluster.clusterManagement' }),
                path: `/${containerRouterPrefix}/`,
              },
              { title: type, path: '/' },
            ]}
          />
          <div className="cluster_edit_info">
            <p className="title" style={{ color: localStorage.getItem('theme') === 'dark' ? '#fff' : '#333' }}>
              {intl.formatMessage({ id: 'multicluster.clusterInfo' })}
            </p>
            <Form {...layout} name="clusetForm" form={clusetForm} layout="vertical">
              <Form.Item
                name="cluster_name"
                label={intl.formatMessage({ id: 'multicluster.clusterNameLabel' })}
                rules={[
                  { required: true, message: intl.formatMessage({ id: 'multicluster.enterClusterName' }) },
                  {
                    min: 1,
                    message: intl.formatMessage({ id: 'multicluster.clusterNameValidation' }),
                  },
                  {
                    max: 63,
                    message: intl.formatMessage({ id: 'multicluster.clusterNameValidation' }),
                  },
                  {
                    pattern: /^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$/,
                    message: intl.formatMessage({ id: 'multicluster.clusterNameValidation' }),
                  },
                ]}
              >
                <Input
                  placeholder={intl.formatMessage({ id: 'common.pleaseInput' })}
                  style={{ width: '620px' }}
                  autoComplete="off"
                />
              </Form.Item>
              <p>{intl.formatMessage({ id: 'multicluster.clusterLabels' })}</p>
              <div style={{ display: 'flex', margin: '8px 0' }}>
                <p className="form_input" style={{ width: '500px', marginRight: '15px' }}>
                  {intl.formatMessage({ id: 'common.key' })}
                </p>
                <p>{intl.formatMessage({ id: 'common.value' })}</p>
              </div>
              <Form.List name={`labelList`} title="标签">
                {(fields, { add, remove }, { errors }) => (
                  <>
                    {fields.map(({ key, name, ...restField }) => (
                      <Space
                        key={key}
                        style={{
                          display: 'flex',
                        }}
                        align="baseline"
                      >
                        <Form.Item {...restField} name={[name, 'key']}>
                          <Input
                            placeholder={intl.formatMessage({ id: 'common.pleaseInput' })}
                            className="form_input"
                            style={{ width: '500px', marginRight: '8px' }}
                          />
                        </Form.Item>
                        <Form.Item {...restField} name={[name, 'value']}>
                          <Input
                            placeholder={intl.formatMessage({ id: 'common.pleaseInput' })}
                            className="form_input"
                            style={{ width: '500px' }}
                          />
                        </Form.Item>
                        <DeleteOutlined className="common_antd_icon" onClick={() => remove(name)} />
                      </Space>
                    ))}
                    <Form.Item>
                      <Button
                        className="icon_btn add_label"
                        type="link"
                        onClick={() => add()}
                        block
                        icon={<PlusCircleOutlined className="primary_icon common_antd_icon" />}
                      >
                        {intl.formatMessage({ id: 'multicluster.addLabel' })}
                      </Button>
                      <Form.ErrorList errors={errors} />
                    </Form.Item>
                  </>
                )}
              </Form.List>
            </Form>
          </div>
          <div className="cluster_config">
            <div className="cluster_config_top">
              <p className="title">
                <span style={{ color: '#e94547', marginRight: '4px', fontSize: '12px' }}>*</span>
                {intl.formatMessage({ id: 'multicluster.kubeconfigConfig' })}
              </p>
              <p className="get_config" onClick={showModal}>
                <QuestionCircleOutlined style={{ width: '16px', height: '16px', color: '#346ac4' }} />
                {intl.formatMessage({ id: 'multicluster.howToGetKubeconfig' })}
              </p>
            </div>
            <CodeMirrorEditor yamlData={''} changeYaml={handleChangeYaml} ref={childCodeMirrorRef} />
          </div>
          <Modal
            className="config_modal"
            getContainer={false}
            title={intl.formatMessage({ id: 'multicluster.getKubeconfig' })}
            open={isModalOpen}
            onCancel={handleCancel}
            footer={[
              <Button onClick={handleCancel} className="cancel_btn">
                {intl.formatMessage({ id: 'common.cancel' })}
              </Button>,
              <Button className="primary_btn" onClick={handleCancel}>
                {intl.formatMessage({ id: 'common.ok' })}
              </Button>,
            ]}
          >
            <p className="config_modal_text">{intl.formatMessage({ id: 'multicluster.kubeconfigStep1' })}</p>
            <p className="config_modal_text">{intl.formatMessage({ id: 'multicluster.kubeconfigStep2' })}</p>
            <div className="config_modal_code">
              <p>kubectl config view --raw</p>
            </div>
            <p className="config_modal_text">{intl.formatMessage({ id: 'multicluster.kubeconfigStep3' })}</p>
          </Modal>
        </Spin>
      </Flex>
      <div className="btns">
        <Button className="cancel_btn" onClick={oncancelEdit}>
          {intl.formatMessage({ id: 'common.cancel' })}
        </Button>
        <Button className="primary_btn" onClick={onFinish}>
          {intl.formatMessage({ id: 'common.ok' })}
        </Button>
      </div>
    </div>
  );
}
