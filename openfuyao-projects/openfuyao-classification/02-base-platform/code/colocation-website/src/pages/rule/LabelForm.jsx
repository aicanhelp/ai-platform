/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Modal, Form, Space, Button, Input, message, Select } from 'antd';
import { PlusCircleOutlined, DeleteOutlined } from '@ant-design/icons';
import { useEffect, useState, useCallback } from 'openinula';
import { useIntl } from 'inula-intl';
import { getNamespaceList } from '../../api/colocationApi';
import '../../styles/colocationRuleManagementPage.less';
import { ResponseCode, ruleWorkloadTypeOptions } from '../../common/constants';

{
  /**
props:{
  dataList //注解列表?标签列表  annotation->注解  label->标签
  type: 类型 注解/标签
  open: 是否打开
  callbackOk: 回调成功
  callbackCancel 回调失败
}
*/
}

export default function LabelForm({ dataList = [], callbackOk, callbackCancel, isEdit }) {
  const intl = useIntl();
  const [labelEditForm] = Form.useForm();
  const [messageApi, contextHolder] = message.useMessage();
  const [namespaceOptions, setNamespaceOptions] = useState([]);

  const handleEditDoneOk = async () => {
    await labelEditForm.validateFields();
    callbackOk(labelEditForm.getFieldValue(`selectList`));
  };

  const handleEditCancel = () => {
    labelEditForm.setFieldValue(`selectList`, dataList); // 还原数据
    callbackCancel();
  };

  const handleEditReset = () => {
    labelEditForm.setFieldValue(`selectList`, dataList);
  };

  // 获取命名空间
  const getLabelNamespaceData = useCallback(async () => {
    let labelNamespaceList = [];
    try {
      const res = await getNamespaceList();
      if (res.status === ResponseCode.OK) {
        res.data.data.items.map((item) => {
          labelNamespaceList.push({ value: item.metadata.name, label: item.metadata.name });
        });
      }
    } catch (e) {
      setNamespaceOptions([]);
    }
    setNamespaceOptions([...labelNamespaceList]);
  }, []);

  const prompt = () => {
    messageApi.error(intl.formatMessage({ id: 'rule.maxTenItems' }));
  };

  useEffect(() => {
    getLabelNamespaceData();
  }, []);

  useEffect(() => {
    labelEditForm.setFieldValue(`selectList`, dataList); // 设置初始值
  }, [dataList]);

  return (
    <div>
      <div className={isEdit ? 'rules_exact_detail_content_box' : 'rules_exact_detail_content_onlyRead_box'}>
        {isEdit ? (
          <p className="rules_exact_detail_content_box_title">{intl.formatMessage({ id: 'rule.labelSelector' })}</p>
        ) : (
          <></>
        )}
        <div className="rules_exact_detail_content_box_form">
          {contextHolder}
          <Form form={labelEditForm} disabled={!isEdit}>
            <Form.List name={`selectList`}>
              {(fields, { add, remove }) => (
                <>
                  {fields.map(({ key, name, ...restField }, index) => (
                    <Space key={key} style={{ display: 'flex' }} align="baseline">
                      <div className="rules-form-col">
                        <Form.Item
                          {...restField}
                          name={[name, 'namespace']}
                          rules={[
                            {
                              required: true,
                              message: intl.formatMessage({ id: 'rule.namespaceRequired' }),
                            },
                          ]}
                          label={index === 0 ? intl.formatMessage({ id: 'table.column.namespace' }) : ''}
                        >
                          <Select
                            placeholder={intl.formatMessage({ id: 'common.select' })}
                            options={namespaceOptions}
                          />
                        </Form.Item>
                      </div>
                      <div className="rules-form-col">
                        <Form.Item
                          {...restField}
                          name={[name, 'workLoadTypes']}
                          rules={[
                            {
                              required: true,
                              message: intl.formatMessage({ id: 'rule.workloadTypeRequired' }),
                            },
                          ]}
                          label={index === 0 ? intl.formatMessage({ id: 'table.column.workloadName' }) : ''}
                        >
                          <Select
                            placeholder={intl.formatMessage({ id: 'common.select' })}
                            options={ruleWorkloadTypeOptions}
                          />
                        </Form.Item>
                      </div>
                      <div className="rules-form-col">
                        <Form.Item
                          {...restField}
                          name={[name, 'key']}
                          rules={[
                            {
                              required: true,
                              message: intl.formatMessage({ id: 'rule.keyRequired' }),
                            },
                          ]}
                          label={index === 0 ? intl.formatMessage({ id: 'rule.key' }) : ''}
                        >
                          <Input placeholder={intl.formatMessage({ id: 'common.input' })} />
                        </Form.Item>
                      </div>
                      <div className="rules-form-col">
                        <Form.Item
                          {...restField}
                          name={[name, 'value']}
                          rules={[
                            {
                              required: true,
                              message: intl.formatMessage({ id: 'rule.valueRequired' }),
                            },
                          ]}
                          label={index === 0 ? intl.formatMessage({ id: 'rule.value' }) : ''}
                        >
                          <Input placeholder="value1,value2" />
                        </Form.Item>
                      </div>
                      <div className="rules-form-col">
                        <Form.Item {...restField} name={[name, 'del']} label={index === 0 ? <></> : ''}>
                          {isEdit && fields.length > 1 ? (
                            <DeleteOutlined className="label_icon_color" onClick={() => remove(name)} />
                          ) : (
                            <></>
                          )}
                        </Form.Item>
                      </div>
                    </Space>
                  ))}
                  {fields.length < 10 ? (
                    <Form.Item>
                      {isEdit ? (
                        <Button
                          className="label_icon_btn"
                          type="link"
                          onClick={() => add()}
                          block
                          icon={<PlusCircleOutlined className="primary_icon common_antd_icon" />}
                        >
                          {intl.formatMessage({ id: 'rule.addLabel' })}
                        </Button>
                      ) : (
                        <></>
                      )}
                    </Form.Item>
                  ) : (
                    <Form.Item>
                      {isEdit ? (
                        <Button
                          className="label_icon_btn"
                          type="link"
                          onClick={() => prompt()}
                          block
                          icon={<PlusCircleOutlined className="primary_icon common_antd_icon" />}
                        >
                          {intl.formatMessage({ id: 'rule.addLabel' })}
                        </Button>
                      ) : (
                        <></>
                      )}
                    </Form.Item>
                  )}
                </>
              )}
            </Form.List>
          </Form>
        </div>
      </div>
      {isEdit ? (
        <div className="child_content_label_footer">
          <div className="label_btn_footer_text"></div>
          <div className="label_btn_footer_button">
            <Button className="cancel_btn" onClick={handleEditCancel}>
              {intl.formatMessage({ id: 'common.cancel' })}
            </Button>
            <Button className="cancel_btn" onClick={handleEditReset}>
              {intl.formatMessage({ id: 'common.reset' })}
            </Button>
            <Button className="primary_btn" onClick={handleEditDoneOk}>
              {intl.formatMessage({ id: 'common.confirm' })}
            </Button>
          </div>
        </div>
      ) : (
        <></>
      )}
    </div>
  );
}
