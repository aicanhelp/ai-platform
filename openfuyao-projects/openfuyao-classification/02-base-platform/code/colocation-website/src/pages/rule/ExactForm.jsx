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

export default function ExactForm({ dataList = [], callbackOk, callbackCancel, isEdit }) {
  const intl = useIntl();
  const [exactEditForm] = Form.useForm();
  const [messageApi, contextHolder] = message.useMessage();
  const [exactNamespaceOptions, setExactNamespaceOptions] = useState([]);

  const handleExactEditDoneOk = async () => {
    await exactEditForm.validateFields();
    callbackOk(exactEditForm.getFieldValue(`exactSelectList`));
  };

  const handleExactEditCancel = () => {
    exactEditForm.setFieldValue(`exactSelectList`, dataList); // 还原数据
    callbackCancel();
  };

  const handleExactEditReset = () => {
    exactEditForm.setFieldValue(`exactSelectList`, dataList); // 还原数据
  };

  // 获取命名空间
  const getNamespaceData = useCallback(async () => {
    let namespaceList = [];
    try {
      const res = await getNamespaceList();
      if (res.status === ResponseCode.OK) {
        res.data.data.items.map((item) => {
          namespaceList.push({ value: item.metadata.name, label: item.metadata.name });
        });
      }
    } catch (e) {
      setExactNamespaceOptions([]);
    }
    setExactNamespaceOptions([...namespaceList]);
  }, []);

  const prompt = () => {
    messageApi.error(intl.formatMessage({ id: 'rule.maxTenItems' }));
  };

  useEffect(() => {
    getNamespaceData();
  }, [getNamespaceData]);

  useEffect(() => {
    exactEditForm.setFieldValue(`exactSelectList`, dataList); // 设置初始值
  }, [dataList]);

  return (
    <div>
      <div className={isEdit ? 'rules_exact_detail_content_box' : 'rules_exact_detail_content_onlyRead_box'}>
        {isEdit ? (
          <p className="rules_exact_detail_content_box_title">{intl.formatMessage({ id: 'rule.exactMatch' })}</p>
        ) : (
          <></>
        )}
        <div className="rules_exact_detail_content_box_form">
          {contextHolder}
          <Form form={exactEditForm} disabled={!isEdit} style={{ width: '100%' }}>
            <Form.List name={`exactSelectList`}>
              {(fields, { add, remove }) => (
                <>
                  {fields.map(({ key, name, ...restField }, index) => (
                    <Space
                      key={key}
                      style={{
                        display: 'flex',
                      }}
                      align="baseline"
                    >
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
                            options={exactNamespaceOptions}
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
                          name={[name, 'name']}
                          rules={[
                            {
                              required: true,
                              message: intl.formatMessage({ id: 'rule.workloadNameRequired' }),
                            },
                          ]}
                          label={index === 0 ? intl.formatMessage({ id: 'table.column.workloadName' }) : ''}
                        >
                          <Input placeholder={intl.formatMessage({ id: 'common.input' })} />
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
            <Button className="cancel_btn" onClick={handleExactEditCancel}>
              {intl.formatMessage({ id: 'common.cancel' })}
            </Button>
            <Button className="cancel_btn" onClick={handleExactEditReset}>
              {intl.formatMessage({ id: 'common.reset' })}
            </Button>
            <Button className="primary_btn" onClick={handleExactEditDoneOk}>
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
