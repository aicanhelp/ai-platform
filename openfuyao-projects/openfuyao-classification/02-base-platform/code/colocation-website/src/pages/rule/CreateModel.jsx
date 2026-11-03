/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { containerRouterPrefix, ResponseCode } from '../../common/constants';
import { useParams, Link, useHistory } from 'inula-router';
import { useEffect, useState, useCallback } from 'openinula';
import { useIntl } from 'inula-intl';
import { Form, message, Select, Input } from 'antd';
import { createColocationRule } from '../../api/colocationApi';
import '../../styles/colocationRuleManagementPage.less';
import LabelForm from './LabelForm';
import ExactForm from './ExactForm';
import Dayjs from 'dayjs';
import { filterStatusName } from '../../utils/common';
import ToolTipComponent from '@/components/ToolTipComponent';
import { CloseOutlined } from '@ant-design/icons';

export default function CreateModel({ tabKeyType, stateType }) {
  const intl = useIntl();
  const history = useHistory();
  const [RuleForm] = Form.useForm();
  const [messageApi, contextHolder] = message.useMessage();
  const [selectType, setSelectType] = useState(stateType);
  const [isShow, setIsShow] = useState(true);
  const handleRuleslabelCreateShowChange = () => {
    setIsShow(false);
  };
  // 标签成功回调
  const handleOk = async (data) => {
    const optData = JSON.parse(JSON.stringify(data));
    const values = await RuleForm.validateFields();
    let arr = [];
    if (values) {
      if (tabKeyType === '1') {
        arr = optData.map((item) => {
          item.workLoadTypes = item.workLoadTypes.split(' ');
          const tempKey = item.key.trim().replace(/\s/g, '');
          const tempValue =
            item.value.length > 1
              ? item.value.trim().replace(/\s/g, '').split(',')
              : item.value.trim().replace(/\s/g, '');
          delete item.key;
          delete item.value;
          return {
            ...item,
            name: '',
            selector: { matchExpressions: [{ key: tempKey, operator: 'In', values: [...tempValue] }] },
          };
        });
      } else {
        arr = optData.map((item) => {
          item.workLoadTypes = item.workLoadTypes.split(' ');
          item.name = item.name.trim().replace(/\s/g, '');
          return {
            ...item,
            selector: null,
          };
        });
      }
      const obj = {
        ruleName: RuleForm.getFieldValue('colocationRuleName').trim().replace(/\s/g, ''),
        createTimestamp: Dayjs().format('YYYY-MM-DD HH:mm'),
        colocationType: selectType,
        selectors: [...arr],
      };
      try {
        const res = await createColocationRule(obj);
        if (res.status === ResponseCode.OK) {
          messageApi.success(intl.formatMessage({ id: 'rule.createSuccess' }));
          setTimeout(() => {
            history.go(-1);
          }, 2000);
        } else {
          messageApi.error(intl.formatMessage({ id: 'rule.createFailed' }));
        }
      } catch (e) {
        messageApi.error(`${intl.formatMessage({ id: 'message.apiError' })}${e.response.data.message}`);
      }
    }
  };

  // 标签失败回调
  const handleCancel = () => {
    history.go(-1);
  };

  return (
    <>
      {contextHolder}
      <div className="rules_form_detail_content">
        <div className={`${isShow ? '' : 'cant_show'}`}>
          <ToolTipComponent>
            <div
              className="daemonsetDetail"
              style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
            >
              <div className="promot-box">{intl.formatMessage({ id: 'rule.tipMessage' })}</div>
              <CloseOutlined onClick={handleRuleslabelCreateShowChange} />
            </div>
          </ToolTipComponent>
        </div>
        <div className={`${isShow ? 'normal_rules_overflow_height' : 'rules_overflow_height'}`}>
          <div className="rules_exact_detail_content_top">
            <div>
              <Form form={RuleForm}>
                <div className="rules_exact_detail_content_top_flex">
                  <Form.Item
                    name="colocationRuleName"
                    label={intl.formatMessage({ id: 'rule.name' })}
                    rules={[
                      {
                        required: true,
                        message: intl.formatMessage({ id: 'rule.nameRequired' }),
                      },
                    ]}
                  >
                    <Input
                      placeholder={intl.formatMessage({ id: 'rule.namePlaceholder' })}
                      className="rules_exact_detail_content_top_select"
                    />
                  </Form.Item>
                  <Form.Item name="colocationRuleType" label={intl.formatMessage({ id: 'rule.colocationType' })}>
                    <Input
                      placeholder={filterStatusName(selectType, intl)}
                      className="rules_exact_detail_content_top_select"
                      disabled
                    />
                  </Form.Item>
                </div>
              </Form>
            </div>
          </div>

          {/* 创建 */}
          {tabKeyType === '1' ? (
            <div>
              <LabelForm
                open={true}
                dataList={[{ namespace: '', workLoadTypes: '', key: '', value: '' }]}
                callbackOk={handleOk}
                callbackCancel={handleCancel}
                isEdit={true}
                isShow={isShow}
              ></LabelForm>
            </div>
          ) : (
            <div>
              <ExactForm
                open={true}
                dataList={[{ namespace: '', workLoadTypes: '', name: '' }]}
                callbackOk={handleOk}
                callbackCancel={handleCancel}
                isEdit={true}
              ></ExactForm>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
