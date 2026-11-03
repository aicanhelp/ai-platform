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
import { useParams, Link, useHistory, useLocation } from 'inula-router';
import { useEffect, useState, useCallback } from 'openinula';
import { useIntl } from 'inula-intl';
import { Form, message, Select, Input, Popover, Space, Button } from 'antd';
import { updateColocationRule, deleteColocationRule } from '../../api/colocationApi';
import '../../styles/colocationRuleManagementPage.less';
import ExactForm from './ExactForm';
import LabelForm from './LabelForm';
import Dayjs from 'dayjs';
import { DownOutlined } from '@ant-design/icons';
import ColocationDeleteModal from '../../components/ColocationDeleteModal';
import { filterStatusName } from '../../utils/common';
import ToolTipComponent from '@/components/ToolTipComponent';
import { CloseOutlined } from '@ant-design/icons';

export default function DetailModel({ modelType }) {
  const intl = useIntl();
  const param = useParams();
  const history = useHistory();
  const location = useLocation();
  const [colocationRuleExactEditForm] = Form.useForm();
  const [messageApi, contextHolder] = message.useMessage();
  const [lookIsEdit, setLookIsEdit] = useState(false);
  const [exactPopOpen, setExactPopOpen] = useState(false);
  const [deleteModal, setDeleteModal] = useState(false);

  // 数据部分
  const [exactDetailData, setExactDetailData] = useState([]); // 详情数据
  const [exactSelectType, setExactSelectType] = useState(location.state.detail[0].colocationType);
  const [isShow, setIsShow] = useState(true);
  const handleRulesExactShowChange = () => {
    setIsShow(false);
  };

  const handleExactPopOpenChange = (open) => {
    setExactPopOpen(open);
  };

  const handleDeleteExact = () => {
    setExactPopOpen(false);
    setDeleteModal(true);
  };

  // 获取colocationRuleList
  const getColocationRuleExactList = () => {
    const arr = location.state.detail[0].selectors;
    if (modelType === 'exact') {
      arr.map((item) => {
        item.workLoadTypes = item.workLoadTypes.toString();
        return {
          ...item,
        };
      });
    } else {
      arr.map((item) => {
        item.workLoadTypes = item.workLoadTypes.toString();
        item.key = item.selector.matchExpressions[0].key;
        item.value = item.selector.matchExpressions[0].values.toString();
        return {
          ...item,
        };
      });
    }
    setExactDetailData(arr);
    colocationRuleExactEditForm.setFieldValue('colocationRuleName', location.state.detail[0].ruleName);
    colocationRuleExactEditForm.setFieldValue('colocationRuleType', filterStatusName(exactSelectType, intl));
  };

  // 标签成功回调
  const handleLabelOk = async (data) => {
    const exactOptData = JSON.parse(JSON.stringify(data));
    const values = await colocationRuleExactEditForm.validateFields();
    let arr = [];
    if (values) {
      if (modelType === 'exact') {
        arr = exactOptData.map((item) => {
          item.workLoadTypes = item.workLoadTypes.split(' ');
          item.name = item.name.trim().replace(/\s/g, '');
          return {
            ...item,
            selector: null,
          };
        });
      } else {
        arr = exactOptData.map((item) => {
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
      }
      const obj = {
        ruleName: colocationRuleExactEditForm.getFieldValue('colocationRuleName').trim().replace(/\s/g, ''),
        createTimestamp: Dayjs().format('YYYY-MM-DD HH:mm'),
        colocationType: exactSelectType,
        selectors: [...arr],
      };
      try {
        const res = await updateColocationRule(obj.ruleName, obj);
        if (res.status === ResponseCode.OK) {
          messageApi.success(intl.formatMessage({ id: 'rule.updateSuccess' }));
          setTimeout(() => {
            history.go(-1);
          }, 2000);
        } else {
          messageApi.error(intl.formatMessage({ id: 'rule.updateFailed' }));
        }
      } catch (e) {
        messageApi.error(`${intl.formatMessage({ id: 'message.apiError' })}${e.response.data.message}`);
      }
    }
  };

  // 标签失败回调
  const handleLabelCancel = () => {
    history.go(-1);
  };

  // 修改
  const handleExactToEdit = () => {
    setLookIsEdit(true);
  };

  // 删除确定
  const handleDeleteConfirmBack = async () => {
    // 此处接口
    try {
      const res = await deleteColocationRule(colocationRuleExactEditForm.getFieldValue('colocationRuleName'));
      if (res.status === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'rule.deleteSuccess' }));
        setTimeout(() => {
          history.go(-1);
        }, 1000);
      } else {
        messageApi.error(intl.formatMessage({ id: 'rule.deleteFailed' }));
      }
    } catch (e) {
      messageApi.error(intl.formatMessage({ id: 'message.apiError' }));
    }
    setDeleteModal(false);
  };

  // 删除取消
  const handleDeleteCancel = () => {
    setDeleteModal(false);
  };

  useEffect(() => {
    getColocationRuleExactList();
  }, [getColocationRuleExactList]);

  return (
    <div className="child_content">
      {contextHolder}
      <div className="rules_detail_title">
        <div style={{ marginRight: '64px' }}>
          <h3>{location.state.detail[0].ruleName}</h3>
        </div>
        <Popover
          placement="bottom"
          getPopupContainer={() => document.getElementsByClassName('rules_detail_title')[0]}
          className="rules_column_pop"
          content={
            <div className="col_button">
              <Button type="link" onClick={handleExactToEdit} disabled={!param.onlyRead}>
                {intl.formatMessage({ id: 'common.edit' })}
              </Button>
              <Button type="link" onClick={handleDeleteExact}>
                {intl.formatMessage({ id: 'common.delete' })}
              </Button>
            </div>
          }
          open={exactPopOpen}
          onOpenChange={handleExactPopOpenChange}
        >
          <Button className="primary_btn">
            {intl.formatMessage({ id: 'common.action' })} <DownOutlined className="small_margin_adjust" />
          </Button>
        </Popover>
      </div>
      <div className="rules_form_detail_content">
        <div className={`${isShow ? '' : 'cant_show'}`}>
          <ToolTipComponent>
            <div
              className="daemonsetDetail"
              style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
            >
              <div className="promot-box">{intl.formatMessage({ id: 'rule.tipMessage' })}</div>
              <CloseOutlined onClick={handleRulesExactShowChange} />
            </div>
          </ToolTipComponent>
        </div>
        <div className={`${isShow ? 'normal_nocreate_rules_overflow_height' : 'rules_nocreate_overflow_height'}`}>
          <div className="rules_exact_detail_content_top">
            <Form
              form={colocationRuleExactEditForm}
              initialValues={{
                colocationRuleType: exactSelectType,
              }}
              disabled={(param.onlyRead && !lookIsEdit) || (!param.onlyRead && lookIsEdit)}
            >
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
                    disabled
                  />
                </Form.Item>
                <Form.Item name="colocationRuleType" label={intl.formatMessage({ id: 'rule.colocationType' })}>
                  <Input
                    placeholder={intl.formatMessage({ id: 'rule.colocationType' })}
                    className="rules_exact_detail_content_top_select"
                    disabled
                  />
                </Form.Item>
              </div>
            </Form>
          </div>

          {(param.onlyRead && !lookIsEdit) || (!param.onlyRead && lookIsEdit) ? (
            <div>
              {modelType === 'exact' ? (
                <ExactForm
                  open={true}
                  dataList={exactDetailData}
                  callbackOk={handleLabelOk}
                  callbackCancel={handleLabelCancel}
                  isEdit={false}
                />
              ) : (
                <LabelForm
                  open={true}
                  dataList={exactDetailData}
                  callbackOk={handleLabelOk}
                  callbackCancel={handleLabelCancel}
                  isEdit={false}
                />
              )}
            </div>
          ) : (
            <div>
              {modelType === 'exact' ? (
                <ExactForm
                  open={true}
                  dataList={exactDetailData}
                  callbackOk={handleLabelOk}
                  callbackCancel={handleLabelCancel}
                  isEdit={true}
                />
              ) : (
                <LabelForm
                  open={true}
                  dataList={exactDetailData}
                  callbackOk={handleLabelOk}
                  callbackCancel={handleLabelCancel}
                  isEdit={true}
                />
              )}
            </div>
          )}
        </div>
      </div>
      <ColocationDeleteModal
        title={intl.formatMessage({ id: 'rule.deleteTitle' })}
        open={deleteModal}
        cancelFn={handleDeleteCancel}
        content={[
          intl.formatMessage(
            { id: 'rule.confirmDeleteByName' },
            { name: colocationRuleExactEditForm.getFieldValue('colocationRuleName') },
          ),
        ]}
        confirmFn={handleDeleteConfirmBack}
        leftText={intl.formatMessage({ id: 'common.delete' })}
      />
    </div>
  );
}
