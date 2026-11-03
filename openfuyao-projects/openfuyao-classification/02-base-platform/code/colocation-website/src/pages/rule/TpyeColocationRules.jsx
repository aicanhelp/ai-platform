/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import BreadCrumbCom from '../../components/BreadCrumbCom';
import ColocationDeleteModal from '../../components/ColocationDeleteModal';
import { containerRouterPrefix, ResponseCode } from '../../common/constants';
import { useParams, Link, useHistory } from 'inula-router';
import { useEffect, useState, useCallback } from 'openinula';
import { SyncOutlined, MoreOutlined, DownOutlined, CloseOutlined } from '@ant-design/icons';
import { Form, Table, Space, Input, Button, message, Popover } from 'antd';

import { getColocationRulesListData, deleteColocationRule } from '../../api/colocationApi';
import '../../styles/colocationRuleManagementPage.less';
import { sorterFirstAlphabe, sortJobByTime } from '../../tools/utils';
import { filterStatusName, reseverStatusName } from '../../utils/common';
import Dayjs from 'dayjs';
import ToolTipComponent from '@/components/ToolTipComponent';
import { useIntl } from 'inula-intl';

/**
 *
 * @param type 类型
 * @returns
 */
export default function ColocationRulesManagementPage({ type }) {
  const intl = useIntl();
  const [isShow, setIsShow] = useState(true);
  const param = useParams();
  const history = useHistory();
  const [colocationRuleForm] = Form.useForm();
  const [messageApi, contextHolder] = message.useMessage();
  const [popOpen, setPopOpen] = useState('');
  const [creatPopOpen, setCreatPopOpen] = useState(false); // 气泡悬浮
  // table部分
  const [originalList, setOriginalList] = useState([]); // 原始数据
  const [colocationRuleLoading, setColocationRuleLoading] = useState(false);
  const [colocationRuleList, setColocationRuleList] = useState([]); // 工作负载table数据集
  const [colocationRulePage, setColocationRulePage] = useState(1);
  const [filterRuleStatus, setFilterRuleStatus] = useState([
    { text: intl.formatMessage({ id: 'rule.online' }), value: '在线' },
    { text: intl.formatMessage({ id: 'rule.offline' }), value: '离线' },
  ]); // 规则混部混部赋值筛选项
  const [filterRuleValue, setFilterRuleValue] = useState(); // 筛选值

  // 删除提醒弹窗
  const [deleteModal, setDeleteModal] = useState(false);
  const [deleteName, setDeleteName] = useState('');

  // 列表项
  const ruleColumns = [
    {
      title: intl.formatMessage({ id: 'rule.name' }),
      key: 'rule_name',
      dataIndex: 'name',
      width: 300,
      sorter: (a, b) => sorterFirstAlphabe(a.ruleName, b.ruleName),
      render: (_, record) => (
        <Space>
          <div style={{ color: '#3f66f5', cursor: 'pointer' }} onClick={() => handleOptRule(record, 'look')}>
            {record.ruleName}
          </div>
        </Space>
      ),
    },
    {
      title: intl.formatMessage({ id: 'rule.matchCount' }),
      key: 'rule_attribute',
      width: 200,
      sorter: (a, b) => sorterFirstAlphabe(a.selectors.length, b.selectors.length),
      render: (_, record) => (
        <Space>
          <div>{record.selectors.length}</div>
        </Space>
      ),
    },
    {
      title: intl.formatMessage({ id: 'table.column.createTime' }),
      key: 'rule_time',
      width: 200,
      sorter: (a, b) => sortJobByTime(a, b),
      render: (_, record) => (
        <Space>{Dayjs(record.creationTimestamp ? record.creationTimestamp : 'none').format('YYYY-MM-DD HH:mm')}</Space>
      ),
    },
    {
      title: intl.formatMessage({ id: 'common.action' }),
      key: 'rule_handle',
      width: 100,
      render: (_, record) => (
        <Space>
          <Popover
            placement="bottom"
            content={
              <div className="colocation_rules_col_popver">
                <Button type="link" onClick={() => handleOptRule(record, 'update')}>
                  {intl.formatMessage({ id: 'common.edit' })}
                </Button>
                <Button type="link" onClick={() => handleDeleteRule(record)}>
                  {intl.formatMessage({ id: 'common.delete' })}
                </Button>
              </div>
            }
            trigger="click"
            open={popOpen === `${record.ruleName}`}
            onOpenChange={(newOpen) => (newOpen ? setPopOpen(`${record.ruleName}`) : setPopOpen(''))}
          >
            <MoreOutlined className="common_antd_icon primary_color" />
          </Popover>
        </Space>
      ),
    },
  ];

  const handleRolePopOpenChange = (open) => {
    setCreatPopOpen(open);
  };

  const handleRulesShowChange = () => {
    setIsShow(false);
  };

  // 获取colocationRuleList
  const getColocationRuleList = useCallback(async (isChange = true) => {
    setColocationRuleLoading(true);
    try {
      const res = await getColocationRulesListData();
      if (res.status === ResponseCode.OK) {
        // 此处先过滤在线属性再进行key遍历
        const typeArr = res.data?.data?.filter((item) => {
          return item.colocationType === type;
        });
        const arr = typeArr.map((item, index) => {
          return {
            ...item,
            key: index,
          };
        });
        setOriginalList(arr);
        handleSearchColocationRule(arr, isChange); // 先搜索
      } else {
        messageApi.error(intl.formatMessage({ id: 'message.apiError' }));
      }
    } catch (e) {
      setColocationRuleList([]); // 数组为空
    }
    setColocationRuleLoading(false);
  }, []);

  // 外部检索
  const handleSearchColocationRule = (totalData = originalList, isChange = true) => {
    const colocationRuleFormName = colocationRuleForm.getFieldValue('colocationRule_name');
    let temporyList = totalData;
    if (colocationRuleFormName) {
      temporyList = temporyList.filter((item) => item.ruleName.includes(colocationRuleFormName));
    }
    setColocationRuleList([...temporyList]);
    isChange ? setColocationRulePage(1) : null;
  };

  // 混部节点移除
  const handleDeleteRule = (record) => {
    setPopOpen('');
    setDeleteModal(true);
    setDeleteName(record.ruleName);
  };

  // 修改
  const handleOptRule = async (record, opt) => {
    setPopOpen('');
    if (opt === 'update') {
      if (record.selectors[0].name !== '' && record.selectors[0].selector === null) {
        history.push({
          pathname: `/ColocationRulesManagement/detail/${record.ruleName}/${'exact'}`,
          state: { detail: originalList.filter((item) => item.ruleName === record.ruleName) },
        });
      } else if (record.selectors[0].name === '' && record.selectors[0].selector !== null) {
        history.push({
          pathname: `/ColocationRulesManagement/detail/${record.ruleName}/${'label'}`,
          state: { detail: originalList.filter((item) => item.ruleName === record.ruleName) },
        });
      } else {
        messageApi.error(intl.formatMessage({ id: 'rule.typeMismatchCannotEdit' }));
      }
    } else {
      if (record.selectors[0].name !== '' && record.selectors[0].selector === null) {
        history.push({
          pathname: `/ColocationRulesManagement/detail/${record.ruleName}/${'exact'}/${'onlyRead'}`,
          state: { detail: originalList.filter((item) => item.ruleName === record.ruleName) },
        });
      } else if (record.selectors[0].name === '' && record.selectors[0].selector !== null) {
        history.push({
          pathname: `/ColocationRulesManagement/detail/${record.ruleName}/${'label'}/${'onlyRead'}`,
          state: { detail: originalList.filter((item) => item.ruleName === record.ruleName) },
        });
      } else {
        messageApi.error(intl.formatMessage({ id: 'rule.typeMismatchCannotView' }));
      }
    }
  };

  const handleTableChange = useCallback((_pagination, filter, _sorter, extra) => {
    if (extra.action === 'filter') {
      setFilterRuleValue(filter.rule_attribute);
    }
  }, []);

  // 重置按钮
  const handleColocationRuleReset = () => {
    getColocationRuleList();
  };

  // 创建入口
  const handleColocationRuleCreate = (createWay, createType) => {
    if (colocationRuleList.length < 100) {
      if (createWay === 'label') {
        history.push(`/ColocationRulesManagement/create/lable/${createType}`);
      } else {
        history.push(`/ColocationRulesManagement/create/exact/${createType}`);
      }
    } else {
      messageApi.error(intl.formatMessage({ id: 'rule.maxLimitReached' }));
    }
  };

  // 删除确定
  const handleDeleteConfirmBack = async () => {
    // 此处接口
    try {
      const res = await deleteColocationRule(deleteName);
      if (res.status === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'rule.deleteSuccess' }));
        setTimeout(() => {
          getColocationRuleList();
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
    getColocationRuleList();
  }, [getColocationRuleList]);

  return (
    <div className="child_content">
      <div className="rules_detail_content">
        {contextHolder}
        <div className={`${isShow ? '' : 'cant_show'}`}>
          <ToolTipComponent>
            <div
              className="statefulSetDetail"
              style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
            >
              <div className="promot-box">{intl.formatMessage({ id: 'rule.tipMessage' })}</div>
              <CloseOutlined onClick={handleRulesShowChange} />
            </div>
          </ToolTipComponent>
        </div>
        <div className={` ${isShow ? 'rules_detail_content_box' : 'normal_rules_detail_content_box'}`}>
          <div className="rules_detail_form">
            <Form className="rules-searchForm form_padding_bottom" form={colocationRuleForm}>
              <Form.Item name="colocationRule_name" className="rules-search-input">
                <Input.Search
                  placeholder={intl.formatMessage({ id: 'rule.searchName' })}
                  onSearch={() => handleSearchColocationRule()}
                  autoComplete="off"
                  maxLength={53}
                />
              </Form.Item>
              <Form.Item>
                <Space>
                  <Popover
                    placement="bottom"
                    content={
                      <Space className="colocation_rules_col_popver">
                        <Button type="link" onClick={() => handleColocationRuleCreate('label', type)}>
                          {intl.formatMessage({ id: 'rule.labelSelectorMode' })}
                        </Button>
                        <Button type="link" onClick={() => handleColocationRuleCreate('exact', type)}>
                          {intl.formatMessage({ id: 'rule.exactMatchMode' })}
                        </Button>
                      </Space>
                    }
                    open={creatPopOpen}
                    onOpenChange={handleRolePopOpenChange}
                  >
                    <Button className="rules_detail_form_button">
                      {intl.formatMessage({ id: 'rule.create' })}
                      <DownOutlined className="small_margin_adjust" />
                    </Button>
                  </Popover>
                  <Button
                    icon={<SyncOutlined />}
                    onClick={handleColocationRuleReset}
                    className="reset_btn"
                    style={{ marginLeft: '16px' }}
                  ></Button>
                </Space>
              </Form.Item>
            </Form>
          </div>
          <div className="rules_detail_content_table">
            <Table
              locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
              loading={colocationRuleLoading}
              columns={ruleColumns}
              dataSource={colocationRuleList}
              onChange={handleTableChange}
              pagination={{
                className: 'colocation-website-page',
                current: colocationRulePage,
                showTotal: (total) => intl.formatMessage({ id: 'pagination.total' }, { total }),
                showSizeChanger: true,
                showQuickJumper: true,
                pageSizeOptions: [10, 20, 50],
                onChange: (page) => setColocationRulePage(page),
              }}
              expandable={{
                expandedRowRender: (record) => (
                  <div style={{ marginLeft: '120px', display: 'flex', flexDirection: 'column' }}>
                    <p style={{ color: '#CCC', marginBottom: '8px' }}>
                      {intl.formatMessage({ id: 'rule.matchingRules' })}:
                    </p>
                    {record.selectors.map((item) => {
                      return (
                        <div style={{ display: 'flex', marginBottom: '6px' }}>
                          <p
                            style={{
                              background: '#cfe7ff',
                              borderRadius: '6px',
                              padding: '2px 12px',
                              marginRight: '8px',
                              color: '#333',
                            }}
                          >
                            namespace={item.namespace ? item.namespace : '--'}
                          </p>
                          <p
                            style={{
                              background: '#cfe7ff',
                              borderRadius: '6px',
                              padding: '2px 12px',
                              marginRight: '8px',
                              color: '#333',
                            }}
                          >
                            workload={item.workLoadTypes[0] ? item.workLoadTypes[0] : '--'}
                          </p>
                          {item.name ? (
                            <p
                              style={{
                                background: '#cfe7ff',
                                borderRadius: '6px',
                                padding: '2px 12px',
                                marginRight: '8px',
                                color: '#333',
                              }}
                            >
                              name={item.name}
                            </p>
                          ) : (
                            <></>
                          )}
                          {item.selector?.matchExpressions.map((innerSelect) => {
                            return (
                              <p
                                style={{
                                  background: '#cfe7ff',
                                  borderRadius: '6px',
                                  padding: '2px 12px',
                                  marginRight: '8px',
                                  color: '#333',
                                }}
                              >
                                {innerSelect.key}={innerSelect.values.join(',')}
                              </p>
                            );
                          })}
                        </div>
                      );
                    })}
                  </div>
                ),
              }}
            ></Table>
          </div>
        </div>
        <ColocationDeleteModal
          title={intl.formatMessage({ id: 'rule.deleteTitle' })}
          open={deleteModal}
          cancelFn={handleDeleteCancel}
          content={[intl.formatMessage({ id: 'rule.confirmDeleteByName' }, { name: deleteName })]}
          confirmFn={handleDeleteConfirmBack}
          leftText={intl.formatMessage({ id: 'common.delete' })}
        />
      </div>
    </div>
  );
}
