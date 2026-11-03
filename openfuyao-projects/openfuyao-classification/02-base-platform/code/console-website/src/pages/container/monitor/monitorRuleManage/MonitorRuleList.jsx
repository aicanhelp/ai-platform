/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { useEffect, useState, useCallback, useStore, useRef } from 'openinula';
import { useTranslation } from 'react-i18next';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { containerRouterPrefix } from '@/constant';
import '@/styles/pages/workload.less';
import { getAlertListData } from '@/api/monitorApi';
import {
  Button,
  Form,
  Space,
  Table,
  Select,
}
  from 'antd';
import {
  DEFAULT_CURRENT_PAGE,
  DEFAULT_PAGE_SIZE,
  ResponseCode,
  alertLevelOptions,
  alertStatusOptions,
  alertStatusEx,
} from '@/common/constants';
import { Link } from 'inula-router';
import { filterAlertStatus, filterAlertStatusStyle } from '@/utils/common';
import { sorterFirstAlphabet } from '@/tools/utils';

/** 告警规则状态筛选项（不含「全部」） */
const alertRuleStatusFilterOptions = alertStatusOptions.filter((o) => o.value !== '');

export default function MonitorRuleList() {
  const { t } = useTranslation('monitoring');
  const [alertRuleForm] = Form.useForm();
  const [groupOptions, setGroupOptions] = useState([]);
  const [alertRulePage, setAlertRulePage] = useState(DEFAULT_CURRENT_PAGE);
  const [alaramRulePageSize, setAlertRulePageSize] = useState(DEFAULT_PAGE_SIZE);
  const [alaramRuleLoading, setAlertRuleLoading] = useState(true);
  const [alertRuleList, setAlertRuleList] = useState([]);
  const alertRuleListTotalRef = useRef([]); // 全量规则（搜索用）
  const themeStore = useStore('theme');

  const handleReset = () => {
    alertRuleForm.resetFields();
    getAlertList();
  };

  const handleSearch = () => {
    setAlertRuleLoading(true);
    let data = alertRuleListTotalRef.current;
    let conditions = alertRuleForm.getFieldsValue();
    if (conditions.group) {
      data = data.filter(item => item.group === conditions.group);
    }
    if (conditions.level) {
      data = data.filter(item => item.severity === conditions.level);
    }
    if (conditions.status) {
      data = data.filter(item => item.status === conditions.status);
    }
    setAlertRulePage(1);
    setAlertRulePageSize(10);
    setAlertRuleList([...data]);
    setAlertRuleLoading(false);
  };

  const solveCurrentOptions = (options) => {
    let filters = [];
    options.map(item => {
      if (item.label !== t('filter.all')) {
        filters.push({ text: item.label, value: item.value });
      }
    });
    return filters;
  };

  const alertRuleColumns = [
    {
      title: t('table.alertGroup'),
      key: 'group',
      sorter: (a, b) => sorterFirstAlphabet(a.group, b.group),
      render: (_, record) => <Link to={`/${containerRouterPrefix}/monitor/monitorRuleManage/detail/${record.name}`}>{record.group}</Link>,
    },
    {
      title: t('table.alertName'),
      key: 'alertName',
      sorter: (a, b) => sorterFirstAlphabet(a.name, b.name),
      render: (_, record) => record.name,
    },
    {
      title: t('table.alertLevel'),
      key: 'level',
      sorter: (a, b) => sorterFirstAlphabet(filterAlertStatus(a.severity), filterAlertStatus(b.severity)),
      filters: solveCurrentOptions(alertLevelOptions),
      onFilter: (value, record) => record.severity === value,
      filterMultiple: false,
      render: (_, record) => <div style={filterAlertStatusStyle(record.severity)} >
        {filterAlertStatus(record.severity)}
      </div >,
    },
    {
      title: t('table.alertStatus'),
      key: 'status',
      filters: alertRuleStatusFilterOptions.map((item) => ({ text: item.label, value: item.value })),
      onFilter: (value, record) => record.status === value,
      filterMultiple: false,
      sorter: (a, b) => sorterFirstAlphabet(a.status, b.status),
      width: 220,
      render: (_, record) => <div className={'status_group'}>
        <span className={`${(record.status).toLowerCase()}_circle`}></span>
        <span>{alertStatusEx[record.status] ?? record.status}</span>
      </div>,
    },
  ];

  const getAlertList = useCallback(async () => {
    setAlertRuleLoading(true);
    try {
      const res = await getAlertListData();
      if (res.status === ResponseCode.OK) {
        setAlertRuleList([...res.data.alertingRules]);
        alertRuleListTotalRef.current = res.data?.alertingRules ?? [];
        // 设置options
        makeOptions(res.data.alertingRules);
      }
    } catch (e) {
      if (e?.response?.data?.code === ResponseCode.NotFound) {
        setAlertRuleList([]); // 数组为空
      }
    }
    setAlertRuleLoading(false);
  }, []);

  const makeOptions = (data) => {
    let groupOptionsFinally = [{ label: t('filter.all'), value: '' }];
    let groupOptionsTempory = [];
    data.map(item => {
      if (!groupOptionsTempory.includes(item.group)) {
        groupOptionsTempory.push(item.group);
        groupOptionsFinally.push({ label: item.group, value: item.group });
      }
    });
    setGroupOptions([...groupOptionsFinally]);
  };

  useEffect(() => {
    getAlertList();
  }, [getAlertList]);

  return <div className="child_content no_breadcrumb">
    <BreadCrumbCom
      className="create_bread"
      items={[
        { title: t('module.title'), path: `/${containerRouterPrefix}/monitor`, disabled: true },
        { title: t('module.ruleList'), path: '/monitorRuleManage' },
      ]} />
    <div className="tab_container container_margin_box backgroundAdjust_box">
      <Form className="alermForm pod_searchForm form_padding_bottom" style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }} form={alertRuleForm} initialValues={{
        group: '',
        level: '',
        status: undefined,
      }}>
        <Form.Item name="group" className="pod_search_input" label={t('filter.alertGroup')}>
          <Select options={groupOptions} />
        </Form.Item>
        <Form.Item name="level" className="pod_search_input" label={t('filter.alertLevel')}>
          <Select options={alertLevelOptions} />
        </Form.Item>
        <Form.Item name="status" className="pod_search_input" label={t('filter.alertStatus')}>
          <Select allowClear options={alertRuleStatusFilterOptions} placeholder={t('filter.alertStatusPlaceholder')} />
        </Form.Item>
        <Form.Item>
          <Space>
            <Button className='cancel_btn' onClick={handleReset}>{t('actions.reset')}</Button>
            <Button className='primary_btn' onClick={handleSearch}>{t('actions.search')}</Button>
          </Space>
        </Form.Item>
      </Form>
      <div className="tab_table_flex monitorRule_table">
        <Table
          className="table_padding"
          loading={alaramRuleLoading}
          columns={alertRuleColumns}
          dataSource={alertRuleList}
          scroll={{ x: 1280 }}
          pagination={{
            className: 'page',
            current: alertRulePage,
            pageSize: alaramRulePageSize,
            onChange: (current, limit) => {
              setAlertRulePage(current);
              setAlertRulePageSize(limit);
            },
            showTotal: (total) => t('table.total', { total }),
            showSizeChanger: true,
            showQuickJumper: true,
            pageSizeOptions: [10, 20, 50],
          }}
        />
      </div>
    </div>
  </div>;
}