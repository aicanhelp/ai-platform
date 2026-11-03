/**
 *  Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  openFuyao is licensed under Mulan PSL v2.
 *  You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  You may obtain a copy of Mulan PSL v2 at:

 *       http://license.coscl.org.cn/MulanPSL2

 *   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *   EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *   MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *   See the Mulan PSL v2 for more details.
 */
import { useEffect, useState, useCallback, useStore } from 'openinula';
import { useTranslation } from 'react-i18next';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { containerRouterPrefix } from '@/constant';
import Dayjs from 'dayjs';
import '@/styles/pages/workload.less';
import {
  Button,
  Form,
  Space,
  Input,
  Table,
  Popover,
  message,
}
  from 'antd';
import { DEFAULT_CURRENT_PAGE, ResponseCode } from '@/common/constants';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { MoreOutlined } from '@ant-design/icons';
import { Link, useHistory } from 'inula-router';
import {
  getCustomResourceDefinitionDetailDescription,
  getResourceInstanceList,
  deleteResourceInstance,
} from '@/api/containerApi';
import { disabledModifyMonitorServiceCr } from '@/common/constants';
import { sorterFirstAlphabet, forbiddenMsg, showSuccessThen } from '@/tools/utils';

export default function MonitorServiceMonitor() {
  const { t } = useTranslation('monitoring');
  const [serviceMonitorForm] = Form.useForm();
  const history = useHistory();
  const [serviceMonitorPage, setServiceMonitorPage] = useState(DEFAULT_CURRENT_PAGE);
  const [serviceMonitorList, setServiceMonitorList] = useState([]); // 数据集
  const [serviceMonitorLoading, setServiceMonitorLoading] = useState(false); // 加载中
  const [popOpen, setPopOpen] = useState(''); // 悬浮框是否展示
  const [prefixObj, setPrefixObj] = useState({}); // 前缀cr实例组
  const themeStore = useStore('theme');
  const [serviceMonitorDelModalOpen, setServiceMonitorDelModalOpen] = useState(false); // 删除对话框展示
  const [serviceMonitorDelName, setServiceMonitorDelName] = useState(''); // 删除的名称
  const [serviceMonitorDelNamespace, setServiceMonitorDelNamespace] = useState(''); // 删除的命名空间
  const [isServiceMonitorDelCheck, setIsServiceMonitorDelCheck] = useState(false); // 是否选中
  const [messageApi, contextHolder] = message.useMessage();
  const [originalList, setOriginalList] = useState([]); // 原始数据
  // 列表项
  const serviceMonitorColumns = [
    {
      title: t('serviceMonitor.table.instanceName'),
      key: 'exampleName',
      sorter: (a, b) => sorterFirstAlphabet(a.metadata.name, b.metadata.name),
      render: (_, record) => <Link to=
        {{
          pathname: `/${containerRouterPrefix}/monitor/monitorGoalManage/serviceMonitor/${record.metadata.namespace}/${record.metadata.name}`, state: {
            ...prefixObj,
          },
        }}>
        {record.metadata.name}
      </Link>,
    },
    {
      title: t('serviceMonitor.table.namespace'),
      key: 'namespace',
      sorter: (a, b) => sorterFirstAlphabet(a.metadata.namespace, b.metadata.namespace),
      render: (_, record) => record.metadata.namespace,
    },
    {
      title: t('serviceMonitor.table.createTime'),
      key: 'created_time',
      sorter: (a, b) => Dayjs(a.metadata.creationTimestamp) - Dayjs(b.metadata.creationTimestamp),
      render: (_, record) => Dayjs(record.metadata.creationTimestamp).format('YYYY-MM-DD HH:mm'),
    },
    {
      title: t('table.operation'),
      width: 120,
      key: 'handle',
      fixed: 'right',
      render: (_, record) => (
        <Space>
          <Popover placement="bottom"
            content={
              <div className="pop_modal">
                <Link className="primary_link" to={{
                  pathname: `/${containerRouterPrefix}/monitor/monitorGoalManage/serviceMonitor/${record.metadata.namespace}/${record.metadata.name}/yaml`,
                  state: { ...prefixObj },
                }}>
                  {t('serviceMonitor.actions.modify')}
                </Link>
                <Button
                  type="link"
                  onClick={() => handleDeleteServiceMonitor(record)}
                >
                  {t('serviceMonitor.actions.delete')}
                </Button>
              </div>
            }
            trigger="click"
            open={popOpen === (record.metadata.uid ?? record.metadata.name) && !disabledModifyMonitorServiceCr.includes(record.metadata.name)}
            onOpenChange={newOpen => newOpen ? setPopOpen(record.metadata.uid ?? record.metadata.name) : setPopOpen('')}>
            <MoreOutlined className={`common_antd_icon ${disabledModifyMonitorServiceCr.includes(record.metadata.name) ? 'disabled_color' : 'primary_color'}`} />
          </Popover>
        </Space>
      ),
    },
  ];

  // 删除按钮
  const handleDeleteServiceMonitor = (record) => {
    // 隐藏气泡框
    setPopOpen('');
    setIsServiceMonitorDelCheck(false);
    setServiceMonitorDelModalOpen(true); // 打开弹窗
    setServiceMonitorDelName(record.metadata.name);
    setServiceMonitorDelNamespace(record.metadata.namespace);
  };

  const handleDelServiceMonitorCancel = () => {
    setServiceMonitorDelModalOpen(false);
    setIsServiceMonitorDelCheck(false);
    setServiceMonitorDelName('');
    setServiceMonitorDelNamespace('');
  };

  const handleDelServiceMonitorConfirm = async () => {
    try {
      const res = await deleteResourceInstance({ ...prefixObj, namespace: serviceMonitorDelNamespace }, serviceMonitorDelName);
      if (res.status === ResponseCode.OK) {
        showSuccessThen(messageApi, t('messages.deleteSuccess'), () => {
          setServiceMonitorDelModalOpen(false);
          setIsServiceMonitorDelCheck(false);
          setServiceMonitorDelName('');
          setServiceMonitorDelNamespace('');
          getServiceMonitorList();
        });
      }
    } catch (error) {
      if (error.response.status === ResponseCode.Forbidden) {
        forbiddenMsg(messageApi, error);
      } else {
        messageApi.error(`${t('messages.deleteFailed')}${error.response.data.message}`);
      }
    }
  };
  const handleServiceMonitorCheckFn = (e) => {
    setIsServiceMonitorDelCheck(e.target.checked);
  };

  const getServiceMonitorList = useCallback(async (isChange = true) => {
    if (Object.keys(prefixObj).length) {
      setServiceMonitorLoading(true);
      try {
        const res = await getResourceInstanceList(prefixObj);
        if (res.status === ResponseCode.OK) {
          setOriginalList([...res.data.items]);
          handleSearchServiceMonitor(res.data.items, isChange); // 先搜索
        }
      } catch (e) {
        if (e.response.data.code === ResponseCode.NotFound) {
          setServiceMonitorList([]); // 数组为空
        }
      }
      setServiceMonitorLoading(false);
    }
  }, [prefixObj]);

  const getCustomResource = useCallback(async () => {
    const res = await getCustomResourceDefinitionDetailDescription('servicemonitors.monitoring.coreos.com');
    if (res.status === ResponseCode.OK) {
      // 处理组数据
      setPrefixObj({
        group: res.data.spec.group,
        version: res.data.spec.versions[0].name,
        plural: res.data.status.acceptedNames.plural,
      });
    }
  }, []);

  useEffect(() => {
    // 获取自定义资源
    getCustomResource();
  }, [getCustomResource]);

  // 检索
  const handleSearchServiceMonitor = (totalData = originalList, isChange = true) => {
    const serviceMonitorFormName = serviceMonitorForm.getFieldValue('service_monitor_name');
    let temporyList;
    if (serviceMonitorFormName) {
      temporyList = totalData.filter(item => (item.metadata.name).toLowerCase().includes(serviceMonitorFormName.toLowerCase()));
    } else {
      temporyList = totalData;
    }
    setServiceMonitorList([...temporyList]);
    isChange ? setServiceMonitorPage(DEFAULT_CURRENT_PAGE) : null;
  };

  useEffect(() => {
    getServiceMonitorList();
  }, [getServiceMonitorList]);

  return <div className="child_content no_breadcrumb">
    <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
      {contextHolder}
    </div>
    <BreadCrumbCom
      className="create_bread"
      items={[
        { title: t('module.title'), path: `/${containerRouterPrefix}/monitor`, disabled: true },
        { title: t('module.targetList'), path: '/monitorGoalManage' },
        { title: t('serviceMonitor.title'), path: '/serviceMonitor' },
      ]} />
    <div className="tab_container container_margin_box">
      <Form className="pod_searchForm form_padding_bottom" form={serviceMonitorForm}>
        <Form.Item name="service_monitor_name" className="pod_search_input">
          <Input.Search placeholder={t('searchPlaceholder', { ns: 'common', name: t('searchName.instance', { ns: 'common' }) })} onSearch={() => handleSearchServiceMonitor()} autoComplete="off" />
        </Form.Item>
        <Form.Item>
          <Space>
            <Button className="primary_btn" onClick={() => history.push({
              pathname: `/${containerRouterPrefix}/monitor/monitorGoalManage/serviceMonitor/create`,
              state: { ...prefixObj, namespace: '' },
            })}>{t('serviceMonitor.create')}</Button>
          </Space>
        </Form.Item>
      </Form>
      <div className="tab_table_flex">
        <Table
          className="table_padding"
          loading={serviceMonitorLoading}
          columns={serviceMonitorColumns}
          dataSource={serviceMonitorList}
          pagination={{
            className: 'page',
            current: serviceMonitorPage,
            showTotal: (total) => t('table.total', { total }),
            showSizeChanger: true,
            showQuickJumper: true,
            pageSizeOptions: [10, 20, 50],
            onChange: page => setServiceMonitorPage(page),
          }}
          scroll={{ x: 1280 }}
        />
        <DeleteInfoModal
          title={t('serviceMonitor.modal.deleteTitle')}
          open={serviceMonitorDelModalOpen}
          cancelFn={handleDelServiceMonitorCancel}
          content={[
            t('serviceMonitor.modal.deleteWarning'),
            t('serviceMonitor.modal.deleteConfirm', { name: serviceMonitorDelName }),
          ]}
          isCheck={isServiceMonitorDelCheck}
          showCheck={true}
          checkFn={handleServiceMonitorCheckFn}
          confirmFn={handleDelServiceMonitorConfirm} />
      </div>
    </div>
  </div>;
}
