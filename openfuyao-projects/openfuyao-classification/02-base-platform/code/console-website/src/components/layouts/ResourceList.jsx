/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import {
  Button,
  Form,
  Space,
  Input,
  Table,
  Popover,
  message,
} from 'antd';
import { SyncOutlined, MoreOutlined } from '@ant-design/icons';
import { useCallback, useEffect, useState, useContext, useMemo } from 'openinula';
import { useTranslation } from 'react-i18next';
import { DEFAULT_CURRENT_PAGE, ResponseCode } from '@/common/constants';
import { Link, useHistory, useLocation } from 'inula-router';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { NamespaceContext } from '@/namespaceContext';
import { forbiddenMsg, showSuccessThen } from '@/tools/utils';
import '@/styles/pages/workload.less';
import ToastMsg from '@/components/ToastMsg';
import useButtonLoading from '@/hooks/useButtonLoading';

export default function ResourceList({
  resourceType,
  columns,
  getResourceFn,
  deleteResourceFn,
  validateDelete,
  deleteSuccessMessage,
  yamlEditPath,
  createLabel,
  createPath,
  createMenuItems = [],
  hideOperations = false,
  hideCreate = false,
}) {
  const { t, i18n } = useTranslation('workload');
  const defaultDeleteSuccessMessage = deleteSuccessMessage || t('messages.deleteSuccessWithRefresh');
  const defaultCreateLabel = createLabel || t('actions.create');
  const [searchForm] = Form.useForm();

  const history = useHistory();
  const location = useLocation();

  const { state } = location;

  const namespace = useContext(NamespaceContext);

  const [messageApi, contextHolder] = message.useMessage();
  const [resourceList, setResourceList] = useState([]); // 数据集
  const [loading, setLoading] = useState(false); // 加载中
  const [popOpen, setPopOpen] = useState(''); // 悬浮框是否展示
  const [pageIndex, setPageIndex] = useState(DEFAULT_CURRENT_PAGE); // 页码
  const [delModalOpen, setDelModalOpen] = useState(false); // 删除对话框展示
  const [delRecord, setDelRecord] = useState(null); // 删除的资源对象
  const [isDelCheck, setIsDelCheck] = useState(false); // 是否选中

  const [originalList, setOriginalList] = useState([]); // 原始数据
  const [createPopOpen, setCreatePopOpen] = useState(false); // 气泡悬浮
  const [refreshLoading, setRefreshLoading] = useButtonLoading(); // 刷新按钮 loading

  // 刷新按钮
  const handleResetWorkload = async () => {
    setRefreshLoading(true);
    try {
      await getResourceList(false);
    } finally {
      setRefreshLoading(false);
    }
  };

  // 打开创建气泡菜单
  const handleRolePopOpenChange = (open) => {
    setCreatePopOpen(open);
  };

  // 删除按钮
  const handleDeleteResource = (record) => {
    if (validateDelete) {
      const errMsg = validateDelete(record);
      if (errMsg) {
        messageApi.error(errMsg);
        return;
      }
    }
    // 隐藏气泡框
    setPopOpen('');
    setDelModalOpen(true); // 打开弹窗
    setDelRecord(record);
    setIsDelCheck(false);
  };

  const handleDelCancel = () => {
    setDelModalOpen(false);
    setDelRecord(null);
  };

  const handleDelConfirm = async () => {
    if (!delRecord) {
      return;
    }
    try {
      const res = await deleteResourceFn(delRecord);
      if (res.status === ResponseCode.OK) {
        await showSuccessThen(messageApi, defaultDeleteSuccessMessage, () => {
          setIsDelCheck(false);
          setDelModalOpen(false);
          setDelRecord(null);
          getResourceList();
        });
      }
    } catch (error) {
      if (error.response?.status === ResponseCode.Forbidden) {
        forbiddenMsg(messageApi, error);
      } else {
        messageApi.error(`${t('messages.deleteFailed')}${error.response?.data?.message}`);
      }
    }
  };

  const handleDelCheckFn = (e) => {
    setIsDelCheck(e.target.checked);
  };

  const handlePopOpenChange = (newOpen, record) => {
    if (newOpen) {
      setPopOpen(record.metadata.uid);
    } else {
      setPopOpen('');
    }
  };

  const handleCreate = useCallback(() => {
    history.push(createPath || `${location.pathname.trim('/')}/create`);
  }, [history, location.pathname, createPath]);

  const tableColumns = useMemo(() => {
    let dataCols = columns;
    if (state?.status) {
      dataCols = columns.map((col) => {
        if (col.key === 'status' && col.filters && !col.defaultFilteredValue) {
          return { ...col, defaultFilteredValue: [state.status.toLowerCase()] };
        }
        return col;
      });
    }

    if (hideOperations) {
      return dataCols;
    }

    return [...dataCols, {
      title: t('actions.operation'),
      width: 120,
      key: 'handle',
      fixed: 'right',
      render: (_, record) => (
        <Space>
          <Popover
            className='pod'
            placement="bottom"
            content={
              <div className="pop_modal">
                <Button type="link"><Link to={yamlEditPath(record)}>{t('actions.modify')}</Link></Button>
                <Button type="link" onClick={() => handleDeleteResource(record)}>{t('actions.delete')}</Button>
              </div>
            }
            trigger="click"
            open={popOpen === record.metadata.uid}
            onOpenChange={e => handlePopOpenChange(e, record)}>
            <MoreOutlined className="common_antd_icon primary_color" />
          </Popover>
        </Space>
      ),
    }];
  }, [columns, state, popOpen, t, yamlEditPath, i18n.language, hideOperations]);

  // 获取资源列表
  const getResourceList = useCallback(async (isChange = true) => {
    setLoading(true);
    try {
      const res = await getResourceFn(namespace);
      if (res.status === ResponseCode.OK) {
        setOriginalList([...res.data.items]);
        handleSearch(res.data.items, isChange); // 先搜索
      }
    } catch (e) {
      if (e.response?.data?.code === ResponseCode.NotFound) {
        setResourceList([]); // 数组为空
        setLoading(false);
        return;
      }
      messageApi.error(`${t('messages.fetchFailed')}: ${e.response?.data?.message}`);
    }
    setLoading(false);
  }, [namespace, getResourceFn]);

  // 检索
  const handleSearch = (totalData = originalList, isChange = true) => {
    const resourceName = searchForm.getFieldValue('resource_name');
    let temporyList = totalData;
    if (resourceName) {
      temporyList = temporyList.filter(item => (item.metadata.name).toLowerCase().includes(resourceName.toLowerCase()));
    }
    setResourceList([...temporyList]);
    isChange ? setPageIndex(DEFAULT_CURRENT_PAGE) : null;
  };

  useEffect(() => {
    if (state && state.status) {
      window.history.replaceState(null, '');
    }
  }, [state]);

  useEffect(() => {
    getResourceList();
  }, [getResourceList]);

  return <div className="tab_container container_margin_box">
    <ToastMsg contextHolder={contextHolder} />
    <Form className="pod_searchForm form_padding_bottom" form={searchForm}>
      <Form.Item name="resource_name" className="pod_search_input">
        <Input.Search placeholder={t('searchPlaceholder', { ns: 'common', name: resourceType })} onSearch={() => handleSearch()} autoComplete="off" />
      </Form.Item>
      <Form.Item>
        <Space>
          {!hideCreate && (
            createMenuItems?.length > 0 ? <Popover placement='bottom'
              content={
                <Space className='column_pop'>
                  {createMenuItems.map(item => (
                    <Button type="link" key={item.key} onClick={item.onClick}>{item.label}</Button>
                  ))}
                </Space>
              }
              open={createPopOpen}
              onOpenChange={handleRolePopOpenChange}
            >
              <Button className="primary_btn" onClick={() => setCreatePopOpen(true)}>{defaultCreateLabel}</Button>
            </Popover> : <Button className="primary_btn" onClick={handleCreate}>{defaultCreateLabel}</Button>
          )}
          <Button icon={<SyncOutlined />} onClick={handleResetWorkload} loading={refreshLoading} className="reset_btn" style={{ marginLeft: '16px' }}></Button>
        </Space>
      </Form.Item>
    </Form>
    <div className="tab_table_flex">
      <Table
        className="table_padding"
        loading={loading}
        columns={tableColumns}
        dataSource={resourceList}
        pagination={{
          className: 'page',
          current: pageIndex,
          showTotal: (total) => t('pagination.total', { ns: 'common', total }),
          showSizeChanger: true,
          showQuickJumper: true,
          pageSizeOptions: [10, 20, 50],
          onChange: (page) => setPageIndex(page),
        }}
        scroll={{ x: 1280 }}
      />
    </div>
    {!hideOperations && (
      <DeleteInfoModal
        title={t('actions.deleteConfirm', { resource: resourceType })}
        open={delModalOpen}
        cancelFn={handleDelCancel}
        content={[
          t('actions.deleteWarning', { resource: resourceType }),
          t('actions.deleteConfirmMessage', { resource: resourceType, name: delRecord?.metadata?.name || '' }),
        ]}
        isCheck={isDelCheck}
        showCheck={true}
        checkFn={handleDelCheckFn}
        confirmFn={handleDelConfirm} />
    )}
  </div>;
}