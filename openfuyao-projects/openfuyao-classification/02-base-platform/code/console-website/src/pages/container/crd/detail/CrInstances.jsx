/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Fragment, useState, useEffect, useCallback, useStore, useMemo, useRef } from 'openinula';
import { SyncOutlined, MoreOutlined } from '@ant-design/icons';
import { Button, Form, Space, Input, Popover, Table, message } from 'antd';
import { useTranslation } from 'react-i18next';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { Link, useHistory } from 'inula-router';
import { solveEncodePath, sorterFirstAlphabet, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import { DEFAULT_CURRENT_PAGE, ResponseCode } from '@/common/constants';
import {
  getResourceInstanceList,
  deleteResourceInstance,
  startVirtualMachine,
  stopVirtualMachine,
  restartVirtualMachine,
} from '@/api/containerApi';
import Dayjs from 'dayjs';
import { containerRouterPrefix } from '@/constant';

export default function CrInstances({ crdDetailData }) {
  const { t, i18n: i18nInstance } = useTranslation(['crd', 'common']);
  const [form] = Form.useForm();
  const history = useHistory();
  const [messageApi, contextHolder] = message.useMessage();
  const themeStore = useStore('theme');

  const [page, setPage] = useState(DEFAULT_CURRENT_PAGE);
  const [list, setList] = useState([]);
  const [loading, setLoading] = useState(false);
  const [actionLoading, setActionLoading] = useState(false); // VM 操作防重
  const [originalList, setOriginalList] = useState([]);
  const listReqSeqRef = useRef(0);
  const [popOpen, setPopOpen] = useState('');
  const [delModalOpen, setDelModalOpen] = useState(false);
  const [delName, setDelName] = useState('');
  const [isDelCheck, setIsDelCheck] = useState(false);
  const [currentNamespace, setCurrentNamespace] = useState('');

  const prefixObj = useMemo(() => ({
    group: crdDetailData?.spec?.group,
    version: crdDetailData?.spec?.versions?.[0]?.name,
    plural: crdDetailData?.spec?.names?.plural,
    scopeNamespace: crdDetailData?.spec?.scope,
  }), [crdDetailData]);

  const yamlExampleObj = useMemo(() => ({
    apiVersion: `${crdDetailData?.spec?.group}/${crdDetailData?.spec?.versions?.[0]?.name}`,
    kind: crdDetailData?.spec?.names?.kind,
    scopeNamespace: crdDetailData?.spec?.scope,
  }), [crdDetailData]);

  const isNamespaced = crdDetailData?.spec?.scope === 'Namespaced';

  const columns = useMemo(() => [
    {
      title: t('table.instanceName'),
      key: 'name',
      sorter: (a, b) => sorterFirstAlphabet(a.metadata.name, b.metadata.name),
      render: (_, record) => {
        const crDetailPath = isNamespaced
          ? `/${containerRouterPrefix}/customResourceDefinition/${solveEncodePath(crdDetailData.metadata.name)}/cr/namespace/${record.metadata.namespace}/${record.metadata.name}`
          : `/${containerRouterPrefix}/customResourceDefinition/${solveEncodePath(crdDetailData.metadata.name)}/cr/${record.metadata.name}`;
        return (
          <Link to={{
            pathname: crDetailPath,
            state: { ...prefixObj, namespace: record.metadata.namespace },
          }}>
            {record.metadata.name}
          </Link>
        );
      },
    },
    {
      title: t('table.namespace'),
      key: 'namespace',
      render: (_, record) => record.metadata.namespace || '--',
    },
    {
      title: t('table.createTime'),
      key: 'createTime',
      sorter: (a, b) => Dayjs(a.metadata.creationTimestamp) - Dayjs(b.metadata.creationTimestamp),
      render: (_, record) => Dayjs(record.metadata.creationTimestamp).format('YYYY-MM-DD HH:mm'),
    },
    {
      title: t('table.operation'),
      key: 'handle',
      fixed: 'right',
      width: 120,
      render: (_, record) => (
        <Space>
          <Popover placement="bottom"
            content={
              <div className="pop_modal">
                {crdDetailData.metadata.name === 'virtualmachines.kubevirt.io' && (
                  <>
                    <Button type="link" onClick={() => handleStartVM(record)}>{t('actions.start')}</Button>
                    <Button type="link" onClick={() => handleStopVM(record)}>{t('actions.stop')}</Button>
                    <Button type="link" onClick={() => handleRestartVM(record)}>{t('actions.restart')}</Button>
                  </>
                )}
                <Link className="primary_link" to={{
                  pathname: isNamespaced
                    ? `/${containerRouterPrefix}/customResourceDefinition/${solveEncodePath(crdDetailData.metadata.name)}/cr/namespace/${record.metadata.namespace}/${record.metadata.name}/yaml`
                    : `/${containerRouterPrefix}/customResourceDefinition/${solveEncodePath(crdDetailData.metadata.name)}/cr/${record.metadata.name}/yaml`,
                  state: { ...prefixObj, namespace: record.metadata.namespace },
                }}>
                  {t('actions.modify')}
                </Link>
                <Button type="link" onClick={() => handleDelete(record)}>{t('actions.delete')}</Button>
              </div>
            }
            trigger="click"
            open={popOpen === record.metadata.uid}
            onOpenChange={newOpen => newOpen ? setPopOpen(record.metadata.uid) : setPopOpen('')}>
            <MoreOutlined className="common_antd_icon primary_color" />
          </Popover>
        </Space>
      ),
    },
  ], [t, i18nInstance.language, crdDetailData, prefixObj, popOpen, isNamespaced]);

  const getList = useCallback(async (isChange = true) => {
    const seq = ++listReqSeqRef.current;
    setLoading(true);
    try {
      const res = await getResourceInstanceList(prefixObj);
      if (seq !== listReqSeqRef.current) { return; }
      if (res.status === ResponseCode.OK) {
        setOriginalList([...(res.data?.items ?? [])]);
        handleSearch(res.data.items ?? [], isChange);
      }
    } catch (e) {
      if (seq !== listReqSeqRef.current) { return; }
      if (e.response?.data?.code === ResponseCode.NotFound) {
        setList([]);
      }
    }
    if (seq === listReqSeqRef.current) {
      setLoading(false);
    }
  }, [prefixObj]);

  const handleSearch = (totalData = originalList, isChange = true) => {
    const searchName = form.getFieldValue('name');
    let tempList = totalData;
    if (searchName) {
      tempList = tempList.filter(item => (item.metadata.name).toLowerCase().includes(searchName.toLowerCase()));
    }
    setList([...tempList]);
    isChange ? setPage(DEFAULT_CURRENT_PAGE) : null;
  };

  const handleReset = () => {
    getList();
  };

  const handleStartVM = async (record) => {
    if (actionLoading) { return; }
    setActionLoading(true);
    setPopOpen('');
    try {
      const res = await startVirtualMachine(record.metadata.namespace, record.metadata.name);
      if (res.status === ResponseCode.OK) {
        messageApi.success(t('messages.startSuccess'));
        getList();
      }
    } catch (error) {
      if (error.response?.status === ResponseCode.Forbidden) {
        forbiddenMsg(messageApi, error, t);
      } else {
        messageApi.error(`${t('messages.startFailed')}${error.response?.data?.message || error.message}`);
      }
    } finally {
      setActionLoading(false);
    }
  };

  const handleStopVM = async (record) => {
    if (actionLoading) { return; }
    setActionLoading(true);
    setPopOpen('');
    try {
      const res = await stopVirtualMachine(record.metadata.namespace, record.metadata.name);
      if (res.status === ResponseCode.OK) {
        messageApi.success(t('messages.stopSuccess'));
        getList();
      }
    } catch (error) {
      if (error.response?.status === ResponseCode.Forbidden) {
        forbiddenMsg(messageApi, error, t);
      } else {
        messageApi.error(`${t('messages.stopFailed')}${error.response?.data?.message || error.message}`);
      }
    } finally {
      setActionLoading(false);
    }
  };

  const handleRestartVM = async (record) => {
    if (actionLoading) { return; }
    setActionLoading(true);
    setPopOpen('');
    try {
      const res = await restartVirtualMachine(record.metadata.namespace, record.metadata.name);
      if (res.status === ResponseCode.OK) {
        messageApi.success(t('messages.restartSuccess'));
        getList();
      }
    } catch (error) {
      if (error.response?.status === ResponseCode.Forbidden) {
        forbiddenMsg(messageApi, error, t);
      } else {
        messageApi.error(`${t('messages.restartFailed')}${error.response?.data?.message || error.message}`);
      }
    } finally {
      setActionLoading(false);
    }
  };

  const handleDelete = (record) => {
    setPopOpen('');
    setDelModalOpen(true);
    setDelName(record.metadata.name);
    setIsDelCheck(false);
    setCurrentNamespace(record.metadata.namespace);
  };

  const handleDelCancel = () => {
    setDelModalOpen(false);
    setDelName('');
    setCurrentNamespace('');
  };

  const handleDelConfirm = async () => {
    try {
      const res = await deleteResourceInstance({ ...prefixObj, namespace: currentNamespace }, delName);
      if (res.status === ResponseCode.OK) {
        showSuccessThen(messageApi, t('messages.deleteSuccess', { ns: 'common' }), () => {
          setDelModalOpen(false);
          setDelName('');
          setCurrentNamespace('');
          setIsDelCheck(false);
          getList();
        });
      }
    } catch (error) {
      if (error.response.status === ResponseCode.Forbidden) {
        forbiddenMsg(messageApi, error, t);
      } else {
        messageApi.error(t('messages.deleteFailed', { ns: 'common', message: error.response?.data?.message }));
      }
    }
  };

  const handleDelCheckFn = (e) => {
    setIsDelCheck(e.target.checked);
  };

  useEffect(() => {
    getList();
  }, [getList]);

  return (
    <Fragment>
      <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
        {contextHolder}
      </div>
      <div className="container_margin_box">
        <Form className="searchForm form_padding_bottom" form={form}>
          <Form.Item name="name" className="search_input">
            <Input.Search placeholder={t('searchPlaceholder', { ns: 'common', name: t('searchName.instance', { ns: 'common' }) })} onSearch={() => handleSearch()} autoComplete="off" />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button
                className="primary_btn borderRadiusRight6"
                onClick={() => history.push({
                  pathname: `/${containerRouterPrefix}/customResourceDefinition/${solveEncodePath(crdDetailData.metadata.name)}/createCR`,
                  state: { ...prefixObj, namespace: '', ...yamlExampleObj },
                })}>
                {t('actions.create')}
              </Button>
              <Button icon={<SyncOutlined />} onClick={handleReset} className="reset_btn" style={{ marginLeft: '16px' }}></Button>
            </Space>
          </Form.Item>
        </Form>
        <div className="tab_table_flex cluster_container_height">
          <Table
            className="table_padding"
            loading={loading}
            columns={columns}
            dataSource={list}
            rowKey={(record) => record.metadata.uid ?? record.metadata.name}
            pagination={{
              className: 'page',
              current: page,
              showTotal: (total) => t('pagination.total', { ns: 'common', total }),
              showSizeChanger: true,
              showQuickJumper: true,
              pageSizeOptions: [10, 20, 50],
              onChange: page => setPage(page),
            }}
            scroll={{ x: 1280 }}
          />
        </div>
        <DeleteInfoModal
          title={t('modal.deleteInstanceTitle')}
          open={delModalOpen}
          cancelFn={handleDelCancel}
          content={[
            t('modal.deleteInstanceWarning'),
            t('modal.deleteInstanceConfirm', { name: delName }),
          ]}
          isCheck={isDelCheck}
          showCheck={true}
          checkFn={handleDelCheckFn}
          confirmFn={handleDelConfirm} />
      </div>
    </Fragment>
  );
}