/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import {
  Table,
  Space,
  Popover,
  Button,
  Input,
  Modal,
  Pagination,
  message,
  Form,
  Radio,
  Tooltip,
} from 'antd';
import { MoreOutlined, InfoCircleFilled } from '@ant-design/icons';
import '@/styles/applicationMarket/index.less';
import '@/styles/pages/helm.less';
import { useEffect, useRef, useState, useStore } from 'openinula';
import { createHelmRepo } from '@/api/applicationMarketApi';
import {
  getHelmRepoList,
  deleteHelmRepo,
  synchronizationHelmRepo,
  updateHelmRepo,
  getsynchronizationStatus,
} from '@/api/applicationMarketApi';
import { Link } from 'inula-router';
import { containerRouterPrefix } from '@/constant';
import local from '@/assets/images/local.png';
import authority from '@/assets/images/authority.png';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { ResponseCode } from '@/common/constants';
import { showSuccessThen } from '@/tools/utils';
import { useTranslation } from 'react-i18next';
import useButtonLoading from '@/hooks/useButtonLoading';
const { Search } = Input;

const limitSearch = 53;
export default function WarehouseConfiguration() {
  const { t } = useTranslation('appMarket');
  const [popOpen, setPopOpen] = useState(false);
  const [total, setTotal] = useState(0);
  const [openAdd, setOpenAdd] = useState(false);
  const [dataSource, setDataSource] = useState([]);
  const [repoName, setRepoName] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [limit, setLimit] = useState(10);
  const [formType, setFormType] = useState('add');
  const [form] = Form.useForm();
  const [isShowWare, setIsShowWare] = useState(true);
  const [loading, setLoading] = useButtonLoading();
  const [formRadioWare, setFormRadioWare] = useState('2');
  const syncIntervalRef = useRef(null);

  useEffect(() => () => clearInterval(syncIntervalRef.current), []);
  const [isRepoDelCheck, setIsRepoDelCheck] = useState(false);
  const [repoDelModalOpen, setRepoDelModalOpen] = useState(false);
  const [deleteName, setDeleteName] = useState('');
  const [messageApi, contextHolder] = message.useMessage();
  const themeStore = useStore('theme');
  const [noneModalOpenWare, setNoneModalOpenWare] = useState(false);
  const [isNoneDelCheckWare, setIsNoneDelCheckWare] = useState(false);
  const [unsafeModalOpen, setUnsafeModalOpen] = useState(false);
  const [certModalOpen, setCertModalOpen] = useState(false);
  const [isCertCheckWare, setIsCertCheckWare] = useState(false);
  const [sortType, setSortType] = useState('');
  const [sortOrder, setSortOrder] = useState('');
  const [syncingRepo, setSyncingRepo] = useState('');

  // 自定义仓库名称校验函数
  const validateRepoName = async (_rule, value) => {
    const regex = /^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$/;
    if (value && !regex.test(value)) {
      return Promise.reject(new Error(t('messages.repoNamePattern')));
    } else {
      return Promise.resolve();
    }
  };

  const handleDelete = async (record) => {
    setPopOpen('');
    setDeleteName(record.name);
    setRepoDelModalOpen(true);
  };
  function handleDelRepoCancel() {
    setDeleteName('');
    setRepoDelModalOpen(false);
    setIsRepoDelCheck(false);
  }
  const handleEdit = (record, type) => {
    setPopOpen('');
    form.setFieldValue('name', record.name);
    form.setFieldValue('url', record.url);
    setFormType(type);
    setOpenAdd((open) => !open);
  };

  function handleCreateWareHouse(type) {
    setPopOpen('');
    form.setFieldValue('name', '');
    form.setFieldValue('url', '');
    setFormType(type);
    setOpenAdd((open) => !open);
  }

  function handleCancel() {
    setOpenAdd((open) => !open);
    setIsShowWare(true);
    setIsNoneDelCheckWare(false);
    setLoading(false);
    form.resetFields();
    setFormRadioWare('2');
    form.setFieldsValue({
      username: '',
      password: '',
    });
  }

  const handleRepoCheckFn = (e) => {
    setIsRepoDelCheck(e.target.checked);
  };

  const handleSynchronization = async (repo) => {
    setSyncingRepo(repo);
    const timeout = 2 * 60 * 1000;
    try {
      if (!repo) {
        return;
      }
      await synchronizationHelmRepo(repo);
      checkUpdateStatus(timeout, repo);
    } catch (error) {
      if (error?.response?.status === ResponseCode.Forbidden) {
        setSyncingRepo('');
        setPopOpen('');
        messageApi.error(t('messages.noPermission'));
      } else if (error?.response?.status === ResponseCode.BadRequest && error?.response?.data?.msg?.includes('syncronizing in progress')) {
        checkUpdateStatus(timeout);
      } else {
        setSyncingRepo('');
        setPopOpen('');
        messageApi.error(t('messages.fetchDataError', { ns: 'application', message: error.response?.data?.msg }));
      }
    }
  };

  const checkUpdateStatus = (timeout, repo) => {
    if (!repo) {
      return;
    }
    const startTime = Date.now();
    let inFlight = false;
    const intervalld = setInterval(async () => {
      if (inFlight) {
        return;
      }
      inFlight = true;
      try {
        const currentTime = Date.now();
        if (currentTime - startTime >= timeout) {
          clearInterval(intervalld);
          setSyncingRepo('');
          setPopOpen('');
          messageApi.error(t('messages.syncFailed', { repo }));
          return;
        }
        try {
          const res = await getsynchronizationStatus(repo);
          if (res.data.msg === 'complete') {
            setSyncingRepo('');
            setPopOpen('');
            messageApi.success(t('messages.localSyncSuccess', { repo }));
            clearInterval(intervalld);
          }
          if (res.data.msg === 'failed') {
            setSyncingRepo('');
            setPopOpen('');
            messageApi.error(t('messages.localSyncFailed'));
            clearInterval(intervalld);
          }
        } catch (error) {
          setSyncingRepo('');
          setPopOpen('');
          messageApi.error(error.response?.data?.msg);
          clearInterval(intervalld);
        }
      } finally {
        inFlight = false;
      }
    }, 3000);
    syncIntervalRef.current = intervalld;
  };

  const columns = [
    {
      title: t('warehouse.repoName'),
      dataIndex: 'name',
      sorter: true,
      render: (_, record) => (
        <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
          <Link
            to={`/${containerRouterPrefix}/appMarket/stash/wareHouse/stashDetails/${record.name}`}
          >
            {record.name}
          </Link>
          <div>
            <Tooltip title={t('warehouse.officialRepo')}>
              {record.name === 'openFuyao' && <img src={authority} />}
            </Tooltip>
            <Tooltip title={t('warehouse.localRepo')}>
              {record.name === 'local' && <img src={local} />}
            </Tooltip>
          </div>
        </div>
      ),
    },
    {
      title: t('warehouse.url'),
      dataIndex: 'url',
    },
    {
      title: t('warehouse.operation'),
      key: 'handle',
      fixed: 'right',
      width: 100,
      render: (_, record) => (
        <Space>
          <Popover
            placement='bottom'
            className='warehouseConfiguration'
            content={
              <div className='pop_modal'>
                <Button
                  type='link'
                  loading={syncingRepo === record.name}
                  onClick={() => handleSynchronization(record.name)}
                  className='warehouseConfiguration'
                >
                  {t('warehouse.syncRepo')}
                </Button>
                <Button
                  type='link'
                  disabled={
                    record.name === 'openFuyao' || record.name === 'local' || !!syncingRepo
                  }
                  onClick={() => handleEdit(record, 'edit')}
                >
                  {t('warehouse.editRepo')}
                </Button>
                <Button
                  type='link'
                  disabled={
                    record.name === 'openFuyao' || record.name === 'local' || !!syncingRepo
                  }
                  onClick={() => handleDelete(record)}
                >
                  {t('warehouse.deleteRepo')}
                </Button>
              </div>
            }
            trigger='click'
            open={popOpen === record.name}
            onOpenChange={(newOpen) =>
              newOpen ? setPopOpen(record.name) : setPopOpen('')
            }
          >
            <MoreOutlined className='common_antd_icon primary_color warehouseConfiguration' />
          </Popover>
        </Space>
      ),
    },
  ];

  const handleChange = (page, pageSize) => {
    setCurrentPage(page);
    setLimit(pageSize);
  };

  const handleInput = (e) => {
    if (e.target.value.length >= limitSearch) {
      messageApi.error(t('market.searchLimit', { limit: limitSearch }));
      return;
    }
    setRepoName(e.target.value);
  };

  const handleSearch = async (e) => {
    try {
      const res = await getHelmRepoList(e, '', currentPage, limit);
      setDataSource(res.data?.data?.items ?? []);
      setTotal(res.data?.data?.totalItems ?? 0);
    } catch (error) {
      messageApi.error(t('messages.fetchDataError', { ns: 'application', message: error.response?.data?.msg }));
    }
  };

  const getHelmRepoListInfo = async () => {
    try {
      const res = await getHelmRepoList(repoName, sortType, currentPage, limit, sortOrder);
      setDataSource(res.data?.data?.items ?? []);
      setTotal(res.data?.data?.totalItems ?? 0);
    } catch (error) {
      messageApi.error(t('messages.fetchDataError', { ns: 'application', message: error.response?.data?.msg }));
    }
  };

  function handleChangeRadioWare(event) {
    setFormRadioWare(event.target.value);
    if (event.target.value === '1') {
      setNoneModalOpenWare(true);
    } else {
      setIsShowWare(true);
    }
  }

  const handleDelNoneWare = () => {
    setNoneModalOpenWare(false);
    setIsNoneDelCheckWare(false);
    setFormRadioWare('2');
  };

  const handleNoneCheckFnWare = (e) => {
    setIsNoneDelCheckWare(e.target.checked);
  };

  const confirmNoneFnWare = () => {
    setIsShowWare(false);
    setNoneModalOpenWare(false);
  };

  const onFinishWare = async () => {
    try {
      await form.validateFields();
    } catch {
      return;
    }
    try {
      setLoading(true);

      // 判断URL协议
      let reg = /(?<urlPrefix>https?):\/\/[\w.]+\/?\S*/i;
      const match = reg.exec(form.getFieldsValue().url);
      if (match) {
        if (match.groups.urlPrefix === 'http') {
          setUnsafeModalOpen(true);
        } else if (match.groups.urlPrefix === 'https') {
          setCertModalOpen(true);
        }
      } else {
        handleContinueAddRepo();
      }
    } catch (error) {
      setLoading(false);
      messageApi.error(t('messages.fetchDataError', { ns: 'application', message: error.response?.data?.msg }));
    }
  };

  const handleContinueAddRepo = async () => {
    let _data = {
      name: form.getFieldsValue().name,
      url: form.getFieldsValue().url,
      username: form.getFieldsValue().username || '',
      password: form.getFieldsValue().password || '',
    };
    let encoder = new TextEncoder();
    if (formType === 'add') {
      _data = Object.assign({}, _data, {
        username: Object.values(encoder.encode(_data.username)),
        password: Object.values(encoder.encode(_data.password)),
      });
      try {
        await createHelmRepo(_data);
        await showSuccessThen(messageApi, t('messages.createSuccess'), () => {
          getHelmRepoListInfo();
          setIsNoneDelCheckWare(false);
          handleCancel();
        });
      } catch (error) {
        if (error.response.status === ResponseCode.Forbidden) {
          messageApi.error(t('messages.noPermission'));
        } else {
          messageApi.error(error.response?.data.msg);
        }
      }
    } else {
      _data = Object.assign({}, _data, {
        username: Object.values(encoder.encode(_data.username)),
        password: Object.values(encoder.encode(_data.password)),
      });
      try {
        await updateHelmRepo(_data);
        await showSuccessThen(messageApi, t('messages.updateSuccess'), () => {
          getHelmRepoListInfo();
          handleCancel();
        });
      } catch (error) {
        if (error.response.status === ResponseCode.Forbidden) {
          messageApi.error(t('messages.noPermission'));
        } else {
          messageApi.error(error.response?.data?.msg);
        }
      }
    }
    setLoading(false);
  };

  const deleteReopFn = async () => {
    try {
      const res = await deleteHelmRepo(deleteName);
      if (res.status === ResponseCode.OK) {
        showSuccessThen(messageApi, t('messages.deleteSuccess', { name: deleteName }), () => {
          setRepoDelModalOpen(false);
          setIsRepoDelCheck(false);
          if (limit !== 10 || currentPage !== 1) {
            handleChange(1, 10);
          } else {
            getHelmRepoListInfo();
          }
        });
      }
    } catch (error) {
      if (error.response.status === ResponseCode.Forbidden) {
        messageApi.error(t('messages.noPermission'));
      } else {
        messageApi.error(error.response?.data?.message);
      }
    }
  };

  const handleUnsafeContinue = () => {
    setUnsafeModalOpen(false);
    handleContinueAddRepo();
  };

  const handleUnSafeConfirm = () => {
    setUnsafeModalOpen(false);
    setLoading(false);
  };

  const handleCertCheckFn = (e) => {
    setIsCertCheckWare(e.target.checked);
  };

  const handleCertCancel = () => {
    setCertModalOpen(false);
    setIsCertCheckWare(false);
    setLoading(false);
  };

  const handleCertContinue = () => {
    setCertModalOpen(false);
    handleContinueAddRepo();
  };

  const handleChange1 = (pagination, filters, sorter) => {
    if (Object.keys(sorter).length) {
      if (sorter.field === 'name') {
        setSortType('name');
        if (sorter.order) {
          setSortOrder(sorter.order === 'ascend' ? 'asc' : 'desc');
        } else {
          setSortOrder('asc');
        }
      } else {
        setSortType('');
        setSortOrder('');
      }
    }
  };

  useEffect(() => {
    getHelmRepoListInfo();
  }, [repoName, currentPage, limit, sortType, sortOrder]);

  return (
    <>
      <div className='prompt' style={{ backgroundColor: themeStore.$s.theme !== 'light' ? 'rgba(119,174,247, 0.12)' : undefined, border: '1px solid rgba(119,174,247, 0.50)' }}>
        <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
          {contextHolder}
        </div>
        <InfoCircleFilled className='prompt-icon' />
        <span>{t('warehouse.description')}</span>
      </div>
      <div style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff', margin: '0 32px' }}>
        <div className='ware_house_search'>
          <Search
            placeholder={t('warehouse.searchPlaceholder')}
            style={{ width: '300px', height: '32px' }}
            onChange={handleInput}
            onSearch={handleSearch}
            maxLength={limitSearch}
          />
          <div style={{ display: 'flex', gap: '23px' }}>
            <Button className='primary_btn' onClick={() => handleCreateWareHouse('add')}>
              {t('warehouse.addRepo')}
            </Button>
          </div>
        </div>
        <Table
          className='table_padding'
          columns={columns}
          dataSource={dataSource}
          pagination={false}
          onChange={handleChange1}
        />
        <Pagination
          className='page'
          showTotal={(curTotal) => t('pagination.total', { ns: 'common', total: curTotal })}
          total={total}
          showSizeChanger
          showQuickJumper
          pageSize={limit}
          current={currentPage}
          pageSizeOptions={[10, 20, 50]}
          onChange={handleChange}
        ></Pagination>
        <DeleteInfoModal
          title={t('modal.securityRiskTitle')}
          open={unsafeModalOpen}
          cancelFn={handleUnSafeConfirm}
          cancelText={t('warehouse.stop')}
          content={[
            t('warehouse.httpWarning'),
          ]}
          confirmText={t('warehouse.continue')}
          confirmFn={handleUnsafeContinue}
        />
        <Modal
          className={themeStore.$s.theme === 'dark' ? 'dark_box' : ''}
          open={openAdd}
          getContainer={false}
          title={formType === 'add' ? t('warehouse.addRepo') : t('warehouse.editRepo')}
          width={720}
          onCancel={handleCancel}
          footer={[
            <Button key='back' className='cancel_btn' onClick={handleCancel}> {t('modal.cancel')}</Button>,
            <Button key='submit' loading={loading} className='primary_btn' onClick={onFinishWare}>{t('actions.confirm', { ns: 'common' })}</Button>,
          ]}
        >
          <Form
            layout='vertical'
            form={form}
            onFinish={onFinishWare}
            name='form_in_modal'
          >
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                minHeight: '200px',
                flexDirection: 'column',
              }}
            >
              <div
                style={{
                  flexDirection: 'column',
                  display: 'flex',
                }}
              >
                <div className='form_container'>
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      flexDirection: 'column',
                    }}
                  >
                    <Form.Item
                      name='name'
                      label={t('warehouse.repoName')}
                      rules={[{ required: true, message: t('warehouse.repoNameRequired') }, { validator: validateRepoName }]}
                    >
                      <Input placeholder={t('warehouse.repoNamePlaceholder')} disabled={formType === 'edit' ? true : false} />
                    </Form.Item>
                    <Form.Item name='url' label={t('warehouse.url')}
                      rules={[{ required: true, message: t('warehouse.repoUrlRequired') }]}
                    >
                      <Input placeholder={t('warehouse.repoUrlPlaceholder')} />
                    </Form.Item>
                  </div>
                </div>
                <div>
                  <div style={{ marginBottom: '15px' }}>{t('warehouse.authMethod')}</div>
                  <div style={{ display: 'flex', gap: '40px' }}>
                    <Radio.Group onChange={handleChangeRadioWare} value={formRadioWare}>
                      <Radio className='custom-checkbox noneUser' value='1'>
                        {t('warehouse.none')}
                      </Radio>
                      <Radio className='custom-checkbox users' value='2'>
                        {t('warehouse.usernamePassword')}
                      </Radio>
                    </Radio.Group>
                  </div>
                </div>
                {isShowWare && (
                  <div
                    className='defalutClass'
                    style={{
                      marginTop: '20px',
                      gap: '20px',
                      display: 'flex',
                      justifyContent: 'space-between',
                    }}
                  >
                    <Form.Item
                      style={{ width: '50%' }}
                      name='username'
                      className='defalutClass'
                      label={t('warehouse.username')}
                      rules={[{ required: true, message: t('warehouse.usernameRequired') }]}
                    >
                      <Input placeholder={t('warehouse.usernamePlaceholder')} />
                    </Form.Item>
                    <Form.Item
                      name='password'
                      label={t('warehouse.password')}
                      rules={[{ required: true, message: t('warehouse.passwordRequired') }]}
                      style={{ width: '50%' }}
                    >
                      <Input.Password placeholder={t('warehouse.passwordPlaceholder')} type='password' onCopy={(event) => event.preventDefault()} />
                    </Form.Item>
                  </div>
                )}
              </div>
            </div>
          </Form>
        </Modal>
        <DeleteInfoModal
          cancelFn={handleDelRepoCancel}
          title={t('modal.securityRiskTitle')}
          open={repoDelModalOpen}
          content={[
            t('warehouse.deleteConfirm'),
            t('warehouse.confirmDelete', { name: deleteName }),
          ]}
          checkFn={handleRepoCheckFn}
          isCheck={isRepoDelCheck}
          showCheck={true}
          confirmFn={deleteReopFn} />
        <DeleteInfoModal
          title={t('modal.securityRiskTitle')}
          open={noneModalOpenWare}
          cancelFn={handleDelNoneWare}
          content={[
            t('warehouse.noneSecurityWarning'),
            t('warehouse.confirmConnect'),
          ]}
          isCheck={isNoneDelCheckWare}
          checkboxText={t('modal.riskAcknowledge')}
          showCheck={true}
          confirmText={t('actions.confirm', { ns: 'common' })}
          checkFn={handleNoneCheckFnWare}
          confirmFn={confirmNoneFnWare}
        />
        <DeleteInfoModal
          title={t('warehouse.certConfigTitle')}
          open={certModalOpen}
          cancelFn={handleCertCancel}
          cancelText={t('modal.cancel')}
          content={[
            t('warehouse.certConfigContent1'),
            t('warehouse.certConfigContent2'),
            t('warehouse.certConfigContent3'),
          ]}
          isCheck={isCertCheckWare}
          checkboxText={t('warehouse.certAcknowledge')}
          showCheck={true}
          confirmText={t('warehouse.continue')}
          checkFn={handleCertCheckFn}
          confirmFn={handleCertContinue}
        />
      </div>
    </>
  );
}
