/* Copyright(c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
    EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON - INFRINGEMENT,
        MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import DetailsContent from '@/pages/applicationMarket/component/DetailsContent';
import '@/styles/pages/stash.less';
import {
  Input,
  Button,
  Table,
  Pagination,
  message,
  Popover,
  Space,
  Modal,
  Form,
  Radio,
  Tooltip,
} from 'antd';
import {
  getHelmRepoDetail,
  synchronizationHelmRepo,
  deleteHelmRepo,
  getsynchronizationStatus,
  updateHelmRepo,
  getNewHelmChartList,
} from '@/api/applicationMarketApi';
import { DownOutlined } from '@ant-design/icons';
import { useEffect, useMemo, useRef, useState, useStore } from 'openinula';
import { useParams, useHistory } from 'inula-router';
import Dayjs from 'dayjs';
import { containerRouterPrefix } from '@/constant';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { showSuccessThen } from '@/tools/utils';
import { ResponseCode } from '@/common/constants';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import applicationExtendDefault from '@/assets/images/application_extend_default.png';
import { useTranslation } from 'react-i18next';
import useButtonLoading from '@/hooks/useButtonLoading';

const { Search } = Input;
const isDelete = (type) => {
  switch (type) {
    case 'openFuyao':
      return false;
    case 'local':
      return false;
    default:
      return true;
  }
};

const isExtension = (arr, t) => {
  const extension = arr.some(item => item === 'openfuyao-extension');
  return extension ? t('table.extension') : t('table.application');
};

export default function StashDetails() {
  const history = useHistory();
  const { repo } = useParams();
  const [dataSource, setDataSource] = useState([]);
  const [chartName, setChartName] = useState('');
  const [total, setTotal] = useState(0);
  const [basicInfo, setBasicInfo] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [limit, setLimit] = useState(10);
  const [syncLoading, setSyncLoading] = useButtonLoading();
  const [podPopOpen, setPodPopOpen] = useState(false);
  const [updateInitValue, setUpdateInitValue] = useState({ name: '', url: '' });
  const [openAdd, setOpenAdd] = useState(false);
  const [form] = Form.useForm();
  const [isShow, setIsShow] = useState(true);
  const [loading, setLoading] = useButtonLoading();
  const [formRadio, setFormRadio] = useState('2');
  const [messageApi, contextHolder] = message.useMessage();
  const [isRepoDelCheck, setIsRepoDelCheck] = useState(false);
  const [repotDelModalOpen, setRepoDelModalOpen] = useState(false);
  const themeStore = useStore('theme');
  const [noneModalOpen, setNoneModalOpen] = useState(false);
  const [isNoneDelCheck, setIsNoneDelCheck] = useState(false);
  const { t, i18n } = useTranslation('stash');

  const [sortType, setSortType] = useState('');

  const [filterType, setfilterType] = useState([]);

  const [sortOrder, setSortOrder] = useState('');
  const repoType = (type) => {
    switch (type) {
      case 'openfuyao':
        return t('repoType.official');
      case 'local':
        return t('repoType.local');
      default:
        return t('repoType.userAdded');
    }
  };
  const limitSearch = 53;
  const contentArray = useMemo(() => [
    {
      title: t('content.repoName'),
      value: basicInfo?.spec?.displayName || '--',
    },
    {
      title: t('content.repoType'),
      value: repoType(basicInfo?.metadata?.name),
    },
    {
      title: t('content.repoUrl'),
      value: basicInfo?.spec?.url || '--',
      copyable: true,
    },
    {
      title: t('content.lastSyncTime'),
      value: basicInfo?.metadata?.annotations?.['openfuyao.com/modificationTimestamp']
        ? Dayjs(basicInfo.metadata.annotations['openfuyao.com/modificationTimestamp']).format(
          'YYYY-MM-DD HH:mm:ss',
        )
        : '--',
    },
    {
      title: t('content.createTime'),
      value: basicInfo?.metadata?.creationTimestamp ? Dayjs(basicInfo.metadata.creationTimestamp).format(
        'YYYY-MM-DD HH:mm:ss',
      ) : '--',
    },
  ], [basicInfo, t, i18n.language]);

  const columns = useMemo(() => [
    {
      title: t('table.name'),
      sorter: true,
      key: 'name',
      render: (_, record) => (
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div><img src={record.icon || applicationExtendDefault} style={{ width: '40px', height: '40px' }} /></div>
          <div>
            <Tooltip title={record.name}>
              <div className='clamp_1'>{record.name}</div>
            </Tooltip>
            <Tooltip title={record.description}>
              <div style={{ color: '#89939b', width: '200px' }} className='clamp_1'>{record.description}</div>
            </Tooltip>
          </div>
        </div>
      ),
      width: 450,
    },
    {
      title: t('table.type'),
      key: 'keywords',
      filters: [
        { text: t('table.application'), value: 'application' },
        { text: t('table.extension'), value: 'openfuyao-extension' },
      ],
      onFilter: (value, record) => {
        const type = record.keywords?.some(item => item === 'openfuyao-extension') ? 'openfuyao-extension' : 'application';
        return type === value;
      },
      filterMultiple: false,
      render: (_, record) =>
        record.keywords ? isExtension(record.keywords, t) : t('table.application'),
    },
    {
      title: t('table.latestVersion'),
      dataIndex: 'version',
    },
    {
      title: t('table.createTime'),
      key: 'created',
      sorter: true,
      render: (_, record) => Dayjs(record.created).format('YYYY-MM-DD HH:mm'),
    },
  ], [t]);

  const syncIntervalRef = useRef(null);

  useEffect(() => {
    return () => clearInterval(syncIntervalRef.current);
  }, []);

  const getHelmRepoDetailInfo = async () => {
    try {
      const res = await getHelmRepoDetail(repo);
      let _updateValue = {};
      _updateValue.name = res.data.data.metadata.name;
      _updateValue.url = res.data.data.spec.url;
      setUpdateInitValue(_updateValue);
      setBasicInfo(res.data.data);
    } catch (error) {
      if (error.response?.status !== ResponseCode.OK) {
        messageApi.error(t('messages.fetchFailed', { ns: 'common', message: error.response?.data?.message }));
      }
    }
  };
  const getNewHelmChartListInfo = async (chartNameInput = chartName) => {
    const params = {
      sortType,
      filterType,
      sortOrder,
    };
    try {
      const res = await getNewHelmChartList(repo, chartNameInput, params, currentPage, limit);
      setTotal(res.data.data.totalItems);
      const arr = res.data.data.items.map((el) => el.metadata);
      setDataSource(arr);
    } catch (error) {
      messageApi.error(error.response.data.msg);
    }
  };

  const handleInput = (e) => {
    if (e.target.value.length >= limitSearch) {
      message.error(t('message.searchLimit', { limit: limitSearch }));
    }
  };

  const handleSearch = (searchName) => {
    setChartName(searchName);
    setCurrentPage(1);
  };

  const handleChange = (page, pageSize) => {
    setCurrentPage(page);
    setLimit(pageSize);
  };

  const handleSynchronization = async () => {
    setSyncLoading(true);
    const timeout = 5 * 60 * 1000;
    try {
      await synchronizationHelmRepo(repo);
      checkUpdateStatus(timeout);
    } catch (error) {
      if (error.response.status === ResponseCode.Forbidden) {
        setSyncLoading(false);
        messageApi.error(t('message.noPermission'));
      } else {
        if (error.response.data.msg.includes('syncronizing in progress')) {
          setSyncLoading(true);
          checkUpdateStatus(timeout);
        } else {
          messageApi.error(error.response.data.msg);
        }
      }
    }
  };

  const checkUpdateStatus = (timeout) => {
    const startTime = Date.now();
    const intervalld = setInterval(async () => {
      const currentTime = Date.now();
      if (currentTime - startTime >= timeout) {
        clearInterval(intervalld);
        messageApi.error(t('message.syncFailed', { repo }));
        setSyncLoading(false);
        return;
      }
      try {
        const res = await getsynchronizationStatus(repo);
        if (res.data.msg === 'complete') {
          messageApi.success(t('message.syncSuccess', { repo }));
          clearInterval(intervalld);
          getNewHelmChartListInfo();
          setSyncLoading(false);
        }
        if (res.data.msg === 'in_progress') {
          setSyncLoading(true);
        }
      } catch (e) {
        messageApi.error(e.response.data.msg);
        clearInterval(intervalld);
        setSyncLoading(false);
      }
    }, 3000);
    syncIntervalRef.current = intervalld;
  };

  const handlePodPopOpenChange = (open) => {
    setPodPopOpen(open);
  };
  const updateRepo = (data) => {
    form.setFieldValue('name', data.name);
    form.setFieldValue('url', data.url);
    setOpenAdd((open) => !open);
    setPodPopOpen(false);
  };
  function handleCancel() {
    setOpenAdd((open) => !open);
    form.resetFields();
    setFormRadio('2');
    setIsShow(true);
  }
  function handleChangeRadio(e) {
    setFormRadio(e.target.value);
    if (e.target.value === '1') {
      setNoneModalOpen(true);
    } else {
      setIsShow(true);
    }
  }

  const handleDelNone = () => {
    setNoneModalOpen(false);
    setIsNoneDelCheck(false);
    setFormRadio('2');
  };
  const handleNoneCheckFn = (e) => {
    setIsNoneDelCheck(e.target.checked);
  };

  const confirmNoneFn = () => {
    setIsShow(false);
    setNoneModalOpen(false);
  };
  const onFinish = async () => {
    setLoading(true);
    let _data = {
      name: form.getFieldsValue().name,
      url: form.getFieldsValue().url,
      username: form.getFieldsValue().username,
      password: form.getFieldsValue().password,
    };
    let encoder = new TextEncoder();
    _data = Object.assign({}, _data, {
      username: Object.values(encoder.encode(_data.username || '')),
      password: Object.values(encoder.encode(_data.password || '')),
    });
    try {
      await updateHelmRepo(_data);
      messageApi.success(t('message.updateSuccess'));
      if (limit !== 10 || currentPage !== 1) {
        handleChange(1, 10);
      } else {
        getNewHelmChartListInfo();
      }
      getHelmRepoDetailInfo();
      setLoading(false);
      setIsShow(true);
      handleCancel();
    } catch (error) {
      if (error?.response?.status === ResponseCode.Forbidden) {
        messageApi.error(t('message.noPermission'));
      } else {
        messageApi.error(error?.response?.data?.msg);
      }
      setLoading(false);
    }
  };
  const handleRepoCheckFn = (e) => {
    setIsRepoDelCheck(e.target.checked);
  };
  const deleteRepo = () => {
    setPodPopOpen(false);
    setRepoDelModalOpen(true);
  };
  function handleDelRepoCancel() {
    setRepoDelModalOpen(false);
    setIsRepoDelCheck(false);
  }
  const deleteReopFn = async () => {
    try {
      const res = await deleteHelmRepo(updateInitValue.name);
      if (res?.status === ResponseCode.OK) {
        await showSuccessThen(messageApi, t('message.deleteSuccess', { name: updateInitValue.name }), () => {
          setRepoDelModalOpen(false);
          setIsRepoDelCheck(false);
          history.push(`/${containerRouterPrefix}/appMarket/stash`);
        });
      }
    } catch (error) {
      if (error?.response?.status === ResponseCode.Forbidden) {
        messageApi.error(t('message.noPermission'));
      } else {
        messageApi.error(error?.response?.data?.message);
      }
    }
  };

  const handleChange1 = (pagination, filters, sorter) => {
    if (filters?.keywords) {
      setCurrentPage(1);
      setfilterType([...Object.values(filters?.keywords)]);
    } else {
      setfilterType([]);
    }
    // 按照类型进行排序
    if (Object.keys(sorter).length) {
      setCurrentPage(1);
      if (sorter.order && (sorter.columnKey === 'created' || sorter.columnKey === 'name')) {
        setSortType(sorter.columnKey === 'created' ? 'time' : 'name');
        setSortOrder(sorter.order === 'ascend' ? 'asc' : 'desc');
      } else {
        setSortType('');
        setSortOrder('');
      }
    }
  };

  useEffect(() => {
    getHelmRepoDetailInfo();
  }, []);

  useEffect(() => {
    getNewHelmChartListInfo();
  }, [sortType, filterType, sortOrder, currentPage, chartName, limit]);

  return (
    <div className='child_content withBread_content'>
      <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
        {contextHolder}
      </div>
      <BreadCrumbCom
        items={[
          { title: t('module.wareHouseConfig'), disabled: true },
          { title: t('module.wareHouse'), path: `/${containerRouterPrefix}/appMarket/stash` },
          { title: t('module.wareHouseDetail') },
        ]}
      />
      <div className='ware_house_detail_header' style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
        <div style={{ fontSize: '20px', fontWeight: 'bold', color: themeStore.$s.theme !== 'light' && '#fff' }}>{repo}</div>
        <Popover placement='bottom'
          content={
            <Space className='column_pop'>
              <Button type='link' disabled={!isDelete(repo)} onClick={() => updateRepo(updateInitValue)}>{t('common.edit')}</Button>
              <Button type='link' disabled={!isDelete(repo)} onClick={() => deleteRepo()}>{t('common.delete')}</Button>
            </Space>
          }
          open={podPopOpen}
          onOpenChange={handlePodPopOpenChange}>
          <Button className='primary_btn'>{t('common.operation')} <DownOutlined className='small_margin_adjust' /></Button>
        </Popover>
      </div>
      <div className='stash_details_container'>
        <DetailsContent contentArray={contentArray} title={t('content.basicInfo')} />
        <div style={{ padding: '20px 32px 0px 32px', backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
          <div className='fz_16 fw_bold mg_b_20' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('details.appExtension')}</div>
          <div className='display_flex mg_b_20'>
            <Search
              placeholder={t('searchPlaceholder', { ns: 'common', name: t('searchName.chart', { ns: 'common' }) })}
              style={{ width: '300px', height: '32px' }}
              onChange={handleInput}
              onSearch={handleSearch}
              maxLength={limitSearch}
            />
            <Button
              className='cancel_btn'
              loading={syncLoading}
              onClick={handleSynchronization}
            >
              {t('common.sync')}
            </Button>
          </div>
          <Table dataSource={dataSource} columns={columns} pagination={false} onChange={handleChange1} />
          <Pagination
            className='page'
            showTotal={(curTotal) => t('pagination.total', { ns: 'common', total: curTotal })}
            showSizeChanger
            showQuickJumper
            total={total}
            current={currentPage}
            pageSize={limit}
            onChange={handleChange}
            pageSizeOptions={[10, 20, 50]}
          ></Pagination>
          <Modal
            open={openAdd}
            title={t('modal.editRepo')}
            width={720}
            onCancel={handleCancel}
            footer={[
              <Button key='back' className='cancel_btn' onClick={handleCancel}>{t('common.cancel')}</Button>,
              <Button key='submit' loading={loading} className='primary_btn' onClick={() => form.validateFields().then(() => onFinish()).catch(() => { })}>{t('common.confirm')}</Button>,
            ]}
          >
            <Form
              layout='vertical'
              form={form}
              name='form_in_modal'
              onFinish={onFinish}
            >
              <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', minHeight: '200px' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', marginBottom: '30px' }}>
                  <div className='form_container'>
                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                      <Form.Item
                        style={{ width: '100%' }}
                        name='name'
                        label={t('content.repoName')}
                      >
                        <Input placeholder={t('common.input')} disabled={true} />
                      </Form.Item>
                      <Form.Item name='url' label={t('content.repoUrl')} style={{ width: '100%' }}>
                        <Input placeholder={t('common.input')} />
                      </Form.Item>
                    </div>
                  </div>
                  <div>
                    <div style={{ marginBottom: '15px' }}>{t('modal.authMethod')}</div>
                    <div style={{ display: 'flex', gap: '40px' }}>
                      <Radio.Group onChange={handleChangeRadio} value={formRadio}>
                        <Radio className='custom-checkbox' value='1'>
                          {t('modal.none')}
                        </Radio>
                        <Radio className='custom-checkbox' value='2'>
                          {t('modal.usernamePassword')}
                        </Radio>
                      </Radio.Group>
                    </div>
                  </div>
                  {isShow && (
                    <div
                      style={{
                        display: 'flex',
                        gap: '20px',
                        justifyContent: 'space-between',
                      }}
                    >
                      <Form.Item
                        style={{ width: '50%' }}
                        name='username'
                        label={t('modal.username')}
                        rules={[{ required: true, message: t('modal.usernameRequired') }]}
                      >
                        <Input placeholder={t('modal.usernamePlaceholder')} />
                      </Form.Item>
                      <Form.Item name='password' label={t('modal.password')} style={{ width: '50%' }}
                        rules={[{ required: true, message: t('modal.passwordRequired') }]}
                      >
                        <Input.Password placeholder={t('modal.passwordPlaceholder')} type='password' onCopy={(e) => e.preventDefault()} />
                      </Form.Item>
                    </div>
                  )}
                </div>
              </div>
            </Form>
          </Modal>
          <DeleteInfoModal
            title={t('common.deleteTitle')}
            open={repotDelModalOpen}
            cancelFn={handleDelRepoCancel}
            content={[
              t('modal.deleteWarning'),
              t('modal.confirmDelete', { name: updateInitValue.name }),
            ]}
            isCheck={isRepoDelCheck}
            showCheck={true}
            checkFn={handleRepoCheckFn}
            confirmFn={deleteReopFn} />
          <DeleteInfoModal
            title={t('common.securityRisk')}
            open={noneModalOpen}
            cancelFn={handleDelNone}
            content={[
              t('modal.noneSecurityWarning'),
              t('modal.confirmConnect'),
            ]}
            isCheck={isNoneDelCheck}
            checkboxText={t('modal.riskAcknowledge')}
            showCheck={true}
            confirmText={t('common.confirm')}
            checkFn={handleNoneCheckFn}
            confirmFn={confirmNoneFn}
          />
        </div>
      </div>
    </div>
  );
}
