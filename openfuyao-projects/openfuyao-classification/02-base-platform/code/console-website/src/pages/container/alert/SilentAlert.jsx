/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { MoreOutlined } from '@ant-design/icons';
import { containerRouterPrefix } from '@/constant';
import { Link } from 'inula-router';
import { Select, Button, Table, Space, Popover, message } from 'antd';
import ToastMsg from '@/components/ToastMsg';
import { useEffect, useState, useStore, useCallback, useMemo } from 'openinula';
import dayjs from 'dayjs';
import { deleteSilentAlert, getSilentList, getAlertOptions, creatSilentAlert } from '@/api/containerAlertApi';
import { ResponseCode } from '@/common/constants';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { filterRepeatByid, filterRepeat } from '@/utils/common';
import UserStore from '@/store/userStore';
import EditSilentAlertModal from './EditSilentAlertModal';
import '@/styles/pages/alert.less';
import { sorterFirstAlphabet, showSuccessThen } from '@/tools/utils';
import { getTimeType } from '@/utils/common';
import { useTranslation } from 'react-i18next';
import useButtonLoading from '@/hooks/useButtonLoading';

let setTimeoutId = 0;
let searchString = '';

export default function SilentAlert() {
  const { t, i18n } = useTranslation('alert');
  const userStore = UserStore();
  const [messageApi, contextHolder] = message.useMessage();
  const [selectSilentName, setSelectSilentName] = useState('all');
  const [selectSilentLabel, setSelectSilentLabel] = useState([]);
  const [popOpen, setPopOpen] = useState('');
  const [formType, setFormType] = useState('');
  const [silentDelModalOpen, setSilentDelModalOpen] = useState(false);
  const [silentName, setSilentName] = useState('');
  const [silentId, setSilentId] = useState('');
  const [silentTableList, setSilentTableList] = useState([]);
  const [silentNameOptions, setSilentNameOptions] = useState([]);
  const [labelOptions, setLabelOptions] = useState([]);
  const [isSilentModalOpen, setIsSilentModalOpen] = useState(false);
  const themeStore = useStore('theme');
  const [isSilentDelCheck, setIsSilentDelCheck] = useState(false);
  const [filterAlertValue, setFilterAlertValue] = useState();
  const [filterAlertStatus, setFilterAlertStatus] = useState([]);
  const [searchLoading, setSearchLoading] = useButtonLoading();

  const defaultOptions = useMemo(() => [{
    label: t('level.all'),
    value: 'all',
  }], [t]);

  const resolveAlertName = (matchers) => {
    const matcherName = matchers.filter(subitem => subitem.name === 'alertname');
    if (matcherName.length > 0) { return matcherName[0].value; }
    if (matchers.length > 0) { return `${matchers[0].name}=${matchers[0].value}`; }
    return '-';
  };

  const [propsChildData, setPropsChildData] = useState();

  const changeSilentName = (value) => {
    setSelectSilentName(value);
  };

  const changeSilentLabel = (value) => {
    setSelectSilentLabel(value);
  };

  const resetSilentSearch = useCallback(() => {
    searchString = '';
    setSelectSilentLabel([]);
    setSelectSilentName('all');
    getSilentLists();
  }, []);

  const searchSilent = async () => {
    setSearchLoading(true);
    try {
      let searchString1 = '';
      let searchString2 = '';
      if (selectSilentName !== 'all') {
        if (selectSilentName.includes('=')) {
          searchString1 = '&filter=' + `${selectSilentName}`;
        } else {
          searchString1 = '&filter=alertname=' + `${selectSilentName}`;
        }
      }
      if (selectSilentLabel.length > 0) {
        selectSilentLabel.map(item => {
          searchString2 += ('&filter=' + `${item}`);
        });
      }
      searchString = searchString1 + searchString2;
      await getSilentLists(searchString);
    } finally {
      setSearchLoading(false);
    }
  };

  const handleEditSilent = (data) => {
    setFormType('edit');
    setSilentId(data.id);
    setIsSilentModalOpen(true);
    let labelList = [];
    data.matchers.map(item => {
      labelList.push({
        key: item.name,
        value: item.value,
      });
    });
    let time = dayjs(dayjs(data.endtime).format('YYYY-MM-DD HH:mm')).diff(dayjs(data.begintime).format('YYYY-MM-DD HH:mm')) / 1000;
    setPropsChildData({
      labelList,
      beginTime: dayjs(data.begintime),
      creators: data.creators,
      remarks: data.remarks,
      timeType: getTimeType(time).type,
      timeNum: getTimeType(time).num,
    });
  };

  function handleDeleteSilent(param) {
    setSilentName(param.name);
    setSilentId(param.id);
    setSilentDelModalOpen(true);
  }

  function handleDelSilentCancel() {
    setSilentDelModalOpen(false);
    setIsSilentDelCheck(false);
    setSilentName('');
  }

  const handleAlertTableChange = useCallback(
    (_pagination, filter, _sorter, extra) => {
      if (extra.action === 'filter') {
        setFilterAlertValue(filter.creators);
      }
    },
    [],
  );

  const silentColumns = useMemo(() => [
    {
      title: t('silentAlert.silentName'),
      dataIndex: 'name',
      key: 'name',
      sorter: (a, b) => sorterFirstAlphabet(a.name, b.name),
      render: (_, record) => (
        <Space size="middle" className="alertName">
          <Link to={{ pathname: `/${containerRouterPrefix}/alert/silentAlert/detail/${record.name}`, state: { currentdetail: record } }}>{record.name}</Link>
        </Space>
      ),
    },
    {
      title: t('table.silentStatus'),
      dataIndex: 'state',
      key: 'state',
      sorter: (a, b) => sorterFirstAlphabet(
        a.state === 'active' ? t('status.active') : t('status.pending'),
        b.state === 'active' ? t('status.active') : t('status.pending'),
      ),
      render: (_, record) => (
        <Space size="middle">
          <span className={record.state === 'active' ? 'alertActive' : 'alertPending'}></span>
          {record.state === 'active' ? t('status.active') : t('status.pending')}
        </Space>
      ),
    },
    {
      title: t('table.beginTime'),
      dataIndex: 'begintime',
      key: 'begintime',
      sorter: (a, b) => sorterFirstAlphabet(dayjs(a.begintime).format('YYYY-MM-DD HH:mm:ss'), dayjs(b.begintime).format('YYYY-MM-DD HH:mm:ss')),
      render: (_, record) => (
        <Space size="middle" className="">{dayjs(record.begintime).format('YYYY-MM-DD HH:mm')}</Space>
      ),
    },
    {
      title: t('table.endTime'),
      dataIndex: 'endtime',
      key: 'endtime',
      sorter: (a, b) => sorterFirstAlphabet(dayjs(a.endtime).format('YYYY-MM-DD HH:mm:ss'), dayjs(b.endtime).format('YYYY-MM-DD HH:mm:ss')),
      render: (_, record) => (
        <Space size="middle" className="">{dayjs(record.endtime).format('YYYY-MM-DD HH:mm')}</Space>
      ),
    },
    {
      title: t('table.creator'),
      dataIndex: 'creators',
      key: 'creators',
      filters: filterAlertStatus,
      filterMultiple: false,
      filteredValue: filterAlertValue ? [filterAlertValue] : [],
      onFilter: (value, record) => (record.creators).toLowerCase() === value.toLowerCase(),
      sorter: (a, b) => sorterFirstAlphabet(a.creators, b.creators),
      render: (_, record) => (
        <Space size="middle" className="">{record.creators}</Space>
      ),
    },
    {
      title: t('table.matchedAlerts'),
      dataIndex: 'alertNum',
      key: 'alertNum',
      sorter: (a, b) => sorterFirstAlphabet(a.alertNum, b.alertNum),
      render: (_, record) => (
        <Space size="middle" className="">{record.alertNum}</Space>
      ),
    },
    {
      title: t('table.remarks'),
      dataIndex: 'remarks',
      key: 'remarks',
      sorter: (a, b) => sorterFirstAlphabet(a.remarks, b.remarks),
      maxWidth: 200,
      render: (_, record) => (
        <Space size="middle" className="remarkWarp">{record.remarks}</Space>
      ),
    },
    {
      title: t('table.operation'),
      key: 'x',
      dataIndex: '',
      render: (_, record) => (
        <Space>
          <Popover
            placement='bottom'
            content={
              <div className='pop_modal deafultClass'>
                <Button type='link' onClick={() => handleEditSilent(record)}>
                  {t('actions.modify')}
                </Button>
                <Button type='link' onClick={() => handleDeleteSilent(record)}>
                  {t('actions.delete')}
                </Button>
              </div>
            }
            trigger='click'
            open={popOpen === record.key}
            onOpenChange={(newOpen) =>
              newOpen ? setPopOpen(record.key) : setPopOpen('')
            }
          >
            <MoreOutlined className='common_antd_icon primary_color' />
          </Popover>
        </Space>
      ),
    },
  ], [t, i18n.language, filterAlertStatus, filterAlertValue, popOpen]);

  const showSilentModal = () => {
    setFormType('add');
    setIsSilentModalOpen(true);
    setPropsChildData({
      beginTime: '',
      labelList: [{ key: '', value: '' }],
      remarks: '',
      creators: userStore.user.name,
      timeType: 'd',
      timeNum: 1,
    });
  };

  const handleChildClick = (data) => {
    let arr = data;
    editSilent(arr);
  };

  const cancelEdit = () => {
    setIsSilentModalOpen(false);
  };

  const editSilent = async (data) => {
    if (formType === 'edit') {
      try {
        const res1 = await deleteSilentAlert(silentId);
        if (res1.status === ResponseCode.OK) {
          setSilentId('');
          const res = await creatSilentAlert(data);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editSilentSuccess'), () => {
              setIsSilentModalOpen(false);
              resetSilentSearch();
              cancelEdit();
            });
          } else {
            messageApi.error(t('messages.editSilentFailed'));
          }
        }
      } catch (error) {
        if (error.response?.status === ResponseCode.Forbidden) {
          messageApi.error(t('messages.noPermission'));
        } else {
          messageApi.error(t('messages.editSilentFailed'));
        }
      }
    } else {
      try {
        const res = await creatSilentAlert(data);
        if (res.status === ResponseCode.OK) {
          showSuccessThen(messageApi, t('messages.createSilentSuccess'), () => {
            setIsSilentModalOpen(false);
            resetSilentSearch();
            cancelEdit();
          });
        }
      } catch (error) {
        if (error.response?.status === ResponseCode.Forbidden) {
          messageApi.error(t('messages.noPermission'));
        } else {
          let errorData;
          if (typeof (error.response?.data) === 'string') {
            errorData = error.response.data;
          } else {
            errorData = error.response?.data?.message ?? error.message;
          }
          messageApi.error(t('messages.createSilentFailed', { reason: errorData }));
        }
      }
    }
  };

  const deleteSilent = async () => {
    try {
      const res = await deleteSilentAlert(silentId);
      if (res.status === ResponseCode.OK) {
        showSuccessThen(messageApi, t('messages.deleteSuccess', { name: silentName }), () => {
          setSilentDelModalOpen(false);
          setIsSilentDelCheck(false);
          setSilentName('');
          getSilentLists();
        });
      }
    } catch (error) {
      if (error.response?.status === ResponseCode.Forbidden) {
        messageApi.error(t('messages.noPermission'));
      }
    }
  };

  const handleSilentCheckFn = (e) => {
    setIsSilentDelCheck(e.target.checked);
  };

  const getAlertLists = async () => {
    let a = [];
    const res = await getAlertOptions();
    if (res.status === ResponseCode.OK) {
      a = res.data;
    }
    return a;
  };

  const getSilentLists = useCallback(async (string = searchString) => {
    try {
      const res = await getSilentList(string);
      let alertlists = await getAlertLists();
      let arr2 = [];
      alertlists.filter(item => item.status.state === 'suppressed').map(item => {
        arr2.push({
          name: item.labels.alertname,
          desc: item.annotations.description,
          level: item.labels.severity,
          time: item.startsAt,
          alertIds: item.status.silencedBy.length > 0 ? item.status.silencedBy : item.status.inhibitedBy,
          label: item.labels,
        });
      });
      let arr = [];
      if (res.status === ResponseCode.OK) {
        res.data.filter(item => item.status.state === 'active' || item.status.state === 'pending').map(item => {
          let alertNames = resolveAlertName(item.matchers);
          let _alertSecondlist = [];
          if (arr2.length > 0) {
            arr2.forEach(arr2item => {
              if (arr2item.alertIds.includes(item.id)) {
                _alertSecondlist.push(arr2item);
              }
            });
          }
          arr.push({
            key: item.id,
            name: alertNames,
            state: item.status.state,
            remarks: item.comment,
            creators: item.createdBy,
            begintime: item.startsAt,
            endtime: item.endsAt,
            alertNum: _alertSecondlist.length,
            id: item.id,
            alertList: _alertSecondlist,
            matchers: item.matchers,
          });
        });
        let silentNameOption = [];
        let silentLabelOptions = [];
        filterRepeatByid(arr).map(item => {
          silentNameOption.push({
            label: item.name,
            value: item.name,
          });
          item.matchers.map(subitem => {
            silentLabelOptions.push({
              label: `${subitem.name}=${subitem.value}`,
              value: `${subitem.name}=${subitem.value}`,
            });
          });
        });
        arr = arr.sort((a, b) => dayjs(b.begintime).unix() - dayjs(a.begintime).unix());
        setLabelOptions(filterRepeat(silentLabelOptions));
        setSilentNameOptions(defaultOptions.concat(filterRepeat(silentNameOption)));
        let filiterCreators = [];
        filterRepeatByid(arr).forEach(item => {
          filiterCreators.push({
            text: item.creators,
            value: item.creators,
          });
        });
        setFilterAlertStatus(filterRepeat(filiterCreators));
        setSilentTableList(filterRepeatByid(arr));
      }
    } catch (e) {
      messageApi.error({ content: t('messages.fetchFailed', { ns: 'common', message: e.response?.data?.message }), key: 'silentListFetchFailed' });
    }
  }, [defaultOptions]);

  const getSilentOptions = useCallback(async () => {
    try {
      const res = await getSilentList();
      let arrOption = [];
      if (res.status === ResponseCode.OK) {
        res.data.filter(item => item.status.state === 'active' || item.status.state === 'pending').map(item => {
          let alertNamesOption = resolveAlertName(item.matchers);
          arrOption.push({
            key: item.id,
            name: alertNamesOption,
            matchers: item.matchers,
          });
        });
        let silentNameOptions2 = [];
        let silentLabelOption = [];
        filterRepeatByid(arrOption).map(item => {
          silentNameOptions2.push({
            label: item.name,
            value: item.name,
          });
          item.matchers.map(subitem => {
            silentLabelOption.push({
              label: `${subitem.name}=${subitem.value}`,
              value: `${subitem.name}=${subitem.value}`,
            });
          });
        });
        setLabelOptions(filterRepeat(silentLabelOption));
        setSilentNameOptions(defaultOptions.concat(filterRepeat(silentNameOptions2)));
      }
    } catch (e) {
      messageApi.error({ content: t('messages.fetchFailed', { ns: 'common', message: e.response?.data?.message }), key: 'silentOptionsFetchFailed' });
    }
  }, [defaultOptions]);

  const timeSearch = useCallback(() => {
    clearTimeout(setTimeoutId);
    getSilentLists(searchString);
    setTimeoutId = setTimeout(() => {
      timeSearch();
    }, 15000);
  }, [getSilentLists]);

  useEffect(() => {
    getSilentOptions();
    setTimeoutId = 0;
    searchString = '';
  }, []);

  useEffect(() => {
    timeSearch();
    return () => clearTimeout(setTimeoutId);
  }, [timeSearch]);

  return (
    <div className='silentAlert child_content'>
      <ToastMsg contextHolder={contextHolder} />
      <BreadCrumbCom items={[
        { title: t('module.title'), path: `/${containerRouterPrefix}/alert/silentAlert`, disabled: true },
        { title: t('silentAlert.title'), path: '/' },
      ]} />
      <div className='silentAlert_top' style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
        <p className='silentAlert_top_name' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('silentAlert.silentNameLabel')}</p>
        <Select
          className='silentAlert_top_nameSearch'
          defaultValue='all'
          style={{ width: 236 }}
          options={silentNameOptions}
          onChange={changeSilentName}
          value={selectSilentName}
        />
        <p className='silentAlert_top_label' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('silentAlert.label')}</p>
        <Select
          className='silentAlert_top_labelSearch'
          mode="multiple"
          allowClear
          placeholder={t('silentAlert.selectLabel')}
          style={{ width: '580px', maxWidth: '500px' }}
          onChange={changeSilentLabel}
          options={labelOptions}
          value={selectSilentLabel}
        />
        <Button onClick={resetSilentSearch} className='silentAlert_top_resetSearch cancel_btn'>{t('silentAlert.reset')}</Button>
        <Button onClick={searchSilent} className='primary_btn' loading={searchLoading}>{t('silentAlert.search')}</Button>
      </div>
      <div className='silentAlert_content' style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
        <Button type="primary" onClick={showSilentModal} className='creatSilentBtn'>{t('silentAlert.createSilence')}</Button>
        <Table
          columns={silentColumns}
          onChange={handleAlertTableChange}
          pagination={{
            className: 'page',
            style: { paddingRight: '0px' },
            showTotal: total => t('pagination.total', { ns: 'common', total }),
            pageSizeOptions: [10, 20, 50],
            showSizeChanger: true,
          }}
          expandable={{
            expandedRowRender: (record) => (
              <div style={{ paddingLeft: '160px' }}>
                <p>{t('silentAlert.silencedAlerts')}</p>
                {record.alertList.map((item, index) => (
                  <div key={`${item.name}-${index}`} style={{ display: 'flex' }}>
                    <p style={{ marginRight: '8px' }}>{`${item.name}:`}</p>
                    <p>{item.desc}</p>
                  </div>
                ))}
              </div>
            ),
            rowExpandable: (record) => record.name !== '',
          }}
          dataSource={silentTableList}
        />
      </div>
      <EditSilentAlertModal formType={formType} isShowModel={isSilentModalOpen} data={propsChildData} cancelEdit={cancelEdit} submitModel={handleChildClick} />
      <DeleteInfoModal
        title={t('modal.deleteSilence')}
        open={silentDelModalOpen}
        cancelFn={handleDelSilentCancel}
        content={[
          t('modal.deleteWarning'),
          t('modal.deleteConfirm', { name: silentName }),
        ]}
        showCheck={true}
        isCheck={isSilentDelCheck}
        checkFn={handleSilentCheckFn}
        confirmFn={deleteSilent} />
    </div >
  );
}