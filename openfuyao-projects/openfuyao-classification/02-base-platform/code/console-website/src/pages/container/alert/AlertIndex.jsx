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
import { containerRouterPrefix } from '@/constant.js';
import { Link, useLocation } from 'inula-router';
import { useEffect, useState, useStore, useMemo, useCallback, useRef } from 'openinula';
import { Select, Form, Button, Table, Space, message } from 'antd';
import ToastMsg from '@/components/ToastMsg';
import { filterAlertStatus, filterAlertStatusStyle, filterAlertTime, filterContinueTime } from '@/utils/common';
import { ResponseCode } from '@/common/constants';
import '@/styles/pages/alert.less';
import { SettingOutlined } from '@ant-design/icons';
import dayjs from 'dayjs';
import { getAlertOptions, creatSilentAlert, getLoki } from '@/api/containerAlertApi';
import { solveAnnotation, sorterFirstAlphabet, showSuccessThen } from '@/tools/utils';
import SetSilentAlertModal from './SetSilentAlertModal';
import { useTranslation } from 'react-i18next';
import useButtonLoading from '@/hooks/useButtonLoading';

export default function AlertIndex() {
  const setTimeoutIdRef = useRef(0);
  const searchStringRef = useRef('');
  const continueTimeRef = useRef([]);
  const { t } = useTranslation('alert');
  const themeStore = useStore('theme');
  const [alertForm] = Form.useForm();
  const [messageApi, contextHolder] = message.useMessage();
  const [isShowEditModal, SetIsShowEditModal] = useState(false);
  const [selectSilentLabel, setSelectSilentLabel] = useState([]);
  const [alertTableList, setAlertTableList] = useState([]);
  const [searchLoading, setSearchLoading] = useButtonLoading();
  const [matchers, setMatchers] = useState([]);
  const [critical, setCritical] = useState(0);
  const [warning, setWarning] = useState(0);
  const [info, setInfo] = useState(0);
  const [remarks, SetRemarks] = useState('');
  const location = useLocation();
  const [alertLabelList, setAlertLabelList] = useState([]);
  const [alertNameList, setAlertNameList] = useState([]);
  const [alertSourceList, setAlertSourceList] = useState(['prometheus']);

  const alertNameOptions = useMemo(() => [
    { label: t('level.all'), value: 'all' },
    ...alertNameList.map(item => ({ label: item, value: item })),
  ], [t, alertNameList]);

  const alertLevelOptions = useMemo(() => [
    { label: t('level.all'), value: 'all' },
    { label: t('level.critical'), value: 'critical' },
    { label: t('level.warning'), value: 'warning' },
    { label: t('level.info'), value: 'info' },
  ], [t]);

  const alertTimeOptions = useMemo(() => [
    { value: 'all', label: t('durationFilter.all') },
    { value: '1d', label: t('durationFilter.within1d') },
    { value: '1-5d', label: t('durationFilter.within1to5d') },
    { value: '5-10d', label: t('durationFilter.within5to10d') },
    { value: '10d', label: t('durationFilter.over10d') },
  ], [t]);

  useEffect(() => {
    getLoki().then(res => {
      if (res.status === ResponseCode.OK && res.data.data?.includes('running')) {
        setAlertSourceList(['loki', 'prometheus']);
      }
    }).catch(e => messageApi.error(t('messages.fetchLokiFailed', { message: e.response?.data?.message ?? '' })));
  }, []);
  const alertSourceOptions = useMemo(() => [
    { label: t('level.all'), value: 'all' },
    ...alertSourceList.map(item => ({ label: item, value: item })),
  ], [t, alertSourceList]);

  const alertLabelOptions = useMemo(() => [
    { label: t('level.all'), value: 'all' },
    ...alertLabelList.map(item => ({ label: item, value: item })),
  ], [t, alertLabelList]);

  const resetSearch = useCallback(() => {
    searchStringRef.current = '';
    alertForm.resetFields();
    setSelectSilentLabel([]);
    getAlert();
  }, [alertForm]);

  const editSilentAlert = (data) => {
    SetIsShowEditModal(true);
    let _matchers = [];
    solveAnnotation(data.labels).map(item => {
      _matchers.push({
        name: item.key,
        value: item.value,
        isRegex: false,
        isEqual: true,
      });
    });
    setMatchers(_matchers);
    SetRemarks(data.comment);
  };
  const cancelEdit = () => {
    SetIsShowEditModal(false);
  };
  const handleChildClick = (data) => {
    let arr = data;
    arr.matchers = matchers;
    return editSilent(arr);
  };
  const editSilent = async (data) => {
    try {
      const res = await creatSilentAlert(data);
      if (res.status === ResponseCode.OK) {
        await showSuccessThen(messageApi, t('messages.setSilentSuccess'), () => {
          cancelEdit();
          getAlert();
        });
      }
    } catch (error) {
      if (error?.response?.status === ResponseCode.Forbidden) {
        messageApi.error(t('messages.noPermission'));
      } else {
        messageApi.error(t('messages.editSilentFailed'));
      }
    }
  };
  // table项
  const stateColumns = useMemo(() => [
    {
      title: t('table.name'),
      key: 'name',
      fixed: 'left',
      sorter: (a, b) => sorterFirstAlphabet(a.name, b.name),
      render: (_, record) => (
        <Space size="middle" className="alertName">
          <Link to={{ pathname: `/${containerRouterPrefix}/alert/alertIndex/detail/${record.name}`, state: { currentdetail: record } }}>{record.name}</Link>
        </Space>
      ),
    },
    {
      title: t('table.description'),
      key: 'desc',
      width: '20%',
      sorter: (a, b) => sorterFirstAlphabet(a.desc, b.desc),
      render: (_, record) => (
        <Space size="middle" className="">{record.desc}</Space>
      ),
    },
    {
      title: t('table.namespace'),
      key: 'namespace',
      sorter: (a, b) => sorterFirstAlphabet(a.namespace, b.namespace),
      render: (_, record) => (
        <Space size="middle" className="">{record.namespace}</Space>
      ),
    },
    {
      title: t('table.pod'),
      key: 'pod',
      sorter: (a, b) => sorterFirstAlphabet(a.pod, b.pod),
      render: (_, record) => (
        <Space size="middle" style={{ paddingRight: '10px', maxWidth: '200px', wordBreak: 'break-all' }}>{record.pod}</Space>
      ),
    },
    {
      title: t('table.level'),
      key: 'level',
      sorter: (a, b) => sorterFirstAlphabet(filterAlertStatus(a.level), filterAlertStatus(b.level)),
      render: (_, record) => (
        <Space size="middle" className="">
          <div style={filterAlertStatusStyle(record.level)}>
            {filterAlertStatus(record.level)}
          </div>
        </Space>
      ),
    },
    {
      title: t('table.duration'),
      key: 'timeDiff',
      sorter: (a, b) => sorterFirstAlphabet(a.timeDiff, b.timeDiff),
      render: (_, record) => (
        <Space size="middle" className="">{filterAlertTime(record.timeDiff)}</Space>
      ),
    },
    {
      title: t('table.triggerTime'),
      key: 'begintime',
      sorter: (a, b) => (dayjs(a.begintime) - dayjs(b.begintime)),
      render: (_, record) => (
        <Space size="middle" className="">{dayjs(record.begintime).format('YYYY-MM-DD HH:mm:ss')}</Space>
      ),
    },
    {
      title: t('table.source'),
      key: 'resource',
      width: '120px',
      sorter: (a, b) => sorterFirstAlphabet(a.resource, b.resource),
      render: (_, record) => (
        <Space>
          <p style={{ wordWrap: 'break-word', width: '100px' }}>{record.resource}</p>
        </Space>
      ),
    },
    {
      title: t('table.silence'),
      width: '100px',
      key: 'silentAlert',
      render: (_, record) => (
        <Space size="middle" style={{ marginLeft: '30px' }}>
          <SettingOutlined onClick={() => editSilentAlert(record)} style={{ color: '#3f66f5ff', fontSize: '20px' }} />
        </Space>
      ),
    },
  ], [t]);

  const changeSilentLabel = (value) => {
    setSelectSilentLabel(value);
  };
  const getFilterAlertList = async () => {
    setSearchLoading(true);
    try {
      let searchStringName = '';
      let searchStringLabel = '';
      const formValues = alertForm.getFieldsValue();
      if (formValues.selectName !== 'all') {
        searchStringName = '&filter=alertname=' + `${formValues.selectName}`;
      }
      if (selectSilentLabel.length > 0) {
        selectSilentLabel.map(item => {
          searchStringLabel += ('&filter=' + `${item}`);
        });
      }
      searchStringRef.current = searchStringName + searchStringLabel;
      await getAlert(searchStringRef.current);
    } finally {
      setSearchLoading(false);
    }
  };
  const handleChangeContinueTime = (e) => {
    if (e !== 'all') {
      continueTimeRef.current = filterContinueTime(e);
    } else {
      continueTimeRef.current = [];
    }
  };

  const getAlert = useCallback(async (string = searchStringRef.current) => {
    try {
      const res = await getAlertOptions(string);
      let alertLists = [];
      if (res.status === ResponseCode.OK) {
        let filterData = res.data.filter(item => item.status.state === 'active');
        // update filter options
        setAlertNameList([...new Set(
          filterData
            .map(item => item.labels?.alertname)
            .filter(Boolean),
        )]);
        setAlertLabelList([...new Set(
          filterData
            .flatMap(item => solveAnnotation(item.labels).map(subitem => `${subitem.key}=${subitem.value}`))
            .filter(Boolean),
        )]);
        // update alert list
        filterData.map(item => {
          let resourceType = '';
          if (item.labels.prometheus) {
            resourceType = 'prometheus';
          }
          if (item.labels.loki) {
            resourceType = 'loki';
          }
          alertLists.push({
            name: item.labels.alertname ? item.labels.alertname : '--',
            desc: item.annotations.description ? item.annotations.description : '--',
            summary: item.annotations.summary ? item.annotations.summary : '--',
            namespace: item.labels.namespace ? item.labels.namespace : '--',
            pod: item.labels.pod ? item.labels.pod : '--',
            level: item.labels.severity ? item.labels.severity : '--',
            begintime: item.startsAt ? item.startsAt : '--',
            time: item.startsAt ? item.startsAt : '--',
            endtime: item.endsAt ? item.endsAt : '--',
            resource: resourceType ? resourceType : '--',
            labels: item.labels,
            resourceType,
            timeDiff: dayjs(dayjs(item.endsAt).format('YYYY-MM-DD HH:mm:ss'))
              .diff(dayjs(item.startsAt).format('YYYY-MM-DD HH:mm:ss')) / 1000,
            createdBy: item.createdBy ? item.createdBy : '',
            comment: item.comment ? item.comment : '',
          });
        });
        const alertCountList = [];
        alertCountList.push(...alertLists);
        const formValues = alertForm.getFieldsValue();
        if (formValues.selectTime !== 'all' && formValues.selectTime !== '10d') {
          alertLists = alertLists.filter(item => item.timeDiff < continueTimeRef.current[1] && item.timeDiff >= continueTimeRef.current[0]);
        }
        if (formValues.selectTime !== 'all' && formValues.selectTime === '10d') {
          alertLists = alertLists.filter(item => item.timeDiff >= continueTimeRef.current[0]);
        }
        if (formValues.selectLevel !== 'all') {
          alertLists = alertLists.filter(item => item.level === formValues.selectLevel);
        }
        if (formValues.selectResource !== 'all') {
          alertLists = alertLists.filter(item => item.resourceType === formValues.selectResource);
        }
        alertLists = alertLists.sort((a, b) => dayjs(b.begintime).unix() - dayjs(a.begintime).unix());
        setCritical(alertCountList.filter((item) => item.level === 'critical').length);
        setWarning(alertCountList.filter((item) => item.level === 'warning').length);
        setInfo(alertCountList.filter((item) => item.level === 'info').length);
        setAlertTableList(alertLists);
      }
    } catch (e) {
      messageApi.error({ content: t('messages.fetchFailed', { ns: 'common', message: e.response?.data?.message }), key: 'alertFetchFailed' });
    }
  }, [alertForm]);

  const timeSearch = useCallback(() => {
    clearTimeout(setTimeoutIdRef.current);
    getAlert(searchStringRef.current);
    const id = setTimeout(() => {
      timeSearch();
    }, 15000);
    setTimeoutIdRef.current = id;
  }, [getAlert]);

  const selectLevelChange = async (type) => {
    if (type === 'critical') {
      alertForm.setFieldValue('selectLevel', 'critical');
    } else if (type === 'warning') {
      alertForm.setFieldValue('selectLevel', 'warning');
    } else {
      alertForm.setFieldValue('selectLevel', 'info');
    }
    getFilterAlertList();
  };

  useEffect(() => {
    if (location.state?.alertType) {
      selectLevelChange(location.state.alertType);
    }
  }, [location.state]);

  useEffect(() => {
    timeSearch();
    return () => clearTimeout(setTimeoutIdRef.current);
  }, [timeSearch]);

  return <div className="child_content">
    <ToastMsg contextHolder={contextHolder} />
    <div className="alert-bread">
      <BreadCrumbCom items={[
        { title: t('module.title'), path: `/${containerRouterPrefix}/alert/alertIndex`, disabled: true },
        { title: t('currentAlert.title'), path: '/' },
      ]} />
    </div>
    <div className='warning-container'>
      <div className='warning-container-top' style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
        <div className='warning-container-top-form'>
          <Form style={{ display: 'flex', flexWrap: 'wrap' }} form={alertForm}
            initialValues={{
              selectName: 'all',
              selectLevel: 'all',
              selectTime: 'all',
              selectResource: 'all',
              selectPods: 'all',
            }}
          >
            <div className="warning-container-top-form-format">
              <Form.Item
                name="selectName"
                label={t('form.alertName')}
                className='warning-280'
              >
                <Select
                  className=''
                  options={alertNameOptions}
                />
              </Form.Item>

              <Form.Item
                name="selectLevel"
                label={t('form.alertLevel')}
                className='warning-280'
              >
                <Select
                  className=''
                  options={alertLevelOptions}
                />
              </Form.Item>

              <Form.Item
                name="selectTime"
                label={t('form.duration')}
                className='warning-280'
              >
                <Select
                  className=''
                  options={alertTimeOptions}
                  onChange={handleChangeContinueTime}
                />
              </Form.Item>

              <Form.Item
                name="selectResource"
                label={t('form.alertSource')}
                className='warning-280'
              >
                <Select
                  className=''
                  options={alertSourceOptions}
                />
              </Form.Item>
            </div>
            <div className="warning-container-top-form-format-nowarp">
              <Form.Item
                name="selectLabels"
                label={t('form.labels')}
              >
                <Select
                  className='silentAlert_top_labelSearch'
                  mode="multiple"
                  allowClear
                  style={{ width: '580px', maxWidth: '500px' }}
                  placeholder={t('form.selectLabels')}
                  onChange={changeSilentLabel}
                  options={alertLabelOptions}
                />
              </Form.Item>

              <Form.Item
              >
                <div className="button-group warning-300">
                  <Button className='cancel_btn' onClick={resetSearch}>{t('silentAlert.reset')}</Button>
                  <Button className='primary_btn' onClick={getFilterAlertList} loading={searchLoading}>{t('silentAlert.search')}</Button>
                </div>
              </Form.Item>
            </div>
          </Form>
        </div>
      </div>
      <div className="warning-container-table" style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
        <div className="alertNum">
          <div className="alertNumOne">
            <p className="alertNumOneType" style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('level.critical')}</p>
            <p className="alertNumOneNumWrong" onClick={() => selectLevelChange('critical')}>{critical}</p>
          </div><div className="alertNumOne">
            <p className="alertNumOneType" style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('level.warning')}</p>
            <p className="alertNumOneNumWarning" onClick={() => selectLevelChange('warning')}>{warning}</p>
          </div>
          <div className="alertNumOne">
            <p className="alertNumOneType" style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('level.info')}</p>
            <p className="alertNumOneNumInfo" onClick={() => selectLevelChange('info')}>{info}</p>
          </div>
        </div>
        <div>
          <Table dataSource={alertTableList} columns={stateColumns} pagination={{
            className: 'page',
            style: { paddingRight: '0px' },
            showTotal: total => t('pagination.total', { ns: 'common', total }),
            pageSizeOptions: [10, 20, 50],
            showSizeChanger: true,
          }} style={{ minHeight: 'calc(100vh - 470px)' }} scroll={{ x: 1300 }} />
        </div>
      </div>
      <SetSilentAlertModal isShowModal={isShowEditModal} cancelEdit={cancelEdit} remarks={remarks} submitModel={handleChildClick} />
    </div >
  </div>;
}