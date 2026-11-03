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
import { containerRouterPrefix } from '@/constant';
import { useHistory } from 'inula-router';
import { Table, Space, Button, Popover, message } from 'antd';
import '@/styles/pages/alert.less';
import { useLocation } from 'inula-router';
import dayjs from 'dayjs';
import { useEffect, useState, useCallback, useStore, useMemo } from 'openinula';
import { solveAnnotation, showSuccessThen } from '@/tools/utils';
import { filterAlertStatus, filterAlertStatusStyle, getTimeType } from '@/utils/common';
import { DownOutlined } from '@ant-design/icons';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { ResponseCode } from '@/common/constants';
import { deleteSilentAlert, getSilentListDetail, getAlertOptions, creatSilentAlert } from '@/api/containerAlertApi';
import EditSilentAlertModal from './EditSilentAlertModal';
import utc from 'dayjs/plugin/utc';
import LabelTag from '@/components/LabelTag';
import { useTranslation } from 'react-i18next';
import DetailBaseInfoList from '@/components/DetailBaseInfoList';
dayjs.extend(utc);

export default function SilentDetail() {
  const { t, i18n } = useTranslation('alert');
  const history = useHistory();
  const location = useLocation();
  const [detailsInfo, setDetailsInfo] = useState();
  const [silentId, setSilentId] = useState(location.state?.currentdetail?.id);
  const [isSilentDelCheck, setIsSilentDelCheck] = useState(false);
  const [messageApi, contextHolder] = message.useMessage();
  const themeStore = useStore('theme');

  const baseInfoItems = useMemo(() => ([
    { key: t('detail.silentName'), value: detailsInfo?.name },
    { key: t('detail.silentBeginTime'), value: detailsInfo?.begintime ? dayjs(detailsInfo.begintime).format('YYYY-MM-DD HH:mm') : '--', keyStyle: { width: 114 } },
    { key: t('detail.silentStatus'), value: detailsInfo?.state },
    { key: t('detail.silentEndTime'), value: detailsInfo?.endtime ? dayjs(detailsInfo.endtime).format('YYYY-MM-DD HH:mm') : '--', keyStyle: { width: 114 } },
    { key: t('detail.creator'), value: detailsInfo?.creators },
    { key: t('detail.lastUpdateTime'), value: detailsInfo?.begintime ? dayjs(detailsInfo.begintime).format('YYYY-MM-DD HH:mm') : '--', keyStyle: { width: 114 } },
    { key: t('detail.remark'), value: detailsInfo?.remarks },
  ]), [detailsInfo, t, i18n.language]);

  const silentDetailColumns = useMemo(() => [
    {
      title: t('table.name'),
      dataIndex: 'name',
      key: 'name',
      render: (_, record) => (
        <Space size="middle" className="alertName">
          {record.name}
        </Space>
      ),
    },
    {
      title: t('table.description'),
      dataIndex: 'desc',
      key: 'desc',
      width: '25%',
    },
    {
      title: t('table.level'),
      dataIndex: 'level',
      key: 'level',
      width: '150px',
      render: (_, record) => (
        <Space size="middle" className="">
          <div style={filterAlertStatusStyle(record.level)}>
            {filterAlertStatus(record.level)}
          </div>
        </Space>
      ),
    },
    {
      title: t('table.triggerTime'),
      dataIndex: 'time',
      key: 'time',
      render: (_, record) => (
        <Space size="middle">
          {dayjs(record.time).format('YYYY-MM-DD HH:mm')}
        </Space>
      ),
    },
    {
      title: t('table.labels'),
      width: '40%',
      dataIndex: 'label',
      key: 'label',
      render: (_, record) => (
        <Space size="middle" style={{ display: 'flex', flexWrap: 'wrap' }}>
          {solveAnnotation(record.label).map(item => (
            <p className='alertDetail_info_labels_content'>{`${item.key}` + '=' + `${item.value}`}</p>
          ))}
        </Space>
      ),
    },
  ], [t, i18n.language]);

  const [propsChildData, setPropsChildData] = useState();
  const formType = 'edit';
  const [silentPopOpen, setSilentPopOpen] = useState(false);
  const [silentDelModalOpen, setSilentDelModalOpen] = useState(false);
  const [isSilentModalOpen, setIsSilentModalOpen] = useState(false);

  const handlSilentPopOpenChange = (open) => {
    setSilentPopOpen(open);
  };

  const handleEditSilent = (data) => {
    setIsSilentModalOpen(true);
    let labelList = [];
    data.matchers.map(item => {
      labelList.push({
        key: item.name,
        value: item.value,
      });
    });
    let time = dayjs(dayjs(detailsInfo?.endtime).format('YYYY-MM-DD HH:mm')).diff(dayjs(detailsInfo?.begintime).format('YYYY-MM-DD HH:mm')) / 1000;
    setPropsChildData({
      beginTime: dayjs(dayjs(detailsInfo?.begintime).format('YYYY-MM-DD HH:mm')),
      labelList,
      creators: data.creators,
      remarks: data.remarks,
      timeType: getTimeType(time).type,
      timeNum: getTimeType(time).num,
    });
  };

  const handleDeleteSilent = () => {
    setSilentPopOpen(false);
    setSilentDelModalOpen(true);
  };

  const deleteSilent = async () => {
    try {
      const res = await deleteSilentAlert(silentId);
      if (res.status === ResponseCode.OK) {
        await showSuccessThen(messageApi, t('messages.deleteSuccess', { name: detailsInfo.name }), () => {
          setSilentDelModalOpen(false);
          history.push(`/${containerRouterPrefix}/alert/silentAlert`);
        });
      }
    } catch (error) {
      if (error.response?.status === ResponseCode.Forbidden) {
        messageApi.error(t('messages.noPermission'));
      } else {
        messageApi.error(t('messages.deleteFailed', { name: detailsInfo.name }));
      }
      setSilentDelModalOpen(false);
      setIsSilentDelCheck(false);
    }
  };

  const handleDelSilentCancel = () => {
    setSilentDelModalOpen(false);
  };

  const handleSilentCheckFn = (e) => {
    setIsSilentDelCheck(e.target.checked);
  };

  const cancelEdit = () => {
    setIsSilentModalOpen(false);
  };

  const handleChildClick = (data) => {
    let arr = data;
    return editSilent(arr);
  };

  const editSilent = async (data) => {
    try {
      const res1 = await deleteSilentAlert(silentId);
      if (res1.status === ResponseCode.OK) {
        const res = await creatSilentAlert(data);
        if (res.status === ResponseCode.OK) {
          await showSuccessThen(messageApi, t('messages.editSilentSuccess'), () => {
            setIsSilentModalOpen(false);
            setSilentId(res.data.silenceID);
            cancelEdit();
            history.push(`/${containerRouterPrefix}/alert/silentAlert`);
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
  };

  const getAlertLists = async () => {
    let a = [];
    const res = await getAlertOptions();
    if (res.status === ResponseCode.OK) {
      a = res.data;
    }
    return a;
  };

  const getSilentDetail = useCallback(async () => {
    const res = await getSilentListDetail(silentId);
    let _alertlist = await getAlertLists();
    let arrAlertList = [];
    _alertlist.filter(item => item.status.state === 'suppressed').map(item => {
      arrAlertList.push({
        name: item.labels.alertname,
        desc: item.annotations.description,
        level: item.labels.severity,
        time: item.startsAt,
        alertIds: item.status.silencedBy.length > 0 ? item.status.silencedBy : item.status.inhibitedBy,
        label: item.labels,
      });
    });
    let silentDetail;
    if (res.status === ResponseCode.OK) {
      let _alertName;
      let _matcherName = res.data.matchers.filter(subitem => subitem.name === 'alertname');
      if (_matcherName.length > 0) {
        _alertName = _matcherName[0].value;
      } else if (res.data.matchers.length) {
        _alertName = `${res.data.matchers[0].name}=${res.data.matchers[0].value}`;
      } else {
        _alertName = '-';
      }
      let _alertSecondlist = [];
      if (arrAlertList.length > 0) {
        arrAlertList.forEach(arr2item => {
          if (arr2item.alertIds.includes(res.data.id)) {
            _alertSecondlist.push(arr2item);
          }
        });
      }
      silentDetail = {
        key: res.data.id,
        name: _alertName,
        state: res.data.status.state,
        remarks: res.data.comment,
        creators: res.data.createdBy,
        begintime: res.data.startsAt,
        endtime: res.data.endsAt,
        alertNum: _alertSecondlist.length,
        id: res.data.id,
        alertList: _alertSecondlist,
        matchers: res.data.matchers,
      };
      setDetailsInfo(silentDetail);
    }
  }, [silentId]);

  useEffect(() => {
    getSilentDetail();
  }, [getSilentDetail]);

  return (
    <div className='alertDetail child_content'>
      <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
        {contextHolder}
      </div>
      <div className='alertDetail_top'>
        <BreadCrumbCom items={[
          { title: t('module.title'), disabled: true },
          { title: t('silentAlert.title'), path: `/${containerRouterPrefix}/alert/silentAlert` },
          { title: t('detail.detail'), path: `/${containerRouterPrefix}/alert/silentAlert/detail` + `${detailsInfo?.name}` },
        ]} />
        <div className='alertDetail_top_row' style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
          <p className='alertDetail_top_name' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{detailsInfo?.name}</p>
          <Popover
            placement='bottom'
            content={
              <Space className='column_pop'>
                <Button type='link' onClick={() => handleEditSilent(detailsInfo)}>{t('actions.modify')}</Button>
                <Button type='link' onClick={handleDeleteSilent}>{t('actions.delete')}</Button>
              </Space>
            }
            open={silentPopOpen}
            onOpenChange={handlSilentPopOpenChange}
          >
            <Button className='primary_btn'>
              {t('actions.operation')} <DownOutlined className='small_margin_adjust' />
            </Button>
          </Popover>
        </div>
      </div>
      <div className='tab_container container_margin_box normal_container_height'>
        <div className="detail_card">
          <h3>{t('detail.basicInfo')}</h3>
          <div className='detail_info_box'>
            <DetailBaseInfoList items={baseInfoItems} columns={2} />
            <div className="annotation">
              <div className='ann_title'>
                <p>{t('detail.labels')}</p>
              </div>
              <div className='key_value'>
                {detailsInfo?.matchers.map(item => (
                  <LabelTag labelKey={item.name} labelValue={item.value} />
                ))}
              </div>
            </div>
          </div>
        </div>
        <div className="detail_card">
          <h3>{t('detail.silencedAlerts')}</h3>
          <div className='detail_info_box'>
            <Table columns={silentDetailColumns} dataSource={detailsInfo?.alertList} className='silentAlertDetailTable' />
          </div>
        </div>
      </div>
      <EditSilentAlertModal formType={formType} isShowModel={isSilentModalOpen} data={propsChildData} cancelEdit={cancelEdit} submitModel={handleChildClick} />
      <DeleteInfoModal
        title={t('modal.deleteSilence')}
        open={silentDelModalOpen}
        cancelFn={handleDelSilentCancel}
        content={[
          t('modal.deleteWarning'),
          t('modal.deleteConfirm', { name: detailsInfo?.name }),
        ]}
        showCheck={true}
        isCheck={isSilentDelCheck}
        checkFn={handleSilentCheckFn}
        confirmFn={deleteSilent} />
    </div>
  );
}