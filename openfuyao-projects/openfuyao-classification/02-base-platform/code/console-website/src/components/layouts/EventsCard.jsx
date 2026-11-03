/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import '@/styles/components/eventsCard.less';
import { Fragment, useEffect, useState, useCallback, useRef, useMemo } from 'openinula';
import { Button, Form, Space, Select, Timeline, FloatButton } from 'antd';
import Dayjs from 'dayjs';
import EmptyData from '@/components/EmptyData';
import { getEventsData } from '@/api/containerApi';
import { ArrowUpOutlined, EllipsisOutlined } from '@ant-design/icons';
import { DEFAULT_PAGE_SIZE, DEFAULT_CURRENT_PAGE, ResponseCode } from '@/common/constants';
import { throttle } from '@/tools/function';
import { filterRepeat } from '@/utils/common';
import { useTranslation } from 'react-i18next';
import useButtonLoading from '@/hooks/useButtonLoading';
/**
 * 
 * @param namespace //命名空间
 * @param name //名称 
 * @param type //类型
 * @param nodeCondition //节点数据
 * @returns 
 */
const eventsPageSize = DEFAULT_PAGE_SIZE;
export default function EventsCard({ namespace = '', name = '', type = '', nodeCondition = {}, isNormal = false }) {
  const { t, i18n } = useTranslation('events');
  const [eventsSearchForm] = Form.useForm();
  const [eventsPage, setEventsPage] = useState(DEFAULT_CURRENT_PAGE);
  const [eventsContinue, setEventsContinue] = useState('');
  const [loading, setLoading] = useState(false);
  const loadingRef = useRef(false);
  const [searchLoading, setSearchLoading] = useButtonLoading();
  const [isScroll, setIsScroll] = useState(true);
  const scrollKey = `eventScroll_${name || 'default'}`;
  const timeLineRef = useRef(null);

  useEffect(() => {
    loadingRef.current = loading;
  }, [loading]);

  const resourceOptions = useMemo(() => [
    {
      label: t('resourceTypes.all'),
      value: '',
    },
    {
      label: t('resourceTypes.pod'),
      value: 'Pod',
    },
    {
      label: t('resourceTypes.deployment'),
      value: 'Deployment',
    },
    {
      label: t('resourceTypes.statefulSet'),
      value: 'StatefulSet',
    },
    {
      label: t('resourceTypes.daemonSet'),
      value: 'DaemonSet',
    },
    {
      label: t('resourceTypes.job'),
      value: 'Job',
    },
    {
      label: t('resourceTypes.cronJob'),
      value: 'CronJob',
    },
  ], [t, i18n.language]);

  const toTop = () => {
    const target = document.querySelector('.container_content');
    target?.scrollTo({ top: 0, left: 0, behavior: 'smooth' });
  };

  const [eventsTimeline, setEventTimeline] = useState([]);

  const getPodEventsTimeLine = useCallback(async (
    page = eventsPage,
    pageSize = eventsPageSize,
    continueToken = eventsContinue,
  ) => {
    setLoading(true);
    let conditions = {
      eventLevel: eventsSearchForm.getFieldsValue().eventLevel,
      passName: name,
      resourceName: eventsSearchForm.getFieldsValue().resourceName || name,
      resourceType: eventsSearchForm.getFieldsValue().resourceType || type,
      isNormal,
    };
    if (nodeCondition.nodeName) {
      conditions.nodeEvents = {
        nodeName: nodeCondition.nodeName,
        type: nodeCondition.type,
      };
    }
    const res = await getEventsData(
      {
        page,
        limit: pageSize,
        continueToken,
      },
      namespace,
      conditions,
    );
    if (res.status === ResponseCode.OK) {
      let data = res.data.items;
      setEventsContinue(res.data.metadata.continue);
      if (res.data.metadata.continue) {
        sessionStorage.setItem(scrollKey, 'true');
        setIsScroll(true);
      } else {
        sessionStorage.setItem(scrollKey, 'false');
        setIsScroll(false);
      }
      setEventTimeline(prev => (eventsPage === 1 ? data : prev.concat(data)));
      setLoading(false);
    }
    // 无论直接调用还是分页重置触发的 useEffect 拉取，均在拉取结束后复位查询按钮 loading
    setSearchLoading(false);
  }, [namespace, eventsPage, eventsPageSize, eventsSearchForm, isNormal]);

  const handleScroll = useCallback(() => {
    const { scrollHeight, clientHeight, scrollTop } = document.querySelector('.container_content');

    if (scrollHeight - scrollTop <= clientHeight + 1) { // add an 'epsilon' to allow looser precision
      if (!loadingRef.current) {
        // 节流
        if (sessionStorage.getItem(scrollKey) === 'true') {
          setEventsPage(pre => pre + 1);
        }
      }
    }
  }, []);

  const handleReset = async () => {
    eventsSearchForm.setFieldsValue({
      resourceType: '',
      resourceName: '',
      eventLevel: '',
    });
    setSearchLoading(true);
    try {
      setEventsContinue('');
      if (eventsPage !== 1) {
        setEventsPage(DEFAULT_CURRENT_PAGE);
      } else {
        await getPodEventsTimeLine(1, 10, '');
      }
    } catch {
      setSearchLoading(false);
    }
  };

  const [resourceNameRaw, setResourceNameRaw] = useState([]);

  useEffect(() => {
    const fetchResourceNameRaw = async () => {
      const res = await getEventsData({ page: 1, limit: 10000, continueToken: '' }, '', '');
      if (res.status === ResponseCode.OK) {
        setResourceNameRaw(res.data.items.map(item => ({
          label: item.regarding?.name,
          value: item.regarding?.name,
        })));
      }
    };
    fetchResourceNameRaw();
  }, []);

  const resourceNameOptions = useMemo(() => {
    if (type !== '') {
      return [{ label: t('filters.all'), value: '' }];
    }
    let resArr = [{ label: t('filters.all'), value: '' }];
    resArr = resArr.concat(resourceNameRaw);
    return filterRepeat(resArr);
  }, [type, resourceNameRaw, t, i18n.language]);

  useEffect(() => {
    if (eventsSearchForm) {
      eventsSearchForm.setFieldsValue({
        resourceType: type,
        resourceName: name,
        eventLevel: '',
      });
    }
  }, [type, name]);

  useEffect(() => {
    getPodEventsTimeLine();
  }, [getPodEventsTimeLine]);

  const filterOption = (input, option) =>
    (option?.label ?? '').toLowerCase().includes(input.toLowerCase());

  const handleSearch = async () => {
    setSearchLoading(true);
    try {
      setEventsContinue('');
      if (eventsPage !== 1) {
        setEventsPage(DEFAULT_CURRENT_PAGE);
      } else {
        await getPodEventsTimeLine(1, 10, '');
      }
    } catch {
      setSearchLoading(false);
    }
  };

  useEffect(() => {
    const throttledHandleScroll = throttle(handleScroll);
    window.addEventListener('wheel', throttledHandleScroll);
    return () => window.removeEventListener('wheel', throttledHandleScroll);
  }, []);

  return <Fragment>
    <Form
      id='formTop'
      className='events_form'
      form={eventsSearchForm}
      onFinish={handleSearch}>
      <div className='select_box'>
        {(!type) && <>
          <Form.Item label={t('filters.resourceName')} name="resourceName" initialValue={name || ''}>
            <Select showSearch filterOption={filterOption} options={resourceNameOptions} style={{ width: 200 }} />
          </Form.Item>
          <Form.Item label={t('filters.resourceType')} name="resourceType" initialValue={type || ''}>
            <Select options={resourceOptions} style={{ width: 200 }} />
          </Form.Item>
        </>
        }
        <Form.Item label={t('filters.eventLevel')} name="eventLevel" initialValue={''}>
          <Select options={[
            { label: t('eventLevel.all'), value: '' },
            { label: t('eventLevel.normal'), value: 'Normal' },
            { label: t('eventLevel.warning'), value: 'Warning' },
          ]} style={{ width: 200 }} />
        </Form.Item>
      </div>
      <Form.Item className='events_search_btn'>
        <Space>
          <Button className='cancel_btn' style={{ marginRight: '8px' }} onClick={handleReset}>{t('filters.reset')}</Button>
          <Button htmlType="submit" className='primary_btn' loading={searchLoading}>{t('filters.search')}</Button>
        </Space>
      </Form.Item>
    </Form>
    <div className='event_timeLine' ref={timeLineRef}>
      {eventsTimeline.length ? <Timeline items={
        eventsTimeline.map(item => ({
          color: '#3f66f5',
          children: <div className="timeLine_box">
            <p className="time_title">{Dayjs(item.metadata.creationTimestamp).format('YYYY-M-DD HH:mm:ss')}</p>
            <div className="time_message_card">
              <div className='time_info'>
                <div className='time_info_item'>
                  <span className={`${item.type === 'Normal' ? 'running_circle' : 'failed_circle'}`}></span>
                  <span className='status_word'>{item.type === 'Normal' ? t('eventLevel.normal') : t('eventLevel.warning')}</span>
                </div>
                <div className='time_info_item'>
                  <span>{t('timeline.namespace')}:</span>
                  <span>{item.metadata.namespace}</span>
                </div>
                <div className='time_info_item'>
                  <span>{t('timeline.resourceName')}:</span>
                  <span>{item.regarding?.name}</span>
                </div>
              </div>
              <p>{item.note}</p>
            </div>
          </div>,
        }))
      } /> : <EmptyData />}
      {!!eventsTimeline.length && isScroll && <div className='bottom_title' onClick={handleScroll} >
        <EllipsisOutlined />
        {loading ? <span>{t('timeline.loading')}</span> : <span>{t('timeline.viewMore')}</span>}
      </div>}
    </div>
    <FloatButton.Group
      shape="square"
      className='float_btn_right'
      style={{
        right: 20,
      }}>
      <FloatButton icon={< ArrowUpOutlined />} onClick={toTop}></FloatButton>
    </FloatButton.Group>
  </Fragment>;
}
