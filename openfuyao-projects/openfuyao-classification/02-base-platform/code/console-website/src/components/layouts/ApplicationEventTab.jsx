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

import '@/styles/pages/helm.less';
import { useEffect, useState, useCallback, useRef, useStore, useMemo } from 'openinula';
import { useTranslation } from 'react-i18next';
import { Button, Form, Space, Select, Timeline, FloatButton } from 'antd';
import Dayjs from 'dayjs';
import EmptyData from '@/components/EmptyData';
import { getHelmEventsData } from '@/api/containerApi';
import { EllipsisOutlined, ArrowUpOutlined, ArrowDownOutlined } from '@ant-design/icons';
import { DEFAULT_PAGE_SIZE, DEFAULT_CURRENT_PAGE, ResponseCode, eventLevelOptions } from '@/common/constants';
import { throttle } from '@/tools/function';
import { getEventLevelLabel } from '@/utils/common';
import useButtonLoading from '@/hooks/useButtonLoading';
const eventsPageSize = DEFAULT_PAGE_SIZE;

/**
 * Application Event Tab Component
 *
 * @param {Object} props
 * @param {string} props.nameProps - Application name
 * @param {string} props.namespaceProps - Namespace
 * @param {Object} props.dataProps - Resource data
 * @param {string} props.type - Component type: 'application' or 'extension'
 */
export default function ApplicationEventTab({ namespaceProps, dataProps, type = 'application' }) {
  const [eventsSearchForm] = Form.useForm();
  const { t } = useTranslation('application');
  const themeStore = useStore('theme');

  // Localized event level options
  const localizedEventLevelOptions = useMemo(() =>
    eventLevelOptions.map(opt => ({
      value: opt.value,
      label: getEventLevelLabel(opt.label),
    })),
  [t]);

  const [eventsPage, setEventsPage] = useState(DEFAULT_CURRENT_PAGE);

  const [loading, setLoading] = useState(false);
  const loadingRef = useRef(false);

  const [searchLoading, setSearchLoading] = useButtonLoading();

  const [isScroll, setIsScroll] = useState(true);

  const timeLineRef = useRef(null);

  const [resourceNameOptions, setResourceNameOptions] = useState([]);

  const [resourceTypeOptions, setResourceTypeOptions] = useState([]);

  const [evenNamespaceOptions, setEvenNamespaceOptions] = useState([]);

  const [initOptions, setInitOptions] = useState([]); // 保存初始值

  const [manageTimeLineItems, setManageTimeLineItems] = useState([]);

  const [eventsContinue, setEventsContinue] = useState('');

  const [defaultResourceName, setDefaultResourceName] = useState('');

  const [defaultResourceType, setDefaultResourceType] = useState('');

  // Storage key based on type
  const scrollKey = type === 'application' ? 'isHelmEventScroll' : 'isExtendEventScroll';

  const helmResourceToTop = () => {
    const target = document.querySelector('.container_content');
    target.scrollTo({ top: 0, left: 0, behavior: 'smooth' });
  };

  const helmResourceToDown = () => {
    const target = document.querySelector('.container_content');
    target.scrollTo({ top: target.scrollHeight - target.clientHeight, left: 0, behavior: 'smooth' });
  };

  // 获取资源名称及类型下拉内容
  const getResourceNameAndTypeValue = () => {
    let _arr = [];
    let options = [];
    if (dataProps.resources) {
      dataProps.resources.forEach(i => {
        if (i.name) {
          _arr.push(i.name);
          const filterList = options.filter(filterItem => filterItem.value === i.name);
          if (filterList.length) {
            options.map(mapItem => {
              if (mapItem.value === i.name) {
                if (mapItem.children.filter(mapFilterItem => mapFilterItem.value === i.kind).length) {
                  const mapFindIndex = mapItem.children.findIndex((mapFilterItem) => mapFilterItem.value === i.kind);
                  mapItem.children[mapFindIndex].children.push({ value: i.name, label: i.name });
                } else {
                  mapItem.children.push({
                    value: `${i.uid}/${i.kind}`,
                    label: i.kind,
                    children: [{ value: i.name, label: i.name }],
                  });
                }
              }
            });
          } else {
            options.push({
              value: i.name,
              label: i.name,
              children: [{
                value: `${i.uid}/${i.kind}`,
                label: i.kind,
                children: [{ value: i.name, label: i.name }],
              }],
            });
          }
        }
      });
    } else {
      return;
    }
    let nameArr = [];
    options.forEach(nameItem => {
      nameArr.push({ value: nameItem.value, label: nameItem.value });
    });
    if (!options.length || !nameArr.length) { return; }

    setInitOptions(options);
    setResourceNameOptions(nameArr);
    setDefaultResourceName(nameArr[0].value);
    let typeOptionArr = [];
    options[0].children.forEach(i => {
      typeOptionArr.push({ value: i.value, label: i.label });
    });
    setResourceTypeOptions(typeOptionArr);
    setDefaultResourceType(typeOptionArr[0].value);

    eventsSearchForm.setFieldValue('resourceName', nameArr[0].value);
    eventsSearchForm.setFieldValue('resourceType', options[0].children[0].value);
    handleChangeResourceType();
  };

  // 改变资源名称
  const handleChangeResourceName = (value) => {
    let namespaceTypeArr = [];
    initOptions.filter(item => {
      if (item.value === value) {
        item.children.forEach(i => {
          namespaceTypeArr.push(i.value);
        });
      }
    });
    eventsSearchForm.setFieldValue('resourceType', namespaceTypeArr[0]);
    let _typeArr = [];
    namespaceTypeArr.forEach(i => {
      _typeArr.push({ label: i.split('/')[1], value: i });
    });
    setResourceTypeOptions(_typeArr);
    handleChangeResourceType();
  };

  // 改变资源类型
  const handleChangeResourceType = (value) => {
    const resourceType = value ?? eventsSearchForm.getFieldValue('resourceType');
    let lastArr = dataProps.resources.filter(i => (i.name === eventsSearchForm.getFieldValue('resourceName') && i.kind === (resourceType || '').split('/')[1]));
    let namespaceArr = [];
    lastArr.forEach(item => {
      if (item.namespace !== '') {
        namespaceArr.push({ value: item.namespace, label: item.namespace });
      } else {
        namespaceArr.push({ value: 'default', label: 'default' });
      }
    });
    setEvenNamespaceOptions(namespaceArr);
    eventsSearchForm.setFieldValue('eventNamespace', namespaceArr.length ? (namespaceArr[0].value || 'default') : 'default');
  };

  const renderTimeLineEvents = (data, temporyType = '') => {
    const newItems = data.map(item => ({
      color: '#3f66f5',
      children: <div className="timeLine_box">
        <p className="time_title">{Dayjs(item.metadata.creationTimestamp).format('YYYY-M-DD HH:mm:ss')}</p>
        <div className="time_message_card">
          <div className='time_info'>
            <div className='time_info_item mange_item'>
              <span className={`${item.type === 'Normal' ? 'running_circle' : 'failed_circle'}`}></span>
              <span>{item.type === 'Normal' ? t('event.normal') : t('event.warning')}</span>
            </div>
            <div className='time_info_item'>
              <span >{t('event.namespace')}：</span>
              <span>{item.metadata.namespace}</span>
            </div>
            <div className='time_info_item'>
              <span>{t('event.resourceNameLabel')}</span>
              <span>{item.metadata.name}</span>
            </div>
          </div>
          <p className='manage_note'>{item.note}</p>
        </div>
      </div>,
    }));
    setManageTimeLineItems(prev => (temporyType ? [...prev, ...newItems] : newItems));
  };

  const getHelmEventsTimeLine = useCallback(async (
    page = eventsPage,
    pageSize = eventsPageSize,
    continueToken = eventsContinue,
    resourceType = eventsSearchForm.getFieldsValue().resourceType,
  ) => {
    setLoading(true);
    let conditions = {
      eventLevel: eventsSearchForm.getFieldsValue().eventLevel,
      resourceName: eventsSearchForm.getFieldsValue().resourceName,
      resourceType: resourceType.split('/')[1],
    };
    if (conditions.resourceName) {
      const res = await getHelmEventsData(
        {
          page,
          pageSize,
          continueToken,
        },
        eventsSearchForm.getFieldsValue().eventNamespace,
        conditions,
      );
      if (res.status === ResponseCode.OK) {
        const data = res.data.items;
        setEventsContinue(res.data.metadata.continue);
        if (res.data.metadata.continue) {
          sessionStorage.setItem(scrollKey, 'true');
          setIsScroll(true);
        } else {
          sessionStorage.setItem(scrollKey, 'false');
          setIsScroll(false);
        }
        eventsPage === 1 ? renderTimeLineEvents(data) : renderTimeLineEvents(data, 'push');
        setLoading(false);
      }
    }
    // 无论直接调用还是分页重置触发的 useEffect 拉取，均在拉取结束后复位查询按钮 loading
    setSearchLoading(false);
  }, [namespaceProps, eventsPage, eventsPageSize, eventsSearchForm]);

  const handleScroller = useCallback(() => {
    const { scrollHeight, clientHeight, scrollTop } = document.querySelector('.child_content');
    if (scrollHeight - scrollTop <= clientHeight) {
      if (!loadingRef.current) {
        // 节流
        if (sessionStorage.getItem(scrollKey) === 'true') {
          setEventsPage(pre => pre + 1);
        }
      }
    }
  }, []);

  // 重置：与 handleSearch 对齐，仅重置状态，由 effect 统一拉取，避免双请求/旧条件请求
  const handleReset = () => {
    eventsSearchForm.setFieldsValue({
      resourceType: (defaultResourceType || '').split('/')[1] || '',
      resourceName: defaultResourceName,
      eventLevel: '',
    });
    setResourceTypeOptions(initOptions[0]?.children ?? []);
    setEventsContinue('');
    if (eventsPage !== DEFAULT_CURRENT_PAGE) {
      setEventsPage(DEFAULT_CURRENT_PAGE);
    } else {
      getHelmEventsTimeLine(1, eventsPageSize, '');
    }
  };

  useEffect(() => {
    loadingRef.current = loading;
  }, [loading]);

  useEffect(() => {
    if (eventsSearchForm) {
      eventsSearchForm.setFieldsValue({
        resourceType: '',
        resourceName: '',
        eventLevel: '',
      });
    }
  }, []);

  useEffect(() => {
    getResourceNameAndTypeValue();
  }, []);

  useEffect(() => {
    getHelmEventsTimeLine();
  }, [getHelmEventsTimeLine]);

  const filterOption = (input, option) =>
    (option?.label ?? '').toLowerCase().includes(input.toLowerCase());

  const handleSearch = async () => {
    setSearchLoading(true);
    try {
      setEventsContinue('');
      if (eventsPage !== 1) {
        setEventsPage(DEFAULT_CURRENT_PAGE);
      } else {
        await getHelmEventsTimeLine(1, 10, '');
      }
    } catch {
      setSearchLoading(false);
    }
  };

  useEffect(() => {
    const throttledHandleScroller = throttle(handleScroller);
    window.addEventListener('wheel', throttledHandleScroller);
    return () => window.removeEventListener('wheel', throttledHandleScroller);
  }, []);

  return <div className="helm_tab_container" style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#ffffff' : '#2a2d34ff' }}>
    <div className='helm_event_card_all'>
      <div className="helm_event_card" style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#ffffff' : '#2a2d34ff' }}>
        <Form
          className='helm_event_flex_box'
          style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#ffffff' : '#2a2d34ff' }}
          form={eventsSearchForm}
          onFinish={handleSearch}>
          <div className='helm_form_item'>
            <Form.Item label={t('event.resourceName')} name="resourceName">
              <Select showSearch filterOption={filterOption} options={resourceNameOptions} onChange={handleChangeResourceName} className='events_options' />
            </Form.Item><Form.Item label={t('event.resourceType')} name="resourceType">
              <Select options={resourceTypeOptions} onChange={handleChangeResourceType} className='events_options' />
            </Form.Item>

            <Form.Item label={t('event.eventLevel')} name="eventLevel">
              <Select options={localizedEventLevelOptions} className='events_options' />
            </Form.Item>
            <Form.Item label={t('event.namespace')} name="eventNamespace">
              <Select options={evenNamespaceOptions} className='events_options' />
            </Form.Item>

            <Form.Item>
              <Space>
                <Button className='cancel_btn' onClick={handleReset} style={{ color: themeStore.$s.theme !== 'light' && '#ffffff' }}>{t('actions.reset')}</Button>
                <Button htmlType="submit" className='primary_btn' loading={searchLoading}>{t('actions.query')}</Button>
              </Space>
            </Form.Item>
          </div>

        </Form>
      </div>
      <div className='helm_event_timeLine' ref={timeLineRef} style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#ffffff' : '#2a2d34ff' }}>
        {manageTimeLineItems.length ? <Timeline items={manageTimeLineItems} /> : <EmptyData />}
        {!!manageTimeLineItems.length && isScroll && <div className='bottom_title' onClick={handleScroller} >
          <EllipsisOutlined />
          {loading ? <span>{t('event.loading')}</span> : <span>{t('event.loadMore')}</span>}
        </div>}
      </div>
    </div>
    <FloatButton.Group
      shape="square"
      style={{
        right: 20,
      }}>
      <FloatButton icon={<ArrowUpOutlined />} onClick={helmResourceToTop} />
      <FloatButton icon={<ArrowDownOutlined />} onClick={helmResourceToDown} />
    </FloatButton.Group>
  </div>;
}
