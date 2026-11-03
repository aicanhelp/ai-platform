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

import { useCallback, useEffect, useState, useStore } from 'openinula';
import { useTranslation } from 'react-i18next';
import { ExportOutlined } from '@ant-design/icons';
import { ResponseCode } from '@/common/constants';
import { message, Button, Select, Form } from 'antd';
import { getHelmDetailLogPodAndContainer, getHelmDetailLogData } from '@/api/containerApi';
import '@/styles/pages/helm.less';
import { exportHelmLogOutPut } from '@/tools/utils';
import ToastMsg from '@/components/ToastMsg';
import useButtonLoading from '@/hooks/useButtonLoading';

/**
 * Application Log Tab Component
 *
 * @param {Object} props
 * @param {string} props.nameProps - Application name
 * @param {string} props.namespaceProps - Namespace
 * @param {Object} props.releaseDataProps - Release Data
 */
export default function ApplicationLogTab({ nameProps, namespaceProps, releaseDataProps }) {
  const [logForm] = Form.useForm();
  const { t } = useTranslation('application');
  const [helmLog, setHelmLog] = useState();
  const [searchLoading, setSearchLoading] = useButtonLoading();
  const [messageApi, contextHolder] = message.useMessage();
  const [podOptions, setPodOptions] = useState([]);
  const [containerOptions, setContainerOptions] = useState([]);

  const resourcesData = releaseDataProps?.resources;
  const helmNamespace = namespaceProps;
  const [allContainer, setAllContainer] = useState([]);

  const themeStore = useStore('theme');

  const getPodNamespace = (podName) => {
    let lastNamespace = '';
    (resourcesData || []).forEach(item => {
      if (podName && podName.startsWith(`${item.name}-`)) {
        if (item.namespace) {
          lastNamespace = item.namespace;
        }
      }
    });
    return lastNamespace;
  };

  const getHelmLog = useCallback(async () => {
    const conditions = logForm.getFieldsValue();
    if (conditions.pod && conditions.container) {
      const temNamespace = getPodNamespace(conditions.pod);
      const res = await getHelmDetailLogData(temNamespace, conditions.pod, conditions.container);
      if (res.status === ResponseCode.OK) {
        setHelmLog(res.data);
      }
    }
  }, []);

  const handleReset = () => {
    if (!allContainer.length) { return; }
    setContainerOptions(allContainer[0].children);
    logForm.setFieldsValue({
      pod: allContainer[0].value,
      container: allContainer[0].children[0].value,
    });
    getHelmLog();
  };

  const handleExportLog = () => {
    if (helmLog) {
      exportHelmLogOutPut(`${nameProps}`, helmLog);
      messageApi.success(t('log.exportSuccess'));
    } else {
      messageApi.error(t('log.noLogToExport'));
    }
  };

  const handleSearch = async () => {
    setSearchLoading(true);
    try {
      await getHelmLog();
    } finally {
      setSearchLoading(false);
    }
  };

  const getPodSelectAndContainerValue = async () => {
    const res = await getHelmDetailLogPodAndContainer(helmNamespace, nameProps);
    if (res.status === ResponseCode.OK) {
      let podArr = [];
      let containerArr = [];
      let obj = res.data.data;
      const newArr = [];
      for (let key in obj) {
        if (Object.prototype.hasOwnProperty.call(obj, key)) {
          if (obj[key].length === 1) {
            newArr.push({ pod: key, container: obj[key].toString() });
          } else {
            obj[key].forEach((item, index) => {
              newArr.push({ pod: key, container: obj[key][index] });
            });
          }
        }
      }
      let _arr = [];
      let options = [];
      newArr.forEach(i => {
        if (i.pod) {
          _arr.push(i.pod);
          const filterList = options.filter(filterItem => filterItem.value === i.pod);
          if (filterList.length) {
            options.map(mapItem => {
              if (mapItem.value === i.pod) {
                if (mapItem.children.filter(mapFilterItem => mapFilterItem.value === i.container).length) {
                  const mapFindIndex = mapItem.children.findIndex((mapFilterItem) => mapFilterItem.value === i.container);
                  mapItem.children[mapFindIndex].children.push({ value: i.pod, label: i.pod });
                } else {
                  mapItem.children.push({
                    value: i.container,
                    label: i.container,
                    children: [{ value: i.pod, label: i.pod }],
                  });
                }
              }
            });
          } else {
            options.push({
              value: i.pod,
              label: i.pod,
              children: [{
                value: i.container,
                label: i.container,
                children: [{ value: i.pod, label: i.pod }],
              }],
            });
          }
        }
      });
      setAllContainer(options);
      options.forEach(podItem => {
        podArr.push({ value: podItem.value, label: podItem.value });
      });
      setPodOptions(podArr);
      if (options.length) {
        options[0].children.forEach(i => {
          containerArr.push({ value: i.value, label: i.label });
        });
      }
      setContainerOptions(containerArr);
      if (logForm) {
        if (podArr.length && containerArr.length) {
          logForm.setFieldsValue({
            pod: podArr[0].value,
            container: containerArr[0].value,
          });
          let podFirstValue = podArr[0].value;
          let containerFirstValue = containerArr[0].value;
          const temNamespace = getPodNamespace(podFirstValue);
          const res1 = await getHelmDetailLogData(temNamespace, podFirstValue, containerFirstValue);
          if (res1.status === ResponseCode.OK) {
            setHelmLog(res1.data);
          }
        } else {
          setHelmLog('');
        }
      }
    }
  };

  const handleChangeContainer = (podValue) => {
    let _newContainer = [];
    allContainer.filter(item => {
      if (item.value === podValue) {
        item.children.forEach(i => {
          _newContainer.push(i.value);
        });
      }
    });
    logForm.setFieldValue('container', _newContainer[0]);
    let filterContainerArr = [...new Set(_newContainer)];
    let _typeArr = [];
    filterContainerArr.forEach(i => {
      _typeArr.push({ label: i, value: i });
    });
    setContainerOptions(_typeArr);
  };

  useEffect(() => {
    getHelmLog();
  }, [getHelmLog]);

  useEffect(() => {
    getPodSelectAndContainerValue();
  }, []);

  return <div className="helm_tab_container" style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#ffffff' : '#2a2d34ff' }}>
    <ToastMsg contextHolder={contextHolder} />
    <div className="helm_log_card" style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#ffffff' : '#2a2d34ff' }}>
      <Form
        form={logForm}
        onFinish={handleSearch}
        className='helm_log_flex_box'
        style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#ffffff' : '#2a2d34ff' }}
      >
        <div className='helm_form_item' style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#ffffff' : '#2a2d34ff' }}>
          <Form.Item label="Pod" name="pod">
            <Select showSearch options={podOptions} className='log_options' onChange={handleChangeContainer} />
          </Form.Item>
          <Form.Item label={t('log.container')} name="container">
            <Select options={containerOptions} className='log_options' />
          </Form.Item>
        </div>
        <Form.Item style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#ffffff' : '#2a2d34ff' }}>
          <Button className='cancel_btn' style={{ marginRight: '16px', color: themeStore.$s.theme !== 'light' && '#fff' }} onClick={handleReset}>{t('actions.reset')}</Button>
          <Button htmlType="submit" className='primary_btn' loading={searchLoading}>{t('actions.query')}</Button>
        </Form.Item>
      </Form>
    </div>

    <div className="helm_log_space_box" style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#ffffff' : '#2a2d34ff' }}>
      <div className="helm_log_space_box_title">
        <p className="box_title_h3" style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('log.title')}</p>
        <div className="helm_log_tool_word_group" onClick={handleExportLog}>
          <ExportOutlined className="common_antd_icon primary_color" />
          <span>{t('log.export')}</span>
        </div>
      </div>
      <div className="helm_log_space_box_content" style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#ffffff' : '#2a2d34ff' }}>
        <div className="helm_log_items">
          <pre className='helm_log_items_content'>{helmLog ? helmLog : t('log.noLog')}</pre>
        </div>
      </div>
    </div>
  </div>;
}