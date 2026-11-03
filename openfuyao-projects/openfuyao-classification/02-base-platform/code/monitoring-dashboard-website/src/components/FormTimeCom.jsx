/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Fragment, forwardRef, useCallback, useEffect, useRef } from 'openinula';
import { Select, DatePicker, Button, Form, ConfigProvider } from 'antd';
import { timePeriodOptions, refreshTimeOptions } from '@/common/constants';
import { SyncOutlined } from '@ant-design/icons';
import { useIntl } from 'inula-intl';
import '@/styles/components/formTimeCom.less';
import dayjs from 'dayjs';
import 'dayjs/locale/zh-cn';
import 'dayjs/locale/en';
let progressId = 0; // 定时器进度条
const FormTimeCom = forwardRef(({ handleResetMonitorFn, refreshTimeValue }, ref) => {
  const intl = useIntl();
  const formRef = useRef(null);
  const dayjsLocale = intl.locale === 'en-US' ? 'en' : 'zh-cn';
  useEffect(() => {
    dayjs.locale(dayjsLocale);
  }, [dayjsLocale]);

  const translatedTimePeriodOptions = timePeriodOptions.map((opt) => ({
    ...opt,
    label: intl.formatMessage({ id: opt.label }),
  }));
  const translatedRefreshTimeOptions = refreshTimeOptions.map((opt) => ({
    ...opt,
    label: intl.formatMessage({ id: opt.label }),
  }));

  const initProgressBar = useCallback(() => {
    const refreshDom = document.querySelector('#border_group_refresh');
    const maxWidth = window.getComputedStyle(refreshDom).width;
    const progressBar = document.querySelector('#progress_bar');
    progressBar.style.width = maxWidth;
    timeOutFn(maxWidth, progressBar);
  }, [refreshTimeValue]);

  // 定时函数
  const timeOutFn = (maxWidth, progressBar) => {
    clearTimeout(progressId);
    if (refreshTimeValue) {
      progressId = setTimeout(() => {
        const progressBarWidth = progressBar.style.width;
        const eachStep = Number(parseFloat(maxWidth)) / refreshTimeValue;
        progressBar.style.setProperty('width', `${Number(parseFloat(progressBarWidth)) - eachStep}px`);
        timeOutFn(maxWidth, progressBar);
      }, 1000);
    }
  };

  useEffect(() => {
    if (formRef.current) {
      initProgressBar();
    }
    return () => clearTimeout(progressId);
  }, [initProgressBar, formRef.current]);

  const EndTimeDatePicker = (props) => (
    <div className="border_group">
      <div className="addbefore">{intl.formatMessage({ id: 'monitor.endTime' })}</div>
      <DatePicker
        className="date_picker"
        showTime
        placeholder={intl.formatMessage({ id: 'monitor.selectTime' })}
        {...props}
      />
    </div>
  );

  const RefreshTimeSelect = (props) => (
    <Fragment>
      <div className="border_group" id="border_group_refresh">
        <Select className="refresh_time_select" options={translatedRefreshTimeOptions} {...props} />
        <Button icon={<SyncOutlined />} onClick={handleResetMonitorFn} className="reset_btn"></Button>
      </div>
      <div id="progress_bar"></div>
    </Fragment>
  );

  return (
    <div className="flex_box" ref={formRef}>
      <Form.Item name="period">
        <Select options={translatedTimePeriodOptions} style={{ minWidth: '100px' }} />
      </Form.Item>
      <Form.Item name="endTime">
        <EndTimeDatePicker />
      </Form.Item>
      <Form.Item>
        <div className="border_group" id="border_group_refresh">
          <Form.Item name="refreshTime" noStyle>
            <Select className="refresh_time_select" options={translatedRefreshTimeOptions} />
          </Form.Item>
          <Button icon={<SyncOutlined />} onClick={handleResetMonitorFn} className="reset_btn"></Button>
          <div id="progress_bar"></div>
        </div>
      </Form.Item>
    </div>
  );
});

export default FormTimeCom;
