/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Modal, Button, Input, Radio } from 'antd';
import { Fragment, useEffect, useState } from 'openinula';
import { useIntl } from 'inula-intl';
import '@/styles/components/createMonitorComponentModal.less';
import areaSimple from '@/assets/images/chart/area.png';
import barSimple from '@/assets/images/chart/bar.png';
import lineSimple from '@/assets/images/chart/line.png';
import dountSimple from '@/assets/images/chart/dount.png';
import numberSimple from '@/assets/images/chart/number.png';
import dashboardSimple from '@/assets/images/chart/dashboard.png';

export default function CreateMonitorComponentModal({ open, cancelFn, okFn, URL, changeURL, loading }) {
  const intl = useIntl();

  const [componentType, setComponentType] = useState('line');

  const handleSelectType = (type) => {
    setComponentType(type);
  };
  const componentTypeList = [
    {
      title: intl.formatMessage({ id: 'chart.line' }),
      type: 'line',
      description: intl.formatMessage({ id: 'chart.lineDesc' }),
      img: lineSimple,
    },
    {
      title: intl.formatMessage({ id: 'chart.area' }),
      type: 'area',
      description: intl.formatMessage({ id: 'chart.areaDesc' }),
      img: areaSimple,
    },
    {
      title: intl.formatMessage({ id: 'chart.number' }),
      type: 'number',
      description: intl.formatMessage({ id: 'chart.numberDesc' }),
      img: numberSimple,
    },
    {
      title: intl.formatMessage({ id: 'chart.dashboard' }),
      type: 'dashboard',
      description: intl.formatMessage({ id: 'chart.dashboardDesc' }),
      img: dashboardSimple,
    },
    {
      title: intl.formatMessage({ id: 'chart.dount' }),
      type: 'dount',
      description: intl.formatMessage({ id: 'chart.dountDesc' }),
      img: dountSimple,
    },
    {
      title: intl.formatMessage({ id: 'chart.bar' }),
      type: 'bar',
      description: intl.formatMessage({ id: 'chart.barDesc' }),
      img: barSimple,
    },
  ];
  const renderComponent = (data) => (
    <div
      className="chart_card"
      onClick={() => handleSelectType(data.type)}
      style={componentType === data.type ? { borderColor: '#346AC4' } : {}}
    >
      <div className="label_grp">
        <Radio checked={componentType === data.type}>{data.title}</Radio>
        <p className="description">{data.description}</p>
      </div>
      <img src={data.img} />
    </div>
  );

  const handleOkFn = (type) => {
    okFn(type);
  };
  return (
    <Modal
      getContainer={false}
      className="monitor_component_modal"
      title={intl.formatMessage({ id: 'monitor.newMonitorComponent' })}
      open={open}
      onOk={okFn}
      onCancel={cancelFn}
      footer={[
        <Fragment>
          <Button className="cancel_btn normal_btn" onClick={cancelFn}>
            {intl.formatMessage({ id: 'common.cancel' })}
          </Button>
          <Button
            className="monitor_primary_btn normal_btn"
            onClick={() => handleOkFn(componentType)}
            loading={loading}
          >
            {intl.formatMessage({ id: 'common.next' })}
          </Button>
        </Fragment>,
      ]}
    >
      <div className="monitor_component_list">
        <div className="source_type">
          <p className="samll_title">{intl.formatMessage({ id: 'monitor.dataSourceType' })}</p>
          <Radio checked={true}>prometheus</Radio>
          <Input
            value={URL}
            onChange={changeURL}
            style={{ width: '300px' }}
            placeholder={intl.formatMessage({ id: 'common.inputPlaceholder' })}
          />
        </div>
        <div className="chart_component">
          <p className="samll_title">{intl.formatMessage({ id: 'monitor.selectComponentType' })}</p>
          <div className="chart_list">{componentTypeList.map((item) => renderComponent(item))}</div>
        </div>
      </div>
    </Modal>
  );
}
