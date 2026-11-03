/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */

import BreadCrumbCom from '@/components/BreadCrumbCom';
import '@/styles/schedulingPolicyConfiguration.less';
import { ExclamationCircleFilled, QuestionCircleOutlined } from '@ant-design/icons';
import { Radio, Switch, message } from 'antd';
import { Link } from 'inula-router';
import {
  getNumaAffinitySchedulingStatus,
  updateNumaAffinitySchedulingStatus,
  getBinpackSchedulingStatus,
  updateBinpackSchedulingStatus,
  getFairSchedulingStatus,
  updateFairSchedulingStatus,
  getGroupSchedulingStatus,
  updateGroupSchedulingStatus,
  getPrioritySchedulingStatus,
  updatePrioritySchedulingStatus,
} from '@/api/schedulingApi';
import { useEffect, useState } from 'openinula';
import noData from '@/assets/images/noData.png';
import { getVolcano, getGang, getBinPack, getPriority, getDrf, getDescVolcano } from '@/assets/md';
import PriorityPreemption from './PriorityPreemption';
import { useIntl } from 'inula-intl';

export default function SchedulingPolicyConfiguration() {
  const intl = useIntl();
  const [numaAffinitySchedulingStatus, setNumaAffinitySchedulingStatus] = useState(false);
  const [numaAffinityPartial, setNumaAffinityPartial] = useState(false);
  const [numaAffinityLoading, setNumaAffinityLoading] = useState(false);
  const [binpackSchedulingStatus, setBinpackSchedulingStatus] = useState(false);
  const [binpackLoading, setBinpackLoading] = useState(false);

  const [fairSchedulingStatus, setFairSchedulingStatus] = useState(false);
  const [fairLoading, setFairLoading] = useState(false);

  const [groupSchedulingStatus, setGroupSchedulingStatus] = useState(false);
  const [groupLoading, setGroupLoading] = useState(false);

  const [prioritySchedulingStatus, setPrioritySchedulingStatus] = useState(false);
  const [priorityLoading, setPriorityLoading] = useState(false);
  const [selectedScheduler, setSelectedScheduler] = useState('volcano');

  const [explanatoryPanelOpen, setExplanatoryPanel] = useState(false);

  const [explanatoryInfo, setExplanatoryInfo] = useState(getVolcano());

  const handleChangeNumaAffinitySchedulingStatus = async (checked) => {
    setNumaAffinityLoading(true);
    try {
      await updateNumaAffinitySchedulingStatus({ policy: checked ? 'open' : 'close' });
      getNumaAffinitySchedulingStatusInfo();
      setNumaAffinityLoading(false);
    } catch (error) {
      if (error.response.status === 403) {
        message.error('操作失败，当前用户没有操作权限，请联系管理员添加权限!');
      } else {
        message.error(error.data.message);
      }
      setNumaAffinityLoading(false);
    }
  };

  const getNumaAffinitySchedulingStatusInfo = async () => {
    const res = await getNumaAffinitySchedulingStatus();
    setNumaAffinitySchedulingStatus(res.data.data !== 'close');
    setNumaAffinityPartial(res.data.data === 'partial');
  };

  const handleChangeBinpackSchedulingStatus = async (checked) => {
    setBinpackLoading(true);
    try {
      await updateBinpackSchedulingStatus({ policy: checked ? 'open' : 'close' });
      getBinpackSchedulingStatusInfo();
      setBinpackLoading(false);
    } catch (error) {
      if (error.response.status === 403) {
        message.error('操作失败，当前用户没有操作权限，请联系管理员添加权限!');
      } else {
        message.error(error.response?.data.message);
      }
      setBinpackLoading(false);
    }
  };

  const getBinpackSchedulingStatusInfo = async () => {
    const res = await getBinpackSchedulingStatus();
    setBinpackSchedulingStatus(res.data.data === 'open');
  };

  const handleFairSchedulingStatus = async (checked) => {
    setFairLoading(true);
    try {
      await updateFairSchedulingStatus({ policy: checked ? 'open' : 'close' });
      getFairSchedulingStatusInfo();
      setBinpackLoading(false);
    } catch (error) {
      if (error.response.status === 403) {
        message.error('操作失败，当前用户没有操作权限，请联系管理员添加权限!');
      } else {
        message.error(error.response?.data.message);
      }
      setFairLoading(false);
    }
  };

  const getFairSchedulingStatusInfo = async () => {
    const res = await getFairSchedulingStatus();
    setFairSchedulingStatus(res.data.data === 'open');
  };

  const handleGroupSchedulingStatus = async (checked) => {
    setGroupLoading(true);
    try {
      await updateGroupSchedulingStatus({ policy: checked ? 'open' : 'close' });
      getGroupSchedulingStatusInfo();
      setGroupLoading(false);
    } catch (error) {
      if (error.response.status === 403) {
        message.error('操作失败，当前用户没有操作权限，请联系管理员添加权限!');
      } else {
        message.error(error.response?.data.message);
      }
      setGroupLoading(false);
    }
  };

  const getGroupSchedulingStatusInfo = async () => {
    const res = await getGroupSchedulingStatus();
    setGroupSchedulingStatus(res.data.data === 'open');
  };

  const handlePrioritySchedulingStatus = async (checked) => {
    setPriorityLoading(true);
    try {
      await updatePrioritySchedulingStatus({ policy: checked ? 'open' : 'close' });
      getPrioritySchedulingStatusInfo();
      setPriorityLoading(false);
    } catch (error) {
      if (error.response.status === 403) {
        message.error('操作失败，当前用户没有操作权限，请联系管理员添加权限!');
      } else {
        message.error(error.response?.data.message);
      }
      setPriorityLoading(false);
    }
  };

  const getPrioritySchedulingStatusInfo = async () => {
    const res = await getPrioritySchedulingStatus();
    setPrioritySchedulingStatus(res.data.data === 'open');
  };

  const handleSchedulerChange = (e) => {
    setSelectedScheduler(e.target.value);
  };

  const handleClosePriorityPreemption = () => {
    setExplanatoryPanel(false);
  };

  const handleNumaAwareOpen = () => {
    setExplanatoryInfo(getVolcano());
    setExplanatoryPanel(true);
  };

  const handleBinPack = () => {
    setExplanatoryInfo(getBinPack());
    setExplanatoryPanel(true);
  };

  const handleDescAbout = () => {
    setExplanatoryInfo(getDescVolcano());
    setExplanatoryPanel(true);
  };

  const handleDrfOpen = () => {
    setExplanatoryInfo(getDrf());
    setExplanatoryPanel(true);
  };

  const handleGangOpen = () => {
    setExplanatoryInfo(getGang());
    setExplanatoryPanel(true);
  };

  const handlePriorityOpen = () => {
    setExplanatoryInfo(getPriority());
    setExplanatoryPanel(true);
  };

  const goToCustome = () => {
    window.location.href = '/container_platform/platformManage/customResourceDefinition';
  };

  useEffect(() => {
    getNumaAffinitySchedulingStatusInfo();
    getBinpackSchedulingStatusInfo();
    getFairSchedulingStatusInfo();
    getGroupSchedulingStatusInfo();
    getPrioritySchedulingStatusInfo();
  }, []);
  return (
    <div className="scheduling_policy_container">
      <BreadCrumbCom
        items={[
          { title: intl.formatMessage({ id: 'common.numaAffinityScheduling' }), disabled: true },
          { title: intl.formatMessage({ id: 'scheduling.schedulingPolicyConfig' }), disabled: true },
        ]}
      />
      <div className="default_scheduling">
        <div className="schedule_title mb_12">{intl.formatMessage({ id: 'scheduling.scheduler' })}</div>
        <Radio.Group
          buttonStyle="solid"
          defaultValue="volcano"
          style={{ marginBottom: '32px' }}
          onChange={handleSchedulerChange}
        >
          <Radio.Button value="volcano" className="radio_schedule">
            volcano
          </Radio.Button>
        </Radio.Group>
        {selectedScheduler === 'volcano' && (
          <div className="custom-switch">
            <div className="schedule_description">
              {intl.formatMessage({ id: 'scheduling.volcanoDesc' })}
              <QuestionCircleOutlined
                onClick={handleDescAbout}
                className="question-icon-hover"
                style={{ marginLeft: '7px' }}
              />
            </div>
            <div className="schedule_title mb_12">
              {intl.formatMessage({ id: 'scheduling.numaAffinityScheduling' })}
              <QuestionCircleOutlined onClick={handleNumaAwareOpen} className="question-icon-hover" />
            </div>
            <Switch
              onChange={handleChangeNumaAffinitySchedulingStatus}
              loading={numaAffinityLoading}
              checked={numaAffinitySchedulingStatus}
              className="mb_12"
            />
            {numaAffinityPartial && (
              <div className="schedule_description" style={{ color: '#d46b08' }}>
                <ExclamationCircleFilled style={{ marginRight: '6px' }} />
                {intl.formatMessage({ id: 'scheduling.partialEnabled' })}
              </div>
            )}
            <div className="schedule_description">{intl.formatMessage({ id: 'scheduling.numaAffinityDesc' })}</div>
            <div className="schedule_title mb_12">
              {intl.formatMessage({ id: 'scheduling.binpackScheduling' })}
              <QuestionCircleOutlined onClick={handleBinPack} className="question-icon-hover" />
            </div>
            <Switch
              checked={binpackSchedulingStatus}
              onChange={handleChangeBinpackSchedulingStatus}
              LOADING={binpackLoading}
              className="mb_12"
            />
            <div className="schedule_description">{intl.formatMessage({ id: 'scheduling.binpackDesc' })}</div>
            <div className="schedule_title mb_12">
              {intl.formatMessage({ id: 'scheduling.fairScheduling' })}
              <QuestionCircleOutlined onClick={handleDrfOpen} className="question-icon-hover" />
            </div>
            <Switch
              onChange={handleFairSchedulingStatus}
              checked={fairSchedulingStatus}
              LOADING={fairLoading}
              className="mb_12"
            />
            <div className="schedule_description">{intl.formatMessage({ id: 'scheduling.fairSchedulingDesc' })}</div>
            <div className="schedule_title mb_12">
              {intl.formatMessage({ id: 'scheduling.groupScheduling' })}
              <QuestionCircleOutlined onClick={handleGangOpen} className="question-icon-hover" />
            </div>
            <Switch
              checked={groupSchedulingStatus}
              loading={groupLoading}
              onChange={handleGroupSchedulingStatus}
              className="mb_12"
            />
            <div className="schedule_description">{intl.formatMessage({ id: 'scheduling.groupSchedulingDesc' })}</div>
            <div className="schedule_title mb_12">
              {intl.formatMessage({ id: 'scheduling.priorityPreemption' })}
              <QuestionCircleOutlined onClick={handlePriorityOpen} className="question-icon-hover" />
            </div>
            <Switch
              checked={prioritySchedulingStatus}
              loading={priorityLoading}
              onChange={handlePrioritySchedulingStatus}
              className="mb_12"
            />
            <div className="schedule_description">
              {intl.formatMessage({ id: 'scheduling.priorityPreemptionDesc' })}
              <span className="check_title">
                <Link to="/SchedulingPolicyConfiguration/priorityClass">
                  {intl.formatMessage({ id: 'scheduling.viewPriorityQueue' })} &gt;
                </Link>
              </span>
            </div>
          </div>
        )}
      </div>
      {selectedScheduler === 'scheduler' && (
        <div className="default_scheduling_no_data">
          <div>
            <img src={noData} style={{ marginBottom: '26px' }} />
            <div style={{ textAlign: 'center', color: '#89939b' }}>
              {intl.formatMessage({ id: 'scheduling.noConfigurablePolicy' })}
              <br />
              {intl.formatMessage({ id: 'scheduling.switchScheduler' })}
            </div>
          </div>
        </div>
      )}
      {selectedScheduler === 'volcano' && explanatoryPanelOpen && (
        <PriorityPreemption onClose={handleClosePriorityPreemption} data={explanatoryInfo} />
      )}
    </div>
  );
}
