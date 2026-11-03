/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Button, Input, message, Pagination, Tooltip } from 'antd';
import { InfoCircleFilled, QuestionCircleOutlined } from '@ant-design/icons';
import { useState, useEffect, useRef, useStore } from 'openinula';
import '@/styles/applicationMarket/index.less';
import CreateApplicationPackage from '@/pages/applicationMarket/component/CreateApplicationPackage';
import {
  getHelmChartDetails,
  deleteHelmChart,
  synchronizationHelmRepo,
  getsynchronizationStatus,
} from '@/api/applicationMarketApi';
import Dayjs from 'dayjs';
import { useHistory } from 'inula-router';
import { containerRouterPrefix } from '@/constant.js';
import helmIcon from '@/assets/images/helmIcon.png';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { showSuccessThen } from '@/tools/utils';
import { ResponseCode } from '@/common/constants';
import ToastMsg from '@/components/ToastMsg';
import { useTranslation } from 'react-i18next';
import useButtonLoading from '@/hooks/useButtonLoading';

const { Search } = Input;
const limitSearch = 53;
export default function ApplicationPackageManagement() {
  const [total, setTotal] = useState(0);
  const [openAdd, setOpenAdd] = useState(false);
  const [dataSource, setDataSource] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [limit, setLimit] = useState(10);
  const [packageName, setPackageName] = useState('');
  const [syncLoading, setSyncLoading] = useButtonLoading();
  const themeStore = useStore('theme');
  const [messageApi, contextHolder] = message.useMessage();
  const { t } = useTranslation(['appMarket', 'common']);
  const syncIntervalRef = useRef(null);

  useEffect(() => () => clearInterval(syncIntervalRef.current), []);

  function handleCreateWareHouse() {
    setOpenAdd((open) => !open);
  }

  const handleCancel = () => {
    getHelmChartListInfo();
    setOpenAdd((open) => !open);
  };

  const handleChange = (page, pageSize) => {
    setCurrentPage(page);
    setLimit(pageSize);
  };

  const getHelmChartListInfo = async () => {
    try {
      const res = await getHelmChartDetails(packageName, currentPage, limit);
      setDataSource(res.data.data.items);
      setTotal(res.data.data.totalItems);
    } catch (error) {
      messageApi.error(error.response?.data?.msg);
    }
  };

  const handleInput = (e) => {
    if (e.target.value.length >= limitSearch) {
      messageApi.error(t('market.searchLimit', { limit: limitSearch }));
      return;
    }
    setPackageName(e.target.value);
  };

  const handleSearch = async (e) => {
    try {
      const res = await getHelmChartDetails(e, currentPage, limit);
      setDataSource(res.data.data.items);
      setTotal(res.data.data.totalItems);
    } catch (error) {
      messageApi.error(error.response?.data?.msg);
    }
  };

  const handleRefresh = () => {
    getHelmChartListInfo();
  };

  const checkUpdateStatus = (timeout) => {
    const startTime = Date.now();
    let inFlight = false;
    const intervalld = setInterval(async () => {
      if (inFlight) {
        return;
      }
      inFlight = true;
      try {
        const currentTime = Date.now();
        if (currentTime - startTime >= timeout) {
          clearInterval(intervalld);
          messageApi.error(t('messages.syncFailed'));
          setSyncLoading(false);
          return;
        }
        try {
          const res = await getsynchronizationStatus('local');
          if (res.data.msg === 'complete') {
            getHelmChartListInfo();
            messageApi.success(t('messages.localSyncSuccess', { repo: t('warehouse.localRepo') }));
            clearInterval(intervalld);
            setSyncLoading(false);
          }
          if (res.data.msg === 'in_progress') {
            setSyncLoading(true);
          }
        } catch (error) {
          messageApi.error(t('messages.fetchDataError', { ns: 'application', message: error.response?.data?.msg }));
          clearInterval(intervalld);
          setSyncLoading(false);
        }
      } finally {
        inFlight = false;
      }
    }, 3000);
    syncIntervalRef.current = intervalld;
  };

  const handleSynchronization = async () => {
    setSyncLoading(true);
    const timeout = 5 * 60 * 1000;
    try {
      await synchronizationHelmRepo('local');
      checkUpdateStatus(timeout);
    } catch (error) {
      if (error?.response?.status === ResponseCode.Forbidden) {
        messageApi.error(t('messages.noPermission'));
      } else if (error?.response?.status === ResponseCode.BadRequest && error?.response?.data?.msg?.includes('syncronizing in progress')) {
        setSyncLoading(true);
        checkUpdateStatus(timeout);
      } else {
        messageApi.error(t('messages.fetchDataError', { ns: 'application', message: error.response?.data?.msg }));
      }
    }
  };
  useEffect(() => {
    getHelmChartListInfo();
  }, [currentPage, limit, packageName]);

  return (
    <div className='child_content withBread_content'>
      <ToastMsg contextHolder={contextHolder} />
      <div className='prompt' style={{ backgroundColor: themeStore.$s.theme !== 'light' ? 'rgba(119,174,247, 0.12)' : undefined, border: '1px solid rgba(119,174,247, 0.50)' }}>
        <InfoCircleFilled className='prompt-icon' />
        <span>{t('package.description')}</span>
      </div>
      <div style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff', margin: '20px 32px', borderRadius: '4px', paddingBottom: '126px' }}>
        <div>
          <div className='ware_house_search'>
            <Search
              placeholder={t('searchPlaceholder', { ns: 'common', name: t('searchName.app', { ns: 'common' }) })}
              style={{ width: '300px', height: '32px' }}
              onChange={handleInput}
              onSearch={handleSearch}
              maxLength={limitSearch}
            />
            <div style={{ display: 'flex', gap: '16px' }}>
              <Button className='primary_btn' onClick={handleCreateWareHouse}>
                {t('package.addPackage')}
              </Button>
              <div>
                <Button loading={syncLoading} className='cancel_btn' onClick={handleSynchronization}>{t('package.sync')}</Button>
                <Tooltip title={t('package.syncTooltip')}>
                  <QuestionCircleOutlined style={{ color: '#89939bff', marginLeft: '8px' }} />
                </Tooltip>
              </div>
            </div>
          </div>
          <div className='package_item_container'>
            {dataSource.map((el, index) => (
              <PackageItem key={`${el?.metadata?.name ?? el?.name ?? ''}_${el?.metadata?.version ?? el?.version ?? index}`} info={el} onRefresh={handleRefresh} />
            ))}
          </div>
          <CreateApplicationPackage
            openAdd={openAdd}
            onCancelFn={handleCancel}
          />
        </div>
        {total > 10 &&
          <Pagination
            className='page'
            showTotal={(curTotal) => t('pagination.total', { ns: 'common', total: curTotal })}
            showSizeChanger
            showQuickJumper
            total={total}
            onChange={handleChange}
            pageSizeOptions={[10, 20, 50]}
          ></Pagination>}
      </div>
    </div>
  );
}

function PackageItem({ info, onRefresh }) {
  const history = useHistory();
  const [deletePackageName, setDeletePackageName] = useState('');
  const [packageNameDelModalOpen, setPackageNameDelModalOpen] = useState(false);
  const [isPackageNameDelCheck, setIsPackageNameDelCheck] = useState(false);
  const themeStore = useStore('theme');
  const [messageApi, contextHolder] = message.useMessage();
  const { t } = useTranslation(['appMarket', 'common']);
  const handleDelete = (name) => {
    setDeletePackageName(name);
    setPackageNameDelModalOpen(true);
  };

  const handleDelPackageNameConfirm = async () => {
    try {
      const res = await deleteHelmChart(deletePackageName);
      if (res.status === ResponseCode.OK) {
        showSuccessThen(messageApi, t('messages.deleteSuccess'), () => {
          onRefresh();
          setPackageNameDelModalOpen(false);
          setIsPackageNameDelCheck(false);
        });
      }
    } catch (error) {
      if (error.response.status === ResponseCode.Forbidden) {
        messageApi.error(t('messages.noPermission'));
      } else {
        messageApi.error(error.response?.data.msg);
      }
    }
  };

  const goToPackageDetail = () => {
    history.push(
      {
        pathname: `/${containerRouterPrefix}/appMarket/stash/packageManageDetails/${info.repo}/${info.metadata.name}`,
        state: {
          pkgDetail: info.metadata,
        },
      },
    );
  };

  const handlePackageNameeCheckFn = (e) => {
    setIsPackageNameDelCheck(e.target.checked);
  };

  const handleDelPackageNameCancel = () => {
    setDeletePackageName('');
    setIsPackageNameDelCheck(false);
    setPackageNameDelModalOpen(false);
  };

  return (
    <div className='package_item' style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff', border: themeStore.$s.theme === 'light' ? '1px solid #dcdcdc' : '1px solid #444' }}>
      <ToastMsg contextHolder={contextHolder} />
      <div onClick={goToPackageDetail} style={{ marginBottom: 16 }}>
        <div
          style={{
            display: 'flex',
            marginBottom: '20px',
            gap: '16px',
          }}
        >
          <div>
            <img
              src={info.metadata.icon ? info.metadata.icon : helmIcon}
              style={{ height: '48px', width: '48px' }}
              alt={t('package.noIcon')}
            />
          </div>
          <div>
            <Tooltip title={info.metadata.name}>
              <div className='fz_16 fw_bold application_title clamp_2'>{info.metadata.name}</div>
            </Tooltip>
            <div>{Dayjs(info.metadata.created).format('YYYY-MM-DD')}</div>
          </div>
        </div>
        <Tooltip title={info.metadata.description}>
          <div className='clamp_3'>{info.metadata.description}</div>
        </Tooltip>
      </div>
      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <Button
          style={{ width: 64, backgroundColor: themeStore.$s.theme !== 'light' && '#2a2d34ff' }}
          onClick={() => handleDelete(info.metadata.name)}
        >
          {t('actions.delete', { ns: 'common' })}
        </Button>
      </div>
      <DeleteInfoModal
        title={t('package.deletePackage')}
        open={packageNameDelModalOpen}
        cancelFn={handleDelPackageNameCancel}
        content={[
          t('package.deleteConfirm'),
          t('package.confirmDelete', { name: deletePackageName }),
        ]}
        isCheck={isPackageNameDelCheck}
        showCheck={true}
        checkFn={handlePackageNameeCheckFn}
        confirmFn={handleDelPackageNameConfirm} />
    </div>
  );
}
